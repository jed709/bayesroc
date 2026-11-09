"""
NumPyro backend for the bayesroc SDT package.

Config-driven model: receives a config dict from R and builds the NumPyro model
dynamically.

Entry point: fit_model(config, chains, warmup, samples, seed)
"""

import os
import sys
import time
import re as _re
from functools import partial

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.special import expit as jax_expit

jax.config.update("jax_enable_x64", os.environ.get("BAYESROC_ENABLE_X64", "1") != "0")
if os.environ.get("BAYESROC_ENABLE_X64", "1") == "0":
    print("[bayesroc] float32 mode (x64 disabled)", flush=True)

_FLOAT = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32

_cache = os.environ.get("BAYESROC_XLA_CACHE", "")
if _cache and _cache != "0":
    try:
        os.makedirs(_cache, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", _cache)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    except Exception:
        pass


_PHI_D = 0.3989422804014327


import numpyro
import numpyro.distributions as dist
from numpyro.infer import HMC, MCMC, NUTS, init_to_uniform
import progress


# =============================================================================
# Detailed progress bar patch for multi-chain runs
# =============================================================================

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _script_dir)


# =============================================================================
# Prior Parsing
# =============================================================================

def _grid_strings(strs, shape):
    arr = np.empty(int(np.prod(shape)) if shape else 1, dtype=object)
    flat = []

    def walk(x):
        if isinstance(x, (list, tuple)):
            for v in x:
                walk(v)
        else:
            flat.append(x)
    walk(strs)
    if len(flat) == 1 and arr.size > 1:
        flat = flat * arr.size
    arr[:] = flat
    return arr


def _sum_zero_basis(n):
    """Orthonormal basis [n, n-1] of the vectors summing to zero (normalised Helmert contrasts, as R's)."""
    H = np.zeros((n, max(n - 1, 0)))
    for j in range(n - 1):
        H[:j + 1, j] = -1.0
        H[j + 1, j] = j + 1.0
    return H / np.sqrt(np.sum(H * H, axis=0)) if n > 1 else H


def _sum_zero_gaps(name, prior_rows, rows=False):
    """Shape gaps that sum to zero in each row: rows [R][n] of prior strings -> [R, n]. The free
    coordinates `name`_z are standard normal, corrected by a factor to each gap's own prior on the
    full vector; `name` is the full vector, as Stan's transformed parameter."""
    R, n = len(prior_rows), len(prior_rows[0])
    H = jnp.asarray(_sum_zero_basis(n), dtype=_FLOAT)
    if n < 2:
        g = jnp.zeros((R, n))
    else:
        z = numpyro.sample(f"{name}_z", dist.Normal(0, 1).expand([R, n - 1]) if R > 1 or rows else dist.Normal(0, 1).expand([n - 1]))
        z = z.reshape(R, n - 1)
        g = z @ H.T
        lp = sum(parse_prior(prior_rows[r][k]).log_prob(g[r, k]) for r in range(R) for k in range(n))
        numpyro.factor(f"{name}_shape", lp - jnp.sum(dist.Normal(0, 1).log_prob(z)))
    numpyro.deterministic(name, g if R > 1 or rows else g[0])
    return g


def _grid_sample(name, strs, shape, transform=None):
    """Sample one array site whose elements carry their own prior strings."""
    tf = transform or (lambda d: d)
    flat = _grid_strings(strs, shape)
    if len(set(flat)) == 1:
        d = tf(parse_prior(flat[0]))
        return numpyro.sample(name, d.expand(list(shape)) if shape else d)
    ds = [parse_prior(s) for s in flat]
    cls = type(ds[0])
    if all(type(d) is cls for d in ds):
        params = {k: np.asarray([getattr(d, k) for d in ds], dtype=np.dtype(_FLOAT)).reshape(shape)
                  for k in cls.arg_constraints}
        return numpyro.sample(name, tf(cls(**params)))
    ds = [tf(d) for d in ds]
    support = ds[0].support
    x = numpyro.sample(name, dist.ImproperUniform(support, (), tuple(shape)))
    xf = x.reshape(-1)
    numpyro.factor(f"{name}_prior", sum(d.log_prob(xf[i]) for i, d in enumerate(ds)))
    return x


def _grid_get(config, key, i=None, j=None, default=None):
    g = config.get(key)
    if g is None:
        return default
    if isinstance(g, str):
        return g
    if i is None:
        return g
    row = g[i] if i < len(g) else None
    if row is None:
        return default
    if j is None or isinstance(row, str):
        return row
    return row[j] if j < len(row) else default


# ---- centred designs -----------------------------------------------------------------------------
# A design with an intercept and other columns is sampled in centred coordinates: the sampled
# intercept is b0_c = b0 + sum_j m_j b_j (R: .centre_info), the model uses b0 and the original X,
# the prior stays on b0, and b0 is emitted under "orig__<site>" so the draws come back in the
# model's coordinates under the original names (_restore_centred).

def _centre_of(obj):
    """(intercept columns, means): one column, or the cell columns of a cell-means design."""
    c = obj.get("centre") if isinstance(obj, dict) else None
    if not c:
        return None
    cols = c["col"] if isinstance(c["col"], list) else [c["col"]]
    return [int(j) for j in cols], jnp.asarray(np.asarray(c["means"][0], dtype=float), dtype=_FLOAT)


def _uncentre(beta_c, centre):
    cols, m = centre
    shift = jnp.dot(m, beta_c)
    beta = beta_c
    for j in cols:
        beta = beta.at[j].add(-shift)
    return beta


def _centre_factor(name, prior_of, centre, b_orig, b_c):
    """the prior stays on the model's intercept(s): prior_of(col) -> distribution"""
    lp = sum(prior_of(j).log_prob(b_orig[j]) - prior_of(j).log_prob(b_c[j]) for j in centre[0])
    numpyro.factor(f"centre__{name}", lp)


def _restore_centred(samples):
    """orig__<site> -> <site>, replacing the sampled (centred) values."""
    for k in [k for k in samples if k.startswith("orig__")]:
        samples[k[6:]] = samples.pop(k)
    return samples


def _centre_init(vals, config):
    """User inits are in the model's coordinates; move each centred intercept to the sampler's."""
    def shift(base, centre, idx_of=None):
        cols, m = centre[0], np.asarray(centre[1], dtype=float)
        if base in vals:
            arr = np.asarray(vals[base], dtype=float)
            if idx_of is None:
                if arr.ndim == 1 and arr.shape[0] == m.shape[0]:
                    arr = arr.copy(); sh = float(m @ arr)
                    for j in cols:
                        arr[j] += sh
                    vals[base] = arr
            elif arr.ndim == 2 and arr.shape[1] == m.shape[0]:
                arr = arr.copy(); sh = arr @ m
                for j in cols:
                    arr[:, j] += sh
                vals[base] = arr
            return
        if idx_of is None:
            keys = [f"{base}[{i + 1}]" for i in range(m.shape[0])]
            if all(k in vals for k in keys):
                b = np.array([float(np.asarray(vals[k]).reshape(-1)[0]) for k in keys])
                for j in cols:
                    vals[keys[j]] = b[j] + float(m @ b)
        else:
            for k in range(idx_of):
                keys = [f"{base}[{k + 1},{i + 1}]" for i in range(m.shape[0])]
                if all(kk in vals for kk in keys):
                    b = np.array([float(np.asarray(vals[kk]).reshape(-1)[0]) for kk in keys])
                    for j in cols:
                        vals[keys[j]] = b[j] + float(m @ b)

    knob = _rkg_knob(config)
    for pname, pc in (config.get("params") or {}).items():
        c = _centre_of(pc)
        if c is None or (pname == "know_crit" and knob is not None):
            continue
        if pname == "criterion2":
            shift("beta_thresh_mid_2", c)
            shift("beta_log_gaps_2", c, idx_of=max(0, int(config.get("K2", 2)) - 2))
            shift("beta_raw_gaps_2", c, idx_of=max(0, int(config.get("K2", 2)) - 2))
        else:
            shift(f"beta_{pname}", c)
    c = _centre_of({"centre": config.get("crit_centre")})
    if c is not None:
        shift("beta_thresh_mid", c)
        n_gaps = int(config.get("n_thresh", int(config["K"]) - 1)) - 1
        shift("beta_log_gaps", c, idx_of=n_gaps)
        shift("beta_raw_gaps", c, idx_of=n_gaps)
    return vals


def parse_prior(prior_str, dim=None):
    """Parse a Stan prior string into a NumPyro distribution.

    Args:
        prior_str: Stan syntax, e.g. "normal(0, 1)", "lkj_corr_cholesky(1)"
        dim: Matrix dimension (required for LKJ)

    Returns:
        A numpyro.distributions.Distribution object
    """
    s = prior_str.strip()

    m = _re.match(r"(\w+)\s*\((.*)\)\s*$", s, _re.DOTALL)
    if not m:
        if s == "std_normal()" or s == "std_normal":
            return dist.Normal(0.0, 1.0)
        raise ValueError(f"Cannot parse prior string: '{prior_str}'")

    name = m.group(1).lower()
    args_str = m.group(2).strip()
    args = [float(x.strip()) for x in args_str.split(",")] if args_str else []

    dispatch = {
        "normal": lambda a: dist.Normal(a[0], a[1]),
        "std_normal": lambda a: dist.Normal(0.0, 1.0),
        "half_normal": lambda a: dist.HalfNormal(a[0]),
        "cauchy": lambda a: dist.Cauchy(a[0], a[1]),
        "half_cauchy": lambda a: dist.HalfCauchy(a[0] if len(a) == 1 else a[1]),
        "student_t": lambda a: dist.StudentT(a[0], a[1], a[2]) if len(a) == 3 else dist.StudentT(a[0]),
        "half_student_t": lambda a: dist.StudentT(a[0], 0.0, a[1]),
        "exponential": lambda a: dist.Exponential(a[0]),
        "beta": lambda a: dist.Beta(a[0], a[1]),
        "gamma": lambda a: dist.Gamma(a[0], a[1]),
        "inv_gamma": lambda a: dist.InverseGamma(a[0], a[1]),
        "weibull": lambda a: dist.Weibull(a[1], a[0]),
        "uniform": lambda a: dist.Uniform(a[0], a[1]),
        "lognormal": lambda a: dist.LogNormal(a[0], a[1]),
        "logistic": lambda a: dist.Logistic(a[0], a[1]),
        "gumbel": lambda a: dist.Gumbel(a[0], a[1]),
        "double_exponential": lambda a: dist.Laplace(a[0], a[1]),
        "laplace": lambda a: dist.Laplace(a[0], a[1]),
    }

    if name in ("lkj_corr_cholesky", "lkj", "lkj_corr"):
        eta = args[0] if args else 2.0
        if dim is None or dim < 2:
            raise ValueError(f"LKJ prior requires dim >= 2, got dim={dim}")
        return dist.LKJCholesky(dim, concentration=eta)

    if name in dispatch:
        return dispatch[name](args)

    raise ValueError(f"Unknown prior distribution: '{name}' in '{prior_str}'")


# =============================================================================
# Link Functions
# =============================================================================

_CLOGLOG_HI = 700.0


def _cloglog_cdf(x):
    """Gumbel-min (smallest extreme value) CDF: F(x) = 1 - exp(-exp(x))."""
    return -jnp.expm1(-jnp.exp(jnp.minimum(x, _CLOGLOG_HI)))


def _cloglog_pdf(x):
    """Gumbel-min density: f(x) = exp(x - exp(x))."""
    xc = jnp.minimum(x, _CLOGLOG_HI)
    return jnp.exp(xc - jnp.exp(xc))


def _logit_pdf(x):
    e = jnp.exp(-jnp.abs(x))
    return e / (1.0 + e) ** 2


def _link_fns(link_name):
    """(CDF, PDF) pair for a latent-distribution link."""
    if link_name == "logit":
        return jax_expit, _logit_pdf
    if link_name == "cloglog":
        return _cloglog_cdf, _cloglog_pdf
    return (lambda x: _probit_triple(x)[0]), (lambda x: _probit_triple(x)[1])


_OPEN = 1e10
_CLOGLOG_MED = -0.36651292058166435


_TAIL_SWITCH = 1e-6


def _pfloor():
    """Smallest cell probability kept before log: the float underflow limit."""
    return 1e-300 if jax.config.jax_enable_x64 else 1e-37


_EMPTY_SLOPE = 10.0


def _empty_cell_lp(violation):
    """Log-likelihood assigned to an observed cell the model gives probability zero: the floor,
    falling linearly in how far the parameters are from making the cell possible, so a chain
    started there is driven back into the support rather than left on a flat floor."""
    return jnp.log(_pfloor()) - _EMPTY_SLOPE * violation


def _link_diff_fn(link_name):
    """P(lo < Z < hi) for the link's latent distribution, differenced on the side of
    the median the interval lies on so upper-tail cells never cancel to zero."""
    if link_name == "logit":
        return lambda lo, hi: jnp.where(lo >= 0, jax_expit(-lo) - jax_expit(-hi),
                                        jax_expit(hi) - jax_expit(lo))
    if link_name == "cloglog":
        def diff(lo, hi):
            surv = lambda x: jnp.exp(-jnp.exp(jnp.minimum(x, _CLOGLOG_HI)))
            return jnp.where(lo >= _CLOGLOG_MED, surv(lo) - surv(hi),
                             _cloglog_cdf(hi) - _cloglog_cdf(lo))
        return diff
    return lambda lo, hi: _phi_diff(lo, hi)[0]


# =============================================================================
# Threshold Construction
# =============================================================================

def _gap_transform(x, gap_link="log"):
    """Apply gap transformation: exp (default) or softplus."""
    if gap_link == "softplus":
        return jax.nn.softplus(x)
    return jnp.exp(x)


def build_thresholds(thresh_mid, log_gaps, n_thresh, gap_link="log"):
    """Build ordered thresholds from mid-anchor + gaps.

    Args:
        thresh_mid: scalar or array of shape [P_crit] dot-producted with X
        log_gaps: array of shape [n_thresh - 1, ...] (log of gap sizes)
        n_thresh: number of thresholds

    Returns:
        Array of shape [n_thresh] with ordered thresholds
    """
    mid = (n_thresh + 1) // 2
    mid_idx = mid - 1
    n_upper = n_thresh - mid

    thresh = jnp.zeros(n_thresh)
    thresh = thresh.at[mid_idx].set(thresh_mid)

    for k in range(mid_idx + 1, n_thresh):
        gap_idx = k - mid_idx - 1
        thresh = thresh.at[k].set(thresh[k - 1] + _gap_transform(log_gaps[gap_idx], gap_link))

    for k in range(mid_idx - 1, -1, -1):
        gap_idx = n_upper + (mid_idx - k - 1)
        thresh = thresh.at[k].set(thresh[k + 1] - _gap_transform(log_gaps[gap_idx], gap_link))

    return thresh


def _location_scale_criteria(location, shape_gaps, log_scale, n_thresh, gap_link="log"):
    scale = jnp.exp(log_scale)
    if shape_gaps.ndim == 1:
        tbar = jnp.mean(build_thresholds(0.0, shape_gaps, n_thresh, gap_link))
        shape_gaps = shape_gaps[:, None]
    else:
        tbar = jnp.mean(build_thresholds_vectorized(jnp.zeros(location.shape[0]), shape_gaps, n_thresh, gap_link), axis=1)
    mid = location - scale * tbar
    if gap_link == "softplus":
        v = scale[None, :] * jax.nn.softplus(shape_gaps)
        gaps = v + jnp.log(-jnp.expm1(-v))
    else:
        gaps = shape_gaps + log_scale[None, :]
    return mid, gaps


def _gap_thresh_pos(g, mid, n_upper):
    return mid + 1 + g if g < n_upper else mid - 1 - (g - n_upper)


def _crit_re_shift(u, idx, conf, n_thresh, mid, n_gaps):
    """Per-observation threshold shifts from one criterion RE term: (mid [N],
    gaps [n_gaps, N]). Dimensions are threshold-major, matching Stan."""
    if conf.get("location_only"):
        if conf.get("type", "index") == "matrix":
            Z = _ensure_2d(jnp.asarray(conf["Z"], dtype=_FLOAT))
            return jnp.sum(Z * u[:, idx].T, axis=1), None
        if conf.get("term_idx") is not None:
            tidx = jnp.array(conf["term_idx"], dtype=jnp.int32) - 1
            return jnp.where(tidx >= 0, u[jnp.clip(tidx, 0, u.shape[0] - 1), idx], 0.0), None
        return u[0, idx], None
    n_upper = n_thresh - mid - 1
    n_dims = u.shape[0]
    if conf.get("type", "index") == "matrix":
        Z = _ensure_2d(jnp.asarray(conf["Z"], dtype=_FLOAT))
        R = n_dims // n_thresh

        def at(k):
            return jnp.sum(Z * u[k * R:(k + 1) * R, idx].T, axis=1)
    elif conf.get("term_idx") is not None:
        R = n_dims // n_thresh
        tidx = jnp.array(conf["term_idx"], dtype=jnp.int32) - 1
        has = tidx >= 0
        tsafe = jnp.clip(tidx, 0, R - 1)

        def at(k):
            return jnp.where(has, u[k * R + tsafe, idx], 0.0)
    else:
        def at(k):
            return u[k, idx]
    mid_shift = at(mid)
    gaps = [at(_gap_thresh_pos(g, mid, n_upper)) for g in range(n_gaps)]
    return mid_shift, (jnp.stack(gaps) if gaps else None)


def _crit2_varying_shift(u, idx, mode, y1_idx, n_thresh, mid, n_gaps, conf=None):
    """Per-observation source-threshold shifts under varying source criteria."""
    n_upper = n_thresh - mid - 1
    Z = None
    if conf is not None and conf.get("type") == "matrix":
        Z = _ensure_2d(jnp.asarray(conf["Z"], dtype=_FLOAT))
    R = 1 if Z is None else Z.shape[1]

    def at(c):
        if Z is None:
            return u[c, idx]
        return sum(Z[:, r] * u[c * R + r, idx] for r in range(R))

    if mode == "full":
        base = y1_idx * n_thresh
        gaps = [at(base + _gap_thresh_pos(g, mid, n_upper)) for g in range(n_gaps)]
        return at(base + mid), (jnp.stack(gaps) if gaps else None)
    if mode == "per_bin" and u.shape[0] > R:
        return at(y1_idx), None
    return at(0), None


def build_thresholds_vectorized(thresh_mid_vec, log_gaps_mat, n_thresh, gap_link="log"):
    """Build thresholds for N observations.

    Args:
        thresh_mid_vec: [N] per-obs mid threshold
        log_gaps_mat: [n_gaps, N] per-obs log gaps (or [n_gaps] if shared)
        n_thresh: int

    Returns:
        [N, n_thresh] array
    """
    N = thresh_mid_vec.shape[0]
    mid = (n_thresh + 1) // 2
    mid_idx = mid - 1
    n_upper = n_thresh - mid

    thresh = jnp.zeros((N, n_thresh))
    thresh = thresh.at[:, mid_idx].set(thresh_mid_vec)

    if log_gaps_mat.ndim == 1:
        log_gaps_mat = jnp.broadcast_to(log_gaps_mat[:, None], (log_gaps_mat.shape[0], N))

    for k in range(mid_idx + 1, n_thresh):
        gap_idx = k - mid_idx - 1
        thresh = thresh.at[:, k].set(thresh[:, k - 1] + _gap_transform(log_gaps_mat[gap_idx], gap_link))

    for k in range(mid_idx - 1, -1, -1):
        gap_idx = n_upper + (mid_idx - k - 1)
        thresh = thresh.at[:, k].set(thresh[:, k + 1] - _gap_transform(log_gaps_mat[gap_idx], gap_link))

    return thresh


# =============================================================================
# Fast Likelihood Primitives (custom JVP for analytic gradients)
# =============================================================================

from jax import custom_jvp

# =============================================================================
# Bivariate normal: rectangle and CDF with exact derivatives
# =============================================================================

_SIN_C = [(-1.0) ** k / float(np.prod(np.arange(1, 2 * k + 2))) for k in range(11)]
_ASIN_P = [4.253011369004428248960E-3, -6.019598008014123785661E-1, 5.444622390564711410273E0,
           -1.626247967210700244449E1, 1.956261983317594739197E1, -8.198089802484824371615E0]
_ASIN_Q = [1.0, -1.474091372988853791896E1, 7.049610280856842141659E1,
           -1.471791292232726029859E2, 1.395105614657485689735E2, -4.918853881490881290097E1]
_ASIN_R = [2.967721961301243206100E-3, -5.634242780008963776856E-1, 6.968710824104713396794E0,
           -2.556901049652824852289E1, 2.853665548261061424989E1]
_ASIN_S = [1.0, -2.194779531642920639778E1, 1.470656354026814941758E2,
           -3.838770957603691357202E2, 3.424398657913078477438E2]


def _horner(c, x):
    acc = c[0]
    for k in c[1:]:
        acc = acc * x + k
    return acc


def _sin_halfpi(x):
    x2 = x * x
    acc = _SIN_C[-1]
    for c in reversed(_SIN_C[:-1]):
        acc = acc * x2 + c
    return x * acc


def _asin(x):
    a = jnp.abs(x)
    z = a * a
    small = a + a * z * _horner(_ASIN_P, z) / _horner(_ASIN_Q, z)
    zz = 1.0 - a
    p = zz * _horner(_ASIN_R, zz) / _horner(_ASIN_S, zz)
    zs = jnp.sqrt(zz + zz)
    big = (0.785398163397448309616 - zs) - (zs * p - 6.123233995736765886130E-17) + 0.785398163397448309616
    return jnp.sign(x) * jnp.where(a > 0.625, big, small)


_DW_RULES = {m: np.polynomial.legendre.leggauss(m) for m in (8, 10, 12, 14, 16, 20, 28, 32, 40, 48, 64, 80, 96, 128)}
_RHO_MAX = 0.999999
_DW_TIER_EDGES = (0.67, 0.79, 0.865, 0.91, 0.94, 0.97, 0.99, 0.994, 0.9978, 0.999, 0.9997, 0.99985, 0.99997)
_DW_TIER_NODES = (8, 10, 12, 14, 16, 20, 28, 32, 40, 48, 64, 80, 96, 128)


def _dw_nodes(rho, m):
    """The Gauss-Legendre nodes of the sine-rule integral on a trailing axis of length m."""
    r = jnp.clip(rho, -_RHO_MAX, _RHO_MAX)
    A = _asin(r)
    X, W = _DW_RULES[m]
    xp = jnp.asarray((X + 1.0) / 2.0, dtype=A.dtype)
    w = jnp.asarray(W, dtype=A.dtype)
    s = _sin_halfpi(A[..., None] * xp)
    ic2 = 1.0 / (1.0 - s * s)
    return r, A, (w, s, ic2, s * ic2, w * xp * jnp.sqrt(ic2))


def _dw_corner(h, k, A, nodes):
    w, s, ic2, sic, wdq = nodes
    a = ((h * h + k * k) / 2.0)[..., None]
    b = (h * k)[..., None]
    e = jnp.exp(-a * ic2 + b * sic)
    we = w * e
    S0 = jnp.sum(we, -1)
    S1 = jnp.sum(we * ic2, -1)
    S2 = jnp.sum(we * sic, -1)
    T = jnp.sum(e * wdq * (b - 2.0 * a * sic + 2.0 * b * s * sic), -1)
    c = A / (4.0 * jnp.pi)
    return c * S0, -c * (h * S1 - k * S2), -c * (k * S1 - h * S2), (S0 + A * T) / (4.0 * jnp.pi)


def _dw_corners(HK, rho):
    """Corner integrals with the Gauss-Legendre order set by the largest |rho| present (same tiers as the
    native kernel: better than 1e-6 relative on rectangles above the tail switch up to |rho| = 0.9995)."""
    r = jnp.clip(rho, -_RHO_MAX, _RHO_MAX)
    tier = jnp.searchsorted(jnp.asarray(_DW_TIER_EDGES), jnp.max(jnp.abs(r)), side="right")

    def branch(m):
        def f(_):
            _, A, nodes = _dw_nodes(rho, m)
            return [_dw_corner(h, k, A, nodes) for h, k in HK]
        return f
    return r, jax.lax.switch(tier, [branch(m) for m in _DW_TIER_NODES], None)


def _phi2_density(x, y, rho):
    """The bivariate normal density at (x, y), the exact derivative of the CDF in rho; zero at an open
    edge and at or past the rho clamp, as the kernels' derivative in rho is."""
    r = jnp.clip(rho, -_RHO_MAX, _RHO_MAX)
    c = (1.0 - r) * (1.0 + r)
    fin = (jnp.abs(x) < 1e9) & (jnp.abs(y) < 1e9) & (jnp.abs(rho) < _RHO_MAX)
    xs, ys = jnp.where(fin, x, 0.0), jnp.where(fin, y, 0.0)
    d = jnp.exp(-(xs * xs - 2.0 * r * xs * ys + ys * ys) / (2.0 * c)) / (2.0 * np.pi * jnp.sqrt(c))
    return jnp.where(fin, d, 0.0)


_ACC_PHI_C = [0.5893276044905619, 0.47231061768868227, 0.1485473376619552, 0.03611913148900286, 0.006378297104388951, 0.0006592194044005224, -1.1534346994167076e-05, -1.541999589462102e-05, -1.4350295963176838e-06, 2.671384604629693e-07, 5.842069382918476e-08, -4.878387335732758e-09, -2.0166300903829314e-09, 1.2065191939381096e-10, 7.135408878679637e-11, -4.885139134318498e-12, -2.630066125081603e-12, 2.7380131795870467e-13, 9.691376820174753e-14]
_ACC_PHI_D = [1.1854995356145523, 1.3097573132537195, 0.45298641056440525, 0.10328147437489583, 0.01295638227198467, -0.0003253821430972744, -0.0004287832995995369, -4.434221633584315e-05, 9.551609080045374e-06, 2.278034508995305e-06, -2.1182703839817368e-07, -9.437945361118253e-08, 6.090771522883504e-09, 3.89287566636132e-09, -2.7866990704357154e-10, -1.638034930910627e-10, 1.8902019919991792e-11, 7.084052091904388e-12]
_ACC_PHI_A = 0.015
_ACC_PHI_B = 1.0


def _clenshaw(co, u):
    b1 = 0.0
    b2 = 0.0
    for k in range(len(co) - 1, 0, -1):
        b1, b2 = 2.0 * u * b1 - b2 + co[k], b1
    return u * b1 - b2 + co[0]


def _acc_phi_and_deriv(x):
    """Phi to ~1e-14 absolute (tail mass to ~2e-13 relative) as phi(x) * t * C(t), C a
    degree-18 Chebyshev series in t = 1 / (1 + 0.2316419 |x|); returns Phi, its exact
    derivative, and the tail mass. Used where bivariate cancellation needs accurate marginals."""
    t = 1.0 / (1.0 + 0.2316419 * jnp.abs(x))
    u = (2.0 * t - (_ACC_PHI_A + _ACC_PHI_B)) / (_ACC_PHI_B - _ACC_PHI_A)
    Cu = _clenshaw(_ACC_PHI_C, u)
    Du = _clenshaw(_ACC_PHI_D, u)
    e = _PHI_D * jnp.exp(-0.5 * x * x)
    q = e * t * Cu
    G = (1.0 - t) / 0.2316419 * Cu + 0.2316419 * t * t * (Cu + t * Du)
    return jnp.where(x >= 0, 1.0 - q, q), e * G, q


def _acc_phi_diff(z_lo, z_hi):
    Ph, dh, Qh = _acc_phi_and_deriv(z_hi)
    Pl, dl, Ql = _acc_phi_and_deriv(z_lo)
    return jnp.where(z_lo >= 0, Ql - Qh, Ph - Pl), -dl, dh


_PROBIT_C = (0.23164256724403792, 0.2316161018958152, 0.21963988229322737, 0.19034147705499127, 0.182843173379805, 0.023627126024141354, 0.3261383197776697, -0.4226294412990229, 0.566393164718276, -0.42518433981895243, 0.14839140546152088, -0.019505298125989647)
_PROBIT_G = (1.0000028804980357, -0.00011425112737684096, 0.001956685225065126, -0.019177470139476724, 0.12026315990835201, -0.510972748965256, 1.5177140282601391, -3.1995984292291655, 4.79844987841736, -5.063837790314006, 3.656934740253874, -1.7097165426172967, 0.46231487164147905, -0.05421893181564818)


_PROBIT32_C = (0.23193557115619975, 0.22641403663893195, 0.25867795829003176, 0.029693243246920797,
               0.5723460328090683, -0.5135716343748198, 0.6254146800669893, -0.17759562815820015)
_PROBIT32_G = (1.0012677808125376, -0.023836510222320806, 0.1930096132521973, -0.8836349552755781,
               2.522398546253241, -4.660402762116997, 5.579909961288564, -4.180390403547215,
               1.780787510474899, -0.3291087099060719)


def _probit_triple(x):
    """Phi(x), its exact derivative, and the tail mass min(Phi, 1 - Phi) as phi(x) t C(t),
    C degree 11 in t = 1 / (1 + 0.2316419 |x|): ~1e-9 relative in both tails. The same
    polynomial as the native kernel."""
    t = 1.0 / (1.0 + 0.2316419 * jnp.abs(x))
    e = _PHI_D * jnp.exp(-0.5 * x * x)
    C, G = (_PROBIT32_C, _PROBIT32_G) if x.dtype == jnp.float32 else (_PROBIT_C, _PROBIT_G)
    q = e * t * _horner(C[::-1], t)
    return jnp.where(x >= 0, 1.0 - q, q), e * _horner(G[::-1], t), q


def _phi_diff(z_lo, z_hi):
    """(Phi(z_hi) - Phi(z_lo), -phi(z_lo), phi(z_hi)), differenced on the side of zero the
    interval lies on; the native kernel (~1e-9 relative in both tails) when it is loaded."""
    z_lo, z_hi = jnp.broadcast_arrays(jnp.asarray(z_lo), jnp.asarray(z_hi))
    if _native_ok(z_lo):
        D, pdf_lo, pdf_hi = _native_call("probit_diff", 3, [z_lo, z_hi])
        return D, -pdf_lo, pdf_hi
    Ph, dh, Qh = _probit_triple(z_hi)
    Pl, dl, Ql = _probit_triple(z_lo)
    dl, dh = _band_pdfs(z_lo, z_hi, dl, dh)
    return _probit_band(z_lo, z_hi, Pl, Ph, Ql, Qh), -dl, dh


_HALF_PHI = (0.3989422804014327, -0.06649038006690544, 0.009973557010035817, -0.0011873282154804543, 0.00011543468761615529, -9.444656259503615e-06, 6.659693516316651e-07, -4.122667414862689e-08, 2.2735298243728065e-09, -1.1301171641619213e-10, 5.1124347902563106e-12, -2.121761474217046e-13, 8.133418984498675e-15, -2.896516732371323e-16, 9.631274849017947e-18, -3.0033007593711877e-19)


def _half_phi(x):
    return x * _horner(_HALF_PHI[::-1], x * x)


def _band_narrow(z_lo, z_hi):
    return (z_lo < 0) & (z_hi >= 0) & (z_hi - z_lo < 1.0)


def _probit_band(z_lo, z_hi, Pl, Ph, Ql, Qh):
    D = jnp.where(z_lo >= 0, Ql - Qh, Ph - Pl)
    if D.dtype != jnp.float64:
        return D
    narrow = _band_narrow(z_lo, z_hi)
    return jnp.where(narrow, _half_phi(jnp.clip(z_hi, -1.0, 1.0)) + _half_phi(jnp.clip(-z_lo, -1.0, 1.0)), D)


def _band_pdfs(z_lo, z_hi, dl, dh):
    """The edge densities to pair with _probit_band: exact where the band is the series (|z| < 1), so that the
    derivative matches the value there."""
    if dl.dtype != jnp.float64:
        return dl, dh
    narrow = _band_narrow(z_lo, z_hi)
    exact = lambda x: jnp.exp(-0.5 * jnp.clip(x, -1.0, 1.0) ** 2 - _LOG_SQRT_2PI)
    return jnp.where(narrow, exact(z_lo), dl), jnp.where(narrow, exact(z_hi), dh)


def _dw_rect_grads_jax(z1_lo, z1_hi, z2_lo, z2_hi, rho):
    r, (hh, lh, hl, ll) = _dw_corners(((z1_hi, z2_hi), (z1_lo, z2_hi), (z1_hi, z2_lo), (z1_lo, z2_lo)), rho)
    D1, dD1_lo, dD1_hi = _acc_phi_diff(z1_lo, z1_hi)
    D2, dD2_lo, dD2_hi = _acc_phi_diff(z2_lo, z2_hi)
    p = D1 * D2 + hh[0] - lh[0] - hl[0] + ll[0]
    return (p,
            dD1_lo * D2 - lh[1] + ll[1],
            dD1_hi * D2 + hh[1] - hl[1],
            D1 * dD2_lo - hl[2] + ll[2],
            D1 * dD2_hi + hh[2] - lh[2],
            _phi2_density(z1_hi, z2_hi, rho) - _phi2_density(z1_lo, z2_hi, rho) - _phi2_density(z1_hi, z2_lo, rho)
            + _phi2_density(z1_lo, z2_lo, rho))


def _dw_cdf_grads_jax(h, k, rho):
    r, (c,) = _dw_corners(((h, k),), rho)
    Ph, dh, _ = _acc_phi_and_deriv(h)
    Pk, dk, _ = _acc_phi_and_deriv(k)
    return Ph * Pk + c[0], dh * Pk + c[1], Ph * dk + c[2], _phi2_density(h, k, rho)


_BVN_FFI = False
if os.environ.get("BAYESROC_BVN_FFI"):
    import ctypes
    try:
        _bvn_lib = ctypes.cdll.LoadLibrary(os.environ["BAYESROC_BVN_FFI"])
        for _t, _sym in (("bvn_rect", "BvnRect"), ("vrdp2d", "Vrdp2d"),
                         ("bvn_cdf2", "BvnCdf2"), ("bvn_cdf4", "BvnCdf4"),
                         ("probit_cell", "ProbitCell"),
                         ("probit_diff", "ProbitDiff"), ("cdp_cell", "CdpCell"), ("metad_cell", "MetadCell"),
                         ("pcq", "Pcq"), ("lnoise_cell", "LnoiseCell"),
                         ("tmap_fwd", "TmapFwd"), ("tmap_bwd", "TmapBwd"),
                         ("tmap_blk_fwd", "TmapBlkFwd"), ("tmap_blk_bwd", "TmapBlkBwd")):
            jax.ffi.register_ffi_target(f"bayesroc_{_t}", jax.ffi.pycapsule(getattr(_bvn_lib, f"Bayesroc{_sym}")),
                                        platform="cpu")
        for _t, _sym in (("bvn_rect", "BvnRect"), ("vrdp2d", "Vrdp2d"),
                         ("bvn_cdf2", "BvnCdf2"), ("bvn_cdf4", "BvnCdf4"),
                         ("probit_cell", "ProbitCell"), ("probit_diff", "ProbitDiff"),
                         ("cdp_cell", "CdpCell"), ("metad_cell", "MetadCell"), ("pcq", "Pcq"),
                         ("lnoise_cell", "LnoiseCell"), ("tmap_fwd", "TmapFwd"), ("tmap_bwd", "TmapBwd"),
                         ("tmap_blk_fwd", "TmapBlkFwd"), ("tmap_blk_bwd", "TmapBlkBwd")):
            jax.ffi.register_ffi_target(f"bayesroc_{_t}_f32", jax.ffi.pycapsule(getattr(_bvn_lib, f"Bayesroc{_sym}F32")),
                                        platform="cpu")
    except Exception as _e:
        raise RuntimeError(f"bayesroc: the native kernel failed to load ({_e}); rebuild it with "
                           "bayesroc:::.bvn_ffi_lib(rebuild = TRUE)") from _e
    _BVN_FFI = True
elif os.environ.get("BAYESROC_PURE_JAX_REFERENCE") != "1":
    raise RuntimeError("bayesroc: no native kernel given (BAYESROC_BVN_FFI); fits must go through "
                       "bayesroc's R interface, which builds it")


def _native_ok(x):
    return _BVN_FFI and jnp.asarray(x).dtype in (jnp.float64, jnp.float32)


def _native_call(name, n_out, args):
    """Call a native kernel on same-shaped arrays; f32 inputs go to the kernel's f32 entry,
    which evaluates in double and rounds the results."""
    dt = args[0].dtype
    sd = jax.ShapeDtypeStruct(args[0].shape, dt)
    target = f"bayesroc_{name}" if dt == jnp.float64 else f"bayesroc_{name}_f32"
    return tuple(jax.ffi.ffi_call(target, (sd,) * n_out, vmap_method="broadcast_all")(
        *[jnp.asarray(a, dtype=dt) for a in args]))


def _probit_cell_native(z_lo, z_hi):
    """log P(z_lo < Z < z_hi) and its derivatives in z_lo and z_hi from the native
    kernel (Phi to ~1e-9 relative in both tails)."""
    return _native_call("probit_cell", 3, jnp.broadcast_arrays(jnp.asarray(z_lo), jnp.asarray(z_hi)))


def _use_probit_native(link_name, x):
    return link_name == "probit" and _native_ok(x)


_GL32_X, _GL32_W = np.polynomial.legendre.leggauss(32)
_LOG_SQRT_2PI = 0.9189385332046727


def _log_q_pos(x):
    t = 1.0 / (1.0 + 0.2316419 * x)
    u = (2.0 * t - (_ACC_PHI_A + _ACC_PHI_B)) / (_ACC_PHI_B - _ACC_PHI_A)
    cheb = -0.5 * x * x - _LOG_SQRT_2PI + jnp.log(t * _clenshaw(_ACC_PHI_C, u))
    xs = jnp.maximum(x, 40.0)
    r = 1.0 / (xs * xs)
    asym = (-0.5 * xs * xs - jnp.log(xs) - _LOG_SQRT_2PI +
            jnp.log1p(r * (-1.0 + r * (3.0 + r * (-15.0 + r * (105.0 - 945.0 * r))))))
    return jnp.where(x > 40.0, asym, cheb)


def _log_q(x):
    ax = jnp.abs(x)
    lq = _log_q_pos(ax)
    return jnp.where(x >= 0, lq, jnp.log1p(-jnp.exp(lq)))


def _log_diff_phi(lo, hi):
    a_up, b_up = _log_q(lo), _log_q(hi)
    a_dn, b_dn = _log_q(-hi), _log_q(-lo)
    up = a_up + jnp.log1p(-jnp.exp(b_up - a_up))
    dn = a_dn + jnp.log1p(-jnp.exp(b_dn - a_dn))
    mid = jnp.log(jnp.maximum(0.5 * (jax.scipy.special.erf(hi * 0.7071067811865476) -
                                     jax.scipy.special.erf(lo * 0.7071067811865476)), 1e-300))
    out = jnp.where(lo >= 0, up, jnp.where(hi <= 0, dn, mid))
    return jnp.where(hi > lo, out, -jnp.inf)


_GL16_X, _GL16_W = np.polynomial.legendre.leggauss(16)
_GRADE_LEVELS = 6


def _graded_log_integral(g, px, ps):
    """log of the integral of exp(g) over (px[..., 0], px[..., -1]), split at the sorted points px
    [..., P], on Gauss-Legendre panels graded geometrically from each point's local scale ps out to
    the middle of each piece (as bvn_tail.h)."""
    A, B = px[..., :-1], px[..., 1:]
    H = 0.5 * jnp.maximum(B - A, 0.0)
    k = jnp.arange(_GRADE_LEVELS + 1.0) / _GRADE_LEVELS
    vals = []
    for end, sc, sign in ((A, ps[..., :-1], 1.0), (B, ps[..., 1:], -1.0)):
        ratio = jnp.maximum(H / jnp.maximum(sc, 1e-300), 1.0)
        offs = jnp.where((ratio > 1.0)[..., None], H[..., None] * ratio[..., None] ** (k - 1.0), H[..., None] * k)
        offs = jnp.concatenate([jnp.zeros_like(offs[..., :1]), offs], axis=-1)
        lo, hi = offs[..., :-1], offs[..., 1:]
        h = 0.5 * (hi - lo)
        xs = end[..., None, None] + sign * ((lo + h)[..., None] + h[..., None] * _GL16_X)
        lw = jnp.log(jnp.maximum(h[..., None] * _GL16_W, 1e-300))
        v = lw + g(xs.reshape(xs.shape[:-3] + (-1,))).reshape(xs.shape)
        v = jnp.where((h > 0.0)[..., None] & jnp.isfinite(v), v, jax.lax.stop_gradient(jnp.full_like(v, -jnp.inf)))
        vals.append(v.reshape(v.shape[:-3] + (-1,)))
    return jax.scipy.special.logsumexp(jnp.concatenate(vals, axis=-1), axis=-1)


def _tail_split_points(dg, m, left, right, kinks):
    """Sorted split points (the ends, the mode and the integrand's transitions inside the range)
    and their local scales; `kinks` is a list of (point, width) with non-finite points ignored. The
    mode's scale is at most the narrowest transition's width, as one may lie beside it."""
    width = jnp.ones_like(m)
    for x, w in kinks:
        width = jnp.where(jnp.isfinite(x), jnp.minimum(width, w), width)
    d = 1e-3 * width
    curv = -(dg(m + d) - dg(m - d)) / (2.0 * d)
    mode_scale = jnp.minimum(1.0 / jnp.sqrt(jnp.maximum(jnp.where(jnp.isfinite(curv), curv, 1.0), 1.0)), width)
    big = jnp.full_like(m, jnp.inf)
    pts = [left, m, right]
    scl = [big, mode_scale, big]
    for x, w in kinks:
        inside = jnp.isfinite(x) & (x > left) & (x < right)
        pts.append(jnp.where(inside, x, m))
        scl.append(jnp.where(inside, w, mode_scale))
    px, ps = jnp.stack(pts, -1), jnp.stack(scl, -1)
    order = jnp.argsort(px, axis=-1)
    return jnp.take_along_axis(px, order, -1), jnp.take_along_axis(ps, order, -1)


def _rect_tail_jax(a1, b1, a2, b2, rho):
    """Tail-accurate P(a1 < Z1 < b1, a2 < Z2 < b2; rho) and its derivatives in
    (a1, b1, a2, b2, rho): log-space Gauss-Legendre over the log-concave integrand
    phi(x) P(a2 < Z2 < b2 | x) around its mode; same algorithm as bvn_tail.h."""
    clamped = ~(jnp.abs(rho) < 0.999999)
    rho = jnp.clip(rho, -0.999999, 0.999999)
    s = jnp.sqrt((1.0 - rho) * (1.0 + rho))
    lo = jnp.maximum(a1, -40.0)
    hi = jnp.minimum(b1, 40.0)
    ok = hi > lo
    hi = jnp.where(ok, hi, lo + 1.0)

    def bc(v, x):
        return v.reshape(v.shape + (1,) * (x.ndim - v.ndim))

    def lcond(x):
        A2, B2, R, S = bc(a2, x), bc(b2, x), bc(rho, x), bc(s, x)
        return _log_diff_phi((A2 - R * x) / S, (B2 - R * x) / S)

    def g(x):
        return -0.5 * x * x - _LOG_SQRT_2PI + lcond(x)

    def dg(x):
        l, u = (a2 - rho * x) / s, (b2 - rho * x) / s
        ld = _log_diff_phi(l, u)
        tl = jnp.where(jnp.abs(l) < 1e9, jnp.exp(-0.5 * l * l - _LOG_SQRT_2PI - ld), 0.0)
        tu = jnp.where(jnp.abs(u) < 1e9, jnp.exp(-0.5 * u * u - _LOG_SQRT_2PI - ld), 0.0)
        return -x + (rho / s) * (tl - tu)

    at_lo = dg(lo) <= 0.0
    at_hi = dg(hi) >= 0.0

    def bis(_, lh):
        L, H = lh
        M = 0.5 * (L + H)
        up = dg(M) > 0.0
        return jnp.where(up, M, L), jnp.where(up, H, M)
    L, H = jax.lax.fori_loop(0, 64, bis, (lo, hi))
    m = jnp.where(at_lo, lo, jnp.where(at_hi, hi, 0.5 * (L + H)))
    gm = g(m)
    steps = 0.25 * (1.6 ** jnp.arange(40.0))
    offs = jnp.cumsum(steps)

    def reach(direction, edge):
        xs = m[..., None] + direction * offs
        past = (xs - edge[..., None]) * direction >= 0.0
        drop = (gm[..., None] - g(jnp.where(past, m[..., None], xs))) > 40.0
        stop = past | drop
        first = jnp.argmax(stop, axis=-1)
        x_stop = jnp.take_along_axis(xs, first[..., None], axis=-1)[..., 0]
        hit_edge = jnp.take_along_axis(past, first[..., None], axis=-1)[..., 0]
        e = jnp.where(hit_edge, edge, x_stop)
        return jnp.where((edge - m) * direction > 0.0, e, m)
    left = reach(-1.0, lo)
    right = reach(1.0, hi)
    safe = jnp.where(rho == 0.0, 1.0, rho)
    kw = s / jnp.abs(safe)
    kinks = [(jnp.where(rho == 0.0, jnp.inf, a2 / safe), kw), (jnp.where(rho == 0.0, jnp.inf, b2 / safe), kw)]
    px, ps = _tail_split_points(dg, m, left, right, kinks)
    lp = _graded_log_integral(g, px, ps)
    lp = jnp.where(ok, lp, -jnp.inf)
    p = jnp.exp(lp)

    def edge_d(x, lo2, hi2):
        fin = jnp.abs(x) < 1e9
        xs = jnp.where(fin, x, 0.0)
        v = jnp.exp(-0.5 * xs * xs - _LOG_SQRT_2PI + _log_diff_phi((lo2 - rho * xs) / s, (hi2 - rho * xs) / s))
        return jnp.where(fin, v, 0.0)
    d1h = edge_d(b1, a2, b2)
    d1l = -edge_d(a1, a2, b2)
    d2h = edge_d(b2, a1, b1)
    d2l = -edge_d(a2, a1, b1)

    def phi2(x, y):
        fin = (jnp.abs(x) < 1e9) & (jnp.abs(y) < 1e9)
        xs, ys = jnp.where(fin, x, 0.0), jnp.where(fin, y, 0.0)
        v = jnp.exp(-(xs * xs - 2.0 * rho * xs * ys + ys * ys) / (2.0 * s * s) - jnp.log(s) - 1.8378770664093455)
        return jnp.where(fin, v, 0.0)
    drho = jnp.where(clamped, 0.0, phi2(b1, b2) - phi2(a1, b2) - phi2(b1, a2) + phi2(a1, a2))
    return p, d1l, d1h, d2l, d2h, drho


def _lin_bounds(Al, Bl, Au, Bu, x):
    l = jnp.where(jnp.isfinite(Al), Al + Bl * x, jax.lax.stop_gradient(jnp.sign(Al) * _OPEN))
    u = jnp.where(jnp.isfinite(Au), Au + Bu * x, jax.lax.stop_gradient(jnp.sign(Au) * _OPEN))
    return l, u


def _lin_log_integral_jax(Al, Bl, Au, Bu, a, b):
    """log of the integral over (a, b) of phi(x) [Phi(Au + Bu x) - Phi(Al + Bl x)]; an infinite
    A is a constant bound. Same algorithm as _rect_tail_jax; the band may vanish at an end
    (a polygon vertex), where the log integrand is -inf and rises into the interval."""
    lo = jnp.maximum(a, -40.0)
    hi = jnp.minimum(b, 40.0)
    ok = hi > lo
    hi = jnp.where(ok, hi, lo + 1.0)

    def bc(v, x):
        return v.reshape(v.shape + (1,) * (x.ndim - v.ndim))

    def g(x):
        l, u = _lin_bounds(bc(Al, x), bc(Bl, x), bc(Au, x), bc(Bu, x), x)
        return -0.5 * x * x - _LOG_SQRT_2PI + _log_diff_phi(l, u)

    def dg(x):
        l, u = _lin_bounds(Al, Bl, Au, Bu, x)
        ld = _log_diff_phi(l, u)
        live = jnp.isfinite(ld)
        tl = jnp.where(live & (jnp.abs(l) < 1e9), jnp.exp(-0.5 * l * l - _LOG_SQRT_2PI - ld), 0.0)
        tu = jnp.where(live & (jnp.abs(u) < 1e9), jnp.exp(-0.5 * u * u - _LOG_SQRT_2PI - ld), 0.0)
        return -x + Bu * tu - Bl * tl

    at_lo = (dg(lo) <= 0.0) & jnp.isfinite(g(lo))
    at_hi = (dg(hi) >= 0.0) & jnp.isfinite(g(hi))

    def bis(_, lh):
        L, H = lh
        M = 0.5 * (L + H)
        up = dg(M) > 0.0
        return jnp.where(up, M, L), jnp.where(up, H, M)
    L, H = jax.lax.fori_loop(0, 64, bis, (lo, hi))
    m = jnp.where(at_lo, lo, jnp.where(at_hi, hi, 0.5 * (L + H)))
    gm = g(m)
    offs = jnp.cumsum(0.25 * (1.6 ** jnp.arange(40.0)))

    def reach(direction, edge):
        xs = m[..., None] + direction * offs
        past = (xs - edge[..., None]) * direction >= 0.0
        drop = (gm[..., None] - g(jnp.where(past, m[..., None], xs))) > 40.0
        stop = past | drop
        first = jnp.argmax(stop, axis=-1)
        x_stop = jnp.take_along_axis(xs, first[..., None], axis=-1)[..., 0]
        hit_edge = jnp.take_along_axis(past, first[..., None], axis=-1)[..., 0]
        e = jnp.where(hit_edge, edge, x_stop)
        return jnp.where((edge - m) * direction > 0.0, e, m)
    left = reach(-1.0, lo)
    right = reach(1.0, hi)
    kinks = []
    for A_, B_ in ((Al, Bl), (Au, Bu)):
        live = jnp.isfinite(A_) & (B_ != 0.0)
        safe = jnp.where(live, B_, 1.0)
        kinks.append((jnp.where(live, -jnp.where(live, A_, 0.0) / safe, jnp.inf), 1.0 / jnp.abs(safe)))
    px, ps = _tail_split_points(dg, m, left, right, kinks)
    lp = _graded_log_integral(g, px, ps)
    return jnp.where(ok & jnp.isfinite(gm), lp, -jnp.inf)


def _gauss_line_jax(A, B, xa, xb):
    """log of the integral over (xa, xb) of phi(x) phi(A + Bx), and the ratio of the x-weighted
    integral to it; closed forms via phi(x) phi(A + Bx) = phi(A/s) phi(s (x + AB/s^2))."""
    s2 = 1.0 + B * B
    s = jnp.sqrt(s2)
    m = A * B / s2
    tl = jnp.where(jnp.isfinite(xa), s * (xa + m), xa)
    th = jnp.where(jnp.isfinite(xb), s * (xb + m), xb)
    ld = _log_diff_phi(tl, th)
    fl = jnp.where(jnp.isfinite(tl) & jnp.isfinite(ld), jnp.exp(-0.5 * tl * tl - _LOG_SQRT_2PI - ld), 0.0)
    fh = jnp.where(jnp.isfinite(th) & jnp.isfinite(ld), jnp.exp(-0.5 * th * th - _LOG_SQRT_2PI - ld), 0.0)
    return -0.5 * (A / s) ** 2 - _LOG_SQRT_2PI - jnp.log(s) + ld, (fl - fh) / s - m


def _cdp_cell_jax(kind, muR, sR, muF, sF, cR, cK, tlo, thi, corr):
    """log P of a CDP R/K/G cell (kind 1 = Remember, 2 = Know, 3 = Guess) as a one-dimensional
    integral over the standardised recollection axis, and its derivatives in
    (muR, sR, muF, sF, cR, cK, tlo, thi, corr); same construction as bvn_tail::cdp_cell. Given R,
    F is normal with mean muF + corr sF x and SD sF sqrt(1 - corr^2)."""
    inf = jnp.inf
    zero = jnp.zeros_like(muR)
    s2 = (1.0 - corr) * (1.0 + corr)
    s = jnp.sqrt(s2)
    sc = sF * s
    cs2 = corr / s2
    B = -(sR + corr * sF) / sc
    BK = -corr / s
    xcR = (cR - muR) / sR
    fin_hi, fin_lo = jnp.isfinite(thi), jnp.isfinite(tlo)
    xkh = jnp.where(fin_hi, (thi - cK - muR) / sR, inf)
    xkl = jnp.where(fin_lo, (tlo - cK - muR) / sR, -inf)
    Alo = jnp.where(fin_lo, (tlo - muR - muF) / sc, -inf)
    Ahi = jnp.where(fin_hi, (thi - muR - muF) / sc, inf)
    AK = (cK - muF) / sc
    Blo = jnp.where(fin_lo, B, 0.0)
    Bhi = jnp.where(fin_hi, B, 0.0)

    def vec(*c):
        return jnp.stack([jnp.broadcast_to(v, muR.shape) for v in c], axis=-1)
    Z = vec(*([zero] * 9))
    dxcR = vec(-1 / sR, -xcR / sR, zero, zero, 1 / sR, zero, zero, zero, zero)
    dxkh = jnp.where(fin_hi[..., None], vec(-1 / sR, -xkh / sR, zero, zero, zero, -1 / sR, zero, 1 / sR, zero), Z)
    dxkl = jnp.where(fin_lo[..., None], vec(-1 / sR, -xkl / sR, zero, zero, zero, -1 / sR, 1 / sR, zero, zero), Z)
    dAlo = jnp.where(fin_lo[..., None], vec(-1 / sc, zero, -1 / sc, -Alo / sF, zero, zero, 1 / sc, zero, Alo * cs2), Z)
    dAhi = jnp.where(fin_hi[..., None], vec(-1 / sc, zero, -1 / sc, -Ahi / sF, zero, zero, zero, 1 / sc, Ahi * cs2), Z)
    dB = vec(zero, -1 / sc, zero, sR / (sF * sc), zero, zero, zero, zero, -sR * corr / (sc * s2) - 1 / (s * s2))
    dBlo = jnp.where(fin_lo[..., None], dB, Z)
    dBhi = jnp.where(fin_hi[..., None], dB, Z)
    dAK = vec(zero, zero, -1 / sc, -AK / sF, zero, 1 / sc, zero, zero, AK * cs2)
    dBK = vec(zero, zero, zero, zero, zero, zero, zero, zero, -1 / (s * s2))

    def pick(c, a, da, b, db):
        return jnp.where(c, a, b), jnp.where(c[..., None], da, db)

    g1b, dg1b = pick(xkh <= xcR, xkh, dxkh, xcR, dxcR)
    g2a, dg2a = pick(xkh >= xkl, xkh, dxkh, xkl, dxkl)
    ub, dub = pick(xcR <= xkh, xcR, dxcR, xkh, dxkh)
    k1b, dk1b = pick(xkl <= ub, xkl, dxkl, ub, dub)
    ninf = jnp.full_like(muR, -inf)
    pinf = jnp.full_like(muR, inf)
    is1, is2 = kind == 1, kind == 2

    def sel(r, k, g):
        if isinstance(r, tuple):
            return tuple(sel(a, b, c) for a, b, c in zip(r, k, g))
        cond1 = is1[..., None] if r.ndim > muR.ndim else is1
        cond2 = is2[..., None] if r.ndim > muR.ndim else is2
        return jnp.where(cond1, r, jnp.where(cond2, k, g))

    pieces = [
        sel((xcR, pinf, Alo, Blo, Ahi, Bhi, dxcR, Z, dAlo, dBlo, dAhi, dBhi),
            (ninf, k1b, Alo, Blo, Ahi, Bhi, Z, dk1b, dAlo, dBlo, dAhi, dBhi),
            (xkl, g1b, Alo, Blo, AK, BK, dxkl, dg1b, dAlo, dBlo, dAK, dBK)),
        sel((pinf, pinf, Alo, Blo, Ahi, Bhi, Z, Z, Z, Z, Z, Z),
            (xkl, ub, AK, BK, Ahi, Bhi, dxkl, dub, dAK, dBK, dAhi, dBhi),
            (g2a, xcR, Alo, Blo, Ahi, Bhi, dg2a, dxcR, dAlo, dBlo, dAhi, dBhi)),
    ]
    lps = [_lin_log_integral_jax(p[2], p[3], p[4], p[5], p[0], p[1]) for p in pieces]
    lp = jnp.logaddexp(lps[0], lps[1])
    d = jnp.zeros(muR.shape + (9,), dtype=muR.dtype)
    for p, lpi in zip(pieces, lps):
        xa0, xb0, Al, Bl, Au, Bu, dxa, dxb, dAl, dBl, dAu, dBu = p
        live = jnp.isfinite(lpi) & jnp.isfinite(lp)
        xa = jnp.maximum(xa0, -40.0)
        xb = jnp.minimum(xb0, 40.0)

        def gat(x):
            l, u = _lin_bounds(Al, Bl, Au, Bu, x)
            return -0.5 * x * x - _LOG_SQRT_2PI + _log_diff_phi(l, u)
        w_xb = jnp.where(live & (xb0 < 40.0), jnp.exp(gat(xb) - lp), 0.0)
        w_xa = jnp.where(live & (xa0 > -40.0), -jnp.exp(gat(xa) - lp), 0.0)
        lJu, ru = _gauss_line_jax(jnp.where(jnp.isfinite(Au), Au, 0.0), Bu, xa, xb)
        lJl, rl = _gauss_line_jax(jnp.where(jnp.isfinite(Al), Al, 0.0), Bl, xa, xb)
        eu = jnp.where(live & jnp.isfinite(Au), jnp.exp(lJu - lp), 0.0)
        el = jnp.where(live & jnp.isfinite(Al), jnp.exp(lJl - lp), 0.0)
        for w, dv in ((w_xa, dxa), (w_xb, dxb), (-el, dAl), (-el * rl, dBl), (eu, dAu), (eu * ru, dBu)):
            d = d + w[..., None] * dv
    return lp, d


def _cdp_tail(flag, kind, muR, sR, muF, sF, cR, cK, tlo, thi, corr):
    """Exact CDP R/K/G cells where `flag` is set: (log p, d log p / d(muR, sR, muF, sF, cR, cK,
    tlo, thi, corr) as a [..., 9] array); zeros elsewhere. Skipped unless some cell is flagged."""
    args = jnp.broadcast_arrays(*[jnp.asarray(a) for a in (muR, sR, muF, sF, cR, cK, tlo, thi, corr)])
    dt = args[0].dtype

    def run(_):
        if _native_ok(args[0]):
            out = _native_call("cdp_cell", 10, [jnp.asarray(flag, dt), jnp.asarray(kind, dt), *args])
            return out[0], jnp.stack(out[1:], axis=-1)
        lp, d = _cdp_cell_jax(jnp.asarray(kind), *args)
        f = jnp.asarray(flag, bool)
        live = f & jnp.isfinite(lp)
        lp = jnp.where(live, lp, jax.lax.stop_gradient(jnp.where(f, -jnp.inf, 0.0)))
        return lp, jnp.where(live[..., None], d, 0.0)

    def skip(_):
        return jnp.zeros_like(args[0]), jnp.zeros(args[0].shape + (9,), dt)
    return jax.lax.cond(jnp.any(flag), run, skip, None)


_TAIL_CHUNK = 256


def _with_rect_tail(out, a1, b1, a2, b2, rho):
    """Replace cells below the tail switch by _rect_tail_jax, in fixed-size chunks that
    are skipped unless they hold such a cell (bounded memory for any N)."""
    shape = jnp.shape(out[0])
    flat = [jnp.ravel(jnp.broadcast_to(x, shape)) for x in (*out, a1, b1, a2, b2, rho)]
    n = flat[0].shape[0]
    pad = (-n) % _TAIL_CHUNK
    fill = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0)
    cols = [jnp.concatenate([x, jnp.full((pad,), v, x.dtype)]).reshape(-1, _TAIL_CHUNK)
            for x, v in zip(flat, fill)]

    def chunk(args):
        o, a = args[:6], args[6:]
        tail = o[0] < _TAIL_SWITCH

        def run(_):
            t = _rect_tail_jax(*a)
            return tuple(jnp.where(tail, ti, oi) for ti, oi in zip(t, o))
        return jax.lax.cond(jnp.any(tail), run, lambda _: tuple(o), None)
    res = jax.lax.map(chunk, tuple(cols))
    return tuple(r.reshape(-1)[:n].reshape(shape) for r in res)


def _bvn_rect(z1_lo, z1_hi, z2_lo, z2_hi, rho):
    """P(z1_lo < Z1 < z1_hi, z2_lo < Z2 < z2_hi; rho) and its exact derivatives in
    (z1_lo, z1_hi, z2_lo, z2_hi, rho): Drezner-Wesolowsky with the Gauss-Legendre order tiered
    on |rho|, and a log-space quadrature for cells below 1e-6."""
    args = [jnp.asarray(a) for a in (z1_lo, z1_hi, z2_lo, z2_hi, rho)]
    args = jnp.broadcast_arrays(*args)
    if _native_ok(args[0]):
        return _native_call("bvn_rect", 6, args)
    return _with_rect_tail(_dw_rect_grads_jax(*args), *args)


_GL4_X, _GL4_W = np.polynomial.legendre.leggauss(4)


def _probit_band_lw(z_lo, z_hi, w, has_lo, has_hi):
    """Pure-JAX probit (P, A, B) of _cell_lw, for cells evaluated at quadrature nodes."""
    inner = has_lo & has_hi
    Ph, dh, Qh = _probit_triple(z_hi)
    Pl, dl, Ql = _probit_triple(z_lo)
    D = _probit_band(z_lo, z_hi, Pl, Ph, Ql, Qh)
    dl, dh = _band_pdfs(z_lo, z_hi, dl, dh)
    narrow, lpn, rA, rB = _probit_narrow(jnp.where(inner, z_lo, 0.0), jnp.where(inner, w, 1.0), inner)
    Dn = jnp.exp(lpn)
    D = jnp.where(narrow, Dn, D)
    A = jnp.where(inner, jnp.where(narrow, rA * Dn, dh - dl), jnp.where(has_lo, -dl, dh))
    B = jnp.where(inner, jnp.where(narrow, rB * Dn, dh), 0.0)
    return D, A, B


def _strip_scale(t_lo, v, a_lo, a_hi, has_alo, has_ahi, rho):
    """How far a strip of width v in t varies the integrand phi(t) P(a_lo < A < a_hi | t), in units of v."""
    r = jnp.clip(rho, -_RHO_MAX, _RHO_MAX)
    s = jnp.sqrt((1.0 - r) * (1.0 + r))
    m = t_lo + 0.5 * v
    big = jnp.asarray(1e300, dtype=m.dtype)
    zl = jnp.where(has_alo, jnp.abs(a_lo - r * m) / s, big)
    zh = jnp.where(has_ahi, jnp.abs(a_hi - r * m) / s, big)
    zmin = jnp.where(has_alo | has_ahi, jnp.minimum(zl, zh), 0.0)
    return v * (jnp.maximum(1.0, jnp.abs(m)) + jnp.abs(r) / s * jnp.maximum(1.0, zmin))


def _rect_strip(t_lo, v, a_lo, a_hi, wa, has_alo, has_ahi, rho):
    """P(t_lo < T < t_lo + v, a_lo < A < a_hi; rho) for a narrow strip in t, by 4-point Gauss-Legendre in t
    with the cell in A by its anchor edge and width wa: (P, dP/dt_lo at fixed v, dP/dv, dP/da_anchor at fixed
    wa, dP/dwa, dP/drho), each the derivative of the quadrature itself."""
    r = jnp.clip(rho, -_RHO_MAX, _RHO_MAX)
    s = jnp.sqrt((1.0 - r) * (1.0 + r))
    x = jnp.asarray((_GL4_X + 1.0) / 2.0, dtype=t_lo.dtype)
    W = jnp.asarray(_GL4_W / 2.0, dtype=t_lo.dtype)
    e = lambda q: q[..., None]
    t = e(t_lo) + e(v) * x
    pt = jnp.exp(-0.5 * t * t - _LOG_SQRT_2PI)
    hlo, hhi = jnp.broadcast_to(e(has_alo), t.shape), jnp.broadcast_to(e(has_ahi), t.shape)
    zlo = jnp.where(hlo, (e(a_lo) - e(r) * t) / e(s), -_OPEN)
    zhi = jnp.where(hhi, (e(a_hi) - e(r) * t) / e(s), _OPEN)
    om = e(wa / s)
    C, Ac, Bc = _probit_band_lw(zlo, zhi, jnp.broadcast_to(om, t.shape), hlo, hhi)
    za = jnp.where(hlo, zlo, zhi)
    g = pt * C
    gp = pt * (-t * C - e(r / s) * Ac)
    Sg = jnp.sum(W * g, -1)
    dza = (-t + za * e(r / s)) / e(s)
    dom = om * e(r / (s * s))
    drho = v * jnp.sum(W * pt * (Ac * dza + Bc * dom), -1)
    return (v * Sg, v * jnp.sum(W * gp, -1), Sg + v * jnp.sum(W * x * gp, -1),
            v * jnp.sum(W * pt * Ac, -1) / s, v * jnp.sum(W * pt * Bc, -1) / s,
            jnp.where(jnp.abs(rho) < _RHO_MAX, drho, 0.0))


def _rect_lw(z1_lo, z1_hi, w1, z2_lo, z2_hi, w2, rho):
    """_bvn_rect with each dimension's cell by its anchor edge (the lower one when finite) and its width w,
    the criterion gap over the SD, so that a narrow rating band keeps full accuracy:
    (P, A1, B1, A2, B2, dP/drho), A = dP/d anchor at fixed width and B = dP/dw (0 for an open cell)."""
    z1_lo, z1_hi, w1, z2_lo, z2_hi, w2, rho = jnp.broadcast_arrays(z1_lo, z1_hi, w1, z2_lo, z2_hi, w2, rho)
    h1l, h1h = z1_lo > -0.5 * _OPEN, z1_hi < 0.5 * _OPEN
    h2l, h2h = z2_lo > -0.5 * _OPEN, z2_hi < 0.5 * _OPEN
    in1, in2 = h1l & h1h, h2l & h2h
    p, d1l, d1h, d2l, d2h, dr = _bvn_rect(z1_lo, z1_hi, z2_lo, z2_hi, rho)
    out = (p, d1l + d1h, jnp.where(in1, d1h, 0.0), d2l + d2h, jnp.where(in2, d2h, 0.0), dr)
    v1, v2 = jnp.where(in1, w1, 1e-3), jnp.where(in2, w2, 1e-3)
    l1, l2 = jnp.where(in1, z1_lo, 0.0), jnp.where(in2, z2_lo, 0.0)
    n2 = in2 & (_strip_scale(l2, v2, z1_lo, z1_hi, h1l, h1h, rho) < _NARROW_U)
    n1 = in1 & ~n2 & (_strip_scale(l1, v1, z2_lo, z2_hi, h2l, h2h, rho) < _NARROW_U)

    def along2(o):
        P, At, Bt, Aa, Ba, R = _rect_strip(l2, jnp.where(n2, v2, 1e-3), z1_lo, z1_hi, jnp.where(in1, w1, 1.0),
                                           h1l, h1h, rho)
        return tuple(jnp.where(n2, a, b) for a, b in zip((P, Aa, Ba, At, Bt, R), o))

    def along1(o):
        P, At, Bt, Aa, Ba, R = _rect_strip(l1, jnp.where(n1, v1, 1e-3), z2_lo, z2_hi, jnp.where(in2, w2, 1.0),
                                           h2l, h2h, rho)
        return tuple(jnp.where(n1, a, b) for a, b in zip((P, At, Bt, Aa, Ba, R), o))
    out = jax.lax.cond(jnp.any(n2), along2, lambda o: o, out)
    return jax.lax.cond(jnp.any(n1), along1, lambda o: o, out)


def _lw_scale(gA, gB, z_a, w, inner, z_b):
    """gA z_a + gB w, a cell's derivative in log sigma less its sign (gA = d/d anchor, gB = d/d width, z_a the
    anchor, w the width, z_b the far edge): for a cell at least half as wide as its anchor is far out taken as
    (gA - gB) z_a + gB z_b, equal since z_b = z_a + w, which does not cancel there (the width form keeps
    narrow cells, where the edge form would)."""
    gB = jnp.where(inner, gB, 0.0)
    wi = jnp.where(inner, w, 0.0)
    wide = inner & (wi > 0.5 * jnp.maximum(1.0, jnp.abs(z_a)))
    return jnp.where(wide, (gA - gB) * z_a + gB * jnp.where(wide, z_b, 0.0), gA * z_a + gB * wi)


def _lw_chain(gA, gB, z_a, w, inner, inv_s, z_b):
    """d/d(mu, sigma, criterion mid) of a cell entering with gA = d/d anchor and gB = d/d width, the anchor
    (c_a - mu) / sigma, the width gap / sigma and the far edge z_b; the gap gradients follow from gA, gB with
    _lw_gap_grads."""
    return -gA * inv_s, -_lw_scale(gA, gB, z_a, w, inner, z_b) * inv_s, gA * inv_s


def _lw_gap_grads(gA, gB, k_a, own, inner, gap_mask, gap_derivs):
    """[N, n_gaps] gap gradients of a cell: the anchor criterion's mask times gA plus the cell's own gap
    times gB, gA and gB already over the SD."""
    gB = jnp.where(inner, gB, 0.0)
    return (jnp.asarray(gap_mask)[:, k_a].T * gA[:, None] + own.T * gB[:, None]) * gap_derivs


def _bvn_cdf_multi(pairs, extras, rho):
    """P(Z1 < h, Z2 < k; rho) and its derivatives for (h, k) pairs sharing rho, plus accurate marginals.
    Returns, per pair, (P, dP/dh, dP/dk, dP/drho, Q(h), Phi'(h), Q(k), Phi'(k)) and, per extra x, (Q(x), Phi'(x)),
    where Q is the tail mass min(Phi, 1 - Phi). One native call when available."""
    n, m = len(pairs), len(extras)
    arrs = jnp.broadcast_arrays(*[jnp.asarray(a) for pair in pairs for a in pair],
                                *[jnp.asarray(x) for x in extras], jnp.asarray(rho))
    hs = arrs[0:2 * n:2]
    ks = arrs[1:2 * n:2]
    xs = arrs[2 * n:2 * n + m]
    r = arrs[-1]
    if _native_ok(r) and (n, m) in ((2, 1), (4, 2)):
        out = _native_call(f"bvn_cdf{n}", 8 * n + 2 * m, [*hs, *ks, *xs, r])
        return [tuple(out[8 * q:8 * q + 8]) for q in range(n)], [tuple(out[8 * n + 2 * j:8 * n + 2 * j + 2])
                                                                 for j in range(m)]
    res = []
    for h, k in zip(hs, ks):
        p, dh, dk, dr = _dw_cdf_grads_jax(h, k, r)
        _, dPh, Qh = _acc_phi_and_deriv(h)
        _, dPk, Qk = _acc_phi_and_deriv(k)
        res.append((p, dh, dk, dr, Qh, dPh, Qk, dPk))
    ext = []
    for x in xs:
        _, dPx, Qx = _acc_phi_and_deriv(x)
        ext.append((Qx, dPx))
    return res, ext


def _phi_from_q(x, q):
    return jnp.where(x >= 0, 1.0 - q, q)


def _band_from_q(z_lo, z_hi, q_lo, q_hi):
    return jnp.where(z_lo >= 0, q_lo - q_hi, _phi_from_q(z_hi, q_hi) - _phi_from_q(z_lo, q_lo))


def _vrdp2d_native(c1_hi, c1_lo, c2_hi, c2_lo, has1_hi, has1_lo, has2_hi, has2_lo,
                   dprime_F, dprime_R, source_d, lambda_val, sigma_item, sigma_S):
    """vrdp2d log p and its partials (dprime_F, dprime_R, source_d, lambda, sigma_item,
    sigma_S, c1_hi, c1_lo, c2_hi, c2_lo) in one native pass; None when unavailable."""
    if not _native_ok(c1_hi):
        return None
    dt = jnp.asarray(c1_hi).dtype
    f = lambda a: jnp.asarray(a, dtype=dt)
    args = [f(c1_hi), f(c1_lo), f(c2_hi), f(c2_lo), f(has1_hi), f(has1_lo), f(has2_hi), f(has2_lo),
            f(dprime_F), f(dprime_R), f(source_d), f(lambda_val), f(sigma_item), f(sigma_S)]
    return _native_call("vrdp2d", 11, jnp.broadcast_arrays(*args))


def _thresh_layout(K):
    """The mid-anchor + gaps threshold layout for K categories: (n_thresh, mid_idx, gap_mask,
    build), gap_mask[g, t] the sign with which gap g enters threshold t, build(mid, gap_vals) the
    [N, n_thresh] thresholds."""
    n_thresh = K - 1
    n_gaps = n_thresh - 1
    mid_idx = (n_thresh - 1) // 2
    n_upper = n_thresh - mid_idx - 1
    gap_mask = np.zeros((n_gaps, n_thresh), dtype=np.float64)
    for g in range(n_gaps):
        if g < n_upper:
            gap_mask[g, mid_idx + 1 + g:] = 1.0
        else:
            gap_mask[g, :mid_idx - (g - n_upper)] = -1.0

    def build(thresh_mid, gap_vals):
        parts = []
        if mid_idx > 0:
            parts.append(thresh_mid[:, None] - jnp.cumsum(gap_vals[:, n_upper:], axis=1)[:, ::-1])
        parts.append(thresh_mid[:, None])
        if n_upper > 0:
            parts.append(thresh_mid[:, None] + jnp.cumsum(gap_vals[:, :n_upper], axis=1))
        return jnp.concatenate(parts, axis=1)

    return n_thresh, mid_idx, gap_mask, build


_NARROW_U = 0.05
_LOG2 = 0.6931471805599453


def _own_gap(K):
    """For K categories, own_idx[y] the gap that is cell y's width (0 for the two open cells) and
    own_mask[g, y] its indicator."""
    n_thresh = K - 1
    n_gaps = n_thresh - 1
    mid_idx = (n_thresh - 1) // 2
    n_upper = n_thresh - mid_idx - 1
    own_idx = np.zeros(K, dtype=np.int32)
    own_mask = np.zeros((max(n_gaps, 0), K), dtype=np.float64)
    for y in range(1, K - 1):
        g = y - 1 - mid_idx if y - 1 >= mid_idx else n_upper + mid_idx - y
        own_idx[y] = g
        own_mask[g, y] = 1.0
    return own_idx, own_mask


def _probit_narrow(z_lo, w, inner):
    """For cells narrow against the scale of the density (w max(1, |m|) < _NARROW_U, m the midpoint):
    the mask, log P by the midpoint series, and dP/dz_lo / P and dP/dw / P, all from (z_lo, w) so that
    no edge difference cancels."""
    m0 = z_lo + 0.5 * w
    narrow = inner & (w * jnp.maximum(1.0, jnp.abs(m0)) < _NARROW_U)
    ws = jnp.where(narrow, w, 1e-3)
    m = jnp.where(narrow, m0, 0.0)
    m2, w2 = m * m, ws * ws
    he2 = m2 - 1.0
    he4 = m2 * (m2 - 6.0) + 3.0
    he6 = m2 * (m2 * (m2 - 15.0) + 45.0) - 15.0
    s = w2 * (he2 / 24.0 + w2 * (he4 / 1920.0 + w2 * he6 / 322560.0))
    lp = jnp.log(ws) - 0.5 * m2 - _LOG_SQRT_2PI + jnp.log1p(s)
    k = jnp.exp(-0.125 * w2) / (ws * (1.0 + s))
    return narrow, lp, -2.0 * k * jnp.sinh(0.5 * m * ws), k * jnp.exp(-0.5 * m * ws)


def _probit_cell_lw_native(z_lo, z_hi, w, has_lo, has_hi):
    """log P, d log P / dz_a at fixed width and d log P / dw of the probit cell of _cell_lw, from the native
    log-space cell kernel and, for narrow cells, the midpoint series."""
    inner = has_lo & has_hi
    lp, g_lo, g_hi = _probit_cell_native(z_lo, z_hi)
    narrow, lpn, rA, rB = _probit_narrow(jnp.where(inner, z_lo, 0.0), jnp.where(inner, w, 1.0), inner)
    lp = jnp.where(narrow, lpn, lp)
    gA = jnp.where(narrow, rA, jnp.where(has_hi, g_hi, 0.0) + jnp.where(has_lo, g_lo, 0.0))
    gB = jnp.where(narrow, rB, jnp.where(inner, g_hi, 0.0))
    return lp, gA, gB


def _logcosh(x):
    a = jnp.abs(x)
    return a + jnp.log1p(jnp.exp(-2.0 * a)) - _LOG2


def _log_logit_pdf(x):
    a = jnp.abs(x)
    return -a - 2.0 * jnp.log1p(jnp.exp(-a))


def _logit_inner(z_lo, w):
    h = z_lo + w
    x = 0.5 * w
    lp = x + jnp.log(-jnp.expm1(-2.0 * x)) - 2.0 * _LOG2 - _logcosh(0.5 * h) - _logcosh(0.5 * z_lo)
    fh, fl = _logit_pdf(h), _logit_pdf(z_lo)
    m = z_lo + x
    am = jnp.maximum(jnp.abs(m), 1e-300)
    xs = jnp.minimum(x, 0.5)
    lA = (2.0 * _LOG2 + am + jnp.log(-jnp.expm1(-2.0 * am)) - _LOG2 + jnp.log(jnp.sinh(xs)) + _log_logit_pdf(h) +
          _log_logit_pdf(z_lo))
    A = jnp.where(w < 1.0, -jnp.sign(m) * jnp.exp(lA), fh - fl)
    return lp, A, fh


def _cloglog_inner(z_lo, w):
    l = jnp.minimum(z_lo, _CLOGLOG_HI)
    el = jnp.exp(l)
    wl = jnp.maximum(w, 1e-300)
    wc = jnp.minimum(wl, 700.0)
    gap = jnp.exp(jnp.minimum(l + wl + jnp.log(-jnp.expm1(-wl)), 700.0))
    lp = -el + jnp.log(-jnp.expm1(-gap))
    fh, fl = _cloglog_pdf(z_lo + w), jnp.exp(l - el)
    return lp, jnp.where(w < 1.0, fl * jnp.expm1(wc - gap), fh - fl), fh


def _cell_lw(link_name, z_lo, z_hi, w, has_lo, has_hi):
    """(P, A, B, z_a) for one cell: an interior cell is (z_lo, z_lo + w), w the criterion gap over the SD,
    with A = dP/dz_lo at fixed w and B = dP/dw; an open cell has A = dP/dz_a in its one finite edge z_a
    and B = 0."""
    inner = has_lo & has_hi
    ws = jnp.where(inner, w, 1.0)
    ls = jnp.where(inner, z_lo, 0.0)
    if link_name == "probit":
        D, neg_pdf_lo, pdf_hi = _phi_diff(z_lo, z_hi)
        pdf_lo = -neg_pdf_lo
        narrow, lpn, rA, rB = _probit_narrow(ls, ws, inner)
        Dn = jnp.exp(lpn)
        D = jnp.where(narrow, Dn, D)
        A_in = jnp.where(narrow, rA * Dn, pdf_hi - pdf_lo)
        B_in = jnp.where(narrow, rB * Dn, pdf_hi)
    else:
        pdf = _link_fns(link_name)[1]
        pdf_lo, pdf_hi = pdf(z_lo), pdf(z_hi)
        lpi, A_in, B_in = (_logit_inner if link_name == "logit" else _cloglog_inner)(ls, ws)
        D = jnp.where(inner, jnp.exp(lpi), _link_diff_fn(link_name)(z_lo, z_hi))
    A = jnp.where(inner, A_in, jnp.where(has_lo, -pdf_lo, pdf_hi))
    B = jnp.where(inner, B_in, 0.0)
    return D, A, B, jnp.where(has_lo, z_lo, z_hi)


def _cells_from_thresh(link_name, y_idx, K, thresh, gap_vals, own_idx, mu, sigma):
    """The cells of responses y_idx (0-based) under N(mu, sigma) and the [N, K - 1] criteria built
    from [N, n_gaps] gap values: (P, A, B, z_a, w, k_a, inner, z_hi), k_a the criterion of z_a."""
    n_thresh = K - 1
    obs = jnp.arange(y_idx.shape[0])
    k_hi = jnp.clip(y_idx, 0, n_thresh - 1)
    k_lo = jnp.clip(y_idx - 1, 0, n_thresh - 1)
    has_hi, has_lo = y_idx < K - 1, y_idx > 0
    z_hi = jnp.where(has_hi, (thresh[obs, k_hi] - mu) / sigma, _OPEN)
    z_lo = jnp.where(has_lo, (thresh[obs, k_lo] - mu) / sigma, -_OPEN)
    if gap_vals.shape[1]:
        w = gap_vals[obs, jnp.asarray(own_idx)[y_idx]] / sigma
    else:
        w = jnp.ones_like(z_lo)
    D, A, B, z_a = _cell_lw(link_name, z_lo, z_hi, w, has_lo, has_hi)
    return D, A, B, z_a, w, jnp.where(has_lo, k_lo, k_hi), has_lo & has_hi, z_hi


def _cell_lw_grads(c, cell, sigma, gap_mask, own):
    """d log-lik / d (mu, sigma, criterion mid, [n_gaps, N] gap values) of a cell entering the
    log-likelihood with weight c = d log-lik / dP."""
    _, A, B, z_a, w, k_a, inner, z_b = cell
    gA = c * A
    gB = jnp.where(inner, c * B, 0.0)
    d_gap_val = jnp.asarray(gap_mask)[:, k_a] * gA[None, :] + own * gB[None, :]
    return (-gA / sigma, -_lw_scale(gA, gB, z_a, w, inner, z_b) / sigma, gA / sigma, d_gap_val / sigma)


def _gap_vals_and_derivs(gaps, softplus):
    if softplus:
        return jax.nn.softplus(gaps), jax.nn.sigmoid(gaps)
    g = jnp.exp(gaps)
    return g, g


def _make_fused_batch_loglik(K, gap_link_is_softplus, link_name="probit"):
    """Fully-vectorized batch log-likelihood with custom JVP: cells from the mid-anchor + gaps
    criteria, each interior cell by its lower edge and its width (the gap over the SD) so that
    narrow cells keep full accuracy."""
    n_thresh, mid_idx, gap_mask, build = _thresh_layout(K)
    own_idx, own_mask = _own_gap(K)

    def cells(y, mu, sigma_val, thresh_mid, gap_vals):
        return _cells_from_thresh(link_name, y - 1, K, build(thresh_mid, gap_vals), gap_vals, own_idx, mu,
                                  sigma_val)

    def native(y, mu, sigma_val, thresh_mid, gap_vals):
        y_idx = y - 1
        obs = jnp.arange(y.shape[0])
        thresh = build(thresh_mid, gap_vals)
        k_hi = jnp.clip(y_idx, 0, n_thresh - 1)
        k_lo = jnp.clip(y_idx - 1, 0, n_thresh - 1)
        has_hi, has_lo = y_idx < K - 1, y_idx > 0
        z_hi = jnp.where(has_hi, (thresh[obs, k_hi] - mu) / sigma_val, _OPEN)
        z_lo = jnp.where(has_lo, (thresh[obs, k_lo] - mu) / sigma_val, -_OPEN)
        w = (gap_vals[obs, jnp.asarray(own_idx)[y_idx]] / sigma_val if gap_vals.shape[1]
             else jnp.ones_like(z_lo))
        inner = has_lo & has_hi
        lp, gA, gB = _probit_cell_lw_native(z_lo, z_hi, w, has_lo, has_hi)
        cell = (None, gA, gB, jnp.where(has_lo, z_lo, z_hi), w, jnp.where(has_lo, k_lo, k_hi), inner, z_hi)
        return lp, cell

    @custom_jvp
    def _batch_fused(y, mu, sigma_val, thresh_mid, gaps):
        gap_vals = _gap_vals_and_derivs(gaps, gap_link_is_softplus)[0]
        if _use_probit_native(link_name, mu):
            return native(y, mu, sigma_val, thresh_mid, gap_vals)[0]
        return jnp.log(jnp.clip(cells(y, mu, sigma_val, thresh_mid, gap_vals)[0], _pfloor()))

    @_batch_fused.defjvp
    def _batch_fused_jvp(primals, tangents):
        y, mu, sigma_val, thresh_mid, gaps = primals
        _, d_mu, d_sigma_val, d_thresh_mid, d_gaps = tangents
        gap_vals, gap_derivs = _gap_vals_and_derivs(gaps, gap_link_is_softplus)
        if _use_probit_native(link_name, mu):
            val, cell = native(y, mu, sigma_val, thresh_mid, gap_vals)
            c = jnp.ones_like(val)
        else:
            cell = cells(y, mu, sigma_val, thresh_mid, gap_vals)
            p = jnp.clip(cell[0], _pfloor())
            val = jnp.log(p)
            c = jnp.where(cell[0] > _pfloor(), 1.0 / p, 0.0)
        g_mu, g_sigma, g_mid, g_gap_val = _cell_lw_grads(c, cell, sigma_val, gap_mask, jnp.asarray(own_mask)[:, y - 1])
        tangent_out = (g_mu * d_mu + g_sigma * d_sigma_val + g_mid * d_thresh_mid +
                       jnp.sum((g_gap_val * gap_derivs.T).T * d_gaps, axis=1))
        return val, tangent_out

    return _batch_fused


_METAD_DEEP = 1e20


def _metad_cell_jax(z_a, z_lo, z_hi, z_c, old):
    """(lp, d/dz_a, d/dz_lo, d/dz_hi, d/dz_c) of log P(R | S) + log P'(cell) - log P'(R | S), every
    term an upper-tail probability after reflecting "new" responses."""
    s = jnp.where(old, 1.0, -1.0).astype(z_a.dtype)
    ua, uc = s * z_a, s * z_c
    ul, uh = jnp.where(old, z_lo, -z_hi), jnp.where(old, z_hi, -z_lo)
    fl = _pfloor()
    Pa, da, qa = _probit_triple(ua)
    Pc, dc, qc = _probit_triple(uc)
    Pl, dl, ql = _probit_triple(ul)
    Ph, dh, qh = _probit_triple(uh)
    Qa = jnp.where(ua >= 0, qa, 1.0 - qa)
    Qc = jnp.where(uc >= 0, qc, 1.0 - qc)
    D = _probit_band(ul, uh, Pl, Ph, ql, qh)
    dl, dh = _band_pdfs(ul, uh, dl, dh)
    ia = jnp.where(Qa > fl, 1.0 / jnp.maximum(Qa, fl), 0.0)
    ib = jnp.where(D > fl, 1.0 / jnp.maximum(D, fl), 0.0)
    ic = jnp.where(Qc > fl, 1.0 / jnp.maximum(Qc, fl), 0.0)
    lp = jnp.log(jnp.maximum(Qa, fl)) + jnp.log(jnp.maximum(D, fl) / jnp.maximum(Qc, fl))
    ga, gc, bl, bh = -s * da * ia, s * dc * ic, -dl * ib, dh * ib
    deep = (Qa < _METAD_DEEP * fl) | (D < _METAD_DEEP * fl) | (Qc < _METAD_DEEP * fl)
    pick = lambda x, safe: jnp.where(deep, x, safe)
    ua_s, uc_s, ul_s, uh_s = pick(ua, 0.0), pick(uc, 0.0), pick(ul, 0.0), pick(uh, 1.0)
    lQa, lQc, lD = _log_q(ua_s), _log_q(uc_s), _log_diff_phi(ul_s, uh_s)
    lphi = lambda x: -0.5 * x * x - _LOG_SQRT_2PI
    edge = lambda x: jnp.abs(x) < 0.5 * _OPEN
    ga = jnp.where(deep, -s * jnp.exp(lphi(ua_s) - lQa), ga)
    gc = jnp.where(deep, s * jnp.exp(lphi(uc_s) - lQc), gc)
    bl = jnp.where(deep, jnp.where(edge(ul_s), -jnp.exp(lphi(ul_s) - lD), 0.0), bl)
    bh = jnp.where(deep, jnp.where(edge(uh_s), jnp.exp(lphi(uh_s) - lD), 0.0), bh)
    lp = jnp.where(deep, lQa + lD - lQc, lp)
    return lp, ga, jnp.where(old, bl, -bh), jnp.where(old, bh, -bl), gc


def _metad_cell_link(z_a, z_lo, z_hi, z_c, old, link_name):
    """The meta-d' cell of _metad_cell_jax under a logistic or Gumbel-min latent distribution: the
    survival on "old" responses and the CDF on "new" ones taken directly, the link not being symmetric."""
    cdf, pdf = _link_fns(link_name)
    diff = _link_diff_fn(link_name)
    if link_name == "logit":
        ccdf = lambda x: jax_expit(-x)
    else:
        ccdf = lambda x: jnp.exp(-jnp.exp(jnp.minimum(x, _CLOGLOG_HI)))
    fl = _pfloor()
    lo_open = z_lo <= -0.5 * _OPEN
    hi_open = z_hi >= 0.5 * _OPEN
    zl = jnp.where(lo_open, 0.0, z_lo)
    zh = jnp.where(hi_open, 0.0, z_hi)
    Pa = jnp.where(old, ccdf(z_a), cdf(z_a))
    Pc = jnp.where(old, ccdf(z_c), cdf(z_c))
    D = jnp.where(lo_open & hi_open, 1.0, jnp.where(lo_open, cdf(zh), jnp.where(hi_open, ccdf(zl), diff(zl, zh))))
    ia = jnp.where(Pa > fl, 1.0 / jnp.maximum(Pa, fl), 0.0)
    ib = jnp.where(D > fl, 1.0 / jnp.maximum(D, fl), 0.0)
    ic = jnp.where(Pc > fl, 1.0 / jnp.maximum(Pc, fl), 0.0)
    s = jnp.where(old, -1.0, 1.0).astype(z_a.dtype)
    lp = jnp.log(jnp.maximum(Pa, fl)) + jnp.log(jnp.maximum(D, fl) / jnp.maximum(Pc, fl))
    ga, gl, gh, gc = s * pdf(z_a) * ia, -pdf(zl) * ib, pdf(zh) * ib, -s * pdf(z_c) * ic
    deep = (Pa < _METAD_DEEP * fl) | (D < _METAD_DEEP * fl) | (Pc < _METAD_DEEP * fl)
    pick = lambda x, safe: jnp.where(deep, x, safe)
    za_s, zc_s, zl_s, zh_s = pick(z_a, 0.0), pick(z_c, 0.0), pick(zl, 0.0), pick(zh, 1.0)
    lS, lF, lpdf = _link_log_tails(link_name)
    lPa = jnp.where(old, lS(za_s), lF(za_s))
    lPc = jnp.where(old, lS(zc_s), lF(zc_s))
    med = 0.0 if link_name == "logit" else _CLOGLOG_MED
    inner = ~lo_open & ~hi_open
    il, ih = jnp.where(inner, zl_s, 0.0), jnp.where(inner, zh_s, 1.0)
    l_in = jnp.where(il >= med, lS(il) + jnp.log(-jnp.expm1(lS(ih) - lS(il))),
                     lF(ih) + jnp.log(-jnp.expm1(lF(il) - lF(ih))))
    lD = jnp.where(lo_open & hi_open, 0.0, jnp.where(lo_open, lF(zh_s), jnp.where(hi_open, lS(zl_s), l_in)))
    ga = jnp.where(deep, s * jnp.exp(lpdf(za_s) - lPa), ga)
    gc = jnp.where(deep, -s * jnp.exp(lpdf(zc_s) - lPc), gc)
    gl = jnp.where(deep, -jnp.exp(lpdf(zl_s) - lD), gl)
    gh = jnp.where(deep, jnp.exp(lpdf(zh_s) - lD), gh)
    lp = jnp.where(deep, lPa + lD - lPc, lp)
    return lp, ga, jnp.where(lo_open, 0.0, gl), jnp.where(hi_open, 0.0, gh), gc


def _link_log_tails(link_name):
    """log survival, log CDF and log density of the logistic or Gumbel-min latent distribution, each
    accurate far into its tails."""
    if link_name == "logit":
        return (lambda x: -jax.nn.softplus(x)), (lambda x: -jax.nn.softplus(-x)), _log_logit_pdf
    lS = lambda x: -jnp.exp(jnp.minimum(x, _CLOGLOG_HI))
    lF = lambda x: jnp.where(x < -20.0, x - 0.5 * jnp.exp(jnp.minimum(x, -20.0)),
                             jnp.log(-jnp.expm1(-jnp.exp(jnp.maximum(jnp.minimum(x, _CLOGLOG_HI), -20.0)))))
    lpdf = lambda x: jnp.minimum(x, _CLOGLOG_HI) - jnp.exp(jnp.minimum(x, _CLOGLOG_HI))
    return lS, lF, lpdf


def _metad_cell(z_a, z_lo, z_hi, z_c, old, link_name="probit"):
    if link_name != "probit":
        return _metad_cell_link(z_a, z_lo, z_hi, z_c, old, link_name)
    if _native_ok(z_a):
        return _native_call("metad_cell", 5, [z_a, z_lo, z_hi, z_c, old.astype(z_a.dtype)])
    return _metad_cell_jax(z_a, z_lo, z_hi, z_c, old)


_PCQ_GL_X, _PCQ_GL_W = np.polynomial.legendre.leggauss(64)
_PCQ_DROP = 40.0


def _pcq_jax(a, c):
    """(log int phi(z) prod_j Phi(a_j z + c_j) dz, d/dc, d/da) over the trailing axis: the log-concave
    integrand split at its mode, each side by Gauss-Legendre out to where it has fallen by e^-40; the
    pure-JAX reference of the native kernel (64 nodes a side, its finest tier)."""
    X = jnp.asarray(_PCQ_GL_X, dtype=a.dtype)
    W = jnp.asarray(_PCQ_GL_W, dtype=a.dtype)

    def lam(x):
        return jnp.exp(-0.5 * x * x - _LOG_SQRT_2PI - _log_q(-x))

    def h(z):
        return -0.5 * z * z + jnp.sum(_log_q(-(a * z[..., None] + c)), -1)

    def gh(z):
        x = a * z[..., None] + c
        lx = lam(x)
        return -z + jnp.sum(a * lx, -1), -1.0 - jnp.sum(a * a * lx * (x + lx), -1)

    def newton(_, st):
        z, zl, zr = st
        g, h2 = gh(z)
        zl = jnp.where(g > 0.0, z, zl)
        zr = jnp.where(g > 0.0, zr, z)
        zn = z - g / h2
        return jnp.where((zn > zl) & (zn < zr), zn, 0.5 * (zl + zr)), zl, zr

    shape = a.shape[:-1]
    zm, _, _ = jax.lax.fori_loop(0, 80, newton, (jnp.zeros(shape, a.dtype), jnp.full(shape, -60.0, a.dtype),
                                                  jnp.full(shape, 60.0, a.dtype)))
    hm = h(zm)
    target = hm - _PCQ_DROP

    def edge(side):
        def step(_, st):
            z, zi, zo = st
            hz = h(z)
            g, _ = gh(z)
            zo = jnp.where(hz < target, z, zo)
            zi = jnp.where(hz < target, zi, z)
            zn = z - (hz - target) / g
            zn = jnp.where((zn - zi) * (zn - zo) < 0.0, zn, 0.5 * (zi + zo))
            return zn, zi, zo
        zo = zm + side * np.sqrt(2.0 * _PCQ_DROP)
        return jax.lax.fori_loop(0, 60, step, (zo, zm, zo))[0]

    zl = edge(-1.0)
    zr = edge(1.0)
    lo = jnp.stack([zl, zm], -1)
    hi = jnp.stack([zm, zr], -1)
    hh = 0.5 * (hi - lo)
    zi = ((lo + hh)[..., None] + hh[..., None] * X).reshape(shape + (2 * X.shape[0],))
    wi = (hh[..., None] * W).reshape(zi.shape)
    hv = jax.vmap(h, in_axes=-1, out_axes=-1)(zi) - hm[..., None]
    e = wi * jnp.exp(hv)
    sm = jnp.sum(e, -1)
    w = e / sm[..., None]
    li = lam(a[..., None, :] * zi[..., :, None] + c[..., None, :])
    dc = jnp.sum(w[..., None] * li, -2)
    da = jnp.sum(w[..., None] * li * zi[..., None], -2)
    return hm + jnp.log(sm) - _LOG_SQRT_2PI, dc, da


def _pcq_raw(a, c):
    if _native_ok(a):
        dt = a.dtype
        target = "bayesroc_pcq" if dt == jnp.float64 else "bayesroc_pcq_f32"
        return tuple(jax.ffi.ffi_call(target, (jax.ShapeDtypeStruct(a.shape[:-1], dt), jax.ShapeDtypeStruct(a.shape, dt),
                                               jax.ShapeDtypeStruct(a.shape, dt)), vmap_method="broadcast_all")(
            jnp.asarray(a, dt), jnp.asarray(c, dt)))
    return _pcq_jax(a, c)


_LN_R = 8.5
_LN_GL_X, _LN_GL_W = np.polynomial.legendre.leggauss(56)
_LN_GH_X, _LN_GH_W = np.polynomial.hermite_e.hermegauss(24)
_LN_GH_LW = np.log(_LN_GH_W) + 0.5 * _LN_GH_X ** 2
_LN_FLOOR = 1e-300


def _ln_lam(x):
    return jnp.exp(-0.5 * x * x - _LOG_SQRT_2PI - _log_q(-x))


def _ln_mode(a, l, s):
    w = jnp.exp(l)
    x = a - w
    L = _ln_lam(x)
    sw = s * w
    zl = jnp.maximum(-L * sw, -40.0)
    d2 = -1.0 - L * s * sw - L * (x + L) * sw * sw
    z = jnp.clip(L * sw / d2, zl, 0.0)

    def body(_, st):
        z, zl, zr = st
        w = jnp.exp(l + s * z)
        x = a - w
        L = _ln_lam(x)
        sw = s * w
        g = -z - L * sw
        d2 = -1.0 - L * s * sw - L * (x + L) * sw * sw
        zl = jnp.where(g > 0.0, z, zl)
        zr = jnp.where(g > 0.0, zr, z)
        zn = z - g / d2
        zn = jnp.where((zn > zl) & (zn < zr), zn, 0.5 * (zl + zr))
        return zn, zl, zr

    z, _, _ = jax.lax.fori_loop(0, 60, body, (z, zl, jnp.zeros_like(z)))
    w = jnp.exp(l + s * z)
    x = a - w
    L = _ln_lam(x)
    sw = s * w
    return z, -1.0 - L * s * sw - L * (x + L) * sw * sw


def _ln_interval(lo, hi):
    return jnp.where(lo >= 0.0, jax.scipy.special.ndtr(-lo) - jax.scipy.special.ndtr(-hi),
                     jax.scipy.special.ndtr(hi) - jax.scipy.special.ndtr(lo))


_LN_GH8_X, _LN_GH8_W = np.polynomial.hermite_e.hermegauss(8)
_LN_GH8_W = _LN_GH8_W / np.sqrt(2.0 * np.pi)
_LN_WSER = 0.01


def _ln_pdf(t):
    return jnp.exp(-0.5 * t * t - _LOG_SQRT_2PI)


def _ln_D(a, w, Pa, Qa):
    """Phi(a) - Phi(a - w) = P(0 < T <= w), by a midpoint Hermite series for w <= 0.01 (no cancellation)."""
    ndtr = jax.scipy.special.ndtr
    u = a - w
    sub = jnp.where(u >= 0.0, ndtr(-u) - Qa, Pa - ndtr(u))
    m = a - 0.5 * w
    h2 = 0.25 * w * w
    m2 = m * m
    he2, he4, he6 = m2 - 1.0, m2 * m2 - 6.0 * m2 + 3.0, m2 * m2 * m2 - 15.0 * m2 * m2 + 45.0 * m2 - 15.0
    ser = w * _ln_pdf(m) * (1.0 + h2 * (he2 / 6.0 + h2 * (he4 / 120.0 + h2 * he6 / 5040.0)))
    return jnp.maximum(jnp.where(w <= _LN_WSER, ser, sub), 1e-300)


def _ln_dD_da(a, w, pa):
    """d/da of _ln_D, with the same series for small w."""
    m = a - 0.5 * w
    h2 = 0.25 * w * w
    m2 = m * m
    he3, he5 = m * (m2 - 3.0), m * (m2 * m2 - 10.0 * m2 + 15.0)
    he7 = m * (m2 * m2 * m2 - 21.0 * m2 * m2 + 105.0 * m2 - 105.0)
    ser = -w * _ln_pdf(m) * (m + h2 * (he3 / 6.0 + h2 * (he5 / 120.0 + h2 * he7 / 5040.0)))
    return jnp.where(w <= _LN_WSER, ser, pa - _ln_pdf(a - w))


def _ln_mode_k(a, l, s, zs, Pa, Qa):
    """Mode of K's integrand phi(z) D(w(z)) on (0, min(zs, 40)) and its curvature; `atzs` where it has no
    interior mode there (the integrand rises up to zs)."""
    zr0 = jnp.minimum(jnp.maximum(zs, 0.0), 40.0)

    def gd(z):
        w = jnp.exp(l + s * z)
        x = a - w
        r = _ln_pdf(x) / _ln_D(a, w, Pa, Qa)
        sw = s * w
        return -z + r * sw, -1.0 + r * s * sw + r * (x - r) * sw * sw

    g_r, d2_r = gd(zr0)
    atzs = (zr0 <= 0.0) | (g_r >= 0.0)
    g0, _ = gd(jnp.zeros_like(a))

    def body(_, st):
        z, zl, zr = st
        g, d2 = gd(z)
        zl = jnp.where(g > 0.0, z, zl)
        zr = jnp.where(g > 0.0, zr, z)
        zn = jnp.where(d2 < 0.0, z - g / jnp.where(d2 < 0.0, d2, -1.0), 0.5 * (zl + zr))
        zn = jnp.where((zn > zl) & (zn < zr), zn, 0.5 * (zl + zr))
        return zn, zl, zr

    z, _, _ = jax.lax.fori_loop(0, 40, body, (jnp.clip(g0, 0.0, zr0), jnp.zeros_like(a), zr0))
    _, d2 = gd(z)
    zk = jnp.where(atzs, jnp.where(zr0 <= 0.0, 0.0, zr0), z)
    d2k = jnp.where(atzs, jnp.where(zr0 <= 0.0, -1.0, d2_r), d2)
    return zk, d2k, atzs


def _ln_mode_e(a, l, s, zs, es):
    """Mode in e = T - a, above es, of C's T-route integrand phi(e) P(zs < Z <= Z(a + e)) (log-concave): it moves
    far above es when s is small and Z(a + e) is deep in the lower tail."""
    ndtr = jax.scipy.special.ndtr

    def slope(e):
        T = a + e
        Z = (jnp.log(T) - l) / s
        diff = jnp.where(zs >= 0.0, ndtr(-zs) - ndtr(-Z), ndtr(Z) - ndtr(zs))
        return -e + _ln_pdf(Z) / jnp.maximum(diff, 1e-300) / (s * T)

    lo0 = es + 1e-9 * jnp.maximum(1.0, jnp.abs(es))
    hi0 = es + 80.0

    def body(_, lh):
        lo, hi = lh
        mid = 0.5 * (lo + hi)
        up = slope(mid) > 0.0
        return jnp.where(up, mid, lo), jnp.where(up, hi, mid)

    lo, hi = jax.lax.fori_loop(0, 50, body, (lo0, hi0))
    return 0.5 * (lo + hi)


def _ln_cum(a, l, s, Pa, Qa, pa):
    """(C, K, dC/da, dK/da, dC/dl, dC/ds) for C = P(T > e^(l + s Z)), T ~ N(a, 1), and K = Phi(a) - C. The
    smaller of the two (picked by an 8-node Gauss-Hermite estimate of C) is integrated around its own mode,
    the other is Phi(a) less it; dK/dl = -dC/dl and dK/ds = -dC/ds. Outside the Gauss-Hermite regime K's
    z-piece runs from max(min(zs, 0) - R, zs - 20) to zs, as its mass sits at z >= 0 up to zs."""
    ndtr = jax.scipy.special.ndtr
    pdf = _ln_pdf
    zs = (-jnp.log(s) - l) / s
    c8 = jnp.sum(_LN_GH8_W * ndtr(a[..., None] - jnp.exp(l[..., None] + s[..., None] * _LN_GH8_X)), -1)
    tk = c8 > 0.5 * Pa
    zm, d2m = _ln_mode(a, l, s)
    zk, d2k, atzs = _ln_mode_k(a, l, s, zs, Pa, Qa)
    zt = jnp.where(tk, zk, zm)
    d2t = jnp.where(tk, d2k, d2m)
    sct = jnp.where(d2t < 0.0, 1.0 / jnp.sqrt(jnp.where(d2t < 0.0, -d2t, 1.0)), 1.0)
    agh = jnp.where(tk, ~atzs & (zs >= zk + _LN_R * jnp.maximum(sct, 1.0)), zs >= zm + _LN_R)
    sign = jnp.where(tk, 1.0, -1.0)[..., None]
    a_, l_, s_, Pa_, Qa_, pa_, tk_ = (v[..., None] for v in (a, l, s, Pa, Qa, pa, tk))

    def target(z, w):
        u = a_ - w
        pu = pdf(u)
        f = jnp.where(tk_, _ln_D(a_, w, Pa_, Qa_), ndtr(u))
        fa = jnp.where(tk_, _ln_dD_da(a_, w, pa_), pu)
        return f, fa, sign * pu * w, sign * pu * w * z

    sc = jnp.where(agh, sct, 1.0)
    wt = jnp.exp(l + s * zt)
    ft = jnp.where(tk, _ln_D(a, wt, Pa, Qa), ndtr(a - wt))
    h0 = jnp.where(agh, -0.5 * zt * zt + jnp.log(jnp.maximum(ft, 1e-300)), 0.0)
    zg = zt[..., None] + sc[..., None] * _LN_GH_X
    wg = jnp.exp(l_ + s_ * zg)
    eg = jnp.exp(-0.5 * zg * zg + _LN_GH_LW - h0[..., None])
    cg = sc * jnp.exp(h0) / np.sqrt(2.0 * np.pi)
    g_agh = [cg * jnp.sum(eg * v, -1) for v in target(zg, wg)]

    zc = jnp.minimum(zt, zs)
    lo = jnp.where(tk, jnp.maximum(jnp.minimum(zs, 0.0) - _LN_R, zs - 20.0), zc - _LN_R)
    hi = jnp.where(tk, zs, jnp.minimum(zc + _LN_R, zs))
    h = 0.5 * jnp.maximum(hi - lo, 0.0)
    z = (lo + h)[..., None] + h[..., None] * _LN_GL_X
    w = jnp.exp(l_ + s_ * z)
    pz = _LN_GL_W * pdf(z)
    vz = [h * jnp.sum(pz * v, -1) for v in target(z, w)]

    es = 1.0 / s - a
    elo = jnp.maximum(es, -_LN_R)
    net = es - jnp.where(zs < 0.0, _ln_lam(zs), 0.0)
    ehi = jnp.where((es > 0.0) & (net * _LN_R > 40.0), es + 40.0 / jnp.where(net > 0.0, net, 1.0),
                    jnp.maximum(es, 0.0) + _LN_R)
    ehi = jnp.where(tk, ehi, jnp.maximum(ehi, _ln_mode_e(a, l, s, zs, es) + _LN_R))
    zsb = zs[..., None]

    def piece_e(lo_e, hi_e):
        he = 0.5 * jnp.maximum(hi_e - lo_e, 0.0)
        e = (lo_e + he)[..., None] + he[..., None] * _LN_GL_X
        ae = jnp.maximum(a_ + e, 1e-300)
        Z = (jnp.log(ae) - l_) / s_
        pe = _LN_GL_W * pdf(e)
        pZ = pdf(Z) / s_
        diff = jnp.where(zsb >= 0.0, ndtr(-zsb) - ndtr(-Z), ndtr(Z) - ndtr(zsb))
        f = jnp.where(tk_, ndtr(-Z), diff)
        return [he * jnp.sum(pe * v, -1) for v in (f, -sign * pZ / ae, sign * pZ, sign * pZ * Z)]

    ve = piece_e(elo, ehi)
    vb = piece_e(jnp.minimum(es, -_LN_R), jnp.full_like(es, -_LN_R))
    Qzs = ndtr(-zs)
    kc = jnp.where(tk, _ln_interval(-a, es) * Qzs, 0.0)
    kca = jnp.where(tk, pa * Qzs, 0.0)
    split = [vz[0] + ve[0] + vb[0] + kc, vz[1] + ve[1] + vb[1] + kca, vz[2] + ve[2] + vb[2], vz[3] + ve[3] + vb[3]]
    T, Ta, Tl, Ts = [jnp.where(agh, x, y) for x, y in zip(g_agh, split)]
    return [jnp.where(tk, Pa - T, T), jnp.where(tk, T, Pa - T), jnp.where(tk, pa - Ta, Ta), jnp.where(tk, Ta, pa - Ta),
            jnp.where(tk, -Tl, Tl), jnp.where(tk, -Ts, Ts)]


def _lnoise_cell_jax(a, ln, lf, s):
    """Pure-JAX reference of the native lognormal meta-noise cell kernel: log P(D_near e^(sZ) < T <=
    D_far e^(sZ)), T ~ N(a, 1), with ln = log D_near (-inf: open, D = 0) and lf = log D_far (+inf: open),
    and its derivatives in a, ln, lf and s."""
    on = jnp.isneginf(ln)
    of = jnp.isposinf(lf)
    Pa = jax.scipy.special.ndtr(a)
    Qa = jax.scipy.special.ndtr(-a)
    pa = jnp.exp(-0.5 * a * a - _LOG_SQRT_2PI)
    n = _ln_cum(a, jnp.where(on, 0.0, ln), s, Pa, Qa, pa)
    f = _ln_cum(a, jnp.where(of, 0.0, lf), s, Pa, Qa, pa)
    z0 = jnp.zeros_like(a)
    n = [jnp.where(on, v, x) for v, x in zip((Pa, z0, pa, z0, z0, z0), n)]
    f = [jnp.where(of, v, x) for v, x in zip((z0, Pa, z0, pa, z0, z0), f)]
    use_k = f[1] < n[0]
    P = jnp.where(use_k, f[1] - n[1], n[0] - f[0])
    ok = P > _LN_FLOOR
    iP = jnp.where(ok, 1.0 / jnp.where(ok, P, 1.0), 0.0)
    lp = jnp.log(jnp.where(ok, P, _LN_FLOOR))
    da = jnp.where(use_k, f[3] - n[3], n[2] - f[2]) * iP
    return lp, da, n[4] * iP, -f[4] * iP, (n[5] - f[5]) * iP


def _lnoise_raw(a, ln, lf, s):
    """(lp, d/da, d/dln, d/dlf, d/ds) of a lognormal meta-noise cell."""
    a, ln, lf, s = jnp.broadcast_arrays(*(jnp.asarray(v) for v in (a, ln, lf, s)))
    if _native_ok(a):
        return _native_call("lnoise_cell", 5, (a, ln, lf, s))
    return _lnoise_cell_jax(a, ln, lf, s)


def _mafc_terms(y0, t0, m, d, sg, b):
    """Slopes a and shifts c [N, M-1] of the product-of-CDFs integral for an m-AFC cell (DeCarlo 2012,
    target SD sg): the slots are the positions other than the chosen one, a lure slot past the trial's m
    alternatives is the empty term (a = 0, c = 40). Also returns the slot roles for the tangent."""
    M = b.shape[-1]
    j = jnp.arange(M - 1)
    pos = jnp.where(j[None, :] < y0[:, None], j[None, :], j[None, :] + 1)
    bj = jnp.take_along_axis(b, pos, axis=1)
    bk = jnp.take_along_axis(b, y0[:, None], axis=1)
    bs = jnp.take_along_axis(b, t0[:, None], axis=1)
    correct = (y0 == t0)[:, None]
    tslot = pos == t0[:, None]
    valid = pos < m[:, None]
    isg = 1.0 / sg[:, None]
    a = jnp.where(correct, sg[:, None], jnp.where(tslot, isg, 1.0))
    c = jnp.where(correct, d[:, None] + bs - bj, jnp.where(tslot, (bk - bs - d[:, None]) * isg, bk - bj))
    return jnp.where(valid, a, 0.0), jnp.where(valid, c, 40.0), (pos, correct, tslot, valid, isg)


@custom_jvp
def fused_loglik_mafc(y0, t0, m, d, sg, b):
    """m-AFC log-likelihood over [N] trials: choice y0 and target t0 (0-based positions), m alternatives,
    sensitivity d, target SD sg and position biases b [N, M] (b[:, M-1] = 0 for the reference)."""
    a, c, _ = _mafc_terms(y0, t0, m, d, sg, b)
    return _pcq_raw(a, c)[0]


@fused_loglik_mafc.defjvp
def _fused_loglik_mafc_jvp(primals, tangents):
    y0, t0, m, d, sg, b = primals
    _, _, _, td, tsg, tb = tangents
    a, c, (pos, correct, tslot, valid, isg) = _mafc_terms(y0, t0, m, d, sg, b)
    lp, dc, da = _pcq_raw(a, c)
    tbj = jnp.take_along_axis(tb, pos, axis=1)
    tbk = jnp.take_along_axis(tb, y0[:, None], axis=1)
    tbs = jnp.take_along_axis(tb, t0[:, None], axis=1)
    ta = jnp.where(correct, tsg[:, None], jnp.where(tslot, -tsg[:, None] * isg * isg, 0.0))
    tc = jnp.where(correct, td[:, None] + tbs - tbj,
                   jnp.where(tslot, (tbk - tbs - td[:, None]) * isg - c * tsg[:, None] * isg, tbk - tbj))
    ta = jnp.where(valid, ta, 0.0)
    tc = jnp.where(valid, tc, 0.0)
    return lp, jnp.sum(dc * tc + da * ta, -1)


def _vec_re_shift(u, idx, conf, n_comp):
    """Per-observation shifts [n_comp, N] of a component-major vector random effect (bias)."""
    R = u.shape[0] // n_comp
    if conf.get("type", "index") == "matrix":
        Z = _ensure_2d(jnp.asarray(conf["Z"], dtype=_FLOAT))
        return jnp.stack([jnp.sum(Z * u[k * R:(k + 1) * R, idx].T, axis=1) for k in range(n_comp)])
    if conf.get("term_idx") is not None:
        tidx = jnp.array(conf["term_idx"], dtype=jnp.int32) - 1
        has = tidx >= 0
        tsafe = jnp.clip(tidx, 0, R - 1)
        return jnp.stack([jnp.where(has, u[k * R + tsafe, idx], 0.0) for k in range(n_comp)])
    return u[:, idx]


def _sample_vec_re(name, conf):
    """One independent random effect of a vector parameter (bias): the same sites as a criterion
    random effect, u [D, G] component-major."""
    group = conf["group"]
    G = int(conf["n_groups"])
    D = int(conf["n_terms"])
    sd_prior = _positive_dist(parse_prior(conf.get("prior_sd", "normal(0, 0.5)")))
    per = conf.get("prior_sd_per_coef")
    if isinstance(per, str):
        per = [per]
    has_per = per is not None and len(per) == D and len(set(per)) > 1
    if conf.get("correlated", True) and D > 1 and conf.get("whiten") is not None and not conf.get("cor_blocks"):
        sds = [_positive_dist(parse_prior(per[t])) for t in range(D)] if has_per else [sd_prior] * D
        _, Lc, L = _whitened_cov(f"{name}_{group}", conf, D, sds)
        numpyro.deterministic(f"corr_{name}_{group}", Lc @ Lc.T)
        u = L @ numpyro.sample(f"z_{name}_{group}", dist.Normal(0, 1).expand([D, G]))
    elif conf.get("correlated", True) and D > 1:
        if has_per:
            sd = jnp.stack([numpyro.sample(f"sigma_{name}_{group}[{t + 1}]", _positive_dist(parse_prior(per[t])))
                            for t in range(D)])
        else:
            sd = numpyro.sample(f"sigma_{name}_{group}", sd_prior.expand([D]))
        Lc = _re_corr_cholesky(f"{name}_{group}", conf, D)
        numpyro.deterministic(f"corr_{name}_{group}", Lc @ Lc.T)
        u = (jnp.diag(sd) @ Lc) @ numpyro.sample(f"z_{name}_{group}", dist.Normal(0, 1).expand([D, G]))
    else:
        u = jnp.zeros((D, G))
        for t in range(D):
            sdp = _positive_dist(parse_prior(per[t])) if has_per else sd_prior
            sd_t = numpyro.sample(f"sigma_{name}_{group}[{t + 1}]", sdp)
            u = u.at[t].set(sd_t * numpyro.sample(f"z_{name}_{group}[{t + 1}]", dist.Normal(0, 1).expand([G])))
    numpyro.deterministic(f"u_{name}_{group}", u.T)
    return u


def _mafc_bias(config, bias_res, cross_contribs, N, M):
    """Position biases [N, M] (the last position 0) from the bias coefficients and random effects."""
    bconf = config.get("bias")
    if bconf is None:
        return jnp.zeros((N, M), dtype=_FLOAT)
    X = jnp.asarray(bconf["X"], dtype=_FLOAT).reshape(N, -1)
    P = X.shape[1]
    cols = bconf.get("col_names", ["V1"])
    cols = [cols] if isinstance(cols, str) else cols
    pri = [bconf.get("priors", {}).get(cols[j], "normal(0, 1)") for j in range(P)]
    beta_c = _grid_sample("beta_bias", [pri] * (M - 1), (M - 1, P)).reshape(M - 1, P)
    centre = _centre_of(bconf) if P > 1 else None
    if centre is not None:
        beta = jnp.stack([_uncentre(beta_c[k], centre) for k in range(M - 1)])
        for k in range(M - 1):
            _centre_factor(f"beta_bias[{k + 1}]", lambda j: parse_prior(pri[j]), centre, beta[k], beta_c[k])
        numpyro.deterministic("orig__beta_bias", beta)
    else:
        beta = beta_c
    bias = (X @ beta.T).T
    contribs = list(cross_contribs)
    for conf in bias_res:
        contribs.append((_sample_vec_re("bias", conf), jnp.array(conf["index"], dtype=jnp.int32) - 1, conf))
    for u, idx, conf in contribs:
        bias = bias + _vec_re_shift(u, idx, conf, M - 1)
    return jnp.concatenate([bias.T, jnp.zeros((N, 1), dtype=_FLOAT)], 1)


_LN_LEVELS = 12
_LN_NARROW = 1e-3


def _ln_bisect(f, lo, hi, increasing, n=80):
    def body(_, b):
        l, h = b
        m = 0.5 * (l + h)
        up = (f(m) > 0.0) != increasing
        return jnp.where(up, m, l), jnp.where(up, h, m)
    return 0.5 * sum(jax.lax.fori_loop(0, n, body, (lo, hi)))


def _ln_density(a, l, s):
    """g(l) = -dC/dl of the lognormal meta-noise cumulative C(l) = E_z Phi(a - exp(l + s z)), with its
    derivatives in (a, l, s): g = int exp(h(t)) / (2 pi s) dt, h(t) = t - (e^t - a)^2 / 2 - (t - l)^2 / (2 s^2).
    h is concave except between the inflections at e^t = (a +- sqrt(a^2 - 8 / s^2)) / 4, so it has at most two
    modes; each is found by bisection and integrated by Gauss-Legendre panels graded outward from it."""
    is2 = 1.0 / (s * s)
    h = lambda t: t - 0.5 * (jnp.exp(t) - a) ** 2 - 0.5 * (t - l) ** 2 * is2
    dh = lambda t: 1.0 - jnp.exp(t) * (jnp.exp(t) - a) - (t - l) * is2
    d2h = lambda t: -jnp.exp(t) * (2.0 * jnp.exp(t) - a) - is2
    disc = a * a - 8.0 * is2
    two = (disc > 0.0) & (a > 0.0)
    sq = jnp.sqrt(jnp.maximum(disc, 0.0))
    t1 = jnp.where(two, jnp.log(jnp.maximum(0.25 * (a - sq), 1e-300)), 0.0)
    t2 = jnp.where(two, jnp.log(jnp.maximum(0.25 * (a + sq), 1e-300)), 0.0)
    LB = jnp.minimum(l, 0.0) - 60.0
    UB = jnp.maximum(l, jnp.log(jnp.abs(a) + 3.0))
    e1 = ~two | (dh(t1) < 0.0)
    e2 = two & (dh(t2) > 0.0)
    m1 = _ln_bisect(dh, LB, jnp.where(two, t1, UB), False)
    m2 = _ln_bisect(dh, jnp.where(two, t2, LB), UB, False)
    v = _ln_bisect(dh, t1, t2, True)
    tstar = jnp.log(0.5 * (a + jnp.sqrt(a * a + 4.0)))
    lower = lambda m: jnp.minimum(jnp.minimum(m, l), tstar) - 12.0 * s
    upper = lambda m: jnp.maximum(jnp.maximum(m, l), tstar) + 12.0 * s
    sides = [(m1, lower(m1), e1), (m1, jnp.where(e2, v, upper(m1)), e1),
             (m2, jnp.where(e1, v, lower(m2)), e2), (m2, upper(m2), e2)]
    x = jnp.asarray((_GL16_X + 1.0) / 2.0, dtype=a.dtype)
    W = jnp.asarray(_GL16_W / 2.0, dtype=a.dtype)
    off = jnp.asarray(0.25 * (2.0 ** np.arange(_LN_LEVELS + 1) - 1.0), dtype=a.dtype)
    ts, ws = [], []
    for m, end, on in sides:
        sc = 1.0 / jnp.sqrt(jnp.maximum(-d2h(m), 1e-24))
        L = jnp.where(on, jnp.abs(end - m), 0.0)
        sg = jnp.where(end >= m, 1.0, -1.0)
        b = jnp.minimum(sc[..., None] * off, L[..., None])
        b = jnp.concatenate([b, L[..., None]], -1)
        lo, wd = b[..., :-1], b[..., 1:] - b[..., :-1]
        ts.append(m[..., None, None] + sg[..., None, None] * (lo[..., None] + wd[..., None] * x))
        ws.append(jnp.broadcast_to(wd[..., None] * W, ts[-1].shape))
    t = jnp.minimum(jnp.concatenate([q.reshape(q.shape[:-2] + (-1,)) for q in ts], -1), 300.0)
    wt = jnp.concatenate([q.reshape(q.shape[:-2] + (-1,)) for q in ws], -1)
    ht = t - 0.5 * (jnp.exp(t) - a[..., None]) ** 2 - 0.5 * (t - l[..., None]) ** 2 * is2[..., None]
    ref =jnp.max(jnp.where(wt > 0.0, ht, -jnp.inf), -1, keepdims=True)
    F = wt * jnp.exp(ht - ref)
    c = jnp.exp(ref[..., 0]) / (2.0 * np.pi * s)
    dl = (t - l[..., None]) * is2[..., None]
    return (c * jnp.sum(F, -1), c * jnp.sum(F * (jnp.exp(t) - a[..., None]), -1), c * jnp.sum(F * dl, -1),
            c * jnp.sum(F * (dl * dl * s[..., None] - 1.0 / s[..., None]), -1))


def _ln_narrow(a, ln, dl, s):
    """A lognormal meta-noise cell between log distances ln and ln + dl for small dl, P = int g over the
    band by 2-point Gauss-Legendre: (P, dP/da, dP/dln at fixed dl, dP/d dl, dP/ds)."""
    out = [0.0] * 5
    for xj in (0.5 - 0.5 / np.sqrt(3.0), 0.5 + 0.5 / np.sqrt(3.0)):
        g, ga, gl, gs = _ln_density(a, ln + xj * dl, s)
        out = [out[0] + 0.5 * dl * g, out[1] + 0.5 * dl * ga, out[2] + 0.5 * dl * gl,
               out[3] + 0.5 * g + 0.5 * dl * xj * gl, out[4] + 0.5 * dl * gs]
    return tuple(out)


def _make_fused_lnoise_loglik(K, gap_link_is_softplus):
    """Lognormal meta-noise (evsd with sigma_meta, Shekhar & Rahnev 2021) log-likelihood with custom JVP:
    the decision on x ~ N(mu, 1) against the middle threshold c0, the confidence criteria on the side of
    the decision at c0 + (c_k - c0) exp(s z), one z ~ N(0, 1) per trial. An interior cell enters by its near
    distance and the log ratio of its distances, log1p(gap / D_near), and a narrow one by _ln_narrow."""
    n_thresh, mid_idx, gap_mask, build = _thresh_layout(K)
    own_idx, own_mask = _own_gap(K)
    half = K // 2

    def _pieces(y, mu, sm, thresh_mid, gaps):
        gap_vals = jax.nn.softplus(gaps) if gap_link_is_softplus else jnp.exp(gaps)
        thresh = build(thresh_mid, gap_vals)
        old = y > half
        sgn = jnp.where(old, 1.0, -1.0)
        open_lo = jnp.where(old, y == half + 1, y == 1)
        open_hi = jnp.where(old, y == K, y == half)
        k_hi = jnp.clip(y - 1, 0, n_thresh - 1)
        k_lo = jnp.clip(y - 2, 0, n_thresh - 1)
        rows = jnp.arange(y.shape[0])
        D_lo = sgn * (thresh[rows, k_lo] - thresh_mid)
        D_hi = sgn * (thresh[rows, k_hi] - thresh_mid)
        open_near = jnp.where(old, open_lo, open_hi)
        open_far = jnp.where(old, open_hi, open_lo)
        inner = ~open_near & ~open_far
        D_near = jnp.where(open_near, 1.0, jnp.where(old, D_lo, D_hi))
        gap = (jnp.where(inner, gap_vals[rows, jnp.asarray(own_idx)[y - 1]], 1.0) if n_thresh > 1
               else jnp.ones_like(mu))
        D_far = jnp.where(open_far, 1.0, jnp.where(inner, D_near + gap, jnp.where(old, D_hi, D_lo)))
        ln = jnp.where(open_near, -jnp.inf, jnp.log(D_near))
        lf = jnp.where(open_far, jnp.inf, jnp.log(D_far))
        a = sgn * (mu - thresh_mid)
        dl = jnp.log1p(gap / D_near)
        out = _lnoise_raw(a, ln, lf, sm)
        narrow = inner & (dl < _LN_NARROW * jnp.minimum(1.0, sm))

        def over(o):
            P, dPa, dPl, dPd, dPs = _ln_narrow(a, jnp.where(narrow, ln, 0.0), jnp.where(narrow, dl, 1e-4), sm)
            Pc = jnp.maximum(P, _pfloor())
            ip = jnp.where(P > _pfloor(), 1.0 / Pc, 0.0)
            return tuple(jnp.where(narrow, u, w) for u, w in zip((jnp.log(Pc), dPa * ip, dPl * ip, dPd * ip,
                                                                  dPs * ip), o))
        wide = (out[0], out[1], out[2], out[3], out[4])
        lp, da, d_ln, d_x, ds = jax.lax.cond(jnp.any(narrow), over, lambda o: o, wide)
        k_near = jnp.where(old, k_lo, k_hi)
        k_far = jnp.where(old, k_hi, k_lo)
        g_near = jnp.where(open_near, 0.0, d_ln / D_near)
        g_far = jnp.where(open_far, 0.0, d_x / D_far)
        GA = jnp.where(narrow, (d_ln - d_x * gap / D_far) / D_near, jnp.where(open_near, g_far, g_near + g_far))
        GB = jnp.where(narrow, d_x / D_far, jnp.where(inner, g_far, 0.0))
        k_a = jnp.where(open_near, k_far, k_near)
        return (lp, da, ds), (sgn, GA, GB, k_a, inner, gap_vals)

    @custom_jvp
    def _lnoise(y, mu, sm, thresh_mid, gaps):
        return _pieces(y, mu, sm, thresh_mid, gaps)[0][0]

    @_lnoise.defjvp
    def _lnoise_jvp(primals, tangents):
        y, mu, sm, thresh_mid, gaps = primals
        _, t_mu, t_sm, t_mid, t_gaps = tangents
        (lp, da, ds), (sgn, GA, GB, k_a, inner, gap_vals) = _pieces(*primals)
        gap_d = jax.nn.sigmoid(gaps) if gap_link_is_softplus else gap_vals
        d_gaps = _lw_gap_grads(sgn * GA, GB, k_a, jnp.asarray(own_mask, dtype=gaps.dtype)[:, y - 1], inner,
                               gap_mask, gap_d)
        tan = sgn * da * t_mu + ds * t_sm - sgn * da * t_mid + jnp.sum(d_gaps * t_gaps, axis=1)
        return lp, tan

    return _lnoise


def fused_loglik_lnoise(y, K, mu, sm, thresh_mid, gaps, gap_link="log"):
    """Lognormal meta-noise log-likelihood; gaps [n_gaps, N] effective log-gaps, or [n_gaps] if shared."""
    N = y.shape[0]
    is_softplus = gap_link == "softplus"
    gaps_per_obs = jnp.broadcast_to(gaps[None, :], (N, K - 2)) if gaps.ndim == 1 else gaps.T
    key = ("lnoise", int(K), bool(is_softplus))
    if key not in _fused_cache:
        _fused_cache[key] = _make_fused_lnoise_loglik(int(K), bool(is_softplus))
    return _fused_cache[key](y, mu, sm, thresh_mid, gaps_per_obs)


def _make_fused_metad_loglik(K, gap_link_is_softplus, q, relative, link_name="probit"):
    """Meta-d' log-likelihood over [N] trials with custom JVP: the type-1 decision and the
    meta-level rating truncated at the meta criterion, the confidence criteria shifted by
    (M - 1)(c - q d) under the relative anchor."""
    n_thresh, mid_idx, gap_mask, build = _thresh_layout(K)
    own_idx, own_mask = _own_gap(K)
    half = K // 2

    def _pieces(y, d, m_old, m_new, a, thresh_mid, gaps):
        gap_vals = jax.nn.softplus(gaps) if gap_link_is_softplus else jnp.exp(gaps)
        thresh = build(thresh_mid, gap_vals)
        old = y > half
        M = jnp.where(old, m_old, m_new)
        c = thresh_mid
        a_c = a - q
        mu_m = q * d + M * d * a_c
        c_mid = c - q * d
        off = ((M - 1.0) * c_mid if relative else 0.0) - mu_m
        k_hi = jnp.clip(y - 1, 0, n_thresh - 1)
        k_lo = jnp.clip(y - 2, 0, n_thresh - 1)
        rows = jnp.arange(y.shape[0])
        z_lo = jnp.where(y > 1, thresh[rows, k_lo] + off, -_OPEN)
        z_hi = jnp.where(y < K, thresh[rows, k_hi] + off, _OPEN)
        inner = (y > 1) & (y < K)
        w = jnp.where(inner, gap_vals[rows, jnp.asarray(own_idx)[y - 1]], 1.0) if n_thresh > 1 else jnp.ones_like(d)
        ls = jnp.where(inner, z_lo, 0.0)
        if link_name == "probit":
            sel, lb, rA, rB = _probit_narrow(ls, w, inner)
        else:
            lb, A, B = (_logit_inner if link_name == "logit" else _cloglog_inner)(ls, w)
            sel = inner
            Db = jnp.exp(lb)
            ib = jnp.where(Db > _pfloor(), 1.0 / jnp.maximum(Db, _pfloor()), 0.0)
            rA, rB = A * ib, B * ib
            lb = jnp.log(jnp.maximum(Db, _pfloor()))
        lp, g_a, g_lo, g_hi, g_c = _metad_cell(c - a * d, jnp.where(sel, -_OPEN, z_lo), jnp.where(sel, _OPEN, z_hi),
                                               c + off, old, link_name)
        g_lo = jnp.where(y > 1, g_lo, 0.0)
        g_hi = jnp.where(y < K, g_hi, 0.0)
        gA = jnp.where(sel, rA, g_lo + g_hi)
        gB = jnp.where(sel, rB, jnp.where(inner, g_hi, 0.0))
        out = (jnp.where(sel, lp + lb, lp), g_a, gA, gB, g_c)
        return out, (old, M, a_c, c_mid, jnp.where(y > 1, k_lo, k_hi), gap_vals)

    @custom_jvp
    def _metad(y, d, m_old, m_new, a, thresh_mid, gaps):
        return _pieces(y, d, m_old, m_new, a, thresh_mid, gaps)[0][0]

    @_metad.defjvp
    def _metad_jvp(primals, tangents):
        y, d, m_old, m_new, a, thresh_mid, gaps = primals
        _, t_d, t_mo, t_mn, _, t_mid, t_gaps = tangents
        (lp, g_a, gA, gB, g_c), (old, M, a_c, c_mid, k_a, gap_vals) = _pieces(*primals)
        G = gA + g_c
        if relative:
            d_mid = g_a + g_c + (M - 1.0) * G
            d_d = -a * g_a - G * M * a
            d_M = G * (c_mid - d * a_c)
        else:
            d_mid = g_a + g_c
            d_d = -a * g_a - G * (q + M * a_c)
            d_M = -G * d * a_c
        gap_d = jax.nn.sigmoid(gaps) if gap_link_is_softplus else gap_vals
        gm = jnp.asarray(gap_mask, dtype=gaps.dtype)
        own = jnp.asarray(own_mask, dtype=gaps.dtype)[:, y - 1]
        d_gaps = (gm[:, k_a].T * gA[:, None] + own.T * gB[:, None]) * gap_d
        tan = (d_d * t_d + jnp.where(old, d_M, 0.0) * t_mo + jnp.where(old, 0.0, d_M) * t_mn +
               (d_mid + gA) * t_mid + jnp.sum(d_gaps * t_gaps, axis=1))
        return lp, tan

    return _metad


def _make_fused_gnoise_loglik(K, gap_link_is_softplus):
    """Gaussian meta-noise (evsd with sigma_meta) log-likelihood with custom JVP: the decision on
    x ~ N(mu, 1) against the middle threshold, the rating on y = x + e, e ~ N(0, s^2), a bivariate
    normal rectangle with correlation 1 / sqrt(1 + s^2); y on the other side of the decision
    criterion gives the lowest confidence. The rating cell enters by its lower edge and width."""
    n_thresh, mid_idx, gap_mask, build = _thresh_layout(K)
    own_idx, own_mask = _own_gap(K)
    half = K // 2

    def _pieces(y, mu, sm, thresh_mid, gaps):
        gap_vals = jax.nn.softplus(gaps) if gap_link_is_softplus else jnp.exp(gaps)
        thresh = build(thresh_mid, gap_vals)
        old = y > half
        tau = jnp.sqrt(1.0 + sm * sm)
        inv_tau = 1.0 / tau
        open_lo = jnp.where(old, y == half + 1, y == 1)
        open_hi = jnp.where(old, y == K, y == half)
        k_hi = jnp.clip(y - 1, 0, n_thresh - 1)
        k_lo = jnp.clip(y - 2, 0, n_thresh - 1)
        rows = jnp.arange(y.shape[0])
        z_lo = jnp.where(open_lo, -_OPEN, (thresh[rows, k_lo] - mu) * inv_tau)
        z_hi = jnp.where(open_hi, _OPEN, (thresh[rows, k_hi] - mu) * inv_tau)
        inner = ~open_lo & ~open_hi
        w = (jnp.where(inner, gap_vals[rows, jnp.asarray(own_idx)[y - 1]], 1.0) * inv_tau if n_thresh > 1
             else jnp.ones_like(mu))
        z_c = thresh_mid - mu
        z1_lo = jnp.where(old, z_c, -_OPEN)
        z1_hi = jnp.where(old, _OPEN, z_c)
        out = _rect_lw(z1_lo, z1_hi, jnp.ones_like(z_c), z_lo, z_hi, w, inv_tau)
        k_a = jnp.where(open_lo, k_hi, k_lo)
        return out, (inner, jnp.where(open_lo, z_hi, z_lo), w, inv_tau, k_a, gap_vals, z_hi)

    @custom_jvp
    def _gnoise(y, mu, sm, thresh_mid, gaps):
        p = _pieces(y, mu, sm, thresh_mid, gaps)[0][0]
        return jnp.log(jnp.clip(p, _pfloor()))

    @_gnoise.defjvp
    def _gnoise_jvp(primals, tangents):
        y, mu, sm, thresh_mid, gaps = primals
        _, t_mu, t_sm, t_mid, t_gaps = tangents
        (p, A1, _, A2, B2, drho), (inner, z_a, w, inv_tau, k_a, gap_vals, z_hi) = _pieces(*primals)
        prob = jnp.clip(p, _pfloor())
        inv_p = jnp.where(p > _pfloor(), 1.0 / prob, 0.0)
        g_c = A1 * inv_p
        gA = A2 * inv_p
        gB = jnp.where(inner, B2 * inv_p, 0.0)
        d_mu = -g_c - gA * inv_tau
        d_tau = -_lw_scale(gA, gB, z_a, w, inner, z_hi) * inv_tau - drho * inv_p * inv_tau * inv_tau
        d_sm = d_tau * sm * inv_tau
        gap_d = jax.nn.sigmoid(gaps) if gap_link_is_softplus else gap_vals
        d_gaps = _lw_gap_grads(gA * inv_tau, gB * inv_tau, k_a, jnp.asarray(own_mask, dtype=gaps.dtype)[:, y - 1],
                               inner, gap_mask, gap_d)
        tan = (d_mu * t_mu + d_sm * t_sm + (g_c + gA * inv_tau) * t_mid + jnp.sum(d_gaps * t_gaps, axis=1))
        return jnp.log(prob), tan

    return _gnoise


def fused_loglik_gnoise(y, K, mu, sm, thresh_mid, gaps, gap_link="log"):
    """Gaussian meta-noise log-likelihood; gaps [n_gaps, N] effective log-gaps, or [n_gaps] if shared."""
    N = y.shape[0]
    is_softplus = gap_link == "softplus"
    gaps_per_obs = jnp.broadcast_to(gaps[None, :], (N, K - 2)) if gaps.ndim == 1 else gaps.T
    key = ("gnoise", int(K), bool(is_softplus))
    if key not in _fused_cache:
        _fused_cache[key] = _make_fused_gnoise_loglik(int(K), bool(is_softplus))
    return _fused_cache[key](y, mu, sm, thresh_mid, gaps_per_obs)


def fused_loglik_metad(y, K, d, m_old, m_new, is_old, thresh_mid, gaps, gap_link="log", q=0.5, relative=False,
                       link_name="probit"):
    """Meta-d' log-likelihood; gaps [n_gaps, N] effective log-gaps, or [n_gaps] if shared."""
    N = y.shape[0]
    is_softplus = gap_link == "softplus"
    gaps_per_obs = jnp.broadcast_to(gaps[None, :], (N, K - 2)) if gaps.ndim == 1 else gaps.T
    key = ("metad", int(K), bool(is_softplus), float(q), bool(relative), str(link_name))
    if key not in _fused_cache:
        _fused_cache[key] = _make_fused_metad_loglik(int(K), bool(is_softplus), float(q), bool(relative),
                                                     str(link_name))
    return _fused_cache[key](y, d, m_old, m_new, is_old, thresh_mid, gaps_per_obs)


def _make_fused_batch_dpsdt_loglik(K, gap_link_is_softplus, link_name="probit", edge=None):
    """Fully-vectorized batch log-likelihood for DPSDT with custom JVP.

    DPSDT: P = lambda * I(y==edge) + (1-lambda) * cell_p, edge = K unless given.
    Operates on entire [N]-shaped arrays directly.
    """
    edge = K if edge is None else int(edge)
    n_thresh, mid_idx, gap_mask, build = _thresh_layout(K)
    own_idx, own_mask = _own_gap(K)

    def parts(y, mu, sigma_val, lambda_val, thresh_mid, gap_vals):
        cell = _cells_from_thresh(link_name, y - 1, K, build(thresh_mid, gap_vals), gap_vals, own_idx, mu,
                                  sigma_val)
        cell_p = jnp.clip(cell[0], _pfloor())
        is_top = (y == edge).astype(_FLOAT)
        p_raw = lambda_val * is_top + (1.0 - lambda_val) * cell_p
        return cell, cell_p, is_top, p_raw

    @custom_jvp
    def _batch_fused(y, mu, sigma_val, lambda_val, thresh_mid, gaps):
        gap_vals = _gap_vals_and_derivs(gaps, gap_link_is_softplus)[0]
        return jnp.log(jnp.clip(parts(y, mu, sigma_val, lambda_val, thresh_mid, gap_vals)[3], _pfloor()))

    @_batch_fused.defjvp
    def _batch_fused_jvp(primals, tangents):
        y, mu, sigma_val, lambda_val, thresh_mid, gaps = primals
        _, d_mu, d_sigma_val, d_lambda, d_thresh_mid, d_gaps = tangents
        gap_vals, gap_derivs = _gap_vals_and_derivs(gaps, gap_link_is_softplus)
        cell, cell_p, is_top, p_raw = parts(y, mu, sigma_val, lambda_val, thresh_mid, gap_vals)
        p = jnp.clip(p_raw, _pfloor())
        val = jnp.log(p)
        inv_p = jnp.where(p_raw > _pfloor(), 1.0 / p, 0.0)
        d_lp_d_lambda = (is_top - cell_p) * inv_p
        c = jnp.where(cell[0] > _pfloor(), (1.0 - lambda_val) * inv_p, 0.0)
        g_mu, g_sigma, g_mid, g_gap_val = _cell_lw_grads(c, cell, sigma_val, gap_mask, jnp.asarray(own_mask)[:, y - 1])
        tangent_out = (g_mu * d_mu + g_sigma * d_sigma_val + d_lp_d_lambda * d_lambda + g_mid * d_thresh_mid +
                       jnp.sum((g_gap_val * gap_derivs.T).T * d_gaps, axis=1))
        return val, tangent_out

    return _batch_fused


def _make_fused_batch_mixture_loglik(K, gap_link_is_softplus, link_name="probit"):
    """Fully-vectorized batch log-likelihood for mixture with custom JVP.

    Mixture: P = lambda * p1(mu1,sig1) + (1-lambda) * p2(mu2,sig2).
    Operates on entire [N]-shaped arrays directly.
    """
    n_thresh, mid_idx, gap_mask, build = _thresh_layout(K)
    own_idx, own_mask = _own_gap(K)

    def parts(y, mu1, sigma1_val, mu2, sigma2_val, lambda_val, thresh_mid, gap_vals):
        thresh = build(thresh_mid, gap_vals)
        c1 = _cells_from_thresh(link_name, y - 1, K, thresh, gap_vals, own_idx, mu1, sigma1_val)
        c2 = _cells_from_thresh(link_name, y - 1, K, thresh, gap_vals, own_idx, mu2, sigma2_val)
        p1, p2 = jnp.clip(c1[0], _pfloor()), jnp.clip(c2[0], _pfloor())
        return c1, c2, p1, p2, lambda_val * p1 + (1.0 - lambda_val) * p2

    @custom_jvp
    def _batch_fused(y, mu1, sigma1_val, mu2, sigma2_val, lambda_val, thresh_mid, gaps):
        gap_vals = _gap_vals_and_derivs(gaps, gap_link_is_softplus)[0]
        p_raw = parts(y, mu1, sigma1_val, mu2, sigma2_val, lambda_val, thresh_mid, gap_vals)[4]
        return jnp.log(jnp.clip(p_raw, _pfloor()))

    @_batch_fused.defjvp
    def _batch_fused_jvp(primals, tangents):
        y, mu1, sigma1_val, mu2, sigma2_val, lambda_val, thresh_mid, gaps = primals
        _, d_mu1, d_sigma1, d_mu2, d_sigma2, d_lambda, d_thresh_mid, d_gaps = tangents
        gap_vals, gap_derivs = _gap_vals_and_derivs(gaps, gap_link_is_softplus)
        c1, c2, p1, p2, p_raw = parts(y, mu1, sigma1_val, mu2, sigma2_val, lambda_val, thresh_mid, gap_vals)
        p = jnp.clip(p_raw, _pfloor())
        val = jnp.log(p)
        inv_p = jnp.where(p_raw > _pfloor(), 1.0 / p, 0.0)
        d_lp_d_lambda = (p1 - p2) * inv_p
        own = jnp.asarray(own_mask)[:, y - 1]
        g1 = _cell_lw_grads(jnp.where(c1[0] > _pfloor(), lambda_val * inv_p, 0.0), c1, sigma1_val, gap_mask, own)
        g2 = _cell_lw_grads(jnp.where(c2[0] > _pfloor(), (1.0 - lambda_val) * inv_p, 0.0), c2, sigma2_val,
                            gap_mask, own)
        tangent_out = (g1[0] * d_mu1 + g1[1] * d_sigma1 + g2[0] * d_mu2 + g2[1] * d_sigma2 +
                       d_lp_d_lambda * d_lambda + (g1[2] + g2[2]) * d_thresh_mid +
                       jnp.sum(((g1[3] + g2[3]) * gap_derivs.T).T * d_gaps, axis=1))
        return val, tangent_out

    return _batch_fused


def _make_fused_batch_source_mixture_loglik(K, gap_link_is_softplus, link_name="probit"):
    """Fully-vectorized batch log-likelihood for source_mixture with custom JVP.

    Source mixture: P = lambda * p_att(d_attended) + (1-lambda) * p_noise(mu=0).
    Both states use sigma=1. Operates on entire [N]-shaped arrays directly.
    """
    n_thresh, mid_idx, gap_mask, build = _thresh_layout(K)
    own_idx, own_mask = _own_gap(K)

    def parts(y, d_attended, lambda_val, thresh_mid, gap_vals):
        thresh = build(thresh_mid, gap_vals)
        one = jnp.ones_like(d_attended)
        ca = _cells_from_thresh(link_name, y - 1, K, thresh, gap_vals, own_idx, d_attended, one)
        cn = _cells_from_thresh(link_name, y - 1, K, thresh, gap_vals, own_idx, jnp.zeros_like(d_attended), one)
        pa, pn = jnp.clip(ca[0], _pfloor()), jnp.clip(cn[0], _pfloor())
        return ca, cn, pa, pn, lambda_val * pa + (1.0 - lambda_val) * pn

    @custom_jvp
    def _batch_fused(y, d_attended, lambda_val, thresh_mid, gaps):
        gap_vals = _gap_vals_and_derivs(gaps, gap_link_is_softplus)[0]
        return jnp.log(jnp.clip(parts(y, d_attended, lambda_val, thresh_mid, gap_vals)[4], _pfloor()))

    @_batch_fused.defjvp
    def _batch_fused_jvp(primals, tangents):
        y, d_attended, lambda_val, thresh_mid, gaps = primals
        _, d_d_attended, d_lambda, d_thresh_mid, d_gaps = tangents
        gap_vals, gap_derivs = _gap_vals_and_derivs(gaps, gap_link_is_softplus)
        ca, cn, pa, pn, p_raw = parts(y, d_attended, lambda_val, thresh_mid, gap_vals)
        p = jnp.clip(p_raw, _pfloor())
        val = jnp.log(p)
        inv_p = jnp.where(p_raw > _pfloor(), 1.0 / p, 0.0)
        d_lp_d_lambda = (pa - pn) * inv_p
        own = jnp.asarray(own_mask)[:, y - 1]
        one = jnp.ones_like(d_attended)
        ga = _cell_lw_grads(jnp.where(ca[0] > _pfloor(), lambda_val * inv_p, 0.0), ca, one, gap_mask, own)
        gn = _cell_lw_grads(jnp.where(cn[0] > _pfloor(), (1.0 - lambda_val) * inv_p, 0.0), cn, one, gap_mask,
                            own)
        tangent_out = (ga[0] * d_d_attended + d_lp_d_lambda * d_lambda + (ga[2] + gn[2]) * d_thresh_mid +
                       jnp.sum(((ga[3] + gn[3]) * gap_derivs.T).T * d_gaps, axis=1))
        return val, tangent_out

    return _batch_fused


# =============================================================================
# Batch-vectorized bivariate_sdt loglik (no vmap)
# =============================================================================

def _make_batch_gap_mask_np(n_gaps, n_thresh, mid_idx, n_upper):
    """Build gap-to-threshold contribution mask as a numpy array.
    Safe to call from within JIT trace — numpy arrays become JAX constants."""
    mask = np.zeros((n_gaps, n_thresh), dtype=np.float64)
    for g in range(n_gaps):
        if g < n_upper:
            for t in range(mid_idx + 1 + g, n_thresh):
                mask[g, t] = 1.0
        else:
            thresh_max = mid_idx - 1 - (g - n_upper)
            for t in range(0, thresh_max + 1):
                mask[g, t] = -1.0
    return mask


def _batch_build_thresh(thresh_mid, gap_vals, n_gaps, mid_idx, n_upper):
    """Build [N, n_thresh] thresholds via cumsum. Shared by all batch families."""
    if n_gaps == 0:
        return thresh_mid[:, None]
    parts = []
    if mid_idx > 0:
        lower_gaps = gap_vals[:, n_upper:]
        lower_cumsum = jnp.cumsum(lower_gaps, axis=1)
        parts.append(thresh_mid[:, None] - lower_cumsum[:, ::-1])
    parts.append(thresh_mid[:, None])
    if n_upper > 0:
        upper_gaps = gap_vals[:, :n_upper]
        upper_cumsum = jnp.cumsum(upper_gaps, axis=1)
        parts.append(thresh_mid[:, None] + upper_cumsum)
    return jnp.concatenate(parts, axis=1)


def _make_fused_batch_bivariate_sdt_loglik(K1, K2, gap_link_is_softplus):
    """Fully-vectorized batch log-likelihood for bivariate_sdt: one bivariate normal rectangle per
    trial, each dimension's cell by its lower edge and width (_rect_lw), with an analytic JVP."""
    n_thresh1 = K1 - 1
    n_gaps1 = n_thresh1 - 1
    mid_idx1 = (n_thresh1 - 1) // 2
    n_upper1 = n_thresh1 - mid_idx1 - 1

    n_thresh2 = K2 - 1
    n_gaps2 = n_thresh2 - 1
    mid_idx2 = (n_thresh2 - 1) // 2
    n_upper2 = n_thresh2 - mid_idx2 - 1

    gap_mask1_np = _make_batch_gap_mask_np(n_gaps1, n_thresh1, mid_idx1, n_upper1)
    gap_mask2_np = _make_batch_gap_mask_np(n_gaps2, n_thresh2, mid_idx2, n_upper2)
    own_idx1, own_mask1 = _own_gap(K1)
    own_idx2, own_mask2 = _own_gap(K2)

    def edges(y, K, c, gv, own_idx, mu, sigma):
        r = y - 1
        n_thresh = K - 1
        obs = jnp.arange(y.shape[0])
        k_hi = jnp.clip(r, 0, n_thresh - 1)
        k_lo = jnp.clip(r - 1, 0, n_thresh - 1)
        has_hi, has_lo = r < K - 1, r > 0
        z_hi = jnp.where(has_hi, (c[obs, k_hi] - mu) / sigma, _OPEN)
        z_lo = jnp.where(has_lo, (c[obs, k_lo] - mu) / sigma, -_OPEN)
        w = gv[obs, jnp.asarray(own_idx)[r]] / sigma if gv.shape[1] else jnp.ones_like(z_lo)
        return z_lo, z_hi, w, jnp.where(has_lo, z_lo, z_hi), jnp.where(has_lo, k_lo, k_hi), has_lo & has_hi

    def pieces(y1, y2, mu1, mu2, sigma1, sigma2, rho_val, thresh1_mid, gv1, thresh2_mid, gv2):
        c1 = _batch_build_thresh(thresh1_mid, gv1, n_gaps1, mid_idx1, n_upper1)
        c2 = _batch_build_thresh(thresh2_mid, gv2, n_gaps2, mid_idx2, n_upper2)
        e1 = edges(y1, K1, c1, gv1, own_idx1, mu1, sigma1)
        e2 = edges(y2, K2, c2, gv2, own_idx2, mu2, sigma2)
        return e1, e2, _rect_lw(e1[0], e1[1], e1[2], e2[0], e2[1], e2[2], rho_val)

    @custom_jvp
    def _batch_fused(y1, y2, item_type, mu1, mu2, sigma1, sigma2, rho_val,
                     thresh1_mid, gaps1, thresh2_mid, gaps2):
        gv1 = _gap_vals_and_derivs(gaps1, gap_link_is_softplus)[0]
        gv2 = _gap_vals_and_derivs(gaps2, gap_link_is_softplus)[0]
        P = pieces(y1, y2, mu1, mu2, sigma1, sigma2, rho_val, thresh1_mid, gv1, thresh2_mid, gv2)[2][0]
        return jnp.log(jnp.clip(P, _pfloor()))

    @_batch_fused.defjvp
    def _batch_fused_jvp(primals, tangents):
        (y1, y2, item_type, mu1, mu2, sigma1, sigma2, rho_val,
         thresh1_mid, gaps1, thresh2_mid, gaps2) = primals
        (_, _, _, d_mu1, d_mu2, d_sigma1, d_sigma2, d_rho,
         d_thresh1_mid, d_gaps1, d_thresh2_mid, d_gaps2) = tangents
        gv1, gd1 = _gap_vals_and_derivs(gaps1, gap_link_is_softplus)
        gv2, gd2 = _gap_vals_and_derivs(gaps2, gap_link_is_softplus)
        e1, e2, (p_raw, A1, B1, A2, B2, dr) = pieces(y1, y2, mu1, mu2, sigma1, sigma2, rho_val, thresh1_mid, gv1,
                                                     thresh2_mid, gv2)
        prob = jnp.clip(p_raw, _pfloor())
        val = jnp.log(prob)
        inv_P = jnp.where(p_raw > _pfloor(), 1.0 / prob, 0.0)
        inv_s1, inv_s2 = 1.0 / sigma1, 1.0 / sigma2
        g1 = _lw_chain(A1 * inv_P, B1 * inv_P, e1[3], e1[2], e1[5], inv_s1, e1[1])
        g2 = _lw_chain(A2 * inv_P, B2 * inv_P, e2[3], e2[2], e2[5], inv_s2, e2[1])
        dg1 = _lw_gap_grads(A1 * inv_P * inv_s1, B1 * inv_P * inv_s1, e1[4], jnp.asarray(own_mask1)[:, y1 - 1],
                            e1[5], gap_mask1_np, gd1)
        dg2 = _lw_gap_grads(A2 * inv_P * inv_s2, B2 * inv_P * inv_s2, e2[4], jnp.asarray(own_mask2)[:, y2 - 1],
                            e2[5], gap_mask2_np, gd2)
        tangent_out = (g1[0] * d_mu1 + g2[0] * d_mu2 + g1[1] * d_sigma1 + g2[1] * d_sigma2 + dr * inv_P * d_rho +
                       g1[2] * d_thresh1_mid + jnp.sum(dg1 * d_gaps1, axis=1) +
                       g2[2] * d_thresh2_mid + jnp.sum(dg2 * d_gaps2, axis=1))
        return val, tangent_out

    return _batch_fused


# =============================================================================
# Batch-vectorized bounded bivariate_sdt loglik (no vmap)
# =============================================================================

def _two_sum(a, b):
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def _veltkamp(a):
    c = 134217729.0 * a
    hi = c - (c - a)
    return hi, a - hi


def _two_prod(a, b):
    p = a * b
    ah, al = _veltkamp(a)
    bh, bl = _veltkamp(b)
    return p, ((ah * bh - p) + ah * bl + al * bh) + al * bl


def _dd_sum(terms):
    """sum(terms) as a double-double (hi, lo), each addition error-free (TwoSum)."""
    hi, lo = terms[0], jnp.zeros_like(terms[0])
    for t in terms[1:]:
        hi, e = _two_sum(hi, t)
        lo = lo + e
    return hi, lo


def _dd_div(a_hi, a_lo, b_hi, b_lo):
    q = a_hi / b_hi
    p, pe = _two_prod(q, b_hi)
    return q, (((a_hi - p) - pe) + a_lo - q * b_lo) / b_hi


_DD_LN2 = (0.6931471805599453, 2.3190468138462996e-17)
_DD_LOG1P = ((1.0, 0.0), (-0.5, 0.0), (0.3333333333333333, 1.850371707708594e-17), (-0.25, 0.0), (0.2, -1.1102230246251566e-17), (-0.16666666666666666, -9.25185853854297e-18), (0.14285714285714285, 7.93016446160826e-18), (-0.125, 0.0))
_DD_INVFACT = ((1.0, 0.0), (1.0, 0.0), (0.5, 0.0), (0.16666666666666666, 9.25185853854297e-18), (0.041666666666666664, 2.3129646346357427e-18), (0.008333333333333333, 1.1564823173178714e-19), (0.001388888888888889, -5.300543954373577e-20), (0.0001984126984126984, 1.7209558293420705e-22), (2.48015873015873e-05, 2.1511947866775882e-23), (2.7557319223985893e-06, -1.858393274046472e-22), (2.755731922398589e-07, 2.3767714622250297e-23), (2.505210838544172e-08, -1.448814070935912e-24), (2.08767569878681e-09, -1.20734505911326e-25), (1.6059043836821613e-10, 1.2585294588752098e-26), (1.1470745597729725e-11, 2.0655512752830745e-28), (7.647163731819816e-13, 7.03872877733453e-30), (4.779477332387385e-14, 4.399205485834081e-31), (2.8114572543455206e-15, 1.6508842730861433e-31), (1.5619206968586225e-16, 1.1910679660273754e-32), (8.22063524662433e-18, 2.2141894119604265e-34), (4.110317623312165e-19, 1.4412973378659527e-36), (1.9572941063391263e-20, -1.3643503830087908e-36), (8.896791392450574e-22, -7.911402614872376e-38), (3.868170170630684e-23, -8.843177655482344e-40))


def _dd_add(a, b):
    hi, e = _two_sum(a[0], b[0])
    return _two_sum(hi, e + a[1] + b[1])


def _dd_mul(a, b):
    p, e = _two_prod(a[0], b[0])
    return _two_sum(p, e + a[0] * b[1] + a[1] * b[0])


def _dd_exp(x):
    """exp(x) as a double-double: x = k ln 2 + r with ln 2 in double-double, exp(r / 256) by its Taylor series and
    eight squarings, scaled by 2^k."""
    k = jnp.round(x / _DD_LN2[0])
    p, pe = _two_prod(k, _DD_LN2[0])
    r = _dd_add(_dd_add((x, jnp.zeros_like(x)), (-p, -pe)), (-k * _DD_LN2[1], jnp.zeros_like(x)))
    r = (r[0] / 256.0, r[1] / 256.0)
    e = tuple(jnp.full_like(x, v) for v in _DD_INVFACT[12])
    for n in range(11, -1, -1):
        e = _dd_add(_dd_mul(e, r), tuple(jnp.full_like(x, v) for v in _DD_INVFACT[n]))
    for _ in range(8):
        e = _dd_mul(e, e)
    ki = k.astype(jnp.int32)
    return jnp.ldexp(e[0], ki), jnp.ldexp(e[1], ki)


def _dd_gap(g, softplus):
    """A gap value, exp(g) or softplus(g) = log(1 + exp(g)), as a double-double (softplus by one Newton step of
    exp(y) = 1 + exp(g) from the double result)."""
    if not softplus:
        return _dd_exp(g)
    y = jax.nn.softplus(g)
    ey = _dd_exp(y)
    rhs = _dd_add((jnp.ones_like(g), jnp.zeros_like(g)), _dd_exp(g))
    num = _dd_add(ey, (-rhs[0], -rhs[1]))
    newton = _two_sum(y, -(num[0] + num[1]) / ey[0])
    v = _dd_exp(jnp.minimum(g, -10.0))
    c = tuple(jnp.full_like(g, t) for t in _DD_LOG1P[-1])
    for t in _DD_LOG1P[-2::-1]:
        c = _dd_add(_dd_mul(c, v), tuple(jnp.full_like(g, u) for u in t))
    series = _dd_mul(c, v)
    small = g < -10.0
    return jnp.where(small, series[0], newton[0]), jnp.where(small, series[1], newton[1])


def _dd_criterion(mid, gaps, softplus, gm, k):
    """Criterion k of every row ([N]) from the midpoint and the [N, n_gaps] gap parameters as a double-double, gm
    the [n_gaps, n_thresh] signs with which the gaps build the criteria."""
    sg = jnp.asarray(gm)[:, k]
    c = (mid, jnp.zeros_like(mid))
    for g in range(gaps.shape[1]):
        v = _dd_gap(gaps[:, g], softplus)
        c = _dd_add(c, (sg[g] * v[0], sg[g] * v[1]))
    return c


def _dd_offset(a, b):
    """a - b for double-doubles a and b, rounded once."""
    hi, lo = _two_sum(a[0], -b[0])
    return hi + (lo + (a[1] - b[1]))


def _band_edge_grads(A, B, inner, has_lo):
    """(dP/d lower edge, dP/d upper edge) of a band given in the anchor-and-width form of _cell_lw."""
    return (jnp.where(inner, A - B, jnp.where(has_lo, A, 0.0)),
            jnp.where(inner, B, jnp.where(has_lo, 0.0, A)))


def _bivariate_edges(y, K, c, gv, own_idx, mu, sigma):
    """A rating dimension's cell under N(mu, sigma): (z_lo, z_hi, w, z_a, k_a, inner, has_lo, has_hi), the width w
    the cell's own gap over sigma and z_a, k_a its anchor edge and criterion."""
    r = y - 1
    n_thresh = K - 1
    obs = jnp.arange(y.shape[0])
    k_hi = jnp.clip(r, 0, n_thresh - 1)
    k_lo = jnp.clip(r - 1, 0, n_thresh - 1)
    has_hi, has_lo = r < K - 1, r > 0
    z_hi = jnp.where(has_hi, (c[obs, k_hi] - mu) / sigma, _OPEN)
    z_lo = jnp.where(has_lo, (c[obs, k_lo] - mu) / sigma, -_OPEN)
    w = gv[obs, jnp.asarray(own_idx)[r]] / sigma if gv.shape[1] else jnp.ones_like(z_lo)
    return (z_lo, z_hi, w, jnp.where(has_lo, z_lo, z_hi), jnp.where(has_lo, k_lo, k_hi), has_lo & has_hi,
            has_lo, has_hi)


def _cp_offset(thresh1_mid, gaps1, softplus, gm1, y1, n_thresh1, mu1, sigma1, mu2, sigma2, rho):
    """The crossing point -mu2 / (rho sigma2) less the detection cell's lower edge (c_lo - mu1) / sigma1, both in
    double-double, rounded once: the width of the cell's part below the crossing point to full precision."""
    k = jnp.clip(y1 - 2, 0, n_thresh1 - 1)
    c = _dd_criterion(thresh1_mid, gaps1, softplus, gm1, k)
    num = _dd_sum([c[0], -mu1, c[1]])
    z1 = _dd_div(num[0], num[1], sigma1, jnp.zeros_like(sigma1))
    d = _two_prod(rho, sigma2)
    d = _dd_sum([d[0], jnp.full_like(d[0], 1e-30), d[1]])
    zc = _dd_div(-mu2, jnp.zeros_like(mu2), d[0], d[1])
    return _dd_offset(zc, z1)


def _make_fused_batch_bounded_bivariate_sdt_loglik(K1, K2, gap_link_is_softplus):
    """Fully-vectorized batch log-likelihood for bounded bivariate_sdt.

    Bounded model: below a crossing point (cp), the source dimension becomes
    independent with mean=0 and conditional SD. Above cp, standard bivariate
    normal applies. For new items, z_cp=-20 so everything is above-cp.

    Each rating cell enters by its lower edge and width (_cell_lw, _rect_lw) wherever the crossing point
    does not split it, with fully analytic JVP gradients.
    """
    n_thresh1 = K1 - 1
    n_gaps1 = n_thresh1 - 1
    mid_idx1 = (n_thresh1 - 1) // 2
    n_upper1 = n_thresh1 - mid_idx1 - 1

    n_thresh2 = K2 - 1
    n_gaps2 = n_thresh2 - 1
    mid_idx2 = (n_thresh2 - 1) // 2
    n_upper2 = n_thresh2 - mid_idx2 - 1

    gap_mask1_np = _make_batch_gap_mask_np(n_gaps1, n_thresh1, mid_idx1, n_upper1)
    gap_mask2_np = _make_batch_gap_mask_np(n_gaps2, n_thresh2, mid_idx2, n_upper2)
    own_idx1, own_mask1 = _own_gap(K1)
    own_idx2, own_mask2 = _own_gap(K2)

    def pieces(y1, y2, item_type, mu1, mu2, sigma1, sigma2, rho_val, thresh1_mid, gv1, thresh2_mid, gv2, gaps1):
        c1 = _batch_build_thresh(thresh1_mid, gv1, n_gaps1, mid_idx1, n_upper1)
        c2 = _batch_build_thresh(thresh2_mid, gv2, n_gaps2, mid_idx2, n_upper2)
        e1 = _bivariate_edges(y1, K1, c1, gv1, own_idx1, mu1, sigma1)
        e2 = _bivariate_edges(y2, K2, c2, gv2, own_idx2, mu2, sigma2)
        z1_lo, z1_hi, w1, _, _, in1, has1_lo, has1_hi = e1
        z2_lo, z2_hi, w2, _, _, in2, has2_lo, has2_hi = e2
        one_minus_rho2 = jnp.clip(1.0 - rho_val ** 2, 1e-20)
        sigma_cond = jnp.maximum(sigma2 * jnp.sqrt(one_minus_rho2), sigma2 * 1e-6)
        inv_sc = 1.0 / sigma_cond
        z_cp = jnp.where(item_type == 1, -1e6, -mu2 / (rho_val * sigma2 + 1e-30))
        z_cp_eff = jnp.clip(z_cp, z1_lo, z1_hi)
        split = (z_cp > z1_lo) & (z_cp < z1_hi)
        above_only = z_cp <= z1_lo
        below_only = z_cp >= z1_hi
        obs = jnp.arange(y2.shape[0])
        k2_hi = jnp.clip(y2 - 1, 0, n_thresh2 - 1)
        k2_lo = jnp.clip(y2 - 2, 0, n_thresh2 - 1)
        zs_hi = jnp.where(has2_hi, c2[obs, k2_hi] * inv_sc, _OPEN)
        zs_lo = jnp.where(has2_lo, c2[obs, k2_lo] * inv_sc, -_OPEN)
        ws = w2 * sigma2 * inv_sc
        det_hi = z_cp_eff < 0.5 * _OPEN
        w_cut = _cp_offset(thresh1_mid, gaps1, gap_link_is_softplus, gap_mask1_np, y1, n_thresh1, mu1, sigma1, mu2,
                           sigma2, rho_val)
        w_det = jnp.where(below_only, w1, jnp.where(above_only, 1.0, jnp.where(has1_lo, w_cut, z_cp_eff - z1_lo)))
        det = _cell_lw("probit", z1_lo, z_cp_eff, jnp.where(has1_lo & det_hi, w_det, 1.0), has1_lo, det_hi)
        src = _cell_lw("probit", zs_lo, zs_hi, ws, has2_lo, has2_hi)
        w_ab = jnp.where(above_only, w1, jnp.where(in1, w1 - w_cut, z1_hi - z_cp_eff))
        rect = _rect_lw(z_cp_eff, z1_hi, w_ab, z2_lo, z2_hi, w2, rho_val)
        geo = (z_cp, z_cp_eff, split, above_only, below_only, zs_lo, zs_hi, ws, det_hi, inv_sc, one_minus_rho2,
               sigma_cond)
        return e1, e2, det, src, rect, geo

    def prob(det, src, rect, geo):
        above_only, below_only = geo[3], geo[4]
        p_det = jnp.where(above_only, 0.0, jnp.clip(det[0], 0.0))
        p_src = jnp.clip(src[0], 0.0)
        p_above = jnp.where(below_only, 0.0, jnp.clip(rect[0], 0.0))
        return p_det, p_src, p_above, p_det * p_src + p_above

    @custom_jvp
    def _batch_fused(y1, y2, item_type, mu1, mu2, sigma1, sigma2, rho_val,
                     thresh1_mid, gaps1, thresh2_mid, gaps2):
        gv1 = _gap_vals_and_derivs(gaps1, gap_link_is_softplus)[0]
        gv2 = _gap_vals_and_derivs(gaps2, gap_link_is_softplus)[0]
        _, _, det, src, rect, geo = pieces(y1, y2, item_type, mu1, mu2, sigma1, sigma2, rho_val, thresh1_mid, gv1,
                                           thresh2_mid, gv2, gaps1)
        return jnp.log(jnp.clip(prob(det, src, rect, geo)[3], _pfloor()))

    @_batch_fused.defjvp
    def _batch_fused_jvp(primals, tangents):
        (y1, y2, item_type, mu1, mu2, sigma1, sigma2, rho_val,
         thresh1_mid, gaps1, thresh2_mid, gaps2) = primals
        (_, _, _, d_mu1, d_mu2, d_sigma1, d_sigma2, d_rho,
         d_thresh1_mid, d_gaps1, d_thresh2_mid, d_gaps2) = tangents
        gv1, gd1 = _gap_vals_and_derivs(gaps1, gap_link_is_softplus)
        gv2, gd2 = _gap_vals_and_derivs(gaps2, gap_link_is_softplus)
        e1, e2, det, src, rect, geo = pieces(y1, y2, item_type, mu1, mu2, sigma1, sigma2, rho_val, thresh1_mid,
                                             gv1, thresh2_mid, gv2, gaps1)
        z1_lo, z1_hi, w1, z1_a, k1_a, in1, has1_lo, has1_hi = e1
        z2_lo, z2_hi, w2, z2_a, k2_a, in2, has2_lo, has2_hi = e2
        (z_cp, z_cp_eff, split, above_only, below_only, zs_lo, zs_hi, ws, det_hi, inv_sc, one_minus_rho2,
         sigma_cond) = geo
        p_det, p_src, p_above, P_raw = prob(det, src, rect, geo)
        P = jnp.clip(P_raw, _pfloor())
        val = jnp.log(P)
        inv_P = jnp.where(P_raw > _pfloor(), 1.0 / P, 0.0)

        det_on = jnp.where((det[0] > 0.0) & ~above_only, p_src, 0.0)
        src_on = jnp.where(src[0] > 0.0, p_det, 0.0)
        above_on = jnp.where((rect[0] > 0.0) & ~below_only, 1.0, 0.0)
        Ad, Bd = (jnp.where(det_on != 0.0, det[j], 0.0) * det_on for j in (1, 2))
        As, Bs = (jnp.where(src_on != 0.0, src[j], 0.0) * src_on for j in (1, 2))
        _, Ar1, Br1, Ar2, Br2, dr = (jnp.where(above_on > 0.0, x, 0.0) for x in rect)

        dd_lo, dd_hi = _band_edge_grads(Ad, Bd, has1_lo & det_hi, has1_lo)
        in_ab = (z_cp_eff > -0.5 * _OPEN) & has1_hi
        dr_lo, dr_hi = _band_edge_grads(Ar1, Br1, in_ab, z_cp_eff > -0.5 * _OPEN)
        cont = split & has1_lo & in_ab & (item_type != 1) & (sigma2 * jnp.sqrt(one_minus_rho2) > sigma2 * 1e-6)
        gA1 = jnp.where(cont, Ad + Ar1, jnp.where(split, dd_lo + dr_hi, jnp.where(below_only, Ad, Ar1)))
        gB1 = jnp.where(split, dr_hi, jnp.where(below_only, Bd, Br1))
        d_zcp = jnp.where(cont, 0.0, jnp.where(split, dd_hi + dr_lo, 0.0))

        inv_s1, inv_s2 = 1.0 / sigma1, 1.0 / sigma2
        g1 = _lw_chain(gA1 * inv_P, gB1 * inv_P, z1_a, w1, in1, inv_s1, z1_hi)
        dg1 = _lw_gap_grads(gA1 * inv_P * inv_s1, gB1 * inv_P * inv_s1, k1_a, jnp.asarray(own_mask1)[:, y1 - 1], in1,
                            gap_mask1_np, gd1)

        zs_a = jnp.where(has2_lo, zs_lo, zs_hi)
        Bs_in = jnp.where(in2, Bs, 0.0)
        g2 = _lw_chain(Ar2 * inv_P, Br2 * inv_P, z2_a, w2, in2, inv_s2, z2_hi)
        src_scale = -(As * zs_a + Bs_in * jnp.where(in2, ws, 0.0)) * inv_P
        dg2 = _lw_gap_grads((Ar2 * inv_s2 + As * inv_sc) * inv_P, (Br2 * inv_s2 + Bs * inv_sc) * inv_P, k2_a,
                            jnp.asarray(own_mask2)[:, y2 - 1], in2, gap_mask2_np, gd2)

        rho_sigma2_safe = jnp.where(jnp.abs(rho_val * sigma2 + 1e-30) > 1e-20, rho_val * sigma2 + 1e-30, 1.0)
        rho_safe = jnp.where(jnp.abs(rho_val) > 1e-20, rho_val, 1.0)
        dz_cp_dmu2 = -1.0 / rho_sigma2_safe
        dz_cp_ds2 = mu2 / (rho_sigma2_safe * sigma2)
        dz_cp_drho = mu2 / (rho_safe * rho_sigma2_safe)
        rho_factor = jnp.where((sigma2 * jnp.sqrt(one_minus_rho2) > sigma2 * 1e-6) & (1.0 - rho_val ** 2 > 1e-20),
                               rho_val / one_minus_rho2, 0.0)
        d_zcp = d_zcp * inv_P

        d_lp_dmu2 = g2[0] + d_zcp * dz_cp_dmu2
        d_lp_dsigma2 = g2[1] + src_scale * inv_s2 + d_zcp * dz_cp_ds2
        d_lp_drho = dr * inv_P + d_zcp * dz_cp_drho - src_scale * rho_factor
        d_lp_d_thresh2_mid = g2[2] + As * inv_sc * inv_P

        tangent_out = (g1[0] * d_mu1 + d_lp_dmu2 * d_mu2 + g1[1] * d_sigma1 + d_lp_dsigma2 * d_sigma2 +
                       d_lp_drho * d_rho + g1[2] * d_thresh1_mid + jnp.sum(dg1 * d_gaps1, axis=1) +
                       d_lp_d_thresh2_mid * d_thresh2_mid + jnp.sum(dg2 * d_gaps2, axis=1))
        return val, tangent_out

    return _batch_fused


# =============================================================================
# Batch-vectorized vrdp2d loglik (no vmap)
# =============================================================================

def _make_fused_batch_vrdp2d_loglik(K1, K2, gap_link_is_softplus):
    """Fully-vectorized batch log-likelihood for VRDP2D.

    P = (1-lambda)*p1_fam*p2_noise + lambda*p1_rec*p2_rec
    All univariate — no bivariate CDF needed. Each cell enters by its lower edge and width; with the
    native kernel loaded, rows holding a narrow cell take the width-form JAX path.
    """
    n_thresh1 = K1 - 1
    n_gaps1 = n_thresh1 - 1
    mid_idx1 = (n_thresh1 - 1) // 2
    n_upper1 = n_thresh1 - mid_idx1 - 1

    n_thresh2 = K2 - 1
    n_gaps2 = n_thresh2 - 1
    mid_idx2 = (n_thresh2 - 1) // 2
    n_upper2 = n_thresh2 - mid_idx2 - 1

    gap_mask1_np = _make_batch_gap_mask_np(n_gaps1, n_thresh1, mid_idx1, n_upper1)
    gap_mask2_np = _make_batch_gap_mask_np(n_gaps2, n_thresh2, mid_idx2, n_upper2)
    own_idx1, own_mask1 = _own_gap(K1)
    own_idx2, own_mask2 = _own_gap(K2)

    def setup(y1, y2, dprime_F, dprime_R, source_d, sigma_item, sigma_S, thresh1_mid, gv1, thresh2_mid, gv2):
        c1 = _batch_build_thresh(thresh1_mid, gv1, n_gaps1, mid_idx1, n_upper1)
        c2 = _batch_build_thresh(thresh2_mid, gv2, n_gaps2, mid_idx2, n_upper2)
        one = jnp.ones_like(dprime_F)
        f1 = _bivariate_edges(y1, K1, c1, gv1, own_idx1, dprime_F, sigma_item)
        r1 = _bivariate_edges(y1, K1, c1, gv1, own_idx1, dprime_F + dprime_R, sigma_item)
        n2 = _bivariate_edges(y2, K2, c2, gv2, own_idx2, jnp.zeros_like(dprime_F), one)
        r2 = _bivariate_edges(y2, K2, c2, gv2, own_idx2, source_d, sigma_S)
        obs = jnp.arange(y1.shape[0])
        cut = lambda c, y, n: (c[obs, jnp.clip(y - 1, 0, n - 1)], c[obs, jnp.clip(y - 2, 0, n - 1)])
        return f1, r1, n2, r2, cut(c1, y1, n_thresh1), cut(c2, y2, n_thresh2)

    def narrow_rows(bands):
        out = jnp.zeros(bands[0][0].shape, dtype=bool)
        for e in bands:
            out = out | _probit_narrow(jnp.where(e[5], e[0], 0.0), jnp.where(e[5], e[2], 1.0), e[5])[0]
        return out

    def jax_path(f1, r1, n2, r2, lambda_val, sigma_item, sigma_S):
        cells = [_cell_lw("probit", e[0], e[1], e[2], e[6], e[7]) for e in (f1, r1, n2, r2)]
        (D1f, A1f, B1f, _), (D1r, A1r, B1r, _), (D2n, A2n, B2n, _), (D2r, A2r, B2r, _) = cells
        fl = _pfloor()
        p1f, p1r, p2n, p2r = (jnp.clip(D, fl) for D in (D1f, D1r, D2n, D2r))
        p_fam, p_rec = p1f * p2n, p1r * p2r
        prob_raw = (1.0 - lambda_val) * p_fam + lambda_val * p_rec
        prob = jnp.clip(prob_raw, fl)
        inv_P = jnp.where(prob_raw > fl, 1.0 / prob, 0.0)
        cf1 = jnp.where(D1f > fl, (1.0 - lambda_val) * p2n * inv_P, 0.0)
        cr1 = jnp.where(D1r > fl, lambda_val * p2r * inv_P, 0.0)
        cn2 = jnp.where(D2n > fl, (1.0 - lambda_val) * p1f * inv_P, 0.0)
        cr2 = jnp.where(D2r > fl, lambda_val * p1r * inv_P, 0.0)
        inv_si, inv_sS = 1.0 / sigma_item, 1.0 / sigma_S
        gf, gr = _lw_chain(cf1 * A1f, cf1 * B1f, f1[3], f1[2], f1[5], inv_si, f1[1]), \
            _lw_chain(cr1 * A1r, cr1 * B1r, r1[3], r1[2], r1[5], inv_si, r1[1])
        gs = _lw_chain(cr2 * A2r, cr2 * B2r, r2[3], r2[2], r2[5], inv_sS, r2[1])
        T1A = gf[2] + gr[2]
        T1B = jnp.where(f1[5], (cf1 * B1f + cr1 * B1r) * inv_si, 0.0)
        T2A = cn2 * A2n + gs[2]
        T2B = jnp.where(n2[5], cn2 * B2n + cr2 * B2r * inv_sS, 0.0)
        return (jnp.log(prob), gf[0] + gr[0], gr[0], gs[0], (p_rec - p_fam) * inv_P, gf[1] + gr[1], gs[1],
                T1A, T1B, T2A, T2B)

    def evaluate(y1, y2, dprime_F, dprime_R, source_d, lambda_val, sigma_item, sigma_S, thresh1_mid, gv1,
                 thresh2_mid, gv2):
        f1, r1, n2, r2, (c1_hi, c1_lo), (c2_hi, c2_lo) = setup(y1, y2, dprime_F, dprime_R, source_d, sigma_item,
                                                               sigma_S, thresh1_mid, gv1, thresh2_mid, gv2)
        has1_lo, has1_hi, has2_lo, has2_hi = f1[6], f1[7], n2[6], n2[7]
        nat = _vrdp2d_native(c1_hi, c1_lo, c2_hi, c2_lo, has1_hi, has1_lo, has2_hi, has2_lo,
                             dprime_F, dprime_R, source_d, lambda_val, sigma_item, sigma_S)
        if nat is None:
            out = jax_path(f1, r1, n2, r2, lambda_val, sigma_item, sigma_S)
        else:
            out = nat[:7] + (nat[7] + nat[8], jnp.where(f1[5], nat[7], 0.0), nat[9] + nat[10],
                             jnp.where(n2[5], nat[9], 0.0))
            narrow = narrow_rows((f1, r1, n2, r2))
            out = jax.lax.cond(jnp.any(narrow),
                               lambda o: tuple(jnp.where(narrow, a, b) for a, b in
                                               zip(jax_path(f1, r1, n2, r2, lambda_val, sigma_item, sigma_S), o)),
                               lambda o: o, out)
        return out, f1, n2

    @custom_jvp
    def _batch_fused(y1, y2, item_type, dprime_F, dprime_R, source_d, lambda_val,
                     sigma_item, sigma_S, thresh1_mid, gaps1, thresh2_mid, gaps2):
        gv1 = _gap_vals_and_derivs(gaps1, gap_link_is_softplus)[0]
        gv2 = _gap_vals_and_derivs(gaps2, gap_link_is_softplus)[0]
        return evaluate(y1, y2, dprime_F, dprime_R, source_d, lambda_val, sigma_item, sigma_S, thresh1_mid, gv1,
                        thresh2_mid, gv2)[0][0]

    @_batch_fused.defjvp
    def _batch_fused_jvp(primals, tangents):
        (y1, y2, item_type, dprime_F, dprime_R, source_d, lambda_val,
         sigma_item, sigma_S, thresh1_mid, gaps1, thresh2_mid, gaps2) = primals
        (_, _, _, d_dprime_F, d_dprime_R, d_source_d, d_lambda,
         d_sigma_item, d_sigma_S, d_thresh1_mid, d_gaps1, d_thresh2_mid, d_gaps2) = tangents
        gv1, gd1 = _gap_vals_and_derivs(gaps1, gap_link_is_softplus)
        gv2, gd2 = _gap_vals_and_derivs(gaps2, gap_link_is_softplus)
        out, f1, n2 = evaluate(y1, y2, dprime_F, dprime_R, source_d, lambda_val, sigma_item, sigma_S, thresh1_mid,
                               gv1, thresh2_mid, gv2)
        val, d_dpF, d_dpR, d_sd, d_lam, d_si, d_sS, T1A, T1B, T2A, T2B = out
        dg1 = _lw_gap_grads(T1A, T1B, f1[4], jnp.asarray(own_mask1)[:, y1 - 1], f1[5], gap_mask1_np, gd1)
        dg2 = _lw_gap_grads(T2A, T2B, n2[4], jnp.asarray(own_mask2)[:, y2 - 1], n2[5], gap_mask2_np, gd2)
        tangent_out = (d_dpF * d_dprime_F + d_dpR * d_dprime_R + d_sd * d_source_d + d_lam * d_lambda +
                       d_si * d_sigma_item + d_sS * d_sigma_S +
                       T1A * d_thresh1_mid + jnp.sum(dg1 * d_gaps1, axis=1) +
                       T2A * d_thresh2_mid + jnp.sum(dg2 * d_gaps2, axis=1))
        return val, tangent_out

    return _batch_fused


# =============================================================================
# Batch-vectorized CDP loglik (no vmap)
# =============================================================================

def _cdp_strip_parts(tau_lo, gap, mu_R, sigma_R, mu_F, sigma_F, c_R, c_K, corr):
    sM = jnp.sqrt(sigma_R ** 2 + sigma_F ** 2 + 2.0 * corr * sigma_R * sigma_F)
    rho = (sigma_R + corr * sigma_F) / sM
    r = jnp.clip(rho, -_RHO_MAX, _RHO_MAX)
    s = jnp.sqrt((1.0 - r) * (1.0 + r))
    return sM, rho, r, s


def _cdp_strip_scale(kind, tau_lo, gap, mu_R, sigma_R, mu_F, sigma_F, c_R, c_K, corr):
    """How far a CDP cell's integrand varies across its band on M, in units of the band's width."""
    sM, _, r, s = _cdp_strip_parts(tau_lo, gap, mu_R, sigma_R, mu_F, sigma_F, c_R, c_K, corr)
    v = gap / sM
    m = (tau_lo + 0.5 * gap - mu_R - mu_F) / sM
    b = ((c_R - mu_R) / sigma_R - r * m) / s
    a = ((tau_lo + 0.5 * gap - c_K - mu_R) / sigma_R - r * m) / s
    u = jnp.maximum(1.0, jnp.abs(m))
    u = u + jnp.where(kind > 0, jnp.abs(r) / s * jnp.maximum(1.0, jnp.abs(b)), 0.0)
    u = u + jnp.where(kind > 2, jnp.abs(sM / sigma_R - r) / s * jnp.maximum(1.0, jnp.abs(a)), 0.0)
    return v * u


def _cdp_strip(kind, tau_lo, gap, mu_R, sigma_R, mu_F, sigma_F, c_R, c_K, corr, kap_off=None):
    """A CDP cell whose band on M is narrow, by 4-point Gauss-Legendre over the band [tau_lo, tau_lo + gap]
    with R | M normal: kind 0 the band, 1 Remember (R > c_R), 2 Know of R/K (R <= c_R), 3 Know of R/K/G
    (R < min(c_R, M - c_K)), 4 Guess (M - c_K <= R <= c_R). Returns P and its derivatives in (mu_R, sigma_R,
    mu_F, sigma_F, c_R, c_K, corr, tau_lo at fixed gap, gap), each the derivative of the quadrature. The Guess
    cell's interval in R, of width (c_R + c_K - M) / sigma_R, enters by its lower edge and that width, from
    `kap_off` = c_R + c_K - tau_lo, so that it keeps full accuracy where it is narrow."""
    sM, rho, r, s = _cdp_strip_parts(tau_lo, gap, mu_R, sigma_R, mu_F, sigma_F, c_R, c_K, corr)
    e = lambda q: q[..., None]
    x = jnp.asarray((_GL4_X + 1.0) / 2.0, dtype=tau_lo.dtype)
    W = jnp.asarray(_GL4_W / 2.0, dtype=tau_lo.dtype)
    M = e(tau_lo) + e(gap) * x
    t = (M - e(mu_R + mu_F)) / e(sM)
    om = e(gap) * W * jnp.exp(-0.5 * t * t - _LOG_SQRT_2PI) / e(sM)
    zcR = (c_R - mu_R) / sigma_R
    zs = (M - e(c_K + mu_R)) / e(sigma_R)
    b = (e(zcR) - e(r) * t) / e(s)
    a = (zs - e(r) * t) / e(s)
    Pb, pb, qb = _probit_triple(b)
    Pa, pa, qa = _probit_triple(a)
    k = e(kind)
    lt = a < b
    zero = jnp.zeros_like(t)
    C = jnp.where(k == 0, 1.0, jnp.where(k == 1, jnp.where(b >= 0, qb, 1.0 - qb), jnp.where(k == 2, Pb,
        jnp.where(k == 3, jnp.where(lt, Pa, Pb), jnp.where(lt, jnp.where(a >= 0, qa - qb, Pb - Pa), 0.0)))))
    Cb = jnp.where(k == 1, -pb, jnp.where(k == 2, pb, jnp.where(k == 3, jnp.where(lt, 0.0, pb),
         jnp.where(k == 4, jnp.where(lt, pb, 0.0), zero))))
    Ca = jnp.where(k == 3, jnp.where(lt, pa, 0.0), jnp.where(k == 4, jnp.where(lt, -pa, 0.0), zero))
    S1, S2, S3 = Cb + Ca, Cb * b + Ca * a, Cb * e(zcR) + Ca * zs
    if kap_off is None:
        kap_off = c_R + c_K - tau_lo
    wR = (e(kap_off) - e(gap) * x) / e(sigma_R * s)
    guess = (k == 4) & (wR > 0.0)
    wg = jnp.where(guess, wR, 1.0)
    one = jnp.ones_like(wg, dtype=bool)
    Dg, Ag, Bg = _probit_band_lw(a, a + wg, wg, one, one)
    C = jnp.where(k == 4, jnp.where(guess, Dg, 0.0), C)
    Cb = jnp.where(k == 4, jnp.where(guess, Bg, 0.0), Cb)
    Ca = jnp.where(k == 4, jnp.where(guess, Ag - Bg, 0.0), Ca)
    S1 = jnp.where(k == 4, jnp.where(guess, Ag, 0.0), S1)
    S2 = jnp.where(k == 4, jnp.where(guess, Ag * a + Bg * wg, 0.0), S2)
    S3 = jnp.where(k == 4, jnp.where(guess, Ag * zs + Bg * wg * e(s), 0.0), S3)
    P = jnp.sum(om * C, -1)
    Kt = om * (-C * t - e(r / s) * S1)
    cM = Kt / e(sM) + om * Ca / e(s * sigma_R)
    d_tau = jnp.sum(cM, -1)
    d_gap = jnp.sum(cM * x, -1) + P / gap
    d_muM = -jnp.sum(Kt, -1) / sM
    d_sM = -jnp.sum(Kt * t + om * C, -1) / sM
    d_zcR = jnp.sum(om * Cb, -1) / s
    S_a = jnp.sum(om * Ca, -1) / s
    d_r = jnp.sum(om * (-t * S1 + S2 * e(r / s)), -1) / s
    d_r = jnp.where(jnp.abs(rho) < _RHO_MAX, d_r, 0.0)
    sM3 = sM ** 3
    drho_dsR = sigma_F ** 2 * (1.0 - corr * corr) / sM3
    drho_dsF = corr / sM - (sigma_R + corr * sigma_F) * (sigma_F + corr * sigma_R) / sM3
    drho_dc = sigma_F / sM - (sigma_R + corr * sigma_F) * sigma_R * sigma_F / sM3
    d_muR = d_muM - jnp.sum(om * S1, -1) / (s * sigma_R)
    d_sR = (d_sM * (sigma_R + corr * sigma_F) / sM - jnp.sum(om * S3, -1) / (s * sigma_R) + d_r * drho_dsR)
    d_sF = d_sM * (sigma_F + corr * sigma_R) / sM + d_r * drho_dsF
    d_corr = d_sM * sigma_R * sigma_F / sM + d_r * drho_dc
    return P, (d_muR, d_sR, d_muM, d_sF, d_zcR / sigma_R, -S_a / sigma_R, d_corr, d_tau, d_gap)


def _cdp_strip_split(kind, tau_lo, gap, mu_R, sigma_R, mu_F, sigma_F, c_R, c_K, corr, g1=None):
    """_cdp_strip with a band that holds the Guess boundary M = c_R + c_K (Know and Guess cells, where the
    integrand has a corner) taken as two strips meeting there; same outputs. `g1`: the boundary's offset in the
    band, c_R + c_K - tau_lo, when it is known more precisely than that difference."""
    if g1 is None:
        g1 = c_R + c_K - tau_lo
    kap = tau_lo + g1
    split = (kind > 2) & (g1 > 0.0) & (g1 < gap)
    P1, d1 = _cdp_strip(kind, tau_lo, jnp.where(split, g1, gap), mu_R, sigma_R, mu_F, sigma_F, c_R, c_K, corr, g1)
    P2, d2 = _cdp_strip(kind, jnp.where(split, kap, tau_lo), jnp.where(split, gap - g1, gap), mu_R, sigma_R, mu_F,
                        sigma_F, c_R, c_K, corr, jnp.where(split, 0.0, g1))
    on = lambda x: jnp.where(split, x, 0.0)
    d_kap = on(d1[8] + d2[7] - d2[8])
    d = [d1[k] + on(d2[k]) for k in range(7)]
    d[4] = d[4] + d_kap
    d[5] = d[5] + d_kap
    return P1 + on(P2), tuple(d) + (d1[7] + on(d2[8] - d1[8]), jnp.where(split, d2[8], d1[8]))


def _make_fused_batch_cdp_loglik(K, J, gap_link_is_softplus, n_rkg=2, has_corr=False):
    """Fully-vectorized batch log-likelihood for CDP.

    CDP uses J tau thresholds and bivariate strip probabilities.
    Supports both n_rkg==2 (R/K) and n_rkg==3 (R/K/G). `corr` is each observation's
    recollection-familiarity correlation; without `has_corr` it is zero and the lower quadrant
    P(R <= ., F <= c_K) of the Guess region stays a product of normal CDFs.
    """
    n_thresh = J
    n_gaps = n_thresh - 1
    mid_idx = (n_thresh - 1) // 2
    n_upper = n_thresh - mid_idx - 1
    gap_mask_np = _make_batch_gap_mask_np(n_gaps, n_thresh, mid_idx, n_upper)
    _own_full, _own_full_mask = _own_gap(J + 1)
    own_tau = _own_full[np.arange(J) + 1]
    own_tau_mask = np.concatenate([_own_full_mask[:, 1:J], np.zeros((n_gaps, 1))], axis=1)

    def _narrow(rk, has_hi, old_idx, gv, gaps, mid, tau_lo, mu_R, sigma_R, mu_F, sigma_F, c_R, c_K, corr):
        inner = has_hi
        k = jnp.clip(old_idx, 0, J - 1)
        gap = jnp.where(inner, gv[jnp.arange(rk.shape[0]), jnp.asarray(own_tau)[k]], 1.0) if n_gaps else \
            jnp.ones_like(mu_R)
        kind = jnp.where(rk == 1, 1, jnp.where(rk == 2, 2 if n_rkg == 2 else 3, 4))
        lo = jnp.where(inner, tau_lo, 0.0)
        narrow = inner & (_cdp_strip_scale(kind, lo, gap, mu_R, sigma_R, mu_F, sigma_F, c_R, c_K, corr) < _NARROW_U)
        gs = jnp.where(narrow, gap, 1e-3)
        g1 = (_dd_offset(_dd_sum([c_R, c_K]), _dd_criterion(mid, gaps, gap_link_is_softplus, gap_mask_np, k))
              if n_gaps else c_R + c_K - lo)
        return narrow, inner, kind, lo, gs, k, g1

    def _rk_tail(tail, is_R, z_cR, z_lo, z_hi, rho):
        """Remember/Know strips as direct rectangles in (R, M) for the cells flagged
        in `tail` (p, dP/dz_cR, dP/dz_lo, dP/dz_hi, dP/drho); zeros elsewhere. Only
        evaluated when some cell is below the tail switch."""
        def run(_):
            lo1 = jnp.where(is_R, z_cR, -_OPEN)
            hi1 = jnp.where(is_R, _OPEN, z_cR)
            p, d1l, d1h, d2l, d2h, dr = _bvn_rect(lo1, hi1, z_lo, z_hi, rho)
            return p, jnp.where(is_R, d1l, d1h), d2l, d2h, dr

        def skip(_):
            z = jnp.zeros_like(z_cR)
            return z, z, z, z, z
        return jax.lax.cond(jnp.any(tail), run, skip, None)

    def _lower_quadrants(z_q_hi, z_q_lo, z_cK_F, corr):
        """Phi2(z, z_cK_F; corr) at the one R bound each Guess edge needs: (P, dP/dz, dP/dz_cK_F, dP/dcorr) each."""
        q, _ = _bvn_cdf_multi([(z_q_hi, z_cK_F), (z_q_lo, z_cK_F)], [z_cK_F], corr)
        return [tuple(q[j][:4]) for j in range(2)]

    @custom_jvp
    def _batch_fused(y, rk, is_old_val, old_idx,
                     mu_R, sigma_R, mu_F, sigma_F, c_R, c_K, corr,
                     thresh_mid, gaps):
        gap_fn = jax.nn.softplus if gap_link_is_softplus else jnp.exp

        gv = gap_fn(gaps)
        c = _batch_build_thresh(thresh_mid, gv, n_gaps, mid_idx, n_upper)

        N = y.shape[0]
        obs_idx = jnp.arange(N)

        mu_M = mu_R + mu_F
        sigma_M = jnp.sqrt(sigma_R**2 + sigma_F**2 + 2.0 * corr * sigma_R * sigma_F)
        rho = (sigma_R + corr * sigma_F) / jnp.clip(sigma_M, 1e-10)

        has_hi = old_idx < J - 1
        k_hi = jnp.clip(old_idx + 1, 0, J - 1)
        tau_lo = c[obs_idx, jnp.clip(old_idx, 0, J - 1)]
        tau_hi = jnp.where(has_hi, c[obs_idx, k_hi], _OPEN)

        z_cR = (c_R - mu_R) / jnp.clip(sigma_R, 1e-10)
        z_lo = (tau_lo - mu_M) / jnp.clip(sigma_M, 1e-10)
        z_hi = jnp.where(has_hi, (tau_hi - mu_M) / jnp.clip(sigma_M, 1e-10), _OPEN)


        if n_rkg == 2:
            (c0, c1_), _ = _bvn_cdf_multi([(z_cR, z_hi), (z_cR, z_lo)], [z_cR], rho)
            bv_cR_hi, bv_cR_lo = c0[0], c1_[0]
            p_band = _band_from_q(z_lo, z_hi, c1_[6], c0[6])
            p_K_strip = bv_cR_hi - bv_cR_lo
            p_rk = jnp.where(rk == 1, p_band - p_K_strip, p_K_strip)
            tail = p_rk < _TAIL_SWITCH
            p_t = _rk_tail(tail, rk == 1, z_cR, z_lo, z_hi, rho)[0]
            p_rk = jnp.clip(jnp.where(tail, p_t, p_rk), _pfloor())
        else:
            z_cK_F = (c_K - mu_F) / jnp.clip(sigma_F, 1e-10)
            s_hi = jnp.fmin(c_R, tau_hi - c_K)
            s_lo = jnp.fmin(c_R, tau_lo - c_K)
            case1_hi = (c_R <= tau_hi - c_K)
            case1_lo = (c_R <= tau_lo - c_K)
            z_s_hi = (s_hi - mu_R) / jnp.clip(sigma_R, 1e-10)
            z_s_lo = (s_lo - mu_R) / jnp.clip(sigma_R, 1e-10)

            (c0, c1_, c2_, c3_), (e_ckf, _) = _bvn_cdf_multi(
                [(z_cR, z_hi), (z_cR, z_lo), (z_s_hi, z_hi), (z_s_lo, z_lo)], [z_cK_F, z_cR], rho)
            bv_cR_hi, bv_cR_lo, bv_s_hi, bv_s_lo = c0[0], c1_[0], c2_[0], c3_[0]
            p_band = _band_from_q(z_lo, z_hi, c1_[6], c0[6])

            p_R_raw = p_band - (bv_cR_hi - bv_cR_lo)

            Phi_zcR = _phi_from_q(z_cR, c0[4])
            Phi_zcKF = _phi_from_q(z_cK_F, e_ckf[0])
            if has_corr:
                qh, ql = _lower_quadrants(jnp.where(case1_hi, z_cR, z_s_hi), jnp.where(case1_lo, z_cR, z_s_lo), z_cK_F, corr)
                P_hi, P_lo = qh[0], ql[0]
            else:
                P_c = Phi_zcR * Phi_zcKF
                P_hi = jnp.where(case1_hi, P_c, _phi_from_q(z_s_hi, c2_[4]) * Phi_zcKF)
                P_lo = jnp.where(case1_lo, P_c, _phi_from_q(z_s_lo, c3_[4]) * Phi_zcKF)
            G_hi = jnp.where(case1_hi, P_hi, P_hi + bv_cR_hi - bv_s_hi)
            G_lo = jnp.where(case1_lo, P_lo, P_lo + bv_cR_lo - bv_s_lo)

            p_G_raw = G_hi - G_lo
            p_K_raw = p_band - p_R_raw - p_G_raw

            p_sel = jnp.where(rk == 1, p_R_raw, jnp.where(rk == 2, p_K_raw, p_G_raw))
            tail = ~(p_sel >= _TAIL_SWITCH)
            t_lp, _ = _cdp_tail(tail, rk, mu_R, sigma_R, mu_F, sigma_F, c_R, c_K,
                                tau_lo, jnp.where(has_hi, tau_hi, jnp.inf), corr)
            p_rk = jnp.where(tail, jnp.exp(t_lp), p_sel)

        val = jnp.log(jnp.clip(p_rk, _pfloor()))
        narrow, _, kind, lo, gs, _, g1 = _narrow(rk, has_hi, old_idx, gv, gaps, thresh_mid, tau_lo, mu_R, sigma_R,
                                                 mu_F, sigma_F, c_R, c_K, corr)
        val = jax.lax.cond(jnp.any(narrow), lambda v: jnp.where(narrow, jnp.log(jnp.clip(
            _cdp_strip_split(kind, lo, gs, mu_R, sigma_R, mu_F, sigma_F, c_R, c_K, corr, g1)[0], _pfloor())), v),
            lambda v: v, val)
        if n_rkg == 3:
            empty = (rk == 3) & (c_R + c_K <= tau_lo)
            val = jnp.where(empty, _empty_cell_lp(tau_lo - c_R - c_K), val)
        return val

    @_batch_fused.defjvp
    def _batch_fused_jvp(primals, tangents):
        (y, rk, is_old_val, old_idx,
         mu_R, sigma_R, mu_F, sigma_F, c_R, c_K, corr,
         thresh_mid_v, gaps_v) = primals
        (_, _, _, _,
         d_mu_R, d_sigma_R, d_mu_F, d_sigma_F, d_c_R, d_c_K, d_corr,
         d_thresh_mid_v, d_gaps_v) = tangents

        if gap_link_is_softplus:
            gv = jax.nn.softplus(gaps_v); gd = jax.nn.sigmoid(gaps_v)
        else:
            gv = jnp.exp(gaps_v); gd = gv

        c = _batch_build_thresh(thresh_mid_v, gv, n_gaps, mid_idx, n_upper)

        N = y.shape[0]
        obs_idx = jnp.arange(N)

        mu_M = mu_R + mu_F
        sigma_M = jnp.sqrt(sigma_R**2 + sigma_F**2 + 2.0 * corr * sigma_R * sigma_F)
        rho = (sigma_R + corr * sigma_F) / jnp.clip(sigma_M, 1e-10)
        inv_sR = 1.0 / jnp.clip(sigma_R, 1e-10)
        inv_sF = 1.0 / jnp.clip(sigma_F, 1e-10)
        inv_sM = 1.0 / jnp.clip(sigma_M, 1e-10)

        has_tau_hi = old_idx < J - 1
        k_tau_lo = jnp.clip(old_idx, 0, n_thresh - 1)
        k_tau_hi = jnp.clip(old_idx + 1, 0, n_thresh - 1)

        tau_lo = c[obs_idx, k_tau_lo]
        tau_hi = jnp.where(has_tau_hi, c[obs_idx, k_tau_hi], _OPEN)

        z_cR = (c_R - mu_R) * inv_sR
        z_hi = jnp.where(has_tau_hi, (tau_hi - mu_M) * inv_sM, _OPEN)
        z_lo = (tau_lo - mu_M) * inv_sM


        if n_rkg == 2:
            cdfs, _ = _bvn_cdf_multi([(z_cR, z_hi), (z_cR, z_lo)], [z_cR], rho)
        else:
            z_s_hi = (jnp.fmin(c_R, tau_hi - c_K) - mu_R) * inv_sR
            z_s_lo = (jnp.fmin(c_R, tau_lo - c_K) - mu_R) * inv_sR
            z_cK_F = (c_K - mu_F) * inv_sF
            cdfs, (e_ckf, _) = _bvn_cdf_multi(
                [(z_cR, z_hi), (z_cR, z_lo), (z_s_hi, z_hi), (z_s_lo, z_lo)], [z_cK_F, z_cR], rho)
        bv_cR_hi, d_bvCRhi_z1, d_bvCRhi_z2, d_bvCRhi_rho = cdfs[0][:4]
        bv_cR_lo, d_bvCRlo_z1, d_bvCRlo_z2, d_bvCRlo_rho = cdfs[1][:4]
        p_band = _band_from_q(z_lo, z_hi, cdfs[1][6], cdfs[0][6])

        phi_hi = jnp.where(has_tau_hi, cdfs[0][7], 0.0)
        phi_lo = cdfs[1][7]

        # d sigma_M / d sigma_R, d sigma_M / d sigma_F and the R-M correlation's derivatives
        sR_over_sM = (sigma_R + corr * sigma_F) * inv_sM
        sF_over_sM = (sigma_F + corr * sigma_R) * inv_sM
        sM3 = jnp.clip(sigma_M ** 3, 1e-20)
        drho_dsR = sigma_F**2 * (1.0 - corr * corr) / sM3
        drho_dsF = corr * inv_sM - (sigma_R + corr * sigma_F) * (sigma_F + corr * sigma_R) / sM3
        dsM_dc = sigma_R * sigma_F * inv_sM
        drho_dc = sigma_F * inv_sM - (sigma_R + corr * sigma_F) * sigma_R * sigma_F / sM3

        if n_rkg == 2:
            p_K_strip = bv_cR_hi - bv_cR_lo
            is_R = (rk == 1)
            p_rk_raw = jnp.where(is_R, p_band - p_K_strip, p_K_strip)
            tail = p_rk_raw < _TAIL_SWITCH
            t_p, t_dzcR, t_dzlo, t_dzhi, t_drho = _rk_tail(tail, is_R, z_cR, z_lo, z_hi, rho)
            p_rk_raw = jnp.where(tail, t_p, p_rk_raw)
            p_rk = jnp.clip(p_rk_raw, _pfloor())
            prob_raw = p_rk_raw
            prob = p_rk
            val = jnp.log(jnp.clip(prob, _pfloor()))
            inv_prob = jnp.where(prob_raw > _pfloor(), 1.0 / jnp.clip(prob, _pfloor()), 0.0)

            g_zcR = jnp.where(tail, t_dzcR, jnp.where(is_R, -d_bvCRhi_z1 + d_bvCRlo_z1,
                                                        d_bvCRhi_z1 - d_bvCRlo_z1))
            g_zhi = jnp.where(tail, t_dzhi, jnp.where(is_R, phi_hi - d_bvCRhi_z2, d_bvCRhi_z2))
            g_zlo = jnp.where(tail, t_dzlo, jnp.where(is_R, -(phi_lo - d_bvCRlo_z2), -d_bvCRlo_z2))
            g_rho = jnp.where(tail, t_drho, jnp.where(is_R, -d_bvCRhi_rho + d_bvCRlo_rho,
                                                        d_bvCRhi_rho - d_bvCRlo_rho))
            dlp_dzcR = g_zcR * inv_prob
            dlp_dzhi = jnp.where(has_tau_hi, g_zhi * inv_prob, 0.0)
            dlp_dzlo = g_zlo * inv_prob
            dlp_drho = g_rho * inv_prob

            d_val_dmuR = (dlp_dzcR * (-inv_sR) +
                          (dlp_dzhi + dlp_dzlo) * (-inv_sM))
            d_val_dmuF = (dlp_dzhi + dlp_dzlo) * (-inv_sM)
            d_val_dsigR = (dlp_dzcR * (-z_cR * inv_sR) +
                           (dlp_dzhi * (-z_hi) + dlp_dzlo * (-z_lo)) * inv_sM * sR_over_sM +
                           dlp_drho * drho_dsR)
            d_val_dsigF = ((dlp_dzhi * (-z_hi) + dlp_dzlo * (-z_lo)) * inv_sM * sF_over_sM +
                           dlp_drho * drho_dsF)
            d_val_dcR = dlp_dzcR * inv_sR
            d_val_dcK = jnp.zeros_like(c_K)
            d_val_dcorr = ((dlp_dzhi * (-z_hi) + dlp_dzlo * (-z_lo)) * inv_sM * dsM_dc +
                           dlp_drho * drho_dc)

            d_val_dtau_hi = dlp_dzhi * inv_sM
            d_val_dtau_lo = dlp_dzlo * inv_sM
            live = prob_raw > _pfloor()
            (d_val_dmuR, d_val_dsigR, d_val_dmuF, d_val_dsigF, d_val_dcR, d_val_dcK, d_val_dcorr, d_val_dtau_lo,
             d_val_dtau_hi) = (jnp.where(live, d, 0.0) for d in (d_val_dmuR, d_val_dsigR, d_val_dmuF, d_val_dsigF,
                                                              d_val_dcR, d_val_dcK, d_val_dcorr, d_val_dtau_lo,
                                                              d_val_dtau_hi))

        else:
            z_cK_F = (c_K - mu_F) * inv_sF
            s_hi = jnp.fmin(c_R, tau_hi - c_K)
            s_lo = jnp.fmin(c_R, tau_lo - c_K)
            case1_hi = (c_R <= tau_hi - c_K)
            case1_lo = (c_R <= tau_lo - c_K)
            z_s_hi = (s_hi - mu_R) * inv_sR
            z_s_lo = (s_lo - mu_R) * inv_sR

            bv_s_hi, d_bvShi_z1, d_bvShi_z2, d_bvShi_rho = cdfs[2][:4]
            bv_s_lo, d_bvSlo_z1, d_bvSlo_z2, d_bvSlo_rho = cdfs[3][:4]

            p_R_raw = p_band - (bv_cR_hi - bv_cR_lo)

            Phi_zcR = _phi_from_q(z_cR, cdfs[0][4])
            Phi_zcKF = _phi_from_q(z_cK_F, e_ckf[0])
            Phi_zs_hi = _phi_from_q(z_s_hi, cdfs[2][4])
            Phi_zs_lo = _phi_from_q(z_s_lo, cdfs[3][4])

            phi_zcR = cdfs[0][5]
            phi_zcKF = e_ckf[1]
            phi_zs_hi = cdfs[2][5]
            phi_zs_lo = cdfs[3][5]
            if has_corr:
                (P_hi, A1h, A2h, Ach), (P_lo, A1l, A2l, Acl) = \
                    _lower_quadrants(jnp.where(case1_hi, z_cR, z_s_hi), jnp.where(case1_lo, z_cR, z_s_lo), z_cK_F, corr)
            else:
                P_c, A1c, A2c, Acc = Phi_zcR * Phi_zcKF, phi_zcR * Phi_zcKF, Phi_zcR * phi_zcKF, phi_zcR * phi_zcKF
                P_hi = jnp.where(case1_hi, P_c, Phi_zs_hi * Phi_zcKF)
                A1h = jnp.where(case1_hi, A1c, phi_zs_hi * Phi_zcKF)
                A2h = jnp.where(case1_hi, A2c, Phi_zs_hi * phi_zcKF)
                Ach = jnp.where(case1_hi, Acc, phi_zs_hi * phi_zcKF)
                P_lo = jnp.where(case1_lo, P_c, Phi_zs_lo * Phi_zcKF)
                A1l = jnp.where(case1_lo, A1c, phi_zs_lo * Phi_zcKF)
                A2l = jnp.where(case1_lo, A2c, Phi_zs_lo * phi_zcKF)
                Acl = jnp.where(case1_lo, Acc, phi_zs_lo * phi_zcKF)

            G_hi = jnp.where(case1_hi, P_hi, P_hi + bv_cR_hi - bv_s_hi)
            G_lo = jnp.where(case1_lo, P_lo, P_lo + bv_cR_lo - bv_s_lo)

            p_G_raw = G_hi - G_lo
            p_K_raw = p_band - p_R_raw - p_G_raw

            p_sel = jnp.where(rk == 1, p_R_raw, jnp.where(rk == 2, p_K_raw, p_G_raw))
            tail = ~(p_sel >= _TAIL_SWITCH)
            t_lp, t_d = _cdp_tail(tail, rk, mu_R, sigma_R, mu_F, sigma_F, c_R, c_K,
                                  tau_lo,
                                  jnp.where(has_tau_hi, tau_hi, jnp.inf), corr)
            p_cell = jnp.where(tail, jnp.exp(t_lp), p_sel)
            p_rk = p_cell
            prob = p_rk
            val = jnp.log(jnp.clip(prob, _pfloor()))
            inv_prob = jnp.where(prob > _pfloor(), 1.0 / jnp.clip(prob, _pfloor()), 0.0)
            closed = ~tail

            dlp_dpR = jnp.where(closed & (rk == 1), inv_prob, 0.0)
            dlp_dpK = jnp.where(closed & (rk == 2), inv_prob, 0.0)
            dlp_dpG = jnp.where(closed & (rk == 3), inv_prob, 0.0)

            eff_dpR = dlp_dpR - dlp_dpK
            eff_dpG = dlp_dpG - dlp_dpK
            eff_dpBand = dlp_dpK


            dGhi_dtau = jnp.where(case1_hi, 0.0,
                A1h * inv_sR + d_bvCRhi_z2 * inv_sM - d_bvShi_z1 * inv_sR - d_bvShi_z2 * inv_sM)
            dGhi_dcR = jnp.where(case1_hi, A1h * inv_sR, d_bvCRhi_z1 * inv_sR)
            dGhi_dcK = jnp.where(case1_hi, A2h * inv_sF,
                A1h * (-inv_sR) + A2h * inv_sF + d_bvShi_z1 * inv_sR)
            dGhi_dmuR = jnp.where(case1_hi, -A1h * inv_sR,
                A1h * (-inv_sR)
                + d_bvCRhi_z1 * (-inv_sR) + d_bvCRhi_z2 * (-inv_sM)
                - d_bvShi_z1 * (-inv_sR) - d_bvShi_z2 * (-inv_sM))
            dGhi_dmuF = jnp.where(case1_hi, -A2h * inv_sF,
                A2h * (-inv_sF) + d_bvCRhi_z2 * (-inv_sM) - d_bvShi_z2 * (-inv_sM))
            dGhi_dsigR = jnp.where(case1_hi, A1h * (-z_cR * inv_sR),
                A1h * (-z_s_hi * inv_sR)
                + d_bvCRhi_z1 * (-z_cR * inv_sR)
                + d_bvCRhi_z2 * (-z_hi * inv_sM) * sR_over_sM + d_bvCRhi_rho * drho_dsR
                - d_bvShi_z1 * (-z_s_hi * inv_sR)
                - d_bvShi_z2 * (-z_hi * inv_sM) * sR_over_sM - d_bvShi_rho * drho_dsR)
            dGhi_dsigF = jnp.where(case1_hi, A2h * (-z_cK_F * inv_sF),
                A2h * (-z_cK_F * inv_sF)
                + d_bvCRhi_z2 * (-z_hi * inv_sM) * sF_over_sM + d_bvCRhi_rho * drho_dsF
                - d_bvShi_z2 * (-z_hi * inv_sM) * sF_over_sM - d_bvShi_rho * drho_dsF)
            dGhi_dcorr = jnp.where(case1_hi, Ach,
                Ach + (d_bvCRhi_z2 - d_bvShi_z2) * (-z_hi * inv_sM) * dsM_dc
                + (d_bvCRhi_rho - d_bvShi_rho) * drho_dc)

            dGlo_dtau = jnp.where(case1_lo, 0.0,
                A1l * inv_sR + d_bvCRlo_z2 * inv_sM - d_bvSlo_z1 * inv_sR - d_bvSlo_z2 * inv_sM)
            dGlo_dcR = jnp.where(case1_lo, A1l * inv_sR, d_bvCRlo_z1 * inv_sR)
            dGlo_dcK = jnp.where(case1_lo, A2l * inv_sF,
                A1l * (-inv_sR) + A2l * inv_sF + d_bvSlo_z1 * inv_sR)
            dGlo_dmuR = jnp.where(case1_lo, -A1l * inv_sR,
                A1l * (-inv_sR)
                + d_bvCRlo_z1 * (-inv_sR) + d_bvCRlo_z2 * (-inv_sM)
                - d_bvSlo_z1 * (-inv_sR) - d_bvSlo_z2 * (-inv_sM))
            dGlo_dmuF = jnp.where(case1_lo, -A2l * inv_sF,
                A2l * (-inv_sF) + d_bvCRlo_z2 * (-inv_sM) - d_bvSlo_z2 * (-inv_sM))
            dGlo_dsigR = jnp.where(case1_lo, A1l * (-z_cR * inv_sR),
                A1l * (-z_s_lo * inv_sR)
                + d_bvCRlo_z1 * (-z_cR * inv_sR)
                + d_bvCRlo_z2 * (-z_lo * inv_sM) * sR_over_sM + d_bvCRlo_rho * drho_dsR
                - d_bvSlo_z1 * (-z_s_lo * inv_sR)
                - d_bvSlo_z2 * (-z_lo * inv_sM) * sR_over_sM - d_bvSlo_rho * drho_dsR)
            dGlo_dsigF = jnp.where(case1_lo, A2l * (-z_cK_F * inv_sF),
                A2l * (-z_cK_F * inv_sF)
                + d_bvCRlo_z2 * (-z_lo * inv_sM) * sF_over_sM + d_bvCRlo_rho * drho_dsF
                - d_bvSlo_z2 * (-z_lo * inv_sM) * sF_over_sM - d_bvSlo_rho * drho_dsF)
            dGlo_dcorr = jnp.where(case1_lo, Acl,
                Acl + (d_bvCRlo_z2 - d_bvSlo_z2) * (-z_lo * inv_sM) * dsM_dc
                + (d_bvCRlo_rho - d_bvSlo_rho) * drho_dc)

            dlp_dzcR = eff_dpR * (-d_bvCRhi_z1 + d_bvCRlo_z1)
            dlp_dzhi = eff_dpR * (phi_hi - d_bvCRhi_z2) + eff_dpBand * phi_hi
            dlp_dzlo = eff_dpR * (-(phi_lo - d_bvCRlo_z2)) + eff_dpBand * (-phi_lo)
            dlp_drho = eff_dpR * (-d_bvCRhi_rho + d_bvCRlo_rho)

            dG_dmuR = dGhi_dmuR - dGlo_dmuR
            dG_dsigR = dGhi_dsigR - dGlo_dsigR
            dG_dmuF = dGhi_dmuF - dGlo_dmuF
            dG_dsigF = dGhi_dsigF - dGlo_dsigF
            dG_dcR = dGhi_dcR - dGlo_dcR
            dG_dcK = dGhi_dcK - dGlo_dcK
            dG_dcorr = dGhi_dcorr - dGlo_dcorr

            d_val_dmuR = (dlp_dzcR * (-inv_sR)
                         + (dlp_dzhi + dlp_dzlo) * (-inv_sM)
                         + eff_dpG * dG_dmuR)
            d_val_dmuF = ((dlp_dzhi + dlp_dzlo) * (-inv_sM)
                         + eff_dpG * dG_dmuF)
            d_val_dsigR = (dlp_dzcR * (-z_cR * inv_sR)
                          + (dlp_dzhi * (-z_hi) + dlp_dzlo * (-z_lo)) * inv_sM * sR_over_sM
                          + dlp_drho * drho_dsR + eff_dpG * dG_dsigR)
            d_val_dsigF = ((dlp_dzhi * (-z_hi) + dlp_dzlo * (-z_lo)) * inv_sM * sF_over_sM
                          + dlp_drho * drho_dsF + eff_dpG * dG_dsigF)
            d_val_dcR = dlp_dzcR * inv_sR + eff_dpG * dG_dcR
            d_val_dcK = eff_dpG * dG_dcK
            d_val_dcorr = ((dlp_dzhi * (-z_hi) + dlp_dzlo * (-z_lo)) * inv_sM * dsM_dc
                           + dlp_drho * drho_dc + eff_dpG * dG_dcorr)

            d_val_dtau_hi = dlp_dzhi * inv_sM + eff_dpG * dGhi_dtau
            d_val_dtau_lo = dlp_dzlo * inv_sM + eff_dpG * (-dGlo_dtau)

            t_w = jnp.where(tail & (prob > _pfloor()), p_cell * inv_prob, 0.0)
            td = [t_w * jnp.where(tail, t_d[..., j], 0.0) for j in range(9)]
            d_val_dmuR = d_val_dmuR + td[0]
            d_val_dsigR = d_val_dsigR + td[1]
            d_val_dmuF = d_val_dmuF + td[2]
            d_val_dsigF = d_val_dsigF + td[3]
            d_val_dcR = d_val_dcR + td[4]
            d_val_dcK = d_val_dcK + td[5]
            d_val_dtau_lo = d_val_dtau_lo + td[6]
            d_val_dtau_hi = d_val_dtau_hi + td[7]
            d_val_dcorr = d_val_dcorr + td[8]

            live = prob > _pfloor()
            (d_val_dmuR, d_val_dsigR, d_val_dmuF, d_val_dsigF, d_val_dcR, d_val_dcK, d_val_dcorr, d_val_dtau_lo,
             d_val_dtau_hi) = (jnp.where(live, d, 0.0) for d in (d_val_dmuR, d_val_dsigR, d_val_dmuF, d_val_dsigF,
                                                              d_val_dcR, d_val_dcK, d_val_dcorr, d_val_dtau_lo,
                                                              d_val_dtau_hi))
            empty = (rk == 3) & (c_R + c_K <= tau_lo)
            val = jnp.where(empty, _empty_cell_lp(tau_lo - c_R - c_K), val)
            d_val_dmuR = jnp.where(empty, 0.0, d_val_dmuR)
            d_val_dsigR = jnp.where(empty, 0.0, d_val_dsigR)
            d_val_dmuF = jnp.where(empty, 0.0, d_val_dmuF)
            d_val_dsigF = jnp.where(empty, 0.0, d_val_dsigF)
            d_val_dcR = jnp.where(empty, _EMPTY_SLOPE, d_val_dcR)
            d_val_dcK = jnp.where(empty, _EMPTY_SLOPE, d_val_dcK)
            d_val_dcorr = jnp.where(empty, 0.0, d_val_dcorr)
            d_val_dtau_lo = jnp.where(empty, -_EMPTY_SLOPE, d_val_dtau_lo)
            d_val_dtau_hi = jnp.where(empty, 0.0, d_val_dtau_hi)

        d_t_k_lo = d_val_dtau_lo
        d_t_k_hi = jnp.where(has_tau_hi, d_val_dtau_hi, 0.0)
        narrow, inner, kind, lo, gs, k_own, g1 = _narrow(rk, has_tau_hi, old_idx, gv, gaps_v, thresh_mid_v, tau_lo,
                                                         mu_R, sigma_R, mu_F, sigma_F, c_R, c_K, corr)
        if n_rkg == 3:
            narrow = narrow & ~empty
        base = (val, d_val_dmuR, d_val_dsigR, d_val_dmuF, d_val_dsigF, d_val_dcR, d_val_dcK, d_val_dcorr,
                d_t_k_lo + d_t_k_hi, jnp.where(inner, d_t_k_hi, 0.0))

        def strip(o):
            P, d = _cdp_strip_split(kind, lo, gs, mu_R, sigma_R, mu_F, sigma_F, c_R, c_K, corr, g1)
            Pc = jnp.clip(P, _pfloor())
            ip = jnp.where(P > _pfloor(), 1.0 / Pc, 0.0)
            return tuple(jnp.where(narrow, a, b) for a, b in zip((jnp.log(Pc),) + tuple(x * ip for x in d), o))
        (val, d_val_dmuR, d_val_dsigR, d_val_dmuF, d_val_dsigF, d_val_dcR, d_val_dcK, d_val_dcorr,
         gA_tau, gB_tau) = jax.lax.cond(jnp.any(narrow), strip, lambda o: o, base)

        d_lp_d_thresh_mid = gA_tau
        k_anchor = k_tau_lo
        d_lp_d_gaps = _lw_gap_grads(gA_tau, gB_tau, k_anchor, jnp.asarray(own_tau_mask)[:, k_own], inner,
                                    gap_mask_np, gd)

        tangent_out = (
            d_val_dmuR * d_mu_R +
            d_val_dmuF * d_mu_F +
            d_val_dsigR * d_sigma_R +
            d_val_dsigF * d_sigma_F +
            d_val_dcR * d_c_R +
            d_val_dcK * d_c_K +
            d_val_dcorr * d_corr +
            d_lp_d_thresh_mid * d_thresh_mid_v +
            jnp.sum(d_lp_d_gaps * d_gaps_v, axis=1)
        )

        return val, tangent_out

    return _batch_fused


_fused_cache = {}


# --- Vectorized fused wrappers ---

def fused_loglik_vrdp2d(y1, y2, item_type, K1, K2, dprime_F, dprime_R, source_d,
                         lambda_val, sigma_item, sigma_S,
                         thresh1_mid, gaps1, thresh2_mid, gaps2, gap_link="log"):
    """Fully-vectorized fused log-likelihood for VRDP2D."""
    N = y1.shape[0]
    n_gaps1 = K1 - 2
    n_gaps2 = K2 - 2
    is_softplus = (gap_link == "softplus")

    if gaps1.ndim == 1:
        gaps1_per_obs = jnp.broadcast_to(gaps1[None, :], (N, n_gaps1))
    else:
        gaps1_per_obs = gaps1.T
    if gaps2.ndim == 1:
        gaps2_per_obs = jnp.broadcast_to(gaps2[None, :], (N, n_gaps2))
    else:
        gaps2_per_obs = gaps2.T

    key = ("batch_vrdp2d", int(K1), int(K2), bool(is_softplus))
    if key not in _fused_cache:
        _fused_cache[key] = _make_fused_batch_vrdp2d_loglik(
            int(K1), int(K2), bool(is_softplus))
    return _fused_cache[key](y1, y2, item_type, dprime_F, dprime_R, source_d,
                              lambda_val, sigma_item, sigma_S,
                              thresh1_mid, gaps1_per_obs, thresh2_mid, gaps2_per_obs)


def fused_loglik_bivariate_sdt(y1, y2, item_type, K1, K2, mu1, mu2, sigma1, sigma2, rho_val,
                                thresh1_mid, gaps1, thresh2_mid, gaps2, gap_link="log"):
    """Fully-vectorized fused log-likelihood for bivariate_sdt."""
    N = y1.shape[0]
    n_gaps1 = K1 - 2
    n_gaps2 = K2 - 2
    is_softplus = (gap_link == "softplus")

    if gaps1.ndim == 1:
        gaps1_per_obs = jnp.broadcast_to(gaps1[None, :], (N, n_gaps1))
    else:
        gaps1_per_obs = gaps1.T
    if gaps2.ndim == 1:
        gaps2_per_obs = jnp.broadcast_to(gaps2[None, :], (N, n_gaps2))
    else:
        gaps2_per_obs = gaps2.T

    key = ("batch_bivariate_sdt", int(K1), int(K2), bool(is_softplus))
    if key not in _fused_cache:
        _fused_cache[key] = _make_fused_batch_bivariate_sdt_loglik(
            int(K1), int(K2), bool(is_softplus))
    return _fused_cache[key](y1, y2, item_type, mu1, mu2, sigma1, sigma2, rho_val,
                              thresh1_mid, gaps1_per_obs, thresh2_mid, gaps2_per_obs)


def fused_loglik_bounded_bivariate_sdt(y1, y2, item_type, K1, K2, mu1, mu2, sigma1, sigma2, rho_val,
                                        thresh1_mid, gaps1, thresh2_mid, gaps2, gap_link="log"):
    """Fully-vectorized fused log-likelihood for bounded bivariate_sdt."""
    N = y1.shape[0]
    n_gaps1 = K1 - 2
    n_gaps2 = K2 - 2
    is_softplus = (gap_link == "softplus")

    if gaps1.ndim == 1:
        gaps1_per_obs = jnp.broadcast_to(gaps1[None, :], (N, n_gaps1))
    else:
        gaps1_per_obs = gaps1.T
    if gaps2.ndim == 1:
        gaps2_per_obs = jnp.broadcast_to(gaps2[None, :], (N, n_gaps2))
    else:
        gaps2_per_obs = gaps2.T

    key = ("batch_bounded_bivariate_sdt", int(K1), int(K2), bool(is_softplus))
    if key not in _fused_cache:
        _fused_cache[key] = _make_fused_batch_bounded_bivariate_sdt_loglik(
            int(K1), int(K2), bool(is_softplus))
    return _fused_cache[key](y1, y2, item_type, mu1, mu2, sigma1, sigma2, rho_val,
                              thresh1_mid, gaps1_per_obs, thresh2_mid, gaps2_per_obs)


def _make_fused_batch_bounded_bivariate_dp_loglik(K1, K2, gap_link_is_softplus):
    """Fully-vectorized batch loglik for BOUNDED bivariate_dp.

    P = (1-R_I) * bounded_biv_prob + R_I*(1-R_S) * bounded_marg_source + R_I*R_S * corner
    bounded_biv_prob: Phi-product below the crossing point cp + bivariate rectangle above it.
    bounded_marg_source: Phi(z_cp) * indep_source + P(Z1 > z_cp, z2_lo < Z2 < z2_hi).
    Each rating cell enters by its lower edge and width wherever the crossing point does not split it.
    """
    n_thresh1 = K1 - 1
    n_gaps1 = n_thresh1 - 1
    mid_idx1 = (n_thresh1 - 1) // 2
    n_upper1 = n_thresh1 - mid_idx1 - 1
    n_thresh2 = K2 - 1
    n_gaps2 = n_thresh2 - 1
    mid_idx2 = (n_thresh2 - 1) // 2
    n_upper2 = n_thresh2 - mid_idx2 - 1
    gap_mask1_np = _make_batch_gap_mask_np(n_gaps1, n_thresh1, mid_idx1, n_upper1)
    gap_mask2_np = _make_batch_gap_mask_np(n_gaps2, n_thresh2, mid_idx2, n_upper2)
    own_idx1, own_mask1 = _own_gap(K1)
    own_idx2, own_mask2 = _own_gap(K2)
    INF = _OPEN

    def _core(y1, y2, item_type, mu1, mu2, sigma1, sigma2, rho_val,
              R_I, R_S, R_I_B, R_S_B, thresh1_mid, gv1, thresh2_mid, gv2, midx, gaps1):
        c1 = _batch_build_thresh(thresh1_mid, gv1, n_gaps1, mid_idx1, n_upper1)
        c2 = _batch_build_thresh(thresh2_mid, gv2, n_gaps2, mid_idx2, n_upper2)
        N = y1.shape[0]
        obs_idx = jnp.arange(N)
        e1 = _bivariate_edges(y1, K1, c1, gv1, own_idx1, mu1, sigma1)
        e2 = _bivariate_edges(y2, K2, c2, gv2, own_idx2, mu2, sigma2)
        z1_lo, z1_hi, w1, _, _, in1, has1_lo, has1_hi = e1
        z2_lo, z2_hi, w2, _, _, in2, has2_lo, has2_hi = e2
        k2_hi = jnp.clip(y2 - 1, 0, n_thresh2 - 1)
        k2_lo = jnp.clip(y2 - 2, 0, n_thresh2 - 1)
        sqrt_1mrr = jnp.sqrt(jnp.clip(1.0 - rho_val**2, 1e-20))
        sc_floor = sigma2 * sqrt_1mrr <= sigma2 * 1e-6
        sigma_cond = jnp.maximum(sigma2 * sqrt_1mrr, sigma2 * 1e-6)
        inv_sc = 1.0 / sigma_cond
        rho_sigma2 = rho_val * sigma2 + 1e-30
        is_new = item_type == 1
        z_cp = jnp.where(is_new, -1e6, -mu2 / rho_sigma2)
        z_cp_eff = jnp.clip(z_cp, z1_lo, z1_hi)
        split = (z_cp > z1_lo) & (z_cp < z1_hi)
        above_only = z_cp <= z1_lo
        below_only = z_cp >= z1_hi
        zs_hi = jnp.where(has2_hi, c2[obs_idx, k2_hi] * inv_sc, INF)
        zs_lo = jnp.where(has2_lo, c2[obs_idx, k2_lo] * inv_sc, -INF)
        ws = w2 * sigma2 * inv_sc

        det_hi = z_cp_eff < 0.5 * INF
        w_cut = _cp_offset(thresh1_mid, gaps1, gap_link_is_softplus, gap_mask1_np, y1, n_thresh1, mu1, sigma1, mu2,
                           sigma2, rho_val)
        w_det = jnp.where(below_only, w1, jnp.where(above_only, 1.0, jnp.where(has1_lo, w_cut, z_cp_eff - z1_lo)))
        det = _cell_lw("probit", z1_lo, z_cp_eff, jnp.where(has1_lo & det_hi, w_det, 1.0), has1_lo, det_hi)
        src = _cell_lw("probit", zs_lo, zs_hi, ws, has2_lo, has2_hi)
        det_on = (det[0] > 0.0) & ~above_only
        src_on = src[0] > 0.0
        p_det = jnp.where(det_on, det[0], 0.0)
        p_src = jnp.where(src_on, src[0], 0.0)
        rect = _rect_lw(z_cp_eff, z1_hi, jnp.where(above_only, w1, jnp.where(in1, w1 - w_cut, z1_hi - z_cp_eff)),
                        z2_lo, z2_hi, w2, rho_val)
        above_on = (rect[0] > 0.0) & ~below_only
        biv_raw = p_det * p_src + jnp.where(above_on, rect[0], 0.0)
        biv_on = biv_raw > _pfloor()
        bounded_biv = jnp.where(biv_on, biv_raw, _pfloor())

        mg = jnp.minimum(midx, N - 1)
        z_cp_g = z_cp[mg]
        Pcp_g, _, dPcp_g = _phi_diff(jnp.full_like(z_cp_g, -INF), z_cp_g)
        marg_g = _rect_lw(z_cp_g, jnp.full_like(z_cp_g, INF), jnp.ones_like(z_cp_g), z2_lo[mg], z2_hi[mg], w2[mg],
                          rho_val[mg])
        put = lambda v: jnp.zeros(N, dtype=v.dtype).at[midx].set(v, mode="drop")
        Pcp, dPcp = put(Pcp_g), put(dPcp_g)
        pm, Am1, _, Am2, Bm2, dmr = (put(v) for v in marg_g)
        marg_raw = Pcp * src[0] + pm
        marg_on = marg_raw > _pfloor()
        bounded_marg = jnp.where(marg_on, marg_raw, _pfloor())

        is_A = item_type == 2
        R_I_eff = jnp.where(is_A, R_I, R_I_B)
        R_S_eff = jnp.where(is_A, R_S, R_S_B)
        is_y1_top = (y1 == K1) & ~is_new
        is_corner = (is_A & (y1 == K1) & (y2 == 1)) | ((item_type == 3) & (y1 == K1) & (y2 == K2))
        wt1 = 1.0 - R_I_eff
        wt2 = jnp.where(is_y1_top, R_I_eff * (1.0 - R_S_eff), 0.0)
        prob_raw = wt1 * bounded_biv + wt2 * bounded_marg + jnp.where(is_corner, R_I_eff * R_S_eff, 0.0)
        prob = jnp.clip(prob_raw, _pfloor())
        parts = dict(e1=e1, e2=e2, det=det, src=src, rect=rect, det_on=det_on, src_on=src_on, above_on=above_on,
                     p_det=p_det, p_src=p_src, biv_on=biv_on, marg_on=marg_on, Pcp=Pcp, dPcp=dPcp, Am1=Am1,
                     Am2=Am2, Bm2=Bm2, dmr=dmr, z_cp=z_cp, z_cp_eff=z_cp_eff, split=split, above_only=above_only,
                     below_only=below_only, det_hi=det_hi, zs_lo=zs_lo, zs_hi=zs_hi, ws=ws, inv_sc=inv_sc,
                     sc_floor=sc_floor, sqrt_1mrr=sqrt_1mrr, rho_sigma2=rho_sigma2, is_new=is_new,
                     R_I_eff=R_I_eff, R_S_eff=R_S_eff, is_A=is_A, is_y1_top=is_y1_top, is_corner=is_corner,
                     w1=wt1, w2=wt2, bounded_biv=bounded_biv, bounded_marg=bounded_marg, prob_raw=prob_raw)
        return jnp.log(prob), prob, parts

    @custom_jvp
    def _batch_fused(y1, y2, item_type, mu1, mu2, sigma1, sigma2, rho_val,
                     R_I, R_S, R_I_B, R_S_B, thresh1_mid, gaps1, thresh2_mid, gaps2, midx):
        gap_fn = jax.nn.softplus if gap_link_is_softplus else jnp.exp
        return _core(y1, y2, item_type, mu1, mu2, sigma1, sigma2, rho_val, R_I, R_S, R_I_B, R_S_B,
                     thresh1_mid, gap_fn(gaps1), thresh2_mid, gap_fn(gaps2), midx, gaps1)[0]

    @_batch_fused.defjvp
    def _batch_fused_jvp(primals, tangents):
        (y1, y2, item_type, mu1, mu2, sigma1, sigma2, rho_val,
         R_I, R_S, R_I_B, R_S_B, thresh1_mid, gaps1, thresh2_mid, gaps2, midx) = primals
        (_, _, _, d_mu1, d_mu2, d_sigma1, d_sigma2, d_rho,
         d_R_I, d_R_S, d_R_I_B, d_R_S_B, d_thresh1_mid, d_gaps1, d_thresh2_mid, d_gaps2, _) = tangents
        gv1, gd1 = _gap_vals_and_derivs(gaps1, gap_link_is_softplus)
        gv2, gd2 = _gap_vals_and_derivs(gaps2, gap_link_is_softplus)
        val, prob, q = _core(y1, y2, item_type, mu1, mu2, sigma1, sigma2, rho_val, R_I, R_S, R_I_B, R_S_B,
                             thresh1_mid, gv1, thresh2_mid, gv2, midx, gaps1)
        inv_P = jnp.where(q["prob_raw"] > _pfloor(), 1.0 / prob, 0.0)
        z1_lo, z1_hi, w1, z1_a, k1_a, in1, has1_lo, has1_hi = q["e1"]
        z2_lo, z2_hi, w2, z2_a, k2_a, in2, has2_lo, has2_hi = q["e2"]

        def f(m):
            return m.astype(prob.dtype)

        gb = q["w1"] * f(q["biv_on"]) * inv_P
        gm = q["w2"] * f(q["marg_on"]) * inv_P
        det_g = gb * f(q["det_on"]) * q["p_src"]
        ab = gb * f(q["above_on"])
        det, src, rect = q["det"], q["src"], q["rect"]
        Ad, Bd = (jnp.where(det_g != 0.0, det[j], 0.0) * det_g for j in (1, 2))
        Ar1, Br1, Ar2, Br2 = (jnp.where(ab != 0.0, rect[j], 0.0) * ab for j in (1, 2, 3, 4))
        z_cp_eff = q["z_cp_eff"]
        dd_lo, dd_hi = _band_edge_grads(Ad, Bd, has1_lo & q["det_hi"], has1_lo)
        dr_lo, dr_hi = _band_edge_grads(Ar1, Br1, (z_cp_eff > -0.5 * _OPEN) & has1_hi, z_cp_eff > -0.5 * _OPEN)
        split, below_only = q["split"], q["below_only"]
        in_ab = (z_cp_eff > -0.5 * _OPEN) & has1_hi
        cont = split & has1_lo & in_ab & ~q["is_new"] & ~q["sc_floor"]
        gA1 = jnp.where(cont, Ad + Ar1, jnp.where(split, dd_lo + dr_hi, jnp.where(below_only, Ad, Ar1)))
        gB1 = jnp.where(split, dr_hi, jnp.where(below_only, Bd, Br1))
        g_zcp = jnp.where(cont, 0.0, jnp.where(split, dd_hi + dr_lo, 0.0)) + gm * (q["dPcp"] * src[0] + q["Am1"])
        g_zcp = jnp.where(q["is_new"], 0.0, g_zcp)

        wsrc = gb * f(q["src_on"]) * q["p_det"] + gm * q["Pcp"]
        As, Bs = src[1] * wsrc, jnp.where(in2, src[2] * wsrc, 0.0)
        gA2 = Ar2 + gm * q["Am2"]
        gB2 = Br2 + gm * q["Bm2"]
        g_rho_direct = jnp.where(ab != 0.0, rect[5], 0.0) * ab + gm * q["dmr"]

        inv_s1, inv_s2, inv_sc = 1.0 / sigma1, 1.0 / sigma2, q["inv_sc"]
        rs2 = q["rho_sigma2"]
        g1 = _lw_chain(gA1, gB1, z1_a, w1, in1, inv_s1, z1_hi)
        g2 = _lw_chain(gA2, gB2, z2_a, w2, in2, inv_s2, z2_hi)
        zs_a = jnp.where(has2_lo, q["zs_lo"], q["zs_hi"])
        zs_term = As * zs_a + Bs * jnp.where(in2, q["ws"], 0.0)
        not_floor = f(~q["sc_floor"] & (1.0 - rho_val**2 > 1e-20))
        d_mu2_c = g2[0] - g_zcp / rs2
        d_s2_c = g2[1] - zs_term * inv_s2 + g_zcp * mu2 * rho_val / (rs2 * rs2)
        d_rho_c = g_rho_direct + g_zcp * mu2 * sigma2 / (rs2 * rs2) + zs_term * not_floor * rho_val / (q["sqrt_1mrr"] ** 2)
        d_gaps1_c = _lw_gap_grads(gA1 * inv_s1, gB1 * inv_s1, k1_a, jnp.asarray(own_mask1)[:, y1 - 1], in1,
                                  gap_mask1_np, gd1)
        d_gaps2_c = _lw_gap_grads(gA2 * inv_s2 + As * inv_sc, gB2 * inv_s2 + Bs * inv_sc, k2_a,
                                  jnp.asarray(own_mask2)[:, y2 - 1], in2, gap_mask2_np, gd2)

        R_I_eff = q["R_I_eff"]
        R_S_eff = q["R_S_eff"]
        top = f(q["is_y1_top"])
        corner = f(q["is_corner"])
        d_RI_c = (-q["bounded_biv"] + top * (1.0 - R_S_eff) * q["bounded_marg"] + corner * R_S_eff) * inv_P
        d_RS_c = (-top * R_I_eff * q["bounded_marg"] + corner * R_I_eff) * inv_P
        d_RI_in = jnp.where(q["is_A"], d_R_I, d_R_I_B)
        d_RS_in = jnp.where(q["is_A"], d_R_S, d_R_S_B)

        tangent_out = (g1[0] * d_mu1 + d_mu2_c * d_mu2 + g1[1] * d_sigma1 + d_s2_c * d_sigma2
                       + d_rho_c * d_rho + d_RI_c * d_RI_in + d_RS_c * d_RS_in
                       + g1[2] * d_thresh1_mid + jnp.sum(d_gaps1_c * d_gaps1, axis=1)
                       + (g2[2] + As * inv_sc) * d_thresh2_mid + jnp.sum(d_gaps2_c * d_gaps2, axis=1))
        return val, tangent_out

    return _batch_fused


def fused_loglik_bounded_bivariate_dp(y1, y2, item_type, K1, K2, mu1, mu2, sigma1, sigma2,
                                       rho_val, R_I, R_S,
                                       R_I_B=None, R_S_B=None,
                                       thresh1_mid=None, gaps1=None,
                                       thresh2_mid=None, gaps2=None, gap_link="log", part=None):
    """Fully-vectorized fused log-likelihood for bounded bivariate_dp; `part` is the padded index of the
    (old, y1 == K1) rows whose marginal rectangle enters the likelihood, or None for all rows."""
    if R_I_B is None:
        R_I_B = R_I
    if R_S_B is None:
        R_S_B = R_S
    N = y1.shape[0]
    n_gaps1 = K1 - 2
    n_gaps2 = K2 - 2
    is_softplus = (gap_link == "softplus")

    if gaps1.ndim == 1:
        gaps1_per_obs = jnp.broadcast_to(gaps1[None, :], (N, n_gaps1))
    else:
        gaps1_per_obs = gaps1.T
    if gaps2.ndim == 1:
        gaps2_per_obs = jnp.broadcast_to(gaps2[None, :], (N, n_gaps2))
    else:
        gaps2_per_obs = gaps2.T

    key = ("batch_bounded_bivariate_dp", int(K1), int(K2), bool(is_softplus))
    if key not in _fused_cache:
        _fused_cache[key] = _make_fused_batch_bounded_bivariate_dp_loglik(
            int(K1), int(K2), bool(is_softplus))
    midx = jnp.arange(N, dtype=jnp.int32) if part is None else jnp.asarray(part, dtype=jnp.int32)
    return _fused_cache[key](y1, y2, item_type, mu1, mu2, sigma1, sigma2,
                              rho_val, R_I, R_S, R_I_B, R_S_B,
                              thresh1_mid, gaps1_per_obs, thresh2_mid, gaps2_per_obs, midx)


def fused_loglik_cdp(y, rk, is_old, old_idx, K, J,
                      mu_R, sigma_R, mu_F, sigma_F, c_R, c_K,
                      thresh_mid, gaps, gap_link="log", n_rkg=2, corr=None):
    """Fully-vectorized fused log-likelihood for CDP, over judged "old" responses (rk >= 1)."""
    N = y.shape[0]
    n_gaps = J - 1
    is_softplus = (gap_link == "softplus")

    if gaps.ndim == 1:
        gaps_per_obs = jnp.broadcast_to(gaps[None, :], (N, n_gaps))
    else:
        gaps_per_obs = gaps.T

    has_corr = corr is not None
    key = ("batch_cdp", int(K), int(J), bool(is_softplus), int(n_rkg), has_corr)
    if key not in _fused_cache:
        _fused_cache[key] = _make_fused_batch_cdp_loglik(
            int(K), int(J), bool(is_softplus), int(n_rkg), has_corr)
    return _fused_cache[key](y, rk, is_old, old_idx,
                              mu_R, sigma_R, mu_F, sigma_F, c_R, c_K,
                              jnp.zeros_like(mu_R) if corr is None else corr,
                              thresh_mid, gaps_per_obs)


def fused_loglik_simple(y, K, mu, sigma_val, thresh_mid, gaps, gap_link="log", link_name="probit"):
    """Fully-vectorized fused log-likelihood for evsdt/uvsdt.

    Args:
        y: [N] int responses (1-indexed)
        K: int
        mu: [N] signal means (is_old * dprime)
        sigma_val: [N] signal SDs
        thresh_mid: [N] effective mid-anchor (pop + crit RE)
        gaps: [n_gaps, N] effective log-gaps (pop + crit RE), or [n_gaps] if shared
        gap_link: "log" or "softplus"
        link_name: latent-distribution link ("probit", "logit", or "cloglog")
    """
    N = y.shape[0]
    n_gaps = K - 2
    is_softplus = (gap_link == "softplus")

    if gaps.ndim == 1:
        gaps_per_obs = jnp.broadcast_to(gaps[None, :], (N, n_gaps))
    else:
        gaps_per_obs = gaps.T

    key = ("batch_simple", int(K), bool(is_softplus), str(link_name))
    if key not in _fused_cache:
        _fused_cache[key] = _make_fused_batch_loglik(int(K), bool(is_softplus), str(link_name))
    return _fused_cache[key](y, mu, sigma_val, thresh_mid, gaps_per_obs)


def fused_loglik_dpsdt(y, K, mu, sigma_val, lambda_val, thresh_mid, gaps,
                       gap_link="log", has_sigma=False, link_name="probit", edge=None):
    """Fully-vectorized fused log-likelihood for DPSDT (old items only)."""
    N = y.shape[0]
    n_gaps = K - 2
    is_softplus = (gap_link == "softplus")

    if gaps.ndim == 1:
        gaps_per_obs = jnp.broadcast_to(gaps[None, :], (N, n_gaps))
    else:
        gaps_per_obs = gaps.T

    key = ("batch_dpsdt", int(K), bool(is_softplus), str(link_name)) + (() if edge is None else (int(edge),))
    if key not in _fused_cache:
        _fused_cache[key] = _make_fused_batch_dpsdt_loglik(int(K), bool(is_softplus), str(link_name), edge)
    return _fused_cache[key](y, mu, sigma_val, lambda_val, thresh_mid, gaps_per_obs)


def fused_loglik_mixture(y, K, mu1, sigma1_val, mu2, sigma2_val, lambda_val,
                         thresh_mid, gaps, gap_link="log", link_name="probit"):
    """Fully-vectorized fused log-likelihood for mixture (old items only)."""
    N = y.shape[0]
    n_gaps = K - 2
    is_softplus = (gap_link == "softplus")

    if gaps.ndim == 1:
        gaps_per_obs = jnp.broadcast_to(gaps[None, :], (N, n_gaps))
    else:
        gaps_per_obs = gaps.T

    key = ("batch_mixture", int(K), bool(is_softplus), str(link_name))
    if key not in _fused_cache:
        _fused_cache[key] = _make_fused_batch_mixture_loglik(int(K), bool(is_softplus), str(link_name))
    return _fused_cache[key](y, mu1, sigma1_val, mu2, sigma2_val, lambda_val,
                             thresh_mid, gaps_per_obs)


def fused_loglik_source_mixture(y, K, d_attended, lambda_val, thresh_mid, gaps,
                                gap_link="log", link_name="probit"):
    """Fully-vectorized fused log-likelihood for source_mixture."""
    N = y.shape[0]
    n_gaps = K - 2
    is_softplus = (gap_link == "softplus")

    if gaps.ndim == 1:
        gaps_per_obs = jnp.broadcast_to(gaps[None, :], (N, n_gaps))
    else:
        gaps_per_obs = gaps.T

    key = ("batch_source_mixture", int(K), bool(is_softplus), str(link_name))
    if key not in _fused_cache:
        _fused_cache[key] = _make_fused_batch_source_mixture_loglik(int(K), bool(is_softplus), str(link_name))
    return _fused_cache[key](y, d_attended, lambda_val, thresh_mid, gaps_per_obs)


def _gap_rows(gaps, i):
    return gaps[:, i] if gaps.ndim == 2 else gaps


def _by_item_type(is_old, old_ll, new_ll, N, is_old_static=None):
    if is_old_static is None:
        idx = jnp.arange(N)
        return jnp.where(is_old > 0, old_ll(idx), new_ll(idx))
    old = np.asarray(is_old_static) > 0
    o, n = np.flatnonzero(old), np.flatnonzero(~old)
    if not len(n):
        return old_ll(o)
    if not len(o):
        return new_ll(n)
    ll = jnp.zeros(N, dtype=_FLOAT)
    return ll.at[o].set(old_ll(o)).at[n].set(new_ll(n))


def univariate_loglik(y, dprime, thresh, is_old, link_name, family_name,
                      sigma=None, lambda_val=None, dprime2=None, sigma2=None,
                      lambda_B=None, dprime_B=None, counts=None,
                      dprime_L=None, sigma_L=None, lambda_L=None,
                      thresh_mid=None, log_gaps=None, gap_link="log",
                      **kwargs):
    """Compute log-likelihood for univariate SDT families.

    Uses custom JVP for analytic gradients through Phi cell probabilities.
    When thresh_mid and log_gaps are provided, uses the fused JVP path
    which also analytically differentiates through threshold construction.
    """
    N = y.shape[0]
    K = thresh.shape[1] + 1

    mu_signal = dprime
    sd_signal = sigma if sigma is not None else jnp.ones(N)
    mu_noise = jnp.zeros(N)
    sd_noise = jnp.ones(N)

    # --- Fused JVP path ---
    has_fused_inputs = (thresh_mid is not None and log_gaps is not None)
    is_simple = family_name in ("evsdt", "uvsdt", "cumulative")

    if family_name == "evsdt" and kwargs.get("sigma_meta") is not None and has_fused_inputs:
        fused = fused_loglik_lnoise if kwargs.get("meta_noise") == "lognormal" else fused_loglik_gnoise
        log_lik = fused(y, K, is_old * mu_signal, kwargs["sigma_meta"], thresh_mid, log_gaps, gap_link)
        if counts is not None:
            log_lik = log_lik * counts
        return log_lik

    if is_simple and has_fused_inputs and lambda_val is None:
        mu_per_obs = is_old * mu_signal
        sd_per_obs = jnp.where(is_old > 0, sd_signal, sd_noise)
        log_lik = fused_loglik_simple(y, K, mu_per_obs, sd_per_obs,
                                       thresh_mid, log_gaps, gap_link,
                                       link_name=link_name)
        if counts is not None:
            log_lik = log_lik * counts
        return log_lik

    if family_name == "dpsdt" and lambda_val is not None and has_fused_inputs:
        has_sigma = (sigma is not None)
        mu_rows = is_old * mu_signal

        def old_ll(i):
            return fused_loglik_dpsdt(y[i], K, mu_rows[i], sd_signal[i], lambda_val[i], thresh_mid[i],
                                      _gap_rows(log_gaps, i), gap_link, has_sigma=has_sigma, link_name=link_name)

        if lambda_L is not None:
            def new_ll(i):
                return fused_loglik_dpsdt(y[i], K, mu_rows[i], sd_noise[i], lambda_L[i], thresh_mid[i],
                                          _gap_rows(log_gaps, i), gap_link, has_sigma=False, link_name=link_name,
                                          edge=1)
        else:
            def new_ll(i):
                return fused_loglik_simple(y[i], K, mu_rows[i], sd_noise[i], thresh_mid[i],
                                           _gap_rows(log_gaps, i), gap_link, link_name=link_name)

        log_lik = _by_item_type(is_old, old_ll, new_ll, N, kwargs.get("is_old_static"))
        if counts is not None:
            log_lik = log_lik * counts
        return log_lik

    if family_name == "mixture" and has_fused_inputs and (lambda_val is not None or lambda_L is not None):
        _dprime2 = dprime2 if dprime2 is not None else jnp.zeros(N)
        _sigma2 = sigma2 if sigma2 is not None else jnp.ones(N)
        mu1_rows = is_old * mu_signal
        mu2_rows = _dprime2 - (1.0 - is_old) * mu_signal

        if lambda_val is not None:
            def old_ll(i):
                return fused_loglik_mixture(y[i], K, mu1_rows[i], sd_signal[i], mu2_rows[i], _sigma2[i], lambda_val[i],
                                            thresh_mid[i], _gap_rows(log_gaps, i), gap_link, link_name=link_name)
        else:
            def old_ll(i):
                return fused_loglik_simple(y[i], K, mu1_rows[i], sd_signal[i], thresh_mid[i],
                                           _gap_rows(log_gaps, i), gap_link, link_name=link_name)

        if dprime_L is not None and lambda_L is not None:
            sd_L_val = sigma_L if sigma_L is not None else jnp.ones(N)

            def new_ll(i):
                return fused_loglik_mixture(y[i], K, -dprime_L[i], sd_L_val[i], mu_noise[i], sd_noise[i],
                                            lambda_L[i], thresh_mid[i], _gap_rows(log_gaps, i), gap_link,
                                            link_name=link_name)
        else:
            def new_ll(i):
                return fused_loglik_simple(y[i], K, mu1_rows[i], sd_noise[i], thresh_mid[i],
                                           _gap_rows(log_gaps, i), gap_link, link_name=link_name)

        log_lik = _by_item_type(is_old, old_ll, new_ll, N, kwargs.get("is_old_static"))
        if counts is not None:
            log_lik = log_lik * counts
        return log_lik

    if family_name == "source_mixture" and lambda_val is not None and has_fused_inputs:
        source = kwargs.get("source")
        if source is not None:
            d_B = dprime_B if dprime_B is not None else -dprime
            d_attended = jnp.where(source == 0, dprime, d_B)
            lam_B = lambda_B if lambda_B is not None else lambda_val
            lam = jnp.where(source == 0, lambda_val, lam_B)

            log_lik = fused_loglik_source_mixture(y, K, d_attended, lam,
                                                    thresh_mid, log_gaps, gap_link,
                                                    link_name=link_name)
            if counts is not None:
                log_lik = log_lik * counts
            return log_lik

    raise RuntimeError("univariate_loglik requires fused inputs "
                       "(thresh_mid / log_gaps), which were not provided")


# =============================================================================
# Bivariate Log-Likelihood
# =============================================================================

def bivariate_loglik(y1, y2, dprime, dprime_B, thresh1, thresh2,
                     item_type, link_name, family_name,
                     sigma=None, sigma2=None, sigma_B=None, sigma2_B=None,
                     rho=None, rho_B=None, rho_N=None,
                     discrim=None, discrim_B=None,
                     lambda_val=None, lambda2=None,
                     lambda_B=None, lambda2_B=None,
                     counts=None,
                     thresh1_mid=None, log_gaps1=None,
                     thresh2_mid=None, log_gaps2=None,
                     gap_link="log", src_obs=None,
                     **kwargs):
    """Compute log-likelihood for bivariate SDT families via the fused kernels;
    requires thresh1_mid/log_gaps1/thresh2_mid/log_gaps2. `src_obs` [N] (1 = the source
    rating was given): a row without one is scored on its detection response alone."""

    N = y1.shape[0]
    K1 = thresh1.shape[1] + 1
    K2 = thresh2.shape[1] + 1

    def _with_marginal(joint, marginal):
        return joint if src_obs is None else jnp.where(src_obs == 1, joint, marginal())

    if family_name == "bivariate_cumulative":
        _has_fused = (thresh1_mid is not None and log_gaps1 is not None and
                      thresh2_mid is not None and log_gaps2 is not None)
        if _has_fused:
            mu1_c = dprime if dprime is not None else jnp.zeros(N)
            mu2_c = discrim if discrim is not None else jnp.zeros(N)
            rho_c = rho if rho is not None else jnp.zeros(N)
            log_lik = _with_marginal(fused_loglik_bivariate_sdt(
                y1, y2, item_type, K1, K2,
                mu1=mu1_c, mu2=mu2_c,
                sigma1=jnp.ones(N), sigma2=jnp.ones(N), rho_val=rho_c,
                thresh1_mid=thresh1_mid, gaps1=log_gaps1,
                thresh2_mid=thresh2_mid, gaps2=log_gaps2, gap_link=gap_link),
                lambda: fused_loglik_simple(y1, K1, mu1_c, jnp.ones(N), thresh1_mid, log_gaps1, gap_link))
            if counts is not None:
                log_lik = log_lik * counts
            return log_lik

    if sigma is None: sigma = jnp.ones(N)
    if sigma2 is None: sigma2 = jnp.ones(N)
    if rho is None: rho = jnp.zeros(N)
    if rho_N is None: rho_N = jnp.zeros(N)

    bounded = kwargs.get("bounded", False)

    if rho_B is None:
        rho_B = rho if bounded else -rho

    is_new = (item_type == 1)
    is_A = (item_type == 2)

    mu1_B = dprime_B if dprime_B is not None else dprime
    mu1 = jnp.where(is_new, 0.0, jnp.where(is_A, dprime, mu1_B))
    sd1_B = sigma_B if sigma_B is not None else sigma
    sd1 = jnp.where(is_new, 1.0, jnp.where(is_A, sigma, sd1_B))

    if discrim is not None:
        if family_name == "vrdp2d":
            mu2_A_val = discrim
            mu2_B_val = -discrim if discrim_B is None else discrim_B
        else:
            if discrim_B is None:
                discrim_B_eff = discrim if bounded else -discrim
            else:
                discrim_B_eff = discrim_B
            mu2_A_val = -discrim if bounded else discrim
            mu2_B_val = discrim_B_eff
        mu2 = jnp.where(is_new, 0.0, jnp.where(is_A, mu2_A_val, mu2_B_val))
    else:
        mu2 = jnp.zeros(N)

    if family_name in ("bivariate_sdt", "bivariate_dp") and bounded:
        rho_for_A = -rho
    else:
        rho_for_A = rho
    rho_for_B = rho_B
    sd2_B = sigma2_B if sigma2_B is not None else sigma2
    sd2 = jnp.where(is_new, 1.0, jnp.where(is_A, sigma2, sd2_B))

    rho_n = jnp.where(is_new, rho_N, jnp.where(is_A, rho_for_A, rho_for_B))

    # --- Fused JVP path ---
    has_fused_inputs = (thresh1_mid is not None and log_gaps1 is not None and
                        thresh2_mid is not None and log_gaps2 is not None)
    bounded = kwargs.get("bounded", False)

    if has_fused_inputs:
        K1 = thresh1.shape[1] + 1
        K2 = thresh2.shape[1] + 1

        if family_name == "vrdp2d":
            dprime2 = kwargs.get("dprime2", jnp.zeros(N))
            R = lambda_val if lambda_val is not None else jnp.zeros(N)
            dprime_masked = jnp.where(is_new, 0.0, dprime)
            dprime2_masked = jnp.where(is_new, 0.0, dprime2)
            R_masked = jnp.where(is_new, 0.0, R)
            log_lik = _with_marginal(fused_loglik_vrdp2d(
                y1, y2, item_type, K1, K2,
                dprime_F=dprime_masked, dprime_R=dprime2_masked, source_d=mu2,
                lambda_val=R_masked, sigma_item=sd1, sigma_S=sd2,
                thresh1_mid=thresh1_mid, gaps1=log_gaps1,
                thresh2_mid=thresh2_mid, gaps2=log_gaps2,
                gap_link=gap_link),
                lambda: fused_loglik_mixture(y1, K1, dprime_masked + dprime2_masked, sd1, dprime_masked, sd1,
                                             R_masked, thresh1_mid, log_gaps1, gap_link))
            if counts is not None:
                log_lik = log_lik * counts
            return log_lik

        elif family_name == "bivariate_sdt" and not bounded:
            log_lik = _with_marginal(fused_loglik_bivariate_sdt(
                y1, y2, item_type, K1, K2,
                mu1=mu1, mu2=mu2, sigma1=sd1, sigma2=sd2, rho_val=rho_n,
                thresh1_mid=thresh1_mid, gaps1=log_gaps1,
                thresh2_mid=thresh2_mid, gaps2=log_gaps2,
                gap_link=gap_link),
                lambda: fused_loglik_simple(y1, K1, mu1, sd1, thresh1_mid, log_gaps1, gap_link))
            if counts is not None:
                log_lik = log_lik * counts
            return log_lik

        elif bounded and family_name == "bivariate_sdt":
            log_lik = _with_marginal(fused_loglik_bounded_bivariate_sdt(
                y1, y2, item_type, K1, K2,
                mu1=mu1, mu2=mu2, sigma1=sd1, sigma2=sd2, rho_val=rho_n,
                thresh1_mid=thresh1_mid, gaps1=log_gaps1,
                thresh2_mid=thresh2_mid, gaps2=log_gaps2,
                gap_link=gap_link),
                lambda: fused_loglik_simple(y1, K1, mu1, sd1, thresh1_mid, log_gaps1, gap_link))
            if counts is not None:
                log_lik = log_lik * counts
            return log_lik

        elif bounded and family_name == "bivariate_dp":
            R_I = lambda_val if lambda_val is not None else jnp.zeros(N)
            R_S = lambda2 if lambda2 is not None else jnp.zeros(N)
            R_I_B = lambda_B if lambda_B is not None else R_I
            R_S_B = lambda2_B if lambda2_B is not None else R_S
            R_I = jnp.where(is_new, 0.0, R_I)
            R_S = jnp.where(is_new, 0.0, R_S)
            R_I_B = jnp.where(is_new, 0.0, R_I_B)
            R_S_B = jnp.where(is_new, 0.0, R_S_B)
            log_lik = _with_marginal(fused_loglik_bounded_bivariate_dp(
                y1, y2, item_type, K1, K2,
                mu1=mu1, mu2=mu2, sigma1=sd1, sigma2=sd2,
                rho_val=rho_n,
                R_I=R_I, R_S=R_S, R_I_B=R_I_B, R_S_B=R_S_B,
                thresh1_mid=thresh1_mid, gaps1=log_gaps1,
                thresh2_mid=thresh2_mid, gaps2=log_gaps2,
                gap_link=gap_link, part=kwargs.get("part")),
                lambda: fused_loglik_dpsdt(y1, K1, mu1, sd1, jnp.where(is_A, R_I, R_I_B), thresh1_mid, log_gaps1,
                                           gap_link))
            if counts is not None:
                log_lik = log_lik * counts
            return log_lik

    raise RuntimeError("bivariate_loglik requires fused inputs "
                       "(thresh*_mid / log_gaps*), which were not provided")


# =============================================================================
# CDP Log-Likelihood
# =============================================================================

def _cdp_index(rk, N):
    rk = np.asarray(rk).reshape(-1)
    cap_u, cap_j = _cdp_caps(rk, N)
    pad = lambda ix, cap: np.concatenate([ix, np.full(cap - len(ix), N)]).astype(np.int32)
    return pad(np.flatnonzero(rk == 0), cap_u), pad(np.flatnonzero(rk != 0), cap_j)


def _bdp_index(item_type, y1, K1, N):
    sel = (np.asarray(item_type).reshape(-1) != 1) & (np.asarray(y1).reshape(-1) == int(K1))
    cap = _bdp_cap(item_type, y1, K1, N)
    ix = np.flatnonzero(sel)
    return np.concatenate([ix, np.full(cap - len(ix), N)]).astype(np.int32)


def _bdp_cap(item_type, y1, K1, N):
    if N <= 256:
        return N
    m = int(((np.asarray(item_type).reshape(-1) != 1) & (np.asarray(y1).reshape(-1) == int(K1))).sum())
    return min(N, max(256, -(-m // 256) * 256))


def _bdp_part(config):
    if config.get("bdp_m") is not None:
        return config["bdp_m"]
    st = config.get("_static", config)
    if config.get("family") != "bivariate_dp" or not config.get("bounded") or st.get("y2") is None:
        return None
    return jnp.asarray(_bdp_index(st["item_type"], st["y"], int(st["K"]), int(st["N"])))


def _cdp_caps(rk, N):
    if N <= 256:
        return N, N
    u = int((np.asarray(rk).reshape(-1) == 0).sum())
    if u == 0:
        return 0, N
    cap_u = min(N, -(-u // 256) * 256)
    return cap_u, min(N, N - cap_u + 256)


def cdp_loglik(y, rk, is_old, thresh, mu_R, sigma_R, mu_F, sigma_F,
               c_R, c_K, J, old_level_map, n_rkg, counts=None,
               thresh_mid=None, log_gaps=None, gap_link="log", rk_static=None, corr=None, part=None):
    """Compute log-likelihood for CDP (Continuous Dual-Process) model: unjudged rows (rk = 0, which include every
    "new" response) by the equal-variance cell on M, judged rows by the CDP kernel. `rk_static`: the data's rk.

    Args:
        y: [N] confidence response (1-indexed)
        rk: [N] R/K(/G) judgment (1=R, 2=K, 3=G)
        is_old: [N] binary 0/1
        thresh: [N, J] thresholds for old-response confidence bins
        mu_R, sigma_R: [N] recollection mean/SD
        mu_F, sigma_F: [N] familiarity mean/SD
        c_R: [N] recollection criterion
        c_K: [N] know criterion (only used if n_rkg==3)
        J: int, number of old-response confidence levels
        old_level_map: [J] 1-indexed response categories that are "old"
        n_rkg: 2 or 3
        counts: optional [N]
        thresh_mid: [N] per-obs mid threshold (for fused JVP path)
        log_gaps: [n_gaps, N] per-obs log gaps (for fused JVP path)
        gap_link: "log" or "softplus"
    """

    N = y.shape[0]
    K = len(old_level_map) + J

    # --- Fused JVP path ---
    if thresh_mid is not None and log_gaps is not None:
        old_level_map_0 = jnp.array(old_level_map, dtype=jnp.int32)
        y_expanded = y[:, None]
        match = (y_expanded == old_level_map_0[None, :])
        old_idx = jnp.argmax(match, axis=1)

        def judged_ll(i, y_i, rk_i, old_idx_i):
            return fused_loglik_cdp(
                y=y_i, rk=rk_i, is_old=is_old[i], old_idx=old_idx_i,
                K=K, J=J,
                mu_R=mu_R[i], sigma_R=sigma_R[i], mu_F=mu_F[i], sigma_F=sigma_F[i],
                c_R=c_R[i], c_K=c_K[i],
                thresh_mid=thresh_mid[i], gaps=_gap_rows(log_gaps, i),
                gap_link=gap_link, n_rkg=n_rkg, corr=None if corr is None else corr[i])

        if part is not None:
            u, j = part
            cap_u, cap_j = int(u.shape[0]), int(j.shape[0])
        else:
            cap_u, cap_j = _cdp_caps(rk_static, N)
            u = jnp.nonzero(rk == 0, size=cap_u, fill_value=N)[0]
            j = jnp.nonzero(rk != 0, size=cap_j, fill_value=N)[0]
        if cap_u == 0:
            log_lik = judged_ll(jnp.arange(N), y, rk, old_idx)
        else:
            unjudged = rk == 0
            y_band = jnp.where(match.any(axis=1), old_idx + 2, 1).astype(jnp.int32)
            mu_M = mu_R + mu_F
            sd_M = jnp.sqrt(sigma_R ** 2 + sigma_F ** 2 + (0.0 if corr is None else 2.0 * corr * sigma_R * sigma_F))
            ug = jnp.minimum(u, N - 1)
            log_lik = jnp.zeros(N, dtype=_FLOAT).at[u].set(
                fused_loglik_simple(y_band[ug], J + 1, mu_M[ug], sd_M[ug], thresh_mid[ug], _gap_rows(log_gaps, ug),
                                    gap_link), mode="drop")
            if cap_j:
                jg = jnp.minimum(j, N - 1)
                y_j = jnp.where(unjudged[jg], old_level_map_0[0], y[jg])
                match_j = y_j[:, None] == old_level_map_0[None, :]
                log_lik = log_lik.at[j].set(
                    judged_ll(jg, y_j, jnp.where(unjudged[jg], 1, rk[jg]), jnp.argmax(match_j, axis=1)), mode="drop")
        if counts is not None:
            log_lik = log_lik * counts
        return log_lik

    raise RuntimeError("cdp_loglik requires fused inputs "
                       "(thresh_mid / log_gaps), which were not provided")


# =============================================================================
# NumPyro Model Function
# =============================================================================

def _ensure_2d(x):
    """Ensure array is at least 2D [N, P]."""
    if x.ndim == 1:
        return x[:, None]
    return x


def _positive_dist(d):
    """Constrain a distribution to positive reals (for SD parameters).

    Stan uses <lower=0> on SD parameters; NumPyro needs an explicit
    positive distribution to match.
    """
    low = getattr(d.support, "lower_bound", None)
    high = getattr(d.support, "upper_bound", None)
    if low is not None and bool(np.all(np.asarray(low) >= 0.0)):
        return d if high is None else _PositivePart(d)
    loc = getattr(d, "loc", None)
    zero_loc = loc is None or bool(np.all(np.asarray(loc) == 0.0))
    if isinstance(d, dist.Normal):
        if zero_loc:
            return dist.HalfNormal(d.scale)
        return dist.TruncatedNormal(d.loc, d.scale, low=0.0)
    if isinstance(d, (dist.Cauchy, dist.StudentT, dist.Laplace, dist.Logistic)):
        if zero_loc:
            return dist.FoldedDistribution(d)
        return _PositivePart(d) if isinstance(d, dist.StudentT) else dist.TruncatedDistribution(d, low=0.0)
    return _PositivePart(d)


class _PositivePart(dist.Distribution):
    """A distribution restricted to positive values, as Stan's <lower=0> declaration
    restricts it: density f(x) / (1 - F(0)) for x > 0. Draws are the first positive of
    _CANDIDATES base draws, which needs no inverse CDF."""
    arg_constraints = {}
    support = dist.constraints.positive
    _CANDIDATES = 64

    def __init__(self, base, validate_args=None):
        self.base_dist = base
        self.low = 0.0
        self._lo = base.cdf(jnp.zeros(base.batch_shape))
        super().__init__(base.batch_shape, base.event_shape, validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        cand = self.base_dist.sample(key, (self._CANDIDATES,) + tuple(sample_shape))
        ok = cand > 0
        first = jnp.take_along_axis(cand, jnp.argmax(ok, axis=0)[None], axis=0)[0]
        x = jnp.where(ok.any(axis=0), first, jnp.abs(cand[0]))
        return jnp.maximum(x, jnp.finfo(x.dtype).tiny)

    def log_prob(self, value):
        ok = (value > 0) & self.base_dist.support(value)
        lp = jnp.where(ok, self.base_dist.log_prob(jnp.where(ok, value, 1.0)), -jnp.inf)
        return lp - jnp.log1p(-self._lo)


def _corr_cholesky_site(name, prior_str, dim):
    """LKJ-distributed Cholesky factor of a correlation matrix; a 1 x 1 block has no free
    correlation, so it is the constant [[1]] (as Stan's cholesky_factor_corr[1])."""
    if dim == 1:
        return numpyro.deterministic(name, jnp.ones((1, 1), dtype=_FLOAT))
    return numpyro.sample(name, parse_prior(prior_str, dim=dim))


def _re_corr_cholesky(prefix, conf, dim):
    """Cholesky factor of a random effect's correlation matrix. With `cor_blocks` (lists of
    0-based dimensions) the matrix is block-diagonal: each block has its own LKJ factor
    L_corr_<prefix>_b<k>, assembled into L_corr_<prefix>, which stays lower-triangular
    because every block keeps its index order."""
    prior_str = conf.get("prior_cor", "lkj_corr_cholesky(1)")
    blocks = conf.get("cor_blocks")
    if not blocks:
        return _corr_cholesky_site(f"L_corr_{prefix}", prior_str, dim)
    L = jnp.eye(dim, dtype=_FLOAT)
    for b, ix in enumerate(blocks):
        ix = jnp.asarray(ix, dtype=jnp.int32)
        Lb = _corr_cholesky_site(f"L_corr_{prefix}_b{b + 1}", prior_str, int(ix.shape[0]))
        L = L.at[ix[:, None], ix[None, :]].set(Lb)
    return numpyro.deterministic(f"L_corr_{prefix}", L)


def _whitened_cov(prefix, conf, dim, sd_dists):
    """Correlated block sampled in coordinates whitened by a fixed W: Sigma = W Sigma~ W^T with
    Sigma~ = diag(e^l) Lc(y) Lc(y)^T diag(e^l) and (l, y) unconstrained. The original priors on
    (sd, L_corr) of Sigma are kept exactly through the log-Jacobian of (l, y) -> (log sd, y_corr).
    Returns (sd, L_corr, L) with Sigma = L L^T."""
    from numpyro.distributions.transforms import CorrCholeskyTransform
    cct = CorrCholeskyTransform()
    W = jnp.asarray(conf["whiten"], dtype=_FLOAT).reshape(dim, dim)
    dx = dim + dim * (dim - 1) // 2
    xw = numpyro.sample(f"xw_{prefix}", dist.ImproperUniform(dist.constraints.real, (), (dx,)))

    def to_x(v):
        Aw = jnp.exp(v[:dim])[:, None] * cct(v[dim:])
        S = W @ Aw @ Aw.T @ W.T
        sd = jnp.sqrt(jnp.diag(S))
        Lc = jnp.linalg.cholesky(S / jnp.outer(sd, sd))
        return jnp.concatenate([jnp.log(sd), cct.inv(Lc)]), Aw

    xu, Aw = to_x(xw)
    J = jax.jacfwd(lambda v: to_x(v)[0])(xw)
    sd = jnp.exp(xu[:dim])
    Lc = cct(xu[dim:])
    lp = sum(sd_dists[t].log_prob(sd[t]) for t in range(dim)) + jnp.sum(xu[:dim])
    lp = lp + parse_prior(conf.get("prior_cor", "lkj_corr_cholesky(1)"), dim=dim).log_prob(Lc)
    lp = lp + cct.log_abs_det_jacobian(xu[dim:], Lc) + jnp.linalg.slogdet(J)[1]
    numpyro.factor(f"whiten_prior_{prefix}", lp)
    numpyro.deterministic(f"sigma_{prefix}", sd)
    numpyro.deterministic(f"L_corr_{prefix}", Lc)
    return sd, Lc, W @ Aw


def _smooth_info_scale(sd, info, w):
    """Information-scaled smooth coefficients: s = sd * v with v = K^-T w, K K^T = I + sd^2 info.

    Returns v and the log density of w, -v'v/2 - log|K|, which with s = sd v gives
    exactly s ~ N(0, sd^2 I)."""
    Kc = jnp.linalg.cholesky(jnp.eye(w.shape[0], dtype=w.dtype) + sd * sd * info)
    v = jax.scipy.linalg.solve_triangular(Kc.T, w, lower=False)
    return v, -0.5 * jnp.dot(v, v) - jnp.sum(jnp.log(jnp.diagonal(Kc)))


def _upper_tail_inv(logq):
    """z with log P(Z > z) = logq, in log space (no underflow)."""
    lim = max(-600.0, float(np.log(np.finfo(np.dtype(logq.dtype)).tiny)) + 7.0)
    lq_s = jnp.maximum(logq, lim)
    z_a = -jax.scipy.special.ndtri(jnp.exp(lq_s))
    t = jnp.maximum(-2.0 * logq, 1.0)
    z_b = jnp.sqrt(jnp.maximum(t - jnp.log(t) - jnp.log(2.0 * jnp.pi), 1e-6))
    z = jnp.where(logq > lim, z_a, z_b)
    for _ in range(3):
        lt = jax.scipy.special.log_ndtr(-z)
        z = z + (lt - logq) * jnp.exp(lt + 0.5 * z ** 2 + 0.5 * jnp.log(2.0 * jnp.pi))
    return z


@jax.custom_jvp
def _trunc_quantile(low, s):
    """z ~ N(0,1) truncated to (low, inf) as the quantile of s ~ N(0,1), from the smaller tail:
    Phi(-z) = Phi(-low) Phi(-s). The JVP differentiates that identity, dz = (phi(low) Phi(-s) dlow +
    Phi(-low) phi(s) ds) / phi(z), in log space."""
    ln = jax.scipy.special.log_ndtr
    logq = ln(-low) + ln(-s)
    logp = jnp.logaddexp(ln(low), ln(s) + ln(-low))
    h = jnp.log(0.5)
    return jnp.where(logq <= logp, _upper_tail_inv(jnp.minimum(logq, h)), -_upper_tail_inv(jnp.minimum(logp, h)))


@_trunc_quantile.defjvp
def _trunc_quantile_jvp(primals, tangents):
    low, s = primals
    dlow, ds = tangents
    z = _trunc_quantile(low, s)
    ln = jax.scipy.special.log_ndtr
    hz = 0.5 * z ** 2
    g_low = jnp.exp(-0.5 * low ** 2 + ln(-s) + hz)
    g_s = jnp.exp(ln(-low) - 0.5 * s ** 2 + hz)
    return z, g_low * dlow + g_s * ds


@jax.custom_jvp
def _trunc_logw(lo, hi):
    """log(Phi(hi) - Phi(lo)), from the smaller tail."""
    ln = jax.scipy.special.log_ndtr
    up = lo + hi > 0
    a = jnp.where(up, ln(-lo), ln(hi))
    b = jnp.where(up, ln(-hi), ln(lo))
    return a + jnp.log(-jnp.expm1(jnp.minimum(b - a, 0.0)))


@_trunc_logw.defjvp
def _trunc_logw_jvp(primals, tangents):
    lo, hi = primals
    dlo, dhi = tangents
    w = _trunc_logw(lo, hi)
    c = 0.5 * jnp.log(2.0 * jnp.pi)
    return w, jnp.exp(-0.5 * hi ** 2 - c - w) * dhi - jnp.exp(-0.5 * lo ** 2 - c - w) * dlo


def _log_phi(x):
    return -0.5 * x ** 2 - 0.5 * jnp.log(2.0 * jnp.pi)


@jax.custom_jvp
def _trunc_lm(lo, logw):
    """log(Phi(lo + W) - Phi(lo)) from lo and logw = log W, accurate for any W > 0: a midpoint
    expansion while W max(1, |lo| + W) < 1e-3, else _trunc_logw."""
    W = jnp.exp(logw)
    m = lo + 0.5 * W
    small = W * jnp.maximum(1.0, jnp.abs(lo) + W) < 1e-3
    Ws = jnp.where(small, W, 0.0)
    ms = jnp.where(small, m, 0.0)
    lm_s = logw + _log_phi(ms) + jnp.log1p(Ws ** 2 * (ms ** 2 - 1.0) / 24.0)
    lo_l = jnp.where(small, 0.0, lo)
    lm_l = _trunc_logw(lo_l, lo_l + jnp.where(small, 1.0, W))
    return jnp.where(small, lm_s, lm_l)


@_trunc_lm.defjvp
def _trunc_lm_jvp(primals, tangents):
    lo, logw = primals
    dlo, dlogw = tangents
    lm = _trunc_lm(lo, logw)
    W = jnp.exp(logw)
    m = lo + 0.5 * W
    hi = lo + W
    small = W * jnp.maximum(1.0, jnp.abs(lo) + W) < 1e-3
    u = jnp.where(small, -W * m, 1.0)
    exprel = jnp.where(jnp.abs(u) < 1e-8, 1.0, jnp.expm1(u) / jnp.where(u == 0.0, 1.0, u))
    g_lo_s = -jnp.where(small, m, 0.0) * exprel * jnp.exp(jnp.where(small, logw + _log_phi(lo) - lm, -jnp.inf))
    g_lo_l = jnp.exp(jnp.where(small, -jnp.inf, _log_phi(hi) - lm)) - jnp.exp(jnp.where(small, -jnp.inf, _log_phi(lo) - lm))
    g_w = jnp.exp(logw + _log_phi(hi) - lm)
    return lm, jnp.where(small, g_lo_s, g_lo_l) * dlo + g_w * dlogw


@jax.custom_jvp
def _trunc2_quantile(lo, hi, s):
    """z ~ N(0,1) truncated to (lo, hi) as the quantile of s ~ N(0,1), from the smaller tail:
    Phi(-z) = Phi(-lo) Phi(-s) + Phi(s) Phi(-hi)."""
    ln = jax.scipy.special.log_ndtr
    logq = jnp.logaddexp(ln(-lo) + ln(-s), ln(s) + ln(-hi))
    logp = jnp.logaddexp(ln(lo) + ln(-s), ln(s) + ln(hi))
    h = jnp.log(0.5)
    return jnp.where(logq <= logp, _upper_tail_inv(jnp.minimum(logq, h)), -_upper_tail_inv(jnp.minimum(logp, h)))


@_trunc2_quantile.defjvp
def _trunc2_quantile_jvp(primals, tangents):
    lo, hi, s = primals
    dlo, dhi, ds = tangents
    z = _trunc2_quantile(lo, hi, s)
    ln = jax.scipy.special.log_ndtr
    hz = 0.5 * z ** 2
    g_lo = jnp.exp(-0.5 * lo ** 2 + ln(-s) + hz)
    g_hi = jnp.exp(-0.5 * hi ** 2 + ln(s) + hz)
    g_s = jnp.exp(_trunc_logw(lo, hi) - 0.5 * s ** 2 + hz)
    return z, g_lo * dlo + g_hi * dhi + g_s * ds


@jax.custom_jvp
def _trunc2w_quantile(lo, logw, s):
    """_trunc2_quantile on (lo, lo + W) from logw = log W, with derivatives that stay accurate as
    W -> 0."""
    return _trunc2_quantile(lo, lo + jnp.exp(logw), s)


@_trunc2w_quantile.defjvp
def _trunc2w_quantile_jvp(primals, tangents):
    lo, logw, s = primals
    dlo, dlogw, ds = tangents
    hi = lo + jnp.exp(logw)
    z = _trunc2_quantile(lo, hi, s)
    ln = jax.scipy.special.log_ndtr
    hz = 0.5 * z ** 2
    g_lo = jnp.exp(-0.5 * lo ** 2 + ln(-s) + hz)
    g_hi = jnp.exp(-0.5 * hi ** 2 + ln(s) + hz)
    g_w = jnp.exp(-0.5 * hi ** 2 + ln(s) + hz + logw)
    g_s = jnp.exp(_trunc_lm(lo, logw) - 0.5 * s ** 2 + hz)
    return z, (g_lo + g_hi) * dlo + g_w * dlogw + g_s * ds


def _rkg_guess(config):
    config = config.get("_static", config)
    if config.get("family") != "cdp" or int(config.get("n_rkg", 2)) != 3:
        return None
    y = np.asarray(config["y"])
    rk = np.asarray(config["rk"])
    olm = np.asarray(config["old_level_map"])
    band = np.full(y.shape[0], -1)
    for j, lev in enumerate(olm):
        band[y == lev] = j
    live = np.ones(y.shape[0], bool) if config.get("counts") is None or config.get("rows_live") else np.asarray(config["counts"]) > 0
    G = np.where((band >= 0) & (rk == 3) & live)[0]
    if G.size == 0:
        return None
    return G, band[G]


def _rkg_plan(config):
    """Static plan for the per-level slack of the CDP R/K/G Guess constraint.

    Picks the grouping factor whose rec_crit/know_crit random effects carry the slack: a set of units,
    each a pair (rec_crit term, know_crit term) with the same loading or a single term: 0/1 loadings
    that partition the trials, else terms of disjoint supports with any loadings, where a level whose
    Guess trials load with both signs bounds its slack on both sides and the Guess trials outside
    every support are left to the know-criterion shift. The terms may sit in one cross-correlated block or in blocks of
    their own. Each block's sampled site is keyed as the model samples it: ("x", cor_id) for a
    cross-correlated block, ("r", index) for a per-parameter block (correlated, whitened, one term
    or batched) and ("t", index, term) for one term of an uncorrelated block. Returns None when no
    factor fits."""
    st = config.get("_static", config)
    g = _rkg_guess(st)
    if g is None:
        return None
    G, band = g
    Nn = int(np.asarray(st["y"]).shape[0])
    res = st.get("random_effects") or []

    def loads(rc):
        nt = int(rc["n_terms"])
        if rc.get("type", "index") == "index":
            if rc.get("term_idx") is not None:
                A = np.zeros((Nn, nt))
                ti = np.asarray(rc["term_idx"], dtype=np.int64)
                A[np.flatnonzero(ti > 0), ti[ti > 0] - 1] = 1.0
                return A
            return np.ones((Nn, nt))
        A = np.asarray(rc["Z"], dtype=float)
        return A[:, None] if A.ndim == 1 else A

    def partition(cols):
        b = [j for j, (_, a) in enumerate(cols) if np.all((a == 0) | (a == 1))]
        ones = [j for j in b if np.all(cols[j][1] == 1)]
        if ones:
            return [ones[0]]
        if b and np.all(sum(cols[j][1] for j in b) == 1):
            return b
        return None

    def disjoint(cols):
        taken, out = np.zeros(Nn, bool), []
        for j, (_, a) in enumerate(cols):
            sup = a != 0
            if np.any(sup[G]) and not np.any(sup & taken):
                out.append(j)
                taken |= sup
        return out or None

    n_th = int(st.get("n_thresh", int(st.get("K", 2)) - 1))
    c_mid = (n_th + 1) // 2 - 1
    scal_groups = st.get("crit_scalar_groups")

    def crit_cols(rc):
        nt = int(rc["n_terms"])
        loc = rc.get("location_only") or (st.get("crit_scalar") and scal_groups is None) or (
            scal_groups is not None and rc.get("group") in scal_groups)
        R = nt if rc.get("location_only") else nt // n_th
        dims = list(range(R)) if loc else [c_mid * R + r for r in range(R)]
        if rc.get("type", "index") == "index":
            if rc.get("term_idx") is not None:
                A = np.zeros((Nn, R))
                ti = np.asarray(rc["term_idx"], dtype=np.int64)
                A[np.flatnonzero(ti > 0), ti[ti > 0] - 1] = 1.0
            else:
                A = np.ones((Nn, R))
        else:
            A = np.asarray(rc["Z"], dtype=float)
            A = A[:, None] if A.ndim == 1 else A
        return [(dims[r], A[:, r]) for r in range(R)]

    cross_off = {}
    for i, rc in enumerate(res):
        cid = rc.get("cor_id")
        if cid is not None:
            o = cross_off.setdefault(cid, [0])
            rc_off = o[0]
            o[0] += int(rc["n_terms"])
            cross_off.setdefault((cid, i), rc_off)

    def site(i, j):
        rc = res[i]
        if rc.get("cor_id") is not None:
            return ("x", rc["cor_id"]), cross_off[(rc["cor_id"], i)] + j
        if int(rc["n_terms"]) > 1 and not rc.get("correlated", True):
            return ("t", i, j), 0
        return ("r", i), j

    best = None
    for group in sorted({rc["group"] for rc in res if rc.get("param") in ("rec_crit", "know_crit", "criterion")}):
        cols = {"rec_crit": [], "know_crit": [], "criterion": []}
        idx = None
        for i, rc in enumerate(res):
            if rc.get("group") != group or rc.get("param") not in cols:
                continue
            if rc["param"] == "criterion":
                cols["criterion"] += [((i, j), a) for j, a in crit_cols(rc)]
            else:
                A = loads(rc)
                for j in range(A.shape[1]):
                    cols[rc["param"]].append(((i, j), A[:, j]))
            idx = rc["index"]

        def pair_k(js):
            out = []
            for j in js:
                k_ref, ind = cols["know_crit"][j]
                m = [r for r, a in cols["rec_crit"] if np.array_equal(a, ind)]
                out.append((m[0] if m else None, k_ref, None, ind))
            return out

        units = None
        for how, src in ((partition, "know_crit"), (partition, "rec_crit"), (partition, "criterion"),
                         (disjoint, "know_crit"), (disjoint, "rec_crit")):
            js = how(cols[src]) if cols[src] else None
            if js is None:
                continue
            if src == "know_crit":
                units = pair_k(js)
            elif src == "rec_crit":
                units = [(cols["rec_crit"][j][0], None, None, cols["rec_crit"][j][1]) for j in js]
            else:
                units = [(None, None, cols["criterion"][j][0], cols["criterion"][j][1]) for j in js]
            break
        if units is None:
            continue
        terms, unit_pos, blocks = [], [], {}
        for r_ref, k_ref, c_ref, ind in units:
            pos = []
            for kind, ref in (("R", r_ref), ("K", k_ref), ("C", c_ref)):
                if ref is None:
                    continue
                key, dim = site(*ref)
                pos.append(len(terms))
                blocks.setdefault(key, []).append((len(terms), dim))
                terms.append(dict(key=key, dim=dim, kind=kind, load=ind))
            unit_pos.append(pos)
        lvl_idx = np.asarray(idx, dtype=np.int64) - 1
        n_groups = int(next(rc["n_groups"] for rc in res if rc.get("group") == group))
        own = np.full(G.shape[0], -1)
        for t, (_, _, _, ind) in enumerate(units):
            own[ind[G] != 0] = t
        kc_fixed = int((st.get("params") or {}).get("know_crit", {}).get("n_coef", 0)) >= 1
        if np.any(own < 0) and not kc_fixed:
            continue
        G_t, band_t, masks, cells, x_t, pos_m, neg_m, pairs = [], [], [], 0, [], [], [], []
        for t in range(len(units)):
            sel = own == t
            G_t.append(G[sel])
            band_t.append(band[sel])
            msk = np.zeros(n_groups, bool)
            msk[np.unique(lvl_idx[G[sel]])] = True
            masks.append(msk)
            cells += int(msk.sum())
            xs = units[t][3][G[sel]]
            lv = lvl_idx[G[sel]]
            x_t.append(xs)
            pm, nm = np.zeros(n_groups, bool), np.zeros(n_groups, bool)
            pm[np.unique(lv[xs > 0])] = True
            nm[np.unique(lv[xs < 0])] = True
            pos_m.append(pm)
            neg_m.append(nm)
            pp, qq = [], []
            for l in np.flatnonzero(pm & nm):
                P_, N_ = np.flatnonzero((lv == l) & (xs > 0)), np.flatnonzero((lv == l) & (xs < 0))
                pp.append(np.repeat(P_, N_.size))
                qq.append(np.tile(N_, P_.size))
            pairs.append((np.concatenate(pp).astype(np.int64), np.concatenate(qq).astype(np.int64)) if pp else None)
        if not kc_fixed and any(p is not None for p in pairs):
            continue
        free = G[own < 0]
        score = G.shape[0] / max(cells, 1)
        mm = len(terms)
        cand = dict(group=group, terms=terms, unit_pos=unit_pos, blocks=blocks, mm=mm,
                    WR=np.stack([t_["load"] if t_["kind"] == "R" else np.zeros(Nn) for t_ in terms], 1),
                    WK=np.stack([t_["load"] if t_["kind"] == "K" else np.zeros(Nn) for t_ in terms], 1),
                    lvl_idx=lvl_idx, n_groups=n_groups, G_t=G_t, band_t=band_t, masks=masks, score=score,
                    ind=np.stack([(ind != 0).astype(float) for _, _, _, ind in units], 1),
                    WC=np.stack([t_["load"] if t_["kind"] == "C" else np.zeros(Nn) for t_ in terms], 1),
                    sgn=np.asarray([-1.0 if t_["kind"] == "C" else 1.0 for t_ in terms]),
                    x_t=x_t, pos_m=pos_m, neg_m=neg_m, pairs=pairs, free=free, free_band=band[own < 0])
        if best is None or score < best["score"]:
            best = cand
    return best


def _force_last(p, back, D):
    """The LKJ order p (a permutation of 0..D-1) with the dims `back` moved last, in that order."""
    rank = np.full(D, -1, dtype=np.int64)
    rank[np.asarray(back, dtype=np.int64)] = np.arange(len(back))
    if isinstance(p, np.ndarray):
        rk = rank[p]
        key = np.where(rk >= 0, D + rk, np.arange(D))
        return p[np.argsort(key, kind="stable")]
    rk = jnp.asarray(rank)[p]
    key = jnp.where(rk >= 0, D + rk, jnp.arange(D))
    return jnp.take(p, jnp.argsort(key))


def _rkg_back(plan, prefix):
    """The model dims of the cross-correlated block `prefix` that carry the R/K/G slack, or None."""
    if plan is None:
        return None
    for key, ds in plan["blocks"].items():
        if key[0] == "x" and prefix == f"cross_{key[1]}_{plan['group']}":
            return [dm for _, dm in ds]
    return None


def _rkg_knob(config):
    """Static setup for sampling the CDP R/K/G know criterion on its feasible set.

    A Guess response in the confidence band starting at tau_k is possible only when
    c_R + c_K > tau_k, so the likelihood has hard walls in the know-criterion
    coefficients. With v solving X v = 1 (the fixed-effect design spans the constant),
    beta = B_perp w + t v and t is solved last as max over Guess trials of the bound plus
    exp(slack): an exact bijection onto the feasible set whose coordinates never meet a
    wall. Returns None when no Guess response constrains the model."""
    config = config.get("_static", config)
    if config.get("family") != "cdp" or int(config.get("n_rkg", 2)) != 3:
        return None
    pconf = config.get("params", {}).get("know_crit")
    if pconf is None or pconf.get("n_coef", 0) < 1:
        return None
    X = np.asarray(pconf["X"], dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    y = np.asarray(config["y"])
    rk = np.asarray(config["rk"])
    olm = np.asarray(config["old_level_map"])
    band = np.full(y.shape[0], -1)
    for j, lev in enumerate(olm):
        band[y == lev] = j
    live = np.ones(y.shape[0], bool) if config.get("counts") is None or config.get("rows_live") else np.asarray(config["counts"]) > 0
    G = np.where((band >= 0) & (rk == 3) & live)[0]
    if G.size == 0:
        return None
    v = np.linalg.lstsq(X, np.ones(X.shape[0]), rcond=None)[0]
    if not np.allclose(X @ v, 1.0, atol=1e-9):
        raise ValueError("know_crit: the fixed-effect design must span the constant (an intercept "
                         "or a full factor coding)")
    d = X.shape[1]
    plan = _rkg_plan(config)
    if plan is None and any(rc.get("param") in ("criterion", "rec_crit", "know_crit") for rc in config.get("random_effects") or []):
        return None
    V, G_t, band_t = v[:, None], [G], [band[G]]
    if config.get("_rkg_V") is not None:
        V = np.asarray(config["_rkg_V"], dtype=float).reshape(d, -1)
        if V.shape[1] > 1:
            if plan is None or len(plan["G_t"]) != V.shape[1]:
                raise RuntimeError("bayesroc: the trimmed configuration lost a unit of the R/K/G slack plan")
            G_t, band_t = list(plan["G_t"]), list(plan["band_t"])
    elif plan is not None and plan["ind"].shape[1] > 1 and plan["free"].size == 0:
        Vp = np.linalg.lstsq(X, plan["ind"], rcond=None)[0]
        if np.allclose(X @ Vp, plan["ind"], atol=1e-9) and np.linalg.matrix_rank(Vp) == Vp.shape[1]:
            V, G_t, band_t = Vp, list(plan["G_t"]), list(plan["band_t"])
    B_perp = np.linalg.svd(V.T)[2][V.shape[1]:].T if d > V.shape[1] else np.zeros((d, 0))
    cover = [[]] if plan is None else ([[j] for j in range(V.shape[1])] if V.shape[1] > 1 else [list(range(len(plan["G_t"])))])
    hard = [plan is not None and (any(plan["pairs"][u] is not None for u in c) or (V.shape[1] == 1 and plan["free"].size > 0))
            for c in cover]
    return {"X": X, "v": v, "V": V, "B_perp": B_perp, "G": G, "band": band[G], "soft": plan is not None,
            "G_t": G_t, "band_t": band_t, "s_t": [1.0 if h else 1.0 / np.sqrt(g.size) for g, h in zip(G_t, hard)],
            "cover": cover, "hard": hard}


def _rkg_shift(knob, knob_state, plan, thresh, c_R, c_K):
    """c_K with the know-criterion shift t (one per column of V) added. A soft column puts t near
    lam, the largest Guess bound of its trials, with the per-level slack absorbing the rest. A hard
    column, whose units have a level bounded on both sides or Guess trials outside every support,
    keeps t above lam, the least shift at which every such level's interval is nonempty: for one
    level, the largest crossing of a positive- and a negative-loading Guess bound."""
    Vk = jnp.asarray(knob["V"], dtype=_FLOAT)
    nT = Vk.shape[1]
    w = numpyro.sample("know_crit_shift", dist.ImproperUniform(dist.constraints.real, (), (nT,)))
    coefs = jnp.stack(knob_state["level"]["coef"]) if "level" in knob_state else None
    ts, sj, lams, rs = [], 0.0, [], []
    floor = knob_state.setdefault("floor", {})
    for j in range(nT):
        sc = knob["s_t"][j]
        if knob["hard"][j]:
            vals, gs = [], {}
            for u in knob["cover"][j]:
                if plan["pairs"][u] is None:
                    continue
                Gu = jnp.asarray(plan["G_t"][u])
                a = thresh[Gu, jnp.asarray(plan["band_t"][u])] - c_R[Gu] - c_K[Gu]
                p, q = (jnp.asarray(ix) for ix in plan["pairs"][u])
                xp = jnp.asarray(plan["x_t"][u][plan["pairs"][u][0]], dtype=_FLOAT)
                xq = -jnp.asarray(plan["x_t"][u][plan["pairs"][u][1]], dtype=_FLOAT)
                gs[u] = (a[p] * xq + a[q] * xp) / (xp + xq)
                vals.append(jnp.max(gs[u]))
            if nT == 1 and plan["free"].size > 0:
                Gf = jnp.asarray(plan["free"])
                vals.append(jnp.max(thresh[Gf, jnp.asarray(plan["free_band"])] - c_R[Gf] - c_K[Gf]))
            lam = jnp.max(jnp.stack(vals))
            r = jnp.asarray(0.0, dtype=_FLOAT)
        else:
            G = jnp.asarray(knob["G_t"][j])
            lam = jnp.max(thresh[G, jnp.asarray(knob["band_t"][j])] - c_R[G] - c_K[G])
            cj = 0.0 if coefs is None else (coefs[j] if nT > 1 else jnp.min(coefs))
            r = cj / jnp.sqrt(cj ** 2 + sc ** 2)
        x = w[j] - lam
        if knob["hard"][j]:
            y = x / sc
            log_gap = jnp.log(sc) + jnp.where(y < -30.0, y, jnp.log(jax.nn.softplus(jnp.maximum(y, -30.0))))
            for u, g in gs.items():
                floor[u] = dict(g=g, lam=lam, log_gap=log_gap)
        ts.append(lam + r * x + (1.0 - r) * sc * jax.nn.softplus(x / sc))
        sj = sj + (jax.nn.log_sigmoid(x / sc) if knob["hard"][j] else jnp.log(r + (1.0 - r) * jax.nn.sigmoid(x / sc)))
        lams.append(lam)
        rs.append(r)
    numpyro.factor("know_crit_shift_jac", sj)
    numpyro.deterministic("kc_shift_lam", jnp.stack(lams))
    numpyro.deterministic("kc_shift_r", jnp.stack(rs))
    v = Vk @ jnp.stack(ts)
    beta_kc = knob_state["beta_perp"] + v
    prior_strs = knob_state["prior_strs"]
    numpyro.factor("know_crit_prior", sum(parse_prior(prior_strs[j]).log_prob(beta_kc[j])
                                          for j in range(knob_state["n_coef"])))
    if knob_state["n_coef"] == 1:
        numpyro.deterministic("beta_know_crit[1]", beta_kc[0])
    elif len(set(prior_strs)) == 1:
        numpyro.deterministic("beta_know_crit", beta_kc)
    else:
        for j in range(knob_state["n_coef"]):
            numpyro.deterministic(f"beta_know_crit[{j+1}]", beta_kc[j])
    return c_K + knob_state["X"] @ v


def _rkg_lift(config, th, to_dict, off):
    """Starting points th [C, D] with each hard know-criterion shift coordinate raised to at least
    lam + max(1, 3 s), so a start is never in the collapsed regime where t sits on its floor."""
    knob = _rkg_knob(config)
    if knob is None or not any(knob.get("hard", ())) or "know_crit_shift" not in off:
        return th
    from numpyro.infer.util import constrain_fn
    th = np.array(th, dtype=np.float64)
    sl = off["know_crit_shift"][0]
    lift = np.asarray([max(1.0, 3.0 * s) if h else -np.inf for s, h in zip(knob["s_t"], knob["hard"])])
    for c in range(th.shape[0]):
        out = constrain_fn(sdt_model, (config,), {}, to_dict(th[c]), return_deterministic=True)
        lam = np.asarray(out["kc_shift_lam"], dtype=np.float64).reshape(-1)
        th[c, sl] = np.maximum(th[c, sl], lam + lift)
    return th


def _apply_crit_scalar(config, n_thresh):
    mid = (n_thresh + 1) // 2 - 1
    for rc in config.get("random_effects") or []:
        if rc.get("param") != "criterion" or rc.get("location_only"):
            continue
        groups = config.get("crit_scalar_groups")
        if groups is not None and rc.get("group") not in groups:
            continue
        Rt = int(rc["n_terms"]) // n_thresh
        per = rc.get("prior_sd_per_coef")
        if isinstance(per, list) and len(per) == int(rc["n_terms"]):
            rc["prior_sd_per_coef"] = per[mid * Rt:(mid + 1) * Rt]
        rc["n_terms"] = Rt
        rc.pop("cor_blocks", None)
        rc["location_only"] = True


def sdt_model(config):
    """NumPyro model function driven by config dict."""
    N = config["N"]
    K = config["K"]
    n_thresh = config.get("n_thresh", K - 1)
    if config.get("crit_scalar") or config.get("crit_scalar_groups") is not None:
        _apply_crit_scalar(config, n_thresh)
    gap_link = config.get("gap_link", "log")
    y = jnp.array(config["y"], dtype=jnp.int32)
    family_name = config["family"]
    link_name = config["link"]

    is_old = jnp.array(config["is_old"], dtype=_FLOAT) if config.get("is_old") is not None else jnp.ones(N)

    counts = jnp.array(config["counts"], dtype=_FLOAT) if config.get("counts") is not None else None

    params_config = config.get("params", {})
    re_configs = config.get("random_effects", [])

    # ---- Sample fixed effects for each parameter ----
    param_values = {}

    needs_ordered = config.get("needs_ordered_dprime", False)
    ordered_done = False

    knob = _rkg_knob(config)
    knob_state = {}
    rkg_plan = _rkg_plan(config)
    rkg_sites = {} if rkg_plan is None else rkg_plan["blocks"]

    def _rkg_hook(key, A, z, emit):
        """The slack plan's sampled site `key` with prior factor A (u = A z): z with its components in the
        subspace that moves only the plan's rows of A removed (orth(A^-1 E_B), an orthonormal basis of
        z's coordinates), and the removed coordinates, their effect on those rows and `emit` (the site's
        deterministics from its full effects) kept for the slack."""
        dims = jnp.asarray([dm for _, dm in rkg_sites[key]])
        E = jnp.zeros((A.shape[0], dims.shape[0]), dtype=A.dtype).at[dims, jnp.arange(dims.shape[0])].set(1.0)
        Q, _ = jnp.linalg.qr(jnp.linalg.solve(A, E))
        zeta = Q.T @ z
        z0 = z - Q @ zeta
        knob_state.setdefault("sites", {})[key] = dict(zeta=zeta, M=(A @ Q)[dims], rows=dims, A=A, z0=z0,
                                                       terms=[tq for tq, _ in rkg_sites[key]], emit=emit)
        return z0

    _batch_scalar_betas = {}
    _batch_skip = set()
    for param_name, pconf in params_config.items():
        if param_name in ("criterion", "criterion2"):
            continue
        if knob is not None and param_name == "know_crit":
            continue
        if needs_ordered and param_name in ("dprime", "dprime2"):
            continue
        if pconf.get("n_coef", 0) != 1:
            continue
        col_names = pconf.get("col_names", ["V1"])
        if isinstance(col_names, str):
            col_names = [col_names]
        cname = col_names[0] if len(col_names) > 0 else "V1"
        prior_str = pconf.get("priors", {}).get(cname, "normal(0, 2.5)")
        _batch_scalar_betas.setdefault(prior_str, []).append(
            (param_name, jnp.array(pconf["X"], dtype=_FLOAT)))

    _batched_beta_values = {}
    for prior_str, entries in _batch_scalar_betas.items():
        if len(entries) < 2:
            continue
        n_batch = len(entries)
        param_tag = "_".join(sorted(p for p, _ in entries))
        batch_sample = numpyro.sample(f"beta_batch_{param_tag}",
                                       parse_prior(prior_str).expand([n_batch]))
        for b, (param_name, X) in enumerate(entries):
            _batched_beta_values[param_name] = batch_sample[b:b+1]
            _batch_skip.add(param_name)
            numpyro.deterministic(f"beta_{param_name}[1]", batch_sample[b])

    for param_name, pconf in params_config.items():
        if param_name in ("criterion", "criterion2"):
            continue

        if needs_ordered and param_name in ("dprime", "dprime2"):
            if ordered_done:
                continue
            if param_name == "dprime2" and "dprime" not in params_config:
                pass
            elif param_name == "dprime" or (param_name == "dprime2" and "dprime" in params_config):
                dp_conf = params_config["dprime"]
                dp2_conf = params_config["dprime2"]
                n_coef_dp = dp_conf["n_coef"]
                X_dp = _ensure_2d(jnp.array(dp_conf["X"], dtype=_FLOAT))
                X_dp2 = _ensure_2d(jnp.array(dp2_conf["X"], dtype=_FLOAT))
                dp_priors = dp_conf.get("priors", {})
                dp2_priors = dp2_conf.get("priors", {})
                dp_col_names = dp_conf.get("col_names", [f"V{i+1}" for i in range(n_coef_dp)])
                dp2_col_names = dp2_conf.get("col_names", [f"V{i+1}" for i in range(n_coef_dp)])
                if isinstance(dp_col_names, str):
                    dp_col_names = [dp_col_names]
                if isinstance(dp2_col_names, str):
                    dp2_col_names = [dp2_col_names]

                n_coef_dp2 = dp2_conf["n_coef"]
                pivot = int(config["dprime_pivot"]) - 1
                xbar1 = jnp.asarray(np.asarray(config["dprime_xbar"], dtype=np.float64).reshape(-1), dtype=_FLOAT)
                xbar2 = jnp.asarray(np.asarray(config["dprime2_xbar"], dtype=np.float64).reshape(-1), dtype=_FLOAT)
                beta_dp2 = jnp.zeros(n_coef_dp2)
                for j in range(n_coef_dp2):
                    cname2 = dp2_col_names[j] if j < len(dp2_col_names) else f"V{j+1}"
                    dp2_prior = parse_prior(dp2_priors.get(cname2, "normal(0, 1)"))
                    beta_dp2 = beta_dp2.at[j].set(numpyro.sample(f"beta_dprime2[{j+1}]", dp2_prior))
                beta_dp = jnp.zeros(n_coef_dp)
                k_free = 0
                for j in range(n_coef_dp):
                    if j == pivot:
                        continue
                    k_free += 1
                    cname = dp_col_names[j] if j < len(dp_col_names) else f"V{j+1}"
                    beta_dp = beta_dp.at[j].set(numpyro.sample(
                        f"beta_dprime_free[{k_free}]", parse_prior(dp_priors.get(cname, "normal(1, 1)"))))
                raw_gap = numpyro.sample("dprime_order_loggap",
                                         dist.ImproperUniform(dist.constraints.real, (), ()))
                numpyro.factor("dprime_order_loggap_jac", raw_gap)
                piv_val = (jnp.exp(raw_gap) + xbar2 @ beta_dp2 - xbar1 @ beta_dp) / xbar1[pivot]
                beta_dp = beta_dp.at[pivot].set(piv_val)
                cpiv = dp_col_names[pivot] if pivot < len(dp_col_names) else f"V{pivot+1}"
                numpyro.factor("beta_dprime_pivot_prior", parse_prior(dp_priors.get(cpiv, "normal(1, 1)")).log_prob(piv_val))
                for j in range(n_coef_dp):
                    numpyro.deterministic(f"beta_dprime[{j+1}]", beta_dp[j])

                param_values["dprime"] = X_dp @ beta_dp
                param_values["dprime2"] = X_dp2 @ beta_dp2
                ordered_done = True
                continue

        n_coef = pconf["n_coef"]
        X = jnp.array(pconf["X"], dtype=_FLOAT)
        if X.ndim == 1:
            X = X[:, None]
        priors = pconf.get("priors", {})
        col_names = pconf.get("col_names", [f"V{i+1}" for i in range(n_coef)])
        if isinstance(col_names, str):
            col_names = [col_names]

        if knob is not None and param_name == "know_crit":
            B_perp = jnp.asarray(knob["B_perp"], dtype=_FLOAT)
            if B_perp.shape[1] > 0:
                w = numpyro.sample("know_crit_free",
                                   dist.ImproperUniform(dist.constraints.real, (), (B_perp.shape[1],)))
                beta_perp = B_perp @ w
            else:
                beta_perp = jnp.zeros(n_coef, dtype=_FLOAT)
            knob_state.update(beta_perp=beta_perp, X=X, n_coef=n_coef,
                              prior_strs=[priors.get(col_names[j] if j < len(col_names) else f"V{j+1}",
                                                     "normal(0, 2.5)") for j in range(n_coef)])
            param_values[param_name] = X @ beta_perp
            continue

        if n_coef > 0:
            if param_name in _batch_skip:
                beta = _batched_beta_values[param_name]
            elif n_coef == 1:
                cname = col_names[0] if len(col_names) > 0 else "V1"
                prior_str = priors.get(cname, "normal(0, 2.5)")
                beta = numpyro.sample(f"beta_{param_name}[1]", parse_prior(prior_str)).reshape(1)
            else:
                prior_strs = [priors.get(col_names[j] if j < len(col_names) else f"V{j+1}", "normal(0, 2.5)")
                              for j in range(n_coef)]
                vector_site = len(set(prior_strs)) == 1
                if vector_site:
                    beta = numpyro.sample(f"beta_{param_name}",
                                          parse_prior(prior_strs[0]).expand([n_coef]))
                else:
                    beta = jnp.zeros(n_coef)
                    for j in range(n_coef):
                        beta_j = numpyro.sample(f"beta_{param_name}[{j+1}]",
                                                parse_prior(prior_strs[j]))
                        beta = beta.at[j].set(beta_j)
                centre = _centre_of(pconf)
                if centre is not None:
                    beta_c, beta = beta, _uncentre(beta, centre)
                    _centre_factor(f"beta_{param_name}", lambda j: parse_prior(prior_strs[j]), centre, beta, beta_c)
                    if vector_site:
                        numpyro.deterministic(f"orig__beta_{param_name}", beta)
                    else:
                        for j in range(n_coef):
                            numpyro.deterministic(f"orig__beta_{param_name}[{j+1}]", beta[j])

            eta = X @ beta
        else:
            eta = jnp.zeros(N)
        param_values[param_name] = eta

    # ---- Sample random effects ----
    criterion_re_configs = [rc for rc in re_configs
                            if rc.get("param") in ("criterion", "criterion2")
                            and rc.get("cor_id") is None]
    bias_re_configs = [rc for rc in re_configs if rc.get("param") == "bias" and rc.get("cor_id") is None]

    cross_cor_groups = {}
    independent_re_idxs = []

    for re_idx, re_conf in enumerate(re_configs):
        param = re_conf.get("param")
        cor_id = re_conf.get("cor_id")
        if param in ("criterion", "criterion2", "bias") and cor_id is None:
            continue
        if cor_id is not None:
            cross_cor_groups.setdefault(cor_id, []).append(re_idx)
        else:
            independent_re_idxs.append(re_idx)

    def _apply_re_offset(param, offset, param_values):
        """Accumulate RE offset into param's linear predictor."""
        if param in param_values:
            param_values[param] = param_values[param] + offset
        else:
            param_values[param] = offset

    def _sample_independent_re(re_conf, param_values):
        """Sample a single independent (non-cross-correlated) RE block."""
        param = re_conf["param"]
        group = re_conf["group"]
        n_groups = int(re_conf["n_groups"])
        n_terms = int(re_conf["n_terms"])
        correlated = re_conf.get("correlated", True)
        re_type = re_conf.get("type", "index")
        sd_prior = _positive_dist(parse_prior(re_conf.get("prior_sd", "normal(0, 0.5)")))
        has_term_idx = re_conf.get("term_idx") is not None
        per_coef = re_conf.get("prior_sd_per_coef")
        if isinstance(per_coef, str):
            per_coef = [per_coef]
        if per_coef is not None and len(per_coef) == n_terms:
            sd_priors = [_positive_dist(parse_prior(p)) for p in per_coef]
            per_coef_varies = len(set(per_coef)) > 1
        else:
            sd_priors = [sd_prior] * n_terms
            per_coef_varies = False

        if n_terms == 1 or not correlated:
            if n_terms == 1:
                sd = numpyro.sample(f"sigma_{param}_{group}[1]", sd_prior)
                z = numpyro.sample(f"z_{param}_{group}[1]",
                                   dist.Normal(0, 1).expand([n_groups]))
                key = ("r", re_conf["_idx"])
                if key in rkg_sites:
                    def _emit1(u_f, nm=f"u_{param}_{group}"):
                        numpyro.deterministic(nm, u_f[0])
                    z = _rkg_hook(key, jnp.reshape(sd, (1, 1)), z[None, :], _emit1)[0]
                u = sd * z
                if key not in rkg_sites:
                    numpyro.deterministic(f"u_{param}_{group}", u)

                if re_type == "index":
                    idx = jnp.array(re_conf["index"], dtype=jnp.int32) - 1
                    offset = u[idx]
                    if has_term_idx:
                        tidx = jnp.array(re_conf["term_idx"], dtype=jnp.int32)
                        offset = jnp.where(tidx > 0, offset, 0.0)
                    _apply_re_offset(param, offset, param_values)
                else:
                    Z = _ensure_2d(jnp.array(re_conf["Z"], dtype=_FLOAT))
                    idx = jnp.array(re_conf["index"], dtype=jnp.int32) - 1
                    _apply_re_offset(param, Z[:, 0] * u[idx], param_values)
            else:
                u_all = jnp.zeros((n_terms, n_groups))
                tkeys = [("t", re_conf["_idx"], t) for t in range(n_terms)]
                defer = any(k in rkg_sites for k in tkeys)
                buf = {}
                for t in range(n_terms):
                    sd_t = numpyro.sample(f"sigma_{param}_{group}[{t+1}]", sd_priors[t])
                    z_t = numpyro.sample(f"z_{param}_{group}[{t+1}]",
                                         dist.Normal(0, 1).expand([n_groups]))
                    if tkeys[t] in rkg_sites:
                        def _emit_t(u_f, t_=t):
                            buf[t_] = u_f[0]
                            if sum(k_ != "u0" for k_ in buf) == sum(k in rkg_sites for k in tkeys):
                                full = buf["u0"]
                                for k_, v_ in buf.items():
                                    if k_ != "u0":
                                        full = full.at[k_].set(v_)
                                numpyro.deterministic(f"u_{param}_{group}", full.T)
                        z_t = _rkg_hook(tkeys[t], jnp.reshape(sd_t, (1, 1)), z_t[None, :], _emit_t)[0]
                    u_t = sd_t * z_t
                    u_all = u_all.at[t].set(u_t)

                    if re_type == "index":
                        idx = jnp.array(re_conf["index"], dtype=jnp.int32) - 1
                        if has_term_idx:
                            tidx = jnp.array(re_conf["term_idx"], dtype=jnp.int32) - 1
                            offset = jnp.where(tidx == t, u_t[idx], 0.0)
                        else:
                            offset = u_t[idx]
                    else:
                        Z = _ensure_2d(jnp.array(re_conf["Z"], dtype=_FLOAT))
                        idx = jnp.array(re_conf["index"], dtype=jnp.int32) - 1
                        offset = Z[:, t] * u_t[idx]

                    _apply_re_offset(param, offset, param_values)

                if defer:
                    buf["u0"] = u_all
                else:
                    numpyro.deterministic(f"u_{param}_{group}", u_all.T)
        elif re_conf.get("whiten") is not None and not re_conf.get("cor_blocks"):
            sd_list = sd_priors if per_coef_varies else [sd_prior] * n_terms
            sds, L_corr, L = _whitened_cov(f"{param}_{group}", re_conf, n_terms, sd_list)
            numpyro.deterministic(f"corr_{param}_{group}", L_corr @ L_corr.T)
            z = numpyro.sample(f"z_{param}_{group}",
                               dist.Normal(0, 1).expand([n_terms, n_groups]))
            key = ("r", re_conf["_idx"])
            if key in rkg_sites:
                def _emitw(u_f, nm=f"u_{param}_{group}"):
                    numpyro.deterministic(nm, u_f.T)
                z = _rkg_hook(key, L, z, _emitw)
            u = L @ z
            if key not in rkg_sites:
                numpyro.deterministic(f"u_{param}_{group}", u.T)
            if re_type == "index":
                idx = jnp.array(re_conf["index"], dtype=jnp.int32) - 1
                if has_term_idx:
                    tidx = jnp.array(re_conf["term_idx"], dtype=jnp.int32) - 1
                    for t in range(n_terms):
                        _apply_re_offset(param, jnp.where(tidx == t, u[t][idx], 0.0), param_values)
                else:
                    for t in range(n_terms):
                        _apply_re_offset(param, u[t][idx], param_values)
            else:
                Z = _ensure_2d(jnp.array(re_conf["Z"], dtype=_FLOAT))
                idx = jnp.array(re_conf["index"], dtype=jnp.int32) - 1
                for t in range(n_terms):
                    _apply_re_offset(param, Z[:, t] * u[t][idx], param_values)
        else:
            if per_coef_varies:
                sds = jnp.stack([numpyro.sample(f"sigma_{param}_{group}[{t+1}]", sd_priors[t])
                                 for t in range(n_terms)])
            else:
                sds = numpyro.sample(f"sigma_{param}_{group}",
                                      sd_prior.expand([n_terms]))

            L_corr = _re_corr_cholesky(f"{param}_{group}", re_conf, n_terms)
            numpyro.deterministic(f"corr_{param}_{group}", L_corr @ L_corr.T)
            L = jnp.diag(sds) @ L_corr

            z = numpyro.sample(f"z_{param}_{group}",
                               dist.Normal(0, 1).expand([n_terms, n_groups]))
            key = ("r", re_conf["_idx"])
            if key in rkg_sites:
                def _emitc(u_f, nm=f"u_{param}_{group}"):
                    numpyro.deterministic(nm, u_f.T)
                z = _rkg_hook(key, L, z, _emitc)
            u = L @ z

            if key not in rkg_sites:
                numpyro.deterministic(f"u_{param}_{group}", u.T)

            if re_type == "index":
                idx = jnp.array(re_conf["index"], dtype=jnp.int32) - 1
                if has_term_idx:
                    tidx = jnp.array(re_conf["term_idx"], dtype=jnp.int32) - 1
                    for t in range(n_terms):
                        offset = jnp.where(tidx == t, u[t][idx], 0.0)
                        _apply_re_offset(param, offset, param_values)
                else:
                    for t in range(n_terms):
                        _apply_re_offset(param, u[t][idx], param_values)
            else:
                Z = _ensure_2d(jnp.array(re_conf["Z"], dtype=_FLOAT))
                idx = jnp.array(re_conf["index"], dtype=jnp.int32) - 1
                for t in range(n_terms):
                    _apply_re_offset(param, Z[:, t] * u[t][idx], param_values)

    _batched_groups = {}
    _unbatched_idxs = []
    for re_idx in independent_re_idxs:
        rc = re_configs[re_idx]
        if (int(rc["n_terms"]) == 1 and rc.get("type", "index") == "index"
                and rc.get("term_idx") is None):
            key = (rc["group"], int(rc["n_groups"]), rc.get("prior_sd", "normal(0, 0.5)"))
            _batched_groups.setdefault(key, []).append(re_idx)
        else:
            _unbatched_idxs.append(re_idx)

    for (group, n_groups, prior_sd_str), batch_idxs in _batched_groups.items():
        if len(batch_idxs) == 1:
            _sample_independent_re({**re_configs[batch_idxs[0]], "_idx": batch_idxs[0]}, param_values)
            continue

        n_batch = len(batch_idxs)
        sd_prior = _positive_dist(parse_prior(prior_sd_str))
        batch_params = [re_configs[i]["param"] for i in batch_idxs]

        batch_tag = "_".join(sorted(batch_params))
        sds = numpyro.sample(f"sigma_batch_{batch_tag}_{group}",
                              sd_prior.expand([n_batch]))
        z_all = numpyro.sample(f"z_batch_{batch_tag}_{group}",
                                dist.Normal(0, 1).expand([n_batch, n_groups]))
        z_used = z_all
        for b, re_idx in enumerate(batch_idxs):
            if ("r", re_idx) in rkg_sites:
                def _emitb(u_f, nm=f"u_{re_configs[re_idx]['param']}_{group}"):
                    numpyro.deterministic(nm, u_f[0])
                z_used = z_used.at[b].set(_rkg_hook(("r", re_idx), jnp.reshape(sds[b], (1, 1)), z_all[b][None, :], _emitb)[0])
        u_all = sds[:, None] * z_used

        for b, re_idx in enumerate(batch_idxs):
            rc = re_configs[re_idx]
            param = rc["param"]
            u = u_all[b]
            numpyro.deterministic(f"sigma_{param}_{group}[1]", sds[b])
            numpyro.deterministic(f"z_{param}_{group}[1]", z_all[b])
            if ("r", re_idx) not in rkg_sites:
                numpyro.deterministic(f"u_{param}_{group}", u)
            idx = jnp.array(rc["index"], dtype=jnp.int32) - 1
            _apply_re_offset(param, u[idx], param_values)

    for re_idx in _unbatched_idxs:
        _sample_independent_re({**re_configs[re_idx], "_idx": re_idx}, param_values)

    cross_cor_criterion_contributions = []
    cross_cor_bias_contributions = []

    for cor_id, member_idxs in cross_cor_groups.items():
        members = [re_configs[i] for i in member_idxs]
        group = members[0]["group"]
        n_groups = int(members[0]["n_groups"])

        total_dim = sum(int(m["n_terms"]) for m in members)
        all_sds = jnp.zeros(total_dim)
        dim_offset = 0
        member_dims = []

        for m in members:
            param = m["param"]
            nt = int(m["n_terms"])
            sd_prior = _positive_dist(parse_prior(m.get("prior_sd", "normal(0, 0.5)")))
            per = m.get("prior_sd_per_coef")
            if isinstance(per, str):
                per = [per]
            if per is None or len(per) != nt:
                per = [m.get("prior_sd", "normal(0, 0.5)")] * nt
            if nt == 1:
                sd_val = numpyro.sample(f"sigma_cross_{cor_id}_{group}[{dim_offset + 1}]",
                                        _positive_dist(parse_prior(per[0])))
                all_sds = all_sds.at[dim_offset].set(sd_val)
            else:
                sd_vals = _grid_sample(f"sigma_cross_{cor_id}_{group}_block_{param}", list(per), (nt,),
                                       transform=_positive_dist)
                all_sds = all_sds.at[dim_offset:dim_offset + nt].set(sd_vals)
            member_dims.append((param, nt, dim_offset, m))
            dim_offset += nt

        cor_prior_str = members[0].get("prior_cor", "lkj_corr_cholesky(1)")
        L_corr = _corr_cholesky_site(f"L_corr_cross_{cor_id}_{group}", cor_prior_str, total_dim)
        # DR-G-HMC's transport map samples the factor with the dimensions in its own order
        # (tmap.TMap.corr): L_corr is then the factor of R[p][:, p], and its rows go back to the
        # model's order here; z follows the sampled order
        cperm = (config.get("_corr_perm") or {}).get(f"cross_{cor_id}_{group}")
        back_ = _rkg_back(rkg_plan, f"cross_{cor_id}_{group}")
        if back_ is not None:
            cperm = _force_last(np.arange(total_dim) if cperm is None else cperm, back_, total_dim)
        if cperm is not None:
            L_corr = jnp.take(L_corr, jnp.argsort(cperm), axis=0)
        numpyro.deterministic(f"corr_cross_{cor_id}_{group}", L_corr @ L_corr.T)
        L = jnp.diag(all_sds) @ L_corr
        numpyro.deterministic(f"sigma_cross_{cor_id}_{group}", all_sds)

        z = numpyro.sample(f"z_cross_{cor_id}_{group}",
                           dist.Normal(0, 1).expand([total_dim, n_groups]))
        u = L @ z
        if ("x", cor_id) in rkg_sites:
            def _emit_cross(u_f, mems=[(param, nt, d_off) for param, nt, d_off, _ in member_dims], grp=group, cid_=cor_id):
                for param_, nt_, d_off_ in mems:
                    u_portion = u_f[d_off_:d_off_ + nt_]
                    numpyro.deterministic(f"u_{param_}_{grp}_from_{cid_}", u_portion[0] if nt_ == 1 else u_portion.T)
            u = L @ _rkg_hook(("x", cor_id), L, z, _emit_cross)
        for param, nt, d_off, m_conf in member_dims:
            u_portion = u[d_off:d_off + nt]
            if ("x", cor_id) not in rkg_sites:
                u_det = u_portion[0] if nt == 1 else u_portion.T
                numpyro.deterministic(f"u_{param}_{group}_from_{cor_id}", u_det)

            if param in ("criterion", "criterion2"):
                idx_crit = jnp.array(m_conf["index"], dtype=jnp.int32) - 1
                cross_cor_criterion_contributions.append((u_portion, idx_crit, m_conf))
                continue
            if param == "bias":
                cross_cor_bias_contributions.append((u_portion, jnp.array(m_conf["index"], dtype=jnp.int32) - 1, m_conf))
                continue

            re_type = m_conf.get("type", "index")
            has_term_idx = m_conf.get("term_idx") is not None
            for t in range(nt):
                u_t = u[d_off + t]
                if re_type == "index":
                    idx = jnp.array(m_conf["index"], dtype=jnp.int32) - 1
                    if has_term_idx:
                        tidx = jnp.array(m_conf["term_idx"], dtype=jnp.int32) - 1
                        offset = jnp.where(tidx == t, u_t[idx], 0.0)
                        _apply_re_offset(param, offset, param_values)
                    else:
                        _apply_re_offset(param, u_t[idx], param_values)
                else:
                    Z = _ensure_2d(jnp.array(m_conf["Z"], dtype=_FLOAT))
                    idx = jnp.array(m_conf["index"], dtype=jnp.int32) - 1
                    _apply_re_offset(param, Z[:, t] * u_t[idx], param_values)

    # ---- Smooth terms: non-centered parameterization ----
    smooth_configs = config.get("smooths", [])
    _criterion_smooth_s = []
    _criterion2_smooth_s = []
    for sm_conf in smooth_configs:
        param = sm_conf["param"]
        label = sm_conf["label"]
        k = sm_conf["component"]
        Zs = jnp.array(sm_conf["Zs"], dtype=_FLOAT)
        dim = sm_conf["dim"]
        n_thresh_sm = sm_conf.get("n_thresh", 0)
        prior_sds_str = sm_conf.get("prior_sds", "student_t(3, 0, 2.5)")

        sds_prior = parse_prior(prior_sds_str)
        if isinstance(sds_prior, (dist.Normal, dist.StudentT)):
            sds_prior = dist.FoldedDistribution(sds_prior)

        info = sm_conf.get("info")
        if info is not None:
            info = jnp.asarray(np.asarray(info, dtype=float).reshape(
                (n_thresh_sm, dim, dim) if n_thresh_sm > 0 else (dim, dim)), dtype=_FLOAT)

        if n_thresh_sm > 0:
            sds = numpyro.sample(f"sds_{param}_{label}_{k}",
                                 sds_prior.expand([n_thresh_sm]))
            zs = numpyro.sample(f"zs_{param}_{label}_{k}",
                                dist.Normal(0, 1).expand([n_thresh_sm, dim]))
            if info is not None:
                v, ld = jax.vmap(_smooth_info_scale)(sds, info, zs)
                numpyro.factor(f"smooth_info_{param}_{label}_{k}",
                               jnp.sum(ld) - jnp.sum(dist.Normal(0, 1).log_prob(zs)))
                s = numpyro.deterministic(f"s_{param}_{label}_{k}", sds[:, None] * v)
            else:
                s = numpyro.deterministic(f"s_{param}_{label}_{k}",
                                          sds[:, None] * zs)
            if param == "criterion2":
                _criterion2_smooth_s.append((Zs, s))
            else:
                _criterion_smooth_s.append((Zs, s))
        else:
            sds = numpyro.sample(f"sds_{param}_{label}_{k}", sds_prior)
            zs = numpyro.sample(f"zs_{param}_{label}_{k}",
                                dist.Normal(0, 1).expand([dim]))
            if info is not None:
                v, ld = _smooth_info_scale(sds, info, zs)
                numpyro.factor(f"smooth_info_{param}_{label}_{k}",
                               ld - jnp.sum(dist.Normal(0, 1).log_prob(zs)))
                s = numpyro.deterministic(f"s_{param}_{label}_{k}", sds * v)
            else:
                s = numpyro.deterministic(f"s_{param}_{label}_{k}", sds * zs)
            offset = Zs @ s
            _apply_re_offset(param, offset, param_values)

    # ---- Apply link functions to get parameter values ----
    trial_params = {}
    for param_name, pconf in params_config.items():
        if param_name in ("criterion", "criterion2"):
            continue
        link = pconf.get("link", "identity")
        eta = param_values.get(param_name, jnp.zeros(N))

        if link == "log":
            trial_params[param_name] = jnp.exp(eta)
        elif link == "logit":
            trial_params[param_name] = jax_expit(eta)
        elif link == "logis":
            trial_params[param_name] = jax_expit(eta)
        elif link in ("tanh", "fisherz"):
            trial_params[param_name] = jnp.tanh(eta)
        else:
            trial_params[param_name] = eta

    if family_name == "mafc":
        M = int(config["K"])
        bias_full = _mafc_bias(config, bias_re_configs, cross_cor_bias_contributions, N, M)
        log_lik = fused_loglik_mafc(y - 1, jnp.array(config["target_pos"], dtype=jnp.int32) - 1,
                                    jnp.array(config["m_alt"], dtype=jnp.int32), trial_params.get("dprime", jnp.zeros(N)),
                                    trial_params.get("sigma", jnp.ones(N)), bias_full)
    else:
        # ---- Build thresholds ----
        P_crit = config.get("P_crit", 1)
        X_crit = jnp.array(config["X_crit"], dtype=_FLOAT)
        if X_crit.ndim == 1:
            X_crit = X_crit[:, None]

        prior_thresh_mid = config.get("prior_thresh_mid", "normal(0, 1.5)")
        prior_log_gaps = config.get("prior_log_gaps", "normal(0, 0.5)")
        prior_thresh_mid2 = config.get("prior_thresh_mid2", prior_thresh_mid)
        prior_log_gaps2 = config.get("prior_log_gaps2", prior_log_gaps)
        per_gap2_priors = config.get("prior_log_gaps2_per_gap")
        has_per_gap2 = per_gap2_priors is not None and len(set(per_gap2_priors)) > 1
        n_gaps = n_thresh - 1

        def _mid_prior(j):
            return _grid_get(config, "prior_thresh_mid_grid", j, None, prior_thresh_mid)

        def _gap_prior(k, j):
            fallback = per_gap_priors[k] if per_gap_priors is not None else prior_log_gaps
            return _grid_get(config, "prior_log_gaps_grid", k, j, fallback)

        per_gap_priors = config.get("prior_log_gaps_per_gap")
        if isinstance(per_gap_priors, str):
            per_gap_priors = [per_gap_priors]
        if P_crit == 1:
            beta_thresh_mid = numpyro.sample("beta_thresh_mid[1]",
                                              parse_prior(_mid_prior(0))).reshape(1)
        else:
            beta_thresh_mid = jnp.zeros(P_crit)
            for j in range(P_crit):
                mid_j = numpyro.sample(f"beta_thresh_mid[{j+1}]",
                                       parse_prior(_mid_prior(j)))
                beta_thresh_mid = beta_thresh_mid.at[j].set(mid_j)

        crit_centre = _centre_of({"centre": config.get("crit_centre")}) if P_crit > 1 else None
        if crit_centre is not None:
            mid_c, beta_thresh_mid = beta_thresh_mid, _uncentre(beta_thresh_mid, crit_centre)
            _centre_factor("beta_thresh_mid", lambda j: parse_prior(_mid_prior(j)), crit_centre, beta_thresh_mid, mid_c)
            for j in range(P_crit):
                numpyro.deterministic(f"orig__beta_thresh_mid[{j+1}]", beta_thresh_mid[j])
        thresh_mid_vec = X_crit @ beta_thresh_mid

        gap_param_prefix = "beta_raw_gaps" if gap_link == "softplus" else "beta_log_gaps"
        beta_log_gaps = jnp.zeros((n_gaps, P_crit)) if n_gaps > 0 else None
        crit_ls = bool(config.get("crit_ls")) and n_gaps > 0
        if crit_ls:
            gap_pri = [_gap_prior(k, 0) for k in range(n_gaps)]
            if config.get("crit_ls_norm"):
                beta_log_gaps = _sum_zero_gaps(gap_param_prefix, [gap_pri])[0].reshape(n_gaps, 1)
            elif P_crit == 1 and len(set(gap_pri)) == 1:
                beta_log_gaps = numpyro.sample(gap_param_prefix,
                                               parse_prior(gap_pri[0]).expand([n_gaps])).reshape(n_gaps, 1)
            else:
                for k in range(n_gaps):
                    site = f"{gap_param_prefix}[{k+1}]" if P_crit == 1 else f"{gap_param_prefix}[{k+1},1]"
                    beta_log_gaps = beta_log_gaps.at[k, 0].set(numpyro.sample(site, parse_prior(gap_pri[k])))
            log_gaps_obs = jnp.zeros((n_gaps, N))
        elif n_gaps > 0 and config.get("crit_scalar") and P_crit > 1:
            for k in range(n_gaps):
                lg_k = numpyro.sample(f"{gap_param_prefix}[{k+1},1]", parse_prior(_gap_prior(k, 0)))
                beta_log_gaps = beta_log_gaps.at[k, 0].set(lg_k)
            log_gaps_obs = beta_log_gaps @ X_crit.T
        elif n_gaps > 0:
            gap_pri = [[_gap_prior(k, j) for j in range(P_crit)] for k in range(n_gaps)]
            uniform_gaps = len({p for row in gap_pri for p in row}) == 1
            if P_crit == 1 and uniform_gaps:
                beta_log_gaps = numpyro.sample(gap_param_prefix,
                                                parse_prior(gap_pri[0][0]).expand([n_gaps])).reshape(n_gaps, 1)
            elif P_crit == 1:
                for k in range(n_gaps):
                    row_k = numpyro.sample(f"{gap_param_prefix}[{k+1}]",
                                            parse_prior(gap_pri[k][0]))
                    beta_log_gaps = beta_log_gaps.at[k, 0].set(row_k)
            else:
                for k in range(n_gaps):
                    for j in range(P_crit):
                        lg_kj = numpyro.sample(f"{gap_param_prefix}[{k+1},{j+1}]",
                                               parse_prior(gap_pri[k][j]))
                        beta_log_gaps = beta_log_gaps.at[k, j].set(lg_kj)
                if crit_centre is not None:
                    gaps_c = beta_log_gaps
                    beta_log_gaps = jnp.stack([_uncentre(gaps_c[k], crit_centre) for k in range(n_gaps)])
                    for k in range(n_gaps):
                        _centre_factor(f"{gap_param_prefix}[{k+1}]", lambda j, k=k: parse_prior(gap_pri[k][j]), crit_centre,
                                       beta_log_gaps[k], gaps_c[k])
                        for j in range(P_crit):
                            numpyro.deterministic(f"orig__{gap_param_prefix}[{k+1},{j+1}]", beta_log_gaps[k, j])

            log_gaps_obs = beta_log_gaps @ X_crit.T
        else:
            log_gaps_obs = jnp.zeros((0, N))

        mid_idx = (n_thresh + 1) // 2 - 1

        crit_re_contributions = list(cross_cor_criterion_contributions)

        for crit_re in criterion_re_configs:
            if crit_re["param"] != "criterion":
                continue
            crit_group = crit_re["group"]
            n_groups_crit = int(crit_re["n_groups"])
            n_dims = int(crit_re["n_terms"])
            ci = next(i for i, rc in enumerate(re_configs) if rc is crit_re)
            ckey = ("r", ci)

            def _emit_crit(u_f, nm=f"u_criterion_{crit_group}"):
                numpyro.deterministic(nm, u_f.T)
            crit_hooked = ckey in rkg_sites
            crit_correlated = crit_re.get("correlated", True)
            crit_sd_prior = _positive_dist(parse_prior(crit_re.get("prior_sd", "normal(0, 0.5)")))
            per_coef_sd = crit_re.get("prior_sd_per_coef")
            has_per_coef_sd = (per_coef_sd is not None
                              and len(per_coef_sd) == n_dims
                              and len(set(per_coef_sd)) > 1)

            if crit_correlated and n_dims > 1 and crit_re.get("whiten") is not None and not crit_re.get("cor_blocks"):
                sd_list = ([_positive_dist(parse_prior(per_coef_sd[t])) for t in range(n_dims)] if has_per_coef_sd
                           else [crit_sd_prior] * n_dims)
                crit_sds, L_corr_crit, L_crit = _whitened_cov(f"criterion_{crit_group}", crit_re, n_dims, sd_list)
                numpyro.deterministic(f"corr_criterion_{crit_group}", L_corr_crit @ L_corr_crit.T)
                z_crit = numpyro.sample(f"z_criterion_{crit_group}",
                                        dist.Normal(0, 1).expand([n_dims, n_groups_crit]))
                if crit_hooked:
                    z_crit = _rkg_hook(ckey, L_crit, z_crit, _emit_crit)
                u_crit = L_crit @ z_crit
            elif crit_correlated and n_dims > 1:
                if has_per_coef_sd:
                    crit_sds_list = []
                    for t in range(n_dims):
                        sd_prior_t = _positive_dist(parse_prior(per_coef_sd[t]))
                        sd_t = numpyro.sample(f"sigma_criterion_{crit_group}[{t+1}]",
                                              sd_prior_t)
                        crit_sds_list.append(sd_t)
                    crit_sds = jnp.stack(crit_sds_list)
                else:
                    crit_sds = numpyro.sample(f"sigma_criterion_{crit_group}",
                                               crit_sd_prior.expand([n_dims]))
                L_corr_crit = _re_corr_cholesky(f"criterion_{crit_group}", crit_re, n_dims)
                numpyro.deterministic(f"corr_criterion_{crit_group}", L_corr_crit @ L_corr_crit.T)
                L_crit = jnp.diag(crit_sds) @ L_corr_crit
                z_crit = numpyro.sample(f"z_criterion_{crit_group}",
                                        dist.Normal(0, 1).expand([n_dims, n_groups_crit]))
                if crit_hooked:
                    z_crit = _rkg_hook(ckey, L_crit, z_crit, _emit_crit)
                u_crit = L_crit @ z_crit
            else:
                u_crit = jnp.zeros((n_dims, n_groups_crit))
                crit_buf = {"u": None, "n": 0, "need": sum((ckey if n_dims == 1 else ("t", ci, t)) in rkg_sites
                                                            for t in range(n_dims))}
                for t in range(n_dims):
                    sd_prior_t = (_positive_dist(parse_prior(per_coef_sd[t]))
                                  if has_per_coef_sd else crit_sd_prior)
                    sd_t = numpyro.sample(f"sigma_criterion_{crit_group}[{t+1}]", sd_prior_t)
                    z_t = numpyro.sample(f"z_criterion_{crit_group}[{t+1}]",
                                         dist.Normal(0, 1).expand([n_groups_crit]))
                    tkey = ckey if n_dims == 1 else ("t", ci, t)
                    if tkey in rkg_sites:
                        crit_hooked = True

                        def _emit_t(u_f, t_=t, buf=crit_buf, nm=f"u_criterion_{crit_group}"):
                            buf["u"] = buf["u"].at[t_].set(u_f[0])
                            buf["n"] += 1
                            if buf["n"] == buf["need"]:
                                numpyro.deterministic(nm, buf["u"].T)
                        z_t = _rkg_hook(tkey, jnp.reshape(sd_t, (1, 1)), z_t[None, :], _emit_t)[0]
                    u_crit = u_crit.at[t].set(sd_t * z_t)
                crit_buf["u"] = u_crit

            if not crit_hooked:
                numpyro.deterministic(f"u_criterion_{crit_group}", u_crit.T)
            idx_crit = jnp.array(crit_re["index"], dtype=jnp.int32) - 1
            crit_re_contributions.append((u_crit, idx_crit, crit_re))

        for u_c, idx_c, conf_c in crit_re_contributions:
            if conf_c.get("param") != "criterion":
                continue
            d_mid, d_gaps = _crit_re_shift(u_c, idx_c, conf_c, n_thresh, mid_idx, n_gaps)
            thresh_mid_vec = thresh_mid_vec + d_mid
            if d_gaps is not None:
                log_gaps_obs = log_gaps_obs + d_gaps

        for (Zs_sm, s_mat) in _criterion_smooth_s:
            thresh_mid_vec = thresh_mid_vec + Zs_sm @ s_mat[0]
            if n_gaps > 0 and not crit_ls:
                for g in range(n_gaps):
                    log_gaps_obs = log_gaps_obs.at[g].add(Zs_sm @ s_mat[1 + g])

        if crit_ls:
            thresh_mid_vec, log_gaps_obs = _location_scale_criteria(
                thresh_mid_vec, beta_log_gaps[:, 0], param_values.get("criterion_scale", jnp.zeros(N)),
                n_thresh, gap_link)

        thresh = build_thresholds_vectorized(thresh_mid_vec, log_gaps_obs, n_thresh, gap_link=gap_link)

        # ---- Compute log-likelihood ----
        is_bivariate = family_name in ("bivariate_sdt", "bivariate_dp", "vrdp2d", "bivariate_cumulative")
        is_cdp = family_name == "cdp"

        if is_cdp:
            rk = jnp.array(config["rk"], dtype=jnp.int32)
            J = int(config["J"])
            old_level_map = config["old_level_map"]
            n_rkg = int(config.get("n_rkg", 2))

            mu_R_raw = param_values.get("dprime", jnp.zeros(N))
            mu_F_raw = param_values.get("dprime2", jnp.zeros(N))
            sigma_R_raw = trial_params.get("sigma", jnp.ones(N))
            sigma_F_raw = trial_params.get("sigma2", jnp.ones(N))

            mu_R = jnp.where(is_old == 1, mu_R_raw, 0.0)
            mu_F = jnp.where(is_old == 1, mu_F_raw, 0.0)
            sigma_R = jnp.where(is_old == 1, sigma_R_raw, 1.0)
            sigma_F = jnp.where(is_old == 1, sigma_F_raw, 1.0)

            c_R = param_values.get("rec_crit", jnp.full(N, 1.5))
            c_K = param_values.get("know_crit", jnp.zeros(N)) if n_rkg == 3 else jnp.zeros(N)
            cdp_corr = None
            if "rho" in trial_params or "rho_N" in trial_params:
                rho_t = trial_params.get("rho", jnp.zeros(N))
                rho_l = trial_params.get("rho_N", rho_t if config.get("cdp_rho_shared") else jnp.zeros(N))
                cdp_corr = jnp.where(is_old == 1, rho_t, rho_l)
            if "sites" in knob_state:
                pl = rkg_plan
                lidx = jnp.asarray(pl["lvl_idx"])
                WR = jnp.asarray(pl["WR"], dtype=_FLOAT)
                WK = jnp.asarray(pl["WK"], dtype=_FLOAT)
                has_c = bool(np.any(pl["sgn"] < 0))
                WC = jnp.asarray(pl["WC"], dtype=_FLOAT)
                sites = knob_state["sites"]
                order = list(sites)
                zeta = jnp.concatenate([sites[k]["zeta"] for k in order], 0)
                Mbig = jnp.zeros((pl["mm"], zeta.shape[0]), dtype=_FLOAT)
                o = 0
                for k in order:
                    m_k = sites[k]["zeta"].shape[0]
                    Mbig = Mbig.at[jnp.asarray(sites[k]["terms"]), o:o + m_k].set(sites[k]["M"])
                    o += m_k
                if has_c:
                    Au = jnp.stack([jnp.sum(jnp.asarray(pl["sgn"][pos], dtype=_FLOAT)[:, None] * Mbig[jnp.asarray(pos)], 0)
                                    for pos in pl["unit_pos"]], 1)
                else:
                    Au = jnp.stack([jnp.sum(Mbig[jnp.asarray(pos)], 0) for pos in pl["unit_pos"]], 1)
                Qu, Ru = jnp.linalg.qr(Au)
                sg = jnp.where(jnp.diagonal(Ru) < 0, -1.0, 1.0)
                Du = Qu * sg[None, :]
                coefs = [jnp.abs(Ru[t, t]) for t in range(len(pl["unit_pos"]))]
                S = Du.T @ zeta
                rest = zeta - Du @ S
                uB = Mbig @ rest
                c_R = c_R + jnp.sum(WR * uB[:, lidx].T, axis=1)
                c_K = c_K + jnp.sum(WK * uB[:, lidx].T, axis=1)
                if has_c:
                    d_c = jnp.sum(WC * uB[:, lidx].T, axis=1)
                    thresh = thresh + d_c[:, None]
                    thresh_mid_vec = thresh_mid_vec + d_c
                knob_state["level"] = dict(coef=coefs)
                numpyro.deterministic("kc_slack_coef", jnp.stack(coefs))
                if knob is not None and knob["soft"]:
                    c_K = _rkg_shift(knob, knob_state, pl, thresh, c_R, c_K)
                jac = 0.0
                gaps = []
                for t in range(len(pl["unit_pos"])):
                    Gj = jnp.asarray(pl["G_t"][t])
                    msk = jnp.asarray(pl["masks"][t])
                    xs = pl["x_t"][t]
                    bnd = (thresh[Gj, jnp.asarray(pl["band_t"][t])] - c_R[Gj] - c_K[Gj]) / coefs[t] / jnp.asarray(xs, dtype=_FLOAT)
                    if not pl["neg_m"][t].any():
                        low = jax.ops.segment_max(bnd, lidx[Gj], num_segments=pl["n_groups"])
                        low = jnp.where(msk, low, 0.0)
                        jac = jac + jnp.sum(jnp.where(msk, jax.scipy.special.log_ndtr(-low), 0.0))
                        z_eff = jnp.where(msk, _trunc_quantile(low, S[t]), S[t])
                        gaps.append(jnp.min(jnp.where(msk, z_eff - low, jnp.inf)))
                    else:
                        lo = jax.ops.segment_max(jnp.where(xs > 0, bnd, -jnp.inf), lidx[Gj], num_segments=pl["n_groups"])
                        hi = jax.ops.segment_min(jnp.where(xs < 0, bnd, jnp.inf), lidx[Gj], num_segments=pl["n_groups"])
                        lo = jnp.where(jnp.asarray(pl["pos_m"][t]), lo, -jnp.inf)
                        hi = jnp.where(jnp.asarray(pl["neg_m"][t]), hi, jnp.inf)
                        mixed = jnp.asarray(pl["pos_m"][t] & pl["neg_m"][t])
                        one = msk & ~mixed
                        lo_n = jnp.where(one, lo, 0.0)
                        hi_n = jnp.where(one, hi, jnp.inf)
                        logw = jnp.zeros(pl["n_groups"], dtype=_FLOAT)
                        if pl["pairs"][t] is not None:
                            fl = knob_state["floor"][t]
                            pp_, qq_ = pl["pairs"][t]
                            xp = np.asarray(xs[pp_], dtype=float)
                            xq = -np.asarray(xs[qq_], dtype=float)
                            d = fl["lam"] - fl["g"]
                            logd = jnp.where(d > 0, jnp.log(jnp.where(d > 0, d, 1.0)), -jnp.inf)
                            lk = jnp.asarray(np.log(xp + xq) - np.log(xp) - np.log(xq), dtype=_FLOAT) - jnp.log(coefs[t])
                            lvp = jnp.asarray(pl["lvl_idx"][pl["G_t"][t][pp_]])
                            logw = jax.ops.segment_min(lk + jnp.logaddexp(fl["log_gap"], logd), lvp, num_segments=pl["n_groups"])
                        lo_m = jnp.where(mixed, lo, 0.0)
                        logw = jnp.where(mixed, logw, 0.0)
                        jac = jac + jnp.sum(jnp.where(one, _trunc_logw(lo_n, hi_n), 0.0)) + jnp.sum(jnp.where(mixed, _trunc_lm(lo_m, logw), 0.0))
                        z_eff = jnp.where(one, _trunc2_quantile(lo_n, hi_n, S[t]), jnp.where(mixed, _trunc2w_quantile(lo_m, logw, S[t]), S[t]))
                        hi_g = jnp.where(mixed, lo_m + jnp.exp(logw), hi_n)
                        lo_g = jnp.where(mixed, lo_m, lo_n)
                        gaps.append(jnp.min(jnp.where(msk, jnp.minimum(z_eff - lo_g, hi_g - z_eff), jnp.inf)))
                    du = (Mbig @ Du[:, t])[:, None] * z_eff[None, :]
                    uB = uB + du
                    dut = du[:, lidx].T
                    c_R = c_R + jnp.sum(WR * dut, axis=1)
                    c_K = c_K + jnp.sum(WK * dut, axis=1)
                    if has_c:
                        d_c = jnp.sum(WC * dut, axis=1)
                        thresh = thresh + d_c[:, None]
                        thresh_mid_vec = thresh_mid_vec + d_c
                numpyro.factor("rkg_level_jac", jac)
                numpyro.deterministic("kc_level_gap", jnp.min(jnp.stack(gaps)))
                for k in order:
                    st_ = sites[k]
                    st_["emit"]((st_["A"] @ st_["z0"]).at[st_["rows"]].add(uB[jnp.asarray(st_["terms"])]))
            elif knob is not None and knob["soft"]:
                c_K = _rkg_shift(knob, knob_state, rkg_plan, thresh, c_R, c_K)
            if knob is not None and not knob["soft"]:
                G = jnp.asarray(knob["G"])
                bound = thresh[G, jnp.asarray(knob["band"])] - c_R[G] - c_K[G]
                slack = numpyro.sample("know_crit_slack", dist.ImproperUniform(dist.constraints.real, (), ()))
                t = jnp.max(bound) + jnp.exp(slack)
                numpyro.deterministic("kc_argmax", jnp.argmax(bound))
                numpyro.deterministic("kc_maxbound", jnp.max(bound))
                numpyro.deterministic("kc_gap12", jnp.max(bound) - jnp.sort(bound)[-2])
                numpyro.factor("know_crit_slack_jac", slack)
                v = jnp.asarray(knob["v"], dtype=_FLOAT)
                beta_kc = knob_state["beta_perp"] + t * v
                prior_strs = knob_state["prior_strs"]
                numpyro.factor("know_crit_prior", sum(parse_prior(prior_strs[j]).log_prob(beta_kc[j])
                                                      for j in range(knob_state["n_coef"])))
                if knob_state["n_coef"] == 1:
                    numpyro.deterministic("beta_know_crit[1]", beta_kc[0])
                elif len(set(prior_strs)) == 1:
                    numpyro.deterministic("beta_know_crit", beta_kc)
                else:
                    for j in range(knob_state["n_coef"]):
                        numpyro.deterministic(f"beta_know_crit[{j+1}]", beta_kc[j])
                c_K = c_K + t * (knob_state["X"] @ v)

            log_lik = cdp_loglik(
                    y=y, rk=rk, is_old=is_old, thresh=thresh,
                    mu_R=mu_R, sigma_R=sigma_R, mu_F=mu_F, sigma_F=sigma_F,
                    c_R=c_R, c_K=c_K, J=J, old_level_map=old_level_map,
                    n_rkg=n_rkg,
                    thresh_mid=thresh_mid_vec, log_gaps=log_gaps_obs,
                    gap_link=gap_link, rk_static=np.asarray(config.get("_static", config)["rk"]),
                    part=(config["cdp_u"], config["cdp_j"]) if config.get("cdp_u") is not None else None,
                    corr=cdp_corr,
                )

        elif family_name == "metadt":
            ones = jnp.ones(N, dtype=_FLOAT)
            m_old = trial_params.get("mratio", ones)
            m_new = m_old if config.get("metad_shared") else trial_params.get("mratio_N", ones)
            log_lik = fused_loglik_metad(y, int(config["K"]), param_values.get("dprime", jnp.zeros(N)), m_old, m_new,
                                         is_old, thresh_mid_vec, log_gaps_obs, gap_link,
                                         q=float(config["is_old_mid"]), relative=bool(config.get("metad_relative")),
                                         link_name=link_name)

        elif is_bivariate:
            y2 = jnp.array(config["y2"], dtype=jnp.int32)
            item_type = jnp.array(config["item_type"], dtype=jnp.int32)

            K2 = config.get("K2", K)
            n_thresh2 = K2 - 1
            varying_source = config.get("varying_source_criteria", False)

            # -- Sample criterion2 REs first (needed for threshold construction) --
            crit2_re_list = [rc for rc in criterion_re_configs if rc["param"] == "criterion2"]
            varying_re_mode = config.get("crit2_re_mode", "full")
            crit2_ls = bool(config.get("crit2_ls")) and K2 - 2 > 0
            s2_eta = param_values.get("criterion2_scale", jnp.zeros(N))
            c2_terms = [(u, i, c) for (u, i, c) in crit_re_contributions
                        if c.get("param") == "criterion2"]

            for crit2_re in crit2_re_list:
                c2_group = crit2_re["group"]
                c2_n_groups = int(crit2_re["n_groups"])
                c2_n_dims = int(crit2_re["n_terms"])
                c2_correlated = crit2_re.get("correlated", True)
                c2_sd_prior = _positive_dist(parse_prior(crit2_re.get("prior_sd", "normal(0, 0.5)")))
                c2_per_coef = crit2_re.get("prior_sd_per_coef")
                if c2_per_coef is not None and len(c2_per_coef) == c2_n_dims:
                    c2_sd_priors = [_positive_dist(parse_prior(p)) for p in c2_per_coef]
                else:
                    c2_sd_priors = [c2_sd_prior] * c2_n_dims

                if c2_correlated and c2_n_dims > 1:
                    c2_sds = jnp.zeros(c2_n_dims)
                    for t in range(c2_n_dims):
                        sd_t = numpyro.sample(f"sigma_criterion2_{c2_group}[{t+1}]", c2_sd_priors[t])
                        c2_sds = c2_sds.at[t].set(sd_t)
                    L_corr_c2 = _re_corr_cholesky(f"criterion2_{c2_group}", crit2_re, c2_n_dims)
                    numpyro.deterministic(f"corr_criterion2_{c2_group}", L_corr_c2 @ L_corr_c2.T)
                    L_c2 = jnp.diag(c2_sds) @ L_corr_c2
                    z_c2 = numpyro.sample(f"z_criterion2_{c2_group}",
                                          dist.Normal(0, 1).expand([c2_n_dims, c2_n_groups]))
                    u_c2 = L_c2 @ z_c2
                else:
                    u_c2 = jnp.zeros((c2_n_dims, c2_n_groups))
                    for t in range(c2_n_dims):
                        sd_t = numpyro.sample(f"sigma_criterion2_{c2_group}[{t+1}]", c2_sd_priors[t])
                        z_t = numpyro.sample(f"z_criterion2_{c2_group}[{t+1}]",
                                             dist.Normal(0, 1).expand([c2_n_groups]))
                        u_c2 = u_c2.at[t].set(sd_t * z_t)

                numpyro.deterministic(f"u_criterion2_{c2_group}",
                                      u_c2[0] if c2_n_dims == 1 else u_c2.T)
                c2_terms.append((u_c2, jnp.array(crit2_re["index"], dtype=jnp.int32) - 1, crit2_re))

            # -- Build source dimension thresholds --
            n_gaps2 = n_thresh2 - 1
            mid2 = (n_thresh2 + 1) // 2 - 1
            _t2_mid_fused = None
            _t2_gaps_fused = None

            if varying_source:
                pv2 = config.get("prior_vary2") or {}
                vbin = config.get("vbin")
                n_vbin = int(config.get("n_vbin", K))
                n_vcell = int(config.get("n_vcell", 1))
                n_vset = n_vbin * n_vcell
                v2_mid = pv2.get("mid", prior_thresh_mid2)
                v2_mid = [v2_mid] * n_vset if isinstance(v2_mid, str) else list(v2_mid)
                v2_gaps = pv2.get("gaps", [[prior_log_gaps2] * n_gaps2] * n_vset)
                v2_gaps = [[g] if isinstance(g, str) else list(g) for g in v2_gaps]
                beta_thresh_mid_2 = _grid_sample("beta_thresh_mid_2_varying", v2_mid, (n_vset,))

                if n_gaps2 > 0:
                    gap2_name = "beta_raw_gaps_2_varying" if gap_link == "softplus" else "beta_log_gaps_2_varying"
                    if config.get("crit2_ls_norm"):
                        pri_rows = [list(r) * n_gaps2 if len(r) == 1 else list(r) for r in v2_gaps]
                        pri_rows = pri_rows * n_vset if len(pri_rows) == 1 else pri_rows
                        beta_log_gaps_2 = _sum_zero_gaps(gap2_name, pri_rows, rows=True)
                    else:
                        beta_log_gaps_2 = _grid_sample(gap2_name, v2_gaps, (n_vset, n_gaps2))
                else:
                    beta_log_gaps_2 = None

                y1_idx = y - 1 if vbin is None else jnp.asarray(np.asarray(vbin, dtype=np.int32).reshape(-1) - 1)[y - 1]
                set_idx = y1_idx
                if n_vcell > 1:
                    set_idx = y1_idx + n_vbin * (jnp.asarray(np.asarray(config["vcell"], dtype=np.int32).reshape(-1)) - 1)

                thresh2_mid_per_obs = beta_thresh_mid_2[set_idx]
                if beta_log_gaps_2 is not None:
                    log_gaps2_per_obs = beta_log_gaps_2[set_idx].T
                else:
                    log_gaps2_per_obs = jnp.zeros((0, N))

                cov2 = config.get("crit2_cov")
                if cov2 is not None:
                    Xc2 = _ensure_2d(jnp.array(cov2["X"], dtype=_FLOAT))
                    P_c2 = Xc2.shape[1]
                    b_mid_cov = _grid_sample("beta_thresh_mid_2_cov",
                                             pv2.get("cov_mid", [prior_thresh_mid2] * P_c2), (P_c2,))
                    thresh2_mid_per_obs = thresh2_mid_per_obs + Xc2 @ b_mid_cov
                    if n_gaps2 > 0 and not crit2_ls:
                        gcov_name = "beta_raw_gaps_2_cov" if gap_link == "softplus" else "beta_log_gaps_2_cov"
                        b_gaps_cov = _grid_sample(gcov_name,
                                                  pv2.get("cov_gaps", [[prior_log_gaps2] * P_c2] * n_gaps2),
                                                  (n_gaps2, P_c2))
                        log_gaps2_per_obs = log_gaps2_per_obs + b_gaps_cov @ Xc2.T

                for (Zs_sm2, s_mat2) in _criterion2_smooth_s:
                    thresh2_mid_per_obs = thresh2_mid_per_obs + Zs_sm2 @ s_mat2[0]
                    if not crit2_ls:
                        for g in range(n_gaps2):
                            log_gaps2_per_obs = log_gaps2_per_obs.at[g].add(Zs_sm2 @ s_mat2[1 + g])

                for u_c2, idx_c2, c2_conf in c2_terms:
                    d_mid, d_gaps = _crit2_varying_shift(u_c2, idx_c2, varying_re_mode, y1_idx,
                                                         n_thresh2, mid2, n_gaps2, c2_conf)
                    thresh2_mid_per_obs = thresh2_mid_per_obs + d_mid
                    if d_gaps is not None:
                        log_gaps2_per_obs = log_gaps2_per_obs + d_gaps

                if crit2_ls:
                    thresh2_mid_per_obs, log_gaps2_per_obs = _location_scale_criteria(
                        thresh2_mid_per_obs, log_gaps2_per_obs, s2_eta, n_thresh2, gap_link)

                thresh2 = build_thresholds_vectorized(thresh2_mid_per_obs, log_gaps2_per_obs, n_thresh2, gap_link=gap_link)
                _t2_mid_fused = thresh2_mid_per_obs
                _t2_gaps_fused = log_gaps2_per_obs

            elif "criterion2" in params_config:
                c2_conf = params_config["criterion2"]
                c2_n_coef = c2_conf["n_coef"]
                X_c2 = _ensure_2d(jnp.array(c2_conf["X"], dtype=_FLOAT))
                c2_col_names = c2_conf.get("col_names", [f"V{i+1}" for i in range(c2_n_coef)])
                if isinstance(c2_col_names, str):
                    c2_col_names = [c2_col_names]

                beta_thresh2_mid = jnp.zeros(c2_n_coef)
                for j in range(c2_n_coef):
                    prior_str = _grid_get(config, "prior_thresh_mid2_grid", j, None, prior_thresh_mid2)
                    t2m_j = numpyro.sample(f"beta_thresh_mid_2[{j+1}]",
                                           parse_prior(prior_str))
                    beta_thresh2_mid = beta_thresh2_mid.at[j].set(t2m_j)

                c2_centre = _centre_of(c2_conf)
                if c2_centre is not None:
                    mid2_c, beta_thresh2_mid = beta_thresh2_mid, _uncentre(beta_thresh2_mid, c2_centre)
                    _centre_factor("beta_thresh_mid_2", lambda j: parse_prior(_grid_get(config, "prior_thresh_mid2_grid", j, None, prior_thresh_mid2)),
                                   c2_centre, beta_thresh2_mid, mid2_c)
                    for j in range(c2_n_coef):
                        numpyro.deterministic(f"orig__beta_thresh_mid_2[{j+1}]", beta_thresh2_mid[j])
                thresh2_mid_vec = X_c2 @ beta_thresh2_mid

                if n_gaps2 > 0:
                    gap2_name_ij = "beta_raw_gaps_2" if gap_link == "softplus" else "beta_log_gaps_2"
                    g2_pri = [[_grid_get(config, "prior_log_gaps2_grid", k, j,
                                         per_gap2_priors[k] if has_per_gap2 else prior_log_gaps2)
                               for j in range(c2_n_coef)] for k in range(n_gaps2)]
                    g2_uniform = len({p for row in g2_pri for p in row}) == 1
                    if crit2_ls and config.get("crit2_ls_norm"):
                        beta_log_gaps2 = jnp.zeros((n_gaps2, c2_n_coef)).at[:, 0].set(
                            _sum_zero_gaps(gap2_name_ij, [[g2_pri[k][0] for k in range(n_gaps2)]])[0])
                    elif crit2_ls:
                        beta_log_gaps2 = jnp.zeros((n_gaps2, c2_n_coef))
                        for k in range(n_gaps2):
                            site = f"{gap2_name_ij}[{k+1}]" if c2_n_coef == 1 else f"{gap2_name_ij}[{k+1},1]"
                            beta_log_gaps2 = beta_log_gaps2.at[k, 0].set(numpyro.sample(site, parse_prior(g2_pri[k][0])))
                    elif c2_n_coef == 1 and not g2_uniform:
                        beta_log_gaps2 = jnp.stack([
                            numpyro.sample(f"{gap2_name_ij}[{k+1}]", parse_prior(g2_pri[k][0]))
                            for k in range(n_gaps2)]).reshape(n_gaps2, 1)
                    elif c2_n_coef == 1:
                        beta_log_gaps2 = numpyro.sample(gap2_name_ij,
                                                         parse_prior(g2_pri[0][0]).expand([n_gaps2])).reshape(n_gaps2, 1)
                    else:
                        beta_log_gaps2 = jnp.zeros((n_gaps2, c2_n_coef))
                        for k in range(n_gaps2):
                            for j in range(c2_n_coef):
                                lg2_kj = numpyro.sample(f"{gap2_name_ij}[{k+1},{j+1}]",
                                                        parse_prior(g2_pri[k][j]))
                                beta_log_gaps2 = beta_log_gaps2.at[k, j].set(lg2_kj)
                        if c2_centre is not None:
                            g2_c = beta_log_gaps2
                            beta_log_gaps2 = jnp.stack([_uncentre(g2_c[k], c2_centre) for k in range(n_gaps2)])
                            for k in range(n_gaps2):
                                _centre_factor(f"{gap2_name_ij}[{k+1}]", lambda j, k=k: parse_prior(g2_pri[k][j]), c2_centre,
                                               beta_log_gaps2[k], g2_c[k])
                                for j in range(c2_n_coef):
                                    numpyro.deterministic(f"orig__{gap2_name_ij}[{k+1},{j+1}]", beta_log_gaps2[k, j])
                    log_gaps2_obs = beta_log_gaps2 @ X_c2.T
                else:
                    log_gaps2_obs = jnp.zeros((0, N))

                for u_c2, idx_c2, conf_c2 in c2_terms:
                    d_mid, d_gaps = _crit_re_shift(u_c2, idx_c2, conf_c2, n_thresh2, mid2, n_gaps2)
                    thresh2_mid_vec = thresh2_mid_vec + d_mid
                    if d_gaps is not None:
                        log_gaps2_obs = log_gaps2_obs + d_gaps

                for (Zs_sm2, s_mat2) in _criterion2_smooth_s:
                    thresh2_mid_vec = thresh2_mid_vec + Zs_sm2 @ s_mat2[0]
                    if n_gaps2 > 0 and not crit2_ls:
                        for g in range(n_gaps2):
                            log_gaps2_obs = log_gaps2_obs.at[g].add(Zs_sm2 @ s_mat2[1 + g])

                if crit2_ls:
                    thresh2_mid_vec, log_gaps2_obs = _location_scale_criteria(
                        thresh2_mid_vec, beta_log_gaps2[:, 0], s2_eta, n_thresh2, gap_link)

                thresh2 = build_thresholds_vectorized(thresh2_mid_vec, log_gaps2_obs, n_thresh2, gap_link=gap_link)
                _t2_mid_fused = thresh2_mid_vec
                _t2_gaps_fused = log_gaps2_obs

            else:
                thresh2_mid = numpyro.sample("beta_thresh_mid_2[1]",
                                             parse_prior(_grid_get(config, "prior_thresh_mid2_grid", 0, None,
                                                                   prior_thresh_mid2)))
                if n_gaps2 > 0:
                    gap2_name_int = "beta_raw_gaps_2" if gap_link == "softplus" else "beta_log_gaps_2"
                    g2i_pri = [_grid_get(config, "prior_log_gaps2_grid", k, 0,
                                         per_gap2_priors[k] if has_per_gap2 else prior_log_gaps2)
                               for k in range(n_gaps2)]
                    if len(set(g2i_pri)) > 1:
                        log_gaps2 = jnp.stack([
                            numpyro.sample(f"{gap2_name_int}[{k+1}]", parse_prior(g2i_pri[k]))
                            for k in range(n_gaps2)])
                    else:
                        log_gaps2 = numpyro.sample(gap2_name_int,
                                                    parse_prior(g2i_pri[0]).expand([n_gaps2]))
                else:
                    log_gaps2 = jnp.zeros(n_gaps2)

                thresh2_single = build_thresholds(thresh2_mid, log_gaps2, n_thresh2, gap_link=gap_link)
                thresh2 = jnp.broadcast_to(thresh2_single[None, :], (N, n_thresh2))

                if c2_terms or _criterion2_smooth_s or crit2_ls:
                    thresh2_mid_vec_re = jnp.broadcast_to(thresh2_mid, (N,))
                    log_gaps2_obs_re = jnp.broadcast_to(log_gaps2[:, None], (n_gaps2, N)) if n_gaps2 > 0 else jnp.zeros((0, N))
                    for u_c2, idx_c2, conf_c2 in c2_terms:
                        d_mid, d_gaps = _crit_re_shift(u_c2, idx_c2, conf_c2, n_thresh2, mid2, n_gaps2)
                        thresh2_mid_vec_re = thresh2_mid_vec_re + d_mid
                        if d_gaps is not None:
                            log_gaps2_obs_re = log_gaps2_obs_re + d_gaps
                    for (Zs_sm2, s_mat2) in _criterion2_smooth_s:
                        thresh2_mid_vec_re = thresh2_mid_vec_re + Zs_sm2 @ s_mat2[0]
                        if n_gaps2 > 0 and not crit2_ls:
                            for g in range(n_gaps2):
                                log_gaps2_obs_re = log_gaps2_obs_re.at[g].add(Zs_sm2 @ s_mat2[1 + g])
                    if crit2_ls:
                        thresh2_mid_vec_re, log_gaps2_obs_re = _location_scale_criteria(
                            thresh2_mid_vec_re, log_gaps2, s2_eta, n_thresh2, gap_link)
                    thresh2 = build_thresholds_vectorized(thresh2_mid_vec_re, log_gaps2_obs_re, n_thresh2, gap_link=gap_link)
                    _t2_mid_fused = thresh2_mid_vec_re
                    _t2_gaps_fused = log_gaps2_obs_re
                else:
                    _t2_mid_fused = jnp.broadcast_to(thresh2_mid, (N,))
                    _t2_gaps_fused = jnp.broadcast_to(log_gaps2[:, None], (n_gaps2, N)) if n_gaps2 > 0 else jnp.zeros((0, N))

            log_lik = bivariate_loglik(
                y, y2,
                src_obs=jnp.asarray(config["src_obs"], dtype=jnp.int32) if config.get("src_obs") is not None else None,
                dprime=trial_params.get("dprime", jnp.zeros(N)),
                dprime_B=trial_params.get("dprime_B"),
                thresh1=thresh, thresh2=thresh2,
                item_type=item_type,
                link_name=link_name,
                family_name=family_name,
                sigma=trial_params.get("sigma"),
                sigma2=trial_params.get("sigma2"),
                sigma_B=trial_params.get("sigma_B"),
                sigma2_B=trial_params.get("sigma2_B"),
                rho=trial_params.get("rho"),
                rho_B=trial_params.get("rho_B"),
                rho_N=trial_params.get("rho_N"),
                discrim=trial_params.get("discrim"),
                discrim_B=trial_params.get("discrim_B"),
                lambda_val=trial_params.get("lambda"),
                lambda2=trial_params.get("lambda2"),
                lambda_B=trial_params.get("lambda_B"),
                lambda2_B=trial_params.get("lambda2_B"),
                dprime2=trial_params.get("dprime2"),
                bounded=config.get("bounded", False),
                part=_bdp_part(config),
                thresh1_mid=thresh_mid_vec,
                log_gaps1=log_gaps_obs,
                thresh2_mid=_t2_mid_fused,
                log_gaps2=_t2_gaps_fused,
                gap_link=gap_link,
            )
        else:
            source = jnp.array(config["source"], dtype=jnp.int32) if config.get("source") is not None else None

            log_lik = univariate_loglik(
                y=y,
                dprime=trial_params.get("dprime", jnp.zeros(N)),
                thresh=thresh,
                is_old=is_old,
                link_name=link_name,
                family_name=family_name,
                sigma=trial_params.get("sigma"),
                lambda_val=trial_params.get("lambda"),
                dprime2=trial_params.get("dprime2"),
                sigma2=trial_params.get("sigma2"),
                lambda_B=trial_params.get("lambda_B"),
                dprime_B=trial_params.get("dprime_B"),
                dprime_L=trial_params.get("dprime_L"),
                sigma_L=trial_params.get("sigma_L"),
                lambda_L=trial_params.get("lambda_L"),
                source=source,
                thresh_mid=thresh_mid_vec,
                log_gaps=log_gaps_obs,
                gap_link=gap_link,
                is_old_static=None if config.get("is_old") is None
                else np.asarray(config.get("_static", config)["is_old"], dtype=float),
                sigma_meta=trial_params.get("sigma_meta"),
                meta_noise=config.get("meta_noise"),
            )

    numpyro.deterministic("log_lik", log_lik)
    if counts is not None:
        log_lik = log_lik * counts

    if config.get("prior_only"):
        return
    ll_offset = config.get("_ll_offset")
    numpyro.factor("obs", jnp.sum(log_lik) if ll_offset is None else jnp.sum(log_lik - ll_offset))


# =============================================================================
# Entry Point
# =============================================================================

def _init_site_value(name, shape, vals):
    if name in vals:
        return name, np.asarray(vals[name], dtype=float).reshape(shape)
    m = _re.fullmatch(r"(.+)\[([0-9,]+)\]", name)
    if m is None or m.group(1) not in vals:
        return None, None
    arr = np.asarray(vals[m.group(1)], dtype=float)
    idx = tuple(int(i) - 1 for i in m.group(2).split(","))
    if arr.ndim == 0 and idx == (0,):
        return m.group(1), arr.reshape(shape)
    if arr.ndim < len(idx):
        raise ValueError(f"init value '{m.group(1)}' has {arr.ndim} dimension(s) but "
                         f"site '{name}' needs {len(idx)}.")
    return m.group(1), np.asarray(arr[idx]).reshape(shape)


def _knob_init_beta(vals, knob):
    d = knob["X"].shape[1]
    if "beta_know_crit" in vals:
        return np.asarray(vals["beta_know_crit"], dtype=float).reshape(d)
    if all(f"beta_know_crit[{j+1}]" in vals for j in range(d)):
        return np.asarray([vals[f"beta_know_crit[{j+1}]"] for j in range(d)], dtype=float)
    return None


def _knob_init_slack(config, knob, z, beta):
    """Slack reproducing the user's know-criterion init: with slack 0 the model sets t to the
    Guess bound + 1, so the bound is read back and the user's t = beta . v / v . v is
    placed above it. An init where some Guess response is impossible keeps slack 0."""
    from numpyro.infer.util import constrain_fn
    v = knob["v"]
    z0 = dict(z)
    z0["know_crit_slack"] = jnp.asarray(0.0)
    out = constrain_fn(sdt_model, (config,), {}, z0, return_deterministic=True)
    d = knob["X"].shape[1]
    if d == 1:
        b0 = np.asarray([out["beta_know_crit[1]"]], dtype=float)
    elif "beta_know_crit" in out:
        b0 = np.asarray(out["beta_know_crit"], dtype=float)
    else:
        b0 = np.asarray([out[f"beta_know_crit[{j+1}]"] for j in range(d)], dtype=float)
    vv = float(v @ v)
    bound = float(b0 @ v) / vv - 1.0
    t_user = float(beta @ v) / vv
    if t_user > bound:
        return jnp.asarray(np.log(t_user - bound))
    print("[bayesroc] the know_crit init makes an observed Guess response impossible; its slack "
          "starts at the boundary + 1 instead", file=sys.stderr, flush=True)
    return jnp.asarray(0.0)


def _knob_init_shift(config, knob, z, beta):
    """Shift coordinates reproducing the user's know-criterion init under the soft wall: each unit's
    t solves V t = beta - B_perp B_perp' beta, and w = lam + x with x the root of
    r x + (1 - r) s softplus(x / s) = t - lam (increasing in x). A t at or below the wall when r = 0
    starts just above it instead."""
    from numpyro.infer.util import constrain_fn
    V, B = np.asarray(knob["V"], dtype=float), np.asarray(knob["B_perp"], dtype=float)
    z0 = dict(z)
    z0["know_crit_shift"] = jnp.zeros(V.shape[1])
    out = constrain_fn(sdt_model, (config,), {}, z0, return_deterministic=True)
    lam, r = np.asarray(out["kc_shift_lam"], dtype=float), np.asarray(out["kc_shift_r"], dtype=float)
    t = np.linalg.lstsq(V, beta - B @ (B.T @ beta), rcond=None)[0]
    w = np.zeros(V.shape[1])
    for j in range(V.shape[1]):
        sc, g = float(knob["s_t"][j]), float(t[j] - lam[j])
        f = lambda x: r[j] * x + (1.0 - r[j]) * sc * np.logaddexp(0.0, x / sc) - g
        if r[j] <= 0 and g <= 0:
            print("[bayesroc] the know_crit init makes an observed Guess response impossible; it starts "
                  "just above the boundary instead", file=sys.stderr, flush=True)
            w[j] = lam[j]
            continue
        lo, hi = -1.0, 1.0
        while f(lo) > 0:
            lo *= 2.0
        while f(hi) < 0:
            hi *= 2.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
        w[j] = lam[j] + 0.5 * (lo + hi)
    return jnp.asarray(w)


def _init_given_else_uniform_site(site=None, values=None, radius=2.0):
    if site["type"] == "sample" and not site["is_observed"] and site["name"] in values:
        return values[site["name"]]
    return init_to_uniform(site, radius=radius)


def _init_given_else_uniform(values, radius):
    return partial(_init_given_else_uniform_site, values=values, radius=radius)


def init_values_to_z(config, init_values, seed, init_radius=2.0):
    """Per-chain user init lists (Stan parameter names, constrained scale) ->
    stacked unconstrained NumPyro params. Sites a list leaves out start from
    uniform(-init_radius, init_radius), as Stan does."""
    from numpyro import handlers
    from numpyro.infer.util import initialize_model, potential_energy

    improper = lambda site: (jnp.zeros(site["fn"].shape())
                             if site.get("type") == "sample" and isinstance(site.get("fn"), dist.ImproperUniform)
                             else None)
    tr = handlers.trace(handlers.seed(handlers.substitute(sdt_model, substitute_fn=improper),
                                      random.PRNGKey(0))).get_trace(config)
    sites = {k: np.shape(v["value"]) for k, v in tr.items()
             if v["type"] == "sample" and not v["is_observed"]}
    base = random.PRNGKey(int(seed) + 7919)
    zs, used = [], set()
    for c, vals in enumerate(init_values):
        vals = _centre_init(dict(vals or {}), config)
        fixed = {}
        for name, shape in sites.items():
            key, v = _init_site_value(name, shape, vals)
            if key is not None:
                fixed[name] = jnp.asarray(v)
                used.add(key)
        knob = _rkg_knob(config)
        kc = None
        if knob is not None:
            kc = _knob_init_beta(vals, knob)
            if kc is not None:
                used.update(k for k in vals if k == "beta_know_crit" or k.startswith("beta_know_crit["))
                if knob["B_perp"].shape[1] > 0:
                    fixed["know_crit_free"] = jnp.asarray(knob["B_perp"].T @ kc)
                if knob["soft"]:
                    fixed["know_crit_shift"] = jnp.zeros(np.asarray(knob["V"]).shape[1])
                else:
                    fixed["know_crit_slack"] = jnp.asarray(0.0)
        for attempt in range(100):
            st = initialize_model(random.fold_in(random.fold_in(base, c), attempt), sdt_model,
                                  init_strategy=_init_given_else_uniform(fixed, float(init_radius)),
                                  model_args=(config,))
            z = dict(st.param_info.z)
            if kc is None:
                break
            if knob["soft"]:
                z["know_crit_shift"] = _knob_init_shift(config, knob, z, kc)
            else:
                z["know_crit_slack"] = _knob_init_slack(config, knob, z, kc)
            pe, g = jax.value_and_grad(lambda q: potential_energy(sdt_model, (config,), {}, q))(z)
            if np.isfinite(float(pe)) and all(bool(jnp.all(jnp.isfinite(v))) for v in jax.tree.leaves(g)):
                break
        else:
            raise ValueError(f"bayesroc: no starting point with a finite log density and gradient was found for "
                             f"chain {c + 1} with the given know_crit init in 100 attempts")
        zs.append(z)
    given = set().union(*[set((v or {}).keys()) for v in init_values])
    unused = sorted(given - used)
    if unused:
        if not used:
            raise ValueError("None of the init names match a model parameter: "
                             + ", ".join(unused) + ". Valid names: "
                             + ", ".join(sorted(sites)))
        print("[bayesroc] init entries matching no parameter were ignored: "
              + ", ".join(unused), file=sys.stderr, flush=True)
    return {k: jnp.stack([jnp.asarray(z[k]) for z in zs]) for k in zs[0]}


class _ThreadStack(list):
    def __init__(self):
        import threading
        super().__init__()
        self._tl = threading.local()

    def _s(self):
        st = getattr(self._tl, "s", None)
        if st is None:
            st = self._tl.s = []
        return st

    def append(self, x):
        self._s().append(x)

    def pop(self, *a):
        return self._s().pop(*a)

    def index(self, *a):
        return self._s().index(*a)

    def insert(self, *a):
        self._s().insert(*a)

    def remove(self, x):
        self._s().remove(x)

    def clear(self):
        self._s().clear()

    def __getitem__(self, k):
        return self._s()[k]

    def __setitem__(self, k, v):
        self._s()[k] = v

    def __delitem__(self, k):
        del self._s()[k]

    def __len__(self):
        return len(self._s())

    def __iter__(self):
        return iter(self._s())

    def __reversed__(self):
        return reversed(self._s())

    def __contains__(self, x):
        return x in self._s()

    def __bool__(self):
        return bool(self._s())


def _thread_local_handlers():
    import importlib
    import numpyro.primitives as prim
    if isinstance(prim._PYRO_STACK, _ThreadStack):
        return
    stack = _ThreadStack()
    for name in ("numpyro.primitives", "numpyro.handlers", "numpyro.contrib.control_flow.cond",
                 "numpyro.contrib.control_flow.scan"):
        mod = importlib.import_module(name)
        if hasattr(mod, "_PYRO_STACK"):
            mod._PYRO_STACK = stack


def _split_parts(cfg):
    if isinstance(cfg, dict) and "_nuts_parts" in cfg:
        return {k: v for k, v in cfg.items() if k != "_nuts_parts"}, cfg["_nuts_parts"]
    return cfg, None


class FastNUTS(NUTS):
    """NUTS whose model setup comes from one abstract trace (drghmc._model_info) instead
    of numpyro's initialize_model, which runs the model eagerly and compiles every
    primitive separately. The potential is still generated from the model arguments at each
    call, so a float32 offset swap works as with NUTS, and the transport map's anchors travel
    in the model arguments. Starting points are always passed as init_params. With `coords`
    (NutsCoords) it samples in those coordinates. Trees use Stan's U-turn checks across
    subtree boundaries (nuts_turn); `integrator` is "leapfrog" or "bcss". `cross_chain`: adapt one step
    size for all chains on their mean acceptance (nuts_turn._cross_chain_adapter; chains on a pmap axis
    named "chain")."""

    def __init__(self, model=None, coords=None, integrator="leapfrog", cross_chain=False, gamma=None, reduce="mean",
                 **kwargs):
        super().__init__(model, **kwargs)
        self._coords = coords
        self._integrator = integrator
        self._cross_chain = cross_chain
        self._gamma = gamma
        self._reduce = reduce

    def _init_state(self, rng_key, model_args, model_kwargs, init_params):
        if self._model is None:
            return super()._init_state(rng_key, model_args, model_kwargs, init_params)
        from numpyro.infer.util import potential_energy, constrain_fn
        from nuts_turn import hmc_for
        model, c = self._model, self._coords
        if c is None:
            pot_gen = lambda *a, **k: (lambda z: potential_energy(model, a, k, z))
            post_gen = lambda *a, **k: (lambda z: constrain_fn(model, a, k, z, return_deterministic=True))
        else:
            pot_gen = lambda *a, **k: (lambda z: c.potential(a[0], z))
            post_gen = lambda *a, **k: (lambda z: constrain_fn(model, (_split_parts(a[0])[0],) + tuple(a[1:]), k,
                                                               c.model_z(z, _split_parts(a[0])[1]),
                                                               return_deterministic=True))
        if self._init_fn is None:
            self._init_fn, self._sample_fn = hmc_for(self._integrator, self._cross_chain, self._gamma, self._reduce)(
                potential_fn_gen=pot_gen, kinetic_fn=self._kinetic_fn, algo=self._algo)
        self._potential_fn_gen = pot_gen
        self._postprocess_fn = post_gen
        if init_params is None:
            raise ValueError("FastNUTS needs init_params.")
        return init_params


class FastHMC(HMC):
    """Static-trajectory HMC with FastNUTS's model setup and coordinates. The trajectory length of
    each transition is the one in its state, which the caller draws afresh every transition."""

    def __init__(self, model=None, coords=None, integrator="leapfrog", cross_chain=False, gamma=None, reduce="mean",
                 **kwargs):
        super().__init__(model, **kwargs)
        self._coords = coords
        self._integrator = integrator
        self._cross_chain = cross_chain
        self._gamma = gamma
        self._reduce = reduce

    _init_state = FastNUTS._init_state


_MAP_WARMUP = 1000
_LAPLACE_PASSES = 2
_CURV_LOOP_MAX = 256
_CURV_PROBES = 32
_HMC_TMAX = float(np.pi)
_HMC_GAMMA = 0.05
_HMC_REDUCE = "mean"
_BLOCK = 50


class NutsCoords:
    """The coordinates JAX NUTS samples in, DR-G-HMC's: every random-effect standard deviation on
    its prior-quantile scale and, with `transport`, the random-effect blocks through the per-group
    transport map, its anchors built by a DR-G-HMC warmup and shared by every chain (the map is
    exact whatever its anchors). NUTS positions keep the model's site names and shapes; model_v
    turns one into DR-G-HMC's flat model-side vector, model_z into the model's own sites. A
    float32 process has the anchors built by a float64 child. `aux` [1, A]: the anchors of a fit
    being resumed; `build`: drghmc._build's result when already at hand."""

    def __init__(self, config, chains, seed, parallel, progress_bar, init_strategy, init_radius,
                 init_z=None, transport=True, aux=None, build=None, announce=True, on_anchors=None,
                 anchors="laplace", label="NUTS", split=None, split_masks=None, split_at="lap", split_cand="nc"):
        import drghmc as dr
        _t0 = time.time()
        self.vg, self.unflatten, self.keys, self.off, self.D, self.pp = build or dr._build(sdt_model, config)
        self.tsub = {"build": time.time() - _t0}
        self.tm, self.aux, self.aux_flat = None, None, None
        th0 = dr._initial_positions(sdt_model, config, self.keys, self.off, self.D, chains, seed,
                                    init_strategy, init_radius, init_z, self.unflatten.to_sampled)
        th0 = _rkg_lift(config, th0, lambda r: self.unflatten(jnp.asarray(r)), self.off)
        self._config = config
        self.start, self._th0, self._init_strategy = th0, th0, init_strategy
        self.source, self.pf, self._split_flags, self.label = None, None, None, label
        self.klog, self.wlog = {}, {}
        self.split_thr, self.split_at, self.split_cand = (float(split) if split else None), split_at, split_cand
        self._cfg = config
        self._tm, self._chains, self._parallel, self._progress = None, chains, parallel, progress_bar
        tm = None
        if transport:
            ctrl = dr._merge_control(None)
            tm = dr._transport(sdt_model, config, ctrl, self.off, self.unflatten)
        self._tm = tm
        if tm is None:
            return
        if aux is not None:
            self.source = "given"
            if progress_bar and announce:
                print(f"[{label}] group transport map on {', '.join(tm.specs)}", flush=True)
            if split_masks is not None:
                import tmap as _tmap
                tm, self._split_flags = _tmap.split_supers(tm, None, masks=split_masks, cand_map=split_cand)
                self._tm = tm
            self.aux_flat = np.asarray(aux)[:1]
            self.tm = tm.cheap_of(self.aux_flat)
            self.aux = tuple(p[0] for p in tm.parts(self.aux_flat))
            return
        if anchors == "laplace" and jax.config.jax_enable_x64:
            self.source = "laplace"
            if progress_bar and announce:
                print(f"[{label}] group transport map on {', '.join(tm.specs)}; anchors from the Laplace-marginal mode",
                      flush=True)
            _t1 = time.time()
            early = on_anchors is not None and not split
            if early:
                self.aux_flat = np.zeros((1, tm.aux_size))
                self.tm = tm.cheap_of(self.aux_flat)
                self.aux = tuple(p[0] for p in tm.parts(self.aux_flat))
                on_anchors(self)
            self.set_aux(tm.anchors_chains(self.vg, th0, self.vg.OFF0, parallel)[:1], th0)
            self._klog("init")
            self.tsub["anchors0"] = time.time() - _t1
            self.pf = self._lapmap_passes(seed, chains)
            self._klog("lap")
            if split:
                self.wlog["lap"] = self._wbar_lap()
            if split and split_at == "lap":
                self._apply_split(self.wlog["lap"], self.pf["center"], self._th_s)
            if on_anchors is not None and split:
                on_anchors(self)
            return
        self.source = "dr"
        if progress_bar and announce:
            how = ("anchors from the Laplace-marginal mode (float64 process)" if not jax.config.jax_enable_x64
                   else "anchors from a DR-G-HMC warmup")
            print(f"[{label}] group transport map on {', '.join(tm.specs)}; {how}", flush=True)
        if not jax.config.jax_enable_x64:
            aux, self.start = _anchors_f64(config, chains, seed, parallel, progress_bar, init_strategy,
                                           init_radius, init_z)
        else:
            vg_s = dr._sampler_vg(self.vg, tm)
            _, _, tht, aux_c, _ = dr._adapt(sdt_model, config, self.keys, self.off, self.vg, vg_s, tm,
                                            self.D, chains, _MAP_WARMUP, seed, ctrl, parallel, progress_bar,
                                            th0, user_init=init_z is not None, title="Transport map warmup")
            aux = np.asarray(aux_c)[:1]
            self.start = tm.inverse_chains(tm.forward_chains(tht, aux_c, self.vg.trim), np.repeat(aux, chains, 0),
                                           self.vg.trim)
        self.aux_flat = np.asarray(aux)[:1]
        self.tm = tm.cheap_of(self.aux_flat)
        self.aux = tuple(p[0] for p in tm.parts(self.aux_flat))

    def _vgc(self):
        if getattr(self, "_vg1", None) is None:
            vgm, OFF = self.tm.wrap(self.vg.pot), self.vg.OFF0
            self._vg1 = jax.jit(lambda v, a: vgm(v, OFF, a))
        return self._vg1

    def _lapmap_passes(self, seed, chains):
        import time as _t
        from scipy.optimize import minimize
        zmask = np.concatenate([np.full(int(np.prod(self.off[k][1])), k.startswith("z_")) for k in self.keys])
        hidx, zidx = np.flatnonzero(~zmask), np.flatnonzero(zmask)
        hid, zid = jnp.asarray(hidx), jnp.asarray(zidx)
        passes = _LAPLACE_PASSES
        OFF, trim, tmv, vg1 = self.vg.OFF0, self.vg.trim, self.tm, self._vgc()

        def solve(base, p):
            b = np.array(base)
            bar = progress.bar(None, f"[{self.label}] Laplace pass {p + 1} of {passes}", unit="L-BFGS iterations",
                               leave=False) if self._progress else None

            def fg(h):
                b[hidx] = h
                u, g = vg1(jnp.asarray(b), self.aux)
                return float(u), np.asarray(g)[hidx]

            r = minimize(fg, b[hidx], jac=True, method="L-BFGS-B", callback=(lambda *_: bar.update()) if bar is not None else None,
                         options=dict(maxiter=1000, maxcor=10, gtol=1e-8, ftol=1e-8, maxls=1000))
            return r.x, int(r.nit), float(r.fun), bar

        fwd_many = jax.jit(lambda V, a: jax.lax.map(lambda v: tmv.to_model(tmv.forward(v, a, trim)[0], a), V))

        info = dict(walls=[], iters=[], h=[], f=[])
        rng = np.random.default_rng(int(seed))
        for p in range(passes):
            t0 = _t.time()
            base = jnp.asarray(self.start[0]).at[zid].set(0.0)
            if "know_crit_shift" in self.off:
                sl = self.off["know_crit_shift"][0]
                thm = np.asarray(tmv.to_model(tmv.forward(base, self.aux, trim)[0], self.aux))[None]
                base = base.at[sl].set(jnp.asarray(_rkg_lift(self._config, thm, lambda r: self.unflatten(jnp.asarray(r)), self.off)[0, sl]))
            hstar, it, fval, bar = solve(base, p)
            center = np.asarray(base.at[hid].set(hstar))
            info["h"].append(np.asarray(hstar))
            info["f"].append(fval)
            starts = np.repeat(center[None], chains, 0)
            starts[:, zmask] = rng.standard_normal((chains, int(zmask.sum())))
            th_all = np.asarray(fwd_many(jnp.asarray(np.concatenate([center[None], starts])), self.aux))
            th_c, th_s = th_all[:1], th_all[1:]
            self.set_aux(self._tm.anchors_chains(self.vg, np.repeat(th_c, chains, 0), OFF, self._parallel)[:1], th_s)
            self._th_s = th_s
            info["walls"].append(_t.time() - t0)
            info["iters"].append(it)
            if bar is not None:
                bar.close(drop=True)
            if self._progress:
                print(f"[{self.label}] Laplace pass {p + 1}: {it} iterations, {info['walls'][-1]:.1f} s", flush=True)
        t0 = _t.time()
        vc = jnp.asarray(self._tm.inverse_chains(th_c, self.aux_flat, trim)[0])
        h_sd_final = 1.0 / np.sqrt(np.maximum(self._fd_hdiag(vc, hidx), 1e-12))
        if self._init_strategy == "laplace":
            v = np.repeat(np.asarray(vc)[None], chains, 0)
            v[:, hidx] += np.minimum(h_sd_final, 1.0) * rng.standard_normal((chains, len(hidx)))
            v[:, zidx] = rng.standard_normal((chains, len(zidx)))
            self.start = v
        else:
            self.start = self._tm.inverse_chains(np.asarray(self._th0), np.repeat(self.aux_flat, chains, 0), trim)
        info["walls"].append(_t.time() - t0)
        return dict(iters=np.asarray(info["iters"]), wall=float(np.sum(info["walls"])), walls=np.asarray(info["walls"]),
                    h=np.asarray(info["h"]), f=np.asarray(info["f"]), hidx=hidx, h_sd_final=h_sd_final, center=th_c)

    def _fd_hdiag(self, vc, idx):
        idx = np.asarray(idx)
        if len(idx) <= _CURV_LOOP_MAX:
            vg1, v0, e = self._vgc(), np.array(vc), 1e-4

            def gi(i, d):
                v = v0.copy()
                v[i] += d
                return float(np.asarray(vg1(jnp.asarray(v), self.aux)[1])[i])

            return np.array([(gi(i, e) - gi(i, -e)) / (2 * e) for i in idx])
        nd = jax.local_device_count()
        per = -(-_CURV_PROBES // nd)
        rng = np.random.default_rng(int(len(idx)))
        Z = np.zeros((nd * per, self.D))
        Z[:, idx] = rng.choice([-1.0, 1.0], size=(nd * per, len(idx)))
        vg1, v0, e = self._vgc(), np.asarray(vc), 1e-4
        g = lambda v: np.asarray(vg1(jnp.asarray(v), self.aux)[1])
        tot = sum(z * (g(v0 + e * z) - g(v0 - e * z)) / (2 * e) for z in Z)
        return tot[idx] / (nd * per)

    def lap_curvature(self):
        vc = self._tm.inverse_chains(self.pf["center"], self.aux_flat, self.vg.trim)[0]
        self.pf["h_sd_final"] = 1.0 / np.sqrt(np.maximum(self._fd_hdiag(vc, self.pf["hidx"]), 1e-12))

    def _sd_pos(self, sb):
        return None if len(sb.blocks) != 1 or sb.blocks[0].L_flat is None else np.asarray(sb.blocks[0].sd_idx)

    def _wbar_lap(self):
        """Each single correlated block's per-dimension map weight w = K / (1 + K) averaged over the Laplace
        approximation of its standard deviation: K = Lam sigma^2 with Lam from the Laplace anchors and sigma =
        scale q(t), t the prior-quantile coordinate, integrated over t ~ N(t_hat, h^-1) by three-point
        Gauss-Hermite. None for a block without one correlated block."""
        from drghmc import _sd_logq
        nodes, wts = np.array([0.0, 3.0 ** 0.5, -(3.0 ** 0.5)]), np.array([2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0])
        hpos = {int(h): i for i, h in enumerate(np.asarray(self.pf["hidx"]))}
        center = np.asarray(self.pf["center"])[0]
        out = []
        for k, sb in enumerate(self._tm.sbs):
            pos = self._sd_pos(sb)
            if pos is None:
                out.append(None)
                continue
            K = np.asarray(self.klog["lap"][k])
            t = center[pos]
            sd = np.array([float(self.pf["h_sd_final"][hpos[int(p)]]) for p in pos])
            lq0 = np.asarray(_sd_logq(jnp.asarray(t)))
            w = np.zeros_like(K)
            for x, a in zip(nodes, wts):
                r2 = np.exp(2.0 * (np.asarray(_sd_logq(jnp.asarray(t + x * sd))) - lq0))
                w = w + a * K * r2 / (1.0 + K * r2)
            out.append(w)
        return out

    def wbar_reb(self, sd2_ratio):
        """The rebuild's per-dimension map weight with sigma^2 averaged over the rebuild window's draws:
        `sd2_ratio` per block is mean(sigma^2 over the window) / sigma^2 at the last draw (the anchors' reference)."""
        out = []
        for k, sb in enumerate(self._tm.sbs):
            if sd2_ratio[k] is None:
                out.append(None)
                continue
            K = np.asarray(self.klog["reb"][k]) * np.asarray(sd2_ratio[k])
            out.append(K / (1.0 + K))
        return out

    def _apply_split(self, wbar, th_ref, th_start):
        """Split the single correlated blocks whose dimensions straddle the threshold on the averaged weights
        `wbar`, and rebuild the anchors in the split map at `th_ref` [1, D] with the chains at `th_start`."""
        import tmap as _tmap
        masks = [None if w is None else (np.asarray(w) < self.split_thr) for w in wbar]
        masks = [None if m is None or not (0 < int((~m).sum()) < len(m)) else m for m in masks]
        if all(m is None for m in masks):
            return False
        tm2, fl = _tmap.split_supers(self._tm, None, masks=masks, cand_map=self.split_cand)
        if fl is None:
            return False
        self._tm, self._split_flags = tm2, fl
        if self._progress:
            print(f"[{self.label}] map: " + "; ".join(f"{sb.name} {len(sb.kept_nat)} of {sb.D} dimensions mapped, "
                                                     f"the rest non-centred" for sb in tm2.sbs if sb.split), flush=True)
        self.set_aux(tm2.anchors_chains(self.vg, np.repeat(np.asarray(th_ref)[:1], self._chains, 0), self.vg.OFF0,
                                        self._parallel)[:1], th_start)
        return True

    def _klog(self, tag):
        """Each super-block's per-dimension map strength K (natural order) at the current anchors."""
        P = self._tm.parts(self.aux_flat)
        self.klog[tag] = [np.asarray(P[8 * k + 5])[0].copy() for k in range(len(self._tm.sbs))]

    @property
    def split_masks(self):
        """The candidate masks of a split map (per super-block, natural order), None without a split."""
        if self._split_flags is None:
            return None
        return [np.asarray(sb.cand).copy() if sb.split else None for sb in self._tm.sbs]

    def rebuild_from(self, last, mean):
        """New anchors at the window mean pooled over the chains (model coordinates [chains, D]); the chains
        restart from `last`."""
        mean = np.asarray(mean)
        ref = np.repeat(mean.mean(0, keepdims=True), mean.shape[0], 0)
        self.set_aux(self._tm.anchors_chains(self.vg, ref, self.vg.OFF0, self._parallel)[:1], np.asarray(last))
        self._klog("reb")

    def set_aux(self, aux, th):
        tm = self._tm
        if self._split_flags is not None:
            aux = tm.with_flags(np.asarray(aux)[:1], self._split_flags)
        self.aux_flat = np.asarray(aux)[:1]
        self.tm = tm.cheap_of(self.aux_flat)
        self.aux = tuple(p[0] for p in tm.parts(self.aux_flat))
        self.start = tm.inverse_chains(np.asarray(th), np.repeat(self.aux_flat, self._chains, 0), self.vg.trim)

    def flat(self, z):
        return jnp.concatenate([jnp.reshape(z[k], (-1,)) for k in self.keys])

    def model_v(self, z, parts=None):
        v = self.flat(z)
        a = self.aux if parts is None else parts
        return v if self.tm is None else self.tm.to_model(self.tm.forward(v, a, self.vg.trim)[0], a)

    def model_z(self, z, parts=None):
        return self.unflatten(self.model_v(z, parts))

    def potential(self, cfg, z):
        from numpyro.infer.util import potential_energy
        cfg, parts = _split_parts(cfg)
        return self._potential_core(cfg, z, parts)

    def _potential_core(self, cfg, z, parts):
        from numpyro.infer.util import potential_energy
        a = self.aux if parts is None else parts
        v = self.flat(z)
        th, ld = (v, 0.0) if self.tm is None else self.tm.forward(v, a, self.vg.trim)
        if self.tm is not None:
            cfg = self.tm.with_order(cfg, a, "_corr_perm")
        return potential_energy(sdt_model, (cfg,), {}, self.unflatten(th)) - self.unflatten.log_jac(th) + ld

    def init_params(self, chains, batched=False):
        z = {k: jnp.asarray(np.asarray(self.start)[:, self.off[k][0]].reshape((chains,) + tuple(self.off[k][1])))
             for k in self.keys}
        return z if chains > 1 or batched else {k: v[0] for k, v in z.items()}

    def positions(self, zs):
        """DR-G-HMC model-side positions [chains, D] from NUTS positions with a chain axis."""
        f = jax.jit(self.model_v)
        C = int(np.shape(zs[self.keys[0]])[0])
        return np.stack([np.asarray(f({k: jnp.asarray(zs[k][c]) for k in self.keys})) for c in range(C)])


def _anchors_f64(config, chains, seed, parallel, progress_bar, init_strategy, init_radius, init_z):
    """NutsCoords' anchors [1, A] and starting positions, built in a float64 child process."""
    import pickle, subprocess, sys, tempfile, shutil
    tmp = tempfile.mkdtemp(prefix="bayesroc_nuts_anchors_")
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        argfile, out = os.path.join(tmp, "args.pkl"), os.path.join(tmp, "anchors.npz")
        with open(argfile, "wb") as fh:
            pickle.dump(dict(config=config, chains=chains, seed=seed, parallel=parallel, progress_bar=progress_bar,
                             init_strategy=init_strategy, init_radius=init_radius, backend_dir=here, out=out,
                             init_z=None if init_z is None else {k: np.asarray(v) for k, v in init_z.items()}), fh)
        env = dict(os.environ)
        env["BAYESROC_ENABLE_X64"] = "1"
        env["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={chains if parallel else 1} --xla_cpu_collective_call_terminate_timeout_seconds=86400"
        code = subprocess.call([sys.executable, os.path.join(here, "autorun.py"), "_nuts_anchors", argfile], env=env)
        if code != 0 or not os.path.exists(out):
            raise RuntimeError(f"bayesroc: the float64 process building the transport map's anchors failed (exit code {code})")
        with np.load(out) as d:
            return d["aux"], d["start"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def nuts_initial_mass(coords):
    """Initial inverse mass from the Laplace stage: the Laplace variances of the hyperparameters in map
    coordinates, capped at NUTS's unit default, and 1 for the effects. None without Laplace anchors."""
    if coords.pf is None or "h_sd_final" not in coords.pf:
        return None
    imm0 = np.ones(coords.D)
    imm0[coords.pf["hidx"]] = np.minimum(coords.pf["h_sd_final"] ** 2, 1.0)
    return imm0


def _divisor(n, cap=_BLOCK):
    return max(d for d in range(1, min(n, cap) + 1) if n % d == 0)


def _rebuild_point(warmup):
    """The end of the first mass-adaptation window: where the anchors are rebuilt at the chains' positions,
    so every later window adapts in the rebuilt map. None when the schedule has no full window."""
    from numpyro.infer.hmc_util import build_adaptation_schedule
    sched = build_adaptation_schedule(warmup)
    if len(sched) < 3:
        return None
    t = sched[1].end + 1
    return t if t < warmup else None


class _FoldPrograms:
    """One program for warmup and sampling, running a block of transitions for every chain (pmap), with the
    map's anchors and the phase as runtime arguments; lowered and compiled in a background thread while the
    Laplace stage runs. `tmax`: draw each transition's trajectory length uniformly on (0, tmax]
    (static-trajectory HMC); None for NUTS. Without `parallel` the program loops over the chains."""

    def __init__(self, config, chains, warmup, thinning, rng_key, make_kernel, tmax, sampling=True, parallel=True,
                 max_steps=1024):
        import math
        self.parallel = parallel
        self.config, self.C, self.warmup, self.thin = config, chains, warmup, int(thinning)
        self.rng_key, self.make_kernel, self.tmax, self.sampling = rng_key, make_kernel, tmax, sampling
        self.max_steps = int(max_steps)
        self.rebuild = _rebuild_point(warmup)
        self.Bw = _divisor(math.gcd(warmup, self.rebuild) if self.rebuild else warmup) if warmup > 0 else _BLOCK
        self.Bs = self.Bw
        self.thread, self.err, self.t = None, None, {}

    def start(self, coords):
        import threading
        _thread_local_handlers()
        t0 = time.time()
        C, warmup, thin, tmax, B = self.C, self.warmup, self.thin, self.tmax, self.Bw
        kernel = self.kernel = self.make_kernel(coords)
        config, D, sampling = self.config, coords.D, self.sampling
        cfgp = self.cfgp = lambda parts: {**config, "_nuts_parts": parts}
        self.keys = random.split(self.rng_key, C)
        jitter = self.jitter = _jitter(tmax, self.max_steps)
        from drghmc import _sd_logq
        sdp_np = self.sdp_np = np.concatenate([np.zeros(0, int)] + [p for p in (coords._sd_pos(sb) for sb in coords._tm.sbs)
                                                                    if p is not None]) if coords._tm is not None else np.zeros(0, int)
        sdp = jnp.asarray(sdp_np)

        def init(key, z, parts):
            return kernel.init(key, warmup, z, (cfgp(parts),), {})

        def run(st, parts, samp, acc):
            args = (cfgp(parts),)
            post = kernel.postprocess_fn(args, {})
            shp = jax.eval_shape(post, st.z) if sampling else None
            n = jnp.where(samp, thin, 1)

            def body(c, _):
                s, tot = c

                def step(_, q):
                    s, dv = q
                    s = kernel.sample(jitter(s), args, {})
                    return s, dv | s.diverging

                s, dv = jax.lax.fori_loop(0, n, step, (s, jnp.zeros((), bool)))

                def add(t):
                    mv = coords.model_v(s.z, parts)
                    return t + jnp.concatenate([mv, jnp.exp(2.0 * _sd_logq(mv[sdp]))])

                tot = jax.lax.cond(acc, add, lambda t: t, tot)
                return (s, tot), (s.z if sampling else None, s.num_steps, s.energy, s.accept_prob, dv, s.adapt_state.step_size)

            (s, tot), ex = jax.lax.scan(body, (st, jnp.zeros(D + len(sdp_np), jnp.result_type(float))), None, length=B)
            if sampling:
                out = jax.lax.cond(samp, lambda zz: jax.lax.map(post, zz),
                                   lambda zz: jax.tree.map(lambda x: jnp.zeros((B,) + x.shape, x.dtype), shp), ex[0])
                ex = (out,) + ex[1:]
            return s, tot, coords.model_v(s.z, parts), ex

        deferred = getattr(kernel, "_cross_chain", None)

        def run_shared(sts, parts, samp, acc):
            args = (cfgp(parts),)
            post = kernel.postprocess_fn(args, {})
            shp = jax.eval_shape(post, jax.tree.map(lambda a: a[0], sts.z)) if sampling else None
            n = jnp.where(samp, thin, 1)
            from numpyro.infer.hmc_util import IntegratorState

            def one(s):
                def step(_, q):
                    s, dv = q
                    s = kernel.sample(jitter(s), args, {})
                    return s, dv | s.diverging

                return jax.lax.fori_loop(0, n, step, (s, jnp.zeros((), bool)))

            def adapt(s, a, upd):
                t = s.i - 1
                zi = IntegratorState(s.z, s.z, s.potential_energy, s.z_grad)
                new = jax.lax.cond(t < warmup, lambda q: upd(t, a, zi, q), lambda q: q, s.adapt_state)
                return s._replace(adapt_state=new)

            def body(c, _):
                ss, tot = c
                ss, dv = jax.lax.map(one, ss)
                from nuts_turn import _stack_mean
                a = _stack_mean(ss.accept_prob, deferred.reduce)
                t0 = ss.i[0] - 1
                ends = jnp.asarray(deferred.pool_at or [-1])
                ss = jax.lax.cond(jnp.any(t0 == ends),
                                  lambda q: jax.vmap(lambda r: adapt(r, a, deferred.update_pooled), axis_name="chain")(q),
                                  lambda q: jax.lax.map(lambda r: adapt(r, a, deferred.update), q), ss)

                def add(t):
                    v = jax.lax.map(lambda z: coords.model_v(z, parts), ss.z)
                    return t + jnp.concatenate([v, jnp.exp(2.0 * _sd_logq(v[:, sdp]))], axis=1)

                tot = jax.lax.cond(acc, add, lambda t: t, tot)
                return (ss, tot), (ss.z if sampling else None, ss.num_steps, ss.energy, ss.accept_prob, dv, ss.adapt_state.step_size)

            (ss, tot), ex = jax.lax.scan(body, (sts, jnp.zeros((C, D + len(sdp_np)), jnp.result_type(float))), None, length=B)
            if sampling:
                out = jax.lax.cond(samp, lambda zz: jax.lax.map(lambda zc: jax.lax.map(post, zc), zz),
                                   lambda zz: jax.tree.map(lambda x: jnp.zeros((B, C) + x.shape, x.dtype), shp), ex[0])
                ex = (out,) + ex[1:]
            ex = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), ex)
            return ss, tot, jax.lax.map(lambda z: coords.model_v(z, parts), ss.z), ex

        if self.parallel:
            self.initf = jax.pmap(init, in_axes=(0, {k: 0 for k in coords.keys}, None), axis_name="chain")
            self.prog = jax.pmap(run, in_axes=(0, None, None, None), axis_name="chain")
            self.shard = jax.pmap(lambda x: x)
        else:
            loop = lambda f, n: jax.jit(lambda *a: jax.lax.map(lambda x: f(*x, *a[n:]), tuple(a[:n])))
            self.initf = loop(init, 2)
            self.prog = jax.jit(run_shared) if hasattr(deferred, "update") else loop(run, 1)
            self.shard = jnp.asarray
        parts0, z0 = coords.aux, coords.init_params(C, batched=True)

        def work():
            try:
                one = jax.tree.map(lambda a: a[0], z0)
                st1 = jax.eval_shape(lambda k, z: kernel.init(k, warmup, z, (cfgp(parts0),), {}), self.keys[0], one)
                sts = jax.tree.map(lambda a: jax.ShapeDtypeStruct((C,) + tuple(a.shape), a.dtype), st1)
                f = jnp.asarray(False)
                low = {"initf": self.initf.lower(self.keys, z0, parts0), "prog": self.prog.lower(sts, parts0, f, f)}
                self.t["lower"] = time.time() - t0
                done = {}

                def comp(k):
                    try:
                        done[k] = low[k].compile()
                    except Exception as e:
                        done[k] = e
                    self.t[k] = time.time() - t0

                side = threading.Thread(target=comp, args=("initf",), daemon=True)
                side.start()
                comp("prog")
                side.join()
                for k in low:
                    if isinstance(done[k], Exception):
                        raise done[k]
                    setattr(self, k, done[k])
            except Exception as e:
                self.err = e

        self.thread = threading.Thread(target=work, daemon=True)
        self.thread.start()

    def warmf(self, st, parts, acc):
        return self.prog(st, parts, jnp.asarray(False), jnp.asarray(acc))

    def sampf(self, st, parts):
        s, _, _, ex = self.prog(st, parts, jnp.asarray(True), jnp.asarray(False))
        return s, ex

    def wait(self):
        t0 = time.time()
        self.thread.join()
        self.t["waited"] = time.time() - t0
        if self.err is not None:
            raise self.err


def _with_mass(st, imm, pg, D):
    am = st.adapt_state
    im = am.inverse_mass_matrix
    if isinstance(im, dict) and len(im) != 1:
        raise RuntimeError("bayesroc: the initial mass needs one diagonal mass block")
    v = np.broadcast_to(np.asarray(imm, dtype=float), (pg.C, D))
    wrap = (lambda x: {k: x for k in im}) if isinstance(im, dict) else (lambda x: x)
    return st._replace(adapt_state=am._replace(inverse_mass_matrix=wrap(pg.shard(jnp.asarray(v))),
                                               mass_matrix_sqrt=wrap(pg.shard(jnp.asarray(1.0 / np.sqrt(v)))),
                                               mass_matrix_sqrt_inv=wrap(pg.shard(jnp.asarray(np.sqrt(v))))))


def _jitter(tmax, nmax=1024):
    """The state with its number of steps drawn as max(1, ceil(tmax u / eps)), u uniform on (0, 1] from the
    state's own key, at the nominal step size eps (Stan, TFP's ChEES and Neal 2011 keep the step and round
    the step count; numpyro's exact-length rescaling to T / ceil(T / eps) hides an oversized step from the
    adaptation), at most nmax steps (NUTS's 2^max_tree_depth; TFP's ChEES max_leapfrog_steps) and one step
    for a non-finite step size; the identity for NUTS (tmax None)."""
    if tmax is None:
        return lambda s: s

    def f(s):
        u = 1.0 - random.uniform(random.fold_in(s.rng_key, 7))
        eps = s.adapt_state.step_size
        n = jnp.maximum(1.0, jnp.ceil(tmax * u / eps))
        n = jnp.where(jnp.isfinite(n), jnp.minimum(n, float(nmax)), 1.0)
        return s._replace(trajectory_length=jnp.asarray((n - 1e-6) * eps, s.trajectory_length.dtype))

    return f


def _chain_adapt(parallel, chains, gamma=_HMC_GAMMA, reduce=_HMC_REDUCE):
    from nuts_turn import _Deferred
    return False if chains < 2 else True if parallel else _Deferred(gamma, reduce)


def fold_setup(config, chains, warmup, thinning, seed, target_accept_prob, max_tree_depth, step_size, progress_bar,
               init_strategy, init_radius, init_z, transport, sampler, sampling=True, build=None, parallel=True,
               gamma=_HMC_GAMMA, integrator="bcss", reduce=_HMC_REDUCE, split=None, split_at="lap", split_cand="nc"):
    """The coordinates (Laplace anchors) and the background-compiled programs of a fold fit. `gamma`: the
    HMC step size's dual averaging gain (Stan's `adapt gamma`)."""
    hmc = sampler == "hmc"
    label = "HMC" if hmc else "NUTS"

    def make_kernel(c):
        kw = {"inverse_mass_matrix": jnp.ones(c.D)} if c.source == "laplace" else {}
        if hmc:
            return FastHMC(sdt_model, coords=c, integrator=integrator, cross_chain=_chain_adapt(parallel, chains, gamma, reduce),
                           gamma=gamma, reduce=reduce,
                           target_accept_prob=target_accept_prob, step_size=float(step_size),
                           trajectory_length=_HMC_TMAX, **kw)
        return FastNUTS(sdt_model, coords=c, target_accept_prob=target_accept_prob, max_tree_depth=max_tree_depth,
                        step_size=float(step_size), **kw)

    pg = _FoldPrograms(config, chains, warmup, thinning, random.PRNGKey(seed), make_kernel,
                       _HMC_TMAX if hmc else None, sampling, parallel and chains > 1, max_steps=2 ** int(max_tree_depth))
    coords = NutsCoords(config, chains, seed, parallel and chains > 1, progress_bar, init_strategy, init_radius, init_z,
                        transport=transport, on_anchors=pg.start, anchors="laplace", label=label, build=build,
                        split=split, split_at=split_at, split_cand=split_cand)
    if pg.thread is None:
        pg.start(coords)
    pg.wait()
    return coords, pg, label


def fold_warmup(pg, coords, warmup, trace, progress_bar, label):
    """Warmup with the programs of `pg`. At the end of the first mass window the anchors are rebuilt at
    the chains' window mean (the window's second half, pooled over the chains), the chains moved into the
    rebuilt map and the initial mass recomputed there; every later window adapts in that map. Returns the
    warmed-up state (chain axis first)."""
    C, D = pg.C, coords.D
    cat = lambda xs: np.concatenate([np.asarray(x) for x in xs], 1)
    t_w = time.time()
    parts = coords.aux
    st = pg.initf(pg.keys, coords.init_params(C, batched=True), parts)
    pe0 = np.asarray(st.potential_energy)
    if not np.all(np.isfinite(pe0)):
        raise RuntimeError(f"bayesroc: the log density is not finite at the starting point of chain(s) "
                           f"{', '.join(str(c + 1) for c in np.flatnonzero(~np.isfinite(pe0)))}; supply other "
                           f"initial values or none")
    imm = nuts_initial_mass(coords)
    if imm is not None:
        st = _with_mass(st, imm, pg, D)
    rb = pg.rebuild if coords.source == "laplace" else None
    tot, n_acc, steps, w_acc, w_eps = np.zeros((C, D + len(pg.sdp_np))), 0, [], [], []
    bar = progress.bar(warmup, f"{label} warmup") if progress_bar else None
    i = 0
    while i < warmup:
        acc = rb is not None and rb // 2 <= i < rb
        st, t_blk, last, (_, ns, _, ap, _, ss) = pg.warmf(st, parts, acc)
        i += pg.Bw
        steps.append(ns)
        w_acc.append(ap)
        w_eps.append(ss)
        if acc:
            tot, n_acc = tot + np.asarray(t_blk), n_acc + pg.Bw
        if rb is not None and i == rb:
            mean = tot / max(n_acc, 1)
            coords.rebuild_from(np.asarray(last), mean[:, :D])
            if pg.sdp_np.size and coords.split_thr is not None:
                from drghmc import _sd_logq
                lastq2 = np.exp(2.0 * np.asarray(_sd_logq(jnp.asarray(np.asarray(last)[:, pg.sdp_np]))))
                ratio = mean[:, D:].mean(0) / lastq2.mean(0)
                per, o = [], 0
                for sb in coords._tm.sbs:
                    p = coords._sd_pos(sb)
                    per.append(None if p is None else ratio[o:o + len(p)])
                    o += 0 if p is None else len(p)
                coords.wlog["reb"] = coords.wbar_reb(per)
                if coords.split_thr is not None and coords.split_at == "reb" and coords._split_flags is None:
                    ref = np.where(coords._tm.scale_mask(D), np.asarray(last), mean[:, :D])
                    if coords._apply_split(coords.wlog["reb"], ref[:1], np.asarray(last)):
                        pg.start(coords)
                        pg.wait()
            if coords.pf is not None:
                coords.lap_curvature()
            parts = coords.aux
            fresh = pg.initf(pg.keys, coords.init_params(C, batched=True), parts)
            st = st._replace(z=fresh.z, z_grad=fresh.z_grad, potential_energy=fresh.potential_energy)
            imm = nuts_initial_mass(coords)
            if imm is not None:
                st = _with_mass(st, imm, pg, D)
        if bar is not None:
            jax.block_until_ready(st.z)
            bar.update(pg.Bw)
    if bar is not None:
        bar.close()
    jax.block_until_ready(st.z)
    if trace is not None:
        trace["warmup_wall"] = time.time() - t_w
        if steps:
            trace["warmup_num_steps"] = cat(steps)
            trace["warmup_accept"] = cat(w_acc)
            trace["warmup_step_size"] = cat(w_eps)
        trace["step_size"] = np.asarray(st.adapt_state.step_size)
        im = st.adapt_state.inverse_mass_matrix
        im = list(im.values()) if isinstance(im, dict) else [im]
        trace["inverse_mass"] = np.concatenate([np.reshape(np.asarray(v), (C, -1)) for v in im], 1)
    return st


def _fold_run(pg, coords, warmup, samples, trace, progress_bar, label):
    """fold_warmup, then `samples` draws. Returns (samples by chain, extra fields, last state)."""
    cat = lambda xs: np.concatenate([np.asarray(x) for x in xs], 1)
    st = fold_warmup(pg, coords, warmup, trace, progress_bar, label)
    parts = coords.aux
    t_s = time.time()
    outs, ex = [], []
    bar = progress.bar(samples, f"{label} sampling") if progress_bar else None
    done = 0
    while done < samples:
        st, (out, ns, en, ap, dv, _) = pg.sampf(st, parts)
        outs.append(out)
        ex.append((ns, en, ap, dv))
        k = min(pg.Bs, samples - done)
        done += pg.Bs
        if bar is not None:
            jax.block_until_ready(st.z)
            bar.update(k)
    if bar is not None:
        bar.close()
    jax.block_until_ready(st.z)
    samples_dict = {k: cat([o[k] for o in outs])[:, :samples] for k in outs[0]}
    extra = {f: cat([e[j] for e in ex])[:, :samples] for j, f in enumerate(("num_steps", "energy", "accept_prob", "diverging"))}
    if trace is not None:
        trace["sampling_wall"] = time.time() - t_s
        trace["sampling_num_steps"] = extra["num_steps"]
        trace["sampling_accept"] = extra["accept_prob"]
        trace["sampling_divergent"] = extra["diverging"]
    return samples_dict, extra, st


def _fit_fold(config, chains, warmup, samples, seed, target_accept_prob, max_tree_depth, step_size, thinning,
              progress_bar, init_strategy, init_radius, init_z, control, save_loglik, keep_re, save_state, sampler,
              parallel=True):
    """sampler "hmc": static-trajectory HMC with the trajectory length drawn uniformly on (0, pi] each
    transition and the two-stage BCSS integrator; "nuts": NUTS (leapfrog). Both in transport-map
    coordinates with Laplace anchors, the Laplace initial mass and the rebuild at the end of the first
    mass window, NumPyro's (Stan's) windowed adaptation, chains in parallel."""
    t0 = time.time()
    coords, pg, label = fold_setup(config, chains, warmup, thinning, seed, target_accept_prob, max_tree_depth,
                                   step_size, progress_bar, init_strategy, init_radius, init_z,
                                   bool((control or {}).get("transport", True)), sampler, parallel=parallel,
                                   gamma=float((control or {}).get("adapt_gamma", _HMC_GAMMA)),
                                   integrator=str((control or {}).get("integrator", "bcss")),
                                   split=_split_threshold(control),
                                   split_at=str((control or {}).get("transport_split_at", "lap")),
                                   split_cand=str((control or {}).get("transport_split_cand", "nc")))
    trace_path = os.environ.get("BAYESROC_NUTS_TRACE")
    trace = {"t_coords": time.time() - t0, **{f"prep_{k}": v for k, v in pg.t.items()}} if trace_path else None
    samples_dict, extra, st = _fold_run(pg, coords, warmup, -(-int(samples) // int(thinning)), trace, progress_bar, label)
    if trace is not None and coords._tm is not None:
        trace["K_blocks"] = np.asarray([sb.name for sb in coords._tm.sbs])
        for k, m in enumerate(coords.split_masks or []):
            if m is not None:
                trace[f"split_cand_{k}"] = np.asarray(m, bool)
        for tag, Ks in coords.klog.items():
            for k, K in enumerate(Ks):
                trace[f"K_{tag}_{k}"] = K
        for tag, ws in coords.wlog.items():
            for k, w in enumerate(ws):
                if w is not None:
                    trace[f"wbar_{tag}_{k}"] = w
    elapsed = time.time() - t0
    if trace is not None:
        trace["elapsed"] = elapsed
        np.savez(trace_path, **{k: np.asarray(v) for k, v in trace.items()})
    samples_np = {}
    for k, v in samples_dict.items():
        if k.startswith("z_") and not keep_re:
            continue
        if k == "log_lik" and not save_loglik:
            continue
        samples_np[k] = np.array(v)
    _restore_centred(samples_np)
    out = {"samples": samples_np, "extra_fields": {k: np.array(v) for k, v in extra.items()}, "elapsed": float(elapsed)}
    if save_state:
        out["state"] = {"hmc": st, "offset": None, "aux": coords.aux_flat, "split": coords.split_masks,
                        "split_cand": coords.split_cand}
    return out


def _split_threshold(control):
    v = (control or {}).get("transport_split", False)
    if v is True:
        return 0.1
    return float(v) if v else None


def fit_model(config, chains=4, warmup=500, samples=1000, seed=0,
              target_accept_prob=0.8, max_tree_depth=10, step_size=1.0,
              thinning=1, chain_method="parallel",
              progress_bar=True,
              init_strategy="numeric", init_radius=0.0,
              sampler="hmc", control=None, save_loglik=True,
              keep_re=False, save_state=False, init_values=None):
    """Fit an SDT model on the JAX backend.

    Args:
        config: dict from R's build_numpyro_config()
        chains: number of MCMC chains
        warmup: warmup iterations
        samples: post-warmup iterations per chain; ceil(samples / thinning) draws are kept
        seed: random seed
        target_accept_prob: target acceptance probability (Stan's adapt_delta)
        max_tree_depth: maximum tree depth for NUTS (default 10)
        step_size: initial step size (default 1.0)
        progress_bar: whether to show progress bars
        init_strategy: "prior" (NumPyro init_to_sample), "random" or "numeric"
            (init_to_uniform with `init_radius`). Default "numeric" with radius 0,
            i.e. every parameter starts at 0 on the unconstrained scale, as in R.
        init_radius: radius for init_to_uniform when init_strategy != "prior".
        init_values: optional per-chain list of {Stan name: value} dicts; used
            instead of init_strategy (see init_values_to_z).
        save_state: also return the end-of-run sampler state (fit_broc's `resume`).
        sampler: "hmc" (default; static-trajectory HMC with jittered length), "nuts" or
            "drghmc" (experimental).
        control: dict of sampler settings (drghmc's tuning; transport for all).

    Returns:
        dict with samples (by chain), extra_fields, elapsed
    """
    init_z = None
    if init_values is not None:
        if len(init_values) != chains:
            raise ValueError(f"init has {len(init_values)} chain entries; expected {chains}.")
        init_z = init_values_to_z(config, init_values, seed, init_radius)

    if sampler == "drghmc":
        from drghmc import run_drghmc
        return run_drghmc(sdt_model, config, chains=chains, warmup=warmup, samples=samples,
                          seed=seed, thinning=thinning, chain_method=chain_method,
                          progress_bar=progress_bar, init_strategy=init_strategy,
                          init_radius=init_radius, control=control,
                          save_loglik=save_loglik, keep_re=keep_re,
                          save_state=save_state, init_z=init_z)

    numpyro.set_host_device_count(chains if chain_method == "parallel" else 1)
    kw = dict(warmup=warmup, samples=samples, target_accept_prob=target_accept_prob, max_tree_depth=max_tree_depth,
              step_size=step_size, thinning=thinning, progress_bar=progress_bar, init_strategy=init_strategy,
              init_radius=init_radius, control=control, save_loglik=save_loglik, keep_re=keep_re,
              save_state=save_state, sampler=sampler)
    if sampler == "hmc" or jax.config.jax_enable_x64:
        return _fit_fold(config, chains, seed=seed, init_z=init_z, parallel=chain_method == "parallel", **kw)
    return _fit_nuts_f32(config, chains, warmup, samples, seed, target_accept_prob, max_tree_depth, step_size,
                         thinning, chain_method, progress_bar, init_strategy, init_radius, init_z, control,
                         save_loglik, keep_re, save_state)


def _fit_nuts_f32(config, chains, warmup, samples, seed, target_accept_prob, max_tree_depth, step_size, thinning,
                  chain_method, progress_bar, init_strategy, init_radius, init_z, control, save_loglik, keep_re,
                  save_state):
    """NUTS in a float32 process: the map's anchors from the Laplace-marginal mode in a float64 child, the
    log-likelihood recentred on the warmed-up chains so that float32 keeps its digits."""
    rng_key = random.PRNGKey(seed)
    t0 = time.time()
    par = chain_method == "parallel" and chains > 1
    coords = NutsCoords(config, chains, seed, par, progress_bar, init_strategy, init_radius, init_z,
                        transport=bool((control or {}).get("transport", True)), anchors="dr")
    kernel = FastNUTS(sdt_model, coords=coords, target_accept_prob=target_accept_prob,
                      max_tree_depth=max_tree_depth, step_size=float(step_size))
    mcmc = MCMC(kernel, num_warmup=warmup, num_samples=samples, num_chains=chains, thinning=int(thinning),
                chain_method=chain_method, progress_bar=bool(progress_bar))
    init_params = coords.init_params(chains)
    with progress.nuts(chains, par, desc="NUTS warmup"):
        mcmc.warmup(rng_key, config, init_params=init_params, collect_warmup=False)
    rng_key = mcmc.post_warmup_state.rng_key
    offset = _nuts_ll_offset(config, mcmc.post_warmup_state, coords)
    config = {**config, "_ll_offset": jnp.asarray(offset)}
    mcmc.post_warmup_state = _refresh_potential(config, mcmc.post_warmup_state, coords)
    with progress.nuts(chains, par, desc="NUTS sampling"):
        mcmc.run(rng_key, config, init_params=None, extra_fields=("num_steps", "energy", "accept_prob", "diverging"))
    jax.block_until_ready(mcmc.last_state.z)
    elapsed = time.time() - t0
    samples_dict = mcmc.get_samples(group_by_chain=True)
    extra = mcmc.get_extra_fields(group_by_chain=True)
    extra_fields = {f: np.array(extra[f]) for f in ("num_steps", "energy", "accept_prob", "diverging") if f in extra}
    samples_np = {}
    for k, v in samples_dict.items():
        if k.startswith("z_") and not keep_re:
            continue
        if k == "log_lik" and not save_loglik:
            continue
        samples_np[k] = np.array(v)
    _restore_centred(samples_np)
    out = {"samples": samples_np, "extra_fields": extra_fields, "elapsed": float(elapsed)}
    if save_state:
        out["state"] = {"hmc": mcmc.last_state, "offset": np.asarray(offset), "aux": coords.aux_flat}
    return out


def _chain0(state):
    batched = jnp.ndim(state.potential_energy) > 0
    return jax.tree_util.tree_map(lambda x: x[0], state.z) if batched else state.z


def _nuts_ll_offset(config, state, coords):
    """Per-observation weighted log-likelihood at chain 0's warmed-up position (see
    drghmc.ll_offset): in float32 the potential is summed relative to it."""
    from numpyro.infer.util import constrain_fn
    ll = constrain_fn(sdt_model, (config,), {}, coords.model_z(_chain0(state)), return_deterministic=True)["log_lik"]
    counts = config.get("counts")
    return ll if counts is None else ll * jnp.asarray(counts, dtype=ll.dtype)


def _refresh_potential(config, state, coords):
    """The warmed-up state caches a potential computed without the offset; recompute it
    so the first transition compares like with like."""
    pe_fn = jax.jit(lambda z: coords.potential(config, z))
    if jnp.ndim(state.potential_energy) > 0:
        pe = jax.lax.map(pe_fn, state.z)
    else:
        pe = pe_fn(state.z)
    old = state.potential_energy
    pe = jnp.asarray(pe, dtype=old.dtype)
    sharding = getattr(old, "sharding", None)
    return state._replace(potential_energy=pe if sharding is None else jax.device_put(pe, sharding))
