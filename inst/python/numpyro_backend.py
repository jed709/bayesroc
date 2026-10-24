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
from numpyro.infer import MCMC, NUTS, init_to_uniform
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

    for pname, pc in (config.get("params") or {}).items():
        c = _centre_of(pc)
        if c is None:
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


def _link_fns(link_name):
    """(CDF, PDF) pair for a latent-distribution link."""
    if link_name == "logit":
        return jax_expit, lambda x: jax_expit(x) * (1.0 - jax_expit(x))
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



def _link_cell_fn(link_name):
    """(P(lo < Z < hi), pdf(lo), pdf(hi)) for the link's latent distribution, with each pdf
    the exact derivative of the CDF used for the probability. Probit uses the native kernel
    (Phi to ~1e-9 relative in both tails) when it is loaded."""
    if link_name == "probit":
        def cell(lo, hi):
            lo, hi = jnp.broadcast_arrays(jnp.asarray(lo), jnp.asarray(hi))
            D, neg_pdf_lo, pdf_hi = _phi_diff(lo, hi)
            return D, -neg_pdf_lo, pdf_hi
        return cell
    diff = _link_diff_fn(link_name)
    pdf = _link_fns(link_name)[1]
    return lambda lo, hi: (diff(lo, hi), pdf(lo), pdf(hi))


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


def _dw_dA_drho(r, rho):
    return jnp.where(jnp.abs(rho) < _RHO_MAX, 1.0 / jnp.sqrt(1.0 - r * r), 0.0)


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


def _acc_phi_deriv(x):
    return _acc_phi_and_deriv(x)[1]


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
    if _BVN_FFI and z_lo.dtype == jnp.float64:
        D, pdf_lo, pdf_hi = _native_call("probit_diff", 3, [z_lo, z_hi])
        return D, -pdf_lo, pdf_hi
    Ph, dh, Qh = _probit_triple(z_hi)
    Pl, dl, Ql = _probit_triple(z_lo)
    return jnp.where(z_lo >= 0, Ql - Qh, Ph - Pl), -dl, dh


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
            (hh[3] - lh[3] - hl[3] + ll[3]) * _dw_dA_drho(r, rho))


def _dw_cdf_grads_jax(h, k, rho):
    r, (c,) = _dw_corners(((h, k),), rho)
    Ph, dh, _ = _acc_phi_and_deriv(h)
    Pk, dk, _ = _acc_phi_and_deriv(k)
    return Ph * Pk + c[0], dh * Pk + c[1], Ph * dk + c[2], c[3] * _dw_dA_drho(r, rho)


_BVN_FFI = False
if os.environ.get("BAYESROC_BVN_FFI"):
    import ctypes
    try:
        _bvn_lib = ctypes.cdll.LoadLibrary(os.environ["BAYESROC_BVN_FFI"])
        for _t, _sym in (("bvn_rect", "BvnRect"), ("bvn_cdf", "BvnCdf"), ("vrdp2d", "Vrdp2d"),
                         ("bvn_cdf2", "BvnCdf2"), ("bvn_cdf3", "BvnCdf3"), ("bvn_cdf4", "BvnCdf4"),
                         ("probit_cell", "ProbitCell"),
                         ("probit_diff", "ProbitDiff"), ("cdp_cell", "CdpCell"), ("metad_cell", "MetadCell"),
                         ("pcq", "Pcq"), ("lnoise_cell", "LnoiseCell"),
                         ("tmap_fwd", "TmapFwd"), ("tmap_bwd", "TmapBwd"),
                         ("tmap_blk_fwd", "TmapBlkFwd"), ("tmap_blk_bwd", "TmapBlkBwd")):
            jax.ffi.register_ffi_target(f"bayesroc_{_t}", jax.ffi.pycapsule(getattr(_bvn_lib, f"Bayesroc{_sym}")),
                                        platform="cpu")
        for _t, _sym in (("bvn_rect", "BvnRect"), ("bvn_cdf", "BvnCdf"), ("vrdp2d", "Vrdp2d"),
                         ("bvn_cdf2", "BvnCdf2"), ("bvn_cdf3", "BvnCdf3"), ("bvn_cdf4", "BvnCdf4"),
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
    return link_name == "probit" and _BVN_FFI and x.dtype == jnp.float64


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


def _rect_tail_jax(a1, b1, a2, b2, rho):
    """Tail-accurate P(a1 < Z1 < b1, a2 < Z2 < b2; rho) and its derivatives in
    (a1, b1, a2, b2, rho): log-space Gauss-Legendre over the log-concave integrand
    phi(x) P(a2 < Z2 < b2 | x) around its mode; same algorithm as bvn_tail.h."""
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
    k = jnp.arange(4.0)
    vals = []
    for A0, B0 in ((left, m), (m, right)):
        w = B0 - A0
        A = A0[..., None] + w[..., None] * k / 4.0
        h = w[..., None] / 8.0
        c0 = A + h
        xs = c0[..., None] + h[..., None] * _GL32_X
        lw = jnp.log(jnp.maximum(h[..., None] * _GL32_W, 1e-300))
        v = lw + g(xs.reshape(xs.shape[:-2] + (-1,))).reshape(xs.shape)
        v = jnp.where((w > 0.0)[..., None, None] & jnp.isfinite(v), v,
                      jax.lax.stop_gradient(jnp.full_like(v, -jnp.inf)))
        vals.append(v.reshape(v.shape[:-2] + (-1,)))
    allv = jnp.concatenate(vals, axis=-1)
    lp = jax.scipy.special.logsumexp(allv, axis=-1)
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
    drho = phi2(b1, b2) - phi2(a1, b2) - phi2(b1, a2) + phi2(a1, a2)
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
    k = jnp.arange(4.0)
    vals = []
    for A0, B0 in ((left, m), (m, right)):
        w = B0 - A0
        A = A0[..., None] + w[..., None] * k / 4.0
        h = w[..., None] / 8.0
        xs = (A + h)[..., None] + h[..., None] * _GL32_X
        lw = jnp.log(jnp.maximum(h[..., None] * _GL32_W, 1e-300))
        v = lw + g(xs.reshape(xs.shape[:-2] + (-1,))).reshape(xs.shape)
        v = jnp.where((w > 0.0)[..., None, None] & jnp.isfinite(v), v,
                      jax.lax.stop_gradient(jnp.full_like(v, -jnp.inf)))
        vals.append(v.reshape(v.shape[:-2] + (-1,)))
    lp = jax.scipy.special.logsumexp(jnp.concatenate(vals, axis=-1), axis=-1)
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


def _bvn_cdf_multi(pairs, extras, rho):
    """_bvn_cdf for (h, k) pairs sharing rho, plus accurate marginals. Returns, per pair,
    (P, dP/dh, dP/dk, dP/drho, Q(h), Phi'(h), Q(k), Phi'(k)) and, per extra x, (Q(x), Phi'(x)),
    where Q is the tail mass min(Phi, 1 - Phi). One native call when available."""
    n, m = len(pairs), len(extras)
    arrs = jnp.broadcast_arrays(*[jnp.asarray(a) for pair in pairs for a in pair],
                                *[jnp.asarray(x) for x in extras], jnp.asarray(rho))
    hs = arrs[0:2 * n:2]
    ks = arrs[1:2 * n:2]
    xs = arrs[2 * n:2 * n + m]
    r = arrs[-1]
    if _native_ok(r) and (n, m) in ((2, 1), (3, 0), (4, 2)):
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


def _bvn_cdf(h, k, rho):
    """P(Z1 < h, Z2 < k; rho) and its exact derivatives in (h, k, rho): Drezner-Wesolowsky with
    the Gauss-Legendre order tiered on |rho|, and the log-space tail path below 1e-6."""
    args = jnp.broadcast_arrays(*[jnp.asarray(a) for a in (h, k, rho)])
    if _native_ok(args[0]):
        return _native_call("bvn_cdf", 4, args)
    h, k, r = args
    out = _dw_cdf_grads_jax(h, k, r)
    ninf = jnp.full_like(h, -_OPEN)
    t = _with_rect_tail((out[0], jnp.zeros_like(h), out[1], jnp.zeros_like(h), out[2], out[3]),
                        ninf, h, ninf, k, r)
    return t[0], t[2], t[4], t[5]


def _phi(x):
    """Standard normal PDF (double precision)."""
    return jnp.exp(-0.5 * x**2) / jnp.sqrt(2.0 * jnp.pi)

# =============================================================================
# Fully-vectorized batch loglik (no vmap)
# =============================================================================

def _make_fused_batch_loglik(K, gap_link_is_softplus, link_name="probit"):
    """Fully-vectorized batch log-likelihood with custom JVP.

    Operates on entire [N]-shaped arrays directly (no per-observation vmap).
    Thresholds are built via cumsum, and
    gap gradients use a precomputed contribution mask.
    """
    n_thresh = K - 1
    n_gaps = n_thresh - 1
    mid_idx = (n_thresh - 1) // 2
    n_upper = n_thresh - mid_idx - 1

    _CELL = _link_cell_fn(link_name)

    # --- Precompute static structures (Python-time, not trace-time) ---


    _gap_mask = np.zeros((n_gaps, n_thresh), dtype=np.float64)
    for g in range(n_gaps):
        if g < n_upper:
            for t in range(mid_idx + 1 + g, n_thresh):
                _gap_mask[g, t] = 1.0
        else:
            thresh_max = mid_idx - 1 - (g - n_upper)
            for t in range(0, thresh_max + 1):
                _gap_mask[g, t] = -1.0
    gap_mask = _gap_mask


    def _build_thresh(thresh_mid, gap_vals):
        """Build [N, n_thresh] thresholds from [N] mid and [N, n_gaps] gap values."""
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

    @custom_jvp
    def _batch_fused(y, mu, sigma_val, thresh_mid, gaps):
        gap_fn = jax.nn.softplus if gap_link_is_softplus else jnp.exp
        gap_vals = gap_fn(gaps)

        thresh = _build_thresh(thresh_mid, gap_vals)

        y_idx = y - 1
        k_hi = jnp.clip(y_idx, 0, n_thresh - 1)
        k_lo = jnp.clip(y_idx - 1, 0, n_thresh - 1)

        obs_idx = jnp.arange(y.shape[0])
        thresh_hi = thresh[obs_idx, k_hi]
        thresh_lo = thresh[obs_idx, k_lo]

        INF = _OPEN
        z_hi = jnp.where(y_idx < K - 1, (thresh_hi - mu) / sigma_val, INF)
        z_lo = jnp.where(y_idx > 0, (thresh_lo - mu) / sigma_val, -INF)

        if _use_probit_native(link_name, z_lo):
            return _probit_cell_native(z_lo, z_hi)[0]
        p = jnp.clip(_CELL(z_lo, z_hi)[0], _pfloor())
        return jnp.log(p)

    @_batch_fused.defjvp
    def _batch_fused_jvp(primals, tangents):
        y, mu, sigma_val, thresh_mid, gaps = primals
        _, d_mu, d_sigma_val, d_thresh_mid, d_gaps = tangents

        if gap_link_is_softplus:
            gap_vals = jax.nn.softplus(gaps)
            gap_derivs = jax.nn.sigmoid(gaps)
        else:
            gap_vals = jnp.exp(gaps)
            gap_derivs = gap_vals

        thresh = _build_thresh(thresh_mid, gap_vals)

        y_idx = y - 1
        k_hi = jnp.clip(y_idx, 0, n_thresh - 1)
        k_lo = jnp.clip(y_idx - 1, 0, n_thresh - 1)

        obs_idx = jnp.arange(y.shape[0])
        thresh_hi = thresh[obs_idx, k_hi]
        thresh_lo = thresh[obs_idx, k_lo]

        INF = _OPEN
        z_hi = jnp.where(y_idx < K - 1, (thresh_hi - mu) / sigma_val, INF)
        z_lo = jnp.where(y_idx > 0, (thresh_lo - mu) / sigma_val, -INF)

        if _use_probit_native(link_name, z_lo):
            val, g_lo, g_hi = _probit_cell_native(z_lo, z_hi)
            d_z_hi = jnp.where(y_idx < K - 1, g_hi, 0.0)
            d_z_lo = jnp.where(y_idx > 0, g_lo, 0.0)
        else:
            D, pdf_lo, pdf_hi = _CELL(z_lo, z_hi)
            p = jnp.clip(D, _pfloor())
            val = jnp.log(p)
            d_z_hi = jnp.where(y_idx < K - 1, pdf_hi / p, 0.0)
            d_z_lo = jnp.where(y_idx > 0, -pdf_lo / p, 0.0)

        d_lp_d_mu = -(d_z_lo + d_z_hi) / sigma_val
        d_lp_d_sigma = -(d_z_lo * z_lo + d_z_hi * z_hi) / sigma_val

        has_hi = (y_idx < K - 1)
        has_lo = (y_idx > 0)
        d_thresh_k_hi = jnp.where(has_hi, d_z_hi / sigma_val, 0.0)
        d_thresh_k_lo = jnp.where(has_lo, d_z_lo / sigma_val, 0.0)

        d_lp_d_thresh_mid = d_thresh_k_hi + d_thresh_k_lo

        _gm = jnp.asarray(gap_mask)
        mask_hi = _gm[:, k_hi]
        mask_lo = _gm[:, k_lo]

        d_lp_d_gap_val = mask_hi * d_thresh_k_hi[None, :] + mask_lo * d_thresh_k_lo[None, :]
        d_lp_d_gaps = (d_lp_d_gap_val * gap_derivs.T).T

        tangent_out = (d_lp_d_mu * d_mu +
                       d_lp_d_sigma * d_sigma_val +
                       d_lp_d_thresh_mid * d_thresh_mid +
                       jnp.sum(d_lp_d_gaps * d_gaps, axis=1))

        return val, tangent_out

    return _batch_fused


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
    D = jnp.where(ul >= 0, ql - qh, Ph - Pl)
    ia = jnp.where(Qa > fl, 1.0 / jnp.maximum(Qa, fl), 0.0)
    ib = jnp.where(D > fl, 1.0 / jnp.maximum(D, fl), 0.0)
    ic = jnp.where(Qc > fl, 1.0 / jnp.maximum(Qc, fl), 0.0)
    lp = jnp.log(jnp.maximum(Qa, fl)) + jnp.log(jnp.maximum(D, fl) / jnp.maximum(Qc, fl))
    bl, bh = -dl * ib, dh * ib
    return lp, -s * da * ia, jnp.where(old, bl, -bh), jnp.where(old, bh, -bl), s * dc * ic


def _metad_cell(z_a, z_lo, z_hi, z_c, old):
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

    def edge(z0):
        def step(_, z):
            g, _ = gh(z)
            return z - (h(z) - target) / g
        return jax.lax.fori_loop(0, 40, step, z0)

    zl = edge(zm - np.sqrt(2.0 * _PCQ_DROP))
    zr = edge(zm + np.sqrt(2.0 * _PCQ_DROP))
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


@custom_jvp
def pcq_logint(a, c):
    """log int phi(z) prod_j Phi(a_j z + c_j) dz over the trailing axis of [N, n] slopes a and shifts c."""
    return _pcq_raw(a, c)[0]


@pcq_logint.defjvp
def _pcq_logint_jvp(primals, tangents):
    a, c = primals
    ta, tc = tangents
    lp, dc, da = _pcq_raw(a, c)
    return lp, jnp.sum(dc * tc + da * ta, -1)


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
    zl = -L * sw
    d2 = -1.0 - L * s * sw - L * (x + L) * sw * sw
    z = -zl / d2

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

    z, _, _ = jax.lax.fori_loop(0, 40, body, (z, zl, jnp.zeros_like(z)))
    w = jnp.exp(l + s * z)
    x = a - w
    L = _ln_lam(x)
    sw = s * w
    return z, -1.0 - L * s * sw - L * (x + L) * sw * sw


def _ln_interval(lo, hi):
    return jnp.where(lo >= 0.0, jax.scipy.special.ndtr(-lo) - jax.scipy.special.ndtr(-hi),
                     jax.scipy.special.ndtr(hi) - jax.scipy.special.ndtr(lo))


def _ln_kdiff(u, Pa, Qa):
    return jnp.where(u >= 0.0, jax.scipy.special.ndtr(-u) - Qa, Pa - jax.scipy.special.ndtr(u))


def _ln_cum(a, l, s, Pa, Qa, pa):
    """(C, K, dC/da, dK/da, dC/dl, dC/ds) for C = P(T > e^(l + s Z)), T ~ N(a, 1), and K = Phi(a) - C, each
    accumulated pointwise; dK/dl = -dC/dl and dK/ds = -dC/ds."""
    zm, d2 = _ln_mode(a, l, s)
    zs = (-jnp.log(s) - l) / s
    agh = zs >= zm + _LN_R
    pdf = lambda t: jnp.exp(-0.5 * t * t - _LOG_SQRT_2PI)
    ndtr = jax.scipy.special.ndtr

    sc = jnp.where(agh, 1.0 / jnp.sqrt(-d2), 1.0)
    h0 = jnp.where(agh, -0.5 * zm * zm + _log_q(-(a - jnp.exp(l + s * zm))), 0.0)
    zg = zm[..., None] + sc[..., None] * _LN_GH_X
    wg = jnp.exp(l[..., None] + s[..., None] * zg)
    ug = a[..., None] - wg
    eg = jnp.exp(-0.5 * zg * zg + _LN_GH_LW - h0[..., None])
    pu = pdf(ug)
    cg = sc * jnp.exp(h0) / np.sqrt(2.0 * np.pi)
    g_agh = [cg * jnp.sum(v, -1) for v in (eg * ndtr(ug), eg * _ln_kdiff(ug, Pa[..., None], Qa[..., None]),
                                            eg * pu, eg * (pa[..., None] - pu), -eg * pu * wg, -eg * pu * wg * zg)]

    zc = jnp.minimum(zm, zs)
    lo = zc - _LN_R
    hi = jnp.minimum(zc + _LN_R, zs)
    h = 0.5 * jnp.maximum(hi - lo, 0.0)
    z = (lo + h)[..., None] + h[..., None] * _LN_GL_X
    w = jnp.exp(l[..., None] + s[..., None] * z)
    u = a[..., None] - w
    pz = _LN_GL_W * pdf(z)
    pu = pdf(u)
    cz = h
    vz = [cz * jnp.sum(v, -1) for v in (pz * ndtr(u), pz * _ln_kdiff(u, Pa[..., None], Qa[..., None]), pz * pu,
                                         pz * (pa[..., None] - pu), -pz * pu * w, -pz * pu * w * z)]

    es = 1.0 / s - a
    elo = jnp.maximum(es, -_LN_R)
    net = es - jnp.where(zs < 0.0, _ln_lam(zs), 0.0)
    ehi = jnp.where((es > 0.0) & (net * _LN_R > 40.0), es + 40.0 / jnp.where(net > 0.0, net, 1.0),
                    jnp.maximum(es, 0.0) + _LN_R)
    he = 0.5 * jnp.maximum(ehi - elo, 0.0)
    e = (elo + he)[..., None] + he[..., None] * _LN_GL_X
    ae = jnp.maximum(a[..., None] + e, 1e-300)
    Z = (jnp.log(ae) - l[..., None]) / s[..., None]
    pe = _LN_GL_W * pdf(e)
    pZ = pdf(Z) / s[..., None]
    zsb = zs[..., None]
    diff = jnp.where(zsb >= 0.0, ndtr(-zsb) - ndtr(-Z), ndtr(Z) - ndtr(zsb))
    ve = [he * jnp.sum(v, -1) for v in (pe * diff, pe * ndtr(-Z), pe * pZ / ae, -pe * pZ / ae, -pe * pZ,
                                        -pe * pZ * Z)]
    Qzs = ndtr(-zs)
    kc = _ln_interval(-a, elo) * Qzs
    split = [vz[0] + ve[0], vz[1] + ve[1] + kc, vz[2] + ve[2], vz[3] + ve[3] + pa * Qzs, vz[4] + ve[4],
             vz[5] + ve[5]]
    return [jnp.where(agh, x, y) for x, y in zip(g_agh, split)]


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
    a, ln, lf, s = jnp.broadcast_arrays(*(jnp.asarray(v) for v in (a, ln, lf, s)))
    if _native_ok(a):
        return _native_call("lnoise_cell", 5, (a, ln, lf, s))
    return _lnoise_cell_jax(a, ln, lf, s)


@custom_jvp
def lnoise_cell_lp(a, ln, lf, s):
    """log P of a lognormal meta-noise cell (Shekhar & Rahnev 2021); see _lnoise_cell_jax."""
    return _lnoise_raw(a, ln, lf, s)[0]


@lnoise_cell_lp.defjvp
def _lnoise_cell_lp_jvp(primals, tangents):
    lp, da, dln, dlf, ds = _lnoise_raw(*primals)
    ta, tln, tlf, ts = tangents
    return lp, da * ta + jnp.where(jnp.isneginf(primals[1]), 0.0, dln * tln) + \
        jnp.where(jnp.isposinf(primals[2]), 0.0, dlf * tlf) + ds * ts


def _make_fused_lnoise_loglik(K, gap_link_is_softplus):
    """Lognormal meta-noise (evsd with sigma_meta, Shekhar & Rahnev 2021) log-likelihood with custom JVP:
    the decision on x ~ N(mu, 1) against the middle threshold c0, the confidence criteria on the side of
    the decision at c0 + (c_k - c0) exp(s z), one z ~ N(0, 1) per trial."""
    n_thresh, mid_idx, gap_mask, build = _thresh_layout(K)
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
        D_near = jnp.where(open_near, 1.0, jnp.where(old, D_lo, D_hi))
        D_far = jnp.where(open_far, 1.0, jnp.where(old, D_hi, D_lo))
        ln = jnp.where(open_near, -jnp.inf, jnp.log(D_near))
        lf = jnp.where(open_far, jnp.inf, jnp.log(D_far))
        out = _lnoise_raw(sgn * (mu - thresh_mid), ln, lf, sm)
        return out, (old, sgn, open_near, open_far, D_near, D_far, k_lo, k_hi, gap_vals)

    @custom_jvp
    def _lnoise(y, mu, sm, thresh_mid, gaps):
        return _pieces(y, mu, sm, thresh_mid, gaps)[0][0]

    @_lnoise.defjvp
    def _lnoise_jvp(primals, tangents):
        y, mu, sm, thresh_mid, gaps = primals
        _, t_mu, t_sm, t_mid, t_gaps = tangents
        (lp, da, dln, dlf, ds), (old, sgn, open_near, open_far, D_near, D_far, k_lo, k_hi, gap_vals) =             _pieces(*primals)
        g_near = jnp.where(open_near, 0.0, dln / D_near)
        g_far = jnp.where(open_far, 0.0, dlf / D_far)
        g_lo = sgn * jnp.where(old, g_near, g_far)
        g_hi = sgn * jnp.where(old, g_far, g_near)
        d_c = -sgn * (da + g_near + g_far)
        gap_d = jax.nn.sigmoid(gaps) if gap_link_is_softplus else gap_vals
        gm = jnp.asarray(gap_mask, dtype=gaps.dtype)
        d_gaps = (gm[:, k_lo].T * g_lo[:, None] + gm[:, k_hi].T * g_hi[:, None]) * gap_d
        tan = (sgn * da * t_mu + ds * t_sm + (d_c + g_lo + g_hi) * t_mid + jnp.sum(d_gaps * t_gaps, axis=1))
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


def _make_fused_metad_loglik(K, gap_link_is_softplus, q, relative):
    """Meta-d' log-likelihood over [N] trials with custom JVP: the type-1 decision and the
    meta-level rating truncated at the meta criterion, the confidence criteria shifted by
    (M - 1)(c - q d) under the relative anchor."""
    n_thresh, mid_idx, gap_mask, build = _thresh_layout(K)
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
        out = _metad_cell(c - a * d, z_lo, z_hi, c + off, old)
        return out, (old, M, a_c, c_mid, k_lo, k_hi, gap_vals)

    @custom_jvp
    def _metad(y, d, m_old, m_new, a, thresh_mid, gaps):
        return _pieces(y, d, m_old, m_new, a, thresh_mid, gaps)[0][0]

    @_metad.defjvp
    def _metad_jvp(primals, tangents):
        y, d, m_old, m_new, a, thresh_mid, gaps = primals
        _, t_d, t_mo, t_mn, _, t_mid, t_gaps = tangents
        (lp, g_a, g_lo, g_hi, g_c), (old, M, a_c, c_mid, k_lo, k_hi, gap_vals) = _pieces(*primals)
        g_lo = jnp.where(y > 1, g_lo, 0.0)
        g_hi = jnp.where(y < K, g_hi, 0.0)
        G = g_lo + g_hi + g_c
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
        d_gaps = (gm[:, k_lo].T * g_lo[:, None] + gm[:, k_hi].T * g_hi[:, None]) * gap_d
        tan = (d_d * t_d + jnp.where(old, d_M, 0.0) * t_mo + jnp.where(old, 0.0, d_M) * t_mn +
               (d_mid + g_lo + g_hi) * t_mid + jnp.sum(d_gaps * t_gaps, axis=1))
        return lp, tan

    return _metad


def _make_fused_gnoise_loglik(K, gap_link_is_softplus):
    """Gaussian meta-noise (evsd with sigma_meta) log-likelihood with custom JVP: the decision on
    x ~ N(mu, 1) against the middle threshold, the rating on y = x + e, e ~ N(0, s^2), a bivariate
    normal rectangle with correlation 1 / sqrt(1 + s^2); y on the other side of the decision
    criterion gives the lowest confidence."""
    n_thresh, mid_idx, gap_mask, build = _thresh_layout(K)
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
        z_c = thresh_mid - mu
        z1_lo = jnp.where(old, z_c, -_OPEN)
        z1_hi = jnp.where(old, _OPEN, z_c)
        out = _bvn_rect(z1_lo, z1_hi, z_lo, z_hi, inv_tau)
        return out, (old, open_lo, open_hi, z_lo, z_hi, inv_tau, k_lo, k_hi, gap_vals)

    @custom_jvp
    def _gnoise(y, mu, sm, thresh_mid, gaps):
        p = _pieces(y, mu, sm, thresh_mid, gaps)[0][0]
        return jnp.log(jnp.clip(p, _pfloor()))

    @_gnoise.defjvp
    def _gnoise_jvp(primals, tangents):
        y, mu, sm, thresh_mid, gaps = primals
        _, t_mu, t_sm, t_mid, t_gaps = tangents
        (p, d1l, d1h, d2l, d2h, drho), (old, open_lo, open_hi, z_lo, z_hi, inv_tau, k_lo, k_hi, gap_vals) = \
            _pieces(*primals)
        prob = jnp.clip(p, _pfloor())
        inv_p = jnp.where(p > _pfloor(), 1.0 / prob, 0.0)
        g_c = jnp.where(old, d1l, d1h) * inv_p
        g_lo = jnp.where(open_lo, 0.0, d2l * inv_p)
        g_hi = jnp.where(open_hi, 0.0, d2h * inv_p)
        zl = jnp.where(open_lo, 0.0, z_lo)
        zh = jnp.where(open_hi, 0.0, z_hi)
        d_mu = -g_c - (g_lo + g_hi) * inv_tau
        d_tau = -(g_lo * zl + g_hi * zh) * inv_tau - drho * inv_p * inv_tau * inv_tau
        d_sm = d_tau * sm * inv_tau
        gap_d = jax.nn.sigmoid(gaps) if gap_link_is_softplus else gap_vals
        gm = jnp.asarray(gap_mask, dtype=gaps.dtype)
        a_lo, a_hi = g_lo * inv_tau, g_hi * inv_tau
        d_gaps = (gm[:, k_lo].T * a_lo[:, None] + gm[:, k_hi].T * a_hi[:, None]) * gap_d
        tan = (d_mu * t_mu + d_sm * t_sm + (g_c + a_lo + a_hi) * t_mid + jnp.sum(d_gaps * t_gaps, axis=1))
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


def fused_loglik_metad(y, K, d, m_old, m_new, is_old, thresh_mid, gaps, gap_link="log", q=0.5, relative=False):
    """Meta-d' log-likelihood; gaps [n_gaps, N] effective log-gaps, or [n_gaps] if shared."""
    N = y.shape[0]
    is_softplus = gap_link == "softplus"
    gaps_per_obs = jnp.broadcast_to(gaps[None, :], (N, K - 2)) if gaps.ndim == 1 else gaps.T
    key = ("metad", int(K), bool(is_softplus), float(q), bool(relative))
    if key not in _fused_cache:
        _fused_cache[key] = _make_fused_metad_loglik(int(K), bool(is_softplus), float(q), bool(relative))
    return _fused_cache[key](y, d, m_old, m_new, is_old, thresh_mid, gaps_per_obs)


def _make_fused_batch_dpsdt_loglik(K, gap_link_is_softplus, link_name="probit"):
    """Fully-vectorized batch log-likelihood for DPSDT with custom JVP.

    DPSDT: P = lambda * I(y==K) + (1-lambda) * cell_p.
    Operates on entire [N]-shaped arrays directly.
    """
    n_thresh = K - 1
    n_gaps = n_thresh - 1
    mid_idx = (n_thresh - 1) // 2
    n_upper = n_thresh - mid_idx - 1

    _CELL = _link_cell_fn(link_name)
    _DIFF = lambda lo, hi: _CELL(lo, hi)[0]

    _gap_mask = np.zeros((n_gaps, n_thresh), dtype=np.float64)
    for g in range(n_gaps):
        if g < n_upper:
            for t in range(mid_idx + 1 + g, n_thresh):
                _gap_mask[g, t] = 1.0
        else:
            thresh_max = mid_idx - 1 - (g - n_upper)
            for t in range(0, thresh_max + 1):
                _gap_mask[g, t] = -1.0
    gap_mask = _gap_mask

    def _build_thresh(thresh_mid, gap_vals):
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

    @custom_jvp
    def _batch_fused(y, mu, sigma_val, lambda_val, thresh_mid, gaps):
        gap_fn = jax.nn.softplus if gap_link_is_softplus else jnp.exp
        gap_vals = gap_fn(gaps)
        thresh = _build_thresh(thresh_mid, gap_vals)

        y_idx = y - 1
        k_hi = jnp.clip(y_idx, 0, n_thresh - 1)
        k_lo = jnp.clip(y_idx - 1, 0, n_thresh - 1)

        obs_idx = jnp.arange(y.shape[0])
        thresh_hi = thresh[obs_idx, k_hi]
        thresh_lo = thresh[obs_idx, k_lo]

        INF = _OPEN
        z_hi = jnp.where(y_idx < K - 1, (thresh_hi - mu) / sigma_val, INF)
        z_lo = jnp.where(y_idx > 0, (thresh_lo - mu) / sigma_val, -INF)
        cell_p = jnp.clip(_DIFF(z_lo, z_hi), _pfloor())

        is_top = (y == K).astype(_FLOAT)
        p = jnp.clip(lambda_val * is_top + (1.0 - lambda_val) * cell_p, _pfloor())
        return jnp.log(p)

    @_batch_fused.defjvp
    def _batch_fused_jvp(primals, tangents):
        y, mu, sigma_val, lambda_val, thresh_mid, gaps = primals
        _, d_mu, d_sigma_val, d_lambda, d_thresh_mid, d_gaps = tangents

        if gap_link_is_softplus:
            gap_vals = jax.nn.softplus(gaps)
            gap_derivs = jax.nn.sigmoid(gaps)
        else:
            gap_vals = jnp.exp(gaps)
            gap_derivs = gap_vals

        thresh = _build_thresh(thresh_mid, gap_vals)

        y_idx = y - 1
        k_hi = jnp.clip(y_idx, 0, n_thresh - 1)
        k_lo = jnp.clip(y_idx - 1, 0, n_thresh - 1)

        obs_idx = jnp.arange(y.shape[0])
        thresh_hi = thresh[obs_idx, k_hi]
        thresh_lo = thresh[obs_idx, k_lo]

        INF = _OPEN
        z_hi = jnp.where(y_idx < K - 1, (thresh_hi - mu) / sigma_val, INF)
        z_lo = jnp.where(y_idx > 0, (thresh_lo - mu) / sigma_val, -INF)
        _D_cell_p, _pl_cell_p, _ph_cell_p = _CELL(z_lo, z_hi)
        cell_p = jnp.clip(_D_cell_p, _pfloor())

        is_top = (y == K).astype(_FLOAT)
        p = jnp.clip(lambda_val * is_top + (1.0 - lambda_val) * cell_p, _pfloor())
        val = jnp.log(p)

        d_lp_d_lambda = (is_top - cell_p) / p

        d_lp_d_cell = (1.0 - lambda_val) / p

        pdf_hi = _ph_cell_p
        pdf_lo = _pl_cell_p
        d_cell_d_z_hi = jnp.where(y_idx < K - 1, pdf_hi, 0.0)
        d_cell_d_z_lo = jnp.where(y_idx > 0, -pdf_lo, 0.0)

        d_z_hi_eff = d_lp_d_cell * d_cell_d_z_hi
        d_z_lo_eff = d_lp_d_cell * d_cell_d_z_lo

        d_lp_d_mu = -(d_z_lo_eff + d_z_hi_eff) / sigma_val
        d_lp_d_sigma = -(d_z_lo_eff * z_lo + d_z_hi_eff * z_hi) / sigma_val

        has_hi = (y_idx < K - 1)
        has_lo = (y_idx > 0)
        d_thresh_k_hi = jnp.where(has_hi, d_z_hi_eff / sigma_val, 0.0)
        d_thresh_k_lo = jnp.where(has_lo, d_z_lo_eff / sigma_val, 0.0)

        d_lp_d_thresh_mid = d_thresh_k_hi + d_thresh_k_lo

        _gm = jnp.asarray(gap_mask)
        mask_hi = _gm[:, k_hi]
        mask_lo = _gm[:, k_lo]

        d_lp_d_gap_val = mask_hi * d_thresh_k_hi[None, :] + mask_lo * d_thresh_k_lo[None, :]
        d_lp_d_gaps = (d_lp_d_gap_val * gap_derivs.T).T

        tangent_out = (d_lp_d_mu * d_mu +
                       d_lp_d_sigma * d_sigma_val +
                       d_lp_d_lambda * d_lambda +
                       d_lp_d_thresh_mid * d_thresh_mid +
                       jnp.sum(d_lp_d_gaps * d_gaps, axis=1))

        return val, tangent_out

    return _batch_fused


def _make_fused_batch_mixture_loglik(K, gap_link_is_softplus, link_name="probit"):
    """Fully-vectorized batch log-likelihood for mixture with custom JVP.

    Mixture: P = lambda * p1(mu1,sig1) + (1-lambda) * p2(mu2,sig2).
    Operates on entire [N]-shaped arrays directly.
    """
    n_thresh = K - 1
    n_gaps = n_thresh - 1
    mid_idx = (n_thresh - 1) // 2
    n_upper = n_thresh - mid_idx - 1

    _CELL = _link_cell_fn(link_name)
    _DIFF = lambda lo, hi: _CELL(lo, hi)[0]

    _gap_mask = np.zeros((n_gaps, n_thresh), dtype=np.float64)
    for g in range(n_gaps):
        if g < n_upper:
            for t in range(mid_idx + 1 + g, n_thresh):
                _gap_mask[g, t] = 1.0
        else:
            thresh_max = mid_idx - 1 - (g - n_upper)
            for t in range(0, thresh_max + 1):
                _gap_mask[g, t] = -1.0
    gap_mask = _gap_mask

    def _build_thresh(thresh_mid, gap_vals):
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

    @custom_jvp
    def _batch_fused(y, mu1, sigma1_val, mu2, sigma2_val, lambda_val, thresh_mid, gaps):
        gap_fn = jax.nn.softplus if gap_link_is_softplus else jnp.exp
        gap_vals = gap_fn(gaps)
        thresh = _build_thresh(thresh_mid, gap_vals)

        y_idx = y - 1
        k_hi = jnp.clip(y_idx, 0, n_thresh - 1)
        k_lo = jnp.clip(y_idx - 1, 0, n_thresh - 1)

        obs_idx = jnp.arange(y.shape[0])
        thresh_hi = thresh[obs_idx, k_hi]
        thresh_lo = thresh[obs_idx, k_lo]

        INF = _OPEN
        z1_hi = jnp.where(y_idx < K - 1, (thresh_hi - mu1) / sigma1_val, INF)
        z1_lo = jnp.where(y_idx > 0, (thresh_lo - mu1) / sigma1_val, -INF)
        p1 = jnp.clip(_DIFF(z1_lo, z1_hi), _pfloor())
        z2_hi = jnp.where(y_idx < K - 1, (thresh_hi - mu2) / sigma2_val, INF)
        z2_lo = jnp.where(y_idx > 0, (thresh_lo - mu2) / sigma2_val, -INF)
        p2 = jnp.clip(_DIFF(z2_lo, z2_hi), _pfloor())

        p = jnp.clip(lambda_val * p1 + (1.0 - lambda_val) * p2, _pfloor())
        return jnp.log(p)

    @_batch_fused.defjvp
    def _batch_fused_jvp(primals, tangents):
        y, mu1, sigma1_val, mu2, sigma2_val, lambda_val, thresh_mid, gaps = primals
        _, d_mu1, d_sigma1, d_mu2, d_sigma2, d_lambda, d_thresh_mid, d_gaps = tangents

        if gap_link_is_softplus:
            gap_vals = jax.nn.softplus(gaps)
            gap_derivs = jax.nn.sigmoid(gaps)
        else:
            gap_vals = jnp.exp(gaps)
            gap_derivs = gap_vals

        thresh = _build_thresh(thresh_mid, gap_vals)

        y_idx = y - 1
        k_hi = jnp.clip(y_idx, 0, n_thresh - 1)
        k_lo = jnp.clip(y_idx - 1, 0, n_thresh - 1)

        obs_idx = jnp.arange(y.shape[0])
        thresh_hi = thresh[obs_idx, k_hi]
        thresh_lo = thresh[obs_idx, k_lo]

        INF = _OPEN

        z1_hi = jnp.where(y_idx < K - 1, (thresh_hi - mu1) / sigma1_val, INF)
        z1_lo = jnp.where(y_idx > 0, (thresh_lo - mu1) / sigma1_val, -INF)
        _D_p1, _pl_p1, _ph_p1 = _CELL(z1_lo, z1_hi)
        p1 = jnp.clip(_D_p1, _pfloor())

        z2_hi = jnp.where(y_idx < K - 1, (thresh_hi - mu2) / sigma2_val, INF)
        z2_lo = jnp.where(y_idx > 0, (thresh_lo - mu2) / sigma2_val, -INF)
        _D_p2, _pl_p2, _ph_p2 = _CELL(z2_lo, z2_hi)
        p2 = jnp.clip(_D_p2, _pfloor())

        p = jnp.clip(lambda_val * p1 + (1.0 - lambda_val) * p2, _pfloor())
        val = jnp.log(p)

        d_lp_d_lambda = (p1 - p2) / p

        d_lp_d_p1 = lambda_val / p
        d_lp_d_p2 = (1.0 - lambda_val) / p

        pdf1_hi = _ph_p1
        pdf1_lo = _pl_p1
        dp1_dz1_hi = jnp.where(y_idx < K - 1, pdf1_hi, 0.0)
        dp1_dz1_lo = jnp.where(y_idx > 0, -pdf1_lo, 0.0)

        pdf2_hi = _ph_p2
        pdf2_lo = _pl_p2
        dp2_dz2_hi = jnp.where(y_idx < K - 1, pdf2_hi, 0.0)
        dp2_dz2_lo = jnp.where(y_idx > 0, -pdf2_lo, 0.0)

        dz1_hi_eff = d_lp_d_p1 * dp1_dz1_hi
        dz1_lo_eff = d_lp_d_p1 * dp1_dz1_lo
        d_lp_d_mu1 = -(dz1_lo_eff + dz1_hi_eff) / sigma1_val
        d_lp_d_sigma1 = -(dz1_lo_eff * z1_lo + dz1_hi_eff * z1_hi) / sigma1_val

        dz2_hi_eff = d_lp_d_p2 * dp2_dz2_hi
        dz2_lo_eff = d_lp_d_p2 * dp2_dz2_lo
        d_lp_d_mu2 = -(dz2_lo_eff + dz2_hi_eff) / sigma2_val
        d_lp_d_sigma2 = -(dz2_lo_eff * z2_lo + dz2_hi_eff * z2_hi) / sigma2_val

        has_hi = (y_idx < K - 1)
        has_lo = (y_idx > 0)

        d_thresh_k_hi = (jnp.where(has_hi, dz1_hi_eff / sigma1_val, 0.0) +
                         jnp.where(has_hi, dz2_hi_eff / sigma2_val, 0.0))
        d_thresh_k_lo = (jnp.where(has_lo, dz1_lo_eff / sigma1_val, 0.0) +
                         jnp.where(has_lo, dz2_lo_eff / sigma2_val, 0.0))

        d_lp_d_thresh_mid = d_thresh_k_hi + d_thresh_k_lo

        _gm = jnp.asarray(gap_mask)
        mask_hi = _gm[:, k_hi]
        mask_lo = _gm[:, k_lo]

        d_lp_d_gap_val = mask_hi * d_thresh_k_hi[None, :] + mask_lo * d_thresh_k_lo[None, :]
        d_lp_d_gaps = (d_lp_d_gap_val * gap_derivs.T).T

        tangent_out = (d_lp_d_mu1 * d_mu1 +
                       d_lp_d_sigma1 * d_sigma1 +
                       d_lp_d_mu2 * d_mu2 +
                       d_lp_d_sigma2 * d_sigma2 +
                       d_lp_d_lambda * d_lambda +
                       d_lp_d_thresh_mid * d_thresh_mid +
                       jnp.sum(d_lp_d_gaps * d_gaps, axis=1))

        return val, tangent_out

    return _batch_fused


def _make_fused_batch_source_mixture_loglik(K, gap_link_is_softplus, link_name="probit"):
    """Fully-vectorized batch log-likelihood for source_mixture with custom JVP.

    Source mixture: P = lambda * p_att(d_attended) + (1-lambda) * p_noise(mu=0).
    Both states use sigma=1. Operates on entire [N]-shaped arrays directly.
    """
    n_thresh = K - 1
    n_gaps = n_thresh - 1
    mid_idx = (n_thresh - 1) // 2
    n_upper = n_thresh - mid_idx - 1

    _CELL = _link_cell_fn(link_name)
    _DIFF = lambda lo, hi: _CELL(lo, hi)[0]

    _gap_mask = np.zeros((n_gaps, n_thresh), dtype=np.float64)
    for g in range(n_gaps):
        if g < n_upper:
            for t in range(mid_idx + 1 + g, n_thresh):
                _gap_mask[g, t] = 1.0
        else:
            thresh_max = mid_idx - 1 - (g - n_upper)
            for t in range(0, thresh_max + 1):
                _gap_mask[g, t] = -1.0
    gap_mask = _gap_mask

    def _build_thresh(thresh_mid, gap_vals):
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

    @custom_jvp
    def _batch_fused(y, d_attended, lambda_val, thresh_mid, gaps):
        gap_fn = jax.nn.softplus if gap_link_is_softplus else jnp.exp
        gap_vals = gap_fn(gaps)
        thresh = _build_thresh(thresh_mid, gap_vals)

        y_idx = y - 1
        k_hi = jnp.clip(y_idx, 0, n_thresh - 1)
        k_lo = jnp.clip(y_idx - 1, 0, n_thresh - 1)

        obs_idx = jnp.arange(y.shape[0])
        thresh_hi = thresh[obs_idx, k_hi]
        thresh_lo = thresh[obs_idx, k_lo]

        INF = _OPEN
        z_att_hi = jnp.where(y_idx < K - 1, thresh_hi - d_attended, INF)
        z_att_lo = jnp.where(y_idx > 0, thresh_lo - d_attended, -INF)
        p_att = jnp.clip(_DIFF(z_att_lo, z_att_hi), _pfloor())

        z_noise_hi = jnp.where(y_idx < K - 1, thresh_hi, INF)
        z_noise_lo = jnp.where(y_idx > 0, thresh_lo, -INF)
        p_noise = jnp.clip(_DIFF(z_noise_lo, z_noise_hi), _pfloor())

        p = jnp.clip(lambda_val * p_att + (1.0 - lambda_val) * p_noise, _pfloor())
        return jnp.log(p)

    @_batch_fused.defjvp
    def _batch_fused_jvp(primals, tangents):
        y, d_attended, lambda_val, thresh_mid, gaps = primals
        _, d_d_attended, d_lambda, d_thresh_mid, d_gaps = tangents

        if gap_link_is_softplus:
            gap_vals = jax.nn.softplus(gaps)
            gap_derivs = jax.nn.sigmoid(gaps)
        else:
            gap_vals = jnp.exp(gaps)
            gap_derivs = gap_vals

        thresh = _build_thresh(thresh_mid, gap_vals)

        y_idx = y - 1
        k_hi = jnp.clip(y_idx, 0, n_thresh - 1)
        k_lo = jnp.clip(y_idx - 1, 0, n_thresh - 1)

        obs_idx = jnp.arange(y.shape[0])
        thresh_hi = thresh[obs_idx, k_hi]
        thresh_lo = thresh[obs_idx, k_lo]

        INF = _OPEN

        z_att_hi = jnp.where(y_idx < K - 1, thresh_hi - d_attended, INF)
        z_att_lo = jnp.where(y_idx > 0, thresh_lo - d_attended, -INF)
        _D_p_att, _pl_p_att, _ph_p_att = _CELL(z_att_lo, z_att_hi)
        p_att = jnp.clip(_D_p_att, _pfloor())

        z_noise_hi = jnp.where(y_idx < K - 1, thresh_hi, INF)
        z_noise_lo = jnp.where(y_idx > 0, thresh_lo, -INF)
        _D_p_noise, _pl_p_noise, _ph_p_noise = _CELL(z_noise_lo, z_noise_hi)
        p_noise = jnp.clip(_D_p_noise, _pfloor())

        p = jnp.clip(lambda_val * p_att + (1.0 - lambda_val) * p_noise, _pfloor())
        val = jnp.log(p)

        d_lp_d_lambda = (p_att - p_noise) / p

        d_lp_d_patt = lambda_val / p
        d_lp_d_pnoise = (1.0 - lambda_val) / p

        pdf_att_hi = _ph_p_att
        pdf_att_lo = _pl_p_att
        dpatt_dz_hi = jnp.where(y_idx < K - 1, pdf_att_hi, 0.0)
        dpatt_dz_lo = jnp.where(y_idx > 0, -pdf_att_lo, 0.0)

        pdf_noise_hi = _ph_p_noise
        pdf_noise_lo = _pl_p_noise
        dpnoise_dz_hi = jnp.where(y_idx < K - 1, pdf_noise_hi, 0.0)
        dpnoise_dz_lo = jnp.where(y_idx > 0, -pdf_noise_lo, 0.0)

        dz_att_hi_eff = d_lp_d_patt * dpatt_dz_hi
        dz_att_lo_eff = d_lp_d_patt * dpatt_dz_lo
        d_lp_d_datt = -(dz_att_lo_eff + dz_att_hi_eff)

        has_hi = (y_idx < K - 1)
        has_lo = (y_idx > 0)

        d_thresh_k_hi = (jnp.where(has_hi, dz_att_hi_eff, 0.0) +
                         jnp.where(has_hi, d_lp_d_pnoise * dpnoise_dz_hi, 0.0))
        d_thresh_k_lo = (jnp.where(has_lo, dz_att_lo_eff, 0.0) +
                         jnp.where(has_lo, d_lp_d_pnoise * dpnoise_dz_lo, 0.0))

        d_lp_d_thresh_mid = d_thresh_k_hi + d_thresh_k_lo

        _gm = jnp.asarray(gap_mask)
        mask_hi = _gm[:, k_hi]
        mask_lo = _gm[:, k_lo]

        d_lp_d_gap_val = mask_hi * d_thresh_k_hi[None, :] + mask_lo * d_thresh_k_lo[None, :]
        d_lp_d_gaps = (d_lp_d_gap_val * gap_derivs.T).T

        tangent_out = (d_lp_d_datt * d_d_attended +
                       d_lp_d_lambda * d_lambda +
                       d_lp_d_thresh_mid * d_thresh_mid +
                       jnp.sum(d_lp_d_gaps * d_gaps, axis=1))

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


def _gap_mask_runs(gap_mask):
    gm = np.asarray(gap_mask)
    lo = np.zeros(gm.shape[0], dtype=np.int32)
    hi = np.full(gm.shape[0], -1, dtype=np.int32)
    sg = np.zeros(gm.shape[0])
    for g in range(gm.shape[0]):
        nz = np.nonzero(gm[g])[0]
        if nz.size:
            lo[g], hi[g], sg[g] = nz[0], nz[-1], gm[g, nz[0]]
            assert np.all(gm[g, nz[0]:nz[-1] + 1] == sg[g])
    return lo, hi, sg


def _batch_gap_grads(d_thresh_k_hi, d_thresh_k_lo, k_hi, k_lo,
                     has_hi, has_lo, gap_derivs, gap_mask):
    """[N, n_gaps] gap gradients. Each gap moves a contiguous run of thresholds by +-1,
    so the threshold-to-gap map is a range test on the threshold index."""
    if not isinstance(gap_mask, np.ndarray):
        gm = jnp.asarray(gap_mask)
        d_lp_d_gap_val = gm[:, k_hi] * d_thresh_k_hi[None, :] + gm[:, k_lo] * d_thresh_k_lo[None, :]
        return (d_lp_d_gap_val * gap_derivs.T).T
    lo, hi, sg = _gap_mask_runs(gap_mask)
    lo = jnp.asarray(lo)[None, :]
    hi = jnp.asarray(hi)[None, :]
    sg = jnp.asarray(sg)[None, :]
    kh = k_hi[:, None]
    kl = k_lo[:, None]
    in_hi = (kh >= lo) & (kh <= hi)
    in_lo = (kl >= lo) & (kl <= hi)
    return sg * (jnp.where(in_hi, d_thresh_k_hi[:, None], 0.0) +
                 jnp.where(in_lo, d_thresh_k_lo[:, None], 0.0)) * gap_derivs


def _make_fused_batch_bivariate_sdt_loglik(K1, K2, gap_link_is_softplus):
    """Fully-vectorized batch log-likelihood for bivariate_sdt.

    Batch operations over all observations:
    - Threshold construction via cumsum for both dimensions
    - GL10 rectangle quadrature vectorized over [N, 10]
    - Analytic JVP with gap_mask for both dimensions
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

    @custom_jvp
    def _batch_fused(y1, y2, item_type, mu1, mu2, sigma1, sigma2, rho_val,
                     thresh1_mid, gaps1, thresh2_mid, gaps2):
        gap_fn = jax.nn.softplus if gap_link_is_softplus else jnp.exp

        gv1 = gap_fn(gaps1)
        gv2 = gap_fn(gaps2)
        c1 = _batch_build_thresh(thresh1_mid, gv1, n_gaps1, mid_idx1, n_upper1)
        c2 = _batch_build_thresh(thresh2_mid, gv2, n_gaps2, mid_idx2, n_upper2)

        r1 = y1 - 1
        r2 = y2 - 1
        obs_idx = jnp.arange(y1.shape[0])
        k1_hi = jnp.clip(r1, 0, n_thresh1 - 1)
        k1_lo = jnp.clip(r1 - 1, 0, n_thresh1 - 1)
        k2_hi = jnp.clip(r2, 0, n_thresh2 - 1)
        k2_lo = jnp.clip(r2 - 1, 0, n_thresh2 - 1)

        INF = _OPEN
        z1_hi = jnp.where(r1 < K1 - 1, (c1[obs_idx, k1_hi] - mu1) / sigma1, INF)
        z1_lo = jnp.where(r1 > 0, (c1[obs_idx, k1_lo] - mu1) / sigma1, -INF)
        z2_hi = jnp.where(r2 < K2 - 1, (c2[obs_idx, k2_hi] - mu2) / sigma2, INF)
        z2_lo = jnp.where(r2 > 0, (c2[obs_idx, k2_lo] - mu2) / sigma2, -INF)

        prob = jnp.clip(_bvn_rect(z1_lo, z1_hi, z2_lo, z2_hi, rho_val)[0], _pfloor())

        return jnp.log(prob)

    @_batch_fused.defjvp
    def _batch_fused_jvp(primals, tangents):
        (y1, y2, item_type, mu1, mu2, sigma1, sigma2, rho_val,
         thresh1_mid, gaps1, thresh2_mid, gaps2) = primals
        (_, _, _, d_mu1, d_mu2, d_sigma1, d_sigma2, d_rho,
         d_thresh1_mid, d_gaps1, d_thresh2_mid, d_gaps2) = tangents

        if gap_link_is_softplus:
            gv1 = jax.nn.softplus(gaps1); gd1 = jax.nn.sigmoid(gaps1)
            gv2 = jax.nn.softplus(gaps2); gd2 = jax.nn.sigmoid(gaps2)
        else:
            gv1 = jnp.exp(gaps1); gd1 = gv1
            gv2 = jnp.exp(gaps2); gd2 = gv2

        c1 = _batch_build_thresh(thresh1_mid, gv1, n_gaps1, mid_idx1, n_upper1)
        c2 = _batch_build_thresh(thresh2_mid, gv2, n_gaps2, mid_idx2, n_upper2)

        r1 = y1 - 1; r2 = y2 - 1
        obs_idx = jnp.arange(y1.shape[0])
        k1_hi = jnp.clip(r1, 0, n_thresh1 - 1)
        k1_lo = jnp.clip(r1 - 1, 0, n_thresh1 - 1)
        k2_hi = jnp.clip(r2, 0, n_thresh2 - 1)
        k2_lo = jnp.clip(r2 - 1, 0, n_thresh2 - 1)
        has1_hi = (r1 < K1 - 1); has1_lo = (r1 > 0)
        has2_hi = (r2 < K2 - 1); has2_lo = (r2 > 0)

        INF = _OPEN
        z1_hi = jnp.where(has1_hi, (c1[obs_idx, k1_hi] - mu1) / sigma1, INF)
        z1_lo = jnp.where(has1_lo, (c1[obs_idx, k1_lo] - mu1) / sigma1, -INF)
        z2_hi = jnp.where(has2_hi, (c2[obs_idx, k2_hi] - mu2) / sigma2, INF)
        z2_lo = jnp.where(has2_lo, (c2[obs_idx, k2_lo] - mu2) / sigma2, -INF)

        p_raw, dp_dz1_lo, dp_dz1_hi, dp_dz2_lo, dp_dz2_hi, dp_drho = _bvn_rect(
            z1_lo, z1_hi, z2_lo, z2_hi, rho_val)
        prob = jnp.clip(p_raw, _pfloor())
        val = jnp.log(prob)

        inv_P = jnp.where(p_raw > _pfloor(), 1.0 / prob, 0.0)

        d_lp_dz1_hi = dp_dz1_hi * inv_P
        d_lp_dz1_lo = dp_dz1_lo * inv_P
        d_lp_dz2_hi = dp_dz2_hi * inv_P
        d_lp_dz2_lo = dp_dz2_lo * inv_P
        d_lp_drho = dp_drho * inv_P

        inv_s1 = 1.0 / sigma1
        inv_s2 = 1.0 / sigma2
        d_lp_d_mu1 = -(d_lp_dz1_hi + d_lp_dz1_lo) * inv_s1
        d_lp_d_mu2 = -(d_lp_dz2_hi + d_lp_dz2_lo) * inv_s2
        d_lp_d_sigma1 = -(d_lp_dz1_hi * z1_hi + d_lp_dz1_lo * z1_lo) * inv_s1
        d_lp_d_sigma2 = -(d_lp_dz2_hi * z2_hi + d_lp_dz2_lo * z2_lo) * inv_s2

        d_t1_k_hi = jnp.where(has1_hi, d_lp_dz1_hi * inv_s1, 0.0)
        d_t1_k_lo = jnp.where(has1_lo, d_lp_dz1_lo * inv_s1, 0.0)
        d_t2_k_hi = jnp.where(has2_hi, d_lp_dz2_hi * inv_s2, 0.0)
        d_t2_k_lo = jnp.where(has2_lo, d_lp_dz2_lo * inv_s2, 0.0)

        d_lp_d_thresh1_mid = d_t1_k_hi + d_t1_k_lo
        d_lp_d_thresh2_mid = d_t2_k_hi + d_t2_k_lo

        d_lp_d_gaps1 = _batch_gap_grads(d_t1_k_hi, d_t1_k_lo, k1_hi, k1_lo,
                                         has1_hi, has1_lo, gd1, gap_mask1_np)
        d_lp_d_gaps2 = _batch_gap_grads(d_t2_k_hi, d_t2_k_lo, k2_hi, k2_lo,
                                         has2_hi, has2_lo, gd2, gap_mask2_np)

        tangent_out = (
            d_lp_d_mu1 * d_mu1 +
            d_lp_d_mu2 * d_mu2 +
            d_lp_d_sigma1 * d_sigma1 +
            d_lp_d_sigma2 * d_sigma2 +
            d_lp_drho * d_rho +
            d_lp_d_thresh1_mid * d_thresh1_mid +
            jnp.sum(d_lp_d_gaps1 * d_gaps1, axis=1) +
            d_lp_d_thresh2_mid * d_thresh2_mid +
            jnp.sum(d_lp_d_gaps2 * d_gaps2, axis=1)
        )

        return val, tangent_out

    return _batch_fused


# =============================================================================
# Batch-vectorized bounded bivariate_sdt loglik (no vmap)
# =============================================================================

def _make_fused_batch_bounded_bivariate_sdt_loglik(K1, K2, gap_link_is_softplus):
    """Fully-vectorized batch log-likelihood for bounded bivariate_sdt.

    Bounded model: below a crossing point (cp), the source dimension becomes
    independent with mean=0 and conditional SD. Above cp, standard bivariate
    normal applies. For new items, z_cp=-20 so everything is above-cp.

    Uses GL10 quadrature for the above-cp region and analytic product for
    the below-cp region, with fully analytic JVP gradients.
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

    @custom_jvp
    def _batch_fused(y1, y2, item_type, mu1, mu2, sigma1, sigma2, rho_val,
                     thresh1_mid, gaps1, thresh2_mid, gaps2):
        gap_fn = jax.nn.softplus if gap_link_is_softplus else jnp.exp

        gv1 = gap_fn(gaps1)
        gv2 = gap_fn(gaps2)
        c1 = _batch_build_thresh(thresh1_mid, gv1, n_gaps1, mid_idx1, n_upper1)
        c2 = _batch_build_thresh(thresh2_mid, gv2, n_gaps2, mid_idx2, n_upper2)

        r1 = y1 - 1
        r2 = y2 - 1
        obs_idx = jnp.arange(y1.shape[0])
        k1_hi = jnp.clip(r1, 0, n_thresh1 - 1)
        k1_lo = jnp.clip(r1 - 1, 0, n_thresh1 - 1)
        k2_hi = jnp.clip(r2, 0, n_thresh2 - 1)
        k2_lo = jnp.clip(r2 - 1, 0, n_thresh2 - 1)

        INF = _OPEN
        inv_s1 = 1.0 / sigma1
        inv_s2 = 1.0 / sigma2

        z1_hi = jnp.where(r1 < K1 - 1, (c1[obs_idx, k1_hi] - mu1) * inv_s1, INF)
        z1_lo = jnp.where(r1 > 0, (c1[obs_idx, k1_lo] - mu1) * inv_s1, -INF)

        z2_hi = jnp.where(r2 < K2 - 1, (c2[obs_idx, k2_hi] - mu2) * inv_s2, INF)
        z2_lo = jnp.where(r2 > 0, (c2[obs_idx, k2_lo] - mu2) * inv_s2, -INF)

        sigma_cond = sigma2 * jnp.sqrt(jnp.clip(1.0 - rho_val**2, 1e-20))
        sigma_cond = jnp.maximum(sigma_cond, sigma2 * 1e-6)
        inv_sc = 1.0 / sigma_cond

        z_cp_raw = -mu2 / (rho_val * sigma2 + 1e-30)
        z_cp = jnp.where(item_type == 1, -1e6, z_cp_raw)

        z_cp_eff = jnp.clip(z_cp, z1_lo, z1_hi)

        zs_hi = jnp.where(r2 < K2 - 1, c2[obs_idx, k2_hi] * inv_sc, INF)
        zs_lo = jnp.where(r2 > 0, c2[obs_idx, k2_lo] * inv_sc, -INF)

        p_det_below = jnp.clip(_phi_diff(z1_lo, z_cp_eff)[0], 0.0)
        p_src_below = jnp.clip(_phi_diff(zs_lo, zs_hi)[0], 0.0)
        p_below = p_det_below * p_src_below
        p_above = jnp.clip(_bvn_rect(z_cp_eff, z1_hi, z2_lo, z2_hi, rho_val)[0], 0.0)
        P = jnp.clip(p_below + p_above, _pfloor())
        return jnp.log(P)

    @_batch_fused.defjvp
    def _batch_fused_jvp(primals, tangents):
        (y1, y2, item_type, mu1, mu2, sigma1, sigma2, rho_val,
         thresh1_mid, gaps1, thresh2_mid, gaps2) = primals
        (_, _, _, d_mu1, d_mu2, d_sigma1, d_sigma2, d_rho,
         d_thresh1_mid, d_gaps1, d_thresh2_mid, d_gaps2) = tangents

        if gap_link_is_softplus:
            gv1 = jax.nn.softplus(gaps1); gd1 = jax.nn.sigmoid(gaps1)
            gv2 = jax.nn.softplus(gaps2); gd2 = jax.nn.sigmoid(gaps2)
        else:
            gv1 = jnp.exp(gaps1); gd1 = gv1
            gv2 = jnp.exp(gaps2); gd2 = gv2

        c1 = _batch_build_thresh(thresh1_mid, gv1, n_gaps1, mid_idx1, n_upper1)
        c2 = _batch_build_thresh(thresh2_mid, gv2, n_gaps2, mid_idx2, n_upper2)

        r1 = y1 - 1; r2 = y2 - 1
        obs_idx = jnp.arange(y1.shape[0])
        k1_hi = jnp.clip(r1, 0, n_thresh1 - 1)
        k1_lo = jnp.clip(r1 - 1, 0, n_thresh1 - 1)
        k2_hi = jnp.clip(r2, 0, n_thresh2 - 1)
        k2_lo = jnp.clip(r2 - 1, 0, n_thresh2 - 1)
        has1_hi = (r1 < K1 - 1); has1_lo = (r1 > 0)
        has2_hi = (r2 < K2 - 1); has2_lo = (r2 > 0)

        inv_s1 = 1.0 / sigma1
        inv_s2 = 1.0 / sigma2
        INF = _OPEN

        z1_hi = jnp.where(has1_hi, (c1[obs_idx, k1_hi] - mu1) * inv_s1, INF)
        z1_lo = jnp.where(has1_lo, (c1[obs_idx, k1_lo] - mu1) * inv_s1, -INF)
        z2_hi = jnp.where(has2_hi, (c2[obs_idx, k2_hi] - mu2) * inv_s2, INF)
        z2_lo = jnp.where(has2_lo, (c2[obs_idx, k2_lo] - mu2) * inv_s2, -INF)

        one_minus_rho2 = jnp.clip(1.0 - rho_val**2, 1e-20)
        sqrt_1mrr = jnp.sqrt(one_minus_rho2)
        sigma_cond = sigma2 * sqrt_1mrr
        sigma_cond = jnp.maximum(sigma_cond, sigma2 * 1e-6)
        inv_sc = 1.0 / sigma_cond

        rho_sigma2 = rho_val * sigma2 + 1e-30
        z_cp_raw = -mu2 / rho_sigma2
        z_cp = jnp.where(item_type == 1, -1e6, z_cp_raw)
        z_cp_eff = jnp.clip(z_cp, z1_lo, z1_hi)

        is_split = (z_cp > z1_lo) & (z_cp < z1_hi)
        is_above_only = (z_cp <= z1_lo)
        is_below_only = (z_cp >= z1_hi)
        is_split_f = is_split.astype(_FLOAT)
        is_above_only_f = is_above_only.astype(_FLOAT)
        is_below_only_f = is_below_only.astype(_FLOAT)

        zs_hi = jnp.where(has2_hi, c2[obs_idx, k2_hi] * inv_sc, INF)
        zs_lo = jnp.where(has2_lo, c2[obs_idx, k2_lo] * inv_sc, -INF)

        D_det, dDdet_lo, dDdet_hi = _phi_diff(z1_lo, z_cp_eff)
        D_src, dDsrc_lo, dDsrc_hi = _phi_diff(zs_lo, zs_hi)
        det_on = (D_det > 0.0).astype(_FLOAT)
        src_on = (D_src > 0.0).astype(_FLOAT)
        p_det_below = jnp.clip(D_det, 0.0)
        p_src_below = jnp.clip(D_src, 0.0)
        p_below = p_det_below * p_src_below
        pa_raw, dpa_da, dpa_db, dpa_dc, dpa_dd, dpa_drho = _bvn_rect(z_cp_eff, z1_hi, z2_lo, z2_hi, rho_val)
        above_on = (pa_raw > 0.0).astype(_FLOAT)
        p_above = jnp.clip(pa_raw, 0.0)
        P_raw = p_below + p_above
        P = jnp.clip(P_raw, _pfloor())
        val = jnp.log(P)
        inv_P = jnp.where(P_raw > _pfloor(), 1.0 / P, 0.0)

        dp_below_dzcp_eff = det_on * dDdet_hi * p_src_below
        dp_below_dz1_lo = jnp.where(has1_lo, det_on * dDdet_lo * p_src_below, 0.0)
        dp_below_dzs_hi = jnp.where(has2_hi, src_on * p_det_below * dDsrc_hi, 0.0)
        dp_below_dzs_lo = jnp.where(has2_lo, src_on * p_det_below * dDsrc_lo, 0.0)
        dp_above_dzcp_eff = above_on * dpa_da
        dp_above_dz1_hi = above_on * dpa_db
        dp_above_dz2_hi = above_on * dpa_dd
        dp_above_dz2_lo = above_on * dpa_dc
        dp_above_drho_direct = above_on * dpa_drho

        # --- Combined z_cp_eff gradient (flows to both below and above) ---
        d_total_dzcp_eff = dp_below_dzcp_eff + dp_above_dzcp_eff

        # ===============================================================
        # Chain rules to SDT parameters
        # ===============================================================

        rho_sigma2_safe = jnp.where(jnp.abs(rho_sigma2) > 1e-20, rho_sigma2, 1.0)
        rho_safe = jnp.where(jnp.abs(rho_val) > 1e-20, rho_val, 1.0)
        dz_cp_dmu2 = -1.0 / rho_sigma2_safe
        dz_cp_ds2 = mu2 / (rho_sigma2_safe * sigma2)
        dz_cp_drho = mu2 / (rho_safe * rho_sigma2_safe)

        d_zcp = d_total_dzcp_eff * is_split_f
        d_zcp_to_z1_lo = d_total_dzcp_eff * is_above_only_f
        d_zcp_to_z1_hi = d_total_dzcp_eff * is_below_only_f

        # --- d_logP / d_mu1 ---
        dP_dmu1 = (dp_below_dz1_lo * (-inv_s1) +
                   dp_above_dz1_hi * (-inv_s1) +
                   d_zcp_to_z1_lo * (-inv_s1) +
                   d_zcp_to_z1_hi * (-inv_s1))
        d_lp_dmu1 = dP_dmu1 * inv_P

        # --- d_logP / d_mu2 ---
        dP_dmu2 = (d_zcp * dz_cp_dmu2 +
                   dp_above_dz2_hi * (-inv_s2) +
                   dp_above_dz2_lo * (-inv_s2))
        d_lp_dmu2 = dP_dmu2 * inv_P

        # --- d_logP / d_sigma1 ---
        dP_ds1 = (dp_below_dz1_lo * (-z1_lo * inv_s1) +
                  dp_above_dz1_hi * (-z1_hi * inv_s1) +
                  d_zcp_to_z1_lo * (-z1_lo * inv_s1) +
                  d_zcp_to_z1_hi * (-z1_hi * inv_s1))
        d_lp_dsigma1 = dP_ds1 * inv_P

        # --- d_logP / d_sigma2 ---
        dP_ds2 = (d_zcp * dz_cp_ds2 +
                  dp_above_dz2_hi * (-z2_hi * inv_s2) +
                  dp_above_dz2_lo * (-z2_lo * inv_s2) +
                  dp_below_dzs_hi * (-zs_hi * inv_s2) +
                  dp_below_dzs_lo * (-zs_lo * inv_s2))
        d_lp_dsigma2 = dP_ds2 * inv_P

        # --- d_logP / d_rho ---
        rho_factor = jnp.where((sigma2 * sqrt_1mrr > sigma2 * 1e-6) & (1.0 - rho_val**2 > 1e-20),
                               rho_val / one_minus_rho2, 0.0)
        dP_drho = (d_zcp * dz_cp_drho +
                   dp_above_drho_direct +
                   dp_below_dzs_hi * (zs_hi * rho_factor) +
                   dp_below_dzs_lo * (zs_lo * rho_factor))
        d_lp_drho = dP_drho * inv_P

        # ===============================================================
        # ===============================================================
        d_t1_k_hi = jnp.where(has1_hi, (dp_above_dz1_hi + d_zcp_to_z1_hi) * inv_s1 * inv_P, 0.0)
        d_t1_k_lo = jnp.where(has1_lo, (dp_below_dz1_lo + d_zcp_to_z1_lo) * inv_s1 * inv_P, 0.0)

        d_t2_k_hi_above = jnp.where(has2_hi, dp_above_dz2_hi * inv_s2, 0.0)
        d_t2_k_hi_below = jnp.where(has2_hi, dp_below_dzs_hi * inv_sc, 0.0)
        d_t2_k_hi = (d_t2_k_hi_above + d_t2_k_hi_below) * inv_P

        d_t2_k_lo_above = jnp.where(has2_lo, dp_above_dz2_lo * inv_s2, 0.0)
        d_t2_k_lo_below = jnp.where(has2_lo, dp_below_dzs_lo * inv_sc, 0.0)
        d_t2_k_lo = (d_t2_k_lo_above + d_t2_k_lo_below) * inv_P

        d_lp_d_thresh1_mid = d_t1_k_hi + d_t1_k_lo
        d_lp_d_thresh2_mid = d_t2_k_hi + d_t2_k_lo

        d_lp_d_gaps1 = _batch_gap_grads(d_t1_k_hi, d_t1_k_lo, k1_hi, k1_lo,
                                         has1_hi, has1_lo, gd1, gap_mask1_np)
        d_lp_d_gaps2 = _batch_gap_grads(d_t2_k_hi, d_t2_k_lo, k2_hi, k2_lo,
                                         has2_hi, has2_lo, gd2, gap_mask2_np)

        tangent_out = (
            d_lp_dmu1 * d_mu1 +
            d_lp_dmu2 * d_mu2 +
            d_lp_dsigma1 * d_sigma1 +
            d_lp_dsigma2 * d_sigma2 +
            d_lp_drho * d_rho +
            d_lp_d_thresh1_mid * d_thresh1_mid +
            jnp.sum(d_lp_d_gaps1 * d_gaps1, axis=1) +
            d_lp_d_thresh2_mid * d_thresh2_mid +
            jnp.sum(d_lp_d_gaps2 * d_gaps2, axis=1)
        )

        return val, tangent_out

    return _batch_fused


# =============================================================================
# Batch-vectorized bivariate_dp loglik (no vmap)
# =============================================================================

def _make_fused_batch_bivariate_dp_loglik(K1, K2, gap_link_is_softplus):
    """Fully-vectorized batch log-likelihood for bivariate_dp.

    3-component mixture:
    P = (1-R_I) * biv_prob + R_I*(1-R_S) * marginal + R_I*R_S * corner
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

    @custom_jvp
    def _batch_fused(y1, y2, item_type, mu1, mu2, sigma1, sigma2, rho_val, rho_N_val,
                     R_I, R_S, R_I_B, R_S_B, thresh1_mid, gaps1, thresh2_mid, gaps2):
        gap_fn = jax.nn.softplus if gap_link_is_softplus else jnp.exp

        gv1 = gap_fn(gaps1)
        gv2 = gap_fn(gaps2)
        c1 = _batch_build_thresh(thresh1_mid, gv1, n_gaps1, mid_idx1, n_upper1)
        c2 = _batch_build_thresh(thresh2_mid, gv2, n_gaps2, mid_idx2, n_upper2)

        N = y1.shape[0]
        r1 = y1 - 1
        r2 = y2 - 1
        obs_idx = jnp.arange(N)
        k1_hi = jnp.clip(r1, 0, n_thresh1 - 1)
        k1_lo = jnp.clip(r1 - 1, 0, n_thresh1 - 1)
        k2_hi = jnp.clip(r2, 0, n_thresh2 - 1)
        k2_lo = jnp.clip(r2 - 1, 0, n_thresh2 - 1)

        INF = _OPEN
        z1_hi = jnp.where(r1 < K1 - 1, (c1[obs_idx, k1_hi] - mu1) / sigma1, INF)
        z1_lo = jnp.where(r1 > 0, (c1[obs_idx, k1_lo] - mu1) / sigma1, -INF)
        z2_hi = jnp.where(r2 < K2 - 1, (c2[obs_idx, k2_hi] - mu2) / sigma2, INF)
        z2_lo = jnp.where(r2 > 0, (c2[obs_idx, k2_lo] - mu2) / sigma2, -INF)

        biv_prob = jnp.clip(_bvn_rect(z1_lo, z1_hi, z2_lo, z2_hi, rho_val)[0], _pfloor())

        is_A = (item_type == 2)
        R_I_eff = jnp.where(is_A, R_I, R_I_B)
        R_S_eff = jnp.where(is_A, R_S, R_S_B)

        p1_comp = (1.0 - R_I_eff) * biv_prob

        p2_source = jnp.clip(_phi_diff(z2_lo, z2_hi)[0], _pfloor())
        is_y1_top = (y1 == K1)
        p2_comp = jnp.where(is_y1_top, R_I_eff * (1.0 - R_S_eff) * p2_source, 0.0)

        is_corner_A = (item_type == 2) & (y1 == K1) & (y2 == 1)
        is_corner_B = (item_type == 3) & (y1 == K1) & (y2 == K2)
        p3_comp = jnp.where(is_corner_A | is_corner_B, R_I_eff * R_S_eff, 0.0)

        prob = jnp.clip(p1_comp + p2_comp + p3_comp, _pfloor())
        return jnp.log(prob)

    @_batch_fused.defjvp
    def _batch_fused_jvp(primals, tangents):
        (y1, y2, item_type, mu1, mu2, sigma1, sigma2, rho_val, rho_N_val,
         R_I, R_S, R_I_B, R_S_B, thresh1_mid, gaps1, thresh2_mid, gaps2) = primals
        (_, _, _, d_mu1, d_mu2, d_sigma1, d_sigma2, d_rho, d_rho_N,
         d_R_I, d_R_S, d_R_I_B, d_R_S_B, d_thresh1_mid, d_gaps1, d_thresh2_mid, d_gaps2) = tangents

        if gap_link_is_softplus:
            gv1 = jax.nn.softplus(gaps1); gd1 = jax.nn.sigmoid(gaps1)
            gv2 = jax.nn.softplus(gaps2); gd2 = jax.nn.sigmoid(gaps2)
        else:
            gv1 = jnp.exp(gaps1); gd1 = gv1
            gv2 = jnp.exp(gaps2); gd2 = gv2

        c1 = _batch_build_thresh(thresh1_mid, gv1, n_gaps1, mid_idx1, n_upper1)
        c2 = _batch_build_thresh(thresh2_mid, gv2, n_gaps2, mid_idx2, n_upper2)

        N = y1.shape[0]
        r1 = y1 - 1; r2 = y2 - 1
        obs_idx = jnp.arange(N)
        k1_hi = jnp.clip(r1, 0, n_thresh1 - 1)
        k1_lo = jnp.clip(r1 - 1, 0, n_thresh1 - 1)
        k2_hi = jnp.clip(r2, 0, n_thresh2 - 1)
        k2_lo = jnp.clip(r2 - 1, 0, n_thresh2 - 1)
        has1_hi = (r1 < K1 - 1); has1_lo = (r1 > 0)
        has2_hi = (r2 < K2 - 1); has2_lo = (r2 > 0)

        inv_s1 = 1.0 / sigma1
        inv_s2 = 1.0 / sigma2
        INF = _OPEN
        z1_hi = jnp.where(has1_hi, (c1[obs_idx, k1_hi] - mu1) * inv_s1, INF)
        z1_lo = jnp.where(has1_lo, (c1[obs_idx, k1_lo] - mu1) * inv_s1, -INF)
        z2_hi = jnp.where(has2_hi, (c2[obs_idx, k2_hi] - mu2) * inv_s2, INF)
        z2_lo = jnp.where(has2_lo, (c2[obs_idx, k2_lo] - mu2) * inv_s2, -INF)

        bp_raw, dp_dz1_lo, dp_dz1_hi, dp_dz2_lo, dp_dz2_hi, dp_biv_drho = _bvn_rect(
            z1_lo, z1_hi, z2_lo, z2_hi, rho_val)
        biv_on = (bp_raw > _pfloor()).astype(_FLOAT)
        dp_dz1_lo = dp_dz1_lo * biv_on
        dp_dz1_hi = dp_dz1_hi * biv_on
        dp_dz2_lo = dp_dz2_lo * biv_on
        dp_dz2_hi = dp_dz2_hi * biv_on
        dp_biv_drho = dp_biv_drho * biv_on
        biv_prob = jnp.clip(bp_raw, _pfloor())

        is_A = (item_type == 2)
        R_I_eff = jnp.where(is_A, R_I, R_I_B)
        R_S_eff = jnp.where(is_A, R_S, R_S_B)

        p1_comp = (1.0 - R_I_eff) * biv_prob

        p2_raw, dp2s_dlo, dp2s_dhi = _phi_diff(z2_lo, z2_hi)
        p2_on = (p2_raw > _pfloor()).astype(_FLOAT)
        p2_source = jnp.clip(p2_raw, _pfloor())
        is_y1_top = (y1 == K1)
        p2_comp = jnp.where(is_y1_top, R_I_eff * (1.0 - R_S_eff) * p2_source, 0.0)

        is_corner_A = (item_type == 2) & (y1 == K1) & (y2 == 1)
        is_corner_B = (item_type == 3) & (y1 == K1) & (y2 == K2)
        is_corner = is_corner_A | is_corner_B
        p3_comp = jnp.where(is_corner, R_I_eff * R_S_eff, 0.0)

        prob_raw = p1_comp + p2_comp + p3_comp
        prob = jnp.clip(prob_raw, _pfloor())
        val = jnp.log(prob)
        inv_P = jnp.where(prob_raw > _pfloor(), 1.0 / prob, 0.0)

        dp2s_dz2_hi = jnp.where(has2_hi, p2_on * dp2s_dhi, 0.0)
        dp2s_dz2_lo = jnp.where(has2_lo, p2_on * dp2s_dlo, 0.0)
        d_lp_d_RI_eff = (-biv_prob + jnp.where(is_y1_top, (1.0 - R_S_eff) * p2_source, 0.0) +
                          jnp.where(is_corner, R_S_eff, 0.0)) * inv_P
        d_lp_d_RS_eff = (jnp.where(is_y1_top, -R_I_eff * p2_source, 0.0) +
                          jnp.where(is_corner, R_I_eff, 0.0)) * inv_P
        d_lp_d_rho = (1.0 - R_I_eff) * dp_biv_drho * inv_P
        d_lp_d_mu1 = (1.0 - R_I_eff) * (dp_dz1_hi + dp_dz1_lo) * (-inv_s1) * inv_P
        d_lp_d_mu2 = ((1.0 - R_I_eff) * (dp_dz2_hi + dp_dz2_lo) * (-inv_s2) +
                       jnp.where(is_y1_top, R_I_eff * (1.0 - R_S_eff) * (dp2s_dz2_hi + dp2s_dz2_lo) * (-inv_s2), 0.0)) * inv_P
        d_lp_d_s1 = (1.0 - R_I_eff) * (dp_dz1_hi * z1_hi + dp_dz1_lo * z1_lo) * (-inv_s1) * inv_P
        d_lp_d_s2 = ((1.0 - R_I_eff) * (dp_dz2_hi * z2_hi + dp_dz2_lo * z2_lo) * (-inv_s2) +
                      jnp.where(is_y1_top, R_I_eff * (1.0 - R_S_eff) * (dp2s_dz2_hi * z2_hi + dp2s_dz2_lo * z2_lo) * (-inv_s2), 0.0)) * inv_P

        d_t1_k_hi = jnp.where(has1_hi, (1.0 - R_I_eff) * dp_dz1_hi * inv_s1, 0.0) * inv_P
        d_t1_k_lo = jnp.where(has1_lo, (1.0 - R_I_eff) * dp_dz1_lo * inv_s1, 0.0) * inv_P
        d_t2_k_hi = (jnp.where(has2_hi, (1.0 - R_I_eff) * dp_dz2_hi * inv_s2, 0.0) +
                      jnp.where(has2_hi & is_y1_top, R_I_eff * (1.0 - R_S_eff) * dp2s_dz2_hi * inv_s2, 0.0)) * inv_P
        d_t2_k_lo = (jnp.where(has2_lo, (1.0 - R_I_eff) * dp_dz2_lo * inv_s2, 0.0) +
                      jnp.where(has2_lo & is_y1_top, R_I_eff * (1.0 - R_S_eff) * dp2s_dz2_lo * inv_s2, 0.0)) * inv_P

        d_lp_d_thresh1_mid = d_t1_k_hi + d_t1_k_lo
        d_lp_d_thresh2_mid = d_t2_k_hi + d_t2_k_lo

        d_lp_d_gaps1 = _batch_gap_grads(d_t1_k_hi, d_t1_k_lo, k1_hi, k1_lo,
                                         has1_hi, has1_lo, gd1, gap_mask1_np)
        d_lp_d_gaps2 = _batch_gap_grads(d_t2_k_hi, d_t2_k_lo, k2_hi, k2_lo,
                                         has2_hi, has2_lo, gd2, gap_mask2_np)

        d_R_I_eff_in = jnp.where(is_A, d_R_I, d_R_I_B)
        d_R_S_eff_in = jnp.where(is_A, d_R_S, d_R_S_B)

        tangent_out = (
            d_lp_d_mu1 * d_mu1 +
            d_lp_d_mu2 * d_mu2 +
            d_lp_d_s1 * d_sigma1 +
            d_lp_d_s2 * d_sigma2 +
            d_lp_d_rho * d_rho +
            0.0 * d_rho_N +
            d_lp_d_RI_eff * d_R_I_eff_in +
            d_lp_d_RS_eff * d_R_S_eff_in +
            d_lp_d_thresh1_mid * d_thresh1_mid +
            jnp.sum(d_lp_d_gaps1 * d_gaps1, axis=1) +
            d_lp_d_thresh2_mid * d_thresh2_mid +
            jnp.sum(d_lp_d_gaps2 * d_gaps2, axis=1)
        )

        return val, tangent_out

    return _batch_fused


# =============================================================================
# Batch-vectorized vrdp2d loglik (no vmap)
# =============================================================================

def _make_fused_batch_vrdp2d_loglik(K1, K2, gap_link_is_softplus):
    """Fully-vectorized batch log-likelihood for VRDP2D.

    P = (1-lambda)*p1_fam*p2_noise + lambda*p1_rec*p2_rec
    All univariate — no bivariate CDF needed.
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

    @custom_jvp
    def _batch_fused(y1, y2, item_type, dprime_F, dprime_R, source_d, lambda_val,
                     sigma_item, sigma_S, thresh1_mid, gaps1, thresh2_mid, gaps2):
        gap_fn = jax.nn.softplus if gap_link_is_softplus else jnp.exp

        gv1 = gap_fn(gaps1)
        gv2 = gap_fn(gaps2)
        c1 = _batch_build_thresh(thresh1_mid, gv1, n_gaps1, mid_idx1, n_upper1)
        c2 = _batch_build_thresh(thresh2_mid, gv2, n_gaps2, mid_idx2, n_upper2)

        N = y1.shape[0]
        r1 = y1 - 1; r2 = y2 - 1
        obs_idx = jnp.arange(N)
        k1_hi = jnp.clip(r1, 0, n_thresh1 - 1)
        k1_lo = jnp.clip(r1 - 1, 0, n_thresh1 - 1)
        k2_hi = jnp.clip(r2, 0, n_thresh2 - 1)
        k2_lo = jnp.clip(r2 - 1, 0, n_thresh2 - 1)

        INF = _OPEN
        inv_si = 1.0 / sigma_item
        inv_sS = 1.0 / sigma_S
        mu_rec1 = dprime_F + dprime_R

        c1_hi = c1[obs_idx, k1_hi]; c1_lo = c1[obs_idx, k1_lo]
        c2_hi = c2[obs_idx, k2_hi]; c2_lo = c2[obs_idx, k2_lo]
        has1_hi = (r1 < K1 - 1); has1_lo = (r1 > 0)
        has2_hi = (r2 < K2 - 1); has2_lo = (r2 > 0)

        nat = _vrdp2d_native(c1_hi, c1_lo, c2_hi, c2_lo, has1_hi, has1_lo, has2_hi, has2_lo,
                             dprime_F, dprime_R, source_d, lambda_val, sigma_item, sigma_S)
        if nat is not None:
            return nat[0]
        z1_fam_hi = jnp.where(has1_hi, (c1_hi - dprime_F) * inv_si, INF)
        z1_fam_lo = jnp.where(has1_lo, (c1_lo - dprime_F) * inv_si, -INF)
        z2_hi     = jnp.where(has2_hi,  c2_hi,                       INF)
        z2_lo     = jnp.where(has2_lo,  c2_lo,                      -INF)
        z1_rec_hi = jnp.where(has1_hi, (c1_hi - mu_rec1) * inv_si,   INF)
        z1_rec_lo = jnp.where(has1_lo, (c1_lo - mu_rec1) * inv_si,  -INF)
        z2_rec_hi = jnp.where(has2_hi, (c2_hi - source_d) * inv_sS,  INF)
        z2_rec_lo = jnp.where(has2_lo, (c2_lo - source_d) * inv_sS, -INF)

        p1_fam   = jnp.clip(_phi_diff(z1_fam_lo, z1_fam_hi)[0], _pfloor())
        p2_noise = jnp.clip(_phi_diff(z2_lo, z2_hi)[0], _pfloor())
        p1_rec   = jnp.clip(_phi_diff(z1_rec_lo, z1_rec_hi)[0], _pfloor())
        p2_rec   = jnp.clip(_phi_diff(z2_rec_lo, z2_rec_hi)[0], _pfloor())

        prob = jnp.clip((1.0 - lambda_val) * p1_fam * p2_noise +
                         lambda_val * p1_rec * p2_rec, _pfloor())
        return jnp.log(prob)

    @_batch_fused.defjvp
    def _batch_fused_jvp(primals, tangents):
        (y1, y2, item_type, dprime_F, dprime_R, source_d, lambda_val,
         sigma_item, sigma_S, thresh1_mid, gaps1, thresh2_mid, gaps2) = primals
        (_, _, _, d_dprime_F, d_dprime_R, d_source_d, d_lambda,
         d_sigma_item, d_sigma_S, d_thresh1_mid, d_gaps1, d_thresh2_mid, d_gaps2) = tangents

        if gap_link_is_softplus:
            gv1 = jax.nn.softplus(gaps1); gd1 = jax.nn.sigmoid(gaps1)
            gv2 = jax.nn.softplus(gaps2); gd2 = jax.nn.sigmoid(gaps2)
        else:
            gv1 = jnp.exp(gaps1); gd1 = gv1
            gv2 = jnp.exp(gaps2); gd2 = gv2

        c1 = _batch_build_thresh(thresh1_mid, gv1, n_gaps1, mid_idx1, n_upper1)
        c2 = _batch_build_thresh(thresh2_mid, gv2, n_gaps2, mid_idx2, n_upper2)

        N = y1.shape[0]
        r1 = y1 - 1; r2 = y2 - 1
        obs_idx = jnp.arange(N)
        k1_hi = jnp.clip(r1, 0, n_thresh1 - 1)
        k1_lo = jnp.clip(r1 - 1, 0, n_thresh1 - 1)
        k2_hi = jnp.clip(r2, 0, n_thresh2 - 1)
        k2_lo = jnp.clip(r2 - 1, 0, n_thresh2 - 1)
        has1_hi = (r1 < K1 - 1); has1_lo = (r1 > 0)
        has2_hi = (r2 < K2 - 1); has2_lo = (r2 > 0)

        INF = _OPEN
        inv_si = 1.0 / sigma_item
        inv_sS = 1.0 / sigma_S
        mu_rec1 = dprime_F + dprime_R

        c1_hi = c1[obs_idx, k1_hi]; c1_lo = c1[obs_idx, k1_lo]
        c2_hi = c2[obs_idx, k2_hi]; c2_lo = c2[obs_idx, k2_lo]

        nat = _vrdp2d_native(c1_hi, c1_lo, c2_hi, c2_lo, has1_hi, has1_lo, has2_hi, has2_lo,
                             dprime_F, dprime_R, source_d, lambda_val, sigma_item, sigma_S)
        if nat is not None:
            (val, d_lp_d_dpF, d_lp_d_dpR, d_lp_d_sd, d_lp_d_lambda, d_lp_d_si, d_lp_d_sS,
             d_t1_k_hi, d_t1_k_lo, d_t2_k_hi, d_t2_k_lo) = nat
            d_lp_d_gaps1 = _batch_gap_grads(d_t1_k_hi, d_t1_k_lo, k1_hi, k1_lo,
                                            has1_hi, has1_lo, gd1, gap_mask1_np)
            d_lp_d_gaps2 = _batch_gap_grads(d_t2_k_hi, d_t2_k_lo, k2_hi, k2_lo,
                                            has2_hi, has2_lo, gd2, gap_mask2_np)
            tangent_out = (
                d_lp_d_dpF * d_dprime_F + d_lp_d_dpR * d_dprime_R + d_lp_d_sd * d_source_d +
                d_lp_d_lambda * d_lambda + d_lp_d_si * d_sigma_item + d_lp_d_sS * d_sigma_S +
                (d_t1_k_hi + d_t1_k_lo) * d_thresh1_mid + jnp.sum(d_lp_d_gaps1 * d_gaps1, axis=1) +
                (d_t2_k_hi + d_t2_k_lo) * d_thresh2_mid + jnp.sum(d_lp_d_gaps2 * d_gaps2, axis=1))
            return val, tangent_out
        z1_fam_hi = jnp.where(has1_hi, (c1_hi - dprime_F) * inv_si,  INF)
        z1_fam_lo = jnp.where(has1_lo, (c1_lo - dprime_F) * inv_si, -INF)
        z2_hi     = jnp.where(has2_hi,  c2_hi,                       INF)
        z2_lo     = jnp.where(has2_lo,  c2_lo,                      -INF)
        z1_rec_hi = jnp.where(has1_hi, (c1_hi - mu_rec1) * inv_si,   INF)
        z1_rec_lo = jnp.where(has1_lo, (c1_lo - mu_rec1) * inv_si,  -INF)
        z2_rec_hi = jnp.where(has2_hi, (c2_hi - source_d) * inv_sS,  INF)
        z2_rec_lo = jnp.where(has2_lo, (c2_lo - source_d) * inv_sS, -INF)

        P1fh, d1fh, q1fh = _probit_triple(z1_fam_hi)
        P1fl, d1fl, q1fl = _probit_triple(z1_fam_lo)
        P2nh, d2nh, q2nh = _probit_triple(z2_hi)
        P2nl, d2nl, q2nl = _probit_triple(z2_lo)
        P1rh, d1rh, q1rh = _probit_triple(z1_rec_hi)
        P1rl, d1rl, q1rl = _probit_triple(z1_rec_lo)
        P2rh, d2rh, q2rh = _probit_triple(z2_rec_hi)
        P2rl, d2rl, q2rl = _probit_triple(z2_rec_lo)
        D1f = jnp.where(z1_fam_lo >= 0, q1fl - q1fh, P1fh - P1fl)
        D2n = jnp.where(z2_lo >= 0, q2nl - q2nh, P2nh - P2nl)
        D1r = jnp.where(z1_rec_lo >= 0, q1rl - q1rh, P1rh - P1rl)
        D2r = jnp.where(z2_rec_lo >= 0, q2rl - q2rh, P2rh - P2rl)
        p1_fam   = jnp.clip(D1f, _pfloor())
        p2_noise = jnp.clip(D2n, _pfloor())
        p1_rec   = jnp.clip(D1r, _pfloor())
        p2_rec   = jnp.clip(D2r, _pfloor())
        p_fam = p1_fam * p2_noise
        p_rec = p1_rec * p2_rec
        one_m_lam = 1.0 - lambda_val
        prob_raw = one_m_lam * p_fam + lambda_val * p_rec
        prob = jnp.clip(prob_raw, _pfloor())
        val = jnp.log(prob)
        inv_P = jnp.where(prob_raw > _pfloor(), 1.0 / prob, 0.0)

        w_fam1 = jnp.where(D1f > _pfloor(), one_m_lam * p2_noise * inv_si * inv_P, 0.0)
        w_rec1 = jnp.where(D1r > _pfloor(), lambda_val * p2_rec * inv_si * inv_P, 0.0)

        phi1f_hi = jnp.where(has1_hi, d1fh, 0.0)
        phi1f_lo = jnp.where(has1_lo, d1fl, 0.0)
        phi2n_hi = jnp.where(has2_hi, d2nh, 0.0)
        phi2n_lo = jnp.where(has2_lo, d2nl, 0.0)
        phi1r_hi = jnp.where(has1_hi, d1rh, 0.0)
        phi1r_lo = jnp.where(has1_lo, d1rl, 0.0)
        phi2r_hi = jnp.where(has2_hi, d2rh, 0.0)
        phi2r_lo = jnp.where(has2_lo, d2rl, 0.0)

        d_t1_k_hi = w_fam1 * phi1f_hi + w_rec1 * phi1r_hi
        d_t1_k_lo = -(w_fam1 * phi1f_lo + w_rec1 * phi1r_lo)

        sum_fam1 = phi1f_hi - phi1f_lo
        sum_rec1 = phi1r_hi - phi1r_lo
        d_lp_d_dpF = -(w_fam1 * sum_fam1 + w_rec1 * sum_rec1)
        d_lp_d_dpR = -w_rec1 * sum_rec1
        d_lp_d_si = -(w_fam1 * (phi1f_hi * z1_fam_hi - phi1f_lo * z1_fam_lo) +
                       w_rec1 * (phi1r_hi * z1_rec_hi - phi1r_lo * z1_rec_lo))

        w_fam2 = jnp.where(D2n > _pfloor(), one_m_lam * p1_fam * inv_P, 0.0)
        w_rec2 = jnp.where(D2r > _pfloor(), lambda_val * p1_rec * inv_sS * inv_P, 0.0)

        d_t2_k_hi = w_fam2 * phi2n_hi + w_rec2 * phi2r_hi
        d_t2_k_lo = -(w_fam2 * phi2n_lo + w_rec2 * phi2r_lo)

        d_lp_d_sd = -w_rec2 * (phi2r_hi - phi2r_lo)
        d_lp_d_sS = -w_rec2 * (phi2r_hi * z2_rec_hi - phi2r_lo * z2_rec_lo)

        d_lp_d_lambda = (p_rec - p_fam) * inv_P

        d_lp_d_thresh1_mid = d_t1_k_hi + d_t1_k_lo
        d_lp_d_thresh2_mid = d_t2_k_hi + d_t2_k_lo

        d_lp_d_gaps1 = _batch_gap_grads(d_t1_k_hi, d_t1_k_lo, k1_hi, k1_lo,
                                         has1_hi, has1_lo, gd1, gap_mask1_np)
        d_lp_d_gaps2 = _batch_gap_grads(d_t2_k_hi, d_t2_k_lo, k2_hi, k2_lo,
                                         has2_hi, has2_lo, gd2, gap_mask2_np)

        tangent_out = (
            d_lp_d_dpF * d_dprime_F +
            d_lp_d_dpR * d_dprime_R +
            d_lp_d_sd * d_source_d +
            d_lp_d_lambda * d_lambda +
            d_lp_d_si * d_sigma_item +
            d_lp_d_sS * d_sigma_S +
            d_lp_d_thresh1_mid * d_thresh1_mid +
            jnp.sum(d_lp_d_gaps1 * d_gaps1, axis=1) +
            d_lp_d_thresh2_mid * d_thresh2_mid +
            jnp.sum(d_lp_d_gaps2 * d_gaps2, axis=1)
        )

        return val, tangent_out

    return _batch_fused


# =============================================================================
# Batch-vectorized CDP loglik (no vmap)
# =============================================================================

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
    def _batch_fused(y, rk, is_old_val, old_idx, is_old_resp_val,
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

        has_lo = (is_old_resp_val == 1)
        has_hi = ~has_lo | (old_idx < J - 1)
        k_hi = jnp.where(has_lo, jnp.clip(old_idx + 1, 0, J - 1), 0)
        tau_lo = jnp.where(has_lo, c[obs_idx, jnp.clip(old_idx, 0, J - 1)], -_OPEN)
        tau_hi = jnp.where(has_hi, c[obs_idx, k_hi], _OPEN)
        judged = rk > 0

        z_cR = (c_R - mu_R) / jnp.clip(sigma_R, 1e-10)
        z_lo = jnp.where(has_lo, (tau_lo - mu_M) / jnp.clip(sigma_M, 1e-10), -_OPEN)
        z_hi = jnp.where(has_hi, (tau_hi - mu_M) / jnp.clip(sigma_M, 1e-10), _OPEN)


        if n_rkg == 2:
            (c0, c1_), _ = _bvn_cdf_multi([(z_cR, z_hi), (z_cR, z_lo)], [z_cR], rho)
            bv_cR_hi, bv_cR_lo = c0[0], c1_[0]
            p_band = _band_from_q(z_lo, z_hi, c1_[6], c0[6])
            p_K_strip = bv_cR_hi - bv_cR_lo
            p_rk = jnp.where(rk == 1, p_band - p_K_strip, p_K_strip)
            tail = judged & (p_rk < _TAIL_SWITCH)
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
            tail = judged & ~(p_sel >= _TAIL_SWITCH)
            t_lp, _ = _cdp_tail(tail, rk, mu_R, sigma_R, mu_F, sigma_F, c_R, c_K,
                                jnp.where(has_lo, tau_lo, -jnp.inf), jnp.where(has_hi, tau_hi, jnp.inf), corr)
            p_rk = jnp.where(tail, jnp.exp(t_lp), p_sel)

        prob = jnp.where(judged, p_rk, p_band)
        val = jnp.log(jnp.clip(prob, _pfloor()))
        if n_rkg == 3:
            empty = judged & (rk == 3) & has_lo & (c_R + c_K <= tau_lo)
            val = jnp.where(empty, _empty_cell_lp(tau_lo - c_R - c_K), val)
        return val

    @_batch_fused.defjvp
    def _batch_fused_jvp(primals, tangents):
        (y, rk, is_old_val, old_idx, is_old_resp_val,
         mu_R, sigma_R, mu_F, sigma_F, c_R, c_K, corr,
         thresh_mid_v, gaps_v) = primals
        (_, _, _, _, _,
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

        has_tau_lo = (is_old_resp_val == 1)
        has_tau_hi = ~has_tau_lo | (old_idx < J - 1)
        k_tau_lo = jnp.clip(old_idx, 0, n_thresh - 1)
        k_tau_hi = jnp.where(has_tau_lo, jnp.clip(old_idx + 1, 0, n_thresh - 1), 0)
        judged = rk > 0

        tau_lo = jnp.where(has_tau_lo, c[obs_idx, k_tau_lo], -_OPEN)
        tau_hi = jnp.where(has_tau_hi, c[obs_idx, k_tau_hi], _OPEN)

        z_cR = (c_R - mu_R) * inv_sR
        z_hi = jnp.where(has_tau_hi, (tau_hi - mu_M) * inv_sM, _OPEN)
        z_lo = jnp.where(has_tau_lo, (tau_lo - mu_M) * inv_sM, -_OPEN)


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
        phi_lo = jnp.where(has_tau_lo, cdfs[1][7], 0.0)

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
            tail = judged & (p_rk_raw < _TAIL_SWITCH)
            t_p, t_dzcR, t_dzlo, t_dzhi, t_drho = _rk_tail(tail, is_R, z_cR, z_lo, z_hi, rho)
            p_rk_raw = jnp.where(tail, t_p, p_rk_raw)
            p_rk = jnp.clip(p_rk_raw, _pfloor())
            prob_raw = jnp.where(judged, p_rk_raw, p_band)
            prob = jnp.where(judged, p_rk, p_band)
            val = jnp.log(jnp.clip(prob, _pfloor()))
            inv_prob = jnp.where(prob_raw > _pfloor(), 1.0 / jnp.clip(prob, _pfloor()), 0.0)

            g_zcR = jnp.where(tail, t_dzcR, jnp.where(is_R, -d_bvCRhi_z1 + d_bvCRlo_z1,
                                                        d_bvCRhi_z1 - d_bvCRlo_z1))
            g_zhi = jnp.where(tail, t_dzhi, jnp.where(is_R, phi_hi - d_bvCRhi_z2, d_bvCRhi_z2))
            g_zlo = jnp.where(tail, t_dzlo, jnp.where(is_R, -(phi_lo - d_bvCRlo_z2), -d_bvCRlo_z2))
            g_rho = jnp.where(tail, t_drho, jnp.where(is_R, -d_bvCRhi_rho + d_bvCRlo_rho,
                                                        d_bvCRhi_rho - d_bvCRlo_rho))
            dlp_dzcR = jnp.where(judged, g_zcR * inv_prob, 0.0)
            dlp_dzhi = jnp.where(has_tau_hi, jnp.where(judged, g_zhi, phi_hi) * inv_prob, 0.0)
            dlp_dzlo = jnp.where(has_tau_lo, jnp.where(judged, g_zlo, -phi_lo) * inv_prob, 0.0)
            dlp_drho = jnp.where(judged, g_rho * inv_prob, 0.0)

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
            tail = judged & ~(p_sel >= _TAIL_SWITCH)
            t_lp, t_d = _cdp_tail(tail, rk, mu_R, sigma_R, mu_F, sigma_F, c_R, c_K,
                                  jnp.where(has_tau_lo, tau_lo, -jnp.inf),
                                  jnp.where(has_tau_hi, tau_hi, jnp.inf), corr)
            p_cell = jnp.where(tail, jnp.exp(t_lp), p_sel)
            p_rk = p_cell
            prob = jnp.where(judged, p_rk, p_band)
            val = jnp.log(jnp.clip(prob, _pfloor()))
            inv_prob = jnp.where(prob > _pfloor(), 1.0 / jnp.clip(prob, _pfloor()), 0.0)
            closed = judged & ~tail

            dlp_dpR = jnp.where(closed & (rk == 1), inv_prob, 0.0)
            dlp_dpK = jnp.where(closed & (rk == 2), inv_prob, 0.0)
            dlp_dpG = jnp.where(closed & (rk == 3), inv_prob, 0.0)

            eff_dpR = dlp_dpR - dlp_dpK
            eff_dpG = dlp_dpG - dlp_dpK
            eff_dpBand = dlp_dpK + jnp.where(judged, 0.0, inv_prob)


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

            empty = judged & (rk == 3) & has_tau_lo & (c_R + c_K <= tau_lo)
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

        d_t_k_lo = jnp.where(has_tau_lo, d_val_dtau_lo, 0.0)
        d_t_k_hi = jnp.where(has_tau_hi, d_val_dtau_hi, 0.0)

        d_lp_d_thresh_mid = d_t_k_lo + d_t_k_hi

        gm = jnp.asarray(gap_mask_np)
        mask_tau_lo = gm[:, k_tau_lo]
        mask_tau_hi = gm[:, k_tau_hi]

        d_lp_d_gap_val = (mask_tau_lo * d_t_k_lo[None, :] +
                          mask_tau_hi * d_t_k_hi[None, :])
        d_lp_d_gaps = (d_lp_d_gap_val * gd.T).T

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
    INF = _OPEN

    def _core(y1, y2, item_type, mu1, mu2, sigma1, sigma2, rho_val,
              R_I, R_S, R_I_B, R_S_B, thresh1_mid, gv1, thresh2_mid, gv2):
        c1 = _batch_build_thresh(thresh1_mid, gv1, n_gaps1, mid_idx1, n_upper1)
        c2 = _batch_build_thresh(thresh2_mid, gv2, n_gaps2, mid_idx2, n_upper2)
        N = y1.shape[0]
        r1 = y1 - 1
        r2 = y2 - 1
        obs_idx = jnp.arange(N)
        k1_hi = jnp.clip(r1, 0, n_thresh1 - 1)
        k1_lo = jnp.clip(r1 - 1, 0, n_thresh1 - 1)
        k2_hi = jnp.clip(r2, 0, n_thresh2 - 1)
        k2_lo = jnp.clip(r2 - 1, 0, n_thresh2 - 1)
        has1_hi = r1 < K1 - 1
        has1_lo = r1 > 0
        has2_hi = r2 < K2 - 1
        has2_lo = r2 > 0
        inv_s1 = 1.0 / sigma1
        inv_s2 = 1.0 / sigma2
        c2_hi = c2[obs_idx, k2_hi]
        c2_lo = c2[obs_idx, k2_lo]
        z1_hi = jnp.where(has1_hi, (c1[obs_idx, k1_hi] - mu1) * inv_s1, INF)
        z1_lo = jnp.where(has1_lo, (c1[obs_idx, k1_lo] - mu1) * inv_s1, -INF)
        z2_hi = jnp.where(has2_hi, (c2_hi - mu2) * inv_s2, INF)
        z2_lo = jnp.where(has2_lo, (c2_lo - mu2) * inv_s2, -INF)
        sqrt_1mrr = jnp.sqrt(jnp.clip(1.0 - rho_val**2, 1e-20))
        sc_floor = sigma2 * sqrt_1mrr <= sigma2 * 1e-6
        sigma_cond = jnp.maximum(sigma2 * sqrt_1mrr, sigma2 * 1e-6)
        inv_sc = 1.0 / sigma_cond
        rho_sigma2 = rho_val * sigma2 + 1e-30
        is_new = item_type == 1
        z_cp = jnp.where(is_new, -1e6, -mu2 / rho_sigma2)
        z_cp_eff = jnp.clip(z_cp, z1_lo, z1_hi)
        zs_hi = jnp.where(has2_hi, c2_hi * inv_sc, INF)
        zs_lo = jnp.where(has2_lo, c2_lo * inv_sc, -INF)

        D_det, dDdet_lo, dDdet_hi = _phi_diff(z1_lo, z_cp_eff)
        D_src, dDsrc_lo, dDsrc_hi = _phi_diff(zs_lo, zs_hi)
        det_on = D_det > 0.0
        src_on = D_src > 0.0
        p_det = jnp.where(det_on, D_det, 0.0)
        p_src = jnp.where(src_on, D_src, 0.0)
        pa, dpa_a, dpa_b, dpa_c, dpa_d, dpa_r = _bvn_rect(z_cp_eff, z1_hi, z2_lo, z2_hi, rho_val)
        above_on = pa > 0.0
        biv_raw = p_det * p_src + jnp.where(above_on, pa, 0.0)
        biv_on = biv_raw > _pfloor()
        bounded_biv = jnp.where(biv_on, biv_raw, _pfloor())

        Pcp, _, dPcp = _phi_diff(jnp.full_like(z_cp, -INF), z_cp)
        pm, dpm_a, _, dpm_c, dpm_d, dpm_r = _bvn_rect(z_cp, jnp.full_like(z_cp, INF), z2_lo, z2_hi, rho_val)
        marg_raw = Pcp * D_src + pm
        marg_on = marg_raw > _pfloor()
        bounded_marg = jnp.where(marg_on, marg_raw, _pfloor())

        is_A = item_type == 2
        R_I_eff = jnp.where(is_A, R_I, R_I_B)
        R_S_eff = jnp.where(is_A, R_S, R_S_B)
        is_y1_top = y1 == K1
        is_corner = (is_A & (y1 == K1) & (y2 == 1)) | ((item_type == 3) & (y1 == K1) & (y2 == K2))
        w1 = 1.0 - R_I_eff
        w2 = jnp.where(is_y1_top, R_I_eff * (1.0 - R_S_eff), 0.0)
        prob_raw = w1 * bounded_biv + w2 * bounded_marg + jnp.where(is_corner, R_I_eff * R_S_eff, 0.0)
        prob = jnp.clip(prob_raw, _pfloor())
        parts = dict(has1_hi=has1_hi, has1_lo=has1_lo, has2_hi=has2_hi, has2_lo=has2_lo,
                     k1_hi=k1_hi, k1_lo=k1_lo, k2_hi=k2_hi, k2_lo=k2_lo,
                     inv_s1=inv_s1, inv_s2=inv_s2, inv_sc=inv_sc, sc_floor=sc_floor,
                     sqrt_1mrr=sqrt_1mrr, rho_sigma2=rho_sigma2, is_new=is_new,
                     z1_hi=z1_hi, z1_lo=z1_lo, z2_hi=z2_hi, z2_lo=z2_lo, zs_hi=zs_hi, zs_lo=zs_lo,
                     z_cp=z_cp, det_on=det_on, src_on=src_on, above_on=above_on,
                     biv_on=biv_on, marg_on=marg_on, p_det=p_det, p_src=p_src, D_src=D_src,
                     dDdet_lo=dDdet_lo, dDdet_hi=dDdet_hi, dDsrc_lo=dDsrc_lo, dDsrc_hi=dDsrc_hi,
                     dpa_a=dpa_a, dpa_b=dpa_b, dpa_c=dpa_c, dpa_d=dpa_d, dpa_r=dpa_r,
                     Pcp=Pcp, dPcp=dPcp, dpm_a=dpm_a, dpm_c=dpm_c, dpm_d=dpm_d, dpm_r=dpm_r,
                     R_I_eff=R_I_eff, R_S_eff=R_S_eff, is_A=is_A, is_y1_top=is_y1_top,
                     is_corner=is_corner, w1=w1, w2=w2, bounded_biv=bounded_biv,
                     bounded_marg=bounded_marg, prob_raw=prob_raw)
        return jnp.log(prob), prob, parts

    @custom_jvp
    def _batch_fused(y1, y2, item_type, mu1, mu2, sigma1, sigma2, rho_val, rho_N_val,
                     R_I, R_S, R_I_B, R_S_B, thresh1_mid, gaps1, thresh2_mid, gaps2):
        gap_fn = jax.nn.softplus if gap_link_is_softplus else jnp.exp
        return _core(y1, y2, item_type, mu1, mu2, sigma1, sigma2, rho_val, R_I, R_S, R_I_B, R_S_B,
                     thresh1_mid, gap_fn(gaps1), thresh2_mid, gap_fn(gaps2))[0]

    @_batch_fused.defjvp
    def _batch_fused_jvp(primals, tangents):
        (y1, y2, item_type, mu1, mu2, sigma1, sigma2, rho_val, rho_N_val,
         R_I, R_S, R_I_B, R_S_B, thresh1_mid, gaps1, thresh2_mid, gaps2) = primals
        (_, _, _, d_mu1, d_mu2, d_sigma1, d_sigma2, d_rho, d_rho_N,
         d_R_I, d_R_S, d_R_I_B, d_R_S_B, d_thresh1_mid, d_gaps1, d_thresh2_mid, d_gaps2) = tangents
        if gap_link_is_softplus:
            gv1 = jax.nn.softplus(gaps1)
            gd1 = jax.nn.sigmoid(gaps1)
            gv2 = jax.nn.softplus(gaps2)
            gd2 = jax.nn.sigmoid(gaps2)
        else:
            gv1 = jnp.exp(gaps1)
            gd1 = gv1
            gv2 = jnp.exp(gaps2)
            gd2 = gv2
        val, prob, q = _core(y1, y2, item_type, mu1, mu2, sigma1, sigma2, rho_val, R_I, R_S, R_I_B, R_S_B,
                             thresh1_mid, gv1, thresh2_mid, gv2)
        inv_P = jnp.where(q["prob_raw"] > _pfloor(), 1.0 / prob, 0.0)

        def f(m):
            return m.astype(prob.dtype)

        gb = q["w1"] * f(q["biv_on"]) * inv_P
        gm = q["w2"] * f(q["marg_on"]) * inv_P
        det_g = f(q["det_on"]) * q["p_src"]
        src_g = f(q["src_on"]) * q["p_det"]
        ab = f(q["above_on"])
        g_zcpeff = gb * (det_g * q["dDdet_hi"] + ab * q["dpa_a"])
        g_z1_lo = gb * det_g * q["dDdet_lo"]
        g_z1_hi = gb * ab * q["dpa_b"]
        g_z2_lo = gb * ab * q["dpa_c"] + gm * q["dpm_c"]
        g_z2_hi = gb * ab * q["dpa_d"] + gm * q["dpm_d"]
        g_zs_lo = (gb * src_g + gm * q["Pcp"]) * q["dDsrc_lo"]
        g_zs_hi = (gb * src_g + gm * q["Pcp"]) * q["dDsrc_hi"]
        g_zcp = gm * (q["dPcp"] * q["D_src"] + q["dpm_a"])
        g_rho_direct = gb * ab * q["dpa_r"] + gm * q["dpm_r"]

        z_cp = q["z_cp"]
        split = (z_cp > q["z1_lo"]) & (z_cp < q["z1_hi"])
        g_zcp = g_zcp + jnp.where(split, g_zcpeff, 0.0)
        g_z1_lo = g_z1_lo + jnp.where(z_cp <= q["z1_lo"], g_zcpeff, 0.0)
        g_z1_hi = g_z1_hi + jnp.where(z_cp >= q["z1_hi"], g_zcpeff, 0.0)
        g_zcp = jnp.where(q["is_new"], 0.0, g_zcp)

        g_z1_hi = jnp.where(q["has1_hi"], g_z1_hi, 0.0)
        g_z1_lo = jnp.where(q["has1_lo"], g_z1_lo, 0.0)
        g_z2_hi = jnp.where(q["has2_hi"], g_z2_hi, 0.0)
        g_z2_lo = jnp.where(q["has2_lo"], g_z2_lo, 0.0)
        g_zs_hi = jnp.where(q["has2_hi"], g_zs_hi, 0.0)
        g_zs_lo = jnp.where(q["has2_lo"], g_zs_lo, 0.0)

        inv_s1 = q["inv_s1"]
        inv_s2 = q["inv_s2"]
        inv_sc = q["inv_sc"]
        rs2 = q["rho_sigma2"]
        d_mu1_c = -(g_z1_hi + g_z1_lo) * inv_s1
        d_s1_c = -(g_z1_hi * q["z1_hi"] + g_z1_lo * q["z1_lo"]) * inv_s1
        d_mu2_c = -(g_z2_hi + g_z2_lo) * inv_s2 - g_zcp / rs2
        zs_term = g_zs_hi * q["zs_hi"] + g_zs_lo * q["zs_lo"]
        not_floor = f(~q["sc_floor"] & (1.0 - rho_val**2 > 1e-20))
        d_s2_c = (-(g_z2_hi * q["z2_hi"] + g_z2_lo * q["z2_lo"]) * inv_s2
                  - zs_term * inv_s2 + g_zcp * mu2 * rho_val / (rs2 * rs2))
        d_rho_c = (g_rho_direct + g_zcp * mu2 * sigma2 / (rs2 * rs2)
                   + zs_term * not_floor * rho_val / (q["sqrt_1mrr"] ** 2))

        d_t1_hi = g_z1_hi * inv_s1
        d_t1_lo = g_z1_lo * inv_s1
        d_t2_hi = g_z2_hi * inv_s2 + g_zs_hi * inv_sc
        d_t2_lo = g_z2_lo * inv_s2 + g_zs_lo * inv_sc
        d_gaps1_c = _batch_gap_grads(d_t1_hi, d_t1_lo, q["k1_hi"], q["k1_lo"], q["has1_hi"], q["has1_lo"],
                                     gd1, gap_mask1_np)
        d_gaps2_c = _batch_gap_grads(d_t2_hi, d_t2_lo, q["k2_hi"], q["k2_lo"], q["has2_hi"], q["has2_lo"],
                                     gd2, gap_mask2_np)

        R_I_eff = q["R_I_eff"]
        R_S_eff = q["R_S_eff"]
        top = f(q["is_y1_top"])
        corner = f(q["is_corner"])
        d_RI_c = (-q["bounded_biv"] + top * (1.0 - R_S_eff) * q["bounded_marg"] + corner * R_S_eff) * inv_P
        d_RS_c = (-top * R_I_eff * q["bounded_marg"] + corner * R_I_eff) * inv_P
        d_RI_in = jnp.where(q["is_A"], d_R_I, d_R_I_B)
        d_RS_in = jnp.where(q["is_A"], d_R_S, d_R_S_B)

        tangent_out = (d_mu1_c * d_mu1 + d_mu2_c * d_mu2 + d_s1_c * d_sigma1 + d_s2_c * d_sigma2
                       + d_rho_c * d_rho + d_RI_c * d_RI_in + d_RS_c * d_RS_in
                       + (d_t1_hi + d_t1_lo) * d_thresh1_mid + jnp.sum(d_gaps1_c * d_gaps1, axis=1)
                       + (d_t2_hi + d_t2_lo) * d_thresh2_mid + jnp.sum(d_gaps2_c * d_gaps2, axis=1))
        return val, tangent_out

    return _batch_fused


def fused_loglik_bounded_bivariate_dp(y1, y2, item_type, K1, K2, mu1, mu2, sigma1, sigma2,
                                       rho_val, rho_N_val, R_I, R_S,
                                       R_I_B=None, R_S_B=None,
                                       thresh1_mid=None, gaps1=None,
                                       thresh2_mid=None, gaps2=None, gap_link="log"):
    """Fully-vectorized fused log-likelihood for bounded bivariate_dp."""
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
    return _fused_cache[key](y1, y2, item_type, mu1, mu2, sigma1, sigma2,
                              rho_val, rho_N_val, R_I, R_S, R_I_B, R_S_B,
                              thresh1_mid, gaps1_per_obs, thresh2_mid, gaps2_per_obs)


def fused_loglik_bivariate_dp(y1, y2, item_type, K1, K2, mu1, mu2, sigma1, sigma2,
                               rho_val, rho_N_val, R_I, R_S,
                               R_I_B=None, R_S_B=None,
                               thresh1_mid=None, gaps1=None,
                               thresh2_mid=None, gaps2=None, gap_link="log"):
    """Fully-vectorized fused log-likelihood for bivariate_dp."""
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

    key = ("batch_bivariate_dp", int(K1), int(K2), bool(is_softplus))
    if key not in _fused_cache:
        _fused_cache[key] = _make_fused_batch_bivariate_dp_loglik(
            int(K1), int(K2), bool(is_softplus))
    return _fused_cache[key](y1, y2, item_type, mu1, mu2, sigma1, sigma2,
                              rho_val, rho_N_val, R_I, R_S, R_I_B, R_S_B,
                              thresh1_mid, gaps1_per_obs, thresh2_mid, gaps2_per_obs)


def fused_loglik_cdp(y, rk, is_old, old_idx, is_old_resp, K, J,
                      mu_R, sigma_R, mu_F, sigma_F, c_R, c_K,
                      thresh_mid, gaps, gap_link="log", n_rkg=2, corr=None):
    """Fully-vectorized fused log-likelihood for CDP."""
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
    return _fused_cache[key](y, rk, is_old, old_idx, is_old_resp,
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
                       gap_link="log", has_sigma=False, link_name="probit"):
    """Fully-vectorized fused log-likelihood for DPSDT (old items only)."""
    N = y.shape[0]
    n_gaps = K - 2
    is_softplus = (gap_link == "softplus")

    if gaps.ndim == 1:
        gaps_per_obs = jnp.broadcast_to(gaps[None, :], (N, n_gaps))
    else:
        gaps_per_obs = gaps.T

    key = ("batch_dpsdt", int(K), bool(is_softplus), str(link_name))
    if key not in _fused_cache:
        _fused_cache[key] = _make_fused_batch_dpsdt_loglik(int(K), bool(is_softplus), str(link_name))
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
                     gap_link="log",
                     **kwargs):
    """Compute log-likelihood for bivariate SDT families via the fused kernels;
    requires thresh1_mid/log_gaps1/thresh2_mid/log_gaps2."""

    N = y1.shape[0]
    K1 = thresh1.shape[1] + 1
    K2 = thresh2.shape[1] + 1

    if family_name == "bivariate_cumulative":
        _has_fused = (thresh1_mid is not None and log_gaps1 is not None and
                      thresh2_mid is not None and log_gaps2 is not None)
        if _has_fused:
            mu1_c = dprime if dprime is not None else jnp.zeros(N)
            mu2_c = discrim if discrim is not None else jnp.zeros(N)
            rho_c = rho if rho is not None else jnp.zeros(N)
            log_lik = fused_loglik_bivariate_sdt(
                y1, y2, item_type, K1, K2,
                mu1=mu1_c, mu2=mu2_c,
                sigma1=jnp.ones(N), sigma2=jnp.ones(N), rho_val=rho_c,
                thresh1_mid=thresh1_mid, gaps1=log_gaps1,
                thresh2_mid=thresh2_mid, gaps2=log_gaps2, gap_link=gap_link)
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
            log_lik = fused_loglik_vrdp2d(
                y1, y2, item_type, K1, K2,
                dprime_F=dprime_masked, dprime_R=dprime2_masked, source_d=mu2,
                lambda_val=R_masked, sigma_item=sd1, sigma_S=sd2,
                thresh1_mid=thresh1_mid, gaps1=log_gaps1,
                thresh2_mid=thresh2_mid, gaps2=log_gaps2,
                gap_link=gap_link)
            if counts is not None:
                log_lik = log_lik * counts
            return log_lik

        elif family_name == "bivariate_sdt" and not bounded:
            log_lik = fused_loglik_bivariate_sdt(
                y1, y2, item_type, K1, K2,
                mu1=mu1, mu2=mu2, sigma1=sd1, sigma2=sd2, rho_val=rho_n,
                thresh1_mid=thresh1_mid, gaps1=log_gaps1,
                thresh2_mid=thresh2_mid, gaps2=log_gaps2,
                gap_link=gap_link)
            if counts is not None:
                log_lik = log_lik * counts
            return log_lik

        elif family_name == "bivariate_dp" and not bounded:
            R_I = lambda_val if lambda_val is not None else jnp.zeros(N)
            R_S = lambda2 if lambda2 is not None else jnp.zeros(N)
            R_I_B = lambda_B if lambda_B is not None else R_I
            R_S_B = lambda2_B if lambda2_B is not None else R_S
            R_I = jnp.where(is_new, 0.0, R_I)
            R_S = jnp.where(is_new, 0.0, R_S)
            R_I_B = jnp.where(is_new, 0.0, R_I_B)
            R_S_B = jnp.where(is_new, 0.0, R_S_B)
            log_lik = fused_loglik_bivariate_dp(
                y1, y2, item_type, K1, K2,
                mu1=mu1, mu2=mu2, sigma1=sd1, sigma2=sd2,
                rho_val=rho_n, rho_N_val=rho_N,
                R_I=R_I, R_S=R_S, R_I_B=R_I_B, R_S_B=R_S_B,
                thresh1_mid=thresh1_mid, gaps1=log_gaps1,
                thresh2_mid=thresh2_mid, gaps2=log_gaps2,
                gap_link=gap_link)
            if counts is not None:
                log_lik = log_lik * counts
            return log_lik

        elif bounded and family_name == "bivariate_sdt":
            log_lik = fused_loglik_bounded_bivariate_sdt(
                y1, y2, item_type, K1, K2,
                mu1=mu1, mu2=mu2, sigma1=sd1, sigma2=sd2, rho_val=rho_n,
                thresh1_mid=thresh1_mid, gaps1=log_gaps1,
                thresh2_mid=thresh2_mid, gaps2=log_gaps2,
                gap_link=gap_link)
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
            log_lik = fused_loglik_bounded_bivariate_dp(
                y1, y2, item_type, K1, K2,
                mu1=mu1, mu2=mu2, sigma1=sd1, sigma2=sd2,
                rho_val=rho_n, rho_N_val=rho_N,
                R_I=R_I, R_S=R_S, R_I_B=R_I_B, R_S_B=R_S_B,
                thresh1_mid=thresh1_mid, gaps1=log_gaps1,
                thresh2_mid=thresh2_mid, gaps2=log_gaps2,
                gap_link=gap_link)
            if counts is not None:
                log_lik = log_lik * counts
            return log_lik

    raise RuntimeError("bivariate_loglik requires fused inputs "
                       "(thresh*_mid / log_gaps*), which were not provided")


# =============================================================================
# CDP Log-Likelihood
# =============================================================================

def cdp_loglik(y, rk, is_old, thresh, mu_R, sigma_R, mu_F, sigma_F,
               c_R, c_K, J, old_level_map, n_rkg, counts=None,
               thresh_mid=None, log_gaps=None, gap_link="log", rk_static=None, y_static=None, corr=None):
    """Compute log-likelihood for CDP (Continuous Dual-Process) model.

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
        is_old_resp = jnp.any(match, axis=1).astype(_FLOAT)
        old_idx = jnp.argmax(match, axis=1)

        def judged_ll(i):
            return fused_loglik_cdp(
                y=y[i], rk=rk[i], is_old=is_old[i],
                old_idx=old_idx[i], is_old_resp=is_old_resp[i],
                K=K, J=J,
                mu_R=mu_R[i], sigma_R=sigma_R[i], mu_F=mu_F[i], sigma_F=sigma_F[i],
                c_R=c_R[i], c_K=c_K[i],
                thresh_mid=thresh_mid[i], gaps=_gap_rows(log_gaps, i),
                gap_link=gap_link, n_rkg=n_rkg, corr=None if corr is None else corr[i])

        unjudged = None if rk_static is None else np.asarray(rk_static) == 0
        y_np = None if y_static is None else np.asarray(y_static)
        if unjudged is None or not unjudged.any():
            log_lik = judged_ll(jnp.arange(N))
        else:
            hit = y_np[:, None] == np.asarray(old_level_map)[None, :]
            y_band = jnp.asarray(np.where(hit.any(axis=1), hit.argmax(axis=1) + 2, 1), dtype=jnp.int32)
            mu_M = mu_R + mu_F
            sd_M = jnp.sqrt(sigma_R ** 2 + sigma_F ** 2 + (0.0 if corr is None else 2.0 * corr * sigma_R * sigma_F))
            u, j = np.flatnonzero(unjudged), np.flatnonzero(~unjudged)
            log_lik = jnp.zeros(N, dtype=_FLOAT).at[u].set(
                fused_loglik_simple(y_band[u], J + 1, mu_M[u], sd_M[u], thresh_mid[u], _gap_rows(log_gaps, u),
                                    gap_link))
            if len(j):
                log_lik = log_lik.at[j].set(judged_ll(j))
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


def _rkg_knob(config):
    """Static setup for sampling the CDP R/K/G know criterion on its feasible set.

    A Guess response in the confidence band starting at tau_k is possible only when
    c_R + c_K > tau_k, so the likelihood has hard walls in the know-criterion
    coefficients. With v solving X v = 1 (the fixed-effect design spans the constant),
    beta = B_perp w + t v and t is solved last as max over Guess trials of the bound plus
    exp(slack): an exact bijection onto the feasible set whose coordinates never meet a
    wall. Returns None when no Guess response constrains the model."""
    if config.get("family") != "cdp" or int(config.get("n_rkg", 2)) != 3:
        return None
    # fixed-effects bounds only: with random effects on the thresholds or criteria the bound
    # couples every effect through a max over trials and sampling collapses
    if any(rc.get("param") in ("criterion", "rec_crit", "know_crit") for rc in config.get("random_effects") or []):
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
    live = np.ones(y.shape[0], bool) if config.get("counts") is None else np.asarray(config["counts"]) > 0
    G = np.where((band >= 0) & (rk == 3) & live)[0]
    if G.size == 0:
        return None
    v = np.linalg.lstsq(X, np.ones(X.shape[0]), rcond=None)[0]
    if not np.allclose(X @ v, 1.0, atol=1e-9):
        raise ValueError("know_crit: the fixed-effect design must span the constant (an intercept "
                         "or a full factor coding)")
    d = X.shape[1]
    B_perp = np.linalg.svd(v[None, :])[2][1:].T if d > 1 else np.zeros((1, 0))
    return {"X": X, "v": v, "B_perp": B_perp, "G": G, "band": band[G]}


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
            if n_coef > 1:
                w = numpyro.sample("know_crit_free",
                                   dist.ImproperUniform(dist.constraints.real, (), (n_coef - 1,)))
                beta_perp = B_perp @ w
            else:
                beta_perp = jnp.zeros(1, dtype=_FLOAT)
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

    cross_cor_groups = {}
    independent_re_idxs = []

    for re_idx, re_conf in enumerate(re_configs):
        param = re_conf.get("param")
        cor_id = re_conf.get("cor_id")
        if param in ("criterion", "criterion2") and cor_id is None:
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
                u = sd * z
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
                for t in range(n_terms):
                    sd_t = numpyro.sample(f"sigma_{param}_{group}[{t+1}]", sd_priors[t])
                    z_t = numpyro.sample(f"z_{param}_{group}[{t+1}]",
                                         dist.Normal(0, 1).expand([n_groups]))
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

                numpyro.deterministic(f"u_{param}_{group}", u_all.T)
        elif re_conf.get("whiten") is not None and not re_conf.get("cor_blocks"):
            sd_list = sd_priors if per_coef_varies else [sd_prior] * n_terms
            sds, L_corr, L = _whitened_cov(f"{param}_{group}", re_conf, n_terms, sd_list)
            numpyro.deterministic(f"corr_{param}_{group}", L_corr @ L_corr.T)
            z = numpyro.sample(f"z_{param}_{group}",
                               dist.Normal(0, 1).expand([n_terms, n_groups]))
            u = L @ z
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
            u = L @ z

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
            _sample_independent_re(re_configs[batch_idxs[0]], param_values)
            continue

        n_batch = len(batch_idxs)
        sd_prior = _positive_dist(parse_prior(prior_sd_str))
        batch_params = [re_configs[i]["param"] for i in batch_idxs]

        batch_tag = "_".join(sorted(batch_params))
        sds = numpyro.sample(f"sigma_batch_{batch_tag}_{group}",
                              sd_prior.expand([n_batch]))
        z_all = numpyro.sample(f"z_batch_{batch_tag}_{group}",
                                dist.Normal(0, 1).expand([n_batch, n_groups]))
        u_all = sds[:, None] * z_all

        for b, re_idx in enumerate(batch_idxs):
            rc = re_configs[re_idx]
            param = rc["param"]
            u = u_all[b]
            numpyro.deterministic(f"sigma_{param}_{group}[1]", sds[b])
            numpyro.deterministic(f"z_{param}_{group}[1]", z_all[b])
            numpyro.deterministic(f"u_{param}_{group}", u)
            idx = jnp.array(rc["index"], dtype=jnp.int32) - 1
            _apply_re_offset(param, u[idx], param_values)

    for re_idx in _unbatched_idxs:
        _sample_independent_re(re_configs[re_idx], param_values)

    cross_cor_criterion_contributions = []

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
        numpyro.deterministic(f"corr_cross_{cor_id}_{group}", L_corr @ L_corr.T)
        L = jnp.diag(all_sds) @ L_corr
        numpyro.deterministic(f"sigma_cross_{cor_id}_{group}", all_sds)

        z = numpyro.sample(f"z_cross_{cor_id}_{group}",
                           dist.Normal(0, 1).expand([total_dim, n_groups]))
        u = L @ z

        for param, nt, d_off, m_conf in member_dims:
            u_portion = u[d_off:d_off + nt]
            u_det = u_portion[0] if nt == 1 else u_portion.T
            numpyro.deterministic(f"u_{param}_{group}_from_{cor_id}", u_det)

            if param in ("criterion", "criterion2"):
                idx_crit = jnp.array(m_conf["index"], dtype=jnp.int32) - 1
                cross_cor_criterion_contributions.append((u_portion, idx_crit, m_conf))
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
        if P_crit == 1 and len(set(gap_pri)) == 1:
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
            u_crit = L_crit @ z_crit
        else:
            u_crit = jnp.zeros((n_dims, n_groups_crit))
            for t in range(n_dims):
                sd_prior_t = (_positive_dist(parse_prior(per_coef_sd[t]))
                              if has_per_coef_sd else crit_sd_prior)
                sd_t = numpyro.sample(f"sigma_criterion_{crit_group}[{t+1}]", sd_prior_t)
                z_t = numpyro.sample(f"z_criterion_{crit_group}[{t+1}]",
                                     dist.Normal(0, 1).expand([n_groups_crit]))
                u_crit = u_crit.at[t].set(sd_t * z_t)

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
        if knob is not None:
            G = jnp.asarray(knob["G"])
            bound = thresh[G, jnp.asarray(knob["band"])] - c_R[G] - c_K[G]
            slack = numpyro.sample("know_crit_slack", dist.ImproperUniform(dist.constraints.real, (), ()))
            t = jnp.max(bound) + jnp.exp(slack)
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
                n_rkg=n_rkg, counts=counts,
                thresh_mid=thresh_mid_vec, log_gaps=log_gaps_obs,
                gap_link=gap_link, rk_static=np.asarray(config["rk"]), y_static=np.asarray(config["y"]),
                corr=cdp_corr,
            )

    elif family_name == "metadt":
        ones = jnp.ones(N, dtype=_FLOAT)
        m_old = trial_params.get("mratio", ones)
        m_new = m_old if config.get("metad_shared") else trial_params.get("mratio_N", ones)
        log_lik = fused_loglik_metad(y, int(config["K"]), param_values.get("dprime", jnp.zeros(N)), m_old, m_new,
                                     is_old, thresh_mid_vec, log_gaps_obs, gap_link,
                                     q=float(config["is_old_mid"]), relative=bool(config.get("metad_relative")))
        if counts is not None:
            log_lik = log_lik * counts

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
                if crit2_ls:
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
            counts=counts,
            dprime2=trial_params.get("dprime2"),
            bounded=config.get("bounded", False),
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
            counts=counts,
            source=source,
            thresh_mid=thresh_mid_vec,
            log_gaps=log_gaps_obs,
            gap_link=gap_link,
            is_old_static=None if config.get("is_old") is None else np.asarray(config["is_old"], dtype=float),
            sigma_meta=trial_params.get("sigma_meta"),
            meta_noise=config.get("meta_noise"),
        )

    if counts is not None:
        log_lik_unweighted = log_lik / counts
    else:
        log_lik_unweighted = log_lik
    numpyro.deterministic("log_lik", log_lik_unweighted)

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
    from numpyro.infer.util import initialize_model

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
                fixed["know_crit_slack"] = jnp.asarray(0.0)
        st = initialize_model(random.fold_in(base, c), sdt_model,
                              init_strategy=_init_given_else_uniform(fixed, float(init_radius)),
                              model_args=(config,))
        z = dict(st.param_info.z)
        if kc is not None:
            z["know_crit_slack"] = _knob_init_slack(config, knob, z, kc)
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


class FastNUTS(NUTS):
    """NUTS whose model setup comes from one abstract trace (drghmc._model_info) instead
    of numpyro's initialize_model, which runs the model eagerly and compiles every
    primitive separately. The potential is still
    generated from the model arguments at each call, so a float32 offset swap works as
    with NUTS. Starting points are always passed as init_params. With `coords` (NutsCoords)
    it samples in those coordinates. Trees use Stan's U-turn checks across subtree boundaries
    (nuts_turn)."""

    def __init__(self, model=None, coords=None, **kwargs):
        super().__init__(model, **kwargs)
        self._coords = coords

    def _init_state(self, rng_key, model_args, model_kwargs, init_params):
        if self._model is None:
            return super()._init_state(rng_key, model_args, model_kwargs, init_params)
        from numpyro.infer.util import potential_energy, constrain_fn
        from nuts_turn import hmc
        model, c = self._model, self._coords
        if c is None:
            pot_gen = lambda *a, **k: (lambda z: potential_energy(model, a, k, z))
            post_gen = lambda *a, **k: (lambda z: constrain_fn(model, a, k, z, return_deterministic=True))
        else:
            pot_gen = lambda *a, **k: (lambda z: c.potential(a[0], z))
            post_gen = lambda *a, **k: (lambda z: constrain_fn(model, a, k, c.model_z(z), return_deterministic=True))
        if self._init_fn is None:
            self._init_fn, self._sample_fn = hmc(potential_fn_gen=pot_gen, kinetic_fn=self._kinetic_fn,
                                                 algo=self._algo)
        self._potential_fn_gen = pot_gen
        self._postprocess_fn = post_gen
        if init_params is None:
            raise ValueError("FastNUTS needs init_params.")
        return init_params


_MAP_WARMUP = 2500


class NutsCoords:
    """The coordinates JAX NUTS samples in, DR-G-HMC's: every random-effect standard deviation on
    its prior-quantile scale and, with `transport`, the random-effect blocks through the per-group
    transport map, its anchors built by a DR-G-HMC warmup and shared by every chain (the map is
    exact whatever its anchors). NUTS positions keep the model's site names and shapes; model_v
    turns one into DR-G-HMC's flat model-side vector, model_z into the model's own sites. A
    float32 process has the anchors built by a float64 child. `aux` [1, A]: the anchors of a fit
    being resumed; `build`: drghmc._build's result when already at hand."""

    def __init__(self, config, chains, seed, parallel, progress_bar, init_strategy, init_radius,
                 init_z=None, transport=True, aux=None, build=None, announce=True):
        import drghmc as dr
        self.vg, self.unflatten, self.keys, self.off, self.D, self.pp = build or dr._build(sdt_model, config)
        self.tm, self.aux, self.aux_flat = None, None, None
        th0 = dr._initial_positions(sdt_model, config, self.keys, self.off, self.D, chains, seed,
                                    init_strategy, init_radius, init_z, self.unflatten.to_sampled)
        self.start = th0
        tm = None
        if transport:
            ctrl = dr._merge_control(None)
            tm = dr._transport(sdt_model, config, ctrl, self.off, self.unflatten)
        if tm is not None:
            if progress_bar and announce:
                print(f"[NUTS] group transport map on {', '.join(tm.specs)}"
                      f"{'' if aux is not None else '; anchors from a DR-G-HMC warmup'}", flush=True)
            if aux is None and not jax.config.jax_enable_x64:
                aux, self.start = _anchors_f64(config, chains, seed, parallel, progress_bar, init_strategy,
                                               init_radius, init_z)
            elif aux is None:
                vg_s = dr._sampler_vg(sdt_model, config, self.unflatten, tm)
                _, _, tht, aux_c, _ = dr._adapt(sdt_model, config, self.keys, self.off, self.vg, vg_s, tm,
                                                self.D, chains, _MAP_WARMUP, seed, ctrl, parallel, progress_bar,
                                                th0, user_init=init_z is not None, title="Transport map warmup")
                aux = np.asarray(aux_c)[:1]
                self.start = tm.inverse_chains(tm.forward_chains(tht, aux_c), np.repeat(aux, chains, 0))
            self.aux_flat = np.asarray(aux)[:1]
            self.tm = tm.cheap_of(self.aux_flat)
            self.aux = tuple(p[0] for p in tm.parts(self.aux_flat))

    def flat(self, z):
        return jnp.concatenate([jnp.reshape(z[k], (-1,)) for k in self.keys])

    def model_v(self, z):
        v = self.flat(z)
        return v if self.tm is None else self.tm.forward(v, self.aux)[0]

    def model_z(self, z):
        return self.unflatten(self.model_v(z))

    def potential(self, cfg, z):
        from numpyro.infer.util import potential_energy
        v = self.flat(z)
        th, ld = (v, 0.0) if self.tm is None else self.tm.forward(v, self.aux)
        return potential_energy(sdt_model, (cfg,), {}, self.unflatten(th)) - self.unflatten.log_jac(th) + ld

    def init_params(self, chains):
        z = {k: jnp.asarray(np.asarray(self.start)[:, self.off[k][0]].reshape((chains,) + tuple(self.off[k][1])))
             for k in self.keys}
        return z if chains > 1 else {k: v[0] for k, v in z.items()}

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
        env["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={chains if parallel else 1}"
        code = subprocess.call([sys.executable, os.path.join(here, "autorun.py"), "_nuts_anchors", argfile], env=env)
        if code != 0 or not os.path.exists(out):
            raise RuntimeError(f"bayesroc: the float64 process building the transport map's anchors failed (exit code {code})")
        with np.load(out) as d:
            return d["aux"], d["start"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def nuts_init_params(config, chains, seed, init_strategy, init_radius, init_z):
    """Per-chain unconstrained starting points as a site dict (leading chain axis when
    chains > 1), from the same rule DR-G-HMC uses."""
    import drghmc as dr
    shapes, _, _ = dr._model_info(sdt_model, config)
    keys = sorted(shapes)
    off, o = {}, 0
    for k in keys:
        sz = int(np.prod(shapes[k]))
        off[k] = (slice(o, o + sz), tuple(shapes[k]))
        o += sz
    th = dr._initial_positions(sdt_model, config, keys, off, o, chains, seed, init_strategy,
                               init_radius, init_z)
    z = {k: jnp.asarray(th[:, off[k][0]].reshape((chains,) + off[k][1])) for k in keys}
    return z if chains > 1 else {k: v[0] for k, v in z.items()}


def fit_model(config, chains=4, warmup=1000, samples=1000, seed=0,
              target_accept_prob=0.8, max_tree_depth=10, step_size=1.0,
              thinning=1, chain_method="parallel",
              progress_bar=True,
              init_strategy="numeric", init_radius=0.0,
              sampler="nuts", control=None, save_loglik=True,
              keep_re=False, save_state=False, init_values=None):
    """Fit an SDT model using NumPyro NUTS or the DR-G-HMC sampler.

    Args:
        config: dict from R's build_numpyro_config()
        chains: number of MCMC chains
        warmup: warmup iterations (NUTS) / adaptation iterations (drghmc)
        samples: post-warmup samples per chain
        seed: random seed
        target_accept_prob: target acceptance probability (Stan's adapt_delta)
        max_tree_depth: maximum tree depth for NUTS (default 10)
        step_size: initial NUTS step size (default 1.0)
        progress_bar: whether to show NumPyro's tqdm progress bar (default True)
        init_strategy: "prior" (NumPyro init_to_sample), "random" or "numeric"
            (init_to_uniform with `init_radius`). Default "numeric" with radius 0,
            i.e. every parameter starts at 0 on the unconstrained scale, as in R.
        init_radius: radius for init_to_uniform when init_strategy != "prior".
        init_values: optional per-chain list of {Stan name: value} dicts; used
            instead of init_strategy (see init_values_to_z).
        save_state: also return the end-of-run sampler state (fit_broc's `resume`).
        sampler: "nuts" (default) or "drghmc".
        control: dict of drghmc tuning knobs (ignored for nuts).

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
    rng_key = random.PRNGKey(seed)

    coords = NutsCoords(config, chains, seed, chain_method == "parallel" and chains > 1, progress_bar,
                        init_strategy, init_radius, init_z, transport=bool((control or {}).get("transport", True)))
    init_params = coords.init_params(chains)
    kernel = FastNUTS(
        sdt_model,
        coords=coords,
        target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth,
        step_size=float(step_size),
    )

    mcmc = MCMC(
        kernel,
        num_warmup=warmup,
        num_samples=samples,
        num_chains=chains,
        thinning=int(thinning),
        chain_method=chain_method,
        progress_bar=bool(progress_bar),
    )

    f32 = not jax.config.jax_enable_x64
    offset = None
    t0 = time.time()
    par = chain_method == "parallel" and chains > 1
    if f32:
        with progress.nuts(chains, par, desc="NUTS warmup"):
            mcmc.warmup(rng_key, config, init_params=init_params, collect_warmup=False)
        rng_key = mcmc.post_warmup_state.rng_key
    if f32:
        offset = _nuts_ll_offset(config, mcmc.post_warmup_state, coords)
        config = {**config, "_ll_offset": jnp.asarray(offset)}
        mcmc.post_warmup_state = _refresh_potential(config, mcmc.post_warmup_state, coords)

    with progress.nuts(chains, par, desc="NUTS sampling" if f32 else None, split=None if f32 else int(warmup)):
        mcmc.run(rng_key, config, init_params=None if f32 else init_params,
                 extra_fields=("num_steps", "energy", "accept_prob", "diverging"))
    elapsed = time.time() - t0

    samples_dict = mcmc.get_samples(group_by_chain=True)

    extra = mcmc.get_extra_fields(group_by_chain=True)
    extra_fields = {}
    for field in ("num_steps", "energy", "accept_prob", "diverging"):
        if field in extra:
            extra_fields[field] = np.array(extra[field])

    samples_np = {}
    for k, v in samples_dict.items():
        if k.startswith("z_") and not keep_re:
            continue
        if k == "log_lik" and not save_loglik:
            continue
        samples_np[k] = np.array(v)
    _restore_centred(samples_np)

    out = {
        "samples": samples_np,
        "extra_fields": extra_fields,
        "elapsed": float(elapsed),
    }
    if save_state:
        out["state"] = {"hmc": mcmc.last_state, "offset": None if offset is None else np.asarray(offset),
                        "aux": coords.aux_flat}
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
