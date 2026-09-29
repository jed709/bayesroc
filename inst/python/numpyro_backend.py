"""
NumPyro backend for the bayesroc SDT package.

Config-driven model: receives a config dict from R, builds the NumPyro model
dynamically. NOT code-generated.

Entry point: fit_model(config, chains, warmup, samples, seed)
"""

import os
import pickle
import sys
import time
import re as _re
from functools import partial

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.stats import norm as _jax_norm_orig
from jax.scipy.special import expit as jax_expit

jax.config.update("jax_enable_x64", os.environ.get("BAYESROC_ENABLE_X64", "1") != "0")
if os.environ.get("BAYESROC_ENABLE_X64", "1") == "0":
    print("[bayesroc] float32 mode (x64 disabled)", flush=True)

_FLOAT = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32

_cache = os.environ.get("BAYESROC_XLA_CACHE", "")
if _cache != "0":
    try:
        if not _cache:
            _cache = os.path.join(os.path.expanduser("~"), ".cache", "bayesroc_xla")
        os.makedirs(_cache, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", _cache)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    except Exception:
        pass


_PHI_D = 0.3989422804014327

def _poly_phi(x):
    """Fast polynomial approximation of the standard normal CDF.
    Max absolute error: 7.5e-8, sufficient for MCMC.
    """
    ax = jnp.abs(x)
    t = 1.0 / (1.0 + 0.2316419 * ax)
    p = _PHI_D * jnp.exp(-0.5 * x * x) * (
        t * (0.319381530 + t * (-0.356563782 + t * (
            1.781477937 + t * (-1.821255978 + t * 1.330274429)))))
    return jnp.where(x >= 0, 1.0 - p, p)

def _phi_and_Phi(x):
    """Compute standard normal PDF and CDF in a single pass, sharing the
    expensive exp(-0.5 x^2) evaluation. ~30% faster than calling _phi and
    _poly_phi separately, especially on Apple Silicon where exp is slow."""
    pdf = jnp.exp(-0.5 * x * x) * _PHI_D
    ax = jnp.abs(x)
    t = 1.0 / (1.0 + 0.2316419 * ax)
    p = pdf * (t * (0.319381530 + t * (-0.356563782 + t *
              (1.781477937 + t * (-1.821255978 + t * 1.330274429)))))
    Phi = jnp.where(x >= 0, 1.0 - p, p)
    return pdf, Phi


class _JaxNormWrapper:
    """Drop-in replacement for jax.scipy.stats.norm with polynomial cdf."""
    @staticmethod
    def cdf(x):
        return _poly_phi(x)
    @staticmethod
    def pdf(x):
        return _jax_norm_orig.pdf(x)
    @staticmethod
    def logpdf(x):
        return _jax_norm_orig.logpdf(x)

jax_norm = _JaxNormWrapper()

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_uniform, init_to_sample


# =============================================================================
# Detailed progress bar patch for multi-chain runs
# =============================================================================

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _script_dir)
try:
    from owens_t_jax import owens_t, binormal_cdf
except ImportError:
    owens_t = None
    binormal_cdf = None


# =============================================================================
# Prior Parsing
# =============================================================================

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

_CLOGLOG_LO, _CLOGLOG_HI = -30.0, 3.5


def _cloglog_cdf(x):
    """Gumbel-min (smallest extreme value) CDF: F(x) = 1 - exp(-exp(x))."""
    return -jnp.expm1(-jnp.exp(jnp.clip(x, _CLOGLOG_LO, _CLOGLOG_HI)))


def _cloglog_pdf(x):
    """Gumbel-min density: f(x) = exp(x - exp(x))."""
    xc = jnp.clip(x, _CLOGLOG_LO, _CLOGLOG_HI)
    return jnp.exp(xc - jnp.exp(xc))


def _link_fns(link_name):
    """(CDF, PDF) pair for a latent-distribution link."""
    if link_name == "logit":
        return jax_expit, lambda x: jax_expit(x) * (1.0 - jax_expit(x))
    if link_name == "cloglog":
        return _cloglog_cdf, _cloglog_pdf
    return _Phi, _phi


def link_cdf(x, link_name):
    """CDF for the link function (probit, logit, or cloglog)."""
    if link_name == "probit":
        return jax_norm.cdf(x)
    elif link_name == "logit":
        return jax_expit(x)
    elif link_name == "cloglog":
        return _cloglog_cdf(x)
    else:
        raise ValueError(f"Unknown link: {link_name}")


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

# --- GL10 quadrature arrays for bivariate rectangle probability ---

_GL10_X = jnp.array([
    -0.9739065285171717, -0.8650633666889845, -0.6794095682990244,
    -0.4333953941292472, -0.1488743389816312,
     0.1488743389816312,  0.4333953941292472,  0.6794095682990244,
     0.8650633666889845,  0.9739065285171717,
])
_GL10_W = jnp.array([
    0.0666713443086881, 0.1494513491505806, 0.2190863625159820,
    0.2692667193099963, 0.2955242247147529,
    0.2955242247147529, 0.2692667193099963, 0.2190863625159820,
    0.1494513491505806, 0.0666713443086881,
])

# --- Bivariate rectangle probability ---

@custom_jvp
def _rect_cell_prob(z1_lo, z1_hi, z2_lo, z2_hi, rho):
    """P(z1_lo < Z1 < z1_hi, z2_lo < Z2 < z2_hi) via GL10 quadrature."""
    a = jnp.clip(z1_lo, -8.0, 8.0)
    b = jnp.clip(z1_hi, -8.0, 8.0)
    sigma_c = jnp.sqrt(jnp.clip(1.0 - rho**2, 1e-20))
    mid = (b + a) / 2.0
    half = (b - a) / 2.0
    x = mid + half * _GL10_X
    phi_x = jnp.exp(-x**2 / 2.0) / jnp.sqrt(2.0 * jnp.pi)
    arg_hi = (z2_hi - rho * x) / sigma_c
    arg_lo = (z2_lo - rho * x) / sigma_c
    phi_diff = jax_norm.cdf(arg_hi) - jax_norm.cdf(arg_lo)
    return jnp.clip(half * jnp.sum(_GL10_W * phi_x * phi_diff), 1e-20)

@_rect_cell_prob.defjvp
def _rect_cell_prob_jvp(primals, tangents):
    z1_lo, z1_hi, z2_lo, z2_hi, rho = primals
    dz1_lo, dz1_hi, dz2_lo, dz2_hi, drho = tangents
    val = _rect_cell_prob(z1_lo, z1_hi, z2_lo, z2_hi, rho)
    sigma_c = jnp.sqrt(jnp.clip(1.0 - rho**2, 1e-20))
    _phi = lambda x: jnp.exp(-0.5 * x**2) / jnp.sqrt(2.0 * jnp.pi)
    _Phi = jax_norm.cdf
    dp_dz1_hi = _phi(z1_hi) * (_Phi((z2_hi - rho*z1_hi)/sigma_c) - _Phi((z2_lo - rho*z1_hi)/sigma_c))
    dp_dz1_lo = -_phi(z1_lo) * (_Phi((z2_hi - rho*z1_lo)/sigma_c) - _Phi((z2_lo - rho*z1_lo)/sigma_c))
    dp_dz2_hi = _phi(z2_hi) * (_Phi((z1_hi - rho*z2_hi)/sigma_c) - _Phi((z1_lo - rho*z2_hi)/sigma_c))
    dp_dz2_lo = -_phi(z2_lo) * (_Phi((z1_hi - rho*z2_lo)/sigma_c) - _Phi((z1_lo - rho*z2_lo)/sigma_c))
    def _bvn_density(x, y):
        return jnp.exp(-(x**2 - 2*rho*x*y + y**2) / (2.0*(1.0 - rho**2))) / (2.0 * jnp.pi * sigma_c)
    dp_drho = (_bvn_density(z1_hi, z2_hi) - _bvn_density(z1_hi, z2_lo)
             - _bvn_density(z1_lo, z2_hi) + _bvn_density(z1_lo, z2_lo))
    tangent_out = (dp_dz1_lo * dz1_lo + dp_dz1_hi * dz1_hi
                 + dp_dz2_lo * dz2_lo + dp_dz2_hi * dz2_hi
                 + dp_drho * drho)
    return val, tangent_out

# --- Bivariate strip probability (for CDP) ---

@custom_jvp
def _strip_prob_upper(z_crit, z_lo, z_hi, rho):
    """P(Z1 > z_crit, z_lo < Z2 < z_hi) — Owen's T forward, analytic JVP."""
    p_hi = jax_norm.cdf(z_hi) - binormal_cdf(z_crit, z_hi, rho)
    p_lo = jax_norm.cdf(z_lo) - binormal_cdf(z_crit, z_lo, rho)
    return jnp.clip(p_hi - p_lo, 1e-20)

@_strip_prob_upper.defjvp
def _strip_prob_upper_jvp(primals, tangents):
    z_crit, z_lo, z_hi, rho = primals
    dz_crit, dz_lo, dz_hi, drho = tangents
    val = _strip_prob_upper(z_crit, z_lo, z_hi, rho)
    sigma_c = jnp.sqrt(jnp.clip(1.0 - rho**2, 1e-20))
    _phi = lambda x: jnp.exp(-0.5 * x**2) / jnp.sqrt(2.0 * jnp.pi)
    _Phi = jax_norm.cdf
    dp_dcrit = _phi(z_crit) * (_Phi((z_lo - rho*z_crit)/sigma_c) - _Phi((z_hi - rho*z_crit)/sigma_c))
    dp_dz_hi = _phi(z_hi) * (1.0 - _Phi((z_crit - rho*z_hi)/sigma_c))
    dp_dz_lo = -_phi(z_lo) * (1.0 - _Phi((z_crit - rho*z_lo)/sigma_c))
    def _bvn_density(x, y):
        return jnp.exp(-(x**2 - 2*rho*x*y + y**2) / (2.0*(1.0 - rho**2))) / (2.0 * jnp.pi * sigma_c)
    dp_drho = -_bvn_density(z_crit, z_hi) + _bvn_density(z_crit, z_lo)
    tangent_out = dp_dcrit * dz_crit + dp_dz_lo * dz_lo + dp_dz_hi * dz_hi + dp_drho * drho
    tangent_out = jnp.where(val > 1e-6, tangent_out, 0.0)
    return val, tangent_out

@custom_jvp
def _strip_prob_lower(z_crit, z_lo, z_hi, rho):
    """P(Z1 < z_crit, z_lo < Z2 < z_hi) — Owen's T forward, analytic JVP."""
    p_hi = binormal_cdf(z_crit, z_hi, rho)
    p_lo = binormal_cdf(z_crit, z_lo, rho)
    return jnp.clip(p_hi - p_lo, 1e-20)

@_strip_prob_lower.defjvp
def _strip_prob_lower_jvp(primals, tangents):
    z_crit, z_lo, z_hi, rho = primals
    dz_crit, dz_lo, dz_hi, drho = tangents
    val = _strip_prob_lower(z_crit, z_lo, z_hi, rho)
    sigma_c = jnp.sqrt(jnp.clip(1.0 - rho**2, 1e-20))
    _phi = lambda x: jnp.exp(-0.5 * x**2) / jnp.sqrt(2.0 * jnp.pi)
    _Phi = jax_norm.cdf
    dp_dcrit = _phi(z_crit) * (_Phi((z_hi - rho*z_crit)/sigma_c) - _Phi((z_lo - rho*z_crit)/sigma_c))
    dp_dz_hi = _phi(z_hi) * _Phi((z_crit - rho*z_hi)/sigma_c)
    dp_dz_lo = -_phi(z_lo) * _Phi((z_crit - rho*z_lo)/sigma_c)
    def _bvn_density(x, y):
        return jnp.exp(-(x**2 - 2*rho*x*y + y**2) / (2.0*(1.0 - rho**2))) / (2.0 * jnp.pi * sigma_c)
    dp_drho = _bvn_density(z_crit, z_hi) - _bvn_density(z_crit, z_lo)
    tangent_out = dp_dcrit * dz_crit + dp_dz_lo * dz_lo + dp_dz_hi * dz_hi + dp_drho * drho
    tangent_out = jnp.where(val > 1e-6, tangent_out, 0.0)
    return val, tangent_out


# =============================================================================
# Univariate Log-Likelihood
# =============================================================================

# =============================================================================
# =============================================================================

def _phi(x):
    """Standard normal PDF (double precision)."""
    return jnp.exp(-0.5 * x**2) / jnp.sqrt(2.0 * jnp.pi)

def _Phi(x):
    """Standard normal CDF — polynomial approximation (max err 7.5e-8).
    Much faster than erfc-based Phi on CPU (no erf hardware call).
    """
    return _poly_phi(x)


# =============================================================================
# Fully-vectorized batch loglik (no vmap)
# =============================================================================

def _make_fused_batch_loglik(K, gap_link_is_softplus, link_name="probit"):
    """Fully-vectorized batch log-likelihood with custom JVP.

    Operates on entire [N]-shaped arrays directly (no per-observation vmap).
    Thresholds are built via cumsum, and
    gap gradients use a precomputed contribution mask instead of a Python loop.
    """
    n_thresh = K - 1
    n_gaps = n_thresh - 1
    mid_idx = (n_thresh - 1) // 2
    n_upper = n_thresh - mid_idx - 1

    _CDF, _PDF = _link_fns(link_name)

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

        INF = 20.0
        z_hi = jnp.where(y_idx < K - 1, (thresh_hi - mu) / sigma_val, INF)
        z_lo = jnp.where(y_idx > 0, (thresh_lo - mu) / sigma_val, -INF)

        p = jnp.clip(_CDF(z_hi) - _CDF(z_lo), 1e-20)
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

        INF = 20.0
        z_hi = jnp.where(y_idx < K - 1, (thresh_hi - mu) / sigma_val, INF)
        z_lo = jnp.where(y_idx > 0, (thresh_lo - mu) / sigma_val, -INF)

        p = jnp.clip(_CDF(z_hi) - _CDF(z_lo), 1e-20)
        val = jnp.log(p)

        # --- Analytic gradients ---
        pdf_hi = _PDF(z_hi)
        pdf_lo = _PDF(z_lo)
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


def _make_fused_batch_dpsdt_loglik(K, gap_link_is_softplus, link_name="probit"):
    """Fully-vectorized batch log-likelihood for DPSDT with custom JVP.

    DPSDT: P = lambda * I(y==K) + (1-lambda) * cell_p.
    Operates on entire [N]-shaped arrays directly.
    """
    n_thresh = K - 1
    n_gaps = n_thresh - 1
    mid_idx = (n_thresh - 1) // 2
    n_upper = n_thresh - mid_idx - 1

    _CDF, _PDF = _link_fns(link_name)

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

        INF = 20.0
        z_hi = jnp.where(y_idx < K - 1, (thresh_hi - mu) / sigma_val, INF)
        z_lo = jnp.where(y_idx > 0, (thresh_lo - mu) / sigma_val, -INF)
        cell_p = jnp.clip(_CDF(z_hi) - _CDF(z_lo), 1e-20)

        is_top = (y == K).astype(jnp.float64)
        p = jnp.clip(lambda_val * is_top + (1.0 - lambda_val) * cell_p, 1e-20)
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

        INF = 20.0
        z_hi = jnp.where(y_idx < K - 1, (thresh_hi - mu) / sigma_val, INF)
        z_lo = jnp.where(y_idx > 0, (thresh_lo - mu) / sigma_val, -INF)
        cell_p = jnp.clip(_CDF(z_hi) - _CDF(z_lo), 1e-20)

        is_top = (y == K).astype(jnp.float64)
        p = jnp.clip(lambda_val * is_top + (1.0 - lambda_val) * cell_p, 1e-20)
        val = jnp.log(p)

        d_lp_d_lambda = (is_top - cell_p) / p

        d_lp_d_cell = (1.0 - lambda_val) / p

        pdf_hi = _PDF(z_hi)
        pdf_lo = _PDF(z_lo)
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

    _CDF, _PDF = _link_fns(link_name)

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

        INF = 20.0
        z1_hi = jnp.where(y_idx < K - 1, (thresh_hi - mu1) / sigma1_val, INF)
        z1_lo = jnp.where(y_idx > 0, (thresh_lo - mu1) / sigma1_val, -INF)
        p1 = jnp.clip(_CDF(z1_hi) - _CDF(z1_lo), 1e-20)
        z2_hi = jnp.where(y_idx < K - 1, (thresh_hi - mu2) / sigma2_val, INF)
        z2_lo = jnp.where(y_idx > 0, (thresh_lo - mu2) / sigma2_val, -INF)
        p2 = jnp.clip(_CDF(z2_hi) - _CDF(z2_lo), 1e-20)

        p = jnp.clip(lambda_val * p1 + (1.0 - lambda_val) * p2, 1e-20)
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

        INF = 20.0

        z1_hi = jnp.where(y_idx < K - 1, (thresh_hi - mu1) / sigma1_val, INF)
        z1_lo = jnp.where(y_idx > 0, (thresh_lo - mu1) / sigma1_val, -INF)
        p1 = jnp.clip(_CDF(z1_hi) - _CDF(z1_lo), 1e-20)

        z2_hi = jnp.where(y_idx < K - 1, (thresh_hi - mu2) / sigma2_val, INF)
        z2_lo = jnp.where(y_idx > 0, (thresh_lo - mu2) / sigma2_val, -INF)
        p2 = jnp.clip(_CDF(z2_hi) - _CDF(z2_lo), 1e-20)

        p = jnp.clip(lambda_val * p1 + (1.0 - lambda_val) * p2, 1e-20)
        val = jnp.log(p)

        d_lp_d_lambda = (p1 - p2) / p

        d_lp_d_p1 = lambda_val / p
        d_lp_d_p2 = (1.0 - lambda_val) / p

        pdf1_hi = _PDF(z1_hi)
        pdf1_lo = _PDF(z1_lo)
        dp1_dz1_hi = jnp.where(y_idx < K - 1, pdf1_hi, 0.0)
        dp1_dz1_lo = jnp.where(y_idx > 0, -pdf1_lo, 0.0)

        pdf2_hi = _PDF(z2_hi)
        pdf2_lo = _PDF(z2_lo)
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

    _CDF, _PDF = _link_fns(link_name)

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

        INF = 20.0
        z_att_hi = jnp.where(y_idx < K - 1, thresh_hi - d_attended, INF)
        z_att_lo = jnp.where(y_idx > 0, thresh_lo - d_attended, -INF)
        p_att = jnp.clip(_CDF(z_att_hi) - _CDF(z_att_lo), 1e-20)

        z_noise_hi = jnp.where(y_idx < K - 1, thresh_hi, INF)
        z_noise_lo = jnp.where(y_idx > 0, thresh_lo, -INF)
        p_noise = jnp.clip(_CDF(z_noise_hi) - _CDF(z_noise_lo), 1e-20)

        p = jnp.clip(lambda_val * p_att + (1.0 - lambda_val) * p_noise, 1e-20)
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

        INF = 20.0

        z_att_hi = jnp.where(y_idx < K - 1, thresh_hi - d_attended, INF)
        z_att_lo = jnp.where(y_idx > 0, thresh_lo - d_attended, -INF)
        p_att = jnp.clip(_CDF(z_att_hi) - _CDF(z_att_lo), 1e-20)

        z_noise_hi = jnp.where(y_idx < K - 1, thresh_hi, INF)
        z_noise_lo = jnp.where(y_idx > 0, thresh_lo, -INF)
        p_noise = jnp.clip(_CDF(z_noise_hi) - _CDF(z_noise_lo), 1e-20)

        p = jnp.clip(lambda_val * p_att + (1.0 - lambda_val) * p_noise, 1e-20)
        val = jnp.log(p)

        d_lp_d_lambda = (p_att - p_noise) / p

        d_lp_d_patt = lambda_val / p
        d_lp_d_pnoise = (1.0 - lambda_val) / p

        pdf_att_hi = _PDF(z_att_hi)
        pdf_att_lo = _PDF(z_att_lo)
        dpatt_dz_hi = jnp.where(y_idx < K - 1, pdf_att_hi, 0.0)
        dpatt_dz_lo = jnp.where(y_idx > 0, -pdf_att_lo, 0.0)

        pdf_noise_hi = _PDF(z_noise_hi)
        pdf_noise_lo = _PDF(z_noise_lo)
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

def _batch_rect_cell_prob(z1_lo, z1_hi, z2_lo, z2_hi, rho):
    """Batch GL10 rectangle probability for [N]-shaped arrays.
    P(z1_lo < Z1 < z1_hi, z2_lo < Z2 < z2_hi | rho)
    Replaces _rect_cell_prob (which works per-obs only) for batch code."""
    a = jnp.clip(z1_lo, -8.0, 8.0)
    b = jnp.clip(z1_hi, -8.0, 8.0)
    sigma_c = jnp.sqrt(jnp.clip(1.0 - rho**2, 1e-20))
    mid_ab = (b + a) / 2.0
    half_ab = (b - a) / 2.0
    x_pts = mid_ab[:, None] + half_ab[:, None] * _GL10_X[None, :]
    phi_x = jnp.exp(-x_pts**2 / 2.0) / jnp.sqrt(2.0 * jnp.pi)
    arg_hi = (z2_hi[:, None] - rho[:, None] * x_pts) / sigma_c[:, None]
    arg_lo = (z2_lo[:, None] - rho[:, None] * x_pts) / sigma_c[:, None]
    phi_diff = _Phi(arg_hi) - _Phi(arg_lo)
    return jnp.clip(half_ab * jnp.sum(_GL10_W[None, :] * phi_x * phi_diff, axis=1), 1e-20)


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


def _batch_gap_grads(d_thresh_k_hi, d_thresh_k_lo, k_hi, k_lo,
                     has_hi, has_lo, gap_derivs, gap_mask):
    """Compute [N, n_gaps] gap gradients via precomputed mask.
    gap_mask can be numpy or jnp — indexing with JAX tracers requires jnp."""
    gm = jnp.asarray(gap_mask)
    mask_hi = gm[:, k_hi]
    mask_lo = gm[:, k_lo]
    d_lp_d_gap_val = mask_hi * d_thresh_k_hi[None, :] + mask_lo * d_thresh_k_lo[None, :]
    return (d_lp_d_gap_val * gap_derivs.T).T


def _make_fused_batch_bivariate_sdt_loglik(K1, K2, gap_link_is_softplus):
    """Fully-vectorized batch log-likelihood for bivariate_sdt.

    Replaces the vmapped per-obs kernel with batch operations:
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

        INF = 20.0
        z1_hi = jnp.where(r1 < K1 - 1, (c1[obs_idx, k1_hi] - mu1) / sigma1, INF)
        z1_lo = jnp.where(r1 > 0, (c1[obs_idx, k1_lo] - mu1) / sigma1, -INF)
        z2_hi = jnp.where(r2 < K2 - 1, (c2[obs_idx, k2_hi] - mu2) / sigma2, INF)
        z2_lo = jnp.where(r2 > 0, (c2[obs_idx, k2_lo] - mu2) / sigma2, -INF)

        a = jnp.clip(z1_lo, -8.0, 8.0)
        b = jnp.clip(z1_hi, -8.0, 8.0)
        sigma_c = jnp.sqrt(jnp.clip(1.0 - rho_val**2, 1e-20))
        mid_ab = (b + a) / 2.0
        half_ab = (b - a) / 2.0
        x_pts = mid_ab[:, None] + half_ab[:, None] * _GL10_X[None, :]
        phi_x = jnp.exp(-x_pts**2 / 2.0) / jnp.sqrt(2.0 * jnp.pi)
        arg_hi = (z2_hi[:, None] - rho_val[:, None] * x_pts) / sigma_c[:, None]
        arg_lo = (z2_lo[:, None] - rho_val[:, None] * x_pts) / sigma_c[:, None]
        phi_diff = _Phi(arg_hi) - _Phi(arg_lo)
        prob = jnp.clip(half_ab * jnp.sum(_GL10_W[None, :] * phi_x * phi_diff, axis=1), 1e-20)

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

        INF = 20.0
        z1_hi = jnp.where(has1_hi, (c1[obs_idx, k1_hi] - mu1) / sigma1, INF)
        z1_lo = jnp.where(has1_lo, (c1[obs_idx, k1_lo] - mu1) / sigma1, -INF)
        z2_hi = jnp.where(has2_hi, (c2[obs_idx, k2_hi] - mu2) / sigma2, INF)
        z2_lo = jnp.where(has2_lo, (c2[obs_idx, k2_lo] - mu2) / sigma2, -INF)

        a = jnp.clip(z1_lo, -8.0, 8.0)
        b = jnp.clip(z1_hi, -8.0, 8.0)
        sigma_c = jnp.sqrt(jnp.clip(1.0 - rho_val**2, 1e-20))
        mid_ab = (b + a) / 2.0
        half_ab = (b - a) / 2.0
        x_pts = mid_ab[:, None] + half_ab[:, None] * _GL10_X[None, :]
        phi_x = jnp.exp(-x_pts**2 / 2.0) / jnp.sqrt(2.0 * jnp.pi)
        arg_hi = (z2_hi[:, None] - rho_val[:, None] * x_pts) / sigma_c[:, None]
        arg_lo = (z2_lo[:, None] - rho_val[:, None] * x_pts) / sigma_c[:, None]
        phi_diff = _Phi(arg_hi) - _Phi(arg_lo)
        prob = jnp.clip(half_ab * jnp.sum(_GL10_W[None, :] * phi_x * phi_diff, axis=1), 1e-20)
        val = jnp.log(prob)

        # --- Analytic gradients of rectangle probability ---
        dp_dz1_hi = _phi(z1_hi) * (_Phi((z2_hi - rho_val * z1_hi) / sigma_c) -
                                     _Phi((z2_lo - rho_val * z1_hi) / sigma_c))
        dp_dz1_lo = -_phi(z1_lo) * (_Phi((z2_hi - rho_val * z1_lo) / sigma_c) -
                                      _Phi((z2_lo - rho_val * z1_lo) / sigma_c))
        dp_dz2_hi = _phi(z2_hi) * (_Phi((z1_hi - rho_val * z2_hi) / sigma_c) -
                                     _Phi((z1_lo - rho_val * z2_hi) / sigma_c))
        dp_dz2_lo = -_phi(z2_lo) * (_Phi((z1_hi - rho_val * z2_lo) / sigma_c) -
                                      _Phi((z1_lo - rho_val * z2_lo) / sigma_c))

        def _bvn_density(x, y):
            return jnp.exp(-(x**2 - 2.0 * rho_val * x * y + y**2) /
                           (2.0 * (1.0 - rho_val**2))) / (2.0 * jnp.pi * sigma_c)
        dp_drho = (_bvn_density(z1_hi, z2_hi) - _bvn_density(z1_hi, z2_lo)
                   - _bvn_density(z1_lo, z2_hi) + _bvn_density(z1_lo, z2_lo))

        inv_P = 1.0 / prob

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

        INF = 20.0
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

        p_det_below = jnp.clip(_Phi(z_cp_eff) - _Phi(z1_lo), 0.0)
        p_src_below = jnp.clip(_Phi(zs_hi) - _Phi(zs_lo), 0.0)
        p_below = p_det_below * p_src_below

        sigma_c = jnp.sqrt(jnp.clip(1.0 - rho_val**2, 1e-20))
        a_above = jnp.clip(z_cp_eff, -8.0, 8.0)
        b_above = jnp.clip(z1_hi, -8.0, 8.0)
        mid_ab = (b_above + a_above) / 2.0
        half_ab = (b_above - a_above) / 2.0
        x_pts = mid_ab[:, None] + half_ab[:, None] * _GL10_X[None, :]
        phi_x = jnp.exp(-x_pts**2 / 2.0) / jnp.sqrt(2.0 * jnp.pi)
        arg_hi = (z2_hi[:, None] - rho_val[:, None] * x_pts) / sigma_c[:, None]
        arg_lo = (z2_lo[:, None] - rho_val[:, None] * x_pts) / sigma_c[:, None]
        phi_diff = _Phi(arg_hi) - _Phi(arg_lo)
        p_above = jnp.clip(half_ab * jnp.sum(_GL10_W[None, :] * phi_x * phi_diff, axis=1), 0.0)

        P = jnp.clip(p_below + p_above, 1e-20)
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
        INF = 20.0

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
        is_split_f = is_split.astype(jnp.float64)
        is_above_only_f = is_above_only.astype(jnp.float64)
        is_below_only_f = is_below_only.astype(jnp.float64)

        zs_hi = jnp.where(has2_hi, c2[obs_idx, k2_hi] * inv_sc, INF)
        zs_lo = jnp.where(has2_lo, c2[obs_idx, k2_lo] * inv_sc, -INF)

        Phi_zcp_eff = _Phi(z_cp_eff)
        Phi_z1_lo = _Phi(z1_lo)
        p_det_below = jnp.clip(Phi_zcp_eff - Phi_z1_lo, 0.0)
        Phi_zs_hi = _Phi(zs_hi)
        Phi_zs_lo = _Phi(zs_lo)
        p_src_below = jnp.clip(Phi_zs_hi - Phi_zs_lo, 0.0)
        p_below = p_det_below * p_src_below

        sigma_c = jnp.sqrt(jnp.clip(1.0 - rho_val**2, 1e-20))
        a_above = jnp.clip(z_cp_eff, -8.0, 8.0)
        b_above = jnp.clip(z1_hi, -8.0, 8.0)
        mid_ab = (b_above + a_above) / 2.0
        half_ab = (b_above - a_above) / 2.0
        x_pts = mid_ab[:, None] + half_ab[:, None] * _GL10_X[None, :]
        phi_x = jnp.exp(-x_pts**2 / 2.0) / jnp.sqrt(2.0 * jnp.pi)
        arg_hi = (z2_hi[:, None] - rho_val[:, None] * x_pts) / sigma_c[:, None]
        arg_lo = (z2_lo[:, None] - rho_val[:, None] * x_pts) / sigma_c[:, None]
        phi_diff = _Phi(arg_hi) - _Phi(arg_lo)
        p_above = jnp.clip(half_ab * jnp.sum(_GL10_W[None, :] * phi_x * phi_diff, axis=1), 0.0)

        P = jnp.clip(p_below + p_above, 1e-20)
        val = jnp.log(P)
        inv_P = 1.0 / P

        safe = (P > 1e-6).astype(jnp.float64)
        inv_P = inv_P * safe

        # ===============================================================
        # Analytic gradients: dP/d(param) for below-cp and above-cp
        # ===============================================================

        phi_zcp_eff = _phi(z_cp_eff)
        phi_z1_lo = _phi(z1_lo)
        phi_z1_hi = _phi(z1_hi)
        phi_zs_hi = _phi(zs_hi)
        phi_zs_lo = _phi(zs_lo)

        # --- Below-cp gradients (product rule on p_det_below * p_src_below) ---
        dp_below_dzcp_eff = phi_zcp_eff * p_src_below
        dp_below_dz1_lo = jnp.where(has1_lo, -phi_z1_lo * p_src_below, 0.0)
        dp_below_dzs_hi = jnp.where(has2_hi, p_det_below * phi_zs_hi, 0.0)
        dp_below_dzs_lo = jnp.where(has2_lo, -p_det_below * phi_zs_lo, 0.0)

        # --- Above-cp gradients (same analytic forms as standard bivariate) ---
        dp_above_dzcp_eff = -phi_zcp_eff * (
            _Phi((z2_hi - rho_val * z_cp_eff) / sigma_c) -
            _Phi((z2_lo - rho_val * z_cp_eff) / sigma_c))

        dp_above_dz1_hi = phi_z1_hi * (
            _Phi((z2_hi - rho_val * z1_hi) / sigma_c) -
            _Phi((z2_lo - rho_val * z1_hi) / sigma_c))

        dp_above_dz2_hi = _phi(z2_hi) * (
            _Phi((z1_hi - rho_val * z2_hi) / sigma_c) -
            _Phi((z_cp_eff - rho_val * z2_hi) / sigma_c))

        dp_above_dz2_lo = -_phi(z2_lo) * (
            _Phi((z1_hi - rho_val * z2_lo) / sigma_c) -
            _Phi((z_cp_eff - rho_val * z2_lo) / sigma_c))

        def _bvn_density(x, y):
            return jnp.exp(-(x**2 - 2.0 * rho_val * x * y + y**2) /
                           (2.0 * (1.0 - rho_val**2))) / (2.0 * jnp.pi * sigma_c)
        dp_above_drho_direct = (_bvn_density(z1_hi, z2_hi) - _bvn_density(z1_hi, z2_lo)
                                - _bvn_density(z_cp_eff, z2_hi) + _bvn_density(z_cp_eff, z2_lo))

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
        rho_factor = rho_val / one_minus_rho2
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

        pass

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

        INF = 20.0
        z1_hi = jnp.where(r1 < K1 - 1, (c1[obs_idx, k1_hi] - mu1) / sigma1, INF)
        z1_lo = jnp.where(r1 > 0, (c1[obs_idx, k1_lo] - mu1) / sigma1, -INF)
        z2_hi = jnp.where(r2 < K2 - 1, (c2[obs_idx, k2_hi] - mu2) / sigma2, INF)
        z2_lo = jnp.where(r2 > 0, (c2[obs_idx, k2_lo] - mu2) / sigma2, -INF)

        a = jnp.clip(z1_lo, -8.0, 8.0)
        b = jnp.clip(z1_hi, -8.0, 8.0)
        sigma_c = jnp.sqrt(jnp.clip(1.0 - rho_val**2, 1e-20))
        mid_ab = (b + a) / 2.0
        half_ab = (b - a) / 2.0
        x_pts = mid_ab[:, None] + half_ab[:, None] * _GL10_X[None, :]
        phi_x = jnp.exp(-x_pts**2 / 2.0) / jnp.sqrt(2.0 * jnp.pi)
        arg_hi = (z2_hi[:, None] - rho_val[:, None] * x_pts) / sigma_c[:, None]
        arg_lo = (z2_lo[:, None] - rho_val[:, None] * x_pts) / sigma_c[:, None]
        phi_diff = _Phi(arg_hi) - _Phi(arg_lo)
        biv_prob = jnp.clip(half_ab * jnp.sum(_GL10_W[None, :] * phi_x * phi_diff, axis=1), 1e-20)

        is_A = (item_type == 2)
        R_I_eff = jnp.where(is_A, R_I, R_I_B)
        R_S_eff = jnp.where(is_A, R_S, R_S_B)

        p1_comp = (1.0 - R_I_eff) * biv_prob

        p2_source = jnp.clip(_Phi(z2_hi) - _Phi(z2_lo), 1e-20)
        is_y1_top = (y1 == K1)
        p2_comp = jnp.where(is_y1_top, R_I_eff * (1.0 - R_S_eff) * p2_source, 0.0)

        is_corner_A = (item_type == 2) & (y1 == K1) & (y2 == 1)
        is_corner_B = (item_type == 3) & (y1 == K1) & (y2 == K2)
        p3_comp = jnp.where(is_corner_A | is_corner_B, R_I_eff * R_S_eff, 0.0)

        prob = jnp.clip(p1_comp + p2_comp + p3_comp, 1e-20)
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
        INF = 20.0
        z1_hi = jnp.where(has1_hi, (c1[obs_idx, k1_hi] - mu1) * inv_s1, INF)
        z1_lo = jnp.where(has1_lo, (c1[obs_idx, k1_lo] - mu1) * inv_s1, -INF)
        z2_hi = jnp.where(has2_hi, (c2[obs_idx, k2_hi] - mu2) * inv_s2, INF)
        z2_lo = jnp.where(has2_lo, (c2[obs_idx, k2_lo] - mu2) * inv_s2, -INF)

        a = jnp.clip(z1_lo, -8.0, 8.0)
        b = jnp.clip(z1_hi, -8.0, 8.0)
        sigma_c = jnp.sqrt(jnp.clip(1.0 - rho_val**2, 1e-20))
        mid_ab = (b + a) / 2.0
        half_ab = (b - a) / 2.0
        x_pts = mid_ab[:, None] + half_ab[:, None] * _GL10_X[None, :]
        phi_x = jnp.exp(-x_pts**2 / 2.0) / jnp.sqrt(2.0 * jnp.pi)
        arg_hi = (z2_hi[:, None] - rho_val[:, None] * x_pts) / sigma_c[:, None]
        arg_lo = (z2_lo[:, None] - rho_val[:, None] * x_pts) / sigma_c[:, None]
        phi_diff = _Phi(arg_hi) - _Phi(arg_lo)
        biv_prob = jnp.clip(half_ab * jnp.sum(_GL10_W[None, :] * phi_x * phi_diff, axis=1), 1e-20)

        is_A = (item_type == 2)
        R_I_eff = jnp.where(is_A, R_I, R_I_B)
        R_S_eff = jnp.where(is_A, R_S, R_S_B)

        p1_comp = (1.0 - R_I_eff) * biv_prob

        p2_source = jnp.clip(_Phi(z2_hi) - _Phi(z2_lo), 1e-20)
        is_y1_top = (y1 == K1)
        p2_comp = jnp.where(is_y1_top, R_I_eff * (1.0 - R_S_eff) * p2_source, 0.0)

        is_corner_A = (item_type == 2) & (y1 == K1) & (y2 == 1)
        is_corner_B = (item_type == 3) & (y1 == K1) & (y2 == K2)
        is_corner = is_corner_A | is_corner_B
        p3_comp = jnp.where(is_corner, R_I_eff * R_S_eff, 0.0)

        prob = jnp.clip(p1_comp + p2_comp + p3_comp, 1e-20)
        val = jnp.log(prob)
        inv_P = 1.0 / prob

        dp_dz1_hi = _phi(z1_hi) * (_Phi((z2_hi - rho_val * z1_hi) / sigma_c) -
                                     _Phi((z2_lo - rho_val * z1_hi) / sigma_c))
        dp_dz1_lo = -_phi(z1_lo) * (_Phi((z2_hi - rho_val * z1_lo) / sigma_c) -
                                      _Phi((z2_lo - rho_val * z1_lo) / sigma_c))
        dp_dz2_hi = _phi(z2_hi) * (_Phi((z1_hi - rho_val * z2_hi) / sigma_c) -
                                     _Phi((z1_lo - rho_val * z2_hi) / sigma_c))
        dp_dz2_lo = -_phi(z2_lo) * (_Phi((z1_hi - rho_val * z2_lo) / sigma_c) -
                                      _Phi((z1_lo - rho_val * z2_lo) / sigma_c))
        def _bvn_density(x, y):
            return jnp.exp(-(x**2 - 2.0 * rho_val * x * y + y**2) /
                           (2.0 * (1.0 - rho_val**2))) / (2.0 * jnp.pi * sigma_c)
        dp_biv_drho = (_bvn_density(z1_hi, z2_hi) - _bvn_density(z1_hi, z2_lo)
                       - _bvn_density(z1_lo, z2_hi) + _bvn_density(z1_lo, z2_lo))

        phi2_hi = _phi(z2_hi)
        phi2_lo = _phi(z2_lo)
        dp2s_dz2_hi = jnp.where(has2_hi, phi2_hi, 0.0)
        dp2s_dz2_lo = jnp.where(has2_lo, -phi2_lo, 0.0)

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

        INF = 20.0
        inv_si = 1.0 / sigma_item
        inv_sS = 1.0 / sigma_S
        mu_rec1 = dprime_F + dprime_R

        c1_hi = c1[obs_idx, k1_hi]; c1_lo = c1[obs_idx, k1_lo]
        c2_hi = c2[obs_idx, k2_hi]; c2_lo = c2[obs_idx, k2_lo]
        has1_hi = (r1 < K1 - 1); has1_lo = (r1 > 0)
        has2_hi = (r2 < K2 - 1); has2_lo = (r2 > 0)

        z1_fam_hi = jnp.where(has1_hi, (c1_hi - dprime_F) * inv_si, INF)
        z1_fam_lo = jnp.where(has1_lo, (c1_lo - dprime_F) * inv_si, -INF)
        z2_hi     = jnp.where(has2_hi,  c2_hi,                       INF)
        z2_lo     = jnp.where(has2_lo,  c2_lo,                      -INF)
        z1_rec_hi = jnp.where(has1_hi, (c1_hi - mu_rec1) * inv_si,   INF)
        z1_rec_lo = jnp.where(has1_lo, (c1_lo - mu_rec1) * inv_si,  -INF)
        z2_rec_hi = jnp.where(has2_hi, (c2_hi - source_d) * inv_sS,  INF)
        z2_rec_lo = jnp.where(has2_lo, (c2_lo - source_d) * inv_sS, -INF)

        z_stack = jnp.stack([z1_fam_hi, z1_fam_lo, z2_hi, z2_lo,
                              z1_rec_hi, z1_rec_lo, z2_rec_hi, z2_rec_lo])
        Phi_stack = _Phi(z_stack)
        p1_fam   = jnp.clip(Phi_stack[0] - Phi_stack[1], 1e-20)
        p2_noise = jnp.clip(Phi_stack[2] - Phi_stack[3], 1e-20)
        p1_rec   = jnp.clip(Phi_stack[4] - Phi_stack[5], 1e-20)
        p2_rec   = jnp.clip(Phi_stack[6] - Phi_stack[7], 1e-20)

        prob = jnp.clip((1.0 - lambda_val) * p1_fam * p2_noise +
                         lambda_val * p1_rec * p2_rec, 1e-20)
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

        INF = 20.0
        inv_si = 1.0 / sigma_item
        inv_sS = 1.0 / sigma_S
        mu_rec1 = dprime_F + dprime_R

        c1_hi = c1[obs_idx, k1_hi]; c1_lo = c1[obs_idx, k1_lo]
        c2_hi = c2[obs_idx, k2_hi]; c2_lo = c2[obs_idx, k2_lo]

        z1_fam_hi = jnp.where(has1_hi, (c1_hi - dprime_F) * inv_si,  INF)
        z1_fam_lo = jnp.where(has1_lo, (c1_lo - dprime_F) * inv_si, -INF)
        z2_hi     = jnp.where(has2_hi,  c2_hi,                       INF)
        z2_lo     = jnp.where(has2_lo,  c2_lo,                      -INF)
        z1_rec_hi = jnp.where(has1_hi, (c1_hi - mu_rec1) * inv_si,   INF)
        z1_rec_lo = jnp.where(has1_lo, (c1_lo - mu_rec1) * inv_si,  -INF)
        z2_rec_hi = jnp.where(has2_hi, (c2_hi - source_d) * inv_sS,  INF)
        z2_rec_lo = jnp.where(has2_lo, (c2_lo - source_d) * inv_sS, -INF)

        z_stack = jnp.stack([z1_fam_hi, z1_fam_lo, z2_hi, z2_lo,
                              z1_rec_hi, z1_rec_lo, z2_rec_hi, z2_rec_lo])
        phi_stack, Phi_stack = _phi_and_Phi(z_stack)
        p1_fam   = jnp.clip(Phi_stack[0] - Phi_stack[1], 1e-20)
        p2_noise = jnp.clip(Phi_stack[2] - Phi_stack[3], 1e-20)
        p1_rec   = jnp.clip(Phi_stack[4] - Phi_stack[5], 1e-20)
        p2_rec   = jnp.clip(Phi_stack[6] - Phi_stack[7], 1e-20)

        p_fam = p1_fam * p2_noise
        p_rec = p1_rec * p2_rec
        one_m_lam = 1.0 - lambda_val
        prob = jnp.clip(one_m_lam * p_fam + lambda_val * p_rec, 1e-20)
        val = jnp.log(prob)
        inv_P = 1.0 / prob

        w_fam1 = one_m_lam * p2_noise * inv_si * inv_P
        w_rec1 = lambda_val * p2_rec * inv_si * inv_P

        phi1f_hi = jnp.where(has1_hi, phi_stack[0], 0.0)
        phi1f_lo = jnp.where(has1_lo, phi_stack[1], 0.0)
        phi2n_hi = jnp.where(has2_hi, phi_stack[2], 0.0)
        phi2n_lo = jnp.where(has2_lo, phi_stack[3], 0.0)
        phi1r_hi = jnp.where(has1_hi, phi_stack[4], 0.0)
        phi1r_lo = jnp.where(has1_lo, phi_stack[5], 0.0)
        phi2r_hi = jnp.where(has2_hi, phi_stack[6], 0.0)
        phi2r_lo = jnp.where(has2_lo, phi_stack[7], 0.0)

        d_t1_k_hi = w_fam1 * phi1f_hi + w_rec1 * phi1r_hi
        d_t1_k_lo = -(w_fam1 * phi1f_lo + w_rec1 * phi1r_lo)

        sum_fam1 = phi1f_hi - phi1f_lo
        sum_rec1 = phi1r_hi - phi1r_lo
        d_lp_d_dpF = -(w_fam1 * sum_fam1 + w_rec1 * sum_rec1)
        d_lp_d_dpR = -w_rec1 * sum_rec1
        d_lp_d_si = -(w_fam1 * (phi1f_hi * z1_fam_hi - phi1f_lo * z1_fam_lo) +
                       w_rec1 * (phi1r_hi * z1_rec_hi - phi1r_lo * z1_rec_lo))

        w_fam2 = one_m_lam * p1_fam * inv_P
        w_rec2 = lambda_val * p1_rec * inv_sS * inv_P

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

def _make_fused_batch_cdp_loglik(K, J, gap_link_is_softplus, n_rkg=2):
    """Fully-vectorized batch log-likelihood for CDP.

    CDP uses J tau thresholds and bivariate strip probabilities.
    Supports both n_rkg==2 (R/K) and n_rkg==3 (R/K/G).
    """
    n_thresh = J
    n_gaps = n_thresh - 1
    mid_idx = (n_thresh - 1) // 2
    n_upper = n_thresh - mid_idx - 1
    _KK_CDP = 10000.0

    gap_mask_np = _make_batch_gap_mask_np(n_gaps, n_thresh, mid_idx, n_upper)

    _INV_2PI = 1.0 / (2.0 * np.pi)

    def _bvn_derivs_batch(z1, z2, rho_v):
        """Closed-form derivatives of binormal_cdf w.r.t. z1, z2, rho."""
        rho2 = rho_v * rho_v
        denom = jnp.sqrt(jnp.clip(1.0 - rho2, 1e-10))
        dz1 = _phi(z1) * _Phi((z2 - rho_v * z1) / denom)
        dz2 = _phi(z2) * _Phi((z1 - rho_v * z2) / denom)
        exp_arg = -(z1**2 - 2.0 * rho_v * z1 * z2 + z2**2) / (2.0 * jnp.clip(1.0 - rho2, 1e-10))
        drho = jnp.exp(exp_arg) * _INV_2PI / denom
        return dz1, dz2, drho

    def _softclamp(x):
        return jax.nn.softplus(_KK_CDP * x) / _KK_CDP

    @custom_jvp
    def _batch_fused(y, rk, is_old_val, old_idx, is_old_resp_val,
                     mu_R, sigma_R, mu_F, sigma_F, c_R, c_K,
                     thresh_mid, gaps):
        gap_fn = jax.nn.softplus if gap_link_is_softplus else jnp.exp

        gv = gap_fn(gaps)
        c = _batch_build_thresh(thresh_mid, gv, n_gaps, mid_idx, n_upper)

        N = y.shape[0]
        obs_idx = jnp.arange(N)

        mu_M = mu_R + mu_F
        sigma_M = jnp.sqrt(sigma_R**2 + sigma_F**2)
        rho = sigma_R / jnp.clip(sigma_M, 1e-10)

        tau_lo = c[obs_idx, jnp.clip(old_idx, 0, J - 1)]
        tau_hi = jnp.where(old_idx < J - 1, c[obs_idx, jnp.clip(old_idx + 1, 0, J - 1)], 20.0)
        tau_first = c[obs_idx, jnp.zeros(N, dtype=jnp.int32)]

        z_cR = (c_R - mu_R) / jnp.clip(sigma_R, 1e-10)
        z_lo = (tau_lo - mu_M) / jnp.clip(sigma_M, 1e-10)
        z_hi = (tau_hi - mu_M) / jnp.clip(sigma_M, 1e-10)
        z_tau1 = (tau_first - mu_M) / jnp.clip(sigma_M, 1e-10)

        p_band = _Phi(z_hi) - _Phi(z_lo)

        if n_rkg == 2:
            bv_cR_hi = binormal_cdf(jnp.float64(z_cR), jnp.float64(z_hi), jnp.float64(rho))
            bv_cR_lo = binormal_cdf(jnp.float64(z_cR), jnp.float64(z_lo), jnp.float64(rho))
            p_K_strip = bv_cR_hi - bv_cR_lo
            p_R = jnp.clip(p_band - p_K_strip, 1e-20)
            p_K = jnp.clip(p_K_strip, 1e-20)
            p_rk = jnp.where(rk == 1, p_R, p_K)
        else:
            z_cK_F = (c_K - mu_F) / jnp.clip(sigma_F, 1e-10)
            s_hi = jnp.fmin(c_R, tau_hi - c_K)
            s_lo = jnp.fmin(c_R, tau_lo - c_K)
            case1_hi = (c_R <= tau_hi - c_K)
            case1_lo = (c_R <= tau_lo - c_K)
            z_s_hi = (s_hi - mu_R) / jnp.clip(sigma_R, 1e-10)
            z_s_lo = (s_lo - mu_R) / jnp.clip(sigma_R, 1e-10)

            bv_cR_hi = binormal_cdf(jnp.float64(z_cR), jnp.float64(z_hi), jnp.float64(rho))
            bv_cR_lo = binormal_cdf(jnp.float64(z_cR), jnp.float64(z_lo), jnp.float64(rho))
            bv_s_hi = binormal_cdf(jnp.float64(z_s_hi), jnp.float64(z_hi), jnp.float64(rho))
            bv_s_lo = binormal_cdf(jnp.float64(z_s_lo), jnp.float64(z_lo), jnp.float64(rho))

            p_R_raw = p_band - (bv_cR_hi - bv_cR_lo)

            Phi_zcR = _Phi(z_cR)
            Phi_zcKF = _Phi(z_cK_F)
            G_hi = jnp.where(case1_hi, Phi_zcR * Phi_zcKF,
                             _Phi(z_s_hi) * Phi_zcKF + bv_cR_hi - bv_s_hi)
            G_lo = jnp.where(case1_lo, Phi_zcR * Phi_zcKF,
                             _Phi(z_s_lo) * Phi_zcKF + bv_cR_lo - bv_s_lo)

            p_G_raw = G_hi - G_lo
            p_K_raw = p_band - p_R_raw - p_G_raw

            p_R = _softclamp(p_R_raw)
            p_G = _softclamp(p_G_raw)
            p_K = _softclamp(p_K_raw)
            p_rk = jnp.where(rk == 1, p_R, jnp.where(rk == 2, p_K, p_G))

        p_new = _Phi(z_tau1)
        prob = jnp.where(is_old_resp_val == 1, p_rk, p_new)
        return jnp.log(jnp.clip(prob, 1e-20))

    @_batch_fused.defjvp
    def _batch_fused_jvp(primals, tangents):
        (y, rk, is_old_val, old_idx, is_old_resp_val,
         mu_R, sigma_R, mu_F, sigma_F, c_R, c_K,
         thresh_mid_v, gaps_v) = primals
        (_, _, _, _, _,
         d_mu_R, d_sigma_R, d_mu_F, d_sigma_F, d_c_R, d_c_K,
         d_thresh_mid_v, d_gaps_v) = tangents

        if gap_link_is_softplus:
            gv = jax.nn.softplus(gaps_v); gd = jax.nn.sigmoid(gaps_v)
        else:
            gv = jnp.exp(gaps_v); gd = gv

        c = _batch_build_thresh(thresh_mid_v, gv, n_gaps, mid_idx, n_upper)

        N = y.shape[0]
        obs_idx = jnp.arange(N)

        mu_M = mu_R + mu_F
        sigma_M = jnp.sqrt(sigma_R**2 + sigma_F**2)
        rho = sigma_R / jnp.clip(sigma_M, 1e-10)
        inv_sR = 1.0 / jnp.clip(sigma_R, 1e-10)
        inv_sF = 1.0 / jnp.clip(sigma_F, 1e-10)
        inv_sM = 1.0 / jnp.clip(sigma_M, 1e-10)
        rho2 = rho * rho
        denom_rho = jnp.sqrt(jnp.clip(1.0 - rho2, 1e-10))

        k_tau_lo = jnp.clip(old_idx, 0, n_thresh - 1)
        k_tau_hi = jnp.clip(old_idx + 1, 0, n_thresh - 1)
        has_tau_hi = (old_idx < J - 1)

        tau_lo = c[obs_idx, k_tau_lo]
        tau_hi = jnp.where(has_tau_hi, c[obs_idx, k_tau_hi], 20.0)
        tau_first = c[obs_idx, jnp.zeros(N, dtype=jnp.int32)]

        z_cR = (c_R - mu_R) * inv_sR
        z_hi = (tau_hi - mu_M) * inv_sM
        z_lo = (tau_lo - mu_M) * inv_sM
        z_tau1 = (tau_first - mu_M) * inv_sM

        p_band = _Phi(z_hi) - _Phi(z_lo)

        d_bvCRhi_z1, d_bvCRhi_z2, d_bvCRhi_rho = _bvn_derivs_batch(z_cR, z_hi, rho)
        d_bvCRlo_z1, d_bvCRlo_z2, d_bvCRlo_rho = _bvn_derivs_batch(z_cR, z_lo, rho)

        phi_hi = _phi(z_hi)
        phi_lo = _phi(z_lo)
        phi_tau1 = _phi(z_tau1)
        p_new = _Phi(z_tau1)

        is_old_resp = (is_old_resp_val == 1)
        is_new_resp = ~is_old_resp

        sR_over_sM = sigma_R * inv_sM
        sF_over_sM = sigma_F * inv_sM
        sM3 = sigma_M ** 3
        drho_dsR = sigma_F**2 / jnp.clip(sM3, 1e-20)
        drho_dsF = -sigma_R * sigma_F / jnp.clip(sM3, 1e-20)

        if n_rkg == 2:
            bv_cR_hi = binormal_cdf(jnp.float64(z_cR), jnp.float64(z_hi), jnp.float64(rho))
            bv_cR_lo = binormal_cdf(jnp.float64(z_cR), jnp.float64(z_lo), jnp.float64(rho))
            p_K_strip = bv_cR_hi - bv_cR_lo
            p_R = jnp.clip(p_band - p_K_strip, 1e-20)
            p_K = jnp.clip(p_K_strip, 1e-20)
            p_rk = jnp.where(rk == 1, p_R, p_K)
            prob = jnp.where(is_old_resp, p_rk, p_new)
            val = jnp.log(jnp.clip(prob, 1e-20))
            inv_prob = jnp.where(prob > 1e-15, 1.0 / jnp.clip(prob, 1e-20), 0.0)

            is_R = (rk == 1)

            dlp_dzcR = jnp.where(is_old_resp,
                jnp.where(is_R, -d_bvCRhi_z1 + d_bvCRlo_z1,
                                  d_bvCRhi_z1 - d_bvCRlo_z1) * inv_prob, 0.0)
            dlp_dzhi = jnp.where(is_old_resp,
                jnp.where(is_R, phi_hi - d_bvCRhi_z2,
                                 d_bvCRhi_z2) * inv_prob, 0.0)
            dlp_dzlo = jnp.where(is_old_resp,
                jnp.where(is_R, -(phi_lo - d_bvCRlo_z2),
                                 -d_bvCRlo_z2) * inv_prob, 0.0)
            dlp_drho = jnp.where(is_old_resp,
                jnp.where(is_R, -d_bvCRhi_rho + d_bvCRlo_rho,
                                  d_bvCRhi_rho - d_bvCRlo_rho) * inv_prob, 0.0)
            dlp_dztau1 = jnp.where(is_new_resp, phi_tau1 * inv_prob, 0.0)

            d_val_dmuR = (dlp_dzcR * (-inv_sR) +
                          (dlp_dzhi + dlp_dzlo + dlp_dztau1) * (-inv_sM))
            d_val_dmuF = (dlp_dzhi + dlp_dzlo + dlp_dztau1) * (-inv_sM)
            d_val_dsigR = (dlp_dzcR * (-z_cR * inv_sR) +
                           (dlp_dzhi * (-z_hi) + dlp_dzlo * (-z_lo) +
                            dlp_dztau1 * (-z_tau1)) * inv_sM * sR_over_sM +
                           dlp_drho * drho_dsR)
            d_val_dsigF = ((dlp_dzhi * (-z_hi) + dlp_dzlo * (-z_lo) +
                            dlp_dztau1 * (-z_tau1)) * inv_sM * sF_over_sM +
                           dlp_drho * drho_dsF)
            d_val_dcR = dlp_dzcR * inv_sR
            d_val_dcK = jnp.zeros_like(c_K)

            d_val_dtau_hi = jnp.where(is_old_resp, dlp_dzhi * inv_sM, 0.0)
            d_val_dtau_lo = jnp.where(is_old_resp, dlp_dzlo * inv_sM, 0.0)
            d_val_dtau_first = jnp.where(is_new_resp, dlp_dztau1 * inv_sM, 0.0)

        else:
            p_R_raw = _batch_rect_cell_prob(z_cR, jnp.full_like(z_cR, 8.0), z_lo, z_hi, rho)
            z_cK_F = (c_K - mu_F) * inv_sF
            s_hi = jnp.fmin(c_R, tau_hi - c_K)
            s_lo = jnp.fmin(c_R, tau_lo - c_K)
            case1_hi = (c_R <= tau_hi - c_K)
            case1_lo = (c_R <= tau_lo - c_K)
            z_s_hi = (s_hi - mu_R) * inv_sR
            z_s_lo = (s_lo - mu_R) * inv_sR

            bv_cR_hi = binormal_cdf(jnp.float64(z_cR), jnp.float64(z_hi), jnp.float64(rho))
            bv_cR_lo = binormal_cdf(jnp.float64(z_cR), jnp.float64(z_lo), jnp.float64(rho))
            bv_s_hi = binormal_cdf(jnp.float64(z_s_hi), jnp.float64(z_hi), jnp.float64(rho))
            bv_s_lo = binormal_cdf(jnp.float64(z_s_lo), jnp.float64(z_lo), jnp.float64(rho))

            Phi_zcR = _Phi(z_cR)
            Phi_zcKF = _Phi(z_cK_F)
            Phi_zs_hi = _Phi(z_s_hi)
            Phi_zs_lo = _Phi(z_s_lo)

            G_hi = jnp.where(case1_hi, Phi_zcR * Phi_zcKF,
                             Phi_zs_hi * Phi_zcKF + bv_cR_hi - bv_s_hi)
            G_lo = jnp.where(case1_lo, Phi_zcR * Phi_zcKF,
                             Phi_zs_lo * Phi_zcKF + bv_cR_lo - bv_s_lo)

            p_G_raw = G_hi - G_lo
            p_K_raw = p_band - p_R_raw - p_G_raw

            p_R = _softclamp(p_R_raw)
            p_K = _softclamp(p_K_raw)
            p_G = _softclamp(p_G_raw)

            p_rk = jnp.where(rk == 1, p_R, jnp.where(rk == 2, p_K, p_G))
            prob = jnp.where(is_old_resp, p_rk, p_new)
            val = jnp.log(jnp.clip(prob, 1e-20))
            inv_prob = jnp.where(prob > 1e-15, 1.0 / jnp.clip(prob, 1e-20), 0.0)

            sig_R = jax.nn.sigmoid(_KK_CDP * p_R_raw)
            sig_K = jax.nn.sigmoid(_KK_CDP * p_K_raw)
            sig_G = jax.nn.sigmoid(_KK_CDP * p_G_raw)

            dlp_dpR = jnp.where(is_old_resp & (rk == 1), sig_R * inv_prob, 0.0)
            dlp_dpK = jnp.where(is_old_resp & (rk == 2), sig_K * inv_prob, 0.0)
            dlp_dpG = jnp.where(is_old_resp & (rk == 3), sig_G * inv_prob, 0.0)

            eff_dpR = dlp_dpR - dlp_dpK
            eff_dpG = dlp_dpG - dlp_dpK
            eff_dpBand = dlp_dpK

            d_bvShi_z1, d_bvShi_z2, d_bvShi_rho = _bvn_derivs_batch(z_s_hi, z_hi, rho)
            d_bvSlo_z1, d_bvSlo_z2, d_bvSlo_rho = _bvn_derivs_batch(z_s_lo, z_lo, rho)

            phi_zcR = _phi(z_cR)
            phi_zcKF = _phi(z_cK_F)
            phi_zs_hi = _phi(z_s_hi)
            phi_zs_lo = _phi(z_s_lo)

            dGhi_dtau = jnp.where(case1_hi, 0.0,
                phi_zs_hi * inv_sR * Phi_zcKF + d_bvCRhi_z2 * inv_sM
                - d_bvShi_z1 * inv_sR - d_bvShi_z2 * inv_sM)
            dGhi_dcR = jnp.where(case1_hi,
                phi_zcR * inv_sR * Phi_zcKF, d_bvCRhi_z1 * inv_sR)
            dGhi_dcK = jnp.where(case1_hi,
                Phi_zcR * phi_zcKF * inv_sF,
                phi_zs_hi * (-inv_sR) * Phi_zcKF + Phi_zs_hi * phi_zcKF * inv_sF + d_bvShi_z1 * inv_sR)
            dGhi_dmuR = jnp.where(case1_hi,
                -phi_zcR * inv_sR * Phi_zcKF,
                phi_zs_hi * (-inv_sR) * Phi_zcKF
                + d_bvCRhi_z1 * (-inv_sR) + d_bvCRhi_z2 * (-inv_sM)
                - d_bvShi_z1 * (-inv_sR) - d_bvShi_z2 * (-inv_sM))
            dGhi_dmuF = jnp.where(case1_hi,
                -Phi_zcR * phi_zcKF * inv_sF,
                Phi_zs_hi * phi_zcKF * (-inv_sF)
                + d_bvCRhi_z2 * (-inv_sM) - d_bvShi_z2 * (-inv_sM))
            dGhi_dsigR = jnp.where(case1_hi,
                phi_zcR * (-z_cR * inv_sR) * Phi_zcKF,
                phi_zs_hi * (-z_s_hi * inv_sR) * Phi_zcKF
                + d_bvCRhi_z1 * (-z_cR * inv_sR)
                + d_bvCRhi_z2 * (-z_hi * inv_sM) * sR_over_sM + d_bvCRhi_rho * drho_dsR
                - d_bvShi_z1 * (-z_s_hi * inv_sR)
                - d_bvShi_z2 * (-z_hi * inv_sM) * sR_over_sM - d_bvShi_rho * drho_dsR)
            dGhi_dsigF = jnp.where(case1_hi,
                Phi_zcR * phi_zcKF * (-z_cK_F * inv_sF),
                Phi_zs_hi * phi_zcKF * (-z_cK_F * inv_sF)
                + d_bvCRhi_z2 * (-z_hi * inv_sM) * sF_over_sM + d_bvCRhi_rho * drho_dsF
                - d_bvShi_z2 * (-z_hi * inv_sM) * sF_over_sM - d_bvShi_rho * drho_dsF)

            dGlo_dtau = jnp.where(case1_lo, 0.0,
                phi_zs_lo * inv_sR * Phi_zcKF + d_bvCRlo_z2 * inv_sM
                - d_bvSlo_z1 * inv_sR - d_bvSlo_z2 * inv_sM)
            dGlo_dcR = jnp.where(case1_lo,
                phi_zcR * inv_sR * Phi_zcKF, d_bvCRlo_z1 * inv_sR)
            dGlo_dcK = jnp.where(case1_lo,
                Phi_zcR * phi_zcKF * inv_sF,
                phi_zs_lo * (-inv_sR) * Phi_zcKF + Phi_zs_lo * phi_zcKF * inv_sF + d_bvSlo_z1 * inv_sR)
            dGlo_dmuR = jnp.where(case1_lo,
                -phi_zcR * inv_sR * Phi_zcKF,
                phi_zs_lo * (-inv_sR) * Phi_zcKF
                + d_bvCRlo_z1 * (-inv_sR) + d_bvCRlo_z2 * (-inv_sM)
                - d_bvSlo_z1 * (-inv_sR) - d_bvSlo_z2 * (-inv_sM))
            dGlo_dmuF = jnp.where(case1_lo,
                -Phi_zcR * phi_zcKF * inv_sF,
                Phi_zs_lo * phi_zcKF * (-inv_sF)
                + d_bvCRlo_z2 * (-inv_sM) - d_bvSlo_z2 * (-inv_sM))
            dGlo_dsigR = jnp.where(case1_lo,
                phi_zcR * (-z_cR * inv_sR) * Phi_zcKF,
                phi_zs_lo * (-z_s_lo * inv_sR) * Phi_zcKF
                + d_bvCRlo_z1 * (-z_cR * inv_sR)
                + d_bvCRlo_z2 * (-z_lo * inv_sM) * sR_over_sM + d_bvCRlo_rho * drho_dsR
                - d_bvSlo_z1 * (-z_s_lo * inv_sR)
                - d_bvSlo_z2 * (-z_lo * inv_sM) * sR_over_sM - d_bvSlo_rho * drho_dsR)
            dGlo_dsigF = jnp.where(case1_lo,
                Phi_zcR * phi_zcKF * (-z_cK_F * inv_sF),
                Phi_zs_lo * phi_zcKF * (-z_cK_F * inv_sF)
                + d_bvCRlo_z2 * (-z_lo * inv_sM) * sF_over_sM + d_bvCRlo_rho * drho_dsF
                - d_bvSlo_z2 * (-z_lo * inv_sM) * sF_over_sM - d_bvSlo_rho * drho_dsF)

            dlp_dzcR = eff_dpR * (-d_bvCRhi_z1 + d_bvCRlo_z1)
            dlp_dzhi = eff_dpR * (phi_hi - d_bvCRhi_z2) + eff_dpBand * phi_hi
            dlp_dzlo = eff_dpR * (-(phi_lo - d_bvCRlo_z2)) + eff_dpBand * (-phi_lo)
            dlp_drho = eff_dpR * (-d_bvCRhi_rho + d_bvCRlo_rho)
            dlp_dztau1 = jnp.where(is_new_resp, phi_tau1 * inv_prob, 0.0)

            dG_dmuR = dGhi_dmuR - dGlo_dmuR
            dG_dsigR = dGhi_dsigR - dGlo_dsigR
            dG_dmuF = dGhi_dmuF - dGlo_dmuF
            dG_dsigF = dGhi_dsigF - dGlo_dsigF
            dG_dcR = dGhi_dcR - dGlo_dcR
            dG_dcK = dGhi_dcK - dGlo_dcK

            d_val_dmuR = (dlp_dzcR * (-inv_sR)
                         + (dlp_dzhi + dlp_dzlo + dlp_dztau1) * (-inv_sM)
                         + eff_dpG * dG_dmuR)
            d_val_dmuF = ((dlp_dzhi + dlp_dzlo + dlp_dztau1) * (-inv_sM)
                         + eff_dpG * dG_dmuF)
            d_val_dsigR = (dlp_dzcR * (-z_cR * inv_sR)
                          + (dlp_dzhi * (-z_hi) + dlp_dzlo * (-z_lo) +
                             dlp_dztau1 * (-z_tau1)) * inv_sM * sR_over_sM
                          + dlp_drho * drho_dsR + eff_dpG * dG_dsigR)
            d_val_dsigF = ((dlp_dzhi * (-z_hi) + dlp_dzlo * (-z_lo) +
                            dlp_dztau1 * (-z_tau1)) * inv_sM * sF_over_sM
                          + dlp_drho * drho_dsF + eff_dpG * dG_dsigF)
            d_val_dcR = dlp_dzcR * inv_sR + eff_dpG * dG_dcR
            d_val_dcK = eff_dpG * dG_dcK

            d_val_dtau_hi = jnp.where(is_old_resp, dlp_dzhi * inv_sM + eff_dpG * dGhi_dtau, 0.0)
            d_val_dtau_lo = jnp.where(is_old_resp, dlp_dzlo * inv_sM + eff_dpG * (-dGlo_dtau), 0.0)
            d_val_dtau_first = jnp.where(is_new_resp, dlp_dztau1 * inv_sM, 0.0)

        d_t_k_lo = d_val_dtau_lo
        d_t_k_hi = jnp.where(has_tau_hi, d_val_dtau_hi, 0.0)
        d_t_first = d_val_dtau_first

        d_lp_d_thresh_mid = d_t_k_lo + d_t_k_hi + d_t_first

        gm = jnp.asarray(gap_mask_np)
        mask_tau_lo = gm[:, k_tau_lo]
        mask_tau_hi = gm[:, k_tau_hi]
        mask_first = gm[:, jnp.zeros(N, dtype=jnp.int32)]

        d_lp_d_gap_val = (mask_tau_lo * d_t_k_lo[None, :] +
                          mask_tau_hi * d_t_k_hi[None, :] +
                          mask_first * d_t_first[None, :])
        d_lp_d_gaps = (d_lp_d_gap_val * gd.T).T

        tangent_out = (
            d_val_dmuR * d_mu_R +
            d_val_dmuF * d_mu_F +
            d_val_dsigR * d_sigma_R +
            d_val_dsigF * d_sigma_F +
            d_val_dcR * d_c_R +
            d_val_dcK * d_c_K +
            d_lp_d_thresh_mid * d_thresh_mid_v +
            jnp.sum(d_lp_d_gaps * d_gaps_v, axis=1)
        )

        return val, tangent_out

    return _batch_fused


_fused_cache = {}


# --- Vectorized fused wrappers for new families ---

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

    bounded_biv_prob: same as bounded bivariate_sdt (Phi-product below cp + GL10 above cp).
    bounded_marg_source: Phi(z_cp)*indep_below + _strip_prob_upper(z_cp, z2_lo, z2_hi, rho).
    """
    n_thresh1 = K1 - 1
    n_gaps1 = n_thresh1 - 1
    mid_idx1 = (n_thresh1 - 1) // 2
    n_upper1 = n_thresh1 - mid_idx1 - 1
    n_thresh2 = K2 - 1
    n_gaps2 = n_thresh2 - 1
    mid_idx2 = (n_thresh2 - 1) // 2
    n_upper2 = n_thresh2 - mid_idx2 - 1

    def _batch_fused(y1, y2, item_type, mu1, mu2, sigma1, sigma2, rho_val, rho_N_val,
                     R_I, R_S, R_I_B, R_S_B, thresh1_mid, gaps1, thresh2_mid, gaps2):
        gap_fn = jax.nn.softplus if gap_link_is_softplus else jnp.exp
        gv1 = gap_fn(gaps1); gv2 = gap_fn(gaps2)
        c1 = _batch_build_thresh(thresh1_mid, gv1, n_gaps1, mid_idx1, n_upper1)
        c2 = _batch_build_thresh(thresh2_mid, gv2, n_gaps2, mid_idx2, n_upper2)

        N = y1.shape[0]
        r1 = y1 - 1; r2 = y2 - 1
        obs_idx = jnp.arange(N)
        k1_hi = jnp.clip(r1, 0, n_thresh1 - 1)
        k1_lo = jnp.clip(r1 - 1, 0, n_thresh1 - 1)
        k2_hi = jnp.clip(r2, 0, n_thresh2 - 1)
        k2_lo = jnp.clip(r2 - 1, 0, n_thresh2 - 1)

        INF = 20.0
        inv_s1 = 1.0 / sigma1
        inv_s2 = 1.0 / sigma2

        z1_hi = jnp.where(r1 < K1 - 1, (c1[obs_idx, k1_hi] - mu1) * inv_s1, INF)
        z1_lo = jnp.where(r1 > 0, (c1[obs_idx, k1_lo] - mu1) * inv_s1, -INF)
        z2_hi = jnp.where(r2 < K2 - 1, (c2[obs_idx, k2_hi] - mu2) * inv_s2, INF)
        z2_lo = jnp.where(r2 > 0, (c2[obs_idx, k2_lo] - mu2) * inv_s2, -INF)

        sigma_c = jnp.sqrt(jnp.clip(1.0 - rho_val**2, 1e-20))
        sigma_cond = sigma2 * sigma_c
        sigma_cond = jnp.maximum(sigma_cond, sigma2 * 1e-6)
        inv_sc = 1.0 / sigma_cond

        rho_sigma2 = rho_val * sigma2 + 1e-30
        z_cp_raw = -mu2 / rho_sigma2
        z_cp = jnp.where(item_type == 1, -1e6, z_cp_raw)
        z_cp_eff = jnp.clip(z_cp, z1_lo, z1_hi)

        # --- bounded biv_prob ---
        zs_hi = jnp.where(r2 < K2 - 1, c2[obs_idx, k2_hi] * inv_sc, INF)
        zs_lo = jnp.where(r2 > 0, c2[obs_idx, k2_lo] * inv_sc, -INF)
        p_det_below = jnp.clip(_Phi(z_cp_eff) - _Phi(z1_lo), 0.0)
        p_src_below = jnp.clip(_Phi(zs_hi) - _Phi(zs_lo), 0.0)
        p_below = p_det_below * p_src_below

        p_above = jax.vmap(_rect_cell_prob)(z_cp_eff, z1_hi, z2_lo, z2_hi, rho_val)

        bounded_biv = jnp.clip(p_below + p_above, 1e-20)

        # --- bounded marginal source ---
        p_marg_below = _Phi(z_cp) * (_Phi(zs_hi) - _Phi(zs_lo))
        p_marg_above = _strip_prob_upper(z_cp, z2_lo, z2_hi, rho_val)
        bounded_marg = jnp.clip(p_marg_below + p_marg_above, 1e-20)

        is_A = (item_type == 2)
        R_I_eff = jnp.where(is_A, R_I, R_I_B)
        R_S_eff = jnp.where(is_A, R_S, R_S_B)

        # --- mixture ---
        p1_comp = (1.0 - R_I_eff) * bounded_biv
        is_y1_top = (y1 == K1)
        p2_comp = jnp.where(is_y1_top, R_I_eff * (1.0 - R_S_eff) * bounded_marg, 0.0)
        is_corner_A = (item_type == 2) & (y1 == K1) & (y2 == 1)
        is_corner_B = (item_type == 3) & (y1 == K1) & (y2 == K2)
        p3_comp = jnp.where(is_corner_A | is_corner_B, R_I_eff * R_S_eff, 0.0)

        prob = jnp.clip(p1_comp + p2_comp + p3_comp, 1e-20)
        return jnp.log(prob)

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
                      thresh_mid, gaps, gap_link="log", n_rkg=2):
    """Fully-vectorized fused log-likelihood for CDP."""
    N = y.shape[0]
    n_gaps = J - 1
    is_softplus = (gap_link == "softplus")

    if gaps.ndim == 1:
        gaps_per_obs = jnp.broadcast_to(gaps[None, :], (N, n_gaps))
    else:
        gaps_per_obs = gaps.T

    key = ("batch_cdp", int(K), int(J), bool(is_softplus), int(n_rkg))
    if key not in _fused_cache:
        _fused_cache[key] = _make_fused_batch_cdp_loglik(
            int(K), int(J), bool(is_softplus), int(n_rkg))
    return _fused_cache[key](y, rk, is_old, old_idx, is_old_resp,
                              mu_R, sigma_R, mu_F, sigma_F, c_R, c_K,
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

    if is_simple and has_fused_inputs and lambda_val is None:
        mu_per_obs = jnp.where(is_old == 1, mu_signal, mu_noise)
        sd_per_obs = jnp.where(is_old == 1, sd_signal, sd_noise)
        log_lik = fused_loglik_simple(y, K, mu_per_obs, sd_per_obs,
                                       thresh_mid, log_gaps, gap_link,
                                       link_name=link_name)
        if counts is not None:
            log_lik = log_lik * counts
        return log_lik

    if family_name == "dpsdt" and lambda_val is not None and has_fused_inputs:
        mu_per_obs = jnp.where(is_old == 1, mu_signal, mu_noise)
        sd_per_obs = jnp.where(is_old == 1, sd_signal, sd_noise)
        has_sigma = (sigma is not None)
        ll_old = fused_loglik_dpsdt(y, K, mu_signal, sd_signal, lambda_val,
                                     thresh_mid, log_gaps, gap_link,
                                     has_sigma=has_sigma, link_name=link_name)
        ll_new = fused_loglik_simple(y, K, mu_noise, sd_noise,
                                      thresh_mid, log_gaps, gap_link,
                                      link_name=link_name)
        log_lik = jnp.where(is_old == 1, ll_old, ll_new)
        if counts is not None:
            log_lik = log_lik * counts
        return log_lik

    if family_name == "mixture" and lambda_val is not None and has_fused_inputs:
        _dprime2 = dprime2 if dprime2 is not None else jnp.zeros(N)
        _sigma2 = sigma2 if sigma2 is not None else jnp.ones(N)
        ll_old = fused_loglik_mixture(y, K, mu_signal, sd_signal,
                                       _dprime2, _sigma2, lambda_val,
                                       thresh_mid, log_gaps, gap_link,
                                       link_name=link_name)

        if dprime_L is not None and lambda_L is not None:
            sd_L_val = sigma_L if sigma_L is not None else jnp.ones(N)
            ll_new = fused_loglik_mixture(y, K, -dprime_L, sd_L_val,
                                           mu_noise, sd_noise, lambda_L,
                                           thresh_mid, log_gaps, gap_link,
                                           link_name=link_name)
        else:
            ll_new = fused_loglik_simple(y, K, mu_noise, sd_noise,
                                          thresh_mid, log_gaps, gap_link,
                                          link_name=link_name)

        log_lik = jnp.where(is_old == 1, ll_old, ll_new)
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
    """Compute log-likelihood for bivariate SDT families.

    Uses vmap over observations — only 4 binormal_cdf calls per obs.
    When thresh1_mid/log_gaps1/thresh2_mid/log_gaps2 are provided, uses fused
    JVP path with analytic gradients through threshold construction.
    """
    if binormal_cdf is None:
        raise ImportError("owens_t_jax is required for bivariate families")

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
               thresh_mid=None, log_gaps=None, gap_link="log"):
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
    if binormal_cdf is None:
        raise ImportError("owens_t_jax is required for CDP")

    N = y.shape[0]
    K = len(old_level_map) + J

    # --- Fused JVP path ---
    if thresh_mid is not None and log_gaps is not None:
        old_level_map_0 = jnp.array(old_level_map, dtype=jnp.int32)
        y_expanded = y[:, None]
        match = (y_expanded == old_level_map_0[None, :])
        is_old_resp = jnp.any(match, axis=1).astype(jnp.float64)
        old_idx = jnp.argmax(match, axis=1)

        log_lik = fused_loglik_cdp(
            y=y, rk=rk, is_old=is_old,
            old_idx=old_idx, is_old_resp=is_old_resp,
            K=K, J=J,
            mu_R=mu_R, sigma_R=sigma_R, mu_F=mu_F, sigma_F=sigma_F,
            c_R=c_R, c_K=c_K,
            thresh_mid=thresh_mid, gaps=log_gaps,
            gap_link=gap_link, n_rkg=n_rkg)
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
    if isinstance(d, (dist.HalfNormal, dist.HalfCauchy, dist.LogNormal,
                      dist.Exponential, dist.Gamma)):
        return d
    if isinstance(d, dist.Normal):
        if float(d.loc) == 0.0:
            return dist.HalfNormal(d.scale)
        return dist.TruncatedNormal(d.loc, d.scale, low=0.0)
    return dist.FoldedDistribution(d)


def sdt_model(config):
    """NumPyro model function driven by config dict."""
    N = config["N"]
    K = config["K"]
    n_thresh = config.get("n_thresh", K - 1)
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

    _batch_scalar_betas = {}
    _batch_skip = set()
    for param_name, pconf in params_config.items():
        if param_name == "criterion":
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
        if param_name == "criterion":
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

                beta_dp = jnp.zeros(n_coef_dp)
                beta_dp2 = jnp.zeros(n_coef_dp)
                for j in range(n_coef_dp):
                    cname = dp_col_names[j] if j < len(dp_col_names) else f"V{j+1}"
                    dp_prior = parse_prior(dp_priors.get(cname, "normal(1, 1)"))
                    cname2 = dp2_col_names[j] if j < len(dp2_col_names) else f"V{j+1}"
                    dp2_prior = parse_prior(dp2_priors.get(cname2, "normal(0, 1)"))

                    val_lo = numpyro.sample(f"beta_dprime2[{j+1}]", dp2_prior)
                    raw_gap = numpyro.sample(
                        f"beta_dprime_loggap[{j+1}]",
                        dist.ImproperUniform(dist.constraints.real, (), ()))
                    gap = jnp.exp(raw_gap)
                    numpyro.factor(f"beta_dprime_loggap_jac[{j+1}]", raw_gap)
                    val_hi = numpyro.deterministic(f"beta_dprime[{j+1}]", val_lo + gap)
                    numpyro.factor(f"beta_dprime_prior[{j+1}]",
                                   dp_prior.log_prob(val_hi))

                    beta_dp2 = beta_dp2.at[j].set(val_lo)
                    beta_dp = beta_dp.at[j].set(val_hi)

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
                if len(set(prior_strs)) == 1:
                    beta = numpyro.sample(f"beta_{param_name}",
                                          parse_prior(prior_strs[0]).expand([n_coef]))
                else:
                    beta = jnp.zeros(n_coef)
                    for j in range(n_coef):
                        beta_j = numpyro.sample(f"beta_{param_name}[{j+1}]",
                                                parse_prior(prior_strs[j]))
                        beta = beta.at[j].set(beta_j)

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

        if n_terms == 1 or not correlated:
            if n_terms == 1:
                sd = numpyro.sample(f"sigma_{param}_{group}[1]", sd_prior)
                z = numpyro.sample(f"z_{param}_{group}[1]",
                                   dist.Normal(0, 1).expand([n_groups]))
                u = sd * z
                numpyro.deterministic(f"u_{param}_{group}", u)

                if re_type == "index":
                    idx = jnp.array(re_conf["index"], dtype=jnp.int32) - 1
                    _apply_re_offset(param, u[idx], param_values)
                else:
                    Z = _ensure_2d(jnp.array(re_conf["Z"], dtype=_FLOAT))
                    idx = jnp.array(re_conf["index"], dtype=jnp.int32) - 1
                    _apply_re_offset(param, Z[:, 0] * u[idx], param_values)
            else:
                u_all = jnp.zeros((n_terms, n_groups))
                for t in range(n_terms):
                    sd_t = numpyro.sample(f"sigma_{param}_{group}[{t+1}]", sd_prior)
                    z_t = numpyro.sample(f"z_{param}_{group}[{t+1}]",
                                         dist.Normal(0, 1).expand([n_groups]))
                    u_t = sd_t * z_t
                    u_all = u_all.at[t].set(u_t)

                    if re_type == "index":
                        idx = jnp.array(re_conf["index"], dtype=jnp.int32) - 1
                        if has_term_idx and n_terms > 1:
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
        else:
            sds = numpyro.sample(f"sigma_{param}_{group}",
                                  sd_prior.expand([n_terms]))

            cor_prior = parse_prior(re_conf.get("prior_cor", "lkj_corr_cholesky(1)"),
                                    dim=n_terms)
            L_corr = numpyro.sample(f"L_corr_{param}_{group}", cor_prior)
            numpyro.deterministic(f"corr_{param}_{group}", L_corr @ L_corr.T)
            L = jnp.diag(sds) @ L_corr

            z = numpyro.sample(f"z_{param}_{group}",
                               dist.Normal(0, 1).expand([n_terms, n_groups]))
            u = L @ z

            numpyro.deterministic(f"u_{param}_{group}", u.T)

            if re_type == "index":
                idx = jnp.array(re_conf["index"], dtype=jnp.int32) - 1
                if has_term_idx and n_terms > 1:
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
            if nt == 1:
                sd_val = numpyro.sample(f"sigma_cross_{cor_id}_{group}[{dim_offset + 1}]", sd_prior)
                all_sds = all_sds.at[dim_offset].set(sd_val)
            else:
                sd_vals = numpyro.sample(f"sigma_cross_{cor_id}_{group}_block_{param}",
                                          sd_prior.expand([nt]))
                all_sds = all_sds.at[dim_offset:dim_offset + nt].set(sd_vals)
            member_dims.append((param, nt, dim_offset, m))
            dim_offset += nt

        cor_prior_str = members[0].get("prior_cor", "lkj_corr_cholesky(1)")
        cor_prior = parse_prior(cor_prior_str, dim=total_dim)
        L_corr = numpyro.sample(f"L_corr_cross_{cor_id}_{group}", cor_prior)
        numpyro.deterministic(f"corr_cross_{cor_id}_{group}", L_corr @ L_corr.T)
        L = jnp.diag(all_sds) @ L_corr

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
                    if has_term_idx and nt > 1:
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

        if n_thresh_sm > 0:
            sds = numpyro.sample(f"sds_{param}_{label}_{k}",
                                 sds_prior.expand([n_thresh_sm]))
            zs = numpyro.sample(f"zs_{param}_{label}_{k}",
                                dist.Normal(0, 1).expand([n_thresh_sm, dim]))
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
            s = numpyro.deterministic(f"s_{param}_{label}_{k}", sds * zs)
            offset = Zs @ s
            _apply_re_offset(param, offset, param_values)

    # ---- Apply link functions to get parameter values ----
    trial_params = {}
    for param_name, pconf in params_config.items():
        if param_name == "criterion":
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
    n_gaps = n_thresh - 1

    if P_crit == 1:
        beta_thresh_mid = numpyro.sample("beta_thresh_mid[1]",
                                          parse_prior(prior_thresh_mid)).reshape(1)
    else:
        beta_thresh_mid = jnp.zeros(P_crit)
        for j in range(P_crit):
            mid_j = numpyro.sample(f"beta_thresh_mid[{j+1}]",
                                   parse_prior(prior_thresh_mid))
            beta_thresh_mid = beta_thresh_mid.at[j].set(mid_j)

    thresh_mid_vec = X_crit @ beta_thresh_mid

    gap_param_prefix = "beta_raw_gaps" if gap_link == "softplus" else "beta_log_gaps"
    per_gap_priors = config.get("prior_log_gaps_per_gap")
    has_per_gap = (per_gap_priors is not None and n_gaps > 0
                   and len(set(per_gap_priors)) > 1)
    beta_log_gaps = jnp.zeros((n_gaps, P_crit)) if n_gaps > 0 else None
    if n_gaps > 0:
        if has_per_gap and P_crit == 1:
            for k in range(n_gaps):
                row_k = numpyro.sample(f"{gap_param_prefix}[{k+1}]",
                                        parse_prior(per_gap_priors[k]))
                beta_log_gaps = beta_log_gaps.at[k, 0].set(row_k)
        elif has_per_gap:
            for k in range(n_gaps):
                for j in range(P_crit):
                    lg_kj = numpyro.sample(f"{gap_param_prefix}[{k+1},{j+1}]",
                                           parse_prior(per_gap_priors[k]))
                    beta_log_gaps = beta_log_gaps.at[k, j].set(lg_kj)
        elif P_crit == 1:
            beta_log_gaps = numpyro.sample(gap_param_prefix,
                                            parse_prior(prior_log_gaps).expand([n_gaps])).reshape(n_gaps, 1)
        else:
            for k in range(n_gaps):
                for j in range(P_crit):
                    lg_kj = numpyro.sample(f"{gap_param_prefix}[{k+1},{j+1}]",
                                           parse_prior(prior_log_gaps))
                    beta_log_gaps = beta_log_gaps.at[k, j].set(lg_kj)

        log_gaps_obs = beta_log_gaps @ X_crit.T
    else:
        log_gaps_obs = jnp.zeros((0, N))

    mid_idx = (n_thresh + 1) // 2 - 1
    n_upper = n_thresh - mid_idx - 1

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

        if crit_correlated and n_dims > 1:
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
            crit_cor_prior = parse_prior(
                crit_re.get("prior_cor", "lkj_corr_cholesky(1)"), dim=n_dims)
            L_corr_crit = numpyro.sample(f"L_corr_criterion_{crit_group}", crit_cor_prior)
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
        has_slopes = n_dims > n_thresh and crit_re.get("term_idx") is not None

        if True:
            if has_slopes:
                n_slope_terms = n_dims // n_thresh
                tidx = jnp.array(crit_re["term_idx"], dtype=jnp.int32) - 1
                has_re = tidx >= 0
                tidx_safe = jnp.clip(tidx, 0, n_slope_terms - 1)

                re_dim_mid = mid_idx * n_slope_terms + tidx_safe
                mid_shift = u_crit[re_dim_mid, idx_crit]
                thresh_mid_vec = thresh_mid_vec + jnp.where(has_re, mid_shift, 0.0)

                if n_gaps > 0:
                    for g in range(n_gaps):
                        if g < n_upper:
                            t_pos = mid_idx + 1 + g
                        else:
                            t_pos = mid_idx - 1 - (g - n_upper)
                        re_dim_g = t_pos * n_slope_terms + tidx_safe
                        gap_shift = u_crit[re_dim_g, idx_crit]
                        log_gaps_obs = log_gaps_obs.at[g].add(jnp.where(has_re, gap_shift, 0.0))
            else:
                u_per_obs = u_crit[:, idx_crit]
                thresh_mid_vec = thresh_mid_vec + u_per_obs[mid_idx]
                if n_gaps > 0:
                    for g in range(n_gaps):
                        if g < n_upper:
                            t_pos = mid_idx + 1 + g
                        else:
                            t_pos = mid_idx - 1 - (g - n_upper)
                        log_gaps_obs = log_gaps_obs.at[g].add(u_per_obs[t_pos])

    if cross_cor_criterion_contributions:
        for u_crit_cc, idx_crit_cc, m_conf_cc in cross_cor_criterion_contributions:
            n_dims_cc = u_crit_cc.shape[0]
            has_slopes_cc = n_dims_cc > n_thresh and m_conf_cc.get("term_idx") is not None
            if has_slopes_cc:
                n_slope_terms = n_dims_cc // n_thresh
                tidx = jnp.array(m_conf_cc["term_idx"], dtype=jnp.int32) - 1
                has_re = tidx >= 0
                tidx_safe = jnp.clip(tidx, 0, n_slope_terms - 1)

                re_dim_mid = mid_idx * n_slope_terms + tidx_safe
                mid_shift = u_crit_cc[re_dim_mid, idx_crit_cc]
                thresh_mid_vec = thresh_mid_vec + jnp.where(has_re, mid_shift, 0.0)

                if n_gaps > 0:
                    for g in range(n_gaps):
                        t_pos = mid_idx + 1 + g if g < n_upper else mid_idx - 1 - (g - n_upper)
                        re_dim_g = t_pos * n_slope_terms + tidx_safe
                        gap_shift = u_crit_cc[re_dim_g, idx_crit_cc]
                        log_gaps_obs = log_gaps_obs.at[g].add(jnp.where(has_re, gap_shift, 0.0))
            else:
                u_per_obs = u_crit_cc[:, idx_crit_cc]
                thresh_mid_vec = thresh_mid_vec + u_per_obs[mid_idx]
                if n_gaps > 0:
                    for g in range(n_gaps):
                        t_pos = mid_idx + 1 + g if g < n_upper else mid_idx - 1 - (g - n_upper)
                        log_gaps_obs = log_gaps_obs.at[g].add(u_per_obs[t_pos])

    for (Zs_sm, s_mat) in _criterion_smooth_s:
        thresh_mid_vec = thresh_mid_vec + Zs_sm @ s_mat[0]
        if n_gaps > 0:
            for g in range(n_gaps):
                log_gaps_obs = log_gaps_obs.at[g].add(Zs_sm @ s_mat[1 + g])

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

        log_lik = cdp_loglik(
                y=y, rk=rk, is_old=is_old, thresh=thresh,
                mu_R=mu_R, sigma_R=sigma_R, mu_F=mu_F, sigma_F=sigma_F,
                c_R=c_R, c_K=c_K, J=J, old_level_map=old_level_map,
                n_rkg=n_rkg, counts=counts,
                thresh_mid=thresh_mid_vec, log_gaps=log_gaps_obs,
                gap_link=gap_link,
            )

    elif is_bivariate:
        y2 = jnp.array(config["y2"], dtype=jnp.int32)
        item_type = jnp.array(config["item_type"], dtype=jnp.int32)

        K2 = config.get("K2", K)
        n_thresh2 = K2 - 1
        varying_source = config.get("varying_source_criteria", False)

        # -- Sample criterion2 REs first (needed for threshold construction) --
        crit2_re_list = [rc for rc in criterion_re_configs if rc["param"] == "criterion2"]
        varying_re_mode = config.get("varying_re", "shared")
        u_c2 = None
        idx_c2 = None

        for crit2_re in crit2_re_list:
            c2_group = crit2_re["group"]
            c2_n_groups = int(crit2_re["n_groups"])
            c2_n_dims = int(crit2_re["n_terms"])
            c2_correlated = crit2_re.get("correlated", True)
            c2_sd_prior = _positive_dist(parse_prior(crit2_re.get("prior_sd", "normal(0, 0.5)")))

            if c2_correlated and c2_n_dims > 1:
                c2_sds = jnp.zeros(c2_n_dims)
                for t in range(c2_n_dims):
                    sd_t = numpyro.sample(f"sigma_criterion2_{c2_group}[{t+1}]", c2_sd_prior)
                    c2_sds = c2_sds.at[t].set(sd_t)
                c2_cor_prior = parse_prior(
                    crit2_re.get("prior_cor", "lkj_corr_cholesky(1)"), dim=c2_n_dims)
                L_corr_c2 = numpyro.sample(f"L_corr_criterion2_{c2_group}", c2_cor_prior)
                numpyro.deterministic(f"corr_criterion2_{c2_group}", L_corr_c2 @ L_corr_c2.T)
                L_c2 = jnp.diag(c2_sds) @ L_corr_c2
                z_c2 = numpyro.sample(f"z_criterion2_{c2_group}",
                                      dist.Normal(0, 1).expand([c2_n_dims, c2_n_groups]))
                u_c2 = L_c2 @ z_c2
            else:
                u_c2 = jnp.zeros((c2_n_dims, c2_n_groups))
                for t in range(c2_n_dims):
                    sd_t = numpyro.sample(f"sigma_criterion2_{c2_group}[{t+1}]", c2_sd_prior)
                    z_t = numpyro.sample(f"z_criterion2_{c2_group}[{t+1}]",
                                         dist.Normal(0, 1).expand([c2_n_groups]))
                    u_c2 = u_c2.at[t].set(sd_t * z_t)

            idx_c2 = jnp.array(crit2_re["index"], dtype=jnp.int32) - 1

        # -- Build source dimension thresholds --
        n_gaps2 = n_thresh2 - 1
        mid2 = (n_thresh2 + 1) // 2 - 1
        n_upper2 = n_thresh2 - mid2 - 1
        _t2_mid_fused = None
        _t2_gaps_fused = None

        if varying_source:
            beta_thresh_mid_2 = numpyro.sample(
                "beta_thresh_mid_2_varying",
                parse_prior(prior_thresh_mid2).expand([K]))

            if n_gaps2 > 0:
                gap2_name = "beta_raw_gaps_2_varying" if gap_link == "softplus" else "beta_log_gaps_2_varying"
                beta_log_gaps_2 = numpyro.sample(
                    gap2_name,
                    parse_prior(prior_log_gaps2).expand([K, n_gaps2]))
            else:
                beta_log_gaps_2 = None

            y1_idx = y - 1

            thresh2_mid_per_obs = beta_thresh_mid_2[y1_idx]
            if beta_log_gaps_2 is not None:
                log_gaps2_per_obs = beta_log_gaps_2[y1_idx].T
            else:
                log_gaps2_per_obs = jnp.zeros((0, N))

            if u_c2 is not None:
                if varying_re_mode == "shared":
                    thresh2_mid_per_obs = thresh2_mid_per_obs + u_c2[0, idx_c2]

                elif varying_re_mode == "per_bin":
                    thresh2_mid_per_obs = thresh2_mid_per_obs + u_c2[y1_idx, idx_c2]

                elif varying_re_mode == "full":
                    block_start = y1_idx * n_thresh2

                    re_mid = block_start + mid2
                    thresh2_mid_per_obs = thresh2_mid_per_obs + u_c2[re_mid, idx_c2]

                    if n_gaps2 > 0:
                        for g in range(n_gaps2):
                            if g < n_upper2:
                                t_pos = mid2 + 1 + g
                            else:
                                t_pos = mid2 - 1 - (g - n_upper2)
                            re_dim_g = block_start + t_pos
                            log_gaps2_per_obs = log_gaps2_per_obs.at[g].add(
                                u_c2[re_dim_g, idx_c2])

            thresh2 = build_thresholds_vectorized(thresh2_mid_per_obs, log_gaps2_per_obs, n_thresh2, gap_link=gap_link)
            _t2_mid_fused = thresh2_mid_per_obs
            _t2_gaps_fused = log_gaps2_per_obs

            new_source_mode = config.get("new_source_criteria")
            if new_source_mode == "shared":
                c2_new_mid = numpyro.sample("beta_thresh_mid_2_new",
                                            parse_prior(prior_thresh_mid2))
                if n_gaps2 > 0:
                    gap2_new_name = "beta_raw_gaps_2_new" if gap_link == "softplus" else "beta_log_gaps_2_new"
                    c2_new_gaps = numpyro.sample(
                        gap2_new_name,
                        parse_prior(prior_log_gaps2).expand([n_gaps2]))
                else:
                    c2_new_gaps = jnp.zeros(0)
                thresh2_new = build_thresholds(c2_new_mid, c2_new_gaps, n_thresh2)

                is_new_response_arr = jnp.array(config["is_new_response"], dtype=jnp.int32)
                is_new_resp = (is_new_response_arr[y - 1] == 1)
                thresh2 = jnp.where(is_new_resp[:, None],
                                    jnp.broadcast_to(thresh2_new[None, :], (N, n_thresh2)),
                                    thresh2)
                if _t2_mid_fused is not None:
                    _t2_mid_fused = jnp.where(is_new_resp, c2_new_mid, _t2_mid_fused)
                    if n_gaps2 > 0:
                        _t2_gaps_fused = jnp.where(
                            is_new_resp[None, :],
                            jnp.broadcast_to(c2_new_gaps[:, None], (n_gaps2, N)),
                            _t2_gaps_fused)

        elif "criterion2" in params_config:
            c2_conf = params_config["criterion2"]
            c2_n_coef = c2_conf["n_coef"]
            X_c2 = _ensure_2d(jnp.array(c2_conf["X"], dtype=_FLOAT))
            c2_priors = c2_conf.get("priors", {})
            c2_col_names = c2_conf.get("col_names", [f"V{i+1}" for i in range(c2_n_coef)])
            if isinstance(c2_col_names, str):
                c2_col_names = [c2_col_names]

            beta_thresh2_mid = jnp.zeros(c2_n_coef)
            for j in range(c2_n_coef):
                cname = c2_col_names[j] if j < len(c2_col_names) else f"V{j+1}"
                prior_str = c2_priors.get(cname, prior_thresh_mid2)
                t2m_j = numpyro.sample(f"beta_thresh_mid_2[{j+1}]",
                                       parse_prior(prior_str))
                beta_thresh2_mid = beta_thresh2_mid.at[j].set(t2m_j)

            thresh2_mid_vec = X_c2 @ beta_thresh2_mid

            if n_gaps2 > 0:
                gap2_name_ij = "beta_raw_gaps_2" if gap_link == "softplus" else "beta_log_gaps_2"
                if c2_n_coef == 1:
                    beta_log_gaps2 = numpyro.sample(gap2_name_ij,
                                                     parse_prior(prior_log_gaps2).expand([n_gaps2])).reshape(n_gaps2, 1)
                else:
                    beta_log_gaps2 = jnp.zeros((n_gaps2, c2_n_coef))
                    for k in range(n_gaps2):
                        for j in range(c2_n_coef):
                            lg2_kj = numpyro.sample(f"{gap2_name_ij}[{k+1},{j+1}]",
                                                    parse_prior(prior_log_gaps2))
                            beta_log_gaps2 = beta_log_gaps2.at[k, j].set(lg2_kj)
                log_gaps2_obs = beta_log_gaps2 @ X_c2.T
            else:
                log_gaps2_obs = jnp.zeros((0, N))

            if u_c2 is not None:
                has_c2_slopes = u_c2.shape[0] > n_thresh2 and crit2_re_list[0].get("term_idx") is not None
                if has_c2_slopes:
                    n_c2_slope_terms = u_c2.shape[0] // n_thresh2
                    tidx_c2 = jnp.array(crit2_re_list[0]["term_idx"], dtype=jnp.int32) - 1
                    has_re_c2 = tidx_c2 >= 0
                    tidx_c2_safe = jnp.clip(tidx_c2, 0, n_c2_slope_terms - 1)
                    re_mid = mid2 * n_c2_slope_terms + tidx_c2_safe
                    thresh2_mid_vec = thresh2_mid_vec + jnp.where(has_re_c2, u_c2[re_mid, idx_c2], 0.0)
                    if n_gaps2 > 0:
                        for g in range(n_gaps2):
                            if g < n_upper2:
                                t_pos = mid2 + 1 + g
                            else:
                                t_pos = mid2 - 1 - (g - n_upper2)
                            re_dim_g = t_pos * n_c2_slope_terms + tidx_c2_safe
                            log_gaps2_obs = log_gaps2_obs.at[g].add(
                                jnp.where(has_re_c2, u_c2[re_dim_g, idx_c2], 0.0))
                else:
                    u_per_obs = u_c2[:, idx_c2]
                    thresh2_mid_vec = thresh2_mid_vec + u_per_obs[mid2]
                    if n_gaps2 > 0:
                        for g in range(n_gaps2):
                            if g < n_upper2:
                                t_pos = mid2 + 1 + g
                            else:
                                t_pos = mid2 - 1 - (g - n_upper2)
                            log_gaps2_obs = log_gaps2_obs.at[g].add(u_per_obs[t_pos])

            for (Zs_sm2, s_mat2) in _criterion2_smooth_s:
                thresh2_mid_vec = thresh2_mid_vec + Zs_sm2 @ s_mat2[0]
                if n_gaps2 > 0:
                    for g in range(n_gaps2):
                        log_gaps2_obs = log_gaps2_obs.at[g].add(Zs_sm2 @ s_mat2[1 + g])

            thresh2 = build_thresholds_vectorized(thresh2_mid_vec, log_gaps2_obs, n_thresh2, gap_link=gap_link)
            _t2_mid_fused = thresh2_mid_vec
            _t2_gaps_fused = log_gaps2_obs

        else:
            thresh2_mid = numpyro.sample("beta_thresh_mid_2[1]",
                                         parse_prior(prior_thresh_mid2))
            if n_gaps2 > 0:
                gap2_name_int = "beta_raw_gaps_2" if gap_link == "softplus" else "beta_log_gaps_2"
                log_gaps2 = numpyro.sample(gap2_name_int,
                                            parse_prior(prior_log_gaps2).expand([n_gaps2]))
            else:
                log_gaps2 = jnp.zeros(n_gaps2)

            thresh2_single = build_thresholds(thresh2_mid, log_gaps2, n_thresh2)
            thresh2 = jnp.broadcast_to(thresh2_single[None, :], (N, n_thresh2))

            if u_c2 is not None or _criterion2_smooth_s:
                thresh2_mid_vec_re = jnp.broadcast_to(thresh2_mid, (N,))
                log_gaps2_obs_re = jnp.broadcast_to(log_gaps2[:, None], (n_gaps2, N)) if n_gaps2 > 0 else jnp.zeros((0, N))
                if u_c2 is not None:
                    u_per_obs = u_c2[:, idx_c2]
                    thresh2_mid_vec_re = thresh2_mid_vec_re + u_per_obs[mid2]
                    if n_gaps2 > 0:
                        for g in range(n_gaps2):
                            if g < n_upper2:
                                t_pos = mid2 + 1 + g
                            else:
                                t_pos = mid2 - 1 - (g - n_upper2)
                            log_gaps2_obs_re = log_gaps2_obs_re.at[g].add(u_per_obs[t_pos])
                for (Zs_sm2, s_mat2) in _criterion2_smooth_s:
                    thresh2_mid_vec_re = thresh2_mid_vec_re + Zs_sm2 @ s_mat2[0]
                    if n_gaps2 > 0:
                        for g in range(n_gaps2):
                            log_gaps2_obs_re = log_gaps2_obs_re.at[g].add(Zs_sm2 @ s_mat2[1 + g])
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
        )

    if counts is not None:
        log_lik_unweighted = log_lik / counts
    else:
        log_lik_unweighted = log_lik
    numpyro.deterministic("log_lik", log_lik_unweighted)

    numpyro.factor("obs", jnp.sum(log_lik))


# =============================================================================
# Entry Point
# =============================================================================

def fit_model(config, chains=4, warmup=1000, samples=1000, seed=0,
              target_accept_prob=0.8, max_tree_depth=10, step_size=1.0,
              thinning=1, chain_method="parallel",
              progress_bar=True,
              init_strategy="prior", init_radius=2.0,
              sampler="nuts", control=None, save_loglik=True,
              keep_re=False, warm_state=None, save_state=False):
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
            (init_to_uniform with `init_radius`). Default "prior".
        init_radius: radius for init_to_uniform when init_strategy != "prior".
        sampler: "nuts" (default) or "drghmc".
        control: dict of drghmc tuning knobs (ignored for nuts).

    Returns:
        dict with samples (by chain), extra_fields, elapsed
    """
    if sampler == "drghmc":
        from drghmc import run_drghmc, _set_model
        _set_model(sdt_model)
        resume = None
        if warm_state is not None:
            with open(warm_state, "rb") as fh:
                resume = pickle.load(fh)
        return run_drghmc(config, chains=chains, warmup=warmup, samples=samples,
                          seed=seed, thinning=thinning, chain_method=chain_method,
                          progress_bar=progress_bar, init_strategy=init_strategy,
                          init_radius=init_radius, control=control,
                          save_loglik=save_loglik, keep_re=keep_re,
                          resume_state=resume, save_state=save_state)

    numpyro.set_host_device_count(chains if chain_method == "parallel" else 1)
    rng_key = random.PRNGKey(seed)

    if init_strategy == "prior":
        kernel_init = init_to_sample
    else:
        kernel_init = init_to_uniform(radius=float(init_radius))

    kernel = NUTS(
        sdt_model,
        init_strategy=kernel_init,
        target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth,
        step_size=float(step_size),
    )

    mcmc = MCMC(
        kernel,
        num_warmup=0 if warm_state is not None else warmup,
        num_samples=samples,
        num_chains=chains,
        thinning=int(thinning),
        chain_method=chain_method,
        progress_bar=bool(progress_bar),
    )

    if warm_state is not None:
        with open(warm_state, "rb") as fh:
            mcmc.post_warmup_state = pickle.load(fh)
        rng_key = mcmc.post_warmup_state.rng_key

    t0 = time.time()
    mcmc.run(rng_key, config,
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
        if k.startswith("L_corr_"):
            continue
        if k.startswith("z_") and not keep_re:
            continue
        if k == "log_lik" and not save_loglik:
            continue
        samples_np[k] = np.array(v)

    out = {
        "samples": samples_np,
        "extra_fields": extra_fields,
        "elapsed": float(elapsed),
    }
    if save_state:
        out["state"] = mcmc.last_state
    return out
