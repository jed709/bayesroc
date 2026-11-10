import json
import os
import sys

import numpy as np

# In float32, logit and cloglog cells far below the format's floor keep their log-probability and gradient:
# open cells at standardised distances of 20, 100 and 300 (logit) and 12, 40 and 80 (cloglog) against closed
# forms in float64, for the simple and mixture kernels. Arguments: the backend directory, the native kernel library.

sys.path.insert(0, sys.argv[1])
os.environ["BAYESROC_BVN_FFI"] = sys.argv[2]
os.environ["BAYESROC_ENABLE_X64"] = "0"
import jax
import jax.numpy as jnp
import numpyro_backend as nb

K = 6
build = nb._thresh_layout(K)[3]
t = np.asarray(build(jnp.zeros(1), jnp.ones((1, K - 2))), dtype=np.float64)[0]
y = np.concatenate([np.ones(3, int), np.full(3, K)])


def reference(link, z):
    if link == "logit":
        lp = np.where(y == 1, -np.logaddexp(0.0, -z), -np.logaddexp(0.0, z))
        d_mu = np.where(y == 1, -1.0 / (1.0 + np.exp(z)), 1.0 / (1.0 + np.exp(-z)))
    else:
        lF = np.log(-np.expm1(-np.exp(z)))
        lp = np.where(y == 1, lF, -np.exp(z))
        d_mu = np.where(y == 1, -np.exp(z - np.exp(z) - lF), np.exp(z))
    return lp, d_mu


def rel(a, b):
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - b) / np.maximum(1.0, np.abs(b))))


f32 = jnp.float32
yj, one, zero = jnp.asarray(y), jnp.ones(len(y), f32), jnp.zeros(len(y), f32)
gaps = jnp.zeros(K - 2, f32)
out = {"f32": not jax.config.jax_enable_x64}
for link, dist in (("logit", (20.0, 100.0, 300.0)), ("cloglog", (12.0, 40.0, 80.0))):
    z = np.concatenate([-np.asarray(dist), dist])
    mu = np.where(y == 1, t[0], t[-1]) - z
    lp, d_mu = reference(link, z)
    simple = lambda m: nb.fused_loglik_simple(yj, K, m, one, zero, gaps, link_name=link)
    m = jnp.asarray(mu, f32)
    out[f"{link}_simple_lp"] = rel(simple(m), lp)
    out[f"{link}_simple_grad"] = rel(jax.grad(lambda v: jnp.sum(simple(v)))(m), d_mu)
    lam, shift = 0.3, np.where(y == 1, -4.0, 4.0)
    lp2, d_mu2 = reference(link, z + shift)
    mix_lp = np.logaddexp(np.log(lam) + lp, np.log1p(-lam) + lp2)
    w1, w2 = np.exp(np.log(lam) + lp - mix_lp), np.exp(np.log1p(-lam) + lp2 - mix_lp)
    m2 = jnp.asarray(mu - shift, f32)
    mixture = lambda a, b: nb.fused_loglik_mixture(yj, K, a, one, b, one, jnp.full(len(y), lam, f32), zero, gaps,
                                                   link_name=link)
    out[f"{link}_mixture_lp"] = rel(mixture(m, m2), mix_lp)
    g1, g2 = jax.grad(lambda a, b: jnp.sum(mixture(a, b)), argnums=(0, 1))(m, m2)
    out[f"{link}_mixture_grad"] = max(rel(g1, w1 * d_mu), rel(g2, w2 * d_mu2))
    out[f"{link}_deepest_lp"] = float(min(np.min(lp), np.min(lp2)))
print("RESULT " + json.dumps(out))
