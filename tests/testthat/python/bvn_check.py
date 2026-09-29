import json
import os
import sys

import numpy as np

sys.path.insert(0, sys.argv[1])
if len(sys.argv) > 2 and sys.argv[2]:
    os.environ["BAYESROC_BVN_FFI"] = sys.argv[2]
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
import numpyro_backend as nb
from scipy.stats import multivariate_normal as mvn
from scipy.special import ndtr


def bvn(h, k, r):
    if h <= -38 or k <= -38:
        return 0.0
    return mvn.cdf([min(h, 38.0), min(k, 38.0)], cov=[[1, r], [r, 1]], abseps=1e-15, releps=1e-15)


rng = np.random.default_rng(7)
n = 400
z1l = rng.normal(-0.5, 1.5, n)
z1h = z1l + rng.uniform(0.05, 3.0, n)
z2l = rng.normal(-0.5, 1.5, n)
z2h = z2l + rng.uniform(0.05, 3.0, n)
z1l[:20] = -20.0
z2h[20:40] = 20.0
rho = rng.uniform(-0.98, 0.98, n)
A = [jnp.asarray(v) for v in (z1l, z1h, z2l, z2h, rho)]

rect = nb._bvn_rect(*A)
ref = np.array([bvn(z1h[i], z2h[i], rho[i]) - bvn(z1l[i], z2h[i], rho[i])
                - bvn(z1h[i], z2l[i], rho[i]) + bvn(z1l[i], z2l[i], rho[i]) for i in range(n)])
cdf = nb._bvn_cdf(A[1], A[3], A[4])
ref_cdf = np.array([bvn(z1h[i], z2h[i], rho[i]) for i in range(n)])
xs = np.linspace(-37.0, 37.0, 2001)
P, _, Q = nb._acc_phi_and_deriv(jnp.asarray(xs))
refQ = ndtr(-np.abs(xs))

out = {
    "ffi_loaded": bool(nb._BVN_FFI),
    "rect_abs_err": float(np.max(np.abs(np.asarray(rect[0]) - ref))),
    "cdf_abs_err": float(np.max(np.abs(np.asarray(cdf[0]) - ref_cdf))),
    "acc_phi_abs_err": float(np.max(np.abs(np.asarray(P) - ndtr(xs)))),
    "acc_phi_tail_rel_err": float(np.max(np.abs(np.asarray(Q) - refQ) / refQ)),
}
if nb._BVN_FFI:
    jr = nb._dw_rect_grads_jax(*A)
    jc = nb._dw_cdf_grads_jax(A[1], A[3], A[4])
    out["ffi_vs_jax_rect"] = float(max(np.max(np.abs(np.asarray(a) - np.asarray(b))) for a, b in zip(rect, jr)))
    out["ffi_vs_jax_cdf"] = float(max(np.max(np.abs(np.asarray(a) - np.asarray(b))) for a, b in zip(cdf, jc)))
print("RESULT " + json.dumps(out))
