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
from scipy.special import ndtr, log_ndtr


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
big = ref > 1e-4

fx = np.genfromtxt(sys.argv[3], delimiter=",", skip_header=1)
fx[:, :4] = np.where(np.isinf(fx[:, :4]), np.sign(fx[:, :4]) * 1e10, fx[:, :4])
tail = nb._bvn_rect(*[jnp.asarray(fx[:, j]) for j in range(5)])
tail_rel = np.abs(np.expm1(np.log(np.asarray(tail[0])) - fx[:, 5]))
tail_fb = nb._with_rect_tail(nb._dw_rect_grads_jax(*[jnp.asarray(fx[:, j]) for j in range(5)]),
                             *[jnp.asarray(fx[:, j]) for j in range(5)])
tail_fb_rel = np.abs(np.expm1(np.log(np.asarray(tail_fb[0])) - fx[:, 5]))
cdf = nb._bvn_cdf(A[1], A[3], A[4])
ref_cdf = np.array([bvn(z1h[i], z2h[i], rho[i]) for i in range(n)])
xs = np.linspace(-37.0, 37.0, 2001)
P, _, Q = nb._acc_phi_and_deriv(jnp.asarray(xs))
refQ = ndtr(-np.abs(xs))

hfx = np.genfromtxt(sys.argv[5], delimiter=",", skip_header=1)
hi_rect, hi_cdf, hi_tail, hi_tail999, hi_jax, hi_point = [], [], [], [], [], []
for rr in (0.9, -0.9, 0.99, -0.99, 0.997, -0.997, 0.9988, -0.9988, 0.999, -0.999, 0.9995, -0.9995, 0.9999, -0.9999, 0.99999, -0.99999):
    m = 120
    a1 = rng.uniform(-5, 5, m); b1 = a1 + rng.uniform(0.05, 4.0, m)
    a2 = rng.uniform(-5, 5, m); b2 = a2 + rng.uniform(0.05, 4.0, m)
    rv = np.full(m, rr)
    args = [jnp.asarray(v) for v in (a1, b1, a2, b2, rv)]
    got = np.asarray(nb._bvn_rect(*args)[0])
    want = np.array([bvn(b1[i], b2[i], rr) - bvn(a1[i], b2[i], rr) - bvn(b1[i], a2[i], rr) + bvn(a1[i], a2[i], rr) for i in range(m)])
    keep = want > 1e-6
    hi_rect.append(np.max(np.abs(got[keep] / want[keep] - 1)))
    cg = np.asarray(nb._bvn_cdf(args[1], args[3], args[4])[0])
    cw = np.array([bvn(b1[i], b2[i], rr) for i in range(m)])
    hi_cdf.append(np.max(np.abs(cg - cw)))
    hx = hfx[np.isclose(hfx[:, 4], rr)]
    if len(hx):
        tg = np.asarray(nb._bvn_rect(*[jnp.asarray(hx[:, j]) for j in range(5)])[0])
        (hi_tail999 if abs(rr) > 0.995 else hi_tail).append(np.max(np.abs(np.expm1(np.log(tg) - hx[:, 5]))))
    hh = rng.uniform(-6, 6, m); kk = rng.uniform(-6, 6, m)
    pg = np.asarray(nb._bvn_cdf(jnp.asarray(hh), jnp.asarray(kk), jnp.asarray(rv))[0])
    hi_point.append(np.max(np.abs(pg - np.array([bvn(hh[i], kk[i], rr) for i in range(m)]))))
    if nb._BVN_FFI:
        nb._BVN_FFI = False
        jg = np.asarray(nb._bvn_rect(*args)[0])
        nb._BVN_FFI = True
        hi_jax.append(np.max(np.abs(jg[keep] / got[keep] - 1)))

out = {
    "hi_rho_rect_rel_err": float(max(hi_rect)),
    "hi_rho_cdf_abs_err": float(max(hi_cdf)),
    "hi_rho_point_abs_err": float(max(hi_point)),
    "hi_rho_tail_rel_err": float(max(hi_tail)),
    "hi_rho_tail_rel_err_999": float(max(hi_tail999)),
    "hi_rho_ffi_vs_jax": float(max(hi_jax)) if hi_jax else 0.0,
    "ffi_loaded": bool(nb._BVN_FFI),
    "rect_abs_err": float(np.max(np.abs(np.asarray(rect[0]) - ref))),
    "rect_rel_err": float(np.max(np.abs(np.asarray(rect[0])[big] / ref[big] - 1))),
    "tail_rel_err": float(tail_rel.max()),
    "tail_fallback_rel_err": float(tail_fb_rel.max()),
    "cdf_abs_err": float(np.max(np.abs(np.asarray(cdf[0]) - ref_cdf))),
    "acc_phi_abs_err": float(np.max(np.abs(np.asarray(P) - ndtr(xs)))),
    "acc_phi_tail_rel_err": float(np.max(np.abs(np.asarray(Q) - refQ) / refQ)),
}
if nb._BVN_FFI:
    nb._BVN_FFI = False
    jr = nb._bvn_rect(*A)
    jc = nb._bvn_cdf(A[1], A[3], A[4])
    nb._BVN_FFI = True
    out["ffi_vs_jax_rect"] = float(np.max(np.abs(np.asarray(rect[0]) / np.asarray(jr[0]) - 1)))
    out["ffi_vs_jax_cdf"] = float(np.max(np.abs(np.asarray(cdf[0]) - np.asarray(jc[0]))))

    lo = np.concatenate([rng.uniform(-37, 37, 4000), np.full(200, -1e10), rng.uniform(-8, 8, 200)])
    hi = lo + np.concatenate([rng.exponential(1.0, 4000), np.zeros(400)])
    hi[4000:4200] = rng.uniform(-37, 37, 200)
    hi[4200:] = 1e10
    up = lo >= 0
    la, lb = np.where(up, log_ndtr(-lo), log_ndtr(hi)), np.where(up, log_ndtr(-hi), log_ndtr(lo))
    ref_lp = la + np.log1p(-np.exp(lb - la))
    ok = np.isfinite(ref_lp) & (ref_lp > -680)
    lp, glo, ghi = (np.asarray(v) for v in nb._probit_cell_native(jnp.asarray(lo), jnp.asarray(hi)))
    out["probit_cell_rel_err"] = float(np.max(np.abs(lp[ok] - ref_lp[ok]) / np.maximum(1, np.abs(ref_lp[ok]))))
    h = 1e-4 * np.minimum(1, np.abs(hi - lo))
    fin = ok & (np.abs(lo) < 30) & (np.abs(hi) < 30)
    fd_lo = (np.asarray(nb._probit_cell_native(jnp.asarray(lo + h), jnp.asarray(hi))[0])
             - np.asarray(nb._probit_cell_native(jnp.asarray(lo - h), jnp.asarray(hi))[0])) / (2 * h)
    hh = h
    fd_hi = (np.asarray(nb._probit_cell_native(jnp.asarray(lo), jnp.asarray(hi + hh))[0])
             - np.asarray(nb._probit_cell_native(jnp.asarray(lo), jnp.asarray(hi - hh))[0])) / (2 * hh)
    cx = np.genfromtxt(sys.argv[4], delimiter=",", skip_header=1)
    cargs = [jnp.asarray(cx[:, j]) for j in range(1, 9)]
    cargs.append(jnp.zeros(cx.shape[0]))
    ref_lp = cx[:, 9]
    cfin = np.isfinite(ref_lp)
    nat_lp = np.asarray(nb._cdp_tail(jnp.ones(cx.shape[0], bool), jnp.asarray(cx[:, 0]), *cargs)[0])
    jax_lp = np.asarray(jax.jit(nb._cdp_cell_jax)(jnp.asarray(cx[:, 0]), *cargs)[0])
    out["cdp_cell_rel_err"] = float(np.max(np.abs(nat_lp[cfin] - ref_lp[cfin]) / np.maximum(1, np.abs(ref_lp[cfin]))))
    out["cdp_cell_jax_rel_err"] = float(np.max(np.abs(jax_lp[cfin] - ref_lp[cfin]) / np.maximum(1, np.abs(ref_lp[cfin]))))
    out["cdp_cell_empty_ok"] = bool(np.all(~np.isfinite(nat_lp[~cfin])) and np.all(~np.isfinite(jax_lp[~cfin])))
    out["probit_cell_fd_err"] = float(max(
        np.max(np.abs(fd_lo[fin] - glo[fin]) / np.maximum(1, np.abs(glo[fin]))),
        np.max(np.abs(fd_hi[fin] - ghi[fin]) / np.maximum(1, np.abs(ghi[fin])))))
print("RESULT " + json.dumps(out))
