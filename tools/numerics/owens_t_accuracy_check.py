"""Accuracy audit for owens_t / binormal_cdf.

Run from the package root: python tools/numerics/owens_t_accuracy_check.py

Records a known defect (found 2026-09-28). Root cause is _poly_phi, the
5-term Zelen-Severo polynomial for Phi (documented error 7.5e-8):

  1. owens_t is discontinuous at |a| = 1, where it switches between the direct
     series and the reflected one. Jumps of 1e-8 to 7e-8.
  2. The |a| > 1 branch calls _poly_phi and carries ~7e-8 absolute error; the
     |a| < 1 branch is Gauss-Legendre only and good to ~1e-11.
  3. binormal_cdf inherits a ~1e-7 absolute error floor at every rho, and its
     analytic d/drho disagrees with its own value by ~9e-7 relative,
     step-size-independent: the derivative belongs to a different function
     than the value.

Scope: binormal_cdf is reached only by the cdp kernel and the strip helpers
used by cdp and bounded bivariate_dp. bivariate_gaussian, bivariate_dp and
vrdp2d use a 10-point Gauss-Legendre rectangle rule whose own error (~5e-7)
dominates this one.

Expected output is the baseline; a fix should drive every number below 1e-10.
"""

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "inst", "python"))
from owens_t_jax import owens_t, binormal_cdf
from scipy.special import owens_t as sp_owens_t
from scipy.stats import multivariate_normal as sp_mvn

TOL = 1e-10


def check_discontinuity():
    print("1) discontinuity of owens_t at |a| = 1")
    worst = 0.0
    for h in (0.25, 0.5, 1.0, 2.0, 3.0):
        lo = float(owens_t(jnp.float64(h), jnp.float64(1 - 1e-9)))
        hi = float(owens_t(jnp.float64(h), jnp.float64(1 + 1e-9)))
        exact = float(sp_owens_t(h, 1.0))
        jump = hi - lo
        worst = max(worst, abs(jump))
        print(f"   h={h:<5} jump={jump:+.3e}   err below={lo - exact:+.1e}  above={hi - exact:+.1e}")
    return worst


def check_owens_grid():
    print("\n2) owens_t absolute error vs scipy")
    H = np.array([0.05, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0])
    A = np.array([0.05, 0.5, 0.9, 0.99, 1.0, 1.01, 1.1, 2.0, 5.0, 20.0])
    hh, aa = np.meshgrid(H, A, indexing="ij")
    got = np.asarray(owens_t(jnp.asarray(hh.ravel()), jnp.asarray(aa.ravel())))
    err = np.abs(got - sp_owens_t(hh.ravel(), aa.ravel()))
    i = err.argmax()
    print(f"   max abs err {err.max():.2e} at h={hh.ravel()[i]}, a={aa.ravel()[i]}")
    print(f"   {int((err > TOL).sum())}/{err.size} points worse than {TOL:.0e}")
    lo = err[aa.ravel() < 1.0].max()
    hi = err[aa.ravel() > 1.0].max()
    print(f"   worst for |a|<1: {lo:.2e}   for |a|>1: {hi:.2e}")
    return err.max()


def check_binormal():
    print("\n3) binormal_cdf absolute error vs scipy (what bivariate families call)")
    zs = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    z1, z2 = np.meshgrid(zs, zs, indexing="ij")
    worst = 0.0
    for rho in (-0.8, -0.5, -0.2, 0.0, 0.2, 0.5, 0.8, 0.95):
        got = np.asarray(binormal_cdf(jnp.asarray(z1.ravel()), jnp.asarray(z2.ravel()),
                                      jnp.float64(rho)))
        ref = np.array([sp_mvn.cdf([a, b], mean=[0, 0], cov=[[1, rho], [rho, 1]])
                        for a, b in zip(z1.ravel(), z2.ravel())])
        e = np.max(np.abs(got - ref))
        worst = max(worst, e)
        print(f"   rho={rho:>5}  max abs err {e:.2e}")
    return worst


def check_gradient_consistency():
    print("\n4) does binormal_cdf's analytic d/drho match its own value?")
    z1 = jnp.array([-0.5, 0.0, 0.5, 1.0])
    z2 = jnp.array([0.3, -0.2, 0.8, -1.0])
    f = lambda r: jnp.sum(binormal_cdf(z1, z2, r))
    an = float(jax.grad(f)(jnp.float64(0.5)))
    worst = 0.0
    for h in (1e-6, 1e-4, 1e-2):
        fd = (float(f(jnp.float64(0.5 + h))) - float(f(jnp.float64(0.5 - h)))) / (2 * h)
        rel = abs(an - fd) / abs(fd)
        worst = max(worst, rel) if h < 1e-2 else worst
        print(f"   h={h:.0e}: analytic {an:.10f}  fd {fd:.10f}  rel {rel:.2e}")
    print("   step-size-independent error => the derivative is not the value's derivative")
    return worst


if __name__ == "__main__":
    results = {
        "owens_t jump at |a|=1": check_discontinuity(),
        "owens_t abs err": check_owens_grid(),
        "binormal_cdf abs err": check_binormal(),
        "binormal_cdf d/drho rel err": check_gradient_consistency(),
    }
    print("\n" + "=" * 62)
    for k, v in results.items():
        print(f"   {k:<32} {v:.2e}   {'OK' if v < TOL else 'FAILS'}")
    print("=" * 62)
