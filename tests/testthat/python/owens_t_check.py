"""Accuracy check for owens_t / binormal_cdf against scipy.

Run: python owens_t_check.py <dir containing owens_t_jax.py>
Prints a JSON summary; every number must be below 1e-10 (tests/testthat/test-numerics.R).
"""

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import os
import sys
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "inst", "python"))
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
        err_lo = lo - float(sp_owens_t(h, 1 - 1e-9))
        err_hi = hi - float(sp_owens_t(h, 1 + 1e-9))
        jump = err_hi - err_lo
        worst = max(worst, abs(jump))
        print(f"   h={h:<5} jump={jump:+.3e}   err below={err_lo:+.1e}  above={err_hi:+.1e}")
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
    cd = lambda h: (float(f(jnp.float64(0.5 + h))) - float(f(jnp.float64(0.5 - h)))) / (2 * h)
    h = 1e-3
    fd = (4.0 * cd(h / 2) - cd(h)) / 3.0
    rel = abs(an - fd) / abs(fd)
    print(f"   analytic {an:.12f}  richardson fd {fd:.12f}  rel {rel:.2e}")
    return rel


if __name__ == "__main__":
    results = {
        "owens_t jump at |a|=1": check_discontinuity(),
        "owens_t abs err": check_owens_grid(),
        "binormal_cdf abs err": check_binormal(),
        "binormal_cdf d/drho rel err": check_gradient_consistency(),
    }
    import json
    print("RESULT " + json.dumps(results))
