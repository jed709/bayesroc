import json
import os
import sys

sys.path.insert(0, sys.argv[1])
os.environ["BAYESROC_BVN_FFI"] = sys.argv[2]
os.environ["BAYESROC_ENABLE_X64"] = "1"
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
import numpyro_backend as nb
import tmap


def reference(Lam, Q, s, delta, zeta):
    Qs = jnp.tril(Q) + jnp.tril(Q, -1).T

    def one(lam, sg, dg, zg):
        L = jnp.linalg.cholesky(lam + Qs)
        y = solve_triangular(L, sg - lam @ dg, lower=True)
        return solve_triangular(L.T, y + zg, lower=False), jnp.sum(jnp.log(jnp.diag(L)))

    u, ld = jax.vmap(one)(Lam, s.T, delta.T, zeta.T)
    return u.T, ld


def worst(D, n=11):
    rng = np.random.default_rng(D)
    B = rng.normal(size=(n, D, D)) / np.sqrt(D)
    Lam = jnp.asarray(B @ np.transpose(B, (0, 2, 1)))
    M = jnp.asarray(rng.normal(size=(D, D)) / np.sqrt(D))
    s, delta, zeta, w = (jnp.asarray(rng.normal(size=(D, n))) for _ in range(4))
    v = jnp.asarray(rng.normal(size=n))
    Lam_t = tmap._tile(Lam)

    def objective(solve):
        def f(M, delta, zeta):
            u, ld = solve(M @ M.T + jnp.eye(D), delta, zeta)
            return jnp.sum(w * u) + jnp.sum(v * ld), (u, ld)
        return jax.grad(f, argnums=(0, 1, 2), has_aux=True)(M, delta, zeta)

    gn, on = objective(lambda Q, d, z: tmap._group_solve(Lam_t, Q, s, d, z))
    gr, orf = objective(lambda Q, d, z: reference(Lam, Q, s, d, z))
    rel = lambda a, b: float(jnp.max(jnp.abs(a - b)) / jnp.max(jnp.abs(b)))
    return max([rel(a, b) for a, b in zip(on, orf)] + [rel(a, b) for a, b in zip(gn, gr)])


print("RESULT " + json.dumps({"ffi_loaded": bool(nb._BVN_FFI), **{f"D{D}": worst(D) for D in map(int, sys.argv[3].split(","))}}))
