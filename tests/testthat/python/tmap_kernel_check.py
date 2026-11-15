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
    errs = [rel(a, b) for a, b in zip(on, orf)] + [rel(a, b) for a, b in zip(gn, gr)]
    empty = jnp.zeros((0,))
    gn0, on0 = objective(lambda Q, d, z: tmap._group_solve(Lam_t, Q, s, empty, z))
    gr0, or0 = objective(lambda Q, d, z: reference(Lam, Q, s, 0.0 * d, z))
    errs += [rel(a, b) for a, b in zip(on0, or0)] + [rel(gn0[0], gr0[0]), rel(gn0[2], gr0[2])]
    return max(errs)


def worst_block(D, n=11):
    rng = np.random.default_rng(100 + D)
    B = rng.normal(size=(n, D, D)) / np.sqrt(D)
    Lam = jnp.asarray(B @ np.transpose(B, (0, 2, 1)))
    Lam_t = tmap._tile(Lam)
    s, delta, z, w, w2 = (jnp.asarray(rng.normal(size=(D, n))) for _ in range(5))
    v = jnp.asarray(rng.normal(size=n))
    perm = jnp.asarray(rng.permutation(D).astype(np.int32))
    M = jnp.asarray(np.tril(rng.normal(size=(D, D)) / np.sqrt(D)) + 2.0 * np.eye(D))
    inv = jnp.argsort(perm)

    def composed(d, z, A):
        Ainv = solve_triangular(A, jnp.eye(D), lower=True)
        Q = Ainv.T @ Ainv
        Qk = Q[perm][:, perm]
        uk, ld = reference(Lam, Qk, s, d, z[perm])
        u = uk[inv]
        return Ainv @ u, ld, u

    def objective(f, d):
        def g(M, dl, z):
            th, ld, u = f(dl if d else jnp.zeros((0,)), z, jnp.tril(M))
            return jnp.sum(w * th) + jnp.sum(v * ld) + jnp.sum(w2 * u), (th, ld, u)
        return jax.grad(g, argnums=(0, 1, 2), has_aux=True)(M, delta, z)

    rel = lambda a, b: float(jnp.max(jnp.abs(a - b)) / jnp.max(jnp.abs(b)))
    errs = []
    for d in (True, False):
        gn, on = objective(lambda dl, z, A: tmap._block_map(Lam_t, s, dl, z, A, perm), d)
        gr, orf = objective(lambda dl, z, A: composed(dl if d else 0.0 * delta, z, A), d)
        errs += [rel(a, b) for a, b in zip(on, orf)] + [rel(gn[0], gr[0]), rel(gn[2], gr[2])]
        if d:
            errs.append(rel(gn[1], gr[1]))
    return max(errs)


print("RESULT " + json.dumps({"ffi_loaded": nb._bvn_lib is not None, **{f"D{D}": worst(D) for D in map(int, sys.argv[3].split(","))},
                              **{f"block_D{D}": worst_block(D) for D in map(int, sys.argv[3].split(",")) if D >= tmap._TRI_MIN}}))
