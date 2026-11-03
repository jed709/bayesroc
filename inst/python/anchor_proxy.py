# =============================================================================
# Geometry proxy for transport-map anchors (test tooling, removed at the freeze)
# =============================================================================

import numpy as np
import jax
import jax.numpy as jnp


def _lanczos(matvec, D, k, rng):
    q = rng.standard_normal(D)
    q /= np.linalg.norm(q)
    Q, a, b = [q], [], []
    for j in range(k):
        w = matvec(Q[-1])
        aj = float(Q[-1] @ w)
        w = w - aj * Q[-1] - (b[-1] * Q[-2] if j > 0 else 0.0)
        Qm = np.array(Q)
        w = w - Qm.T @ (Qm @ w)
        bj = float(np.linalg.norm(w))
        a.append(aj)
        if bj < 1e-10 or j == k - 1:
            break
        b.append(bj)
        Q.append(w / bj)
    T = np.diag(a) + np.diag(b[:len(a) - 1], 1) + np.diag(b[:len(a) - 1], -1)
    return np.linalg.eigvalsh(T)


def proxy(coords, ref, hmask, k=60, eps=1e-4, seed=0):
    tm, aux, trim, Um = coords.tm, coords.aux, coords.vg.trim, coords._u_model()

    def U(v):
        th, ld = tm.forward(v, aux, trim)
        return Um(th) + ld

    g = jax.jit(jax.grad(U))
    V = coords._tm.inverse_chains(np.asarray(ref), np.repeat(coords.aux_flat, ref.shape[0], 0), trim)
    rng = np.random.default_rng(seed)
    out = []
    for x in np.asarray(V):
        x = jnp.asarray(x)
        hv = lambda v: (np.asarray(g(x + eps * jnp.asarray(v))) - np.asarray(g(x - eps * jnp.asarray(v)))) / (2 * eps)
        D = x.shape[0]
        hidx = np.flatnonzero(hmask)
        hd = np.ones(D)
        for i in hidx:
            e = np.zeros(D)
            e[i] = 1.0
            hd[i] = max(hv(e)[i], 1e-8)
        p = 1.0 / np.sqrt(hd)
        ev = _lanczos(lambda v: p * hv(p * v), D, k, rng)
        lmax, lmin_pos = float(ev.max()), float(ev[ev > 0].min()) if np.any(ev > 0) else np.nan
        out.append(dict(lmax=lmax, lmin=lmin_pos, neg=int(np.sum(ev <= 0)), kappa=lmax / lmin_pos,
                        hyper_curv_med=float(np.median(hd[hidx]))))
    return out
