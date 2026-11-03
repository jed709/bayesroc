# =============================================================================
# Pathfinder (Zhang, Carpenter, Gelman & Vehtari, 2022): one path, BlackJAX's L-BFGS
# =============================================================================

import time

import jax
import jax.numpy as jnp
import numpy as np
from blackjax.optimizers.lbfgs import minimize_lbfgs, lbfgs_inverse_hessian_factors


def _approx(x, g_logp, alpha, beta, gamma):
    k = beta.shape[1]
    Q, R = jnp.linalg.qr(beta / jnp.sqrt(alpha)[:, None])
    L = jnp.linalg.cholesky(jnp.eye(k) + R @ gamma @ R.T)
    logdet = jnp.sum(jnp.log(alpha)) + 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
    mu = x + alpha * g_logp + beta @ (gamma @ (beta.T @ g_logp))
    return mu, Q, L, logdet


def _draw(key, n, mu, alpha, Q, L, logdet):
    D, k = mu.shape[0], L.shape[0]
    u = jax.random.normal(key, (n, D), dtype=mu.dtype)
    v = u + ((L - jnp.eye(k)) @ (Q.T @ u.T)).T @ Q.T
    phi = mu + jnp.sqrt(alpha) * v
    logq = -0.5 * (logdet + jnp.sum(u * u, 1) + D * jnp.log(2.0 * jnp.pi))
    return phi, logq


def _diag_cov(alpha, beta, gamma):
    return alpha + jnp.sum((beta @ gamma) * beta, 1)


def pathfinder(U, x0, seed, n_draws, maxiter=1000, maxcor=5, ftol=1e-5, gtol=1e-8, maxls=1000, n_elbo=25):
    t0 = time.time()
    x0 = jnp.asarray(x0)
    last, hist = jax.jit(lambda x: minimize_lbfgs(U, x, maxiter, maxcor, gtol, ftol, maxls))(x0)
    n = int(last.state.iter_num)
    X, G, A = hist.x[:n + 1], hist.g[:n + 1], hist.alpha[:n + 1]
    mask = hist.update_mask[1:n + 1]
    s = jnp.where(mask, jnp.diff(X, axis=0), 0.0)
    z = jnp.where(mask, jnp.diff(G, axis=0), 0.0)
    s = jnp.pad(s, ((maxcor, 0), (0, 0)))
    z = jnp.pad(z, ((maxcor, 0), (0, 0)))
    idx = jnp.arange(n + 1)[:, None] + jnp.arange(maxcor)[None, :]
    keys = jax.random.split(jax.random.PRNGKey(seed), n + 2)

    def body(a):
        key, S, Z, al, x, g = a
        beta, gamma = lbfgs_inverse_hessian_factors(S.T, Z.T, al)
        mu, Q, L, ld = _approx(x, -g, al, beta, gamma)
        phi, logq = _draw(key, n_elbo, mu, al, Q, L, ld)
        logp = -jax.lax.map(U, phi)
        e = jnp.mean(logp - logq)
        return jnp.where(jnp.isfinite(e), e, -jnp.inf)

    elbo = np.asarray(jax.jit(lambda *a: jax.lax.map(body, a))(keys[:n + 1], s[idx], z[idx], A, X, G))
    best = int(np.argmax(elbo))
    beta, gamma = lbfgs_inverse_hessian_factors(s[idx[best]].T, z[idx[best]].T, A[best])
    mu, Q, L, ld = _approx(X[best], -G[best], A[best], beta, gamma)
    draws, _ = _draw(keys[-1], n_draws, mu, A[best], Q, L, ld)
    return dict(mu=np.asarray(mu), position=np.asarray(X[best]), draws=np.asarray(draws),
                diag=np.asarray(_diag_cov(A[best], beta, gamma)), elbo=elbo, best=best, iters=n,
                wall=time.time() - t0)
