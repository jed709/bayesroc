"""Warmup adaptation of the jittered static HMC trajectory length.

"chees": ChEES-HMC (Hoffman, Radul and Sountsov 2021, Algorithm 1): the maximum trajectory length T,
jittered by a Halton sequence shared by the chains, follows Adam steps on the acceptance-weighted
cross-chain mean of the ChEES gradient, with the chain means as centres; the sampling cap is the
moving average of T.

"snaper": SNAPER-HMC (Sountsov and Hoffman 2022, Algorithm 1 and Appendix C): the trajectory length is
drawn uniformly on (0, 2 tau_bar), one step for the first iterations; tau_bar follows Adam steps on the
gradient of the ESJD of (z'w)^2 divided by the trajectory length, w the first principal component of
the centred states (minibatch Oja) and the centre a running mean; the sampling cap is twice the moving
average of tau_bar.

"esjd": as "chees" with the expected squared jump distance |z' - z|^2 (Pasarica and Gelman 2010) as
the criterion.

"rule45": T = 4.5 sigma_max (Hoffman, Radul and Sountsov 2021, section 2.1: the first-moment optimum of
uniformly jittered HMC on a Gaussian), sigma_max the scale of the first principal component of the
states in mass-whitened coordinates (minibatch Oja and a running variance), jittered as "chees".

"ehmc": eHMC (Wu, Stoehr and Robert 2018; Wu, Pudlo, Robert and Stoehr 2025): each warmup transition
also records the first U-turn time of the trajectory from its start (their equation 4); warmup
trajectories are jittered up to twice the running mean of those times, and sampling draws each
trajectory time from their empirical distribution over the second half of warmup.

The step size and the mass matrix are adapted as without trajectory adaptation (numpyro's windowed
scheme on the chain-mean acceptance)."""
import jax
import jax.numpy as jnp
from jax import random
from jax.flatten_util import ravel_pytree
from numpyro.infer import hmc_util as hu
from numpyro.infer.hmc import momentum_generator

from nuts_turn import _bcss

MODES = ("chees", "snaper", "esjd", "rule45", "ehmc")
_LR = {"chees": 0.025, "snaper": 0.05, "esjd": 0.025}
_B2 = 0.95
_HOLD = {"chees": 0, "snaper": 100, "esjd": 0, "rule45": 0, "ehmc": 0}
_KAPPA = 8.0
_ETA_W = 8.0
_AVG = 0.1
_RULE = 4.5
_MAX_STEPS = 1000
_MAX_DE = 1000.0


def init(mode, D, eps0, dtype):
    lt = jnp.log(jnp.asarray(eps0, dtype))
    z = jnp.zeros((), dtype)
    return dict(logT=lt, v=z, k=z, Tavg=jnp.exp(lt), mu=jnp.zeros(D, dtype),
                w=jnp.ones(D, dtype) / jnp.sqrt(jnp.asarray(D, dtype)), n=z, lam=z, tmean=z)


def cap(mode, ta):
    return ta["Tavg"] * (2.0 if mode == "snaper" else 1.0)


def _halton2(i):
    def body(_, c):
        n, f, r = c
        f = f / 2.0
        return n // 2, f, r + f * (n % 2)

    return jax.lax.fori_loop(0, 31, body, (i, 1.0, 0.0))[2]


def _length(mode, s, T):
    if mode == "snaper":
        u = 1.0 - random.uniform(random.fold_in(s.rng_key, 11))
        return jnp.where(s.i < _HOLD["snaper"], s.adapt_state.step_size, 2.0 * T * u)
    return _halton2(s.i + 1) * T


def _flat_sqrt_imm(am):
    m = am.mass_matrix_sqrt_inv
    return ravel_pytree(list(m.values()) if isinstance(m, dict) else m)[0]


def transition(mode, s, T, pot):
    """One jittered static HMC transition (numpyro's momentum, kinetic energy, two-stage BCSS update,
    exact trajectory time and accept rule). Returns the new state and a dict with the start and proposal
    positions, the proposal velocity, the acceptance, the trajectory time, the mass-whitening scale and,
    for "ehmc", the first U-turn time from the start."""
    kinetic = hu.euclidean_kinetic_energy
    t = _length(mode, s, T)
    key, km, ka = random.split(s.rng_key, 3)
    am = s.adapt_state
    imm, eps = am.inverse_mass_matrix, am.step_size
    r = momentum_generator(s.z, am.mass_matrix_sqrt, km)
    vv0 = hu.IntegratorState(s.z, r, s.potential_energy, s.z_grad)
    n = jnp.clip(jnp.ceil(t / eps), 1, _MAX_STEPS)
    h = t / n
    _, upd = _bcss(pot, kinetic)
    vv1 = jax.lax.fori_loop(0, n.astype(jnp.int32), lambda _, v: upd(h, imm, v), vv0)
    e0 = s.potential_energy + kinetic(imm, r)
    e1 = vv1.potential_energy + kinetic(imm, vv1.r)
    de = jnp.where(jnp.isnan(e1 - e0), jnp.inf, e1 - e0)
    a = jnp.clip(jnp.exp(-de), None, 1.0)
    take = random.bernoulli(ka, a)
    vv, e = jax.tree.map(lambda x, y: jnp.where(take, x, y), (vv1, e1), (vv0, e0))
    vel = hu._kinetic_grad(kinetic, imm, vv1.r)
    z0f, z1f, vf = ravel_pytree(vv0.z)[0], ravel_pytree(vv1.z)[0], ravel_pytree(vel)[0]
    fin = jnp.all(jnp.isfinite(z1f)) & jnp.all(jnp.isfinite(vf))
    z1f, vf = jnp.where(fin, z1f, z0f), jnp.where(fin, vf, 0.0)
    tstar = jnp.zeros((), t.dtype)
    if mode == "ehmc":
        def cond(c):
            k, _, done = c
            return (~done) & (k < _MAX_STEPS)

        def body(c):
            k, w, _ = c
            w = upd(eps, imm, w)
            d = ravel_pytree(w.z)[0] - z0f
            u = ravel_pytree(hu._kinetic_grad(kinetic, imm, w.r))[0]
            return k + 1, w, ~(jnp.sum(d * u) >= 0)

        k, _, _ = jax.lax.while_loop(cond, body, (jnp.zeros((), jnp.int32), vv0, jnp.zeros((), bool)))
        tstar = k * eps
    s2 = s._replace(i=s.i + 1, z=vv.z, z_grad=vv.z_grad, potential_energy=vv.potential_energy,
                    energy=jnp.asarray(e, s.energy.dtype), trajectory_length=jnp.asarray(t, s.trajectory_length.dtype),
                    num_steps=n.astype(s.num_steps.dtype), accept_prob=jnp.asarray(a, s.accept_prob.dtype),
                    diverging=de > _MAX_DE, rng_key=key)
    return s2, dict(z0=z0f, z1=z1f, vel=vf, a=jnp.where(fin, a, 0.0), t=t, sc=_flat_sqrt_imm(am), tstar=tstar)


def _oja(ta, x, cm):
    dot = lambda p, q: jnp.sum(p * q, -1)
    eta = 1.0 / (jnp.ceil(ta["n"] / _KAPPA) + 1.0)
    mu = (1.0 - eta) * ta["mu"] + eta * cm(x)
    xc = x - mu
    w = ta["w"]
    wt = cm(xc * dot(xc, w)[..., None])
    wt = wt / (jnp.linalg.norm(wt, axis=-1, keepdims=True) + 1e-30)
    w = w + _ETA_W / (ta["n"] + 1.0) * wt
    w = w / jnp.linalg.norm(w, axis=-1, keepdims=True)
    return mu, w, xc, eta


def update(mode, ta, i, info, znew, cm):
    """`cm`: the cross-chain mean, broadcast back to each chain."""
    dot = lambda x, y: jnp.sum(x * y, -1)
    z0, z1, vel, a, t = info["z0"], info["z1"], info["vel"], info["a"], info["t"]
    out = dict(ta)
    if mode == "rule45":
        mu, w, xc, eta = _oja(ta, znew / info["sc"], cm)
        lam = (1.0 - eta) * ta["lam"] + eta * cm(dot(xc, w) ** 2)
        lt = jnp.log(_RULE * jnp.sqrt(jnp.maximum(lam, 1e-12)))
        lt = jnp.where(jnp.isfinite(lt) & (ta["n"] > 0), lt, ta["logT"])
        Tavg = (1.0 - _AVG) * ta["Tavg"] + _AVG * jnp.exp(lt)
        out.update(logT=lt, Tavg=Tavg, mu=mu, w=w, lam=lam, n=ta["n"] + 1.0)
        return out
    if mode == "ehmc":
        eta = 1.0 / (jnp.ceil(ta["n"] / _KAPPA) + 1.0)
        tm = (1.0 - eta) * ta["tmean"] + eta * cm(info["tstar"])
        lt = jnp.log(jnp.maximum(2.0 * tm, 1e-12))
        Tavg = (1.0 - _AVG) * ta["Tavg"] + _AVG * jnp.exp(lt)
        out.update(logT=lt, Tavg=Tavg, tmean=tm, n=ta["n"] + 1.0)
        return out
    if mode == "chees":
        d0, d1 = z0 - cm(z0), z1 - cm(z1)
        g = t * (dot(d1, d1) - dot(d0, d0)) * dot(d1, vel)
    elif mode == "esjd":
        g = 2.0 * t * dot(z1 - z0, vel)
    else:
        mu, w = ta["mu"], ta["w"]
        p0, p1 = dot(z0 - mu, w), dot(z1 - mu, w)
        A = p1 ** 2 - p0 ** 2
        g = 4.0 * A * p1 * dot(w, vel) - A ** 2 / t
        mu, w, _, _ = _oja(ta, znew, cm)
        out.update(mu=mu, w=w, n=ta["n"] + 1.0)
    g = jnp.where(jnp.isfinite(g) & (a > 0), g, 0.0)
    gbar = cm(a * g) / (cm(a) + 1e-30)
    on = (i >= _HOLD[mode]) & jnp.isfinite(gbar)
    k = ta["k"] + on
    v = jnp.where(on, _B2 * ta["v"] + (1.0 - _B2) * gbar ** 2, ta["v"])
    step = _LR[mode] * gbar / (jnp.sqrt(v / (1.0 - _B2 ** jnp.maximum(k, 1.0))) + 1e-8)
    lt = jnp.where(on, ta["logT"] + step, ta["logT"])
    Tavg = jnp.where(on, (1.0 - _AVG) * ta["Tavg"] + _AVG * jnp.exp(lt), ta["Tavg"])
    out.update(logT=lt, v=v, k=k, Tavg=Tavg)
    return out
