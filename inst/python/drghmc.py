"""DR-G-HMC sampler (Modi et al. 2024, arXiv:2406.02741). NUTS-free.

Self-adapting f64 donor -> f64/f32 sampling. The sampler only touches
value_and_grad(potential_fn), so every family's custom-JVP kernel is reused
unchanged. run_drghmc returns the same dict contract as fit_model's NUTS path.
"""
import os, sys, json, time, pickle, subprocess, tempfile, shutil
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import jax
import jax.numpy as jnp
from numpyro.infer.util import initialize_model
from numpyro.infer import init_to_uniform

_LAM = 0.19318332750378357
_B1, _A1 = 0.11888010966548, 0.29619504261126
_B2, _A2 = 0.5 - _B1, 1.0 - 2.0 * _A1
_SCHED10 = (0.15, 0.08, 0.08, 0.05, 0.05, 0.05, 0.05, 0.03, 0.02, 0.02)

_DEFAULTS = dict(acc_target=0.92, gamma=0.08, integrator="bcss3", dr_k=2,
                 dr_r=4.0, adapt_windows=10, eps_c=1.0,
                 mass_burn_frac=0.4, mass_end_frac=0.7, precision="f64")
_INTEGRATORS = ("bcss3", "bcss2", "leapfrog")


def _merge_control(control):
    c = dict(_DEFAULTS)
    if control:
        unknown = sorted(set(control) - set(_DEFAULTS))
        if unknown:
            raise ValueError("unknown DR-G-HMC control key(s): " + ", ".join(unknown)
                             + ". Valid keys: " + ", ".join(sorted(_DEFAULTS)))
        c.update({k: v for k, v in control.items() if v is not None})
    checks = [
        (0.0 < float(c["acc_target"]) < 1.0, "acc_target must be in (0, 1)"),
        (0.0 < float(c["gamma"]) <= 1.0, "gamma must be in (0, 1]"),
        (c["integrator"] in _INTEGRATORS, "integrator must be one of " + ", ".join(_INTEGRATORS)),
        (int(c["dr_k"]) == c["dr_k"] and int(c["dr_k"]) in (1, 2, 3), "dr_k must be 1, 2 or 3"),
        (float(c["dr_r"]) > 1.0, "dr_r must be > 1"),
        (int(c["adapt_windows"]) == c["adapt_windows"] and int(c["adapt_windows"]) >= 1,
         "adapt_windows must be a positive integer"),
        (float(c["eps_c"]) > 0.0, "eps_c must be > 0"),
        (0.0 <= float(c["mass_burn_frac"]) < float(c["mass_end_frac"]) <= 1.0,
         "need 0 <= mass_burn_frac < mass_end_frac <= 1"),
        (c["precision"] in ("f64", "f32"), "precision must be 'f64' or 'f32'"),
    ]
    for ok, msg in checks:
        if not ok:
            raise ValueError("DR-G-HMC control: " + msg)
    c["dr_k"] = int(c["dr_k"])
    c["adapt_windows"] = int(c["adapt_windows"])
    return c


def _root(seed):
    return jax.random.PRNGKey(int(seed))


def _tagged(seed, tag):
    return jax.random.fold_in(_root(seed), tag)


def _pbar(total, desc):
    try:
        from tqdm.auto import tqdm
        return tqdm(total=int(total), desc=desc)
    except Exception:
        return None


def _convert_matrices(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("X", "Z", "X_crit") and isinstance(v, list) and v and isinstance(v[0], list):
                obj[k] = [list(map(float, r)) for r in v]
            else:
                _convert_matrices(v)
    elif isinstance(obj, list):
        for it in obj:
            if isinstance(it, dict):
                _convert_matrices(it)


def _trim_config(config, n_keep=1):
    """A tiny (n_keep-row) copy of config. u_*/corr_*/params are data-independent,
    so constraining against this yields identical draws with no per-obs likelihood
    cost. Slices length-N arrays without copying the full data."""
    n = config.get("N")

    def t(o):
        if isinstance(o, dict):
            return {k: (v[:n_keep] if isinstance(v, list) and n is not None
                        and len(v) == n else t(v)) for k, v in o.items()}
        if isinstance(o, list):
            return [t(it) if isinstance(it, dict) else it for it in o]
        return o

    c = t(config)
    c["N"] = n_keep
    return c


def _fast_constrain_fn(model, config, full_pp, unflatten, ths):
    """Postprocess against a 1-row config (skips the likelihood recompute). Returns
    None -> caller falls back to full_pp unless the trimmed and full postprocess
    agree on the first, middle and last draw of every chain."""
    try:
        _, _, trim_pp = _model_info(model, _trim_config(config))
        full_j, trim_j = jax.jit(full_pp), jax.jit(trim_pp)
        chains, kept = ths.shape[:2]
        for c in range(chains):
            for i in sorted({0, kept // 2, kept - 1}):
                z = unflatten(jnp.asarray(ths[c, i]))
                a, b = full_j(z), trim_j(z)
                ka = {k for k in a if k != "log_lik"}
                if ka != set(b) - {"log_lik"}:
                    return None
                for k in ka:
                    if k.startswith("z_") or k.startswith("L_corr_"):
                        continue
                    if not np.allclose(np.asarray(a[k]), np.asarray(b[k]),
                                       rtol=1e-10, atol=1e-12):
                        return None
        return trim_pp
    except Exception:
        return None


def _model_info(model, config):
    """Unconstrained site shapes, potential and postprocess (constrained sites plus
    deterministics) from one abstract trace. numpyro's initialize_model gives the same
    three but runs the model eagerly, compiling every primitive separately."""
    from numpyro import handlers
    from numpyro.distributions.transforms import biject_to
    from numpyro.infer.util import potential_energy, constrain_fn

    supports = {}

    def run():
        init = handlers.substitute(handlers.seed(model, rng_seed=0),
                                   substitute_fn=init_to_uniform(radius=0.0))
        tr = handlers.trace(init).get_trace(config)
        out = {}
        for name, site in tr.items():
            if site["type"] == "sample" and not site["is_observed"]:
                supports[name] = site["fn"].support
                out[name] = site["value"]
        return out

    vals = jax.eval_shape(run)
    shapes = {k: tuple(biject_to(supports[k]).inv.forward_shape(vals[k].shape)) for k in vals}
    pot = lambda z: potential_energy(model, (config,), {}, z)
    post = lambda z: constrain_fn(model, (config,), {}, z, return_deterministic=True)
    return shapes, pot, post


def _build(model, config):
    """(vg, unflatten, keys, offsets, D, postprocess_fn) for flat unconstrained vectors."""
    shapes, potential_fn, postprocess_fn = _model_info(model, config)
    keys = sorted(shapes)
    off, o = {}, 0
    for k in keys:
        sz = int(np.prod(shapes[k]))
        off[k] = (slice(o, o + sz), shapes[k])
        o += sz
    D = o

    def unflatten(v):
        return {k: v[off[k][0]].reshape(off[k][1]) for k in keys}

    pot = lambda v: potential_fn(unflatten(v))
    return jax.value_and_grad(pot), unflatten, keys, off, D, postprocess_fn


def _build_offset(model, config, unflatten):
    """value_and_grad of the potential with a runtime per-observation log-likelihood
    offset (see ll_offset): the sampler takes it as an argument, so a compiled
    sampler serves any offset."""
    from numpyro.infer.util import potential_energy

    def pot(v, offset):
        return potential_energy(model, ({**config, "_ll_offset": offset},), {}, unflatten(v))
    return jax.value_and_grad(pot)


def ll_offset(postprocess_fn, unflatten, config, th):
    """Per-observation weighted log-likelihood at the unconstrained point `th`. Summing
    log_lik - offset keeps the potential near zero around `th`, so its float32 value
    resolves energy differences (a sum of ~|U| rounds to |U| * 6e-8, 0.03 at |U| = 5e5)."""
    ll = jnp.asarray(jax.jit(lambda v: postprocess_fn(unflatten(v))["log_lik"])(jnp.asarray(th)))
    counts = config.get("counts")
    return ll if counts is None else ll * jnp.asarray(counts, dtype=ll.dtype)


def _leap_flip(vg, integ, th, rho, g_th, eps, imm):
    """One reversible integrator step + momentum flip (involution)."""
    if integ == "bcss3":
        rho1 = rho - _B1 * eps * g_th
        th1 = th + _A1 * eps * imm * rho1
        _, g1 = vg(th1)
        rho2 = rho1 - _B2 * eps * g1
        th2 = th1 + _A2 * eps * imm * rho2
        _, g2 = vg(th2)
        rho3 = rho2 - _B2 * eps * g2
        th3 = th2 + _A1 * eps * imm * rho3
        U3, g3 = vg(th3)
        rho4 = rho3 - _B1 * eps * g3
        return th3, -rho4, U3, g3
    if integ == "bcss2":
        rho1 = rho - _LAM * eps * g_th
        th1 = th + 0.5 * eps * imm * rho1
        _, g1 = vg(th1)
        rho2 = rho1 - (1.0 - 2.0 * _LAM) * eps * g1
        th2 = th1 + 0.5 * eps * imm * rho2
        U2, g2 = vg(th2)
        rho3 = rho2 - _LAM * eps * g2
        return th2, -rho3, U2, g2
    rho_h = rho - 0.5 * eps * g_th
    th2 = th + eps * imm * rho_h
    U2, g2 = vg(th2)
    rho2 = rho_h - 0.5 * eps * g2
    return th2, -rho2, U2, g2


def _self_adapt(vg, D, chains, warmup, seed, ctrl, parallel, progress_bar=False,
                th0=None):
    """DR-native adaptation: per-chain RM on log(eps) -> acc_target, progressive
    diagonal mass from accumulated within-chain Welford variance. `warmup`
    iterations are split exactly across the adaptation windows. Returns
    (imm, eps, final positions, wall). th0 [chains, D] sets the starting points."""
    acc_t = ctrl["acc_target"]
    gamma = ctrl["gamma"]
    integ = ctrl["integrator"]
    sched = _SCHED10 if ctrl["adapt_windows"] == 10 else tuple(
        float(x) for x in np.geomspace(0.15, 0.02, ctrl["adapt_windows"]))
    H = len(sched)
    if warmup < H:
        raise ValueError(f"DR-G-HMC needs warmup >= adapt_windows ({H}); got {warmup}")
    lengths = [warmup // H + (1 if w < warmup % H else 0) for w in range(H)]
    _map = jax.pmap if parallel else jax.vmap

    def make_window(nw):
        def window(key, th0, rho0, U0, g0, le0, imm, lr):
            msq = jnp.sqrt(1.0 / imm)

            def step(c, k):
                th, rho, U, g, le, mean, m2, cnt = c
                k1, k2 = jax.random.split(k)
                xi = msq * jax.random.normal(k1, (D,))
                rho = jnp.sqrt(1.0 - gamma) * rho + jnp.sqrt(gamma) * xi
                eps = jnp.exp(le)
                th1, rho1, U1, g1 = _leap_flip(vg, integ, th, rho, g, eps, imm)
                H0 = U + 0.5 * jnp.sum(imm * rho * rho)
                H1 = U1 + 0.5 * jnp.sum(imm * rho1 * rho1)
                a = jnp.minimum(1.0, jnp.exp(H0 - H1))
                a = jnp.where(jnp.isfinite(a), a, 0.0)
                acc = jax.random.uniform(k2) < a
                th, U, g = jax.tree.map(
                    lambda x, y: jnp.where(acc, x, y), (th1, U1, g1), (th, U, g))
                rho = jnp.where(acc, -rho1, -rho)
                le = le + lr * (a - acc_t)
                cnt = cnt + 1.0
                d = th - mean
                mean = mean + d / cnt
                m2 = m2 + d * (th - mean)
                return (th, rho, U, g, le, mean, m2, cnt), a

            (thf, rhof, Uf, gf, lef, mean, m2, _), accs = jax.lax.scan(
                step, (th0, rho0, U0, g0, le0, jnp.zeros(D), jnp.zeros(D), 0.0),
                jax.random.split(key, nw))
            return thf, rhof, Uf, gf, lef, mean, m2, jnp.mean(accs)

        return _map(window, in_axes=(0, 0, 0, 0, 0, 0, None, None))

    pws = {n: make_window(n) for n in sorted(set(lengths))}
    t0 = time.time()
    th = jnp.zeros((chains, D)) if th0 is None else jnp.asarray(th0)
    U, g = _map(vg)(th)
    rho = _map(lambda k: jax.random.normal(k, (D,)))(
        jax.random.split(_tagged(seed, 4), chains))
    imm = jnp.ones(D)
    le = jnp.full((chains,), float(np.log(0.02)))
    acc_mean = np.zeros((chains, D)); acc_m2 = np.zeros((chains, D))
    acc_n = np.zeros(chains)
    burn = max(1, int(round(ctrl["mass_burn_frac"] * H)))
    mass_end = max(burn + 1, int(round(ctrl["mass_end_frac"] * H)))
    bar = _pbar(warmup, "DR-G-HMC adapt") if progress_bar else None
    for w, lr0 in enumerate(sched):
        nw = lengths[w]
        lr = (0.10 * 0.65 ** (w - mass_end)) if w >= mass_end else lr0
        keys = jax.random.split(jax.random.fold_in(_tagged(seed, 3), w), chains)
        th, rho, U, g, le, mn, m2, am = pws[nw](keys, th, rho, U, g, le, imm, lr)
        th = jax.block_until_ready(th)
        if bar is not None:
            bar.update(nw)
        if burn <= w < mass_end:
            mns, m2s = np.asarray(mn), np.asarray(m2)
            nwf = float(nw)
            delta = mns - acc_mean
            tot = acc_n[:, None] + nwf
            acc_mean = acc_mean + delta * (nwf / tot)
            acc_m2 = acc_m2 + m2s + delta ** 2 * (acc_n[:, None] * nwf / tot)
            acc_n = acc_n + nwf
            within = acc_m2.sum(0) / acc_n.sum()
            ne = float(acc_n.sum())
            imm = jnp.asarray((ne / (ne + 5.0)) * within + 1e-3 * (5.0 / (ne + 5.0)))
        if w < H - 1 and float(np.asarray(am).mean()) > 0.97:
            le = le + float(np.log(2.0))
    if bar is not None:
        bar.close()
    eps = float(np.exp(np.median(np.asarray(le))))
    return np.asarray(imm), eps, np.asarray(th), time.time() - t0


def _sampler_fns(vg_off, D, ctrl, parallel):
    """State-independent sampler functions (setup_p, chunk_p). (imm, eps) are
    runtime arguments, so chunk_p can be lowered/compiled before adaptation ends.
    The key for global step t of a chain is fold_in(chain_key, t), so a run split
    into rounds draws exactly the keys of one long run."""
    integ = ctrl["integrator"]
    kmax = int(ctrl["dr_k"])
    rfac = float(ctrl["dr_r"])
    gamma = ctrl["gamma"]
    RSC = jnp.asarray([(1.0 / rfac) ** k for k in range(3)])
    SQ1MG, SQG = jnp.sqrt(1.0 - gamma), jnp.sqrt(gamma)

    def scanner(c, ts, ckey, IMM, MSQ, E0, OFF):
        vg = lambda v: vg_off(v, OFF)
        eps_k = E0 * RSC
        tiny = jnp.finfo(c[0].dtype).tiny

        def kinetic(rho):
            return 0.5 * jnp.sum(IMM * rho * rho)

        def alpha1(th, rho, U_th, g_th):
            th1, rho1, U1, g1 = _leap_flip(vg, integ, th, rho, g_th, eps_k[0], IMM)
            H1 = U1 + kinetic(rho1)
            a1 = jnp.minimum(1.0, jnp.exp((U_th + kinetic(rho)) - H1))
            a1 = jnp.where(jnp.isfinite(a1), a1, 0.0)
            return a1, (th1, rho1, U1, g1)

        def dr_step(carry, t):
            th, rho, U_th, g_th = carry
            k1, k2 = jax.random.split(jax.random.fold_in(ckey, t))
            xi = MSQ * jax.random.normal(k1, (D,))
            rho = SQ1MG * rho + SQG * xi
            H0 = U_th + kinetic(rho)
            u = jax.random.uniform(k2, (3,))
            a1, (th1, rho1, U1, g1) = alpha1(th, rho, U_th, g_th)
            acc1 = u[0] < a1

            def stage2(_):
                th2, rho2, U2, g2 = _leap_flip(vg, integ, th, rho, g_th, eps_k[1], IMM)
                a1g, _ = alpha1(th2, rho2, U2, g2)
                H2 = U2 + kinetic(rho2)
                raw = jnp.exp(H0 - H2) * (1.0 - a1g) / jnp.maximum(1.0 - a1, tiny)
                return th2, rho2, U2, g2, jnp.where(jnp.isfinite(raw), jnp.minimum(1.0, raw), 0.0)

            def stage3(th2, rho2, U2, g2, a2):
                th3, rho3, U3, g3 = _leap_flip(vg, integ, th, rho, g_th, eps_k[2], IMM)
                a1g3, _ = alpha1(th3, rho3, U3, g3)
                th2g, rho2g, U2g, g2g = _leap_flip(vg, integ, th3, rho3, g3, eps_k[1], IMM)
                a1gg, _ = alpha1(th2g, rho2g, U2g, g2g)
                H3 = U3 + kinetic(rho3)
                H2g = U2g + kinetic(rho2g)
                raw2g = jnp.exp(H3 - H2g) * (1.0 - a1gg) / jnp.maximum(1.0 - a1g3, tiny)
                a2g = jnp.where(jnp.isfinite(raw2g), jnp.minimum(1.0, raw2g), 0.0)
                raw3 = (jnp.exp(H0 - H3)
                        * (1.0 - a1g3) / jnp.maximum(1.0 - a1, tiny)
                        * (1.0 - a2g) / jnp.maximum(1.0 - a2, tiny))
                return th3, rho3, U3, g3, jnp.where(jnp.isfinite(raw3), jnp.minimum(1.0, raw3), 0.0)

            def reject_all(_):
                return (th, -rho, U_th, g_th, -1.0)

            def rejected1(_):
                th2, rho2, U2, g2, a2 = stage2(None)
                acc2 = u[1] < a2

                def rejected2(_):
                    if kmax >= 3:
                        th3, rho3, U3, g3, a3 = stage3(th2, rho2, U2, g2, a2)
                        acc3 = u[2] < a3
                        return jax.tree.map(
                            lambda a, b: jnp.where(acc3, a, b),
                            (th3, rho3, U3, g3, 3.0), (th, -rho, U_th, g_th, -1.0))
                    return reject_all(None)

                def accepted2(_):
                    return (th2, rho2, U2, g2, 2.0)

                return jax.lax.cond(acc2, accepted2, rejected2, None)

            def accepted1(_):
                return (th1, rho1, U1, g1, 1.0)

            on_reject = reject_all if kmax == 1 else rejected1
            th_n, rho_n, U_n, g_n, stage = jax.lax.cond(acc1, accepted1, on_reject, None)
            rho_n = jnp.where(stage > 0, -rho_n, rho_n)
            return (th_n, rho_n, U_n, g_n), (th_n, stage, H0)

        return jax.lax.scan(dr_step, c, ts)

    def setup(th0, rho0, OFF):
        U0, g0 = vg_off(th0, OFF)
        return (th0, rho0, U0, g0)

    return (_chain_map(setup, (0, 0, None), parallel),
            _chain_map(scanner, (0, None, 0, None, None, None, None), parallel))


def _chain_map(f, in_axes, parallel):
    """pmap across devices, or one chain at a time under jit: lax.map keeps each chain's
    delayed-rejection lax.cond a real branch, where vmap would run every stage for all."""
    if parallel:
        return jax.pmap(f, in_axes=in_axes)
    mapped = [i for i, ax in enumerate(in_axes) if ax == 0]

    def run(*args):
        def one(xs):
            full = list(args)
            for i, x in zip(mapped, xs):
                full[i] = x
            return f(*full)
        return jax.lax.map(one, tuple(args[i] for i in mapped))
    return jax.jit(run)


def _chunk_len(samples):
    return max(1, samples // 100)


def _sample_phase(vg_off, D, chains, samples, thinning, seed, ctrl,
                  imm, eps, state, parallel, progress_bar=False,
                  donor_loader=None, prebuilt=None, n_obs=None, offset_fn=None):
    """DR-G-HMC sampling from a fixed (imm, eps). `state` holds the chains'
    positions `th` [chains, D], optional momenta `rho`, and the global step
    offset `t0` (0 for a fresh run). Returns (thinned draws [chains, kept, D],
    stages, energy, end state)."""
    step = _chunk_len(samples)
    if prebuilt is not None:
        setup_p, chunk_c, chunk_raw = prebuilt
    else:
        setup_p, chunk_raw = _sampler_fns(vg_off, D, ctrl, parallel)
        chunk_c = chunk_raw
    chain_keys = jax.random.split(_tagged(seed, 1), chains)
    zero_off = jnp.zeros(n_obs)

    if donor_loader is not None:
        dummy = jnp.zeros((chains, D))
        c0 = jax.tree.map(np.asarray, setup_p(dummy, dummy, zero_off))
        cw = chunk_c(c0, jnp.arange(step), chain_keys, jnp.ones(D), jnp.ones(D), jnp.asarray(1e-3),
                     zero_off)
        jax.block_until_ready(cw)
        del c0, cw, dummy
        imm, eps, state = donor_loader()
    OFF = zero_off if offset_fn is None else offset_fn(state)

    IMM = jnp.asarray(imm)
    MSQ = jnp.sqrt(1.0 / IMM)
    E0 = jnp.asarray(ctrl["eps_c"] * float(eps))
    th0 = jnp.asarray(state["th"])
    rho0 = state.get("rho")
    if rho0 is None:
        rho0 = MSQ * jax.random.normal(_tagged(seed, 2), (chains, D))
    t0 = int(state.get("t0", 0))
    carry = jax.tree.map(np.asarray, setup_p(th0, jnp.asarray(rho0), OFF))

    bar = _pbar(samples, "DR-G-HMC sample") if progress_bar else None
    tp, sp, ep = [], [], []
    for i in range(t0, t0 + samples, step):
        ts = jnp.arange(i, min(i + step, t0 + samples))
        fn = chunk_c if ts.shape[0] == step else chunk_raw
        carry, (t, s, e) = fn(carry, ts, chain_keys, IMM, MSQ, E0, OFF)
        carry = jax.tree.map(np.asarray, carry)
        tp.append(np.asarray(t)); sp.append(np.asarray(s)); ep.append(np.asarray(e))
        if bar is not None:
            bar.update(int(ts.shape[0]))
    if bar is not None:
        bar.close()
    sl = slice(None, None, thinning)
    end = dict(th=np.asarray(carry[0]), rho=np.asarray(carry[1]), t0=t0 + samples,
               offset=None if offset_fn is None else np.asarray(OFF))
    return (np.concatenate(tp, 1)[:, sl], np.concatenate(sp, 1)[:, sl],
            np.concatenate(ep, 1)[:, sl], end)


def _postprocess_chunked(postprocess_fn, unflatten, ths, save_loglik=True,
                         keep_re=False, progress_bar=False, chunk=500):
    """Constrain flat draws -> named sites (+ deterministics). Chains constrain
    in parallel across devices (pmap) when the device count allows; draw-chunks
    stream into preallocated outputs to bound peak memory. Drops raw
    z_* unless keep_re, to match NUTS get_samples; drops log_lik (the largest array by
    far) when it is not needed."""
    chains, kept, _ = ths.shape
    named1 = jax.vmap(lambda z: postprocess_fn(unflatten(z)))
    par = chains > 1 and jax.local_device_count() >= chains
    namedC = jax.pmap(named1) if par else None
    n_chunks = (kept + chunk - 1) // chunk
    bar = _pbar(n_chunks if par else chains * n_chunks,
                "DR-G-HMC constrain") if progress_bar else None

    def _keep(k):
        if k.startswith("z_") and not keep_re:
            return False
        return not (k == "log_lik" and not save_loglik)

    out = {}
    if par:
        for i in range(0, kept, chunk):
            block = jax.block_until_ready(namedC(jnp.asarray(ths[:, i:i + chunk])))
            for k, v in block.items():
                if not _keep(k):
                    continue
                v = np.asarray(v)
                if k not in out:
                    out[k] = np.empty((chains, kept) + v.shape[2:], dtype=v.dtype)
                out[k][:, i:i + v.shape[1]] = v
            if bar is not None:
                bar.update(1)
    else:
        for c in range(chains):
            for i in range(0, kept, chunk):
                block = jax.block_until_ready(named1(jnp.asarray(ths[c, i:i + chunk])))
                for k, v in block.items():
                    if not _keep(k):
                        continue
                    v = np.asarray(v)
                    if k not in out:
                        out[k] = np.empty((chains, kept) + v.shape[1:], dtype=v.dtype)
                    out[k][c, i:i + v.shape[0]] = v
                if bar is not None:
                    bar.update(1)
    if bar is not None:
        bar.close()
    return out


def _f32_spawn(config, chains, samples, thinning, seed, ctrl, parallel,
               backend_dir, progress_bar=False):
    """Launch the x64-disabled f32 sampling process before adaptation so its
    import and XLA compile overlap the parent's f64 adaptation. The child
    compiles against dummy state, then blocks until the donor file appears.
    Returns (proc, tmp, donor_path, out_path)."""
    tmp = tempfile.mkdtemp(prefix="drghmc_f32_")
    donor = os.path.join(tmp, "donor.pkl")
    cfg = os.path.join(tmp, "config.json")
    out = os.path.join(tmp, "out.npz")
    with open(cfg, "w") as fh:
        json.dump(config, fh)
    payload = dict(chains=chains, samples=samples, thinning=thinning, seed=seed,
                   ctrl=ctrl, parallel=parallel, donor=donor, config=cfg, out=out,
                   backend_dir=backend_dir, progress_bar=progress_bar)
    argfile = os.path.join(tmp, "args.json")
    with open(argfile, "w") as fh:
        json.dump(payload, fh)
    env = dict(os.environ)
    env["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={chains if parallel else 1}"
    env["BAYESROC_ENABLE_X64"] = "0"
    proc = subprocess.Popen([sys.executable, os.path.abspath(__file__),
                             "_sample_f32", argfile], env=env)
    return proc, tmp, donor, out


def _f32_send_donor_and_collect(proc, tmp, donor, out, imm, eps, state):
    """Hand the adapted (imm, eps, state) to the waiting f32 child (atomic rename
    so the child never reads a partial pickle), then collect results."""
    try:
        with open(donor + ".tmp", "wb") as fh:
            pickle.dump(dict(imm=np.asarray(imm), eps=float(eps),
                             state={k: (np.asarray(v) if v is not None and k != "t0" else v)
                                    for k, v in state.items()}), fh)
        os.replace(donor + ".tmp", donor)
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"f32 sampling subprocess failed (exit {rc})")
        d = np.load(out)
        end = dict(th=d["end_th"], rho=d["end_rho"], t0=int(d["end_t0"]), offset=d["offset"])
        return d["ths"], d["stages"], d["energy"], end
    finally:
        if proc.poll() is None:
            proc.kill()
        shutil.rmtree(tmp, ignore_errors=True)


def _initial_positions(model, config, keys, off, D, chains, seed, init_strategy,
                       init_radius, init_z=None):
    """Per-chain starting points on the unconstrained scale, matching the NUTS
    init strategies: user values (`init_z`), 'prior' draws from the prior,
    otherwise uniform(-r, r)."""
    base = _tagged(seed, 5)
    if init_z is not None:
        th = np.zeros((chains, D))
        for k in keys:
            th[:, off[k][0]] = np.asarray(init_z[k]).reshape(chains, -1)
        return th
    if init_strategy == "prior":
        from numpyro.infer import init_to_sample
        th = np.zeros((chains, D))
        for c in range(chains):
            st = initialize_model(jax.random.fold_in(base, c), model,
                                  init_strategy=init_to_sample, model_args=(config,))
            z = st.param_info.z
            for k in keys:
                th[c, off[k][0]] = np.asarray(z[k]).reshape(-1)
        return th
    r = float(init_radius)
    if r == 0.0:
        return np.zeros((chains, D))
    return np.asarray(jax.random.uniform(base, (chains, D), minval=-r, maxval=r))


def run_drghmc(model, config, chains=4, warmup=2500, samples=10000, seed=0,
               thinning=1, chain_method="parallel", progress_bar=True,
               init_strategy="numeric", init_radius=0.0, control=None,
               save_loglik=True, keep_re=False, save_state=False, init_z=None):
    """With `save_state` the return carries the end-of-run `state` (positions, momenta,
    step offset, adaptation), from which autorun continues the same chains exactly."""
    ctrl = _merge_control(control)
    parallel = chain_method == "parallel"
    _convert_matrices(config)
    t0 = time.time()

    f32_proc = None
    if ctrl["precision"] == "f32":
        if progress_bar:
            print("[DR-G-HMC] launching f32 sampling process (compiles during "
                  "adaptation)...", flush=True)
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        f32_proc, f32_tmp, f32_donor, f32_out = _f32_spawn(
            config, chains, samples, thinning, seed, ctrl, parallel,
            backend_dir, progress_bar)

    if progress_bar:
        print("[DR-G-HMC] building model + compiling adaptation...", flush=True)
    try:
        vg, unflatten, keys, off, D, postprocess_fn = _build(model, config)

        smp_thread = None
        if ctrl["precision"] != "f32":
            import threading
            setup_p, chunk_raw = _sampler_fns(_build_offset(model, config, unflatten), D, ctrl, parallel)
            step = _chunk_len(samples)
            _ck = jax.random.split(_tagged(seed, 1), chains)
            _th0 = jnp.zeros((chains, D))
            _c0 = (_th0, _th0, jnp.zeros(chains), _th0)
            _off0 = jnp.zeros(int(config["N"]))
            low_setup = setup_p.lower(_th0, _th0, _off0)
            low_chunk = chunk_raw.lower(_c0, jnp.arange(step), _ck, jnp.ones(D), jnp.ones(D),
                                        jnp.asarray(1e-3), _off0)
            _res = {}

            def _worker():
                try:
                    _res["setup"] = low_setup.compile()
                    _res["chunk"] = low_chunk.compile()
                except BaseException as e:
                    _res["err"] = e

            if progress_bar:
                print("[DR-G-HMC] compiling sampler (overlapped with "
                      "adaptation)...", flush=True)
            smp_thread = threading.Thread(target=_worker, daemon=True)
            smp_thread.start()

        th_init = _initial_positions(model, config, keys, off, D, chains, seed,
                                     init_strategy, init_radius, init_z)
        imm, eps, th_adapted, wall_warm = _self_adapt(
            vg, D, chains, warmup, seed, ctrl, parallel, progress_bar,
            th0=th_init)
        state = dict(th=th_adapted, rho=None, t0=0)
    except Exception:
        if f32_proc is not None:
            f32_proc.kill()
            shutil.rmtree(f32_tmp, ignore_errors=True)
        raise

    if ctrl["precision"] == "f32":
        ths, stages, energy, end = _f32_send_donor_and_collect(
            f32_proc, f32_tmp, f32_donor, f32_out, imm, eps, state)
        ths = np.asarray(ths, dtype=np.float64)
    else:
        if smp_thread is not None:
            smp_thread.join()
        if "err" in _res:
            raise _res["err"]
        ths, stages, energy, end = _sample_phase(
            None, D, chains, samples, thinning, seed, ctrl, imm, eps, state,
            parallel, progress_bar,
            prebuilt=(_res["setup"], _res["chunk"], chunk_raw), n_obs=int(config["N"]))

    pp_fn = postprocess_fn
    if not save_loglik:
        fast = _fast_constrain_fn(model, config, postprocess_fn, unflatten, ths)
        if fast is not None:
            pp_fn = fast
    if progress_bar:
        note = (" (log_lik retained; large)" if save_loglik
                else " (log_lik skipped)" if pp_fn is not postprocess_fn
                else " (log_lik skipped; full recompute)")
        print("[DR-G-HMC] constraining draws" + note + "...", flush=True)
    samples_np = _postprocess_chunked(pp_fn, unflatten, ths,
                                      save_loglik=save_loglik,
                                      keep_re=keep_re,
                                      progress_bar=progress_bar)
    elapsed = time.time() - t0

    extra_fields = dict(energy=np.asarray(energy),
                        accepted=(stages > 0).astype(np.float64),
                        dr_stage=np.asarray(stages))
    if progress_bar:
        fr = [float((stages == v).mean()) for v in (1.0, 2.0, 3.0, -1.0)]
        print(f"[DR-G-HMC] {chains}x{samples} adapt {wall_warm:.1f}s "
              f"stage 1/2/3/rej {fr[0]:.2f}/{fr[1]:.2f}/{fr[2]:.2f}/{fr[3]:.2f}", flush=True)
    state_out = None
    if save_state:
        state_out = dict(imm=np.asarray(imm), eps=float(eps), th=end["th"], rho=end["rho"],
                         t0=int(end["t0"]), seed=int(seed), offset=end.get("offset"))
    return {"samples": samples_np, "extra_fields": extra_fields,
            "elapsed": float(elapsed), "state": state_out}


def _main_sample_f32(argfile):
    with open(argfile) as fh:
        p = json.load(fh)
    if p["backend_dir"]:
        sys.path.insert(0, p["backend_dir"])
    import numpyro_backend as backend
    with open(p["config"]) as fh:
        config = json.load(fh)
    _convert_matrices(config)
    vg, unflatten, keys, off, D, postprocess_fn = _build(backend.sdt_model, config)
    vg_off = _build_offset(backend.sdt_model, config, unflatten)

    def load_donor(timeout=3600.0):
        t0 = time.time()
        while not os.path.exists(p["donor"]):
            if time.time() - t0 > timeout:
                raise TimeoutError("f32 child: donor never arrived")
            time.sleep(0.2)
        with open(p["donor"], "rb") as fh:
            don = pickle.load(fh)
        return don["imm"], don["eps"], don["state"]

    ths, stages, energy, end = _sample_phase(
        vg_off, D, p["chains"], p["samples"], p["thinning"], p["seed"], p["ctrl"],
        None, None, None, p["parallel"], p.get("progress_bar", False),
        donor_loader=load_donor, n_obs=int(config["N"]),
        offset_fn=lambda st: (jnp.asarray(st["offset"]) if st.get("offset") is not None else
                              ll_offset(postprocess_fn, unflatten, config, np.asarray(st["th"])[0])))
    np.savez(p["out"], ths=ths, stages=stages, energy=energy,
             end_th=end["th"], end_rho=end["rho"], end_t0=end["t0"], offset=end["offset"])


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "_sample_f32":
        import jax as _j
        _j.config.update("jax_enable_x64", False)
        _main_sample_f32(sys.argv[2])
