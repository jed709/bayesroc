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
                 dr_r=4.0, adapt_windows=10, eps_c=1.0, mass_start=1,
                 mass_burn_frac=0.4, mass_end_frac=0.7, precision="f64")


def _merge_control(control):
    c = dict(_DEFAULTS)
    if control:
        c.update({k: v for k, v in control.items() if v is not None})
    return c


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


def _fast_constrain_fn(config, full_pp, unflatten, ths):
    """Postprocess against a 1-row config (skips the likelihood recompute). Returns
    None -> caller falls back to full_pp if trimmed draws don't match on a probe."""
    try:
        st = initialize_model(jax.random.PRNGKey(0), _MODEL,
                              init_strategy=init_to_uniform(radius=0.0),
                              model_args=(_trim_config(config),))
        z = unflatten(jnp.asarray(ths[0, 0]))
        a, b = full_pp(z), st.postprocess_fn(z)
        for k in a:
            if k == "log_lik" or k.startswith("z_") or k.startswith("L_corr_"):
                continue
            if not np.allclose(np.asarray(a[k]), np.asarray(b[k]), atol=1e-6):
                return None
        return st.postprocess_fn
    except Exception:
        return None


def _build(config):
    """initialize_model -> (vg, unflatten, keys, offsets, D, postprocess_fn).

    Init is pinned to zeros: the returned init values are never used, and
    numpyro's default random init search can abort on non-finite potentials
    under float32."""
    st = initialize_model(jax.random.PRNGKey(0), _MODEL,
                          init_strategy=init_to_uniform(radius=0.0),
                          model_args=(config,))
    z = st.param_info.z
    keys = sorted(z.keys())
    off, o = {}, 0
    for k in keys:
        sz = int(np.prod(z[k].shape))
        off[k] = (slice(o, o + sz), z[k].shape)
        o += sz
    D = o

    def unflatten(v):
        return {k: v[off[k][0]].reshape(off[k][1]) for k in keys}

    pot = lambda v: st.potential_fn(unflatten(v))
    return jax.value_and_grad(pot), unflatten, keys, off, D, st.postprocess_fn


_MODEL = None


def _set_model(model):
    global _MODEL
    _MODEL = model


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
    diagonal mass from accumulated within-chain Welford variance. Returns
    (imm, eps, inits, wall). th0 [chains, D] sets the starting points
    (default: all zeros on the unconstrained scale, fit_broc's init = 0)."""
    acc_t = ctrl["acc_target"]
    gamma = ctrl["gamma"]
    integ = ctrl["integrator"]
    sched = _SCHED10 if ctrl["adapt_windows"] == 10 else tuple(
        float(x) for x in np.geomspace(0.15, 0.02, ctrl["adapt_windows"]))
    nw = max(1, warmup // len(sched))
    _map = jax.pmap if parallel else jax.vmap

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

    pw = _map(window, in_axes=(0, 0, 0, 0, 0, 0, None, None))
    t0 = time.time()
    th = jnp.zeros((chains, D)) if th0 is None else jnp.asarray(th0)
    U, g = _map(vg)(th)
    rho = _map(lambda k: jax.random.normal(k, (D,)))(
        jax.random.split(jax.random.PRNGKey(seed + 23), chains))
    imm = jnp.ones(D)
    le = jnp.full((chains,), float(np.log(0.02)))
    acc_mean = np.zeros((chains, D)); acc_m2 = np.zeros((chains, D))
    acc_n = np.zeros(chains)
    H = len(sched)
    burn = max(1, int(round(ctrl["mass_burn_frac"] * H)))
    mass_end = max(burn + 1, int(round(ctrl["mass_end_frac"] * H)))
    imm_frozen = None
    bar = _pbar(H * nw, "DR-G-HMC adapt") if progress_bar else None
    for w, lr0 in enumerate(sched):
        lr = (0.10 * 0.65 ** (w - mass_end)) if w >= mass_end else lr0
        keys = jax.random.split(jax.random.PRNGKey(seed + 100 + w), chains)
        th, rho, U, g, le, mn, m2, am = pw(keys, th, rho, U, g, le, imm, lr)
        th = jax.block_until_ready(th)
        if bar is not None:
            bar.update(nw)
        if w == burn:
            acc_mean[:] = 0.0; acc_m2[:] = 0.0; acc_n[:] = 0.0
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
            imm_frozen = imm
        if w < H - 1 and float(np.asarray(am).mean()) > 0.97:
            le = le + float(np.log(2.0))
    if imm_frozen is not None:
        imm = imm_frozen
    if bar is not None:
        bar.close()
    eps = float(np.exp(np.median(np.asarray(le))))
    return np.asarray(imm), eps, np.asarray(th), time.time() - t0


def _sampler_fns(vg, D, chains, samples, ctrl, parallel):
    """Build the state-independent sampler executables: (setup_p, chunk_p,
    step). (imm, eps) enter as runtime arguments, not baked constants, so
    these can be lowered/compiled before adaptation finishes."""
    integ = ctrl["integrator"]
    kmax = int(ctrl["dr_k"])
    rfac = float(ctrl["dr_r"])
    gamma = ctrl["gamma"]
    kk = max(kmax, 3)
    RSC = jnp.asarray([(1.0 / rfac) ** k for k in range(kk)])
    SQ1MG, SQG = jnp.sqrt(1.0 - gamma), jnp.sqrt(gamma)

    def scanner(c, keys, IMM, MSQ, E0):
        eps_k = E0 * RSC

        def kinetic(rho):
            return 0.5 * jnp.sum(IMM * rho * rho)

        def alpha1(th, rho, U_th, g_th):
            th1, rho1, U1, g1 = _leap_flip(vg, integ, th, rho, g_th, eps_k[0], IMM)
            a1 = jnp.minimum(1.0, jnp.exp((U_th + kinetic(rho)) - (U1 + kinetic(rho1))))
            return a1, (th1, rho1, U1, g1)

        def dr_step(carry, key):
            th, rho, U_th, g_th = carry
            k1, k2 = jax.random.split(key)
            xi = MSQ * jax.random.normal(k1, (D,))
            rho = SQ1MG * rho + SQG * xi
            H0 = U_th + kinetic(rho)
            u = jax.random.uniform(k2, (kk,))
            a1, (th1, rho1, U1, g1) = alpha1(th, rho, U_th, g_th)
            acc1 = u[0] < a1

            def stage2(_):
                th2, rho2, U2, g2 = _leap_flip(vg, integ, th, rho, g_th, eps_k[1], IMM)
                a1g, _ = alpha1(th2, rho2, U2, g2)
                H2 = U2 + kinetic(rho2)
                raw = jnp.exp(H0 - H2) * (1.0 - a1g) / jnp.maximum(1.0 - a1, 1e-300)
                return th2, rho2, U2, g2, jnp.minimum(1.0, raw)

            def stage3(th2, rho2, U2, g2, a2):
                th3, rho3, U3, g3 = _leap_flip(vg, integ, th, rho, g_th, eps_k[2], IMM)
                a1g3, _ = alpha1(th3, rho3, U3, g3)
                th2g, rho2g, U2g, g2g = _leap_flip(vg, integ, th3, rho3, g3, eps_k[1], IMM)
                a1gg, _ = alpha1(th2g, rho2g, U2g, g2g)
                H3 = U3 + kinetic(rho3)
                H2g = U2g + kinetic(rho2g)
                raw2g = jnp.exp(H3 - H2g) * (1.0 - a1gg) / jnp.maximum(1.0 - a1g3, 1e-300)
                a2g = jnp.minimum(1.0, raw2g)
                raw3 = (jnp.exp(H0 - H3)
                        * (1.0 - a1g3) / jnp.maximum(1.0 - a1, 1e-300)
                        * (1.0 - a2g) / jnp.maximum(1.0 - a2, 1e-300))
                return th3, rho3, U3, g3, jnp.minimum(1.0, raw3)

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
                    return (th, -rho, U_th, g_th, -1.0)

                def accepted2(_):
                    return (th2, rho2, U2, g2, 2.0)

                return jax.lax.cond(acc2, accepted2, rejected2, None)

            def accepted1(_):
                return (th1, rho1, U1, g1, 1.0)

            th_n, rho_n, U_n, g_n, stage = jax.lax.cond(acc1, accepted1, rejected1, None)
            rho_n = jnp.where(stage > 0, -rho_n, rho_n)
            return (th_n, rho_n, U_n, g_n), (th_n, stage, H0)

        return jax.lax.scan(dr_step, c, keys)

    def setup(key, th0, MSQ):
        U0, g0 = vg(th0)
        k1, k2 = jax.random.split(key)
        rho0 = MSQ * jax.random.normal(k1, (D,))
        return (th0, rho0, U0, g0), jax.random.split(k2, samples)

    _map = jax.pmap if parallel else jax.vmap
    return (_map(setup, in_axes=(0, 0, None)),
            _map(scanner, in_axes=(0, 0, None, None, None)),
            max(1, samples // 100))


def _sample_phase(vg, D, chains, samples, thinning, seed, ctrl,
                  imm, eps, inits, parallel, progress_bar=False,
                  donor_loader=None, prebuilt=None):
    """DR-G-HMC sampling from a fixed (imm, eps, inits). Returns flat draws
    [chains, kept, D], stages [chains, kept], energy [chains, kept], and the
    final unthinned positions [chains, D].

    With `donor_loader` (f32 subprocess), the sampler compiles against dummy
    state first, then blocks on donor_loader() for the real state. With
    `prebuilt` (in-process overlap), already-compiled executables from
    _sampler_fns lower/compile are used directly."""
    if prebuilt is not None:
        setup_p, chunk_p, step = prebuilt
    else:
        setup_p, chunk_p, step = _sampler_fns(vg, D, chains, samples, ctrl,
                                              parallel)

    if donor_loader is not None:
        dummy_th = jnp.zeros((chains, D))
        dummy_keys = jax.vmap(lambda k: jax.random.split(k, samples))(
            jax.random.split(jax.random.PRNGKey(0), chains))
        c0, _ = setup_p(jax.random.split(jax.random.PRNGKey(0), chains),
                        dummy_th, jnp.ones(D))
        cw, _ow = chunk_p(c0, dummy_keys[:, :step], jnp.ones(D), jnp.ones(D),
                          jnp.asarray(1e-3))
        jax.block_until_ready(cw)
        del c0, cw, _ow, dummy_th, dummy_keys
        imm, eps, inits = donor_loader()

    IMM = jnp.asarray(imm)
    MSQ = jnp.sqrt(1.0 / IMM)
    E0 = jnp.asarray(ctrl["eps_c"] * float(eps))

    carry, allkeys = setup_p(
        jax.random.split(jax.random.PRNGKey(seed + 7), chains),
        jnp.asarray(inits), MSQ)

    bar = _pbar(samples, "DR-G-HMC sample") if progress_bar else None
    tp, sp, ep = [], [], []
    for i in range(0, samples, step):
        kc = allkeys[:, i:i + step]
        carry, (t, s, e) = chunk_p(carry, kc, IMM, MSQ, E0)
        carry = jax.block_until_ready(carry)
        tp.append(np.asarray(t)); sp.append(np.asarray(s)); ep.append(np.asarray(e))
        if bar is not None:
            bar.update(kc.shape[1])
    if bar is not None:
        bar.close()
    sl = slice(None, None, thinning)
    return (np.concatenate(tp, 1)[:, sl], np.concatenate(sp, 1)[:, sl],
            np.concatenate(ep, 1)[:, sl], np.asarray(carry[0]))


def _postprocess_chunked(postprocess_fn, unflatten, ths, save_loglik=True,
                         keep_re=False, progress_bar=False, chunk=500):
    """Constrain flat draws -> named sites (+ deterministics). Chains constrain
    in parallel across devices (pmap) when the device count allows; draw-chunks
    stream into preallocated outputs to bound peak memory. Drops raw
    z_*/L_corr_* to match NUTS get_samples; drops log_lik (the largest array by
    far) when it is not needed."""
    chains, kept, _ = ths.shape
    named1 = jax.vmap(lambda z: postprocess_fn(unflatten(z)))
    par = chains > 1 and jax.local_device_count() >= chains
    namedC = jax.pmap(named1) if par else None
    n_chunks = (kept + chunk - 1) // chunk
    bar = _pbar(n_chunks if par else chains * n_chunks,
                "DR-G-HMC constrain") if progress_bar else None

    def _keep(k):
        if k.startswith("L_corr_"):
            return False
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


def _f32_send_donor_and_collect(proc, tmp, donor, out, imm, eps, inits):
    """Hand the adapted (imm, eps, inits) to the waiting f32 child (atomic
    rename so the child never reads a partial pickle), then collect results."""
    try:
        with open(donor + ".tmp", "wb") as fh:
            pickle.dump(dict(imm=np.asarray(imm), eps=float(eps),
                             inits=np.asarray(inits)), fh)
        os.replace(donor + ".tmp", donor)
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"f32 sampling subprocess failed (exit {rc})")
        d = np.load(out)
        return d["ths"], d["stages"], d["energy"], d["final_th"]
    finally:
        if proc.poll() is None:
            proc.kill()
        shutil.rmtree(tmp, ignore_errors=True)


def run_drghmc(config, chains=4, warmup=2500, samples=10000, seed=0,
               thinning=1, chain_method="parallel", progress_bar=True,
               init_strategy="numeric", init_radius=0.0, control=None,
               save_loglik=True, keep_re=False, resume_state=None,
               save_state=False):
    """With `resume_state` (imm/eps/inits from a prior round) adaptation is
    skipped and the same chains continue from the saved unthinned positions.
    With `save_state` the return carries `state` for the next round."""
    ctrl = _merge_control(control)
    warm = resume_state is not None
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
        vg, unflatten, keys, off, D, postprocess_fn = _build(config)

        th_init = None
        if init_strategy == "prior" or float(init_radius) != 0.0:
            print("[DR-G-HMC] WARNING: init strategies other than 0 are not "
                  "supported by the DR-G-HMC adaptation; starting from 0 on "
                  "the unconstrained scale (use sampler='nuts' if a custom "
                  "init is required).", flush=True)

        smp_thread = None
        if ctrl["precision"] != "f32":
            import threading
            setup_p, chunk_p, step = _sampler_fns(vg, D, chains, samples,
                                                  ctrl, parallel)
            _k0 = jax.random.split(jax.random.PRNGKey(0), chains)
            _th0 = jnp.zeros((chains, D))
            _keys = jax.vmap(lambda k: jax.random.split(k, step))(_k0)
            _c0 = (_th0, _th0, jnp.zeros(chains), _th0)
            low_setup = setup_p.lower(_k0, _th0, jnp.ones(D))
            low_chunk = chunk_p.lower(_c0, _keys, jnp.ones(D), jnp.ones(D),
                                      jnp.asarray(1e-3))
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

        if warm:
            imm = jnp.asarray(resume_state["imm"])
            eps = float(resume_state["eps"])
            inits = np.asarray(resume_state["inits"])
            wall_warm = 0.0
        else:
            imm, eps, inits, wall_warm = _self_adapt(
                vg, D, chains, warmup, seed, ctrl, parallel, progress_bar,
                th0=th_init)
    except Exception:
        if f32_proc is not None:
            f32_proc.kill()
            shutil.rmtree(f32_tmp, ignore_errors=True)
        raise

    if ctrl["precision"] == "f32":
        ths, stages, energy, final_th = _f32_send_donor_and_collect(
            f32_proc, f32_tmp, f32_donor, f32_out, imm, eps, inits)
        ths = np.asarray(ths, dtype=np.float64)
    else:
        smp_thread.join()
        if "err" in _res:
            raise _res["err"]
        ths, stages, energy, final_th = _sample_phase(
            vg, D, chains, samples, thinning, seed, ctrl, imm, eps, inits,
            parallel, progress_bar,
            prebuilt=(_res["setup"], _res["chunk"], step))

    pp_fn = postprocess_fn
    if not save_loglik:
        fast = _fast_constrain_fn(config, postprocess_fn, unflatten, ths)
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

    accept_prob = (stages > 0).astype(np.float64)
    extra_fields = dict(energy=np.asarray(energy),
                        accept_prob=accept_prob,
                        dr_stage=np.asarray(stages))
    if progress_bar:
        fr = [float((stages == v).mean()) for v in (1.0, 2.0, 3.0, -1.0)]
        print(f"[DR-G-HMC] {chains}x{samples} adapt {wall_warm:.1f}s "
              f"stage 1/2/3/rej {fr[0]:.2f}/{fr[1]:.2f}/{fr[2]:.2f}/{fr[3]:.2f}",
              flush=True)
    state_out = None
    if save_state and final_th is not None:
        state_out = dict(imm=np.asarray(imm), eps=float(eps),
                         inits=np.asarray(final_th))
    return {"samples": samples_np, "extra_fields": extra_fields,
            "elapsed": float(elapsed), "state": state_out}


def _main_sample_f32(argfile):
    with open(argfile) as fh:
        p = json.load(fh)
    if p["backend_dir"]:
        sys.path.insert(0, p["backend_dir"])
    import numpyro_backend as backend
    _set_model(backend.sdt_model)
    with open(p["config"]) as fh:
        config = json.load(fh)
    _convert_matrices(config)
    vg, unflatten, keys, off, D, _ = _build(config)

    def load_donor(timeout=3600.0):
        t0 = time.time()
        while not os.path.exists(p["donor"]):
            if time.time() - t0 > timeout:
                raise TimeoutError("f32 child: donor never arrived")
            time.sleep(0.2)
        with open(p["donor"], "rb") as fh:
            don = pickle.load(fh)
        return don["imm"], don["eps"], don["inits"]

    ths, stages, energy, final_th = _sample_phase(
        vg, D, p["chains"], p["samples"], p["thinning"], p["seed"], p["ctrl"],
        None, None, None, p["parallel"], p.get("progress_bar", False),
        donor_loader=load_donor)
    np.savez(p["out"], ths=ths, stages=stages, energy=energy, final_th=final_th)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "_sample_f32":
        import jax as _j
        _j.config.update("jax_enable_x64", False)
        _main_sample_f32(sys.argv[2])
