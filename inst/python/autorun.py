"""Adaptive-length sampling in the persistent worker (fit_broc(iter_sampling = "auto")).

A session adapts once, compiles one fixed-length sampling block and loops it. Draws stay
here: row t is kept when (t + 1) % stride == 0, and the stride doubles whenever more than
`max_kept` rows per chain would be kept, so storage is bounded however long the run.
Each round returns only the monitored parameters of the rows it added, and the next round
is already sampling while the caller checks convergence. finish() constrains the kept
rows once."""
import os, sys, json, time, pickle, re, threading, subprocess, tempfile, shutil
import numpy as np
import jax
import jax.numpy as jnp

import drghmc as dr
import progress
from essrhat import ess_rhat

_UNMONITORED = re.compile(r"^(log_lik|lp__|z_|u_|zs_|s_|L_corr_)")
_BLOCK_BYTES = 64 * 2 ** 20
_MON_CHUNK = 64
_FINAL_CHUNK = 256


def _blocker(ess, rhat, classes, names, targets, rhat_target):
    lim = np.array([float(targets.get(c, 0.0)) for c in classes])
    bad = (np.isfinite(ess) & (ess < lim)) | (np.isfinite(rhat) & (rhat > rhat_target))
    idx = np.flatnonzero(bad)
    return names[int(idx[0])] if idx.size == 1 else None


def _block_plan(check_every, chains, D, itemsize, bcap):
    """Block length b and blocks per round with b * nblocks == check_every exactly; b is
    capped by the block's memory and by `bcap` (how long a stop waits on a running block)."""
    n = int(check_every)
    bmax = max(1, min(int(bcap), _BLOCK_BYTES // max(1, chains * D * itemsize)))
    nblocks = -(-n // bmax)
    while n % nblocks:
        nblocks += 1
    return n // nblocks, nblocks


def _round_bar(T, n):
    """The bar of one round: draws T to T + n per chain, gone once the round is checked."""
    return progress.bar(T + n, "Sampling", start=T, unit="draws", leave=False)


def _kept(t, stride, burn):
    """Row t is kept when it lies past the burn-in and (t - burn + 1) % stride == 0."""
    t = np.asarray(t)
    return (t >= burn) & ((t - burn + 1) % stride == 0)


def _grow_stride(stride, T, max_kept):
    while T // stride > max_kept:
        stride *= 2
    return stride


def _padded_rows(f, rows, chunk):
    """Apply f ([chunk, D] -> pytree with leading chunk axis) to rows [n, D] in fixed-size
    chunks, padding the last one so f compiles once. Yields (start, count, numpy dict)."""
    n = rows.shape[0]
    for i in range(0, n, chunk):
        blk = rows[i:i + chunk]
        m = blk.shape[0]
        if m < chunk:
            blk = np.concatenate([blk, np.repeat(blk[-1:], chunk - m, 0)], 0)
        out = f(jnp.asarray(blk))
        yield i, m, {k: np.asarray(v)[:m] for k, v in out.items()}


def _padded_chains(f, rows, chunk):
    """As _padded_rows over rows [C, n, D] with f mapping [C, chunk, D]."""
    n = rows.shape[1]
    for i in range(0, n, chunk):
        blk = rows[:, i:i + chunk]
        m = blk.shape[1]
        if m < chunk:
            blk = np.concatenate([blk, np.repeat(blk[:, -1:], chunk - m, 1)], 1)
        out = f(jnp.asarray(blk))
        yield i, m, {k: np.asarray(v)[:, :m] for k, v in out.items()}


def _batched(state, chains):
    if jnp.ndim(state.potential_energy) == 0:
        return jax.tree.map(lambda x: jnp.asarray(x)[None], state)
    return jax.tree.map(jnp.asarray, state)


# =============================================================================
# Running totals over every transition (kept or not)
# =============================================================================

class _Totals:
    _KEYS = ("n", "div", "depth", "acc", "stage", "e_n", "e_mean", "e_m2", "e_de2", "e_last")

    def __init__(self, chains, prev=None):
        z = lambda: np.zeros(chains)
        self.n, self.div, self.depth, self.acc = z(), z(), z(), z()
        self.stage = np.zeros((chains, 4))
        self.e_n, self.e_mean, self.e_m2, self.e_de2 = z(), z(), z(), z()
        self.e_last = np.full(chains, np.nan)
        if prev:
            for k in self._KEYS:
                if k in prev and prev[k] is not None:
                    v = np.asarray(prev[k], dtype=float)
                    setattr(self, k, v.reshape(getattr(self, k).shape))

    def add(self, stats, max_steps):
        e = np.asarray(stats["energy"], dtype=float)
        L = e.shape[1]
        bm, bm2 = e.mean(1), ((e - e.mean(1, keepdims=True)) ** 2).sum(1)
        tot = self.e_n + L
        d = bm - self.e_mean
        self.e_m2 = self.e_m2 + bm2 + d ** 2 * self.e_n * L / tot
        self.e_mean = self.e_mean + d * L / tot
        self.e_n = tot
        jump = np.where(np.isfinite(self.e_last), (e[:, 0] - self.e_last) ** 2, 0.0)
        self.e_de2 = self.e_de2 + (np.diff(e, axis=1) ** 2).sum(1) + jump
        self.e_last = e[:, -1].copy()
        if "diverging" in stats:
            self.div += np.asarray(stats["diverging"]).sum(1)
            self.depth += (np.asarray(stats["num_steps"]) >= max_steps).sum(1)
            self.acc += np.asarray(stats["accept_prob"], dtype=float).sum(1)
        if "dr_stage" in stats:
            s = np.asarray(stats["dr_stage"])
            for j, v in enumerate((1.0, 2.0, 3.0, -1.0)):
                self.stage[:, j] += (s == v).sum(1)
            self.acc += (s > 0).sum(1)
        self.n += L

    def to_dict(self):
        return {k: np.asarray(getattr(self, k)).tolist() for k in self._KEYS}

    def diagnostics(self, sampler):
        if sampler == "drghmc":
            n = max(float(self.n.sum()), 1.0)
            return dict(accept=float(self.acc.sum() / n),
                        dr_stage=dict(s1=float(self.stage[:, 0].sum() / n),
                                      s2=float(self.stage[:, 1].sum() / n),
                                      s3=float(self.stage[:, 2].sum() / n),
                                      rej=float(self.stage[:, 3].sum() / n)))
        ebfmi = np.where(self.e_m2 > 0, self.e_de2 / np.maximum(self.e_m2, 1e-300), np.nan)
        return dict(num_divergent=self.div.astype(int).tolist(),
                    num_max_treedepth=self.depth.astype(int).tolist() if sampler != "hmc" else None,
                    ebfmi=ebfmi.tolist())


# =============================================================================
# Samplers: one compiled block, looped
# =============================================================================

class _InProcess:
    def run_round(self, carry, T, nblocks, b, stride, cancel, burn=0, show=False):
        zs, ts, stats = [], [], {}
        t = T
        bar = _round_bar(T, nblocks * b) if show else None
        for _ in range(nblocks):
            if cancel.is_set():
                if bar is not None:
                    bar.close(drop=True)
                return None
            pos = np.arange(t, t + b)
            keep = _kept(pos, stride, burn)
            carry, z, st = self.block(carry, t, keep)
            sel = np.nonzero(keep)[0]
            zs.append(np.asarray(z)[:, sel])
            ts.append(pos[sel])
            for name, v in st.items():
                stats.setdefault(name, []).append(np.asarray(v))
            t += b
            if bar is not None:
                bar.update(b)
        if bar is not None:
            bar.close()
        return (carry, np.concatenate(zs, 1), np.concatenate(ts),
                {n: np.concatenate(v, 1) for n, v in stats.items()})

    def close(self):
        pass

    def ordered(self, data):
        """`data` for constraining the kept rows: DR-G-HMC's samplers store the model's
        coordinates with the correlated blocks in the transport map's orders (tmap.TMap.corr)."""
        tm = getattr(self, "tm_order", None)
        return data if tm is None else tm.with_order_of(data, self.aux)

    def fix_draws(self, out):
        tm = getattr(self, "tm_order", None)
        if tm is not None:
            tm.fix_draws(out, self.aux)


def _with_offset(cfg, offset):
    return cfg if offset is None else {**cfg, "_ll_offset": jnp.asarray(offset)}


class _Fold(_InProcess):
    """samplers "hmc" and "nuts": fold_setup's coordinates and warmup, then blocks of the same transitions
    (HMC's trajectory length jittered each one). A resumed fit keeps its anchors, its adapted state and, in
    float32, the log-likelihood offset its potential is summed relative to."""

    def __init__(self, s, model, config, a, resume):
        import numpyro_backend as nb
        C = s.chains
        hmc = s.sampler == "hmc"
        tad, mtd, ss = float(a.get("target_accept_prob", 0.8)), int(a.get("max_tree_depth", 10)), float(a.get("step_size", 1.0))
        build = (s.vg, s.unflatten, s.keys, s.off, s.D, s.pp)
        if resume is not None:
            aux = resume.get("aux")
            coords = nb.NutsCoords(config, C, int(a["seed"]), s.parallel, s.progress,
                                   a.get("init_strategy", "numeric"), float(a.get("init_radius", 0.0)), None,
                                   transport=aux is not None, aux=aux, build=build, label="HMC" if hmc else "NUTS")
            kernel = (nb.FastHMC(model, coords=coords, integrator=str((a.get("control") or {}).get("integrator", "bcss")),
                                 cross_chain=nb._chain_adapt(s.parallel, C), gamma=nb._HMC_GAMMA, reduce=nb._HMC_REDUCE,
                                 target_accept_prob=tad,
                                 step_size=ss, trajectory_length=nb._HMC_TMAX) if hmc else
                      nb.FastNUTS(model, coords=coords, target_accept_prob=tad, max_tree_depth=mtd, step_size=ss))
            state = _batched(resume["hmc"], C)
            offset = resume.get("offset")
            kernel.init(jax.random.PRNGKey(0), 0, jax.tree.map(lambda x: x[0], state.z),
                        (_with_offset({**config, "_nuts_parts": coords.aux}, offset),), {})
        else:
            coords, pg, label = nb.fold_setup(config, C, int(a["warmup"]), 1, int(a["seed"]), tad, mtd, ss, s.progress,
                                              a.get("init_strategy", "numeric"), float(a.get("init_radius", 0.0)),
                                              s.init_z, bool((a.get("control") or {}).get("transport", True)),
                                              s.sampler, sampling=False, build=build, parallel=s.parallel,
                                              gamma=float((a.get("control") or {}).get("adapt_gamma", nb._HMC_GAMMA)),
                                              integrator=str((a.get("control") or {}).get("integrator", "bcss")))
            kernel = pg.kernel
            state = nb.fold_warmup(pg, coords, int(a["warmup"]), None, s.progress, label)
            offset = getattr(coords, "offset", None)
        self.coords = coords
        self.max_steps = np.inf if hmc else 2 ** mtd
        self.offset = offset
        jit = nb._jitter(nb._HMC_TMAX if hmc else None, 2 ** mtd)
        parts = coords.aux
        args = (_with_offset({**config, "_nuts_parts": parts}, offset),)

        def body(st, _):
            st = kernel.sample(jit(st), args, {})
            return st, (coords.model_v(st.z, parts), st.num_steps, st.energy, st.accept_prob, st.diverging)

        run = lambda st: jax.lax.scan(body, st, None, length=s.b)
        self.fn = jax.pmap(run, axis_name="chain") if s.parallel else jax.jit(lambda st: jax.lax.map(run, st))
        state = jax.tree.map(np.asarray, state)
        self.carry0 = state
        self.positions = coords.positions(state.z)

    def block(self, carry, t, keep=None):
        carry, (z, ns, en, ap, dv) = self.fn(carry)
        return carry, z, dict(num_steps=ns, energy=en, accept_prob=ap, diverging=dv)

    def state(self, carry, T):
        return {"hmc": jax.tree.map(np.asarray, carry), "offset": None if self.offset is None else np.asarray(self.offset),
                "aux": self.coords.aux_flat}


def _dr_store(s, model, config, a, ctrl, seed, tm=None):
    """The compiled-program store of this session's signature, checked against the traced
    potential when it comes from an earlier fit (drghmc._verify_store)."""
    sig = dr._signature(config, s.chains, s.parallel, ctrl, s.b, None if tm is None else tm.structure_key())
    store = dr._programs(sig)
    th0 = dr._initial_positions(model, config, s.keys, s.off, s.D, s.chains, seed,
                                a.get("init_strategy", "numeric"), a.get("init_radius", 0.0),
                                s.init_z, s.unflatten.to_sampled)
    if not dr._verify_store(s.vg, store, th0):
        store = dr._programs(sig, fresh=True)
        dr._verify_store(s.vg, store, th0)
    s.store = store
    return store


def _dr_adapt(s, model, config, a, ctrl, resume, seed, tm=None, vg_s=None, init=None, on_decide=None, store=None):
    """(imm, eps, positions in sampler coordinates, momenta, transport anchors)."""
    if resume is not None:
        return (np.asarray(resume["imm"]), float(resume["eps"]), np.asarray(resume["th"]),
                None if resume.get("rho") is None else np.asarray(resume["rho"]),
                None if tm is None else np.asarray(resume["aux"]))
    th0 = dr._initial_positions(model, config, s.keys, s.off, s.D, s.chains, seed,
                                a.get("init_strategy", "numeric"), a.get("init_radius", 0.0),
                                s.init_z, s.unflatten.to_sampled)
    imm, eps, th, aux, _ = dr._adapt(model, config, s.keys, s.off, s.vg, vg_s, tm, s.D, s.chains,
                                     int(a["warmup"]), seed, ctrl, s.parallel, s.progress, th0,
                                     user_init=s.init_z is not None, init=init, on_decide=on_decide, store=store)
    return np.asarray(imm), float(eps), np.asarray(th), None, aux


def _resume_ctrl(ctrl, resume):
    """A resumed fit keeps the coordinates it was sampled in."""
    if resume is not None:
        had = bool(resume.get("transport"))
        if had and not ctrl["transport"]:
            raise ValueError("the fit being resumed was sampled with control$transport = TRUE; "
                             "it cannot be continued with transport = FALSE")
        ctrl = {**ctrl, "transport": had}
    return ctrl


def _model_positions(tm, th, aux, trim):
    return np.asarray(th) if tm is None else tm.forward_chains(th, aux, trim)


class _Dr(_InProcess):
    def __init__(self, s, model, config, a, resume):
        ctrl = _resume_ctrl(dr._merge_control(a.get("control")), resume)
        C, D = s.chains, s.D
        self.seed = int(resume["seed"]) if resume is not None else int(a["seed"])
        tm = dr._transport(model, config, ctrl, s.off, s.unflatten, s.progress)
        vg_s = dr._sampler_vg(s.vg, tm)
        if tm is not None and resume is not None and resume.get("aux") is not None:
            tm = tm.cheap_of(resume["aux"])
            vg_s = tm.wrap(tm.pot_off)
        store = _dr_store(s, model, config, a, ctrl, self.seed, tm)
        ck = jax.random.split(dr._tagged(self.seed, 1), C)
        z0 = jnp.zeros((C, D))
        off0 = s.vg.OFF0

        def lower(tm_v, vg_v):
            setup_p, chunk = dr._sampler_fns(vg_v, D, ctrl, s.parallel, tm_v)
            aux0 = dr.aux_zeros(tm_v, C)
            return (setup_p.lower(z0, z0, off0, aux0),
                    chunk.lower((z0, z0, jnp.zeros(C), z0), jnp.arange(s.b), jnp.ones(s.b, bool), ck, jnp.ones(D),
                                jnp.ones(D), jnp.asarray(1e-3), off0, aux0))

        def compile_on_thread(flags, low, comp, ready):
            def _compile():
                try:
                    comp["setup"] = low[0].compile()
                    ready.set()
                    comp["chunk"] = low[1].compile()
                    store["sampler"][flags] = dict(setup=comp["setup"], chunk=comp["chunk"])
                    msg = dr.memory_check(comp["chunk"], C, s.parallel)
                    if msg is not None:
                        print(f"[auto] warning: {msg}", flush=True)
                except BaseException as e:
                    comp["err"] = e
                    ready.set()
            t = threading.Thread(target=_compile, daemon=True)
            t.start()
            return t

        def start(flags, tm_v, vg_v, comp, ready):
            hit = store["sampler"].get(flags)
            if hit is not None:
                comp.update(hit)
                ready.set()
                return None
            return compile_on_thread(flags, lower(tm_v, vg_v), comp, ready)

        comp, final = {}, {}
        setup_ready = threading.Event()
        th_c = start(dr._sampler_key(tm), tm, vg_s, comp, setup_ready)

        def on_decide(flags):
            if any(flags):
                final["tm"] = tm.view(flags)
                final["thread"] = start(tuple(int(f) for f in flags), final["tm"], final["tm"].wrap(tm.pot_off),
                                        final, threading.Event())

        imm, eps, th, rho, aux = _dr_adapt(s, model, config, a, ctrl, resume, self.seed, tm, vg_s,
                                           dr._setup_init(setup_ready, comp, z0, off0), on_decide, store=store)
        if th_c is not None:
            th_c.join()
        if "err" in comp:
            raise comp["err"]
        if final:
            if final["thread"] is not None:
                final["thread"].join()
            if "err" in final:
                raise final["err"]
            comp, tm = final, final["tm"]

        IMM = jnp.asarray(imm)
        MSQ = jnp.sqrt(1.0 / IMM)
        if rho is None:
            rho = MSQ * jax.random.normal(dr._tagged(self.seed, 2), (C, D))
        self.imm, self.eps, self.b = imm, eps, s.b
        self.chunk = comp["chunk"]
        if tm is not None:
            self.aux = np.asarray(aux)
        else:
            self.aux = np.zeros(1)
        self.transport = tm is not None
        self.tm_order = tm
        self.args = (ck, IMM, MSQ, jnp.asarray(ctrl["eps_c"] * eps), off0, dr.aux_in(tm, self.aux))
        self.carry0 = jax.tree.map(np.asarray, comp["setup"](jnp.asarray(th), jnp.asarray(rho), off0,
                                                             dr.aux_in(tm, self.aux)))
        self.positions = _model_positions(tm, th, self.aux, s.vg.trim)

    def block(self, carry, t, keep):
        ck, IMM, MSQ, E0, OFF, AUX = self.args
        carry, (th, stage, H0) = self.chunk(carry, jnp.arange(t, t + self.b), jnp.asarray(keep), ck, IMM, MSQ, E0, OFF, AUX)
        return jax.tree.map(np.asarray, carry), th, dict(dr_stage=stage, energy=H0)

    def state(self, carry, T):
        return dict(imm=np.asarray(self.imm), eps=float(self.eps), th=np.asarray(carry[0]),
                    rho=np.asarray(carry[1]), t0=int(T), seed=self.seed, offset=None, aux=self.aux,
                    transport=self.transport)


class _DrF32:
    """DR-G-HMC sampling in an x64-disabled child that serves one round per request; the
    parent adapts in float64 meanwhile and constrains the returned rows."""

    def __init__(self, s, model, config, a, resume):
        ctrl = _resume_ctrl(dr._merge_control(a.get("control")), resume)
        self.seed = int(resume["seed"]) if resume is not None else int(a["seed"])
        self.tmp = tempfile.mkdtemp(prefix="bayesroc_auto_f32_")
        cfg = os.path.join(self.tmp, "config.json")
        with open(cfg, "w") as fh:
            json.dump(config, fh)
        args = dict(chains=s.chains, b=s.b, seed=self.seed, ctrl=ctrl, parallel=s.parallel,
                    config=cfg, dir=self.tmp,
                    backend_dir=os.path.dirname(os.path.abspath(__file__)))
        argfile = os.path.join(self.tmp, "args.json")
        with open(argfile, "w") as fh:
            json.dump(args, fh)
        env = dict(os.environ)
        env["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={s.chains if s.parallel else 1} --xla_cpu_collective_call_terminate_timeout_seconds=86400"
        env["BAYESROC_ENABLE_X64"] = "0"
        self.proc = subprocess.Popen([sys.executable, os.path.abspath(__file__),
                                      "_dr_f32_server", argfile], env=env)
        try:
            tm = dr._transport(model, config, ctrl, s.off, s.unflatten, s.progress)
            vg_s = dr._sampler_vg(s.vg, tm)
            store = _dr_store(s, model, config, a, ctrl, self.seed, tm)
            imm, eps, th, rho, aux = _dr_adapt(s, model, config, a, ctrl, resume, self.seed, tm, vg_s,
                                               store=store)
            off = None if resume is None else resume.get("offset")
            donor = os.path.join(self.tmp, "donor.pkl")
            with open(donor + ".tmp", "wb") as fh:
                pickle.dump(dict(imm=imm, eps=eps, th=th, rho=rho, aux=aux,
                                 offset=None if off is None else np.asarray(off)), fh)
            os.replace(donor + ".tmp", donor)
        except BaseException:
            self.close()
            raise
        self.imm, self.eps, self.aux = imm, eps, aux
        self.transport = tm is not None
        self.tm_order = tm
        self.carry0 = dict(th=th, rho=rho, offset=off)
        self.positions = _model_positions(tm, th, aux, s.vg.trim)
        self.n = 0

    ordered, fix_draws = _InProcess.ordered, _InProcess.fix_draws

    def run_round(self, carry, T, nblocks, b, stride, cancel, burn=0, show=False):
        n = self.n
        self.n += 1
        req = os.path.join(self.tmp, f"req_{n}.json")
        with open(req + ".tmp", "w") as fh:
            json.dump(dict(T=int(T), nblocks=int(nblocks), stride=int(stride), burn=int(burn), show=bool(show)), fh)
        os.replace(req + ".tmp", req)
        res = os.path.join(self.tmp, f"res_{n}.npz")
        while not os.path.exists(res):
            if cancel.is_set():
                open(os.path.join(self.tmp, f"cancel_{n}"), "w").close()
                return None
            if self.proc.poll() is not None:
                raise RuntimeError(f"f32 sampling process exited (code {self.proc.returncode})")
            time.sleep(0.02)
        with np.load(res) as d:
            out = {k: d[k] for k in d.files}
        os.remove(res)
        carry = dict(th=out["th"], rho=out["rho"], offset=out["offset"])
        return (carry, out["z"].astype(np.float64), out["t"],
                dict(dr_stage=out["stage"], energy=out["energy"]))

    def state(self, carry, T):
        return dict(imm=np.asarray(self.imm), eps=float(self.eps), th=np.asarray(carry["th"]),
                    rho=None if carry["rho"] is None else np.asarray(carry["rho"]), t0=int(T),
                    seed=self.seed,
                    offset=None if carry["offset"] is None else np.asarray(carry["offset"]),
                    aux=None if self.aux is None else np.asarray(self.aux), transport=self.transport)

    def close(self):
        try:
            open(os.path.join(self.tmp, "stop"), "w").close()
            self.proc.wait(timeout=10)
        except Exception:
            pass
        if self.proc.poll() is None:
            self.proc.kill()
        shutil.rmtree(self.tmp, ignore_errors=True)


def _nuts_anchors_server(argfile):
    with open(argfile, "rb") as fh:
        p = pickle.load(fh)
    sys.path.insert(0, p["backend_dir"])
    import numpyro_backend as nb
    c = nb.NutsCoords(p["config"], p["chains"], p["seed"], p["parallel"], p["progress_bar"], p["init_strategy"],
                      p["init_radius"], p["init_z"], transport=True, announce=False)
    np.savez(p["out"], aux=c.aux_flat, start=np.asarray(c.start))


def _fold_f64_server(argfile):
    """The float64 parts of a float32 fold fit, in a float64 process kept for the fit's warmup: it builds the
    coordinates with Laplace anchors (as a float64 fit does) and writes the anchors, the starting positions, the
    Laplace stage's results and the per-observation log-likelihood at the first start; then, for each path read
    from stdin, rebuilds the anchors at the window mean it names, until "exit" or the end of stdin."""
    with open(argfile, "rb") as fh:
        p = pickle.load(fh)
    sys.path.insert(0, p["backend_dir"])
    import numpyro_backend as nb
    c = nb.NutsCoords(p["config"], p["chains"], p["seed"], p["parallel"], p["progress_bar"], p["init_strategy"],
                      p["init_radius"], p["init_z"], transport=p["transport"], anchors="laplace", label=p["label"])
    z0 = {k: v[0] for k, v in c.init_params(p["chains"], batched=True).items()}
    off = np.asarray(jax.jit(lambda z: nb._ll_at(p["config"], c, z))(z0))
    out = dict(aux=None if c.aux_flat is None else np.asarray(c.aux_flat), start=np.asarray(c.start),
               source=c.source, pf=c.pf, offset=off)
    with open(p["out"], "wb") as fh:
        pickle.dump(out, fh)
    print("@@READY", flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line or line == "exit":
            break
        with open(line, "rb") as fh:
            req = pickle.load(fh)
        mean = np.asarray(req["mean"])
        ref = np.repeat(mean.mean(0, keepdims=True), mean.shape[0], 0)
        aux = np.asarray(c._tm.anchors_chains(c.vg, ref, c.vg.OFF0, c._parallel)[:1])
        with open(req["out"], "wb") as fh:
            pickle.dump(dict(aux=aux), fh)
        print("@@DONE", flush=True)


def _dr_f32_server(argfile):
    with open(argfile) as fh:
        p = json.load(fh)
    sys.path.insert(0, p["backend_dir"])
    import numpyro_backend as nb
    with open(p["config"]) as fh:
        config = json.load(fh)
    dr._convert_matrices(config)
    C, b, seed, d = int(p["chains"]), int(p["b"]), int(p["seed"]), p["dir"]
    vg, unflatten, keys, off, D, pp = dr._build(nb.sdt_model, config)
    tm = dr._transport(nb.sdt_model, config, p["ctrl"], off, unflatten)
    setup_p, chunk = dr._sampler_fns(dr._sampler_vg(vg, tm), D, p["ctrl"], bool(p["parallel"]), tm)
    ck = jax.random.split(dr._tagged(seed, 1), C)
    z0 = jnp.zeros((C, D))
    aux0 = dr.aux_zeros(tm, C)
    c0 = setup_p(z0, z0, vg.OFF0, aux0)
    c0 = jax.tree.map(np.asarray, c0)
    jax.block_until_ready(chunk(c0, jnp.arange(b), jnp.ones(b, bool), ck, jnp.ones(D), jnp.ones(D),
                                jnp.asarray(1e-3), vg.OFF0, aux0))
    stop = os.path.join(d, "stop")
    donor = os.path.join(d, "donor.pkl")
    while not os.path.exists(donor):
        if os.path.exists(stop):
            return
        time.sleep(0.05)
    with open(donor, "rb") as fh:
        don = pickle.load(fh)
    th = np.asarray(don["th"])
    if tm is not None and tuple(tm.flags(don["aux"])) != tm.static:
        tm = tm.cheap_of(don["aux"])
        setup_p, chunk = dr._sampler_fns(tm.wrap(tm.pot_off), D, p["ctrl"], bool(p["parallel"]), tm)
    AUX = dr.aux_in(tm, don["aux"] if don.get("aux") is not None else np.zeros(1))
    OFF = {**vg.OFF0, "off": (jnp.asarray(don["offset"]) if don.get("offset") is not None
                              else dr.ll_offset(pp, unflatten, config,
                                                (th[:1] if tm is None
                                                 else tm.forward_chains(th[:1], np.asarray(don["aux"])[:1], vg.trim))[0],
                                                vg.data))}
    IMM = jnp.asarray(don["imm"])
    MSQ = jnp.sqrt(1.0 / IMM)
    E0 = jnp.asarray(p["ctrl"]["eps_c"] * float(don["eps"]))
    rho = don.get("rho")
    rho = (MSQ * jax.random.normal(dr._tagged(seed, 2), (C, D)) if rho is None
           else jnp.asarray(rho))
    carry = jax.tree.map(np.asarray, setup_p(jnp.asarray(th), rho, OFF, AUX))
    n = 0
    while True:
        req = os.path.join(d, f"req_{n}.json")
        while not os.path.exists(req):
            if os.path.exists(stop):
                return
            time.sleep(0.02)
        with open(req) as fh:
            r = json.load(fh)
        t, k = int(r["T"]), int(r["stride"])
        zs, ts, ss, es = [], [], [], []
        cancelled = False
        bar = _round_bar(t, int(r["nblocks"]) * b) if r.get("show") else None
        for _ in range(int(r["nblocks"])):
            if os.path.exists(os.path.join(d, f"cancel_{n}")):
                cancelled = True
                break
            pos = np.arange(t, t + b)
            keep = _kept(pos, k, int(r.get("burn", 0)))
            carry, (thb, st, e) = chunk(carry, jnp.arange(t, t + b), jnp.asarray(keep), ck, IMM, MSQ, E0, OFF, AUX)
            carry = jax.tree.map(np.asarray, carry)
            sel = np.nonzero(keep)[0]
            zs.append(np.asarray(thb)[:, sel]); ts.append(pos[sel])
            ss.append(np.asarray(st)); es.append(np.asarray(e))
            t += b
            if bar is not None:
                bar.update(b)
        if bar is not None:
            bar.close(drop=cancelled)
        if cancelled:
            n += 1
            continue
        tmp = os.path.join(d, f"res_{n}.part.npz")
        np.savez(tmp, z=np.concatenate(zs, 1), t=np.concatenate(ts), stage=np.concatenate(ss, 1),
                 energy=np.concatenate(es, 1), th=np.asarray(carry[0]), rho=np.asarray(carry[1]),
                 offset=np.asarray(OFF["off"]))
        os.replace(tmp, os.path.join(d, f"res_{n}.npz"))
        n += 1


# =============================================================================
# Session
# =============================================================================

class AutoSession:
    def __init__(self, model, config, a):
        self.t_start = time.time()
        self.a = a
        self.sampler = a.get("sampler", "nuts")
        self.chains = C = int(a["chains"])
        self.parallel = a.get("chain_method", "parallel") == "parallel" and C > 1
        self.progress = bool(a.get("progress_bar", True))
        self.max_kept = int(a["max_kept"])
        self.T = int(a.get("t0") or 0)
        self.stride = int(a.get("stride") or 1)
        self.burn = int(a.get("burnin") or 0)
        self.config = config
        self.store = None
        self.vg, self.unflatten, self.keys, self.off, self.D, self.pp = dr._build(model, config)
        itemsize = 8 if jax.config.jax_enable_x64 else 4
        self.b, self.nblocks = _block_plan(int(a["check_every"]), C, self.D, itemsize,
                                           25 if self.sampler in ("nuts", "hmc") else 100)
        self.round_len = self.b * self.nblocks
        self.min_draws = int(a.get("min_draws") or 0)
        self.init_z = None
        resume = None
        if a.get("resume_state"):
            with open(a["resume_state"], "rb") as fh:
                resume = pickle.load(fh)
            if isinstance(resume, dict) and "t0" in resume and resume["t0"] is not None:
                self.T = int(resume["t0"])
        elif a.get("init_values") is not None:
            import numpyro_backend as nb
            self.init_z = nb.init_values_to_z(config, a["init_values"], int(a["seed"]),
                                              float(a.get("init_radius", 2.0)))
        f32_dr = (self.sampler == "drghmc"
                  and dr._merge_control(a.get("control"))["precision"] == "f32")
        fold = self.sampler in ("hmc", "nuts")
        cls = _Fold if fold else (_DrF32 if f32_dr else _Dr)
        self.ad = cls(self, model, config, a, resume)
        self.max_steps = getattr(self.ad, "max_steps", 0)
        self.carry = self.ad.carry0
        self.tot = _Totals(C, a.get("totals"))

        full1 = lambda z, d: self.pp(self.unflatten(z), d)
        shapes = jax.eval_shape(full1, jnp.zeros(self.D), self.vg.data)
        self.mon_keys = sorted(k for k in shapes if not _UNMONITORED.match(k))
        self.mon_shapes = {k: list(shapes[k].shape) for k in self.mon_keys}
        self.mon_cols = None
        self.mon = np.empty((C, 0, 0))
        self.mon_t = np.empty(0, dtype=np.int64)
        self.mon_n = 0
        self.rows = np.empty((C, 0, self.D), dtype=np.float64 if itemsize == 8 else np.float32)
        self.row_t = np.empty(0, dtype=np.int64)
        self.extra = {}
        self.n = 0
        self.cancel = threading.Event()
        self.mon_ready = threading.Event()
        self.pending = None
        self.last_dg = None
        if self.sampler in ("nuts", "hmc"):
            self._verify(model, config, full1, itemsize)
            self._launch()
        else:
            self._launch()
            self._verify(model, config, full1, itemsize)

    def _verify(self, model, config, full1, itemsize):
        """Monitored parameters come from a 1-row config (no likelihood) when it reproduces
        the full constrain at the warmed-up positions. NUTS traces the model when its first
        round compiles, and numpyro's handler stack is global, so for NUTS this runs before
        the first round instead of beside it."""
        try:
            cache = None if self.store is None else self.store.setdefault("auto", {})
            if cache is not None and "verify" in cache:
                trim1, fa_fn, fb_fn = cache["verify"]
            else:
                _, _, trim_post = dr._model_info(model, dr._trim_config(config))
                trim1 = lambda z, d: trim_post(self.unflatten(z), d)
                fa_fn, fb_fn = jax.jit(jax.vmap(full1, (0, None))), jax.jit(jax.vmap(trim1, (0, None)))
                if cache is not None:
                    cache["verify"] = (trim1, fa_fn, fb_fn)
            pos = jnp.asarray(self.ad.positions)
            fa = fa_fn(pos, self.vg.data)
            fb = fb_fn(pos, self.vg.trim)
            ok = set(fa) - {"log_lik"} == set(fb) - {"log_lik"} and all(
                np.allclose(np.asarray(fa[k]), np.asarray(fb[k]), rtol=1e-10 if itemsize == 8 else 1e-5,
                            atol=1e-12 if itemsize == 8 else 1e-6)
                for k in fa if k != "log_lik" and not k.startswith(("z_", "L_corr_")))
            self.trim1 = trim1 if ok else None
            self.mon_data = self.ad.ordered(self.vg.trim if ok else self.vg.data)
            base = trim1 if ok else full1
            mk = self.mon_keys
            key = ("mon", bool(ok), tuple(mk))
            if cache is not None and key in cache:
                self.mon_fn = cache[key]
            else:
                self.mon_fn = jax.jit(jax.vmap(lambda z, d: {k: v for k, v in base(z, d).items() if k in mk}, (0, None)))
                if cache is not None:
                    cache[key] = self.mon_fn
        finally:
            self.mon_ready.set()

    def start_info(self):
        return dict(b=self.b, nblocks=self.nblocks, round_len=self.round_len, D=self.D,
                    T=self.T, stride=self.stride, mon_keys=self.mon_keys,
                    mon_shapes=self.mon_shapes)

    def set_map(self, a):
        """Monitored columns in the fit's (R) naming: `cols` index the flattened monitored
        sites, `names`/`classes` label them. `old_mon` [chains, kept, len(cols)] holds the
        monitored draws of a resumed fit, kept every `old_stride` draws."""
        self.mon_cols = np.asarray(a["cols"], dtype=np.int64).reshape(-1)
        as_list = lambda v: [v] if isinstance(v, str) else list(v)
        self.mon_names = as_list(a["names"])
        self.mon_class = np.asarray(as_list(a["classes"]))
        V = len(self.mon_cols)
        self.mon = np.empty((self.chains, 0, V))
        if a.get("old_mon"):
            old = np.load(a["old_mon"])
            t = np.arange(1, old.shape[1] + 1, dtype=np.int64) * int(a.get("old_stride", 1)) - 1
            keep = (t + 1) % self.stride == 0
            self.mon = np.ascontiguousarray(old[:, keep])
            self.mon_t = t[keep]
            self.mon_n = int(keep.sum())

    def _flat_mon(self, mon, m):
        if not m:
            return np.empty((self.chains, 0, len(self.mon_cols)))
        flat = np.concatenate([np.asarray(mon[k], dtype=np.float64).reshape(self.chains, m, -1)
                               for k in self.mon_keys], axis=2)
        return flat[:, :, self.mon_cols]

    def _diagnose(self):
        if self.mon_n < 4:
            return dict(ess_pop=0.0, ess_sd=0.0, ess_cor=0.0, rhat=None, worst=None, worst_rhat=None)
        x = np.transpose(self.mon[:, :self.mon_n], (1, 0, 2))
        bulk, tail, rhat = ess_rhat(x, sampler=self.sampler)
        ess = np.minimum(bulk, tail)
        self.last_ess, self.last_rhat = ess, rhat
        out = {}
        for c in ("pop", "sd", "cor"):
            v = ess[(self.mon_class == c) & np.isfinite(ess)]
            out["ess_" + c] = float(v.min()) if v.size else None
        fin = np.isfinite(rhat)
        out["rhat"] = float(rhat[fin].max()) if fin.any() else 1.0
        e2 = np.where(np.isfinite(ess), ess, np.inf)
        out["worst"] = self.mon_names[int(np.argmin(e2))] if np.isfinite(e2).any() else None
        out["worst_rhat"] = self.mon_names[int(np.argmax(np.where(fin, rhat, -np.inf)))] if fin.any() else None
        return out

    def _mon(self, z):
        C, m, D = z.shape
        out = {}
        for i, cnt, blk in _padded_rows(lambda r: self.mon_fn(r, self.mon_data), z.reshape(C * m, D), _MON_CHUNK):
            for k, v in blk.items():
                if k not in out:
                    out[k] = np.empty((C * m,) + v.shape[1:], dtype=v.dtype)
                out[k][i:i + cnt] = v
        return {k: v.reshape((C, m) + v.shape[1:]) for k, v in out.items()}

    def _round(self, carry, T, stride):
        t0 = time.time()
        nb = -(-(self.min_draws - T) // self.b) if T < self.min_draws else self.nblocks
        T_end = T + nb * self.b
        k = _grow_stride(stride, max(T_end - self.burn, 0), self.max_kept)
        r = self.ad.run_round(carry, T, nb, self.b, k, self.cancel, self.burn, self.progress)
        if r is None:
            return None
        carry_end, z, tk, stats = r
        self.mon_ready.wait()
        mon = self._mon(z) if z.shape[1] else {}
        sel = np.nonzero(_kept(np.arange(T, T_end), k, self.burn))[0]
        return dict(carry=carry_end, T=T_end, stride=k, z=z, t=tk, stats=stats,
                    kept_stats={n: v[:, sel] for n, v in stats.items()}, mon=mon,
                    wall=time.time() - t0)

    def _launch(self, after=None):
        res = {}
        carry, T, stride = ((self.carry, self.T, self.stride) if after is None
                            else (after["carry"], after["T"], after["stride"]))

        def work():
            try:
                res["out"] = self._round(carry, T, stride)
            except BaseException as e:
                res["err"] = e

        th = threading.Thread(target=work, daemon=True)
        th.start()
        self.pending = (th, res)

    def _stop_pending(self):
        if self.pending is None:
            return
        self.cancel.set()
        self.pending[0].join()
        self.pending = None

    def _commit(self, r):
        k = r["stride"]
        if k != self.stride and self.n:
            idx = np.nonzero(_kept(self.row_t[:self.n], k, self.burn))[0]
            m = len(idx)
            self.rows[:, :m] = self.rows[:, idx]
            self.row_t[:m] = self.row_t[idx]
            for name in self.extra:
                self.extra[name][:, :m] = self.extra[name][:, idx]
            self.n = m
        m = r["z"].shape[1]
        if self.n + m > self.rows.shape[1]:
            cap = min(self.max_kept, max(self.n + m, 2 * self.rows.shape[1]))
            grow = lambda a, shape: np.concatenate([a[..., :self.n], np.empty(shape[:-1] + (cap - self.n,), a.dtype)], -1)
            self.rows = np.concatenate([self.rows[:, :self.n],
                                        np.empty((self.chains, cap - self.n, self.D), self.rows.dtype)], 1)
            self.row_t = np.concatenate([self.row_t[:self.n], np.empty(cap - self.n, np.int64)])
            for name in self.extra:
                self.extra[name] = grow(self.extra[name], self.extra[name].shape)
        self.rows[:, self.n:self.n + m] = r["z"]
        self.row_t[self.n:self.n + m] = r["t"]
        for name, v in r["kept_stats"].items():
            if name not in self.extra:
                self.extra[name] = np.empty((self.chains, self.rows.shape[1]), dtype=np.float64)
            self.extra[name][:, self.n:self.n + m] = v
        self.n += m
        self.T, self.stride, self.carry = r["T"], k, r["carry"]
        self.tot.add(r["stats"], self.max_steps)
        if self.mon_cols is not None:
            if self.mon_n:
                keep = np.nonzero(_kept(self.mon_t[:self.mon_n], k, self.burn))[0]
                self.mon[:, :len(keep)] = self.mon[:, keep]
                self.mon_t[:len(keep)] = self.mon_t[keep]
                self.mon_n = len(keep)
            new = self._flat_mon(r["mon"], m)
            if self.mon_n + m > self.mon.shape[1]:
                cap = max(self.mon_n + m, min(self.max_kept, 2 * max(self.mon.shape[1], m)))
                self.mon = np.concatenate([self.mon[:, :self.mon_n],
                                           np.empty((self.chains, cap - self.mon_n, self.mon.shape[2]))], 1)
                self.mon_t = np.concatenate([self.mon_t[:self.mon_n], np.empty(cap - self.mon_n, np.int64)])
            self.mon[:, self.mon_n:self.mon_n + m] = new
            self.mon_t[self.mon_n:self.mon_n + m] = r["t"]
            self.mon_n += m

    def step(self, out_dir, a=None):
        """One round: start the next round from the finished one's end (unless this is the last
        check, `force`), then commit and diagnose while it samples. When the stopping rule
        (`targets`, `rhat`; never with `fixed` draws) is met the new round is cancelled and its
        draws dropped, so the kept draws and the saved state end at the committed round."""
        a = a or {}
        if self.pending is None:
            self._launch()
        th, res = self.pending
        th.join()
        self.pending = None
        if "err" in res:
            raise res["err"]
        r = res["out"]
        if not a.get("force"):
            self._launch(after=r)
        self._commit(r)
        os.makedirs(out_dir, exist_ok=True)
        checked = self.T >= int(a.get("min_draws") or 0) or bool(a.get("force"))
        t0 = time.time()
        rhat_t = float(a["rhat"]) if a.get("rhat") is not None else np.inf
        check_s = 0.0
        if checked:
            self.last_dg = self._diagnose()
            check_s = time.time() - t0
            self.blocker = (None if getattr(self, "last_ess", None) is None else
                            _blocker(self.last_ess, self.last_rhat, self.mon_class, self.mon_names,
                                     a.get("targets") or {}, rhat_t))
        dg = dict(self.last_dg or {}, blocker=getattr(self, "blocker", None))
        met = checked and not a.get("fixed") and all(
            dg.get("ess_" + c) is None or dg["ess_" + c] >= float(t)
            for c, t in (a.get("targets") or {}).items()) and (
            dg.get("rhat") is None or dg["rhat"] <= rhat_t)
        if met:
            self._stop_pending()
        n_all = max(float(self.tot.n.sum()), 1.0)
        kept_bytes = (self.n * self.chains * self.D * self.rows.itemsize
                      + self.mon_n * self.mon.shape[0] * self.mon.shape[2] * self.mon.itemsize)
        info = dict(T=self.T, stride=self.stride, kept=int(self.mon_n), seconds=float(r["wall"]),
                    checked=bool(checked), kept_bytes=float(kept_bytes),
                    check_seconds=float(check_s), divergent=int(self.tot.div.sum()),
                    rejected=float(self.tot.stage[:, 3].sum() / n_all),
                    max_treedepth=int(self.tot.depth.sum()), **dg)
        with open(os.path.join(out_dir, "step.json"), "w") as fh:
            json.dump(info, fh)

    def finish(self, out_dir, f):
        self._stop_pending()
        t0 = time.time()
        save_loglik = bool(f.get("save_loglik", True))
        keep_re = bool(f.get("keep_re", False))
        ll_dt = np.float32 if f.get("loglik_precision", "f32") == "f32" else np.float64
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "state.pkl"), "wb") as fh:
            pickle.dump(self.ad.state(self.carry, self.T), fh)
        rows = self.rows[:, :self.n]
        use_trim = not save_loglik and self.trim1 is not None
        base = self.trim1 if use_trim else (lambda z, d: self.pp(self.unflatten(z), d))
        dat = self.ad.ordered(self.vg.trim if use_trim else self.vg.data)
        keep = lambda k: not ((k.startswith("z_") and not keep_re) or (k == "log_lik" and not save_loglik))
        one = lambda z, d: {k: v for k, v in base(z, d).items() if keep(k)}
        shapes = jax.eval_shape(one, jax.ShapeDtypeStruct(rows.shape[2:], rows.dtype), dat)
        per_draw = sum(int(np.prod(v.shape)) * np.dtype(v.dtype).itemsize for v in shapes.values())
        chunk = int(max(1, min(_FINAL_CHUNK, dr._CONSTRAIN_BYTES // max(1, per_draw * self.chains))))
        cache = None if self.store is None else self.store.setdefault("auto", {})
        key = ("finish", use_trim, bool(save_loglik), bool(keep_re))
        if cache is not None and key in cache:
            fn = cache[key]
        else:
            batch = jax.vmap(one, (0, None))
            par = self.chains > 1 and jax.local_device_count() >= self.chains
            fn = jax.pmap(batch, axis_name="chain", in_axes=(0, None)) if par else jax.jit(jax.vmap(batch, (0, None)))
            if cache is not None:
                cache[key] = fn

        old_ll, old_sel = f.get("old_loglik"), None
        if save_loglik and old_ll:
            old = np.load(old_ll, mmap_mode="r")
            ok = int(old.shape[1])
            j = np.arange(1, ok + 1)
            old_sel = np.nonzero((j * int(f.get("old_stride", 1))) % self.stride == 0)[0]
        n_old = 0 if old_sel is None else len(old_sel)

        out, files = {}, {}
        centred = {k[6:] for k in shapes if k.startswith("orig__")}
        bar = dr._pbar(-(-self.n // chunk), "Constraining draws", unit="chunks") if self.progress and self.n else None
        for i, m, blk in _padded_chains(lambda b: fn(b, dat), rows, chunk):
            for k, v in blk.items():
                if k.startswith("z_") and not keep_re:
                    continue
                if k == "log_lik" and not save_loglik:
                    continue
                if k in centred:
                    continue
                if k.startswith("orig__"):
                    k = k[6:]
                if k not in out:
                    fname = "sample_" + re.sub(r"[^A-Za-z0-9_\[\]]", "_", k) + ".npy"
                    lead = n_old if k == "log_lik" else 0
                    dtype = ll_dt if k == "log_lik" else v.dtype
                    out[k] = np.lib.format.open_memmap(
                        os.path.join(out_dir, fname), mode="w+", dtype=dtype,
                        shape=(self.chains, lead + self.n) + v.shape[2:])
                    files[k] = fname
                lead = n_old if k == "log_lik" else 0
                out[k][:, lead + i:lead + i + m] = v
            if bar is not None:
                bar.update(1)
        if bar is not None:
            bar.close()
        self.ad.fix_draws(out)
        if n_old and "log_lik" in out:
            for s in range(0, n_old, 256):
                e = min(s + 256, n_old)
                out["log_lik"][:, s:e] = old[:, old_sel[s:e]]
        for v in out.values():
            v.flush()
        del out
        extras = {}
        names = (("num_steps", "energy", "accept_prob", "diverging") if self.sampler in ("nuts", "hmc")
                 else ("energy", "dr_stage"))
        for name in names:
            if name in self.extra:
                extras[name] = self.extra[name][:, :self.n]
        if "dr_stage" in extras:
            extras["accepted"] = (extras["dr_stage"] > 0).astype(np.float64)
        efiles = {}
        for k, v in extras.items():
            fname = f"extra_{k}.npy"
            np.save(os.path.join(out_dir, fname), v)
            efiles[k] = fname
        meta = dict(elapsed=float(time.time() - self.t_start), samples=files, extra_fields=efiles,
                    state="state.pkl", T=self.T, stride=self.stride, kept_new=self.n,
                    loglik_old_rows=n_old, diagnostics=self.tot.diagnostics(self.sampler),
                    totals=self.tot.to_dict(), constrain_seconds=float(time.time() - t0))
        with open(os.path.join(out_dir, "results.json"), "w") as fh:
            json.dump(meta, fh)

    def close(self):
        try:
            self._stop_pending()
        finally:
            self.ad.close()
            self.rows = None


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "_dr_f32_server":
        jax.config.update("jax_enable_x64", False)
        _dr_f32_server(sys.argv[2])
    elif len(sys.argv) >= 3 and sys.argv[1] == "_nuts_anchors":
        _nuts_anchors_server(sys.argv[2])
    elif len(sys.argv) >= 3 and sys.argv[1] == "_fold_f64":
        _fold_f64_server(sys.argv[2])
