"""DR-G-HMC sampler (Modi et al. 2024, arXiv:2406.02741). NUTS-free.

Self-adapting f64 donor -> f64/f32 sampling. The sampler only touches
value_and_grad(potential_fn), so every family's custom-JVP kernel is reused
unchanged. run_drghmc returns the same dict contract as fit_model's NUTS path.
"""
import os, sys, json, time, pickle, subprocess, tempfile, shutil
import functools
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import jax
import jax.numpy as jnp
import phases
from numpyro.infer.util import initialize_model
from numpyro.infer import init_to_uniform

_LAM = 0.19318332750378357
_B1, _A1 = 0.11888010966548, 0.29619504261126
_B2, _A2 = 0.5 - _B1, 1.0 - 2.0 * _A1
_DR_EPS = (0, 1, 0, 2, 0, 1, 0)
_SCHED10 = (0.15, 0.08, 0.08, 0.05, 0.05, 0.05, 0.05, 0.03, 0.02, 0.02)

_DEFAULTS = dict(acc_target=0.92, gamma=0.32, integrator="bcss3", dr_k=2,
                 dr_r=4.0, adapt_windows=10, eps_c=1.0,
                 mass_burn_frac=0.4, mass_end_frac=0.7, precision="f64",
                 transport=True, transport_links=True, transport_wide=True)
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
        (isinstance(c["transport"], (bool, np.bool_)), "transport must be TRUE or FALSE"),
        (isinstance(c["transport_links"], (bool, np.bool_)), "transport_links must be TRUE or FALSE"),
        (isinstance(c["transport_wide"], (bool, np.bool_)), "transport_wide must be TRUE or FALSE"),
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


def memory_check(compiled, chains, parallel):
    """Scratch memory of the compiled sampler for all chains, against available RAM.
    Returns a message when it will not fit, else None."""
    try:
        per = compiled.memory_analysis().temp_size_in_bytes
    except Exception:
        return None
    need = per * (chains if parallel else 1)
    try:
        import psutil
        avail = psutil.virtual_memory().available
    except Exception:
        return None
    if need > 0.8 * avail:
        return (f"sampler needs about {need / 2**30:.1f} GB of working memory for {chains} chains "
                f"but only {avail / 2**30:.1f} GB is free")
    return None


def _pbar(total, desc, **kw):
    import progress
    return progress.bar(total, desc, **kw)


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


def _trim_rows(config, n_keep):
    """The rows a trimmed config keeps: the first n_keep, plus whatever row keeps the
    model's parameterisation the same as on the full data (the CDP R/K/G know-criterion
    slack transform exists only when a Guess trial is present)."""
    rows = list(range(n_keep))
    if config.get("family") == "cdp" and int(config.get("n_rkg", 2)) == 3:
        y = np.asarray(config["y"]); rk = np.asarray(config["rk"])
        olm = np.asarray(config["old_level_map"])
        live = np.ones(y.shape[0], bool) if config.get("counts") is None else np.asarray(config["counts"]) > 0
        g = np.where(np.isin(y, olm) & (rk == 3) & live)[0]
        if g.size and int(g[0]) not in rows:
            rows.append(int(g[0]))
    return rows


_TRIM_ABOVE = 4096


def _trim_config(config, n_keep=1):
    """A tiny copy of config (n_keep rows, see _trim_rows). u_*/corr_*/params are
    data-independent, so constraining against this yields identical draws with no per-obs
    likelihood cost. Slices length-N arrays without copying the full data. A list of dicts
    (one per random-effect block, ...) is structure, never rows. A config of at most
    _TRIM_ABOVE rows is returned whole: trimming saves little there, and a short structural
    list of numbers (one per threshold, coefficient, ...) could have the length of the data."""
    n = config.get("N")
    if n is not None and n <= _TRIM_ABOVE:
        return dict(config)
    rows = _trim_rows(config, n_keep) if n is not None else list(range(n_keep))

    def per_row(v):
        return isinstance(v, list) and n is not None and len(v) == n and not any(isinstance(it, dict) for it in v)

    def t(o):
        if isinstance(o, dict):
            return {k: ([v[i] for i in rows] if per_row(v) else t(v)) for k, v in o.items()}
        if isinstance(o, list):
            return [t(it) if isinstance(it, dict) else it for it in o]
        return o

    c = t(config)
    c["N"] = len(rows)
    return c


_DATA_TOP = ("y", "is_old", "counts", "source", "rk", "y2", "item_type", "target_pos", "m_alt", "X_crit")
_DATA_INT = ("y", "source", "rk", "y2", "item_type", "target_pos", "m_alt")


def _data_of(config):
    """The data of a config as device arrays: responses, trial-type columns, design matrices
    and random-effect index vectors. Every program takes them as an argument (the `OFF`
    bundle), so a compiled program, and its key in the persistent compilation cache, depends
    on the data's shapes only."""
    def arr(v, integer=False):
        return jnp.asarray(np.asarray(v, dtype=np.int32 if integer else np.float64))
    top = {k: arr(config[k], k in _DATA_INT) for k in _DATA_TOP if config.get(k) is not None}
    params = {p: arr(pc["X"]) for p, pc in (config.get("params") or {}).items() if pc.get("X") is not None}
    res = [{k: arr(rc[k], k != "Z") for k in ("Z", "index", "term_idx") if rc.get(k) is not None}
           for rc in (config.get("random_effects") or [])]
    extra = {k: arr(config[k]["X"]) for k in ("bias", "crit2_cov")
             if config.get(k) is not None and config[k].get("X") is not None}
    return dict(top=top, params=params, re=res, extra=extra)


def _with_data(config, data, offset=None):
    """`config` with its data replaced by `data` (traced inside a program); the original stays
    under "_static" for the structural decisions that need concrete values."""
    import numpyro_backend as backend
    if config.get("crit_scalar") or config.get("crit_scalar_groups") is not None:
        backend._apply_crit_scalar(config, config.get("n_thresh", config["K"] - 1))
    c = dict(config)
    c.update(data["top"])
    c["params"] = {p: ({**pc, "X": data["params"][p]} if p in data["params"] else pc)
                   for p, pc in (config.get("params") or {}).items()}
    c["random_effects"] = [{**rc, **d} for rc, d in zip(config.get("random_effects") or [], data["re"])]
    for k, X in data["extra"].items():
        c[k] = {**config[k], "X": X}
    c["_static"] = config
    if offset is not None:
        c["_ll_offset"] = offset
    return c


def _off0(config, data=None, trim=None):
    """The sampler's runtime bundle: per-observation log-likelihood offset (zero), the data
    and the data of the trimmed config the transport map constrains against."""
    data = _data_of(config) if data is None else data
    trim = _data_of(_trim_config(config)) if trim is None else trim
    return dict(off=jnp.zeros(int(config["N"])), data=data, trim=trim)


_PROGRAMS = {}
_PROGRAMS_MAX = 8
_STRUCTURAL_RESPONSES = ("cdp", "metadt")


def _signature(config, chains, parallel, ctrl, step):
    """Key of the compiled programs a fit can reuse: the config with its responses replaced by
    their lengths (kept whole for the families whose program structure reads them), the chain
    count, the control and the sampler's chunk length. Everything else that shapes a program
    (design, priors, grouping) is in the config."""
    import hashlib
    keep = config.get("family") in _STRUCTURAL_RESPONSES or bool(config.get("varying_source_criteria"))
    c = {k: v for k, v in config.items() if k != "_static"}
    for k in ("y", "y2", "rk"):
        if not keep and isinstance(c.get(k), list):
            c[k] = ["<rows>", len(c[k])]
    body = dict(config=c, chains=int(chains), parallel=bool(parallel), ctrl=ctrl, step=int(step),
                x64=bool(jax.config.jax_enable_x64))
    text = json.dumps(body, sort_keys=True, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return hashlib.sha256(text.encode()).hexdigest()


def _programs(sig, fresh=False):
    """The store of compiled programs for a signature, kept across the fits of one process
    (the persistent worker); the least recently used stores go beyond _PROGRAMS_MAX."""
    store = None if fresh else _PROGRAMS.pop(sig, None)
    if store is None:
        _PROGRAMS.pop(sig, None)
        store = dict(plain=None, anchors=None, mapped={}, fwdinv={}, sampler={}, constrain={}, fast=None)
    _PROGRAMS[sig] = store
    while len(_PROGRAMS) > _PROGRAMS_MAX:
        _PROGRAMS.pop(next(iter(_PROGRAMS)))
    return store


def _sampler_key(tm):
    """Store key of the sampler built on `tm` (its flag configuration), None without a map."""
    return None if tm is None else tuple(int(f) for f in tm.static)


def _verify_store(vg, store, th0, chains, D):
    """Programs reused from an earlier fit of the same signature: their plain potential at the
    starting point against the model evaluated eagerly there."""
    th = jnp.asarray(np.asarray(th0, dtype=np.float64))
    U = np.asarray(store["plain"]["vg0"](th, jnp.zeros((chains, D + 1)), vg.OFF0)[0])
    with jax.disable_jit():
        U0 = float(vg.pot(th[0], vg.OFF0))
    ok = bool(np.isfinite(U0) and abs(float(U[0]) - U0) <= 1e-6 * max(1.0, abs(U0)))
    phases.mark(f"program cache {'verified' if ok else 'REJECTED'}")
    if not ok:
        print("[DR-G-HMC] cached programs disagree with the model at the starting point; recompiling", flush=True)
    return ok


def _fast_constrain_fn(model, config, full_pp, unflatten, ths, data, trim, store=None):
    """Postprocess against a 1-row config (skips the likelihood recompute). Returns
    (postprocess, its data) or None -> caller falls back to full_pp unless the trimmed and
    full postprocess agree on the first, middle and last draw of every chain."""
    try:
        if store is not None and store["fast"] is not None:
            full_j, trim_j, trim_pp = store["fast"]
        else:
            _, _, trim_pp = _model_info(model, _trim_config(config))
            full_j, trim_j = jax.jit(full_pp), jax.jit(trim_pp)
            if store is not None:
                store["fast"] = (full_j, trim_j, trim_pp)
        chains, kept = ths.shape[:2]
        for c in range(chains):
            for i in sorted({0, kept // 2, kept - 1}):
                z = unflatten(jnp.asarray(ths[c, i]))
                a, b = full_j(z, data), trim_j(z, trim)
                ka = {k for k in a if k != "log_lik"}
                if ka != set(b) - {"log_lik"}:
                    return None
                for k in ka:
                    if k.startswith("z_") or k.startswith("L_corr_"):
                        continue
                    if not np.allclose(np.asarray(a[k]), np.asarray(b[k]),
                                       rtol=1e-10, atol=1e-12):
                        return None
        return trim_pp, trim
    except Exception:
        return None


_SQ2PI = 2.5066282746310002


@jax.custom_jvp
def _sd_logq(t):
    """log q(t), q the half-normal quantile at the normal probability of t: q = Phi^-1((1 + Phi(t)) / 2).
    A standard deviation sampled as scale * q(t) has a standard normal prior on t under a
    half-normal(scale) prior, and where the data say nothing about it its lower tail is the
    normal's instead of the exponential one it has on the log scale."""
    from jax.scipy.special import ndtr, ndtri, log_ndtr
    d = 0.5 * ndtr(jnp.minimum(t, -3.0))
    a = _SQ2PI * d
    low = jnp.log(0.5 * _SQ2PI) + log_ndtr(jnp.minimum(t, -3.0)) + jnp.log1p(a * a / 6.0 + 7.0 * a ** 4 / 120.0)
    high = jnp.log(-ndtri(0.5 * ndtr(-jnp.maximum(t, -3.0))))
    return jnp.where(t < -3.0, low, high)


@_sd_logq.defjvp
def _sd_logq_jvp(primals, tangents):
    t, = primals
    lq = _sd_logq(t)
    return lq, tangents[0] * jnp.exp(_sd_logslope(t, lq))


def _sd_logslope(t, lq):
    """log d log q / dt."""
    return 0.5 * (jnp.exp(2.0 * lq) - t * t) - jnp.log(2.0) - lq


def _sd_t(q):
    """The inverse of q(t), in numpy."""
    from scipy.special import ndtr, ndtri, erf
    q = np.asarray(q, dtype=np.float64)
    return np.where(q < 1.0, ndtri(np.clip(erf(q / np.sqrt(2.0)), 1e-300, 1.0)),
                    -ndtri(np.clip(2.0 * ndtr(-q), 1e-300, 1.0)))


def _sd_prior_scale(fn):
    """The scale of a standard deviation's prior when it has positive density at zero, else None."""
    import numpyro.distributions as dist
    while True:
        if isinstance(fn, (dist.HalfNormal, dist.HalfCauchy)):
            return fn.scale
        if isinstance(fn, dist.Exponential):
            return 1.0 / fn.rate
        low = getattr(fn, "low", None)
        base = getattr(fn, "base_dist", None)
        if low is not None and base is not None and hasattr(base, "scale"):
            return base.scale if bool(np.all(np.asarray(low) == 0.0)) else None
        if base is None:
            return None
        fn = base


def _model_info(model, config):
    """Unconstrained site shapes, potential and postprocess (constrained sites plus
    deterministics) from one abstract trace. numpyro's initialize_model gives the same
    three but runs the model eagerly, compiling every primitive separately."""
    from numpyro import handlers
    from numpyro.distributions.transforms import biject_to
    from numpyro.infer.util import potential_energy, constrain_fn

    supports, sd_scale = {}, {}

    def run():
        init = handlers.substitute(handlers.seed(model, rng_seed=0),
                                   substitute_fn=init_to_uniform(radius=0.0))
        tr = handlers.trace(init).get_trace(config)
        out = {}
        for name, site in tr.items():
            if site["type"] == "sample" and not site["is_observed"]:
                supports[name] = site["fn"].support
                out[name] = site["value"]
                sc = _sd_prior_scale(site["fn"]) if name.startswith("sigma_") else None
                if sc is not None and not isinstance(sc, jax.core.Tracer):
                    sd_scale[name] = np.broadcast_to(np.asarray(sc, dtype=np.float64), np.shape(site["value"]))
        return out

    vals = jax.eval_shape(run)
    shapes = {k: tuple(biject_to(supports[k]).inv.forward_shape(vals[k].shape)) for k in vals}
    pot = lambda z, OFF: potential_energy(model, (_with_data(config, OFF["data"], OFF["off"]),), {}, z)
    pot.sd_scale = sd_scale
    post = lambda z, data: constrain_fn(model, (_with_data(config, data),), {}, z, return_deterministic=True)
    return shapes, pot, post


def _build(model, config):
    """(vg, unflatten, keys, offsets, D, postprocess_fn) for flat unconstrained vectors, the standard
    deviations on their prior-quantile scale (the coordinates both samplers draw in). vg(v, OFF)
    takes the runtime bundle; vg.OFF0 is the bundle with a zero offset, vg.data / vg.trim its
    data, vg.pot the potential, and postprocess_fn(z, data) constrains against `data`."""
    shapes, potential_fn, postprocess_fn = _model_info(model, config)
    keys = sorted(shapes)
    off, o = {}, 0
    for k in keys:
        sz = int(np.prod(shapes[k]))
        off[k] = (slice(o, o + sz), shapes[k])
        o += sz
    D = o

    # The sites partition v in key order, so the cotangent of v is their cotangents
    # concatenated. Left to autodiff, XLA assembles it from one padded full-length vector
    # per site, which costs more than the likelihood gradient itself whenever v is not
    # the program's own input (any reparameterisation in front of the model).
    @jax.custom_vjp
    def split(v):
        return {k: v[off[k][0]].reshape(off[k][1]) for k in keys}

    def split_fwd(v):
        return split(v), None

    def split_bwd(_, ct):
        return (jnp.concatenate([jnp.reshape(ct[k], (-1,)) for k in keys]),)

    split.defvjp(split_fwd, split_bwd)
    sd_keys = [k for k in keys if k in potential_fn.sd_scale]
    sd_idx = np.concatenate([np.arange(off[k][0].start, off[k][0].stop) for k in sd_keys]).astype(int) \
        if sd_keys else np.zeros(0, int)
    sd_log = np.log(np.concatenate([potential_fn.sd_scale[k].reshape(-1) for k in sd_keys])) if sd_keys else np.zeros(0)

    def unflatten(v):
        if not sd_idx.size:
            return split(v)
        return split(v.at[sd_idx].set(_sd_logq(v[sd_idx]) + jnp.asarray(sd_log, dtype=v.dtype)))

    def log_jac(v):
        if not sd_idx.size:
            return 0.0
        t = v[sd_idx]
        return jnp.sum(_sd_logslope(t, _sd_logq(t)))

    def to_sampled(th):
        th = np.array(th, dtype=np.float64)
        if sd_idx.size:
            th[..., sd_idx] = _sd_t(np.exp(th[..., sd_idx] - sd_log))
        return th

    unflatten.log_jac, unflatten.to_sampled = log_jac, to_sampled
    # jitted: every program that embeds the potential then reuses one traced jaxpr instead
    # of re-tracing the model (XLA inlines the call, so the compiled code is unchanged)
    pot = jax.jit(lambda v, OFF: potential_fn(unflatten(v), OFF) - log_jac(v))
    vg = jax.value_and_grad(pot)
    vg.pot = pot
    vg.OFF0 = _off0(config)
    vg.data, vg.trim = vg.OFF0["data"], vg.OFF0["trim"]
    return vg, unflatten, keys, off, D, postprocess_fn


def _transport(model, config, ctrl, off, unflatten, progress_bar=False):
    """tmap.build when ctrl['transport'] and the model has random-effect blocks, else None."""
    if not ctrl.get("transport", True):
        return None
    import tmap
    tm = tmap.build(model, config, off, unflatten, links=bool(ctrl.get("transport_links", True)))
    if tm is not None and progress_bar:
        print(f"[DR-G-HMC] group transport map on {', '.join(tm.specs)}", flush=True)
    return tm


def _sampler_vg(vg, tm):
    """The sampler's value-and-grad: (v, OFF) -> (U, g), or (v, OFF, AUX) with a transport map,
    whose potential the map keeps so that a view of it (another block configuration) wraps the
    same potential."""
    if tm is None:
        return vg
    tm.pot_off = vg.pot
    return tm.wrap(vg.pot)


_PILOT_ABOVE = 7500
_DISAGREE = 2.0
_PILOT_LEVELS = 5000


def _subset_config(config, group, levels):
    """The model restricted to the rows of `levels` (0-based) of grouping factor `group`:
    every length-N array is cut to those rows and the factor is re-indexed to 1..len(levels)."""
    N = int(config["N"])
    idx = next(np.asarray(r["index"], dtype=int) - 1 for r in config["random_effects"]
               if r["group"] == group)
    new_id = np.full(int(idx.max()) + 1, -1)
    new_id[levels] = np.arange(len(levels))
    rows = np.nonzero(new_id[idx] >= 0)[0]

    def cut(o):
        if isinstance(o, dict):
            return {k: ([v[i] for i in rows] if isinstance(v, list) and len(v) == N else cut(v))
                    for k, v in o.items()}
        if isinstance(o, list):
            return [cut(it) if isinstance(it, dict) else it for it in o]
        return o

    c = cut(config)
    c["N"] = int(len(rows))
    for r in c["random_effects"]:
        if r["group"] == group:
            r["index"] = (new_id[np.asarray(r["index"], dtype=int) - 1] + 1).tolist()
            r["n_groups"] = int(len(levels))
    return c


def _pilot_start(model, config, keys, off, D, tm, chains, warmup, seed, ctrl, parallel, progress_bar):
    """Starting positions [chains, D] from the warmup of the same model on a random subset of
    the levels of its largest grouping factor: every site of unchanged shape is copied, the
    sampled levels keep their standardised effects and the others start at zero."""
    sb = max(tm.sbs, key=lambda s: s.n)
    levels = np.sort(np.random.default_rng(int(seed)).choice(sb.n, _PILOT_LEVELS, replace=False))
    if progress_bar:
        print(f"[DR-G-HMC] starting values from a pilot warmup on {_PILOT_LEVELS} of {sb.n} "
              f"levels of {sb.name}", flush=True)
    cfg_p = _subset_config(config, sb.name, levels)
    vg_p, unfl_p, keys_p, off_p, D_p, _ = _build(model, cfg_p)
    tm_p = _transport(model, cfg_p, ctrl, off_p, unfl_p)
    vg_s_p = _sampler_vg(vg_p, tm_p)
    with _group(progress_bar, warmup, "Pilot warmup"):
        _, _, tht, aux, _ = _adapt_stages(vg_p, vg_s_p, tm_p, D_p, chains, warmup, seed + 7, ctrl, parallel,
                                          progress_bar, np.zeros((chains, D_p)), True, decide=False)
    th_p = tm_p.forward_chains(tht, aux, vg_p.trim)
    th0 = np.zeros((chains, D))
    for k in keys:
        if k not in off_p:
            continue
        (sl, shp), (sl_p, shp_p) = off[k], off_p[k]
        if tuple(shp) == tuple(shp_p):
            th0[:, sl] = th_p[:, sl_p]
        elif len(shp) == len(shp_p) and shp[:-1] == shp_p[:-1] and shp[-1] == sb.n and shp_p[-1] == len(levels):
            full = np.zeros((chains,) + tuple(shp))
            full[..., levels] = th_p[:, sl_p].reshape((chains,) + tuple(shp_p))
            th0[:, sl] = full.reshape(chains, -1)
    return th0


class _Programs:
    """The warmup's compiled programs: the plain window and its gradient, the anchors program,
    and per map configuration (a tuple of 0/1 flags over the super-blocks: 1 = that block's
    candidates non-centred) the mapped window and the gradient over the chains. Everything is
    lowered on the calling thread (NumPyro's handler stack is not shared safely across threads)
    and compiled on one worker thread in the order queued, so that the plain stage can start
    at once; a configuration never queued is compiled when first asked for."""

    def __init__(self, vg_plain, tm, D, chains, ctrl, parallel, off0, store=None):
        import threading
        self.tm, self.D, self.chains, self.ctrl, self.parallel, self.off0 = tm, D, chains, ctrl, parallel, off0
        self.progs, self.ready, self.err, self.queue = {}, {}, {}, []
        self.lock = threading.Lock()
        self.store = store
        self.plain = {}
        self.plain_ready = threading.Event()
        self.anchors_ready = threading.Event()
        hit = store is not None and store["plain"] is not None
        if hit:
            self.plain = store["plain"]
            self.plain_ready.set()
            tm._anch = store["anchors"]
            self.anchors_ready.set()
            for flags, progs in store["mapped"].items():
                self.progs[flags] = progs
                self.ready[flags] = threading.Event()
                self.ready[flags].set()
            for flags, (fwd, inv) in store["fwdinv"].items():
                view = tm.view(flags)
                view._fwd, view._inv = fwd, inv
        else:
            low_vg0 = _chain_map(vg_plain, (0, 0, None), parallel).lower(jnp.zeros((chains, D)),
                                                                        jnp.zeros((chains, D + 1)), off0)
            low_plain = _lower_window(vg_plain, True, D, chains, ctrl, parallel, jnp.zeros((chains, D + 1)), off0)
            self.queue.append(("plain", (low_vg0, low_plain)))
        self.thread = threading.Thread(target=self._work, daemon=True)
        self.thread.start()
        if not hit:
            self.queue.append(("anchors", tm.lower_anchors(vg_plain_base(vg_plain, D), chains, D, off0, parallel)))
        self.base = (0,) * len(tm.sbs)
        self.add(self.base)
        phases.mark("warmup programs reused" if hit else "warmup programs lowered")

    def vg_m(self, flags):
        f = self.tm.view(flags).wrap(self.tm.pot_off)
        return lambda v, a, OFF: f(v, OFF, a)

    def add(self, flags):
        """Lower a configuration's programs (calling thread) and queue them for compilation."""
        import threading
        flags = tuple(int(f) for f in flags)
        with self.lock:
            if flags in self.ready:
                return
            self.ready[flags] = threading.Event()
        vg_m = self.vg_m(flags)
        zeros = aux_zeros(self.tm, self.chains)
        low_w = _lower_window(vg_m, True, self.D, self.chains, self.ctrl, self.parallel, zeros, self.off0)
        low_g = _chain_map(vg_m, (0, 0, None), self.parallel).lower(jnp.zeros((self.chains, self.D)), zeros, self.off0)
        self.queue.append((flags, (low_w, low_g)))

    def _work(self):
        done = 0
        while True:
            if done >= len(self.queue):
                if getattr(self, "closed", False):
                    return
                time.sleep(0.01)
                continue
            key, low = self.queue[done]
            done += 1
            try:
                if key == "plain":
                    self.plain["vg0"] = low[0].compile()
                    self.plain["window"] = low[1].compile()
                    self.plain_ready.set()
                    phases.mark("plain programs compiled")
                    if self.store is not None:
                        self.store["plain"] = self.plain
                elif key == "anchors":
                    self.tm._anch = low.compile()
                    self.anchors_ready.set()
                    phases.mark("anchors program compiled")
                    if self.store is not None:
                        self.store["anchors"] = self.tm._anch
                else:
                    self.progs[key] = dict(window=low[0].compile(), vg0=low[1].compile())
                    self.ready[key].set()
                    phases.mark(f"mapped programs compiled {key}")
                    if self.store is not None:
                        self.store["mapped"][key] = self.progs[key]
            except BaseException as e:
                self.err[key] = e
                (self.plain_ready if key == "plain" else self.anchors_ready if key == "anchors" else self.ready[key]).set()

    def _check(self, key):
        if key in self.err:
            raise self.err.pop(key)

    def wait_plain(self):
        self.plain_ready.wait()
        self._check("plain")
        return self.plain

    def wait_anchors(self):
        self.anchors_ready.wait()
        self._check("anchors")

    def get(self, flags):
        """The mapped window and gradient for a configuration, compiled if need be."""
        flags = tuple(int(f) for f in flags)
        self.add(flags)
        self.ready[flags].wait()
        self._check(flags)
        return self.progs[flags]

    def close(self):
        self.closed = True
        if self.store is not None:
            for flags, view in self.tm._views.items():
                if view._fwd is not None or view._inv is not None:
                    self.store["fwdinv"][flags] = (view._fwd, view._inv)


def vg_plain_base(vg_plain, D):
    """The plain gradient (v, OFF) -> (U, g) from the flagged window gradient (flag off)."""
    return lambda v, OFF: vg_plain(v, jnp.zeros(D + 1), OFF)


def _wide_flags(tm, ctrl, aux):
    """The anchors with every super-block's wide blocks non-centred under
    control$transport_wide = FALSE, as they are otherwise."""
    if bool(ctrl.get("transport_wide", True)):
        return aux
    return tm.with_flags(aux, [1.0 if sb.cand is not None else 0.0 for sb in tm.sbs])


def _adapt_stages(vg, vg_s, tm, D, chains, warmup, seed, ctrl, parallel, progress_bar, th0, cold,
                  init=None, decide=True, on_decide=None, store=None):
    H = int(ctrl["adapt_windows"])
    if warmup < 5 * H:
        raise ValueError(f"DR-G-HMC with transport needs warmup >= {5 * H}; got {warmup}")
    off0 = vg.OFF0
    mask_np = tm.scale_mask(D)
    mask = jnp.asarray(mask_np)

    def vg_plain(v, ax, OFF):
        # ax = [fixed values, flag]: with the flag set the scale coordinates are held at the
        # fixed values (unit-normal dummies in their place), so the plain and the frozen
        # stages share one compiled window
        on = mask & (ax[D] > 0.5)
        U, g = vg(jnp.where(on, ax[:D], v), OFF)
        return U + 0.5 * jnp.sum(jnp.where(on, v * v, 0.0)), jnp.where(on, v, g)

    progs = _Programs(vg_plain, tm, D, chains, ctrl, parallel, off0, store)
    c_plain = progs.wait_plain()

    def mapped(aux):
        return progs.get(tm.flags(aux))

    def vg_m(aux):
        return progs.vg_m(tm.flags(aux))

    def build(th, w, sd, flags=None):
        _, _, tf, _, mf = _self_adapt(vg_plain, D, chains, w, sd, ctrl, parallel, progress_bar,
                                      th0=np.where(mask_np, 0.0, th), return_mean=True, cache=c_plain, stage="map",
                                      aux=np.concatenate([th, np.ones((chains, 1))], 1), OFF=off0)
        phases.mark("map build: frozen window done")
        progs.wait_anchors()
        aux = tm.anchors_chains(vg, np.where(mask_np, th, mf), off0, parallel)
        phases.mark("map build: anchors computed")
        return np.where(mask_np, th, tf), (aux if flags is None else tm.with_flags(aux, flags))

    th = np.asarray(th0)
    if cold:
        w0, wf, w1 = max(H, warmup // 5), max(H, warmup // 10), max(H, warmup // 5)
        ci, ce, th, _ = _self_adapt(vg_plain, D, chains, w0, seed, ctrl, parallel, progress_bar, th0=th,
                                    cache=c_plain, aux=np.zeros((chains, D + 1)), stage="start", OFF=off0)
        phases.mark("stage start done")
        th, aux = build(th, wf, seed + 1)
        phases.mark("stage map build 1 done")
        ci, ce, tht, _ = _self_adapt(vg_m(aux), D, chains, w1, seed + 2, ctrl, parallel, progress_bar,
                                     th0=tm.inverse_chains(th, aux, vg.trim), aux=tm.parts(aux), cache=mapped(aux),
                                     init=init, imm0=ci, eps0=ce, stage="mapped", OFF=off0)
        phases.mark("stage mapped done")
        th = tm.forward_chains(tht, aux, vg.trim)
        th, aux = build(th, wf, seed + 3)
        phases.mark("stage map build 2 done")
        rest = warmup - w0 - 2 * wf - w1
    else:
        wf = max(H, warmup // 4)
        th, aux = build(th, wf, seed + 1)
        rest = warmup - wf
        ci = ce = None
    if decide:
        aux = _wide_flags(tm, ctrl, aux)
        if on_decide is not None:
            on_decide(tuple(int(f) for f in tm.flags(aux)))
    init = init if tuple(tm.flags(aux)) == progs.base else None
    imm, eps, tht, _, (between, within) = _self_adapt(
        vg_m(aux), D, chains, rest, seed + 4, ctrl, parallel, progress_bar,
        th0=tm.inverse_chains(th, aux, vg.trim), aux=tm.parts(aux), cache=mapped(aux), spread=True, init=init,
        imm0=ci, eps0=ce, stage="tuning", OFF=off0)
    phases.mark("stage tuning done")
    slow = mask_np & (between > _DISAGREE * within)
    if slow.any():
        if progress_bar:
            print(f"[DR-G-HMC] chains still disagree on {int(slow.sum())} scale parameter(s); "
                  f"extending warmup by {max(H, rest // 2) + wf + rest} iterations", flush=True)
        imm, eps, tht, _ = _self_adapt(vg_m(aux), D, chains, max(H, rest // 2), seed + 5, ctrl, parallel, progress_bar,
                                       th0=tht, aux=tm.parts(aux), cache=mapped(aux), pooled=True, init=init,
                                       stage="extended", OFF=off0)
        th, aux = build(tm.forward_chains(tht, aux, vg.trim), wf, seed + 6, tm.flags(aux))
        imm, eps, tht, _ = _self_adapt(vg_m(aux), D, chains, rest, seed + 7, ctrl, parallel, progress_bar,
                                       th0=tm.inverse_chains(th, aux, vg.trim), aux=tm.parts(aux), cache=mapped(aux),
                                       imm0=imm, eps0=eps, init=init, stage="tuning", OFF=off0)
        phases.mark("stage extended done")
    progs.close()
    return imm, eps, tht, aux, None


def _adapt(model, config, keys, off, vg, vg_s, tm, D, chains, warmup, seed, ctrl, parallel,
           progress_bar, th0, user_init=False, init=None, on_decide=None, title="DR-G-HMC warmup", store=None):
    """Adaptation. Without a transport map this is _self_adapt. With one:

    1. the scale parameters of the mapped blocks are held where they are while everything
       else is sampled, so the groups' effects settle around those scales;
    2. each chain builds its map at its own effects averaged over the second half of 1;
    3. the rest adapts the step size and masses in mapped coordinates.

    Chains that do not start from a pilot first run a fifth of the warmup in the model's own
    coordinates, then 1-2, a fifth in mapped coordinates to reach the posterior, and 1-3 again
    so the map used for sampling is built there.

    If, over the draws the final masses were estimated from, the chains' means of some scale
    parameter of a mapped block vary more than _DISAGREE times its variance within a chain,
    that parameter is still travelling: its within-chain variance understates how far it has
    to go, which makes its steps small and keeps it slow. Warmup is then extended: a pass
    whose masses are the variances over all chains' draws together, so the spread between
    chains counts and the chains can finish travelling; the maps rebuilt where they arrive;
    and a final pass that starts from those masses and that step size and re-estimates within
    chains. Models whose chains agree are unaffected.

    A model whose largest grouping factor has more than _PILOT_ABOVE levels first runs this
    warmup on a random _PILOT_LEVELS of them and starts from where that ends, which puts the
    scale parameters near their posterior before the full data are used. Returns (imm, eps,
    positions in sampler coordinates, aux [chains, A], wall)."""
    t0 = time.time()
    if tm is None:
        with _group(progress_bar, warmup, title):
            imm, eps, th, wall = _self_adapt(vg, D, chains, warmup, seed, ctrl, parallel, progress_bar, th0=th0,
                                             stage="tuning", OFF=vg.OFF0)
        return imm, eps, th, None, wall
    cold = True
    if not user_init and max(sb.n for sb in tm.sbs) > _PILOT_ABOVE:
        th0 = _pilot_start(model, config, keys, off, D, tm, chains, warmup, seed, ctrl, parallel, progress_bar)
        cold = False
    with _group(progress_bar, warmup, title):
        imm, eps, tht, aux, _ = _adapt_stages(vg, vg_s, tm, D, chains, warmup, seed, ctrl, parallel,
                                              progress_bar, th0, cold, init, on_decide=on_decide, store=store)
    return imm, eps, tht, aux, time.time() - t0


def _group(on, total, title):
    import contextlib
    import progress
    return progress.group(total, title) if on else contextlib.nullcontext()


def ll_offset(postprocess_fn, unflatten, config, th, data):
    """Per-observation weighted log-likelihood at the unconstrained point `th`. Summing
    log_lik - offset keeps the potential near zero around `th`, so its float32 value
    resolves energy differences (a sum of ~|U| rounds to |U| * 6e-8, 0.03 at |U| = 5e5)."""
    ll = jnp.asarray(jax.jit(lambda v, d: postprocess_fn(unflatten(v), d)["log_lik"])(jnp.asarray(th), data))
    counts = config.get("counts")
    return ll if counts is None else ll * jnp.asarray(counts, dtype=ll.dtype)


_COEF = {
    "bcss3": ((_B1, _B2, _B2, _B1), (_A1, _A2, _A1)),
    "bcss2": ((_LAM, 1.0 - 2.0 * _LAM, _LAM), (0.5, 0.5)),
    "leapfrog": ((0.5, 0.5), (1.0,)),
}


def _leap_flip(vg, integ, th, rho, g_th, eps, imm):
    """One reversible integrator step + momentum flip (involution). The sub-steps run in a
    fori_loop so the gradient is compiled once per program, not once per sub-step."""
    B, A = (jnp.asarray(c, dtype=th.dtype) for c in _COEF[integ])
    n = len(_COEF[integ][1])

    def sub(i, c):
        th, rho, U, g = c
        rho = rho - B[i] * eps * g
        th = th + A[i] * eps * imm * rho
        U, g = vg(th)
        return th, rho, U, g
    th, rho, U, g = jax.lax.fori_loop(0, n, sub, (th, rho, jnp.zeros((), th.dtype), g_th))
    return th, -(rho - B[n] * eps * g), U, g


_KEEP = 64


def _dynamic_window(vg_in, with_aux, D, ctrl, parallel):
    """One adaptation window of run-time length n (pmap over chains): a fori_loop of
    single-proposal steps with per-chain Robbins-Monro on log(eps) and Welford moments."""
    acc_t, gamma, integ = ctrl["acc_target"], ctrl["gamma"], ctrl["integrator"]
    _map = lambda f, in_axes=None: _chain_map(f, in_axes, parallel)

    def window(key, th0, rho0, U0, g0, le0, ax, imm, lr, n, OFF):
        vg = (lambda v: vg_in(v, ax, OFF)) if with_aux else (lambda v: vg_in(v, OFF))
        msq = jnp.sqrt(1.0 / imm)

        def body(i, c):
            th, rho, U, g, le, mean, m2, cnt, asum, gs, buf = c
            k1, k2 = jax.random.split(jax.random.fold_in(key, i))
            xi = msq * jax.random.normal(k1, (D,))
            rho = jnp.sqrt(1.0 - gamma) * rho + jnp.sqrt(gamma) * xi
            th1, rho1, U1, g1 = _leap_flip(vg, integ, th, rho, g, jnp.exp(le), imm)
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
            return (th, rho, U, g, le, mean, m2, cnt, asum + a, gs + g * g, buf.at[(i * jnp.minimum(_KEEP, n)) // n].set(th))

        thf, rhof, Uf, gf, lef, mean, m2, cnt, asum, gs, buf = jax.lax.fori_loop(
            0, n, body, (th0, rho0, U0, g0, le0, jnp.zeros(D), jnp.zeros(D), 0.0, 0.0, jnp.zeros(D),
                         jnp.zeros((_KEEP, D))))
        return thf, rhof, Uf, gf, lef, mean, m2, asum / cnt, gs, buf

    return _map(window, in_axes=(0, 0, 0, 0, 0, 0, 0, None, None, None, None))


def _lower_window(vg_in, with_aux, D, chains, ctrl, parallel, aux0, OFF0):
    """The dynamic window lowered on representative arguments; its compiled executable takes
    any input placement, so it serves every later call of any length."""
    z = jnp.zeros((chains, D))
    args = (jax.random.split(_tagged(0, 4), chains), z, z, jnp.zeros(chains), z, jnp.zeros(chains),
            aux0, jnp.ones(D), 0.1, jnp.asarray(1), OFF0)
    return _dynamic_window(vg_in, with_aux, D, ctrl, parallel).lower(*args)


def _self_adapt(vg, D, chains, warmup, seed, ctrl, parallel, progress_bar=False,
                th0=None, aux=None, return_mean=False, cache=None, imm0=None, eps0=None,
                spread=False, pooled=False, init=None, stage=None, OFF=None):
    """DR-native adaptation: per-chain RM on log(eps) -> acc_target, progressive
    diagonal mass from accumulated within-chain Welford variance. `warmup`
    iterations are split exactly across the adaptation windows. Returns
    (imm, eps, final positions, wall). th0 [chains, D] sets the starting points.
    With `cache` (a dict kept by the caller for one `vg`) the window takes its length at
    run time, so repeated calls of any length compile once. `imm0` and `eps0` start from an
    earlier call's masses and step size instead of unit masses. With `spread` the return
    ends with (between, within): per coordinate, the variance of the chains' means and the
    average variance within a chain over the draws the masses were estimated from. With
    `pooled` every mass is the variance over all chains' draws together. `init(th, aux)`
    -> (U, g), when given, supplies the starting values from an already compiled program
    instead of compiling `vg` over the chains for that alone."""
    acc_t = ctrl["acc_target"]
    gamma = ctrl["gamma"]
    integ = ctrl["integrator"]
    sched = _SCHED10 if ctrl["adapt_windows"] == 10 else tuple(
        float(x) for x in np.geomspace(0.15, 0.02, ctrl["adapt_windows"]))
    H = len(sched)
    if warmup < H:
        raise ValueError(f"DR-G-HMC needs warmup >= adapt_windows ({H}); got {warmup}")
    lengths = [warmup // H + (1 if w < warmup % H else 0) for w in range(H)]
    _map = lambda f, in_axes=None: _chain_map(f, in_axes, parallel)
    vg_in = vg

    def make_window(nw):
        def window(key, th0, rho0, U0, g0, le0, ax, imm, lr, OFF):
            vg = (lambda v: vg_in(v, OFF)) if aux is None else (lambda v: vg_in(v, ax, OFF))
            msq = jnp.sqrt(1.0 / imm)

            def step(c, k):
                th, rho, U, g, le, mean, m2, cnt, gs, buf = c
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
                slot = jnp.floor((cnt - 1.0) * min(_KEEP, nw) / nw + 1e-6).astype(jnp.int32)
                return (th, rho, U, g, le, mean, m2, cnt, gs + g * g, buf.at[slot].set(th)), a

            (thf, rhof, Uf, gf, lef, mean, m2, _, gs, buf), accs = jax.lax.scan(
                step, (th0, rho0, U0, g0, le0, jnp.zeros(D), jnp.zeros(D), 0.0, jnp.zeros(D),
                       jnp.zeros((_KEEP, D))),
                jax.random.split(key, nw))
            return thf, rhof, Uf, gf, lef, mean, m2, jnp.mean(accs), gs, buf

        return _map(window, in_axes=(0, 0, 0, 0, 0, 0, 0, None, None, None))

    vg0_axes = (0, None) if aux is None else (0, 0, None)
    if cache is None:
        pws = {n: make_window(n) for n in sorted(set(lengths))}
        vg0 = _map(vg_in, vg0_axes)
    else:
        if "window" not in cache:
            cache["window"] = _dynamic_window(vg_in, aux is not None, D, ctrl, parallel)
        if "vg0" not in cache:
            cache["vg0"] = _map(vg_in, vg0_axes)
        pws = {n: (lambda *a, n=n: cache["window"](*a[:-1], jnp.asarray(n), a[-1])) for n in set(lengths)}
        vg0 = cache["vg0"]
    t0 = time.time()
    # fresh inputs go through an identity map first: pmap compiles its program once for
    # host-placed arguments and again for its own outputs, so every window call must see
    # the same placement
    ident = _ident_map(parallel)
    th = ident(jnp.zeros((chains, D)) if th0 is None else jnp.asarray(th0))
    AX = jnp.zeros((chains, 1)) if aux is None else jax.tree.map(jnp.asarray, aux)
    if init is not None:
        U, g = init(th, AX)
    else:
        U, g = vg0(th, OFF) if aux is None else vg0(th, AX, OFF)
    rho = _normal_map(D, parallel)(jax.random.split(_tagged(seed, 4), chains))
    imm = jnp.ones(D) if imm0 is None else jnp.asarray(imm0)
    le = ident(jnp.asarray(np.full(chains, np.log(0.02 if eps0 is None else eps0))))
    acc_mean = np.zeros((chains, D)); acc_m2 = np.zeros((chains, D))
    acc_n = np.zeros(chains)
    acc_g2 = np.zeros((chains, D))
    kept = []
    burn = max(1, int(round(ctrl["mass_burn_frac"] * H)))
    mass_end = max(burn + 1, int(round(ctrl["mass_end_frac"] * H)))
    late_sum, late_n = 0.0, 0
    spread_out = None
    bar = _pbar(warmup, "DR-G-HMC warmup", stage=stage) if progress_bar else None
    for w, lr0 in enumerate(sched):
        nw = lengths[w]
        lr = (0.10 * 0.65 ** (w - mass_end)) if w >= mass_end else lr0
        keys = jax.random.split(jax.random.fold_in(_tagged(seed, 3), w), chains)
        th, rho, U, g, le, mn, m2, am, gsq, buf = pws[nw](keys, th, rho, U, g, le, AX, imm, lr, OFF)
        # a chain that accepts nothing while the others sample is parked somewhere the shared
        # step cannot leave (a pilot start the full data rule out): restart it from the chain that
        # is doing best; the draws it would have given are not draws of anything
        am_np = np.asarray(am)
        stuck = np.where((am_np < 0.02) & (np.median(am_np) > 0.2))[0]
        if stuck.size and stuck.size < chains:
            best = int(np.argmax(am_np))
            th_np, rho_np, le_np = np.array(th), np.array(rho), np.array(le)
            jit = np.random.default_rng(int(seed) + 101 + w).normal(size=(stuck.size, D)) * 1e-3
            th_np[stuck] = th_np[best] + jit
            rho_np[stuck] = rho_np[best]
            le_np[stuck] = le_np[best]
            th, rho, le = ident(jnp.asarray(th_np)), ident(jnp.asarray(rho_np)), ident(jnp.asarray(le_np))
            U, g = (init(th, AX) if init is not None else (vg0(th, OFF) if aux is None else vg0(th, AX, OFF)))
            if progress_bar:
                print(f"[DR-G-HMC] chain(s) {', '.join(str(int(c) + 1) for c in stuck)} accepted nothing in a warmup "
                      f"window; restarted from chain {best + 1}", flush=True)
        if return_mean and w >= H // 2:
            late_sum = late_sum + np.asarray(mn) * nw
            late_n += nw
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
            grand = (acc_mean * acc_n[:, None]).sum(0) / acc_n.sum()
            spread_out = ((acc_n[:, None] * (acc_mean - grand) ** 2).sum(0) / acc_n.sum(), within)
            acc_g2 = acc_g2 + np.asarray(gsq)
            kept.append(np.asarray(buf)[:, :min(_KEEP, nw)].reshape(-1, D))
            if pooled:
                within = within + spread_out[0]
            else:
                q = np.quantile(np.concatenate(kept), [0.25, 0.75], axis=0)
                within = np.sqrt(((q[1] - q[0]) / 1.3489795) ** 2 * (acc_n.sum() / np.maximum(acc_g2.sum(0), 1e-300)))
            ne = float(acc_n.sum())
            imm = jnp.asarray((ne / (ne + 5.0)) * within + 1e-3 * (5.0 / (ne + 5.0)))
        if w < H - 1 and float(np.asarray(am).mean()) > 0.97:
            le = le + float(np.log(2.0))
    if bar is not None:
        bar.close()
    eps = float(np.exp(np.median(np.asarray(le))))
    if spread:
        return np.asarray(imm), eps, np.asarray(th), time.time() - t0, spread_out
    if return_mean:
        return np.asarray(imm), eps, np.asarray(th), time.time() - t0, late_sum / late_n
    return np.asarray(imm), eps, np.asarray(th), time.time() - t0


def _sampler_fns(vg_off, D, ctrl, parallel, tm=None):
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

    last = {1: 0, 2: 2, 3: 6}[kmax]

    def scanner(c, ts, keep, ckey, IMM, MSQ, E0, OFF, AUX):
        vg = (lambda v: vg_off(v, OFF)) if tm is None else (lambda v: vg_off(v, OFF, AUX))
        eps_k = E0 * RSC
        tiny = jnp.finfo(c[0].dtype).tiny
        EIDX = jnp.asarray(_DR_EPS)

        def kinetic(rho):
            return 0.5 * jnp.sum(IMM * rho * rho)

        def ap(x):
            return jnp.where(jnp.isfinite(x), jnp.minimum(1.0, x), 0.0)

        def sel(p, a, b):
            return jax.tree.map(lambda x, y: jnp.where(p, x, y), a, b)

        def dr_step(carry, x):
            t, kept = x
            # Delayed rejection as a while_loop over at most 7 integrator steps (one compiled
            # copy of the gradient): 0 proposal 1; 1 proposal 2; 2 its ghost; 3 proposal 3;
            # 4 its ghost; 5 proposal 2 from 3; 6 that one's ghost. S2 holds proposal 2 and
            # later the ghost-chain proposal, S3 proposal 3, R the accepted state.
            th, rho, U_th, g_th = carry
            k1, k2 = jax.random.split(jax.random.fold_in(ckey, t))
            xi = MSQ * jax.random.normal(k1, (D,))
            rho = SQ1MG * rho + SQG * xi
            H0 = U_th + kinetic(rho)
            u = jax.random.uniform(k2, (3,))
            X0 = (th, rho, U_th, g_th)
            z = jnp.zeros((), th.dtype)

            def body(s):
                k, done, S2, S3, R, stage, H2, H3, a1, a2, a1g3 = s
                src = sel((k == 2) | (k == 6), S2, sel((k == 4) | (k == 5), S3, X0))
                L = _leap_flip(vg, integ, src[0], src[1], src[3], eps_k[EIDX[k]], IMM)
                HL = L[2] + kinetic(L[1])
                a1n = ap(jnp.exp(H0 - HL))
                a1gn = ap(jnp.exp(H2 - HL))
                a2n = ap(jnp.exp(H0 - H2) * (1.0 - a1gn) / jnp.maximum(1.0 - a1, tiny))
                a1g3n = ap(jnp.exp(H3 - HL))
                a2g = ap(jnp.exp(H3 - H2) * (1.0 - a1gn) / jnp.maximum(1.0 - a1g3, tiny))
                a3 = ap(jnp.exp(H0 - H3) * (1.0 - a1g3) / jnp.maximum(1.0 - a1, tiny)
                        * (1.0 - a2g) / jnp.maximum(1.0 - a2, tiny))
                a1 = jnp.where(k == 0, a1n, a1)
                a2 = jnp.where(k == 2, a2n, a2)
                a1g3 = jnp.where(k == 4, a1g3n, a1g3)
                hit1 = (k == 0) & (u[0] < a1n)
                hit2 = (k == 2) & (u[1] < a2n)
                hit3 = (k == 6) & (u[2] < a3)
                R = sel(hit1, L, sel(hit2, S2, sel(hit3, S3, R)))
                stage = jnp.where(hit1, 1.0, jnp.where(hit2, 2.0, jnp.where(hit3, 3.0, stage)))
                S2 = sel((k == 1) | (k == 5), L, S2)
                S3 = sel(k == 3, L, S3)
                H2 = jnp.where((k == 1) | (k == 5), HL, H2)
                H3 = jnp.where(k == 3, HL, H3)
                return (k + 1, hit1 | hit2 | hit3 | (k == last), S2, S3, R, stage, H2, H3, a1, a2, a1g3)

            s = jax.lax.while_loop(lambda s: ~s[1], body, (
                0, False, X0, X0, (th, -rho, U_th, g_th), -1.0 + z, z, z, z, z, z))
            th_n, rho_n, U_n, g_n = s[4]
            stage = s[5]
            rho_n = jnp.where(stage > 0, -rho_n, rho_n)
            out = th_n if tm is None else jax.lax.cond(kept, lambda v: tm.forward(v, AUX, OFF["trim"])[0],
                                                       jnp.zeros_like, th_n)
            return (th_n, rho_n, U_n, g_n), (out, stage, H0)

        return jax.lax.scan(dr_step, c, (ts, keep))

    def setup(th0, rho0, OFF, AUX):
        U0, g0 = vg_off(th0, OFF) if tm is None else vg_off(th0, OFF, AUX)
        return (th0, rho0, U0, g0)

    ax = None if tm is None else 0
    return (_chain_map(setup, (0, 0, None, ax), parallel),
            _chain_map(scanner, (0, None, None, 0, None, None, None, None, ax), parallel))


def _identity(x):
    return x


def _normal(k, D):
    return jax.random.normal(k, (D,))


@functools.lru_cache(maxsize=None)
def _ident_map(parallel):
    """The identity over chains, built once per process: pmap keeps every function it has
    wrapped, so a fresh one per call would hold a compiled program per adaptation stage."""
    return _chain_map(_identity, None, parallel)


@functools.lru_cache(maxsize=None)
def _normal_map(D, parallel):
    return _chain_map(functools.partial(_normal, D=D), None, parallel)


def _chain_map(f, in_axes=None, parallel=True):
    """pmap across devices, or one chain at a time under jit: lax.map keeps each chain's
    delayed-rejection lax.cond a real branch, where vmap would run every stage for all, and
    the native kernels have no batching rule. in_axes None maps every argument."""
    if parallel:
        return jax.pmap(f, axis_name="chain", in_axes=0 if in_axes is None else in_axes)

    def run(*args):
        mapped = range(len(args)) if in_axes is None else [i for i, ax in enumerate(in_axes) if ax == 0]
        def one(xs):
            full = list(args)
            for i, x in zip(mapped, xs):
                full[i] = x
            return f(*full)
        return jax.lax.map(one, tuple(args[i] for i in mapped))
    return jax.jit(run)


def aux_zeros(tm, chains):
    """A placeholder with the shape of the sampler's AUX input: a transport map's anchors in
    parts, one row per chain, a single zero otherwise."""
    return aux_in(tm, np.zeros((chains, tm.aux_size))) if tm is not None else jnp.zeros(1)


def aux_in(tm, aux):
    """Stored anchors [chains, A] as the compiled programs take them."""
    return jnp.asarray(aux) if tm is None else tm.parts(aux)


def _chunk_len(samples):
    return max(1, samples // 100)


def _setup_init(ready, res, zeros, off0):
    """(th, aux) -> (U, g) from the sampler's compiled setup program, waiting for its compile."""
    def init(th, aux):
        ready.wait()
        if "err" in res:
            raise res["err"]
        return res["setup"](th, zeros, off0, aux)[2:4]
    return init


def _sample_phase(vg_off, D, chains, samples, thinning, seed, ctrl,
                  imm, eps, state, parallel, progress_bar=False,
                  donor_loader=None, prebuilt=None, off0=None, offset_fn=None, tm=None):
    """DR-G-HMC sampling from a fixed (imm, eps). `state` holds the chains'
    positions `th` [chains, D] in sampler coordinates, optional momenta `rho`, the
    global step offset `t0` (0 for a fresh run) and, with a transport map, its anchors
    `aux`. `off0` is the runtime bundle (vg.OFF0); `offset_fn(state)` replaces it with one
    carrying a log-likelihood offset. Returns (thinned draws [chains, kept, D] in the model's
    coordinates, stages, energy, end state)."""
    step = _chunk_len(samples)
    if prebuilt is not None:
        setup_p, chunk_c, chunk_raw = prebuilt
    else:
        setup_p, chunk_raw = _sampler_fns(vg_off, D, ctrl, parallel, tm)
        chunk_c = chunk_raw
    chain_keys = jax.random.split(_tagged(seed, 1), chains)
    zero_off = off0

    if donor_loader is not None:
        dummy = jnp.zeros((chains, D))
        aux0 = aux_zeros(tm, chains)
        c0 = jax.tree.map(np.asarray, setup_p(dummy, dummy, zero_off, aux0))
        cw = chunk_c(c0, jnp.arange(step), jnp.ones(step, bool), chain_keys, jnp.ones(D), jnp.ones(D), jnp.asarray(1e-3),
                     zero_off, aux0)
        jax.block_until_ready(cw)
        del c0, cw, dummy
        imm, eps, state = donor_loader()
        if tm is not None and tuple(tm.flags(state["aux"])) != tm.static:
            tm = tm.cheap_of(state["aux"])
            setup_p, chunk_raw = _sampler_fns(tm.wrap(tm.pot_off), D, ctrl, parallel, tm)
            chunk_c = chunk_raw
    OFF = zero_off if offset_fn is None else offset_fn(state)

    IMM = jnp.asarray(imm)
    MSQ = jnp.sqrt(1.0 / IMM)
    E0 = jnp.asarray(ctrl["eps_c"] * float(eps))
    th0 = jnp.asarray(state["th"])
    rho0 = state.get("rho")
    if rho0 is None:
        rho0 = MSQ * jax.random.normal(_tagged(seed, 2), (chains, D))
    t0 = int(state.get("t0", 0))
    if state.get("aux") is None:
        state["aux"] = np.zeros(1)
    AUX = aux_in(tm, state["aux"])
    carry = jax.tree.map(np.asarray, setup_p(th0, jnp.asarray(rho0), OFF, AUX))

    bar = _pbar(samples, "DR-G-HMC sampling") if progress_bar else None
    tp, sp, ep = [], [], []
    for i in range(t0, t0 + samples, step):
        ts = jnp.arange(i, min(i + step, t0 + samples))
        fn = chunk_c if ts.shape[0] == step else chunk_raw
        keep = (np.asarray(ts) - t0) % thinning == 0
        carry, (t, s, e) = fn(carry, ts, jnp.asarray(keep), chain_keys, IMM, MSQ, E0, OFF, AUX)
        carry = jax.tree.map(np.asarray, carry)
        tp.append(np.asarray(t)[:, keep]); sp.append(np.asarray(s)[:, keep]); ep.append(np.asarray(e)[:, keep])
        if bar is not None:
            bar.update(int(ts.shape[0]))
    if bar is not None:
        bar.close()
    sl = slice(None)
    end = dict(th=np.asarray(carry[0]), rho=np.asarray(carry[1]), t0=t0 + samples,
               offset=None if offset_fn is None else np.asarray(OFF["off"]), aux=np.asarray(state["aux"]))
    return (np.concatenate(tp, 1)[:, sl], np.concatenate(sp, 1)[:, sl],
            np.concatenate(ep, 1)[:, sl], end)


_CONSTRAIN_BYTES = 256 * 2 ** 20


def _postprocess_chunked(postprocess_fn, unflatten, ths, data, save_loglik=True,
                         keep_re=False, progress_bar=False, chunk=500, store=None, tag="full"):
    """Constrain flat draws -> named sites (+ deterministics). Chains constrain
    in parallel across devices (pmap) when the device count allows; draw-chunks
    stream into preallocated outputs to bound peak memory. Drops raw
    z_* unless keep_re, to match NUTS get_samples; drops log_lik (the largest array by
    far) when it is not needed."""
    chains, kept, _ = ths.shape

    def _keep(k):
        if k.startswith("z_") and not keep_re:
            return False
        return not (k == "log_lik" and not save_loglik)

    par = chains > 1 and jax.local_device_count() >= chains
    key = (tag, bool(save_loglik), bool(keep_re))
    cached = None if store is None else store["constrain"].get(key)
    if cached is None:
        one = lambda z, d: {k: v for k, v in postprocess_fn(unflatten(z), d).items() if _keep(k)}
        named1 = jax.vmap(one, (0, None))
        namedC = jax.pmap(named1, axis_name="chain", in_axes=(0, None)) if par else None
        if store is not None:
            store["constrain"][key] = (one, named1, namedC)
    else:
        one, named1, namedC = cached
    shapes = jax.eval_shape(one, jax.ShapeDtypeStruct(ths.shape[2:], ths.dtype), data)
    per_draw = sum(int(np.prod(v.shape)) * np.dtype(v.dtype).itemsize for v in shapes.values())
    chunk = int(max(1, min(chunk, _CONSTRAIN_BYTES // max(1, per_draw * chains))))
    n_chunks = (kept + chunk - 1) // chunk
    bar = _pbar(n_chunks if par else chains * n_chunks,
                "Constraining draws", unit="chunks") if progress_bar else None

    out = {}
    if par:
        for i in range(0, kept, chunk):
            block = jax.block_until_ready(namedC(jnp.asarray(ths[:, i:i + chunk]), data))
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
                block = jax.block_until_ready(named1(jnp.asarray(ths[c, i:i + chunk]), data))
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
    import numpyro_backend as backend
    return backend._restore_centred(out)


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
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
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
        end = dict(th=d["end_th"], rho=d["end_rho"], t0=int(d["end_t0"]), offset=d["offset"],
                   aux=state.get("aux"))
        return d["ths"], d["stages"], d["energy"], end
    finally:
        if proc.poll() is None:
            proc.kill()
        shutil.rmtree(tmp, ignore_errors=True)


def _initial_positions(model, config, keys, off, D, chains, seed, init_strategy,
                       init_radius, init_z=None, to_sampled=None):
    """Per-chain starting points on the unconstrained scale, matching the NUTS
    init strategies: user values (`init_z`), 'prior' draws from the prior,
    otherwise uniform(-r, r)."""
    base = _tagged(seed, 5)
    if init_z is not None:
        th = np.zeros((chains, D))
        for k in keys:
            th[:, off[k][0]] = np.asarray(init_z[k]).reshape(chains, -1)
        return th if to_sampled is None else to_sampled(th)
    if init_strategy == "prior":
        from numpyro.infer import init_to_sample
        th = np.zeros((chains, D))
        for c in range(chains):
            st = initialize_model(jax.random.fold_in(base, c), model,
                                  init_strategy=init_to_sample, model_args=(config,))
            z = st.param_info.z
            for k in keys:
                th[c, off[k][0]] = np.asarray(z[k]).reshape(-1)
        return th if to_sampled is None else to_sampled(th)
    r = float(init_radius)
    if r == 0.0:
        return np.zeros((chains, D))
    return np.asarray(jax.random.uniform(base, (chains, D), minval=-r, maxval=r))


def run_drghmc(model, config, chains=4, warmup=1000, samples=10000, seed=0,
               thinning=1, chain_method="parallel", progress_bar=True,
               init_strategy="numeric", init_radius=0.0, control=None,
               save_loglik=True, keep_re=False, save_state=False, init_z=None):
    """With `save_state` the return carries the end-of-run `state` (positions, momenta,
    step offset, adaptation), from which autorun continues the same chains exactly."""
    ctrl = _merge_control(control)
    parallel = chain_method == "parallel"
    _convert_matrices(config)
    t0 = time.time()
    sig = _signature(config, chains, parallel, ctrl, _chunk_len(samples))
    store = _programs(sig)

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
        phases.mark("model built")
        tm = _transport(model, config, ctrl, off, unflatten, progress_bar)
        phases.mark("transport map built")
        vg_s = _sampler_vg(vg, tm)
        th_init = _initial_positions(model, config, keys, off, D, chains, seed,
                                     init_strategy, init_radius, init_z, unflatten.to_sampled)
        if store["plain"] is not None and not _verify_store(vg, store, th_init, chains, D):
            store = _programs(sig, fresh=True)

        smp_thread, init, on_decide, final = None, None, None, {}
        if ctrl["precision"] != "f32":
            import threading
            step = _chunk_len(samples)
            _ck = jax.random.split(_tagged(seed, 1), chains)
            _th0 = jnp.zeros((chains, D))
            _c0 = (_th0, _th0, jnp.zeros(chains), _th0)
            _off0 = vg.OFF0

            def lower_sampler(tm_v, vg_v):
                setup_p, chunk_raw = _sampler_fns(vg_v, D, ctrl, parallel, tm_v)
                _aux0 = aux_zeros(tm_v, chains)
                return (setup_p.lower(_th0, _th0, _off0, _aux0),
                        chunk_raw.lower(_c0, jnp.arange(step), jnp.ones(step, bool), _ck, jnp.ones(D), jnp.ones(D),
                                        jnp.asarray(1e-3), _off0, _aux0), chunk_raw)

            def compile_sampler(flags, low, res, ready):
                def work():
                    try:
                        res["setup"] = low[0].compile()
                        ready.set()
                        res["chunk"] = low[1].compile()
                        res["chunk_raw"] = low[2]
                        phases.mark("sampler compiled")
                        store["sampler"][flags] = dict(setup=res["setup"], chunk=res["chunk"],
                                                       chunk_raw=res["chunk_raw"])
                        msg = memory_check(res["chunk"], chains, parallel)
                        if msg is not None:
                            print(f"[DR-G-HMC] warning: {msg}", flush=True)
                    except BaseException as e:
                        res["err"] = e
                        ready.set()
                t = threading.Thread(target=work, daemon=True)
                t.start()
                return t

            def start_sampler(flags, tm_v, vg_v, res, ready):
                hit = store["sampler"].get(flags)
                if hit is not None:
                    res.update(hit)
                    ready.set()
                    phases.mark("sampler reused")
                    return None
                return compile_sampler(flags, lower_sampler(tm_v, vg_v), res, ready)

            _res = {}
            setup_ready = threading.Event()
            if progress_bar:
                print("[DR-G-HMC] compiling sampler (overlapped with "
                      "adaptation)...", flush=True)
            smp_thread = start_sampler(_sampler_key(tm), tm, vg_s, _res, setup_ready)
            phases.mark("sampler lowered")
            init = _setup_init(setup_ready, _res, _th0, _off0)
            if tm is not None:
                def on_decide(flags):
                    if any(flags):
                        tm_v = tm.view(flags)
                        final["tm"] = tm_v
                        final["thread"] = start_sampler(tuple(int(f) for f in flags), tm_v, tm_v.wrap(tm.pot_off),
                                                        final, threading.Event())

        imm, eps, th_adapted, aux_adapted, wall_warm = _adapt(
            model, config, keys, off, vg, vg_s, tm, D, chains, warmup, seed, ctrl, parallel,
            progress_bar, th_init, user_init=init_z is not None, init=init, on_decide=on_decide, store=store)
        state = dict(th=th_adapted, rho=None, t0=0, aux=aux_adapted)
        phases.mark("adaptation done")
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
        if final:
            if final["thread"] is not None:
                final["thread"].join()
            if "err" in final:
                raise final["err"]
            _res, tm = final, final["tm"]

        ths, stages, energy, end = _sample_phase(
            None, D, chains, samples, thinning, seed, ctrl, imm, eps, state,
            parallel, progress_bar,
            prebuilt=(_res["setup"], _res["chunk"], _res["chunk_raw"]), off0=vg.OFF0, tm=tm)

    phases.mark("sampling done")
    pp_fn, pp_data = postprocess_fn, vg.data
    if not save_loglik:
        fast = _fast_constrain_fn(model, config, postprocess_fn, unflatten, ths, vg.data, vg.trim, store)
        if fast is not None:
            pp_fn, pp_data = fast
        phases.mark("fast constrain check done")
    if progress_bar:
        note = (" (log_lik retained; large)" if save_loglik
                else " (log_lik skipped)" if pp_fn is not postprocess_fn
                else " (log_lik skipped; full recompute)")
        print("[DR-G-HMC] constraining draws" + note + "...", flush=True)
    samples_np = _postprocess_chunked(pp_fn, unflatten, ths, pp_data,
                                      save_loglik=save_loglik,
                                      keep_re=keep_re,
                                      progress_bar=progress_bar, store=store,
                                      tag="full" if pp_fn is postprocess_fn else "trim")
    phases.mark("constrain done")
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
                         t0=int(end["t0"]), seed=int(seed), offset=end.get("offset"),
                         aux=end.get("aux"), transport=tm is not None)
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
    tm = _transport(backend.sdt_model, config, p["ctrl"], off, unflatten)
    vg_off = _sampler_vg(vg, tm)

    def first_position(st):
        th = np.asarray(st["th"])[:1]
        return (th if tm is None else tm.forward_chains(th, np.asarray(st["aux"])[:1], vg.trim))[0]

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
        donor_loader=load_donor, off0=vg.OFF0, tm=tm,
        offset_fn=lambda st: {**vg.OFF0, "off": (jnp.asarray(st["offset"]) if st.get("offset") is not None else
                                                 ll_offset(postprocess_fn, unflatten, config, first_position(st),
                                                           vg.data))})
    np.savez(p["out"], ths=ths, stages=stages, energy=energy,
             end_th=end["th"], end_rho=end["rho"], end_t0=end["t0"], offset=end["offset"])


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "_sample_f32":
        import jax as _j
        _j.config.update("jax_enable_x64", False)
        _main_sample_f32(sys.argv[2])
