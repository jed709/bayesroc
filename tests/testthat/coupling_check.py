"""Independent check of the transport map's coefficient coupling (test-transport-coupling.R).

For each config: at three random points, the per-observation log-likelihood Jacobian in the
effects u (u_g = A z_g) and in every population coefficient. Within a group, a coefficient whose
Jacobian column lies exactly in the span of the group's effect columns trades off with those
effects; the map's weights (one-to-one pairs, projected pairs) must reproduce it. Prints one line
per config: "<name> ok", or "<name> ISSUES" followed by the mismatches.
Usage: python coupling_check.py <package python dir> <config dir>"""
import os, sys, json, glob
sys.path.insert(0, sys.argv[1])
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpyro_backend as nb, drghmc as dr, tmap

EXACT = 1e-7
TOL = 1e-6


def audit(path):
    cfg = json.load(open(path)); cfg = cfg.get("config", cfg); dr._convert_matrices(cfg)
    vg, unflatten, keys, off, D, post = dr._build(nb.sdt_model, cfg)
    tm = tmap.build(nb.sdt_model, cfg, off, unflatten)
    sbs = tm.sbs if tm is not None else []
    confs = cfg.get("random_effects") or []
    mapped = {(b.prefix) for sb in sbs for b in sb.blocks}
    want = set()
    for c in confs:
        want.add(f"cross_{c['cor_id']}_{c['group']}" if c.get("cor_id") is not None else f"{c.get('param')}_{c['group']}")
    unmapped = sorted(want - mapped)
    rng = np.random.default_rng(3)
    ths = [rng.normal(0.0, 0.4, D) for _ in range(3)]
    ll = jax.jit(lambda v: post(unflatten(v))["log_lik"])
    Js = [np.array(jax.jacrev(ll)(jnp.asarray(th))) for th in ths]
    if "dprime_order_loggap" in off:
        f = off["dprime_order_loggap"][0].start
        for Jk, th in zip(Js, ths):
            Jk[:, f] /= np.exp(th[f])
    J = Js[0]
    bnames, bflats = [], []
    for k in keys:
        if k.startswith("beta_") or k.startswith("dprime_order"):
            sl, shape = off[k]
            for i in range(sl.stop - sl.start):
                bnames.append(k if sl.stop - sl.start == 1 else f"{k}[{np.unravel_index(i, shape) if shape else i}]")
                bflats.append(sl.start + i)
    bflats = np.array(bflats, int)
    col = {f: j for j, f in enumerate(bflats)}
    psts = [tm.post(jnp.asarray(th)) for th in ths] if tm is not None else None
    issues = []
    for sb in sbs:
        Ainvs = [np.linalg.inv(np.asarray(sb.A(p))) for p in psts]
        dims = [f"{b.prefix}:{d}" for b in sb.blocks for d in range(b.dim)]
        for g in range(sb.n):
            rows = np.nonzero(sb.index == g)[0]
            if not len(rows):
                continue
            Ju = np.concatenate([Jk[np.ix_(rows, sb.pos_np[:, g])] @ Ai for Jk, Ai in zip(Js, Ainvs)], 0)
            Jb = np.concatenate([Jk[np.ix_(rows, bflats)] for Jk in Js], 0)
            Wt = np.zeros((sb.D, len(bflats)))
            for i, d in enumerate(sb.bdims):
                Wt[d, col[int(sb.bflat[i])]] += 1.0
            for i, d in enumerate(sb.gdims):
                for c, f in enumerate(sb.bflat):
                    Wt[d, col[int(f)]] += sb.gW[i, g, c]
            act = np.linalg.norm(Ju, axis=0) > 1e-10 * max(1.0, np.abs(J).max())
            if not act.any():
                continue
            Ja = Ju[:, act]
            W, *_ = np.linalg.lstsq(Ja, Jb, rcond=None)
            res = np.linalg.norm(Jb - Ja @ W, axis=0)
            nb_ = np.linalg.norm(Jb, axis=0)
            for j in range(len(bflats)):
                if nb_[j] < 1e-12:
                    continue
                n_rows = len(rows)
                informative = np.sum((np.abs(Jb[:, j]) > 1e-12 * nb_[j]).reshape(len(Js), n_rows).any(0))
                exact = res[j] <= EXACT * nb_[j] and informative >= np.linalg.matrix_rank(Ja) + 3
                wt = Wt[act, j]
                if exact:
                    dif = np.abs(W[:, j] - wt)
                    res_t = np.linalg.norm(Jb[:, j] - Ja @ wt)
                    if res_t > 1e-6 * nb_[j]:
                        k = int(np.argmax(dif))
                        ad = np.nonzero(act)[0][k]
                        kind = "MISSING" if np.all(np.abs(wt) < 1e-12) else "WRONG"
                        issues.append((kind, sb.name, g, dims[ad], bnames[j], float(W[k, j]), float(wt[k])))
                else:
                    if np.abs(wt).max() > 1e-12 and np.abs(W[:, j]).max() < 1e-9:
                        k = int(np.argmax(np.abs(wt)))
                        issues.append(("SPURIOUS", sb.name, g, dims[np.nonzero(act)[0][k]], bnames[j], 0.0, float(wt[k])))
    global LINKS
    LINKS = [(sb.name, sb.D, len(sb.links), int(len(sb.bdims)), int(len(sb.gdims))) for sb in sbs]
    return unmapped, issues


if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(sys.argv[2], "*.json")))
    for f in files:
        name = os.path.basename(f)[:-5]
        try:
            unmapped, issues = audit(f)
        except Exception as e:
            print(f"{name:34s} ERROR {type(e).__name__}: {str(e)[:160]}", flush=True)
            continue
        agg = {}
        for kind, sbn, g, dim, b, wi, wt in issues:
            key = (kind, dim.rsplit(":", 1)[0], b)
            agg.setdefault(key, [0, wi, wt, g])[0] += 1
        status = "ok" if not issues and not unmapped else "ISSUES"
        print(f"{name:34s} {status}" + (f" | unmapped: {unmapped}" if unmapped else "")
              + " | blocks " + ", ".join(f"{a}(D{b} links{c} pairs{d}+{e})" for a, b, c, d, e in LINKS), flush=True)
        for (kind, blk, b), (cnt, wi, wt, g) in sorted(agg.items())[:12]:
            print(f"    {kind:8s} {blk:28s} <- {b:34s} groups {cnt:3d} (e.g. g{g}: implied {wi:+.3f}, map {wt:+.3f})", flush=True)
        if len(agg) > 12:
            print(f"    ... {len(agg) - 12} more", flush=True)
