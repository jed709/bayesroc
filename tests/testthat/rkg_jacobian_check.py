"""Independent check that the CDP R/K/G slack reparameterisation is exact (test-rkg-jacobian.R).

With the hyperparameters fixed, the model maps its sampled effect coordinates x (every z site plus
know_crit_free / know_crit_shift) to theta = (every block's effects u, the know-criterion
coefficients). Exactness on the feasible set requires
    log N(z; 0, I) + rkg_level_jac + know_crit_shift_jac
        = sum_blocks log N(u_b; 0, Sigma_b) + log |det d theta / d x| + const,
the constant collecting terms that do not depend on x. The Jacobian is taken by forward-mode
autodiff of the model itself; the difference D is evaluated at several points of x and must not
vary. Prints one line per config: "<name> ok <spread>", "<name> skip <reason>" or "<name> FAIL ...".
Usage: python rkg_jacobian_check.py <package python dir> <config dir>"""
import os, sys, json, glob
sys.path.insert(0, sys.argv[1])
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from numpyro import handlers
from numpyro.infer.util import initialize_model, constrain_fn
from numpyro.infer.initialization import init_to_uniform
import numpyro_backend as nb, drghmc as dr

TOL = 1e-7


def blocks(cfg, sites):
    res = cfg.get("random_effects") or []
    batched, seen = {}, {}
    for i, rc in enumerate(res):
        if rc.get("cor_id") is not None or rc.get("param") in ("criterion", "criterion2", "bias"):
            continue
        if int(rc["n_terms"]) == 1 and rc.get("type", "index") == "index" and rc.get("term_idx") is None:
            batched.setdefault((rc["group"], int(rc["n_groups"]), rc.get("prior_sd", "normal(0, 0.5)")), []).append(i)
    out = {}
    for k in sites:
        if k.startswith("z_cross_"):
            cid, g = next((rc["cor_id"], rc["group"]) for rc in res
                          if rc.get("cor_id") is not None and k == f"z_cross_{rc['cor_id']}_{rc['group']}")
            mem = [rc for rc in res if rc.get("cor_id") == cid and rc["group"] == g]
            out[k] = ("cross", [f"u_{rc['param']}_{g}_from_{cid}" for rc in mem], f"sigma_cross_{cid}_{g}", f"corr_cross_{cid}_{g}")
            continue
        if k.startswith("z_batch_"):
            grp = next(b for b in batched.values() if len(b) > 1 and k == "z_batch_{}_{}".format(
                "_".join(sorted(res[i]["param"] for i in b)), res[b[0]]["group"]))
            out[k] = ("batch", [f"u_{res[i]['param']}_{res[i]['group']}" for i in grp],
                      [f"sigma_{res[i]['param']}_{res[i]['group']}[1]" for i in grp], None)
            continue
        base, term = (k.split("[")[0], int(k.split("[")[1].rstrip("]")) - 1) if k.endswith("]") else (k, None)
        rc = next((rc for rc in res if rc.get("cor_id") is None and base == f"z_{rc['param']}_{rc['group']}"), None)
        if rc is None:
            raise KeyError(f"unrecognised effect site {k}")
        u = f"u_{rc['param']}_{rc['group']}"
        nt = int(rc["n_terms"])
        if term is not None:
            out[k] = ("term", u, f"sigma_{rc['param']}_{rc['group']}[{term + 1}]", (term, nt))
        else:
            out[k] = ("corr", u, (f"sigma_{rc['param']}_{rc['group']}", nt), f"corr_{rc['param']}_{rc['group']}")
    return out


def mvn_logpdf(U, sd, C):
    Lc = jnp.linalg.cholesky(C)
    W = jax.scipy.linalg.solve_triangular(Lc, (U / sd[None, :]).T, lower=True)
    n, d = U.shape
    return (-0.5 * jnp.sum(W ** 2) - n * jnp.sum(jnp.log(jnp.diagonal(Lc))) - n * jnp.sum(jnp.log(sd))
            - 0.5 * n * d * jnp.log(2.0 * jnp.pi))


def check(path):
    cfg = json.load(open(path)); cfg = cfg.get("config", cfg); dr._convert_matrices(cfg)
    if nb._rkg_plan(cfg) is None:
        return "skip no slack plan"
    model = nb.sdt_model
    init = initialize_model(jax.random.PRNGKey(11), model, model_args=(cfg,), init_strategy=init_to_uniform(radius=1.0))
    z0 = init.param_info.z
    vals = constrain_fn(model, (cfg,), {}, z0)
    X = sorted(k for k in z0 if k.startswith("z_") or k in ("know_crit_free", "know_crit_shift"))
    shp = {k: np.shape(z0[k]) for k in X}
    sz = {k: int(np.prod(shp[k])) for k in X}
    bl = blocks(cfg, [k for k in X if k.startswith("z_")])

    def trace(x):
        d, o = dict(vals), 0
        for k in X:
            d[k] = jnp.reshape(x[o:o + sz[k]], shp[k])
            o += sz[k]
        return handlers.trace(handlers.substitute(model, data=d)).get_trace(cfg)

    def det(tr, nm):
        return tr[nm]["value"]

    def sd_of(tr, spec):
        nm, nt = spec
        return det(tr, nm) if nm in tr else jnp.stack([det(tr, f"{nm}[{t + 1}]") for t in range(nt)])

    def parts(tr):
        th, lp = [], 0.0
        for k, (kind, u, s, c) in bl.items():
            if kind == "cross":
                U = jnp.concatenate([jnp.reshape(det(tr, nm), (det(tr, nm).shape[0], -1)) for nm in u], 1)
                sd = det(tr, s)
                C = det(tr, c)
            elif kind == "batch":
                U = None
                for nm, snm in zip(u, s):
                    th.append(det(tr, nm).reshape(-1))
                    lp = lp + jnp.sum(jax.scipy.stats.norm.logpdf(det(tr, nm), 0.0, det(tr, snm)))
                continue
            elif kind == "term":
                t, nt = c
                Ua = det(tr, u)
                col = Ua if Ua.ndim == 1 else Ua[:, t]
                th.append(col.reshape(-1))
                lp = lp + jnp.sum(jax.scipy.stats.norm.logpdf(col, 0.0, det(tr, s)))
                continue
            else:
                U = det(tr, u)
                U = U[:, None] if U.ndim == 1 else U
                sd = sd_of(tr, s)
                C = det(tr, c) if c in tr else jnp.eye(sd.shape[0])
            th.append(U.reshape(-1))
            lp = lp + mvn_logpdf(U, sd, C)
        if not any(k in X for k in ("know_crit_free", "know_crit_shift")):
            pass
        elif "beta_know_crit" in tr:
            th.append(jnp.reshape(det(tr, "beta_know_crit"), (-1,)))
        else:
            j = 1
            while f"beta_know_crit[{j}]" in tr:
                th.append(jnp.reshape(det(tr, f"beta_know_crit[{j}]"), (1,)))
                j += 1
        return jnp.concatenate(th), lp

    def theta(x):
        return parts(trace(x))[0]

    def lhs(x):
        tr = trace(x)
        v = 0.0
        for k in X:
            if k.startswith("z_"):
                v = v + jnp.sum(tr[k]["fn"].log_prob(tr[k]["value"]))
        for f in ("rkg_level_jac", "know_crit_shift_jac", "know_crit_slack_jac"):
            if f in tr:
                v = v + tr[f]["fn"].log_prob(tr[f]["value"]) if tr[f]["type"] == "sample" else v + tr[f]["value"]
        return v - parts(tr)[1]

    xdim = sum(sz.values())
    x0 = np.concatenate([np.asarray(z0[k], dtype=np.float64).reshape(-1) for k in X])
    rng = np.random.default_rng(5)
    xs = [x0] + [x0 + rng.normal(0.0, 1.0, xdim) for _ in range(3)]
    th0 = theta(jnp.asarray(x0))
    if th0.shape[0] != xdim:
        return f"FAIL dimension: {th0.shape[0]} effects+coefficients vs {xdim} sampled coordinates"
    jac = jax.jit(jax.jacfwd(theta))
    lh = jax.jit(lhs)
    Ds = []
    for x in xs:
        J = np.asarray(jac(jnp.asarray(x)))
        sgn, ld = np.linalg.slogdet(J)
        if sgn == 0 or not np.isfinite(ld):
            return "FAIL singular coordinate map"
        Ds.append(float(lh(jnp.asarray(x))) - ld)
    spread = max(Ds) - min(Ds)
    if not np.all(np.isfinite(Ds)) or spread > TOL * (1.0 + abs(Ds[0])):
        return f"FAIL D = {np.round(Ds, 8).tolist()}"
    return f"ok {spread:.2e}"


for path in sorted(glob.glob(os.path.join(sys.argv[2], "*.json"))):
    name = os.path.splitext(os.path.basename(path))[0]
    try:
        print(name, check(path), flush=True)
    except Exception as e:
        print(name, "FAIL", type(e).__name__, str(e)[:300], flush=True)
