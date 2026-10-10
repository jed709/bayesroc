"""Per-group transport map for random-effect blocks.

With non-centred random effects u_g = A(x) z_g the scale parameters x and the standardised
effects z are tied together through the likelihood, more tightly the more groups there are, and
samplers crawl along that tie on large data. Here every group's effects are instead sampled as

    u_g = LP_g^-T (zeta_g + LP_g^-1 s_g),    LP_g LP_g' = Lam_g + Sigma(x)^-1,

with LP_g upper triangular, so that where a group's data say nothing (Lam_g = 0, s_g = 0) this is
a non-centred form of the effects. Dimensions whose effects are almost entirely prior (shrinkage
weight c / (1 + c) below _UNINFORMED, with c the mean likelihood curvature times the prior
variance) are placed last in the triangle, so a correlation with such an effect moves that
effect and leaves the informed ones where the data put them,

where (Lam_g, s_g) is a fixed quadratic approximation of the group's log-likelihood in u_g. If the
approximation were exact, zeta would be standard normal and independent of x. Blocks that share a
grouping factor are mapped jointly. s_g is shifted by -Lam_g Delta_g when the fixed effects or
earlier blocks that enter the same linear-predictor terms move by Delta_g, so population means,
parent levels and crossed factors stay untied as well. For fixed (Lam, s) the map
(x, beta, zeta) -> (x, beta, z) is a triangular bijection; sampling zeta with its log-Jacobian
leaves the posterior of the model's own parameters unchanged."""
import os
import re
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular, block_diag


_RE_PARAMS = ("dprime", "sigma", "lambda", "dprime2", "sigma2", "criterion", "criterion2", "discrim", "rho")
_FD_STEP = 1e-3
_NEWTON = 3
_NEWTON_CAP = 0.5
_UNINFORMED = 0.1
_MIN_TRIALS_PER_DIM = 2.0
_THIN_MAX_DIM = 10
_MAX_LINK = 4_000_000


class _Block:
    """Coordinates and geometry of one random-effect block in the flat unconstrained vector."""

    def __init__(self, conf, param, off, cross_members=None, batch=None):
        g = conf["group"]
        self.n = int(conf["n_groups"])
        cid = conf.get("cor_id")
        if batch is not None:
            z_site, sd_site, row = batch
            self.dim, self.dim_offset = 1, 0
            self.prefix = f"{param}_{g}"
            self.z_sites = [(z_site, row)]
            self.sd_sites = [(sd_site, row)]
            self.sd_det = None
        elif cid is not None:
            members = cross_members[cid]
            self.dim = sum(int(m["n_terms"]) for m in members)
            self.dim_offset = 0
            for m in members:
                if m is conf:
                    break
                self.dim_offset += int(m["n_terms"])
            self.prefix = f"cross_{cid}_{g}"
            self.z_sites = [(f"z_cross_{cid}_{g}", None)]
            self.sd_det = f"sigma_cross_{cid}_{g}"
            sd_sites = []
            o = 0
            for m in members:
                nt = int(m["n_terms"])
                if nt == 1:
                    sd_sites.append((f"sigma_cross_{cid}_{g}[{o + 1}]", None))
                else:
                    sd_sites += [(f"sigma_cross_{cid}_{g}_block_{m['param']}", t) for t in range(nt)]
                o += nt
            self.sd_sites = sd_sites
        else:
            nt = int(conf["n_terms"])
            self.dim, self.dim_offset = nt, 0
            self.prefix = f"{param}_{g}"
            correlated = conf.get("correlated", True) and nt > 1
            vector = f"sigma_{self.prefix}" in off
            if correlated:
                self.z_sites = [(f"z_{self.prefix}", None)]
            elif nt == 1:
                self.z_sites = [(f"z_{self.prefix}[1]", None)]
            else:
                self.z_sites = [(f"z_{self.prefix}[{t + 1}]", None) for t in range(nt)]
            self.sd_sites = ([(f"sigma_{self.prefix}", t) for t in range(nt)] if vector
                             else [(f"sigma_{self.prefix}[{t + 1}]", None) for t in range(nt)])
            self.sd_det = None
        self.L_sites = [k for k in off if re.fullmatch(f"L_corr_{re.escape(self.prefix)}(_b\\d+)?", k)]
        for s, _ in self.z_sites + self.sd_sites:
            if s not in off:
                raise KeyError(s)
        zi = []
        for s, row in self.z_sites:
            sl = off[s][0]
            zi.append(np.arange(sl.start, sl.stop) if row is None
                      else sl.start + row * self.n + np.arange(self.n))
        self.z_idx = np.concatenate(zi).reshape(self.dim, self.n)
        self.sd_idx = np.array([off[s][0].start + (0 if t is None else t) for s, t in self.sd_sites])
        self.x_idx = np.concatenate([self.sd_idx] + [np.arange(off[k][0].start, off[k][0].stop) for k in self.L_sites])

    def A(self, post):
        if self.sd_det is not None:
            sd = post[self.sd_det]
        else:
            sd = jnp.stack([jnp.reshape(post[s], (-1,))[0 if t is None else t] for s, t in self.sd_sites])
        Lk = f"L_corr_{self.prefix}"
        L = post[Lk] if Lk in post else jnp.eye(self.dim, dtype=sd.dtype)
        return sd[:, None] * L

    def Ainv(self, post):
        """(sd * L)^-1 = L^-1 diag(1/sd); a diagonal when the block is uncorrelated."""
        if self.sd_det is not None:
            sd = post[self.sd_det]
        else:
            sd = jnp.stack([jnp.reshape(post[s], (-1,))[0 if t is None else t] for s, t in self.sd_sites])
        Lk = f"L_corr_{self.prefix}"
        if Lk in post:
            return solve_triangular(post[Lk], jnp.eye(self.dim, dtype=sd.dtype), lower=True) / sd[None, :]
        return jnp.diag(1.0 / sd)



def _gap_of_component(k, mid, n_thresh):
    n_upper = n_thresh - mid - 1
    if k > mid:
        return k - mid - 1
    return n_upper + (mid - 1 - k)


def _re_columns(conf, N):
    """The N-vectors multiplying each term of one random-effect conf (before threshold components)."""
    if conf.get("type", "index") == "matrix":
        Z = np.asarray(conf["Z"], dtype=float)
        return Z.reshape(N, -1)
    t = conf.get("term_idx")
    if t is not None:
        t = np.asarray(t, dtype=int)
        R = int(t.max())
        return np.stack([(t == r + 1).astype(float) for r in range(R)], 1)
    return np.ones((N, 1))


def _beta_site(off, prefix, idx):
    """Flat index of a fixed-effect coefficient given candidate site spellings."""
    for name, pos in idx:
        if name in off:
            sl, shape = off[name][0], off[name][1]
            if not shape:
                return sl.start if pos is None else None
            if pos is not None and len(shape) == len(pos):
                return sl.start + int(np.ravel_multi_index(pos, shape))
    return None


def _batched_betas(config, off):
    """Single-coefficient parameters that share a prior are sampled by the model as one site
    (beta_batch_<params> [members]): param -> flat index of its coefficient."""
    import numpyro_backend as nb
    knob = nb._rkg_knob(config)
    ordered = config.get("needs_ordered_dprime", False)
    groups = {}
    for name, pc in (config.get("params") or {}).items():
        if (name in ("criterion", "criterion2") or (knob is not None and name == "know_crit")
                or (ordered and name in ("dprime", "dprime2")) or pc.get("n_coef", 0) != 1):
            continue
        cols = pc.get("col_names", ["V1"])
        cols = [cols] if isinstance(cols, str) else cols
        prior = (pc.get("priors") or {}).get(cols[0] if len(cols) else "V1", "normal(0, 2.5)")
        groups.setdefault(prior, []).append(name)
    out = {}
    for members in groups.values():
        site = "beta_batch_" + "_".join(sorted(members))
        if len(members) > 1 and site in off and tuple(off[site][1]) == (len(members),):
            for row, name in enumerate(members):
                out[name] = off[site][0].start + row
    return out


def _varying_sets(config, conf):
    """(sets, components per set) of a criterion2 random effect under varying source criteria, or
    None when its dimensions are not one per set (and threshold component)."""
    n_sets = int(config.get("n_vbin") or config["K"])
    nth = int(config["K2"]) - 1
    return (n_sets, nth) if int(conf["n_terms"]) % (n_sets * nth) == 0 else None


def _fixed_map_varying(config, conf, off, Zc, gap_prefix):
    """Varying source criteria: an intercept random effect's dimension for (set, threshold
    component) enters exactly the rows the set's own coefficient enters."""
    lay = _varying_sets(config, conf)
    if lay is None or Zc.shape[1] != 1 or not np.all(Zc[:, 0] == 1.0):
        return []
    n_sets, n_comp = lay
    nth = int(config["K2"]) - 1
    mid = (nth + 1) // 2 - 1
    out = []
    for b in range(n_sets):
        for c in range(n_comp):
            if n_comp == 1 or c == mid:
                flat = _beta_site(off, None, [("beta_thresh_mid_2_varying", (b,))])
            else:
                flat = _beta_site(off, None, [(gap_prefix + "_2_varying", (b, _gap_of_component(c, mid, nth)))])
            if flat is not None:
                out.append((b * n_comp + c, flat))
    return out


def _fixed_map(config, conf, param, off, n_thresh, mid, gap_prefix, batched=None):
    """[(dim_within_conf, flat beta index)] for the conf's terms that equal a fixed-effect column."""
    N = int(config["N"])
    Zc = _re_columns(conf, N)
    R = Zc.shape[1]
    out = []
    if param == "criterion2" and config.get("varying_source_criteria"):
        return _fixed_map_varying(config, conf, off, Zc, gap_prefix)
    if param in ("criterion", "criterion2"):
        mid_name = "beta_thresh_mid"
        if param == "criterion2":
            pc = (config.get("params") or {}).get("criterion2")
            X = np.asarray(pc["X"], dtype=float).reshape(N, -1) if pc and pc.get("X") is not None else np.ones((N, 1))
            n_thresh = int(config["K2"]) - 1
            mid = (n_thresh + 1) // 2 - 1
            mid_name, gap_prefix = "beta_thresh_mid_2", gap_prefix + "_2"
        else:
            X = np.asarray(config["X_crit"], dtype=float).reshape(N, -1)
        P = X.shape[1]
        scalar = bool(conf.get("location_only"))
        n_comp = 1 if scalar else n_thresh
        if int(conf["n_terms"]) != n_comp * R:
            return []
        n_gaps = n_thresh - 1
        for r in range(R):
            js = [j for j in range(P) if np.array_equal(Zc[:, r], X[:, j])]
            if not js:
                continue
            j = js[0]
            for k in range(n_comp):
                if scalar or k == mid:
                    flat = _beta_site(off, None, [(f"{mid_name}[{j + 1}]", None)])
                else:
                    g = _gap_of_component(k, mid, n_thresh)
                    flat = _beta_site(off, None, [(f"{gap_prefix}[{g + 1},{j + 1}]", None),
                                                  (f"{gap_prefix}[{g + 1}]", None) if P == 1 else ("", None),
                                                  (gap_prefix, (g,)) if P == 1 else ("", None)])
                if flat is not None:
                    out.append((k * R + r, flat))
        return out
    pc = (config.get("params") or {}).get(param)
    if not pc or int(pc.get("n_coef", 0)) < 1:
        return []
    X = np.asarray(pc["X"], dtype=float).reshape(N, -1)
    if int(conf["n_terms"]) != R:
        return []
    for r in range(R):
        js = [j for j in range(X.shape[1]) if np.array_equal(Zc[:, r], X[:, j])]
        if not js:
            continue
        j = js[0]
        flat = _beta_site(off, None, [(f"beta_{param}[{j + 1}]", None), (f"beta_{param}", (j,))])
        if flat is None and X.shape[1] == 1 and batched:
            flat = batched.get(param)
        if flat is not None:
            out.append((r, flat))
    return out


def _group_fixed_map(config, conf, param, off, n_thresh, mid, gap_prefix, batched, index, n, taken):
    """Terms that equal no population column as a whole but whose rows, within every group,
    share one row of the population design (a subject's effect of a within-subject factor when
    the population formula crosses it with a between-subject factor):
    [(dim_within_conf, flat beta indices [P'], weights [n, P'])], the weights being each group's
    design row. Such an effect and that combination of coefficients enter the same rows."""
    N = int(config["N"])
    Zc = _re_columns(conf, N)
    R, nt = Zc.shape[1], int(conf["n_terms"])
    if (param == "criterion2" and config.get("varying_source_criteria")) or not np.all((Zc == 0) | (Zc == 1)):
        return []
    comps = []
    if param in ("criterion", "criterion2"):
        mid_name = "beta_thresh_mid"
        if param == "criterion2":
            pc = (config.get("params") or {}).get("criterion2")
            X = np.asarray(pc["X"], dtype=float).reshape(N, -1) if pc and pc.get("X") is not None else np.ones((N, 1))
            n_thresh = int(config["K2"]) - 1
            mid = (n_thresh + 1) // 2 - 1
            mid_name, gap_prefix = "beta_thresh_mid_2", gap_prefix + "_2"
        else:
            X = np.asarray(config["X_crit"], dtype=float).reshape(N, -1)
        P = X.shape[1]
        scalar = bool(conf.get("location_only"))
        n_comp = 1 if scalar else n_thresh
        if nt != n_comp * R:
            return []
        for k in range(n_comp):
            flats = []
            for j in range(P):
                if scalar or k == mid:
                    flats.append(_beta_site(off, None, [(f"{mid_name}[{j + 1}]", None)]))
                else:
                    g = _gap_of_component(k, mid, n_thresh)
                    flats.append(_beta_site(off, None, [(f"{gap_prefix}[{g + 1},{j + 1}]", None),
                                                        (f"{gap_prefix}[{g + 1}]", None) if P == 1 else ("", None),
                                                        (gap_prefix, (g,)) if P == 1 else ("", None)]))
            comps.append((k * R, flats))
    else:
        pc = (config.get("params") or {}).get(param)
        if not pc or int(pc.get("n_coef", 0)) < 1 or nt != R:
            return []
        X = np.asarray(pc["X"], dtype=float).reshape(N, -1)
        P = X.shape[1]
        flats = [_beta_site(off, None, [(f"beta_{param}[{j + 1}]", None), (f"beta_{param}", (j,))]) for j in range(P)]
        if P == 1 and flats[0] is None and batched:
            flats = [batched.get(param)]
        comps.append((0, flats))
    out = []
    for r in range(R):
        on = Zc[:, r] == 1
        if not on.any():
            continue
        g, Xr = index[on], X[on]
        mean = np.zeros((n, P))
        np.add.at(mean, g, Xr)
        mean /= np.maximum(np.bincount(g, minlength=n), 1)[:, None]
        if np.max(np.abs(Xr - mean[g])) > 1e-12:
            continue
        used = np.where(np.any(mean != 0.0, 0))[0]
        for d0, flats in comps:
            if d0 + r in taken or not len(used) or any(flats[j] is None for j in used):
                continue
            out.append((d0 + r, [int(flats[j]) for j in used], mean[:, used]))
    return out


def _coef_design(config, conf, param, off, n_thresh, mid, gap_prefix, batched):
    """The population design a random-effect conf's terms act on, in the sampler's (centred)
    coordinates, and the flat coefficient indices of each threshold component:
    (Xc [N, P], [(dim offset of the component, flats [P] or None)]) or None."""
    N = int(config["N"])
    R, nt = _re_columns(conf, N).shape[1], int(conf["n_terms"])
    comps = []
    if param in ("criterion", "criterion2"):
        mid_name = "beta_thresh_mid"
        if param == "criterion2":
            pc = (config.get("params") or {}).get("criterion2")
            X = np.asarray(pc["X"], dtype=float).reshape(N, -1) if pc and pc.get("X") is not None else np.ones((N, 1))
            centre = pc.get("centre") if pc else None
            n_thresh = int(config["K2"]) - 1
            mid = (n_thresh + 1) // 2 - 1
            mid_name, gap_prefix = "beta_thresh_mid_2", gap_prefix + "_2"
        else:
            X = np.asarray(config["X_crit"], dtype=float).reshape(N, -1)
            centre = config.get("crit_centre")
        P = X.shape[1]
        scalar = bool(conf.get("location_only"))
        n_comp = 1 if scalar else n_thresh
        if nt != n_comp * R:
            return None
        for k in range(n_comp):
            flats = []
            for j in range(P):
                if scalar or k == mid:
                    flats.append(_beta_site(off, None, [(f"{mid_name}[{j + 1}]", None)]))
                else:
                    g = _gap_of_component(k, mid, n_thresh)
                    flats.append(_beta_site(off, None, [(f"{gap_prefix}[{g + 1},{j + 1}]", None),
                                                        (f"{gap_prefix}[{g + 1}]", None) if P == 1 else ("", None),
                                                        (gap_prefix, (g,)) if P == 1 else ("", None)]))
            comps.append((k * R, flats))
    else:
        pc = (config.get("params") or {}).get(param)
        cum_loc = config.get("family") in ("cumulative", "bivariate_cumulative") and param in ("dprime", "discrim")
        if nt != R or (not cum_loc and (not pc or int(pc.get("n_coef", 0)) < 1)):
            return None
        if pc and int(pc.get("n_coef", 0)) >= 1:
            X = np.asarray(pc["X"], dtype=float).reshape(N, -1)
            centre = pc.get("centre")
            P = X.shape[1]
            flats = [_beta_site(off, None, [(f"beta_{param}[{j + 1}]", None), (f"beta_{param}", (j,))]) for j in range(P)]
            if P == 1 and flats[0] is None and batched:
                flats = [batched.get(param)]
        else:
            X, centre, flats = np.zeros((N, 0)), None, []       # mu with no coefficients of its own
        comps.append((0, flats))
    if centre:
        cols = centre["col"] if isinstance(centre["col"], list) else [centre["col"]]
        m = np.asarray(centre["means"][0], dtype=float)
        X = X - np.outer(X[:, [int(j) for j in cols]].sum(1), m)
    # cumulative families: only cutpoint - mu is identified, so a shift of the cutpoints'
    # midpoint is a shift of mu with the opposite sign; the mu effects answer to it too
    fam = config.get("family")
    if fam in ("cumulative", "bivariate_cumulative") and param in ("dprime", "discrim"):
        if param == "dprime":
            Xm = np.asarray(config["X_crit"], dtype=float).reshape(N, -1)
            cm, mid_name = config.get("crit_centre"), "beta_thresh_mid"
        else:
            pc2 = (config.get("params") or {}).get("criterion2")
            Xm = np.asarray(pc2["X"], dtype=float).reshape(N, -1) if pc2 and pc2.get("X") is not None else np.ones((N, 1))
            cm, mid_name = (pc2.get("centre") if pc2 else None), "beta_thresh_mid_2"
        if cm:
            cols = cm["col"] if isinstance(cm["col"], list) else [cm["col"]]
            mm = np.asarray(cm["means"][0], dtype=float)
            Xm = Xm - np.outer(Xm[:, [int(j) for j in cols]].sum(1), mm)
        mid_flats = [_beta_site(off, None, [(f"{mid_name}[{j + 1}]", None)]) for j in range(Xm.shape[1])]
        X = np.concatenate([X, -Xm], 1)
        comps = [(d0, flats + mid_flats) for d0, flats in comps]
    return X, comps


def _proj_map(config, conf, param, off, n_thresh, mid, gap_prefix, batched, index, n):
    """How a change of the population coefficients moves each group's effects: within every
    group, the least-squares projection of the (centred) population design onto the term's
    columns, W_g = (Z_g' Z_g)^-1 Z_g' X_g. A column that one term reproduces exactly gets weight
    one there and nothing elsewhere (the one-to-one pairs); a between-group column, an
    interaction with a within-group factor or a continuous covariate gets its share in every
    group it enters. Independent of how the factors are coded.
    Returns (one-to-one pairs [(dim, flat)], weighted pairs [(dim, flats, W [n, P'])])."""
    des = _coef_design(config, conf, param, off, n_thresh, mid, gap_prefix, batched)
    if des is None:
        return [], []
    X, comps = des
    N = int(config["N"])
    Zc = _re_columns(conf, N)
    R, P = Zc.shape[1], X.shape[1]
    W = np.zeros((n, R, P))
    act = np.zeros((n, R), bool)
    order = np.argsort(index, kind="stable")
    bounds = np.searchsorted(index[order], np.arange(n + 1))
    for g in range(n):
        rows = order[bounds[g]:bounds[g + 1]]
        if not len(rows):
            continue
        act[g] = np.any(Zc[rows] != 0.0, 0)
        W[g] = np.linalg.lstsq(Zc[rows], X[rows], rcond=None)[0]
    W[np.abs(W) < 1e-10] = 0.0
    bp, gp = [], []
    for d0, flats in comps:
        for r in range(R):
            Wr = W[:, r, :]
            # weight one wherever the term is active and nothing elsewhere: the column is the term
            one = [j for j in range(P) if flats[j] is not None and act[:, r].any()
                   and np.all(np.abs(Wr[act[:, r], j] - 1.0) < 1e-9) and np.all(Wr[~act[:, r], j] == 0.0)]
            rest = [j for j in range(P) if flats[j] is not None and j not in one and np.any(Wr[:, j] != 0.0)]
            for j in one:
                bp.append((d0 + r, int(flats[j])))
            if rest:
                gp.append((d0 + r, [int(flats[j]) for j in rest], Wr[:, rest].copy()))
    return bp, gp


class _Super:
    """The blocks of one grouping factor, mapped jointly."""

    def __init__(self, name, blocks):
        self.name = name
        self.blocks = blocks
        self.n = blocks[0].n
        self.D = sum(b.dim for b in blocks)
        self.starts = [int(r[0]) for b in blocks for r in b.z_idx]
        self.index = blocks[0].index
        self.zcol = []
        self.slots, bd, bf, gp, o = [], [], [], [], 0
        for b in blocks:
            self.slots += b.slots
            self.zcol += b.zcol
            bd += [o + d for d, _ in b.bpairs]
            bf += [f for _, f in b.bpairs]
            gp += [(o + d, flats, W) for d, flats, W in getattr(b, "gpairs", [])]
            o += b.dim
        # coefficients read through group-specific weights follow the one-to-one pairs in bflat
        pos = {}
        for _, flats, _ in gp:
            for f in flats:
                pos.setdefault(f, len(bf) + len(pos))
        self.bdims, self.bflat = np.array(bd, int), np.array(bf + list(pos), int)
        self.gdims = np.array([d for d, _, _ in gp], int)
        self.gW = np.zeros((len(gp), self.n, len(self.bflat)))
        for i, (_, flats, W) in enumerate(gp):
            for c, f in enumerate(flats):
                self.gW[i, :, pos[f]] += W[:, c]
        self.links = []
        # one gather / one scatter for the whole [D, n] block: D chained dynamic_update_slices
        # each copied the full parameter vector, forward and again in the VJP
        self.pos = jnp.asarray(np.add.outer(np.array(self.starts, int), np.arange(self.n)))

    def A(self, post):
        return block_diag(*[b.A(post) for b in self.blocks])

    def Ainv(self, post):
        return block_diag(*[b.Ainv(post) for b in self.blocks])

    def get(self, v):
        return v[self.pos]

    def put(self, v, z):
        return v.at[self.pos].set(z)


def _batched_sites(res, off):
    """Intercept-only uncorrelated effects that share a grouping factor and a prior are sampled
    by the model as one site pair (z_batch_<params>_<group> [members, groups] and
    sigma_batch_<params>_<group> [members]): id(conf) -> (z site, sd site, row) for those."""
    groups = {}
    for conf in res:
        if (conf.get("cor_id") is None and int(conf["n_terms"]) == 1 and conf.get("type", "index") == "index"
                and conf.get("term_idx") is None and f"z_{conf.get('param')}_{conf['group']}[1]" not in off):
            key = (conf["group"], int(conf["n_groups"]), conf.get("prior_sd", "normal(0, 0.5)"))
            groups.setdefault(key, []).append(conf)
    out = {}
    for (g, n, _), members in groups.items():
        tag = "_".join(sorted(m["param"] for m in members))
        z_site, sd_site = f"z_batch_{tag}_{g}", f"sigma_batch_{tag}_{g}"
        if z_site in off and sd_site in off and tuple(off[z_site][1]) == (len(members), n):
            for row, m in enumerate(members):
                out[id(m)] = (z_site, sd_site, row)
    return out


def _supers(config, off, split=False, thin_rule=True):
    """Super-blocks and the blocks left unmapped: wider than _THIN_MAX_DIM with fewer than _MIN_TRIALS_PER_DIM
    trials per group per dimension, where the per-group expansion is poorly determined and costs D^3 per group
    (HANDOFF_bayesroc_tmap_wide_blocks, REPORT_tmap_wide_blocks_resolution)."""
    res = config.get("random_effects") or []
    batched = _batched_sites(res, off)
    batched_betas = _batched_betas(config, off)
    N = int(config["N"])
    n_thresh = int(config.get("n_thresh", int(config["K"]) - 1))
    mid = (n_thresh + 1) // 2 - 1
    gap_prefix = "beta_raw_gaps" if config.get("gap_link") == "softplus" else "beta_log_gaps"
    cross = {}
    for r in res:
        if r.get("cor_id") is not None:
            cross.setdefault(r["cor_id"], []).append(r)
    by_group, seen, skipped = {}, set(), []
    for conf in res:
        cid = conf.get("cor_id")
        key = ("cross", cid, conf["group"]) if cid is not None else ("re", conf.get("param"), conf["group"])
        if key in seen:
            continue
        seen.add(key)
        if conf.get("index") is None:
            continue
        try:
            B = _Block(conf, conf.get("param"), off, cross, batched.get(id(conf)))
        except KeyError:
            continue
        if any(not np.array_equal(r, np.arange(r[0], r[0] + B.n)) for r in B.z_idx):
            continue
        members = cross[cid] if cid is not None else [conf]
        B.index = np.asarray(conf["index"], dtype=int).reshape(-1) - 1
        B.slots, B.zcol, B.bpairs, B.gpairs, o = [None] * B.dim, [None] * B.dim, [], [], 0
        for m in members:
            mp, nt = m.get("param"), int(m["n_terms"])
            Zc = _re_columns(m, N)
            R = Zc.shape[1]
            n_comp = None
            if mp == "criterion" and not m.get("location_only"):
                n_comp = n_thresh
            elif mp == "criterion2" and not m.get("location_only") and config.get("K2") is not None:
                if config.get("varying_source_criteria"):
                    lay = _varying_sets(config, m)
                    n_comp = None if lay is None else lay[0] * lay[1]
                else:
                    n_comp = int(config["K2"]) - 1
            if n_comp is not None and nt == n_comp * R:
                for k in range(n_comp):
                    for r in range(R):
                        B.slots[o + k * R + r] = (mp, k, Zc[:, r].tobytes())
                        B.zcol[o + k * R + r] = Zc[:, r]
            elif nt == R:
                for r in range(R):
                    B.slots[o + r] = (mp, 0, Zc[:, r].tobytes())
                    B.zcol[o + r] = Zc[:, r]
            if mp in _RE_PARAMS and os.environ.get("BAYESROC_LEGACY_COORDS"):    # benchmarking only: column-matching pairs
                pairs = _fixed_map(config, m, mp, off, n_thresh, mid, gap_prefix, batched_betas)
                B.bpairs += [(o + d, f) for d, f in pairs]
                B.gpairs += [(o + d, flats, W) for d, flats, W in
                             _group_fixed_map(config, m, mp, off, n_thresh, mid, gap_prefix, batched_betas,
                                              B.index, B.n, {d for d, _ in pairs})]
            elif mp in _RE_PARAMS:
                if mp == "criterion2" and config.get("varying_source_criteria"):
                    B.bpairs += [(o + d, f) for d, f in _fixed_map(config, m, mp, off, n_thresh, mid, gap_prefix, batched_betas)]
                else:
                    bp, gp = _proj_map(config, m, mp, off, n_thresh, mid, gap_prefix, batched_betas, B.index, B.n)
                    B.bpairs += [(o + d, f) for d, f in bp]
                    B.gpairs += [(o + d, flats, W) for d, flats, W in gp]
            o += nt
        if o != B.dim:
            continue
        per_dim = N / (B.n * B.dim)
        if thin_rule and B.dim > _THIN_MAX_DIM and per_dim < _MIN_TRIALS_PER_DIM:
            skipped.append(f"{B.prefix} ({B.dim} x {B.n}, {per_dim:.1f} trials per group per dimension)")
            continue
        by_group.setdefault((B.prefix if split else conf["group"], B.n), []).append(B)
    sbs = sorted([_Super(g, bl) for (g, _), bl in by_group.items()], key=lambda s: s.n)
    # links: a later block's effects on (param, component) answer to an earlier block's effects on the
    # same (param, component): within each later group, the earlier effects' per-row contributions are
    # projected onto the later block's columns for that component (an intercept takes the mean over the
    # group's rows, a slope the difference it explains), whatever the terms' columns. Without this a
    # population coefficient both blocks absorb would be compensated twice.
    legacy = bool(os.environ.get("BAYESROC_LEGACY_COORDS"))
    for k, sb in enumerate(sbs):
        for e in range(k):
            E = sbs[e]
            if sb.n * E.n > _MAX_LINK:
                continue
            if legacy:                                            # benchmarking only: links between identical columns
                cd = [i for i, si in enumerate(sb.slots) if si is not None and si in E.slots]
                if cd:
                    ed = [E.slots.index(sb.slots[i]) for i in cd]
                    cnt = np.zeros((sb.n, E.n))
                    np.add.at(cnt, (sb.index, E.index), 1.0)
                    W = cnt / np.maximum(cnt.sum(1, keepdims=True), 1.0)
                    sb.links.append((e, np.array(cd, int), np.array(ed, int), np.stack([W] * len(cd))))
                continue
            comps, pairs = {}, []
            for i, si in enumerate(sb.slots):
                if si is not None:
                    comps.setdefault((si[0], si[1]), []).append(i)
            order = np.argsort(sb.index, kind="stable")
            bounds = np.searchsorted(sb.index[order], np.arange(sb.n + 1))
            for key, cdims in comps.items():
                edims = [j for j, tj in enumerate(E.slots) if tj is not None and (tj[0], tj[1]) == key]
                if not edims:
                    continue
                Zc = np.stack([sb.zcol[c] for c in cdims], 1)
                W = np.zeros((len(cdims), len(edims), sb.n, E.n))
                for g in range(sb.n):
                    rows = order[bounds[g]:bounds[g + 1]]
                    if not len(rows):
                        continue
                    M = np.linalg.pinv(Zc[rows])                       # [R_c, rows]
                    eg = E.index[rows]
                    for b, d in enumerate(edims):
                        contrib = M * E.zcol[d][rows][None, :]          # [R_c, rows]
                        for a in range(len(cdims)):
                            np.add.at(W[a, b, g], eg, contrib[a])
                W[np.abs(W) < 1e-10] = 0.0
                for a, c in enumerate(cdims):
                    for b, d in enumerate(edims):
                        if np.any(W[a, b] != 0.0):
                            pairs.append((c, d, W[a, b]))
            # one contraction per earlier block: W [P, n, n_e] over the pairs, scattered onto the dims
            if pairs:
                sb.links.append((e, np.array([c for c, _, _ in pairs], int), np.array([d for _, d, _ in pairs], int),
                                 np.stack([w for _, _, w in pairs])))
    return sbs, skipped


def _chol(Lam, Q):
    D = Q.shape[0]
    L = [[None] * D for _ in range(D)]
    for j in range(D):
        d = Lam[j, j] + Q[j, j]
        for k in range(j):
            d = d - L[j][k] * L[j][k]
        L[j][j] = jnp.sqrt(d)
        inv = 1.0 / L[j][j]
        for i in range(j + 1, D):
            v = Lam[i, j] + Q[i, j]
            for k in range(j):
                v = v - L[i][k] * L[j][k]
            L[i][j] = v * inv
    return L


def _fsolve(L, b):
    y = []
    for i in range(len(L)):
        v = b[i]
        for k in range(i):
            v = v - L[i][k] * y[k]
        y.append(v / L[i][i])
    return y


def _bsolve(L, b):
    D = len(L)
    x = [None] * D
    for i in reversed(range(D)):
        v = b[i]
        for k in range(i + 1, D):
            v = v - L[k][i] * x[k]
        x[i] = v / L[i][i]
    return x


def _n4(n):
    return -(-n // 4)


def _tile(Lam):
    """[n, D, D] -> [n4, D, D, 4] (zero-padded past n)."""
    n, D = Lam.shape[0], Lam.shape[1]
    pad = _n4(n) * 4 - n
    Lam = jnp.concatenate([Lam, jnp.zeros((pad, D, D), Lam.dtype)]) if pad else Lam
    return jnp.transpose(Lam.reshape(_n4(n), 4, D, D), (0, 2, 3, 1))


def _untile(tiled, n):
    """[n4, D, D, 4] -> [D, D, n]."""
    D = tiled.shape[1]
    return jnp.transpose(tiled, (1, 2, 0, 3)).reshape(D, D, -1)[:, :, :n]


def _native():
    import numpyro_backend as nb
    return nb._BVN_FFI


_CHUNKS = 8
_NATIVE_MAX_DIM = 64  # the kernel's kMaxD; wider blocks take the batched path
_UNROLL_MAX = 20  # without the kernel: unrolled up to here, batched linear algebra above


def _ffi(name, outs, *args):
    dt = args[0].dtype
    target = f"bayesroc_{name}" if dt == jnp.float64 else f"bayesroc_{name}_f32"
    return jax.ffi.ffi_call(target, [jax.ShapeDtypeStruct(sh, dt) for sh in outs])(*args)


@jax.custom_vjp
def _group_solve(Lam_t, Q, s, delta, zeta):
    """Per group g (dimensions in the kernel's order): r_g = s_g - Lam_g delta_g,
    L_g = chol(Lam_g + Q), u_g = L_g^-T (L_g^-1 r_g + zeta_g), ld_g = sum log diag L_g.
    Lam_t [n4, D, D, 4] tiled, Q [D, D], s, delta, zeta [D, n] -> u [D, n], ld [n]. Native,
    with the adjoint in closed form (Lam and s fixed)."""
    D, n = s.shape
    return _ffi("tmap_fwd", [(D, n), (n,)], Lam_t, Q, s, delta, zeta)


def _group_solve_fwd(Lam_t, Q, s, delta, zeta):
    return _group_solve(Lam_t, Q, s, delta, zeta), (Lam_t, Q, s, delta, zeta)


def _group_solve_bwd(res, cts):
    Lam_t, Q, s, delta, zeta = res
    ubar, ldbar = cts
    D, n = s.shape
    zetabar, deltabar, Qbar = _ffi("tmap_bwd", [(D, n), (D, n), (_CHUNKS, D, D)],
                                   Lam_t, Q, s, delta, zeta, ubar, ldbar)
    return (jnp.zeros_like(Lam_t), jnp.sum(Qbar, 0), jnp.zeros_like(s), deltabar, zetabar)


_group_solve.defvjp(_group_solve_fwd, _group_solve_bwd)


class TMap:
    def __init__(self, sbs, post):
        self.sbs = sbs
        self.post = post
        self.sizes = [(sb.D * sb.D * _n4(sb.n) * 4, sb.D * sb.n, len(sb.bflat), sb.D * sb.n, sb.D) for sb in sbs]
        self.aux_size = int(sum(sum(s) for s in self.sizes))
        self.specs = [f"{sb.name} ({sb.D} x {sb.n})" for sb in sbs]
        self._fwd = self._inv = self._anch = None

    def scale_mask(self, D):
        """Flat positions of the mapped blocks' standard deviations and correlations."""
        m = np.zeros(D, bool)
        for sb in self.sbs:
            for b in sb.blocks:
                m[b.x_idx] = True
        return m

    def _unpack(self, aux):
        out, o = [], 0
        for sb, (a, b, c, d, e) in zip(self.sbs, self.sizes):
            perm = aux[o + a + b + c + d:o + a + b + c + d + e].astype(jnp.int32)
            tiled = aux[o:o + a].reshape(_n4(sb.n), sb.D, sb.D, 4)
            out.append(dict(Lam_t=tiled, Lam=_untile(tiled, sb.n)[::-1, ::-1],
                            s=aux[o + a:o + a + b].reshape(sb.D, sb.n),
                            bref=aux[o + a + b:o + a + b + c],
                            uref=aux[o + a + b + c:o + a + b + c + d].reshape(sb.D, sb.n),
                            perm=perm, inv=jnp.argsort(perm)))
            o += a + b + c + d + e
        return out

    def _delta(self, k, v, an, us, ans):
        """The shift of each group's effects implied by the fixed effects and earlier blocks,
        [D, n] in the block's permuted order, or None when nothing couples."""
        sb = self.sbs[k]
        rows = [None] * sb.D
        if len(sb.bflat):
            db = v[sb.bflat] - an["bref"]
            for i, d in enumerate(sb.bdims):
                rows[int(d)] = jnp.broadcast_to(db[i], (sb.n,))
            if len(sb.gdims):
                dg = jnp.asarray(sb.gW, dtype=v.dtype) @ db
                for i, d in enumerate(sb.gdims):
                    rows[int(d)] = dg[i] if rows[int(d)] is None else rows[int(d)] + dg[i]
        for e, cd, ed, W in sb.links:
            du = jnp.einsum("pge,pe->pg", jnp.asarray(W, dtype=v.dtype), (us[e] - ans[e]["uref"])[ed])
            acc = jax.ops.segment_sum(du, jnp.asarray(cd), num_segments=sb.D)
            for c in np.unique(cd):
                rows[int(c)] = acc[c] if rows[int(c)] is None else rows[int(c)] + acc[c]
        if all(r is None for r in rows):
            return None
        zero = jnp.zeros((sb.n,), dtype=v.dtype)
        return jnp.take(jnp.stack([zero if r is None else r for r in rows]), an["perm"], axis=0)

    def _linear(self, k, v, an, us, ans):
        sb = self.sbs[k]
        delta = self._delta(k, v, an, us, ans)
        if delta is None:
            return [an["s"][a] for a in range(sb.D)]
        out = []
        for a in range(sb.D):
            acc = an["s"][a]
            for b in range(sb.D):
                acc = acc - an["Lam"][a, b] * delta[b]
            out.append(acc)
        return out

    def _prior(self, k, v, post, an):
        sb = self.sbs[k]
        A = sb.A(post).astype(v.dtype)
        Ainv = sb.Ainv(post).astype(v.dtype)
        Q = Ainv.T @ Ainv
        return A, Ainv, jnp.take(jnp.take(Q, an["perm"], axis=0), an["perm"], axis=1)[::-1, ::-1]

    def _parts(self, k, v, post, ans, us):
        sb, an = self.sbs[k], ans[k]
        A, Ainv, Q = self._prior(k, v, post, an)
        L = _chol(an["Lam"][::-1, ::-1], Q)
        return A, Ainv, L, _fsolve(L, self._linear(k, v, an, us, ans)[::-1])

    def _dense(self, k, v, post, ans, us):
        """_parts as batched linear algebra, for wide blocks: L [n, D, D] and y [n, D] in the
        kernel's order, with an operation count that does not grow with D."""
        sb, an = self.sbs[k], ans[k]
        A, Ainv, Q = self._prior(k, v, post, an)
        Lam = jnp.transpose(_untile(an["Lam_t"], sb.n), (2, 0, 1))
        r = an["s"][::-1].T
        delta = self._delta(k, v, an, us, ans)
        if delta is not None:
            r = r - jnp.einsum("nab,nb->na", Lam, delta[::-1].T)
        L = jnp.linalg.cholesky(Lam + Q[None])
        return A, Ainv, L, solve_triangular(L, r[..., None], lower=True)[..., 0]

    def forward(self, tht, aux):
        """Sampler coordinates -> the model's unconstrained coordinates, and log|dz/dzeta|^-1."""
        ans = self._unpack(aux.astype(tht.dtype))
        post = self.post(tht)
        th, ld, us = tht, 0.0, {}
        native = _native()
        for k, sb in enumerate(self.sbs):
            an = ans[k]
            zeta = jnp.take(sb.get(tht), an["perm"], axis=0)[::-1]
            if native and sb.D <= _NATIVE_MAX_DIM:
                A, Ainv, Q = self._prior(k, tht, post, an)
                delta = self._delta(k, tht, an, us, ans)
                delta = jnp.zeros_like(zeta) if delta is None else delta[::-1]
                u_rev, ldg = _group_solve(an["Lam_t"], Q, an["s"][::-1], delta, zeta)
                u = jnp.take(u_rev[::-1], an["inv"], axis=0)
                ld = ld + jnp.sum(ldg)
            elif sb.D <= _UNROLL_MAX:
                A, Ainv, L, y = self._parts(k, tht, post, ans, us)
                u = jnp.take(jnp.stack(_bsolve(L, [y[i] + zeta[i] for i in range(sb.D)])[::-1]),
                             an["inv"], axis=0)
                ld = ld + sum(jnp.sum(jnp.log(L[i][i])) for i in range(sb.D))
            else:
                A, Ainv, L, y = self._dense(k, tht, post, ans, us)
                u_rev = solve_triangular(jnp.swapaxes(L, 1, 2), (y + zeta.T)[..., None], lower=False)[..., 0]
                u = jnp.take(u_rev.T[::-1], an["inv"], axis=0)
                ld = ld + jnp.sum(jnp.log(jnp.diagonal(L, axis1=1, axis2=2)))
            us[k] = u
            th = sb.put(th, Ainv @ u)
            ld = ld + sb.n * jnp.sum(jnp.log(jnp.diag(A)))
        return th, ld

    def inverse(self, th, aux):
        ans = self._unpack(aux.astype(th.dtype))
        post = self.post(th)
        tht, us = th, {}
        for k, sb in enumerate(self.sbs):
            dense = sb.D > _UNROLL_MAX
            A, Ainv, L, y = (self._dense if dense else self._parts)(k, th, post, ans, us)
            u = A @ sb.get(th)
            us[k] = u
            ur = jnp.take(u, ans[k]["perm"], axis=0)[::-1]
            if dense:
                zp = (jnp.einsum("nji,nj->ni", L, ur.T) - y).T[::-1]
            else:
                zp = jnp.stack([sum(L[j][i] * ur[j] for j in range(i, sb.D)) - y[i] for i in range(sb.D)][::-1])
            tht = sb.put(tht, jnp.take(zp, ans[k]["inv"], axis=0))
        return tht

    def wrap(self, pot_off):
        """pot_off(v, OFF) -> value_and_grad of the mapped potential, (v, OFF, AUX) -> (U, g)."""
        def pot(v, OFF, AUX):
            th, ld = self.forward(v, AUX)
            return pot_off(th, OFF) + ld
        return jax.value_and_grad(pot)

    def forward_chains(self, tht, aux):
        """Sampler -> model coordinates for positions [chains, D] with anchors [chains, A]."""
        if self._fwd is None:
            self._fwd = jax.jit(lambda v, a: self.forward(v, a)[0])
        return np.stack([np.asarray(self._fwd(jnp.asarray(x), jnp.asarray(a)))
                         for x, a in zip(np.asarray(tht), np.asarray(aux))])

    def inverse_chains(self, th, aux):
        if self._inv is None:
            self._inv = jax.jit(self.inverse)
        return np.stack([np.asarray(self._inv(jnp.asarray(x), jnp.asarray(a)))
                         for x, a in zip(np.asarray(th), np.asarray(aux))])

    def lower_anchors(self, vg, chains, D, parallel=True):
        """The anchors program over the chains (pmap, or lax.map under jit when not parallel),
        lowered and ready to compile."""
        f = lambda th: self.anchors(vg, th)
        prog = jax.pmap(f, axis_name="chain") if parallel else jax.jit(lambda r: jax.lax.map(f, r))
        return prog.lower(jnp.zeros((chains, D)))

    def anchors_chains(self, vg, ref, parallel=True):
        """Packed anchors [chains, A], each chain's from its own reference position. The
        dimension order, which defines the coordinate system, is pooled over the chains, so
        every chain samples in the same coordinates with the shared mass matrix."""
        ref = np.asarray(ref, dtype=np.float64)
        C = ref.shape[0]
        if self._anch is None:
            self._anch = self.lower_anchors(vg, C, ref.shape[1], parallel).compile()
        raw = jax.tree.map(np.asarray, self._anch(jnp.asarray(ref)))
        packed = [[] for _ in range(C)]
        for sb, r in zip(self.sbs, raw):
            Lam = r["Lam"]
            s_ = -np.swapaxes(r["q0"], 1, 2) + np.einsum("cnab,cnb->cna", Lam, np.swapaxes(r["u"], 1, 2))
            strength = np.mean(np.einsum("cnaa->cna", Lam), axis=(0, 1)) * r["A2"].mean(0)
            weight = strength / (1.0 + strength)
            idx = np.arange(sb.D)
            # informed dims first in reversed natural order, then the rest by falling weight
            perm = np.lexsort((-idx, np.where(weight >= _UNINFORMED, -idx - 1.0, -weight)))
            for c in range(C):
                Lp = Lam[c][:, perm][:, :, perm][:, ::-1, ::-1]
                packed[c] += [np.asarray(_tile(jnp.asarray(Lp))).reshape(-1), s_[c][:, perm].T.reshape(-1),
                              ref[c, sb.bflat], r["u0"][c].reshape(-1), perm.astype(np.float64)]
        return np.stack([np.concatenate(p) for p in packed])

    def anchors(self, vg, th):
        """Raw anchors at one position th (model coordinates), one dict per super-block: each
        group's log-likelihood expanded to second order in its effects near that group's
        conditional mode (up to _NEWTON capped Newton steps from its effects at th with
        everything else held fixed, each kept only for the groups whose conditional gradient it
        shrinks): Lam [n, D, D] (PSD part), q0 and u [D, n], the starting u0 and diag(A A^T).
        Traceable: the
        expansion's D + 1 gradients run in a lax.map and the Newton steps in a fori_loop, so the
        gradient is compiled once per super-block."""
        post0 = self.post(th)
        out = []
        for sb in self.sbs:
            A = sb.A(post0)
            Ainv = solve_triangular(A, jnp.eye(sb.D, dtype=A.dtype), lower=True)
            Q = Ainv.T @ Ainv
            cap = _NEWTON_CAP * jnp.sqrt(jnp.diag(A @ A.T))[:, None]
            eye = jnp.eye(sb.D, dtype=A.dtype)

            def neg_grad(u):
                z = solve_triangular(A, u, lower=True)
                g = sb.get(vg(sb.put(th, z))[1]) - z
                return solve_triangular(A.T, g, lower=False)

            def expand(u):
                qs = jax.lax.map(lambda j: neg_grad(u + _FD_STEP * jnp.where(j < 0, 0.0, eye[jnp.maximum(j, 0)])[:, None]),
                                 jnp.arange(-1, sb.D))
                q0 = qs[0]
                H = jnp.transpose((qs[1:] - q0[None]) / _FD_STEP, (2, 1, 0))
                ev, V = jnp.linalg.eigh(0.5 * (H + jnp.swapaxes(H, 1, 2)))
                return q0, jnp.einsum("nab,nb,ncb->nac", V, jnp.maximum(ev, 0.0), V)

            def merit(u, q0):
                r = A.T @ (q0 + Q @ u)
                return jnp.sum(r * r, 0)

            def newton(_, c):
                u, q0, Lam, m = c
                step = jnp.linalg.solve(Lam + Q[None], -(q0.T + u.T @ Q)[..., None])[..., 0].T
                u1 = u + jnp.clip(step, -cap, cap)
                q1, Lam1 = expand(u1)
                m1 = merit(u1, q1)
                ok = jnp.isfinite(m1) & (m1 < m)
                return (jnp.where(ok[None, :], u1, u), jnp.where(ok[None, :], q1, q0),
                        jnp.where(ok[:, None, None], Lam1, Lam), jnp.where(ok, m1, m))

            u0 = A @ sb.get(th)
            q0, Lam = expand(u0)
            u, q0, Lam, _ = jax.lax.fori_loop(0, _NEWTON, newton, (u0, q0, Lam, merit(u0, q0)))
            out.append(dict(Lam=Lam, q0=q0, u=u, u0=u0, A2=jnp.diag(A @ A.T)))
        return out


def build(model, config, off, unflatten, split=False, thin_rule=True, links=None):
    """A TMap over every random-effect block it can represent, or None when there is none. Thin wide blocks are left
    unmapped by the design rule unless thin_rule is False. `links`: True keeps the links between super-blocks, False
    cuts them, None cuts them only when a block was left unmapped (measured on vrdp2d: 1.6-2.5x on the mapped classes)."""
    import drghmc as dr
    sbs, skipped = _supers(config, off, split, thin_rule)
    if not sbs:
        return None
    keep_links = (not skipped) if links is None else bool(links)
    if not keep_links:
        for sb in sbs:
            sb.links = []
    _, _, post_trim = dr._model_info(model, dr._trim_config(config))
    tm = TMap(sbs, lambda th: post_trim(unflatten(th)))
    tm.skipped, tm.links_kept = skipped, keep_links
    return tm
