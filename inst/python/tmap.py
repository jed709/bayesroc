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
import re
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular, block_diag


_FD_STEP = 1e-3
_NEWTON = 3
_NEWTON_CAP = 0.5
_UNINFORMED = 0.1
_CAND_MIN_DIM = 10
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

    def sd(self, post):
        if self.sd_det is not None:
            return post[self.sd_det]
        ts = [t for _, t in self.sd_sites]
        if len({s for s, _ in self.sd_sites}) == 1 and ts == list(range(len(ts))):
            return jnp.reshape(post[self.sd_sites[0][0]], (-1,))[:len(ts)]
        return jnp.stack([jnp.reshape(post[s], (-1,))[0 if t is None else t] for s, t in self.sd_sites])

    def A(self, post):
        sd = self.sd(post)
        Lk = f"L_corr_{self.prefix}"
        L = post[Lk] if Lk in post else jnp.eye(self.dim, dtype=sd.dtype)
        return sd[:, None] * L

    def Ainv(self, post):
        """(sd * L)^-1 = L^-1 diag(1/sd); a diagonal when the block is uncorrelated."""
        sd = self.sd(post)
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


def _varying_row_bins(config):
    """0-based set of varying source criteria each observation reads (from its detection response)."""
    y = np.asarray(config["y"], dtype=int).reshape(-1) - 1
    vbin = config.get("vbin")
    return y if vbin is None else np.asarray(vbin, dtype=int).reshape(-1)[y] - 1


def _proj_map_varying(config, conf, off, gap_prefix, index, n):
    """_proj_map for a criterion2 random effect under varying source criteria. Its dimension for
    (bin b, threshold component k, term r) acts on the rows whose detection response reads bin b;
    the population coefficients there are each cell's set coefficient for (b, k) and the shared
    covariate coefficients for k. Within every group the effect is projected onto them as in
    _proj_map, so a group nested in a cell answers to that cell's set only."""
    N = int(config["N"])
    Zc = _re_columns(conf, N)
    R = Zc.shape[1]
    nth = int(config["K2"]) - 1
    mid = (nth + 1) // 2 - 1
    n_vbin = int(config.get("n_vbin") or config["K"])
    n_vcell = int(config.get("n_vcell") or 1)
    nt = int(conf["n_terms"])
    if nt == n_vbin * nth * R:
        comps = list(range(nth))
    elif nt == n_vbin * R:
        comps = [mid]
    else:
        return [], []
    rb = _varying_row_bins(config)
    cell = (np.asarray(config["vcell"], dtype=int).reshape(-1) - 1 if n_vcell > 1 else np.zeros(N, int))
    cov = config.get("crit2_cov")
    Xcov = np.asarray(cov["X"], dtype=float).reshape(N, -1) if cov else np.zeros((N, 0))
    gv, gcov = gap_prefix + "_2_varying", gap_prefix + "_2_cov"
    order = np.argsort(index, kind="stable")
    bounds = np.searchsorted(index[order], np.arange(n + 1))
    bp, gp = [], []
    for b in range(n_vbin):
        on = (rb == b).astype(float)
        Zb = Zc * on[:, None]
        for ci, k in enumerate(comps):
            g = None if k == mid else _gap_of_component(k, mid, nth)
            flats, cols = [], []
            for c in range(n_vcell):
                st = b + n_vbin * c
                flats.append(_beta_site(off, None, [("beta_thresh_mid_2_varying", (st,))] if g is None
                                        else [(gv, (st, g))]))
                cols.append(on * (cell == c))
            for q in range(Xcov.shape[1]):
                flats.append(_beta_site(off, None, [("beta_thresh_mid_2_cov", (q,))] if g is None
                                        else [(gcov, (g, q))]))
                cols.append(on * Xcov[:, q])
            X = np.stack(cols, 1)
            P = X.shape[1]
            W = np.zeros((n, R, P))
            act = np.zeros((n, R), bool)
            for grp in range(n):
                rows = order[bounds[grp]:bounds[grp + 1]]
                if not len(rows):
                    continue
                act[grp] = np.any(Zb[rows] != 0.0, 0)
                W[grp] = np.linalg.lstsq(Zb[rows], X[rows], rcond=None)[0]
            W[np.abs(W) < 1e-10] = 0.0
            for r in range(R):
                d = (b * len(comps) + ci) * R + r
                Wr = W[:, r, :]
                one = [j for j in range(P) if flats[j] is not None and act[:, r].any()
                       and np.all(np.abs(Wr[act[:, r], j] - 1.0) < 1e-9) and np.all(Wr[~act[:, r], j] == 0.0)]
                rest = [j for j in range(P) if flats[j] is not None and j not in one and np.any(Wr[:, j] != 0.0)]
                for j in one:
                    bp.append((d, int(flats[j])))
                if rest:
                    gp.append((d, [int(flats[j]) for j in rest], Wr[:, rest].copy()))
    return bp, gp


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
        if param == "dprime" and config.get("needs_ordered_dprime") and "dprime_order_loggap" in off:
            X, flats = _ordered_dprime_design(config, off, N)
            centre = None
        elif pc and int(pc.get("n_coef", 0)) >= 1:
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


def _ordered_dprime_design(config, off, N):
    """mixture with dprime > dprime2 on the old items' average: dprime's pivot coefficient is
    (exp(gap) + xbar2' b2 - sum_{j != pivot} xbar1_j b1_j) / xbar1_pivot, so X1 b1 is linear in
    the free dprime coefficients, the dprime2 coefficients and exp(gap) (the gap enters the map
    through exp, see _Super.bexp)."""
    pcs = config.get("params") or {}
    X1 = np.asarray(pcs["dprime"]["X"], dtype=float).reshape(N, -1)
    X2 = np.asarray(pcs["dprime2"]["X"], dtype=float).reshape(N, -1)
    piv = int(config["dprime_pivot"]) - 1
    x1 = np.asarray(config["dprime_xbar"], dtype=float).reshape(-1)
    x2 = np.asarray(config["dprime2_xbar"], dtype=float).reshape(-1)
    cols, flats, k = [], [], 0
    for j in range(X1.shape[1]):
        if j == piv:
            continue
        k += 1
        cols.append(X1[:, j] - X1[:, piv] * x1[j] / x1[piv])
        flats.append(_beta_site(off, None, [(f"beta_dprime_free[{k}]", None), ("beta_dprime_free", (k - 1,))]))
    for j in range(X2.shape[1]):
        cols.append(X1[:, piv] * x2[j] / x1[piv])
        flats.append(_beta_site(off, None, [(f"beta_dprime2[{j + 1}]", None), ("beta_dprime2", (j,))]))
    cols.append(X1[:, piv] / x1[piv])
    flats.append(off["dprime_order_loggap"][0].start)
    return np.stack(cols, 1), flats


_OLD_ONLY = ("dprime", "sigma", "lambda", "dprime2", "sigma2", "dprime_B", "lambda_B", "discrim", "discrim_B",
             "sigma_B", "sigma2_B", "rho", "rho_B", "lambda2", "lambda2_B")
_NEW_ONLY = ("dprime_L", "sigma_L", "lambda_L", "rho_N")


def _param_rows(config, param, N):
    """The observations a parameter enters the likelihood through (old items for the signal
    parameters, new items for the lure ones), or None for all: a group's effects trade off
    with the population coefficients on those rows only."""
    if config.get("family") in ("cumulative", "bivariate_cumulative", "source_mixture"):
        return None
    if config.get("family") == "metadt":
        if param not in ("mratio", "mratio_N") or config.get("metad_shared"):
            return None
        side = 2 * np.asarray(config["y"], dtype=int).reshape(-1) > int(config["K"])
        if len(side) != N:
            return None
        return side if param == "mratio" else ~side
    if param not in _OLD_ONLY and param not in _NEW_ONLY:
        return None
    if param == "rho" and config.get("cdp_rho_shared"):
        return None
    if config.get("item_type") is not None:
        old = np.asarray(config["item_type"], dtype=int).reshape(-1) != 1
    elif config.get("is_old") is not None:
        io = np.asarray(config["is_old"], dtype=float).reshape(-1)
        old = io == io.max()
    else:
        return None
    if len(old) != N:
        return None
    return old if param in _OLD_ONLY else ~old


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
    used = _param_rows(config, param, N)
    for g in range(n):
        rows = order[bounds[g]:bounds[g + 1]]
        if used is not None:
            rows = rows[used[rows]]
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
        cand, self.cand_names, o = np.zeros(self.D, bool), [], 0
        for b in blocks:
            if b.dim > _CAND_MIN_DIM:
                cand[o:o + b.dim] = True
                self.cand_names.append(b.prefix)
            o += b.dim
        self.cand = cand if cand.any() else None
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
        self.bexp = np.zeros(len(self.bflat), bool)
        self.gdims = np.array([d for d, _, _ in gp], int)
        self.gW = np.zeros((len(gp), self.n, len(self.bflat)))
        for i, (_, flats, W) in enumerate(gp):
            for c, f in enumerate(flats):
                self.gW[i, :, pos[f]] += W[:, c]
        self.links = []
        # one gather / one scatter for the whole [D, n] block: D chained dynamic_update_slices
        # would each copy the full parameter vector, forward and again in the VJP
        self.pos_np = np.add.outer(np.array(self.starts, int), np.arange(self.n))
        self.pos = jnp.asarray(self.pos_np)
        self.linked = False
        if self.cand is not None:
            self.kept_nat, self.cand_nat = np.nonzero(~cand)[0], np.nonzero(cand)[0]
            self.kept_index, self.cand_index = np.full(self.D, -1), np.full(self.D, -1)
            self.kept_index[self.kept_nat] = np.arange(len(self.kept_nat))
            self.cand_index[self.cand_nat] = np.arange(len(self.cand_nat))
            self.pos1, self.pos2 = jnp.asarray(self.pos_np[self.kept_nat]), jnp.asarray(self.pos_np[self.cand_nat])
            self.kept_blocks = [b for b in blocks if b.dim <= _CAND_MIN_DIM]
            self.cand_blocks = [b for b in blocks if b.dim > _CAND_MIN_DIM]

    def A(self, post):
        return block_diag(*[b.A(post) for b in self.blocks])

    def Ainv(self, post):
        return block_diag(*[b.Ainv(post) for b in self.blocks])

    def A1(self, post):
        return block_diag(*[b.A(post) for b in self.kept_blocks])

    def Ainv1(self, post):
        return block_diag(*[b.Ainv(post) for b in self.kept_blocks])

    def A2(self, post):
        return block_diag(*[b.A(post) for b in self.cand_blocks])

    def Ainv2(self, post):
        return block_diag(*[b.Ainv(post) for b in self.cand_blocks])

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


def _supers(config, off, split=False):
    """Super-blocks: the random-effect blocks of each grouping factor, mapped together."""
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
    by_group, seen = {}, set()
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
                rb = (_varying_row_bins(config) if mp == "criterion2" and config.get("varying_source_criteria")
                      else None)
                nth2 = (int(config["K2"]) - 1) if rb is not None else 1
                for k in range(n_comp):
                    zk = Zc if rb is None else Zc * (rb == k // nth2)[:, None]
                    for r in range(R):
                        B.slots[o + k * R + r] = (mp, k, zk[:, r].tobytes())
                        B.zcol[o + k * R + r] = zk[:, r]
            elif nt == R:
                for r in range(R):
                    B.slots[o + r] = (mp, 0, Zc[:, r].tobytes())
                    B.zcol[o + r] = Zc[:, r]
            if mp == "criterion2" and config.get("varying_source_criteria"):
                bp, gp = _proj_map_varying(config, m, off, gap_prefix, B.index, B.n)
            else:
                bp, gp = _proj_map(config, m, mp, off, n_thresh, mid, gap_prefix, batched_betas, B.index, B.n)
            B.bpairs += [(o + d, f) for d, f in bp]
            B.gpairs += [(o + d, flats, W) for d, flats, W in gp]
            o += nt
        if o != B.dim:
            continue
        by_group.setdefault((B.prefix if split else conf["group"], B.n), []).append(B)
    sbs = sorted((_Super(key[0], bl) for key, bl in by_group.items()), key=lambda s: s.n)
    if "dprime_order_loggap" in off:
        for sb in sbs:
            sb.bexp = sb.bflat == off["dprime_order_loggap"][0].start
    # links: a later block's effects on (param, component) answer to an earlier block's effects on the
    # same (param, component): within each later group, the earlier effects' per-row contributions are
    # projected onto the later block's columns for that component (an intercept takes the mean over the
    # group's rows, a slope the difference it explains), whatever the terms' columns. Without this a
    # population coefficient both blocks absorb would be compensated twice.
    for k, sb in enumerate(sbs):
        for e in range(k):
            E = sbs[e]
            if sb.n * E.n > _MAX_LINK:
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
            # one contraction per earlier block and distinct weight matrix (intercept-only terms share
            # one): W [n, n_e] against the pairs' earlier effects, scattered onto the dims
            if pairs:
                groups = []
                for c, d, w in pairs:
                    for g in groups:
                        if np.array_equal(g[2], w):
                            g[0].append(c)
                            g[1].append(d)
                            break
                    else:
                        groups.append(([c], [d], w))
                for cs, ds, w in groups:
                    sb.links.append((e, np.array(cs, int), np.array(ds, int), w))
        sb.links_kept = sb.links
        if sb.cand is not None:
            sb.links_kept = [(e, cd[m], ed[m], W) for e, cd, ed, W in sb.links for m in [~sb.cand[cd]] if m.any()]
        for e, _, _, _ in sb.links:
            sbs[e].linked = True
    return sbs


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
_TRI_MIN = 16  # tmap_kernel.h kTriMin: wider blocks keep their factors for the adjoint
_UNROLL_MAX = 20  # without the kernel: unrolled up to here, batched linear algebra above


def _ffi(name, outs, *args):
    dt = args[0].dtype
    target = f"bayesroc_{name}" if dt == jnp.float64 else f"bayesroc_{name}_f32"
    return jax.ffi.ffi_call(target, [jax.ShapeDtypeStruct(sh, dt) for sh in outs])(*args)


@jax.custom_vjp
def _group_solve(Lam_t, Q, s, delta, zeta):
    """Per group g (dimensions in the kernel's order): r_g = s_g - Lam_g delta_g,
    L_g = chol(Lam_g + Q), u_g = L_g^-T (L_g^-1 r_g + zeta_g), ld_g = sum log diag L_g.
    Lam_t [n4, D, D, 4] tiled, Q [D, D], s, zeta [D, n], delta [D, n] or empty (no shift) ->
    u [D, n], ld [n]. Native, with the adjoint in closed form (Lam and s fixed)."""
    return _group_solve_fwd(Lam_t, Q, s, delta, zeta)[0]


def _tri_shapes(D, n):
    return [(_n4(n), D * (D + 1) // 2 * 4), (D, n)] if D >= _TRI_MIN else [(1,), (1,)]


def _group_solve_fwd(Lam_t, Q, s, delta, zeta):
    D, n = s.shape
    u, ld, lfac, yres = _ffi("tmap_fwd", [(D, n), (n,)] + _tri_shapes(D, n), Lam_t, Q, s, delta, zeta)
    return (u, ld), (Lam_t, Q, s, delta, zeta, lfac, yres, u)


def _group_solve_bwd(res, cts):
    Lam_t, Q, s, delta, zeta, lfac, yres, u = res
    ubar, ldbar = cts
    D, n = s.shape
    zetabar, deltabar, Qbar = _ffi("tmap_bwd", [(D, n), (D, n) if delta.size else (1,), (_CHUNKS, D, D)],
                                   Lam_t, Q, s, delta, zeta, ubar, ldbar, lfac, yres, u)
    deltabar = deltabar if delta.size else jnp.zeros_like(delta)
    return (jnp.zeros_like(Lam_t), jnp.sum(Qbar, 0), jnp.zeros_like(s), deltabar, zetabar)


_group_solve.defvjp(_group_solve_fwd, _group_solve_bwd)


@jax.custom_vjp
def _block_map(Lam_t, s, delta, z, A, perm):
    """A wide block's whole map step, natively: z [D, n] (sampler coordinates, natural order),
    A [D, D] the prior factor (lower triangular), perm [D] int32 (kernel row i is natural row
    perm[i]); Lam_t, s, delta as in _group_solve, in the kernel's order -> th = A^-1 u [D, n],
    ld [n], u [D, n] (natural order), with A^-1 and Q = A^-T A^-1 formed inside."""
    return _block_map_fwd(Lam_t, s, delta, z, A, perm)[0]


def _block_map_fwd(Lam_t, s, delta, z, A, perm):
    D, n = s.shape
    th, ld, u, lfac, yres, uk, Qk, Ainv = _ffi("tmap_blk_fwd", [(D, n), (n,), (D, n)] + _tri_shapes(D, n) + [(D, n), (D, D), (D, D)],
                                               Lam_t, s, delta, z, A, perm)
    return (th, ld, u), (Ainv, perm, Qk, lfac, yres, uk, u, delta, Lam_t, s)


def _block_map_bwd(res, cts):
    Ainv, perm, Qk, lfac, yres, uk, u, delta, Lam_t, s = res
    thbar, ldbar, ubar = cts
    D, n = s.shape
    zbar, deltabar, Abar = _ffi("tmap_blk_bwd", [(D, n), (D, n) if delta.size else (1,), (D, D)],
                                Ainv, perm, Qk, lfac, yres, uk, u, thbar, ldbar, ubar, delta)
    deltabar = deltabar if delta.size else jnp.zeros_like(delta)
    return (jnp.zeros_like(Lam_t), jnp.zeros_like(s), deltabar, zbar, Abar,
            np.zeros(perm.shape, dtype=jax.dtypes.float0))


_block_map.defvjp(_block_map_fwd, _block_map_bwd)


def _link_groups(links):
    """Links from the same earlier block with as many pairs each, stacked so that one contraction
    serves them all: (e, ed [K, m], cd [K, m], W [K, n, n_e])."""
    groups = {}
    for e, cd, ed, W in links:
        groups.setdefault((e, len(ed)), []).append((cd, ed, W))
    return [(e, np.stack([x[1] for x in g]), np.stack([x[0] for x in g]), np.stack([x[2] for x in g]))
            for (e, _), g in groups.items()]


class TMap:
    def __init__(self, sbs, post):
        self.sbs = sbs
        self.post = post
        self.sizes = [(sb.D * sb.D * _n4(sb.n) * 4, sb.D * sb.n, len(sb.bflat), sb.D * sb.n, sb.D) for sb in sbs]
        self.body = int(sum(sum(s) for s in self.sizes))
        self.kept_dims = [0 if sb.cand is None else sb.D - int(sb.cand.sum()) for sb in sbs]
        self.kept_sizes = [d * d * _n4(sb.n) * 4 for d, sb in zip(self.kept_dims, sbs)]
        self.aux_size = self.body + sum(self.kept_sizes) + len(sbs)
        self.specs = [f"{sb.name} ({sb.D} x {sb.n})" for sb in sbs]
        self.static = (0,) * len(sbs)
        self._views = {self.static: self}
        self._fwd = self._inv = self._anch = None

    def view(self, flags):
        """The map with the super-blocks flagged 1 treated as in `_cheap` (their candidate blocks
        non-centred): one object per configuration, so each traces and compiles its own
        programs; the anchors program and the views are shared."""
        import copy
        flags = tuple(int(f) for f in flags)
        if flags not in self._views:
            v = copy.copy(self)
            v.static = flags
            v._fwd = v._inv = None
            self._views[flags] = v
        return self._views[flags]

    def cheap_of(self, aux):
        """The view matching the flags stored in `aux`."""
        return self.view(self.flags(aux))

    def flags(self, aux):
        """Per super-block: 1 where its candidate blocks are sampled non-centred, else 0."""
        aux = np.asarray(aux)
        return aux[0, -len(self.sbs):].copy() if aux.shape[1] == self.aux_size else np.zeros(len(self.sbs))

    def _complete(self, aux):
        """[chains, aux_size] from packed anchors, deriving the kept-dimension tiles and
        flags when they are absent."""
        aux = np.asarray(aux, dtype=np.float64)
        if aux.shape[1] == self.aux_size:
            return aux
        if aux.shape[1] != self.body:
            raise ValueError(f"transport-map anchors have {aux.shape[1]} values; expected {self.aux_size}")
        C, tail, o = aux.shape[0], [], 0
        aux = aux.copy()
        for sb, (a, b, c, d, e), D1 in zip(self.sbs, self.sizes, self.kept_dims):
            aux[:, o + a:o + a + b] = aux[:, o + a:o + a + b].reshape(C, sb.D, sb.n)[:, ::-1].reshape(C, -1)
            if D1:
                perm = aux[0, o + a + b + c + d:o + a + b + c + d + e].astype(int)
                ks1 = self._kept(sb, perm)[0]
                tiles = aux[:, o:o + a].reshape(C, _n4(sb.n), sb.D, sb.D, 4)
                tail.append(tiles[:, :, ks1][:, :, :, ks1].reshape(C, -1))
            o += a + b + c + d + e
        return np.concatenate([aux] + tail + [np.zeros((C, len(self.sbs)))], 1)

    def with_flags(self, aux, flags):
        out = self._complete(aux).copy()
        out[:, -len(self.sbs):] = np.asarray(flags, dtype=np.float64)[None]
        return out

    def scale_mask(self, D):
        """Flat positions of the mapped blocks' standard deviations and correlations."""
        m = np.zeros(D, bool)
        for sb in self.sbs:
            for b in sb.blocks:
                m[b.x_idx] = True
        return m

    def parts(self, aux):
        """Flat anchors [chains, A] -> per block (Lam tiles, s, bref, uref, perm), each [chains, ...]:
        the form the compiled programs take, so that none of them slices the anchors per gradient."""
        aux = self._complete(aux)
        C, out, o, ok = aux.shape[0], [], 0, self.body
        for k, (sb, (a, b, c, d, e), D1, f) in enumerate(zip(self.sbs, self.sizes, self.kept_dims, self.kept_sizes)):
            out += [aux[:, o:o + a].reshape(C, _n4(sb.n), sb.D, sb.D, 4),
                    aux[:, o + a:o + a + b].reshape(C, sb.D, sb.n),
                    aux[:, o + a + b:o + a + b + c],
                    aux[:, o + a + b + c:o + a + b + c + d].reshape(C, sb.D, sb.n),
                    aux[:, o + a + b + c + d:o + a + b + c + d + e].astype(np.int32),
                    aux[:, ok:ok + f].reshape(C, _n4(sb.n), D1, D1, 4),
                    aux[:, self.body + sum(self.kept_sizes) + k]]
            o += a + b + c + d + e
            ok += f
        return tuple(jnp.asarray(x) for x in out)

    def _unpack(self, parts, dtype):
        out = []
        for k, sb in enumerate(self.sbs):
            tiled, s_k, bref, uref = (x.astype(dtype) for x in parts[7 * k:7 * k + 4])
            perm, kept, flag = parts[7 * k + 4], parts[7 * k + 5].astype(dtype), parts[7 * k + 6]
            perm_rev = perm[::-1]
            out.append(dict(Lam_t=tiled, Lam=_untile(tiled, sb.n)[::-1, ::-1], s=s_k[::-1], s_k=s_k, bref=bref,
                            uref=uref, perm=perm, perm_rev=perm_rev, inv=jnp.argsort(perm),
                            inv_rev=jnp.argsort(perm_rev), Lam1_t=kept, flag=flag))
        return out

    @staticmethod
    def _kept(sb, perm):
        """Kernel-order positions of the dimensions kept in the map when the candidates are
        non-centred, and of the candidates best informed first (numpy or jax `perm`)."""
        xp = np if isinstance(perm, np.ndarray) else jnp
        mask = xp.asarray(sb.cand)[perm][::-1]
        D2 = int(sb.cand.sum())
        if xp is np:
            return np.nonzero(~mask)[0], np.nonzero(mask)[0][::-1]
        return jnp.nonzero(~mask, size=sb.D - D2)[0], jnp.nonzero(mask, size=D2)[0][::-1]

    def _split(self, sb, an):
        """For a super-block with candidates: kernel-order positions of the kept dimensions and
        the permutation of the kept dimensions (natural order) to the kernel's order with its
        inverse."""
        ks1 = self._kept(sb, an["perm"])[0]
        perm1 = jnp.take(jnp.asarray(sb.kept_index), jnp.take(an["perm_rev"], ks1))
        return ks1, perm1, jnp.argsort(perm1)

    def _kept_dense(self, an, Q1, s1, delta):
        """Batched factor L [n, D1, D1] of the kept dimensions' Lam + Q and y = L^-1 (s - Lam delta)."""
        Lam1 = jnp.transpose(_untile(an["Lam1_t"], s1.shape[1]), (2, 0, 1))
        r = s1.T if delta is None else s1.T - jnp.einsum("nab,nb->na", Lam1, delta.T)
        L1 = jnp.linalg.cholesky(Lam1 + Q1[None])
        return L1, solve_triangular(L1, r[..., None], lower=True)[..., 0]

    def _kept_solve(self, k, v, an, us, ans, post, ks1, perm1, inv1, native):
        """The kept dimensions mapped on their own anchors: u [D1, n] in their natural order, the
        log-determinant term and the prior factors A1, A1^-1."""
        sb = self.sbs[k]
        A1, Ainv1 = sb.A1(post).astype(v.dtype), sb.Ainv1(post).astype(v.dtype)
        Q1 = Ainv1.T @ Ainv1
        Q1 = jnp.take(jnp.take(Q1, perm1, axis=0), perm1, axis=1)
        s1 = jnp.take(an["s_k"], ks1, axis=0)
        delta = self._delta(k, v, an, us, ans, kept=perm1)
        zeta = jnp.take(v[sb.pos1], perm1, axis=0)
        if native:
            u_k, ld = _group_solve(an["Lam1_t"], Q1, s1, jnp.zeros((0,), zeta.dtype) if delta is None else delta, zeta)
        else:
            L1, y1 = self._kept_dense(an, Q1, s1, delta)
            u_k = solve_triangular(jnp.swapaxes(L1, 1, 2), (y1 + zeta.T)[..., None], lower=False)[..., 0].T
            ld = jnp.sum(jnp.log(jnp.diagonal(L1, axis1=1, axis2=2)), 1)
        return jnp.take(u_k, inv1, axis=0), jnp.sum(ld) + sb.n * jnp.sum(jnp.log(jnp.diag(A1))), A1, Ainv1

    def _cheap(self, k, v, an, us, ans, post, native):
        """(th, log-determinant term, u [D, n] in natural order) for super-block k with its
        candidate blocks non-centred (their standardised effects sampled as they are) and the
        kept dimensions mapped on their own anchors."""
        sb = self.sbs[k]
        ks1, perm1, inv1 = self._split(sb, an)
        th, ld = v, 0.0
        u = jnp.zeros((sb.D, sb.n), v.dtype)
        if ks1.shape[0]:
            u1, ld, A1, Ainv1 = self._kept_solve(k, v, an, us, ans, post, ks1, perm1, inv1, native)
            th = th.at[sb.pos1].set(Ainv1 @ u1)
            if sb.linked:
                u = u.at[sb.kept_nat].set(u1)
        if sb.linked:
            u = u.at[sb.cand_nat].set(sb.A2(post).astype(v.dtype) @ v[sb.pos2])
        return th, ld, u

    def _delta(self, k, v, an, us, ans, kept=None):
        """The shift of each group's effects implied by the fixed effects and earlier blocks,
        [D, n] in the kernel's order, or None when nothing couples. With `kept` (the kept
        dimensions' permutation to the kernel's order) the kept dimensions only, [D1, n]: the
        candidates have no anchors to shift."""
        sb = self.sbs[k]
        if kept is None:
            index, Dout, links, order = np.arange(sb.D), sb.D, sb.links, an["perm_rev"]
        else:
            index, Dout, links, order = sb.kept_index, len(sb.kept_nat), sb.links_kept, kept
        delta, coupled = jnp.zeros((Dout, sb.n), dtype=v.dtype), False
        if len(sb.bflat):
            vb = v[sb.bflat]
            db = (jnp.where(jnp.asarray(sb.bexp), jnp.exp(vb) - jnp.exp(an["bref"]), vb - an["bref"])
                  if sb.bexp.any() else vb - an["bref"])
            sel = np.nonzero(index[sb.bdims] >= 0)[0]
            if sel.size:
                delta, coupled = delta.at[index[sb.bdims[sel]]].add(db[sel][:, None]), True
            gsel = np.nonzero(index[sb.gdims] >= 0)[0] if len(sb.gdims) else np.zeros(0, int)
            if gsel.size:
                delta, coupled = delta.at[index[sb.gdims[gsel]]].add(jnp.asarray(sb.gW[gsel], dtype=v.dtype) @ db), True
        parts, rows = [], []
        for e, ed, cd, W in _link_groups(links):
            du = jnp.einsum("kmj,knj->kmn", (us[e] - ans[e]["uref"])[ed], jnp.asarray(W, dtype=v.dtype))
            parts.append(du.reshape(-1, sb.n))
            rows.append(index[cd.reshape(-1)])
        if parts:
            du = parts[0] if len(parts) == 1 else jnp.concatenate(parts, 0)
            delta, coupled = delta + jax.ops.segment_sum(du, jnp.asarray(np.concatenate(rows)), num_segments=Dout), True
        return jnp.take(delta, order, axis=0) if coupled else None

    def _linear(self, k, v, an, us, ans):
        sb = self.sbs[k]
        delta = self._delta(k, v, an, us, ans)
        if delta is None:
            return [an["s"][a] for a in range(sb.D)]
        delta = delta[::-1]
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
        return A, Ainv, jnp.take(jnp.take(Q, an["perm_rev"], axis=0), an["perm_rev"], axis=1)

    def _parts(self, k, v, post, ans, us):
        an = ans[k]
        A, Ainv, Q = self._prior(k, v, post, an)
        L = _chol(an["Lam"][::-1, ::-1], Q)
        return A, Ainv, L, _fsolve(L, self._linear(k, v, an, us, ans)[::-1])

    def _dense(self, k, v, post, ans, us):
        """_parts as batched linear algebra, for wide blocks: L [n, D, D] and y [n, D] in the
        kernel's order, with an operation count that does not grow with D."""
        sb, an = self.sbs[k], ans[k]
        A, Ainv, Q = self._prior(k, v, post, an)
        Lam = jnp.transpose(_untile(an["Lam_t"], sb.n), (2, 0, 1))
        r = an["s_k"].T
        delta = self._delta(k, v, an, us, ans)
        if delta is not None:
            r = r - jnp.einsum("nab,nb->na", Lam, delta.T)
        L = jnp.linalg.cholesky(Lam + Q[None])
        return A, Ainv, L, solve_triangular(L, r[..., None], lower=True)[..., 0]

    def forward(self, tht, aux):
        """Sampler coordinates -> the model's unconstrained coordinates, and log|dz/dzeta|^-1.
        `aux`: one chain's anchors as parts() gives them."""
        ans = self._unpack(aux, tht.dtype)
        post = self.post(tht)
        th, ld, us = tht, 0.0, {}
        native = _native()
        for k, sb in enumerate(self.sbs):
            an = ans[k]

            def full(th):
                if native and sb.D >= _TRI_MIN:
                    A = sb.A(post).astype(th.dtype)
                    delta = self._delta(k, th, an, us, ans)
                    thb, ldg, u = _block_map(an["Lam_t"], an["s_k"], jnp.zeros((0,), th.dtype) if delta is None else delta,
                                             sb.get(th), A, an["perm_rev"])
                    return sb.put(th, thb), jnp.sum(ldg) + sb.n * jnp.sum(jnp.log(jnp.diag(A))), u
                zeta = jnp.take(sb.get(th), an["perm_rev"], axis=0)
                A, Ainv, Q = self._prior(k, th, post, an)
                if native:
                    delta = self._delta(k, th, an, us, ans)
                    u_rev, ldg = _group_solve(an["Lam_t"], Q, an["s_k"], jnp.zeros((0,), zeta.dtype) if delta is None else delta, zeta)
                    ldg = jnp.sum(ldg)
                elif sb.D <= _UNROLL_MAX:
                    _, _, L, y = self._parts(k, th, post, ans, us)
                    u_rev = jnp.stack(_bsolve(L, [y[i] + zeta[i] for i in range(sb.D)]))
                    ldg = sum(jnp.sum(jnp.log(L[i][i])) for i in range(sb.D))
                else:
                    _, _, L, y = self._dense(k, th, post, ans, us)
                    u_rev = solve_triangular(jnp.swapaxes(L, 1, 2), (y + zeta.T)[..., None], lower=False)[..., 0].T
                    ldg = jnp.sum(jnp.log(jnp.diagonal(L, axis1=1, axis2=2)))
                u = jnp.take(u_rev, an["inv_rev"], axis=0)
                return sb.put(th, Ainv @ u), ldg + sb.n * jnp.sum(jnp.log(jnp.diag(A))), u

            if sb.cand is not None and self.static[k]:
                th, ldg, u = self._cheap(k, th, an, us, ans, post, native)
            else:
                th, ldg, u = full(th)
            us[k] = u
            ld = ld + ldg
        return th, ld

    def inverse(self, th, aux):
        ans = self._unpack(aux, th.dtype)
        post = self.post(th)
        tht, us = th, {}
        for k, sb in enumerate(self.sbs):
            an = ans[k]
            dense = sb.D > _UNROLL_MAX

            def full(tht):
                A, Ainv, Q = self._prior(k, th, post, an)
                u = A @ sb.get(th)
                ur = jnp.take(u, an["perm_rev"], axis=0)
                _, _, L, y = (self._dense if dense else self._parts)(k, th, post, ans, us)
                if dense:
                    zp = (jnp.einsum("nji,nj->ni", L, ur.T) - y).T
                else:
                    zp = jnp.stack([sum(L[j][i] * ur[j] for j in range(i, sb.D)) - y[i] for i in range(sb.D)])
                return sb.put(tht, jnp.take(zp, an["inv_rev"], axis=0)), u

            def cheap(tht):
                ks1, perm1, inv1 = self._split(sb, an)
                u = jnp.zeros((sb.D, sb.n), th.dtype)
                if ks1.shape[0]:
                    A1, Ainv1 = sb.A1(post).astype(th.dtype), sb.Ainv1(post).astype(th.dtype)
                    u1 = A1 @ th[sb.pos1]
                    Q1 = Ainv1.T @ Ainv1
                    Q1 = jnp.take(jnp.take(Q1, perm1, axis=0), perm1, axis=1)
                    L1, y1 = self._kept_dense(an, Q1, jnp.take(an["s_k"], ks1, axis=0),
                                              self._delta(k, th, an, us, ans, kept=perm1))
                    zk = (jnp.einsum("nji,nj->ni", L1, jnp.take(u1, perm1, axis=0).T) - y1).T
                    tht = tht.at[sb.pos1].set(jnp.take(zk, inv1, axis=0))
                    u = u.at[sb.kept_nat].set(u1)
                return tht, u.at[sb.cand_nat].set(sb.A2(post).astype(th.dtype) @ th[sb.pos2])

            tht, u = cheap(tht) if sb.cand is not None and self.static[k] else full(tht)
            us[k] = u
        return tht

    def wrap(self, pot_off):
        """pot_off(v, OFF) -> value_and_grad of the mapped potential, (v, OFF, AUX) -> (U, g)."""
        def pot(v, OFF, AUX):
            th, ld = self.forward(v, AUX)
            return pot_off(th, OFF) + ld
        return jax.value_and_grad(pot)

    def forward_chains(self, tht, aux):
        """Sampler -> model coordinates for positions [chains, D] with anchors [chains, A], in
        the configuration the anchors' flags name."""
        v = self.cheap_of(aux)
        if v._fwd is None:
            v._fwd = jax.jit(lambda x, a: v.forward(x, a)[0])
        P = self.parts(aux)
        return np.stack([np.asarray(v._fwd(jnp.asarray(x), tuple(p[c] for p in P)))
                         for c, x in enumerate(np.asarray(tht))])

    def inverse_chains(self, th, aux):
        v = self.cheap_of(aux)
        if v._inv is None:
            v._inv = jax.jit(v.inverse)
        P = self.parts(aux)
        return np.stack([np.asarray(v._inv(jnp.asarray(x), tuple(p[c] for p in P)))
                         for c, x in enumerate(np.asarray(th))])

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
        packed, kept = [[] for _ in range(C)], [[] for _ in range(C)]
        for sb, r in zip(self.sbs, raw):
            Lam = r["Lam"]
            s_ = -np.swapaxes(r["q0"], 1, 2) + np.einsum("cnab,cnb->cna", Lam, np.swapaxes(r["u"], 1, 2))
            strength = np.mean(np.einsum("cnaa->cna", Lam), axis=(0, 1)) * r["A2"].mean(0)
            weight = strength / (1.0 + strength)
            idx = np.arange(sb.D)
            # informed dims first in reversed natural order, then the rest by falling weight
            perm = np.lexsort((-idx, np.where(weight >= _UNINFORMED, -idx - 1.0, -weight)))
            ks1 = None if sb.cand is None else self._kept(sb, perm)[0]
            for c in range(C):
                Lp = Lam[c][:, perm][:, :, perm][:, ::-1, ::-1]
                packed[c] += [np.asarray(_tile(jnp.asarray(Lp))).reshape(-1), s_[c][:, perm].T[::-1].reshape(-1),
                              ref[c, sb.bflat], r["u0"][c].reshape(-1), perm.astype(np.float64)]
                if ks1 is not None:
                    kept[c].append(np.asarray(_tile(jnp.asarray(Lp[:, ks1][:, :, ks1]))).reshape(-1))
        return np.stack([np.concatenate(p + q + [np.zeros(len(self.sbs))]) for p, q in zip(packed, kept)])

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


def build(model, config, off, unflatten, split=False, links=True):
    """A TMap over every random-effect block it can represent, or None when there is none. `links` False cuts
    the links between super-blocks."""
    import drghmc as dr
    sbs = _supers(config, off, split)
    if not sbs:
        return None
    if not links:
        for sb in sbs:
            sb.links = []
    _, _, post_trim = dr._model_info(model, dr._trim_config(config))
    return TMap(sbs, lambda th: post_trim(unflatten(th)))
