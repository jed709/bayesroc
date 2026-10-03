"""Exact population-mean moves for random-effect blocks.

When a random-effect term's design column equals a fixed-effect column on every row (e.g. the
intercept, or a slope on a predictor that also has a fixed effect), raising that coefficient by
Delta and lowering every group's effect by Delta leaves every linear predictor, and so the
likelihood and its gradient, unchanged. With many observations per group the posterior is a
narrow ridge along exactly this direction, which short-trajectory samplers cross slowly. The move
draws Delta from the Gaussian conditional implied by the groups' standardised effects (u = A z,
z ~ N(0, I)) and accepts on the ratio of the coefficients' own priors, which makes it exact for
any prior; the potential and gradient are carried through by re-evaluating only the prior."""
import re
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
from numpyro.distributions.transforms import CorrCholeskyTransform


class _Block:
    """Coordinates and geometry of one random-effect block in the flat unconstrained vector."""

    def __init__(self, conf, param, off, cross_members=None):
        g = conf["group"]
        self.n = int(conf["n_groups"])
        cid = conf.get("cor_id")
        if cid is not None:
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
            self.member_priors = []
            for m in members:
                per = m.get("prior_sd_per_coef")
                per = [per] if isinstance(per, str) else per
                nt = int(m["n_terms"])
                if per is None or len(per) != nt:
                    per = [m.get("prior_sd", "normal(0, 0.5)")] * nt
                self.member_priors += list(per)
        else:
            nt = int(conf["n_terms"])
            self.dim, self.dim_offset = nt, 0
            self.prefix = f"{param}_{g}"
            correlated = conf.get("correlated", True) and nt > 1
            per = conf.get("prior_sd_per_coef")
            per = [per] if isinstance(per, str) else per
            if per is None or len(per) != nt:
                per = [conf.get("prior_sd", "normal(0, 0.5)")] * nt
            self.member_priors = list(per)
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
        for s, _ in self.z_sites:
            sl = off[s][0]
            zi.append(np.arange(sl.start, sl.stop))
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


def _accessor(idx):
    shape = idx.shape
    flat = idx.reshape(-1)
    if np.array_equal(flat, np.arange(flat[0], flat[0] + flat.size)):
        a, b = int(flat[0]), int(flat[0]) + flat.size
        return ((lambda v: v[a:b].reshape(shape)),
                (lambda v, x: jax.lax.dynamic_update_slice(v, x.reshape(-1), (a,))))
    j = jnp.asarray(flat)
    return (lambda v: v[j].reshape(shape)), (lambda v, x: v.at[j].set(x.reshape(-1)))


def _neg_lp(x, pri):
    return -jnp.stack([pri[j].log_prob(jnp.exp(x[j])) + x[j] for j in range(len(pri))])


def _neg_lp_grad(x, pri):
    return jax.grad(lambda v: jnp.sum(_neg_lp(v, pri)))(x)


def _jump(key, x0, pri, Pd, M, n_newton, max_step):
    """One exact MH jump of the log-SD shifts a along a ridge scaling. The target is
    F(a) = sum_j log p_j(e^(x0_j + a_j)) + a_j - c.Pd - c'Mc/2 with c = e^a - 1; the proposal is Gaussian at
    the Newton mode of F, and the reverse proposal is built the same way from the proposed point.
    Returns the accepted shift (zero if rejected) and F(shift) - F(0)."""
    n = len(pri)
    lp = [lambda y, d=d: d.log_prob(jnp.exp(y)) + y for d in pri]
    lp1 = [jax.grad(f) for f in lp]
    lp2 = [jax.grad(f) for f in lp1]

    def F(a):
        c = jnp.exp(a) - 1.0
        return -jnp.sum(_neg_lp(x0 + a, pri)) - c @ Pd - 0.5 * c @ M @ c

    def grad_hess(a):
        ea = jnp.exp(a)
        y = x0 + a
        r = ea * (Pd + M @ (ea - 1.0))
        gr = jnp.stack([lp1[j](y[j]) for j in range(n)]) - r
        H = jnp.diag(jnp.stack([lp2[j](y[j]) for j in range(n)]) - r) - jnp.outer(ea, ea) * M
        return gr, H

    def laplace(a):
        for _ in range(n_newton):
            gr, H = grad_hess(a)
            a = a + jnp.clip(jnp.linalg.solve(-H, gr), -max_step, max_step)
        return a, jnp.linalg.cholesky(-grad_hess(a)[1])

    def log_q(b, m, Lh):
        return jnp.sum(jnp.log(jnp.diag(Lh))) - 0.5 * jnp.sum(jnp.square(Lh.T @ (b - m)))

    k_p, k_u = jax.random.split(key)
    zero = jnp.zeros(n, dtype=x0.dtype)
    m_f, L_f = laplace(zero)
    a_s = m_f + solve_triangular(L_f.T, jax.random.normal(k_p, (n,), dtype=x0.dtype), lower=False)
    m_r, L_r = laplace(a_s)
    dF = F(a_s) - F(zero)
    log_acc = dF + log_q(zero, m_r, L_r) - log_q(a_s, m_f, L_f)
    acc = jnp.isfinite(log_acc) & (jnp.log(jax.random.uniform(k_u, dtype=x0.dtype)) < log_acc)
    return jnp.where(acc, a_s, zero), jnp.where(acc, dF, 0.0)




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


def _fixed_map(config, conf, param, off, n_thresh, mid, gap_prefix):
    """[(dim_within_conf, flat beta index)] for the conf's terms that equal a fixed-effect column."""
    N = int(config["N"])
    Zc = _re_columns(conf, N)
    R = Zc.shape[1]
    out = []
    if param in ("criterion",):
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
                    flat = _beta_site(off, None, [(f"beta_thresh_mid[{j + 1}]", None)])
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
        if flat is not None:
            out.append((r, flat))
    return out


def build(model, config, keys, off, unflatten):
    """Return move(key, th, U, g) -> (th, U, g) applying one exact population-mean move per
    random-effect block with matched fixed effects, or None when no block qualifies."""
    import drghmc as dr
    from numpyro.handlers import block as hblock

    res = config.get("random_effects") or []
    if not res:
        return None
    n_thresh = int(config.get("n_thresh", int(config["K"]) - 1))
    mid = (n_thresh + 1) // 2 - 1
    gap_prefix = "beta_raw_gaps" if config.get("gap_link") == "softplus" else "beta_log_gaps"
    cross = {}
    for r in res:
        if r.get("cor_id") is not None:
            cross.setdefault(r["cor_id"], []).append(r)

    plans, seen = [], set()
    for conf in res:
        param = conf.get("param")
        if param not in ("dprime", "sigma", "lambda", "dprime2", "sigma2", "criterion", "discrim", "rho"):
            continue
        cid = conf.get("cor_id")
        key = ("cross", cid, conf["group"]) if cid is not None else ("re", param, conf["group"])
        if key in seen:
            continue
        seen.add(key)
        members = cross[cid] if cid is not None else [conf]
        try:
            B = _Block(conf, param, off, cross)
        except KeyError:
            continue
        pairs, dim_off = [], 0
        for mconf in members:
            mp = mconf.get("param")
            nt = int(mconf["n_terms"])
            if mp in ("dprime", "sigma", "lambda", "dprime2", "sigma2", "criterion", "discrim", "rho"):
                pairs += [(dim_off + d, flat) for d, flat in
                          _fixed_map(config, mconf, mp, off, n_thresh, mid, gap_prefix)]
            dim_off += nt
        if dim_off != B.dim:
            continue
        flats = [f for _, f in pairs]
        if len(set(flats)) != len(flats):
            pairs, flats = [], []
        plans.append(dict(B=B, dims=np.array([d for d, _ in pairs], dtype=int), betas=np.array(flats, dtype=int),
                          name=B.prefix, cov=False))
    if not plans:
        return None

    _, _, post_trim = dr._model_info(model, dr._trim_config(config))
    _, pot_prior, _ = dr._model_info(lambda c: hblock(model, hide=["obs"])(c), config)
    vg_prior = jax.value_and_grad(lambda v: pot_prior(unflatten(v)))
    cct = CorrCholeskyTransform()
    probe = jnp.asarray(np.random.default_rng(7).normal(0, 0.3, max(v[0].stop for v in off.values())))

    for p in plans:
        B = p["B"]
        Lk = f"L_corr_{B.prefix}"
        if B.dim < 2 or B.L_sites != [Lk] or B.n <= 2 * B.dim + 1:
            continue
        p["L_idx"] = np.arange(off[Lk][0].start, off[Lk][0].stop)
        x = jnp.concatenate([probe[jnp.asarray(B.sd_idx)], probe[jnp.asarray(p["L_idx"])]])
        A_model = B.A(post_trim(unflatten(probe)))
        A_x = jnp.exp(x[:B.dim])[:, None] * cct(x[B.dim:])
        if bool(jnp.max(jnp.abs(A_model - A_x)) < 1e-10):
            p["cov"] = True
    plans = [p for p in plans if len(p["dims"]) or p["cov"]]
    if not plans:
        return None

    def A_x_of(x, d):
        return jnp.exp(x[:d])[:, None] * cct(x[d:])

    def vech(M):
        return M[jnp.triu_indices(M.shape[0])]

    def log_q(x, d, nu, S):
        Sig = A_x_of(x, d) @ A_x_of(x, d).T
        J = jax.jacfwd(lambda xx: vech(A_x_of(xx, d) @ A_x_of(xx, d).T))(x)
        return (-0.5 * (nu + d + 1) * jnp.linalg.slogdet(Sig)[1] - 0.5 * jnp.trace(jnp.linalg.solve(Sig, S))
                + jnp.linalg.slogdet(J)[1])

    def sample_iw(key, nu, S, d):
        Vc = jnp.linalg.cholesky(jnp.linalg.inv(S))
        k1, k2 = jax.random.split(key)
        chi = jnp.sqrt(jax.random.gamma(k1, 0.5 * (nu - jnp.arange(d)), (d,), dtype=S.dtype) * 2.0)
        Bm = jnp.tril(jax.random.normal(k2, (d, d), dtype=S.dtype), -1) + jnp.diag(chi)
        return jnp.linalg.inv(Vc @ Bm @ Bm.T @ Vc.T)

    def mean_step(key, th, Up, gp, g_lik, plan):
        B, dims, betas = plan["B"], plan["dims"], plan["betas"]

        def A_of(t):
            return B.A(post_trim(unflatten(t)))

        A, A_vjp = jax.vjp(A_of, th)
        zi = jnp.asarray(B.z_idx)
        z = th[zi]
        E = jnp.eye(B.dim, dtype=th.dtype)[:, dims]
        W = solve_triangular(A, E, lower=True)
        G = z.shape[1]
        Lp = jnp.linalg.cholesky(G * (W.T @ W))
        mu = jax.scipy.linalg.cho_solve((Lp, True), W.T @ z.sum(1))
        kp, ku = jax.random.split(key)
        delta = mu + solve_triangular(Lp.T, jax.random.normal(kp, (len(dims),), dtype=th.dtype), lower=False)
        shift = W @ delta
        z1 = z - shift[:, None]
        th1 = th.at[jnp.asarray(betas)].add(delta).at[zi.reshape(-1)].set(z1.reshape(-1))
        Up1, gp1 = vg_prior(th1)
        dz = 0.5 * (jnp.sum(z1 * z1) - jnp.sum(z * z))
        acc = jnp.log(jax.random.uniform(ku, dtype=th.dtype)) < -((Up1 - Up) - dz)
        s_u = solve_triangular(A.T, g_lik[zi].sum(1), lower=False)
        g_lik1 = g_lik + A_vjp(-jnp.outer(s_u, shift))[0]
        return (jnp.where(acc, th1, th), jnp.where(acc, Up1, Up), jnp.where(acc, gp1, gp),
                jnp.where(acc, g_lik1, g_lik))

    def cov_step(key, th, Up, gp, g_lik, plan):
        B = plan["B"]
        d, n = B.dim, B.n
        sdi, Li = jnp.asarray(B.sd_idx), jnp.asarray(plan["L_idx"])
        zi = jnp.asarray(B.z_idx)
        x0 = jnp.concatenate([th[sdi], th[Li]])
        A0 = A_x_of(x0, d)
        z0 = th[zi]
        u = A0 @ z0
        S = u @ u.T
        nu = n - d - 1
        kp, ku = jax.random.split(key)
        Sig = sample_iw(kp, nu, S, d)
        sd = jnp.sqrt(jnp.diag(Sig))
        Lc = jnp.linalg.cholesky(Sig / jnp.outer(sd, sd))
        x1 = jnp.concatenate([jnp.log(sd), cct.inv(Lc)])
        A1 = A_x_of(x1, d)
        z1 = solve_triangular(A1, u, lower=True)
        th1 = th.at[sdi].set(x1[:d]).at[Li].set(x1[d:]).at[zi.reshape(-1)].set(z1.reshape(-1))
        Up1, gp1 = vg_prior(th1)
        log_t0 = -Up - n * jnp.sum(jnp.log(jnp.diag(A0)))
        log_t1 = -Up1 - n * jnp.sum(jnp.log(jnp.diag(A1)))
        log_acc = (log_t1 - log_q(x1, d, nu, S)) - (log_t0 - log_q(x0, d, nu, S))
        acc = jnp.isfinite(log_acc) & (jnp.log(jax.random.uniform(ku, dtype=th.dtype)) < log_acc)
        g_u = solve_triangular(A0.T, g_lik[zi], lower=False)
        _, vjp1 = jax.vjp(lambda xx: A_x_of(xx, d), x1)
        gx1 = vjp1(g_u @ z1.T)[0]
        g_lik1 = g_lik.at[zi.reshape(-1)].set((A1.T @ g_u).reshape(-1)).at[sdi].set(gx1[:d]).at[Li].set(gx1[d:])
        return (jnp.where(acc, th1, th), jnp.where(acc, Up1, Up), jnp.where(acc, gp1, gp),
                jnp.where(acc, g_lik1, g_lik))

    def move(key, th, U, g):
        Up, gp = vg_prior(th)
        U_lik, g_lik = U - Up, g - gp
        for kp, plan in zip(jax.random.split(key, len(plans)), plans):
            km, kc = jax.random.split(kp)
            if len(plan["dims"]):
                th, Up, gp, g_lik = mean_step(km, th, Up, gp, g_lik, plan)
            if plan["cov"]:
                th, Up, gp, g_lik = cov_step(kc, th, Up, gp, g_lik, plan)
        return th, U_lik + Up, g_lik + gp

    move.specs = [f"{p['name']} (mean {len(p['dims'])}{', covariance' if p['cov'] else ''})" for p in plans]
    move.plans = plans
    return move
