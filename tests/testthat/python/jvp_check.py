import json
import os
import sys

os.environ.pop("BAYESROC_BVN_FFI", None)

import numpy as np

sys.path.insert(0, sys.argv[1])
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
import numpyro_backend as nb



def check(f, args, cont_idx, seed=0):
    """Custom-JVP primal vs the plain value, and custom-JVP tangent vs autodiff of the
    undecorated value function (f.fun). Run without BAYESROC_BVN_FFI so f.fun is pure JAX."""
    rng = np.random.default_rng(seed)
    plain = np.asarray(f(*args))
    if not hasattr(f, "fun"):
        return {"custom_jvp": False, "primal_abs": float("nan"), "tangent_abs": float("nan"),
                "tangent_rel": float("nan"), "finite": bool(np.all(np.isfinite(plain)))}
    worst_primal = 0.0
    worst_tan = 0.0
    worst_abs = 0.0
    for i in cont_idx:
        v = jnp.asarray(rng.normal(size=np.shape(args[i])))
        tangents = [jnp.zeros_like(a) if jnp.issubdtype(a.dtype, jnp.floating)
                    else np.zeros(np.shape(a), dtype=jax.dtypes.float0) for a in args]
        tangents[i] = v
        primal, tan = jax.jit(lambda a, t: jax.jvp(f, a, t))(tuple(args), tuple(tangents))
        worst_primal = max(worst_primal, float(np.max(np.abs(np.asarray(primal) - plain))))
        ref = np.asarray(jax.jit(lambda a, t: jax.jvp(f.fun, a, t))(tuple(args), tuple(tangents))[1])
        err = np.abs(np.asarray(tan) - ref)
        worst_abs = max(worst_abs, float(np.max(err)))
        worst_tan = max(worst_tan, float(np.max(err / np.maximum(np.abs(ref), 1.0))))
    return {"custom_jvp": True, "primal_abs": worst_primal, "tangent_abs": worst_abs, "tangent_rel": worst_tan,
            "finite": bool(np.all(np.isfinite(plain)))}


def cdp_case(n_rkg, N=300, J=5, seed=1):
    rng = np.random.default_rng(seed)
    n_gaps = J - 1
    f = nb._make_fused_batch_cdp_loglik(J + 5, J, False, n_rkg=n_rkg)
    is_old_resp = rng.integers(0, 2, N).astype(np.float64)
    args = [
        jnp.asarray(rng.integers(1, J + 6, N), dtype=jnp.int32),
        jnp.asarray(rng.integers(1, n_rkg + 1, N), dtype=jnp.int32),
        jnp.asarray(rng.integers(0, 2, N).astype(np.float64)),
        jnp.asarray(rng.integers(0, J, N), dtype=jnp.int32),
        jnp.asarray(is_old_resp),
        jnp.asarray(rng.normal(0.8, 0.3, N)),
        jnp.asarray(rng.uniform(0.6, 1.4, N)),
        jnp.asarray(rng.normal(0.6, 0.3, N)),
        jnp.asarray(rng.uniform(0.6, 1.4, N)),
        jnp.asarray(rng.normal(1.5, 0.3, N)),
        jnp.asarray(rng.normal(0.4, 0.3, N)),
        jnp.asarray(rng.normal(0.3, 0.2, N)),
        jnp.asarray(rng.normal(-0.5, 0.2, (N, n_gaps))),
    ]
    return f, args, list(range(5, 13))


def _u(rng, lo, hi, N):
    return jnp.asarray(rng.uniform(lo, hi, N))


def _n(rng, m, sd, shape):
    return jnp.asarray(rng.normal(m, sd, shape))


def uni_case(kind, link, N=400, K=6, seed=2):
    rng = np.random.default_rng(seed)
    ng = K - 2
    y = jnp.asarray(rng.integers(1, K + 1, N), dtype=jnp.int32)
    tail = [_n(rng, 0.0, 0.2, N), _n(rng, -0.4, 0.2, (N, ng))]
    if kind == "simple":
        f = nb._make_fused_batch_loglik(K, False, link)
        args = [y, _n(rng, 1.0, 1.0, N), _u(rng, 0.7, 1.5, N)] + tail
    elif kind == "dpsdt":
        f = nb._make_fused_batch_dpsdt_loglik(K, False, link)
        args = [y, _n(rng, 1.0, 1.0, N), _u(rng, 0.7, 1.5, N), _u(rng, 0.05, 0.6, N)] + tail
    elif kind == "mixture":
        f = nb._make_fused_batch_mixture_loglik(K, False, link)
        args = [y, _n(rng, 1.5, 0.8, N), _u(rng, 0.7, 1.5, N), _n(rng, 0.3, 0.5, N),
                _u(rng, 0.7, 1.5, N), _u(rng, 0.05, 0.9, N)] + tail
    elif kind == "source_mixture":
        f = nb._make_fused_batch_source_mixture_loglik(K, False, link)
        args = [y, _n(rng, 1.0, 1.0, N), _u(rng, 0.05, 0.9, N)] + tail
    return f, args, list(range(1, len(args)))


def biv_case(kind, N=400, K=6, seed=3):
    rng = np.random.default_rng(seed)
    ng = K - 2
    ints = [jnp.asarray(rng.integers(1, K + 1, N), dtype=jnp.int32),
            jnp.asarray(rng.integers(1, K + 1, N), dtype=jnp.int32),
            jnp.asarray(rng.integers(1, 4, N), dtype=jnp.int32)]
    th = [_n(rng, 0.0, 0.2, N), _n(rng, -0.4, 0.2, (N, ng)), _n(rng, 0.0, 0.2, N), _n(rng, -0.4, 0.2, (N, ng))]
    if kind == "vrdp2d":
        f = nb._make_fused_batch_vrdp2d_loglik(K, K, False)
        cont = [_n(rng, 1.0, 0.5, N), _n(rng, 1.0, 0.5, N), _n(rng, 0.8, 0.5, N),
                _u(rng, 0.05, 0.6, N), _u(rng, 0.7, 1.5, N), _u(rng, 0.7, 1.5, N)]
    else:
        base = [_n(rng, 1.0, 0.7, N), _n(rng, 0.5, 0.7, N), _u(rng, 0.7, 1.5, N),
                _u(rng, 0.7, 1.5, N), _u(rng, -0.7, 0.7, N)]
        if kind in ("bivariate_dp", "bounded_bivariate_dp"):
            base += [_u(rng, -0.7, 0.7, N), _u(rng, 0.05, 0.6, N), _u(rng, 0.05, 0.9, N),
                     _u(rng, 0.05, 0.6, N), _u(rng, 0.05, 0.9, N)]
        f = {"bivariate_sdt": nb._make_fused_batch_bivariate_sdt_loglik,
             "bounded_bivariate_sdt": nb._make_fused_batch_bounded_bivariate_sdt_loglik,
             "bivariate_dp": nb._make_fused_batch_bivariate_dp_loglik,
             "bounded_bivariate_dp": nb._make_fused_batch_bounded_bivariate_dp_loglik}[kind](K, K, False)
        cont = base
    args = ints + cont + th
    return f, args, list(range(3, len(args)))


CASES = {
    "cdp_rk": lambda: cdp_case(2),
    "cdp_rkg": lambda: cdp_case(3),
}
for _k in ("simple", "dpsdt", "mixture", "source_mixture"):
    for _l in ("probit", "logit", "cloglog"):
        CASES[f"{_k}_{_l}"] = (lambda k=_k, l=_l: uni_case(k, l))
for _k in ("bivariate_sdt", "bounded_bivariate_sdt", "bivariate_dp", "bounded_bivariate_dp", "vrdp2d"):
    CASES[_k] = (lambda k=_k: biv_case(k))

if __name__ == "__main__":
    names = sys.argv[2].split(",") if len(sys.argv) > 2 else list(CASES)
    out = {}
    for name in names:
        f, args, cont = CASES[name]()
        out[name] = check(f, args, cont)
        r = out[name]
        print(f"{name:<24} primal {r['primal_abs']:.1e}  tan_abs {r['tangent_abs']:.1e}  tan_rel {r['tangent_rel']:.1e}  finite {r['finite']}", file=sys.stderr)
    print("RESULT " + json.dumps(out))
