import json
import os
import sys

sys.path.insert(0, sys.argv[1])
os.environ["BAYESROC_BVN_FFI"] = sys.argv[2]
os.environ["BAYESROC_ENABLE_X64"] = "1"
import numpy as np
import jax
import jax.numpy as jnp
import numpyro_backend as nb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jvp_check


def fd_check(f, args, cont_idx, seed=0, steps=(1e-4, 1e-5, 1e-6)):
    """Custom-JVP tangent vs central differences of the kernel's own value, with the native kernels
    loaded: each row scored at the step that agrees best (one step's truncation or round-off can spoil
    it at a sharply curved row; a wrong tangent agrees at none)."""
    rng = np.random.default_rng(seed)
    fj = jax.jit(f)
    jv = jax.jit(lambda a, t: jax.jvp(f, a, t))
    worst = 0.0
    for i in cont_idx:
        v = jnp.asarray(rng.normal(size=np.shape(args[i])))
        tangents = [jnp.zeros_like(a) if jnp.issubdtype(a.dtype, jnp.floating)
                    else np.zeros(np.shape(a), dtype=jax.dtypes.float0) for a in args]
        tangents[i] = v
        _, tan = jv(tuple(args), tuple(tangents))
        errs = []
        for h in steps:
            up = list(args)
            dn = list(args)
            up[i] = args[i] + h * v
            dn[i] = args[i] - h * v
            fd = (np.asarray(fj(*up)) - np.asarray(fj(*dn))) / (2 * h)
            errs.append(np.abs(np.asarray(tan) - fd) / np.maximum(np.abs(fd), 1.0))
        worst = max(worst, float(np.max(np.min(np.stack(errs), axis=0))))
    return worst


names = sys.argv[3].split(",")
out = {"ffi_loaded": nb._bvn_lib is not None}
for name in names:
    f, args, cont = jvp_check.CASES[name]()
    out[name] = fd_check(f, args, cont)
    print(f"{name:<24} fd_rel {out[name]:.1e}", file=sys.stderr)
print("RESULT " + json.dumps(out))
