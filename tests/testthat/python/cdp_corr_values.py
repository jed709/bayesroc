import json
import os
import sys

sys.path.insert(0, sys.argv[1])
if len(sys.argv) > 2 and sys.argv[2]:
    os.environ["BAYESROC_BVN_FFI"] = sys.argv[2]
else:
    os.environ.pop("BAYESROC_BVN_FFI", None)
    os.environ["BAYESROC_PURE_JAX_REFERENCE"] = "1"
os.environ["BAYESROC_ENABLE_X64"] = "1"
import numpy as np
import jax
import jax.numpy as jnp
import numpyro_backend as nb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jvp_check

out = {"ffi_loaded": bool(nb._BVN_FFI)}
for name, n_rkg, crit in (("rk", 2, (1.5, 0.4)), ("rkg", 3, (1.5, 0.4)), ("rkg_tail", 3, (4.0, 3.5))):
    f, args, _ = jvp_check.cdp_case(n_rkg, crit=crit, corr=os.environ.get("CDP_NO_CORR") != "1")
    val = np.asarray(jax.jit(f)(*args))
    J = 5
    n_gaps = J - 1
    mid_idx = (J - 1) // 2
    c = np.asarray(nb._batch_build_thresh(args[11], jnp.exp(args[12]), n_gaps, mid_idx, J - mid_idx - 1))
    old_idx = np.asarray(args[3])
    has_hi = old_idx < J - 1
    rows = np.arange(len(val))
    tau_lo = c[rows, np.clip(old_idx, 0, J - 1)]
    tau_hi = np.where(has_hi, c[rows, np.clip(old_idx + 1, 0, J - 1)], np.inf)
    out[name] = {"rk": np.asarray(args[1]).tolist(), "mu_R": np.asarray(args[4]).tolist(),
                 "sigma_R": np.asarray(args[5]).tolist(), "mu_F": np.asarray(args[6]).tolist(),
                 "sigma_F": np.asarray(args[7]).tolist(), "c_R": np.asarray(args[8]).tolist(),
                 "c_K": np.asarray(args[9]).tolist(), "corr": np.asarray(args[10]).tolist(),
                 "tau_lo": [str(v) for v in tau_lo], "tau_hi": [str(v) for v in tau_hi],
                 "lp": val.tolist()}
print("RESULT " + json.dumps(out))
