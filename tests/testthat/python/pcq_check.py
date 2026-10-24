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

groups = {}
for line in open(sys.argv[3]):
    n, a, c, r = line.strip().split(";")
    groups.setdefault(int(n), []).append(([float(x) for x in a.split(",")], [float(x) for x in c.split(",")], float(r)))
ref_jax = jax.jit(nb._pcq_jax)
out = {"native": 0.0, "jax": 0.0, "native_vs_jax_grad": 0.0, "steep_rows": 0}
for n, rows in groups.items():
    a = jnp.asarray([r[0] for r in rows])
    c = jnp.asarray([r[1] for r in rows])
    ref = np.array([r[2] for r in rows])
    nat = [np.asarray(v) for v in nb._pcq_raw(a, c)]
    jx = [np.asarray(v) for v in ref_jax(a, c)]
    out["native"] = max(out["native"], float(np.max(np.abs(nat[0] - ref))))
    out["jax"] = max(out["jax"], float(np.max(np.abs(jx[0] - ref))))
    out["native_vs_jax_grad"] = max(out["native_vs_jax_grad"],
                                    float(max(np.max(np.abs(nat[k] - jx[k])) for k in (1, 2))))
    out["steep_rows"] += int(np.sum(np.max(np.asarray(a), -1) > 3))
out["ffi_loaded"] = bool(nb._BVN_FFI)
print("RESULT " + json.dumps(out))
