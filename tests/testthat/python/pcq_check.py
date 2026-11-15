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
out = {"native": 0.0, "steep_rows": 0}
for n, rows in groups.items():
    a = jnp.asarray([r[0] for r in rows])
    c = jnp.asarray([r[1] for r in rows])
    ref = np.array([r[2] for r in rows])
    nat = [np.asarray(v) for v in nb._pcq_raw(a, c)]
    out["native"] = max(out["native"], float(np.max(np.abs(nat[0] - ref))))
    out["steep_rows"] += int(np.sum(np.max(np.asarray(a), -1) > 3))
rng = np.random.default_rng(4)
ab = jnp.asarray(rng.uniform(0.5, 2.0, (3, 50, 4)))
cb = jnp.asarray(rng.normal(0.5, 1.0, (3, 50, 4)))
vm = [np.asarray(v) for v in jax.vmap(nb._pcq_raw)(ab, cb)]
rows = [[np.asarray(v) for v in nb._pcq_raw(ab[i], cb[i])] for i in range(3)]
out["vmap_vs_rows"] = float(max(np.max(np.abs(vm[k][i] - rows[i][k])) for i in range(3) for k in range(3)))
out["ffi_loaded"] = nb._bvn_lib is not None
print("RESULT " + json.dumps(out))
