import json
import os
import subprocess
import sys

import numpy as np

if len(sys.argv) > 3 and sys.argv[3] == "f64":
    sys.path.insert(0, sys.argv[1])
    os.environ["BAYESROC_BVN_FFI"] = sys.argv[2]
    os.environ["BAYESROC_ENABLE_X64"] = "1"
else:
    sys.path.insert(0, sys.argv[1])
    os.environ["BAYESROC_BVN_FFI"] = sys.argv[2]
    os.environ["BAYESROC_ENABLE_X64"] = "0"
import jax.numpy as jnp
import numpyro_backend as nb

rng = np.random.default_rng(11)
n = 3000
z1l = rng.normal(-0.5, 1.5, n)
z1h = z1l + rng.uniform(0.05, 3.0, n)
z2l = rng.normal(-0.5, 1.5, n)
z2h = z2l + rng.uniform(0.05, 3.0, n)
rho = rng.uniform(-0.97, 0.97, n)
lo = rng.uniform(-9, 9, n)
hi = lo + rng.exponential(1.0, n)
inputs = [np.asarray(v, dtype=np.float32).astype(np.float64) for v in (z1l, z1h, z2l, z2h, rho, lo, hi)]
dt = jnp.float64 if os.environ["BAYESROC_ENABLE_X64"] == "1" else jnp.float32
z1l, z1h, z2l, z2h, rho, lo, hi = (jnp.asarray(v, dtype=dt) for v in inputs)

res = {
    "rect": nb._bvn_rect(z1l, z1h, z2l, z2h, rho),
    "cdf": nb._bvn_cdf(z1h, z2h, rho),
    "cdf2": nb._bvn_cdf_multi([(z1h, z2h), (z1l, z2l)], [z1l], rho)[0][0],
    "cdf4": nb._bvn_cdf_multi([(z1h, z2h), (z1l, z2l), (z1h, z2l), (z1l, z2h)], [z1l, z2h], rho)[0][3],
    "probit_diff": nb._phi_diff(lo, hi),
}
flags = [jnp.asarray((np.arange(n) + s) % 4 != 0, dtype=dt) for s in range(4)]
f32r = lambda v: jnp.asarray(np.asarray(v, dtype=np.float32).astype(np.float64), dtype=dt)
res["vrdp2d"] = nb._vrdp2d_native(z1h, z1l, z2h, z2l, *flags, f32r(inputs[5] * 0.2 + 1.0),
                                  f32r(inputs[6] * 0.1 + 1.2), f32r(inputs[4] + 0.5),
                                  f32r(np.abs(inputs[4]) * 0.5 + 0.05), f32r(np.abs(inputs[0]) * 0.1 + 0.8),
                                  f32r(np.abs(inputs[2]) * 0.1 + 0.8))
res["metad"] = nb._metad_cell(lo, z1l, z1h, f32r(inputs[0] - np.abs(inputs[4])), flags[0] > 0)
pc = f32r(np.stack([inputs[0], inputs[2] + 1.0, inputs[5] * 0.3], 1))
res["pcq"] = nb._pcq_raw(jnp.ones_like(pc), pc)
out = {k: [np.asarray(a, dtype=np.float64).tolist() for a in v] for k, v in res.items()}

if len(sys.argv) > 3 and sys.argv[3] == "f64":
    print("F64 " + json.dumps(out))
    sys.exit(0)

ref = subprocess.run([sys.executable, __file__, sys.argv[1], sys.argv[2], "f64"], capture_output=True, text=True)
line = [l for l in ref.stdout.splitlines() if l.startswith("F64 ")]
ref = json.loads(line[0][4:])
err = {}
for k in out:
    worst = 0.0
    for a, b in zip(out[k], ref[k]):
        a, b = np.asarray(a), np.asarray(b)
        worst = max(worst, float(np.max(np.abs(a - b) / np.maximum(1e-3, np.abs(b)))))
    err[k] = worst
print("RESULT " + json.dumps({"ffi_loaded": bool(nb._BVN_FFI), "f32": dt == jnp.float32, "rel_err": err}))
