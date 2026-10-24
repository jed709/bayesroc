import csv
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

rows = list(csv.DictReader(open(sys.argv[3])))
K = 6
sets = {}
for r in rows:
    sets.setdefault((r["link"], int(r["rel"])), []).append(r)

worst_r = worst_fd = worst_f32 = 0.0
rng = np.random.default_rng(2)
for (lk, rel), rs in sets.items():
    f = nb._make_fused_metad_loglik(K, False, 0.5, bool(rel), lk)
    col = lambda k: jnp.asarray([float(r[k]) for r in rs])
    y = jnp.asarray([int(r["y"]) for r in rs], dtype=jnp.int32)
    a = col("a")
    args = [col("d"), col("mo"), col("mn"), col("c0"), jnp.stack([col(f"X{j}") for j in range(1, K - 1)], 1)]
    run = lambda d, mo, mn, c0, g: f(y, d, mo, mn, a, c0, g)
    lp = np.asarray(run(*args))
    ref = np.array([float(r["lp"]) for r in rs])
    worst_r = max(worst_r, float(np.max(np.abs(lp - ref) / np.maximum(1.0, np.abs(ref)))))
    tan = [jnp.asarray(rng.normal(size=v.shape)) for v in args]
    _, jv = jax.jvp(run, args, tan)
    h = 1e-6
    fd = (np.asarray(run(*[v + h * t for v, t in zip(args, tan)])) -
          np.asarray(run(*[v - h * t for v, t in zip(args, tan)]))) / (2 * h)
    worst_fd = max(worst_fd, float(np.max(np.abs(np.asarray(jv) - fd) / np.maximum(1.0, np.abs(fd)))))
    lp32 = np.asarray(f(y, *[v.astype(jnp.float32) for v in args[:3]], a.astype(jnp.float32),
                        *[v.astype(jnp.float32) for v in args[3:]]))
    ok = lp > np.log(1e-30)
    worst_f32 = max(worst_f32, float(np.max(np.abs(lp32 - lp)[ok] / np.maximum(1.0, np.abs(lp[ok])))))

print("RESULT " + json.dumps({"jax_vs_r": worst_r, "jvp_vs_fd": worst_fd, "f32_vs_f64": worst_f32}))
