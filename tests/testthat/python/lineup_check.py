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
cols = [np.array([float(r[k]) for r in rows]) for k in ("kind", "k", "m", "s", "lo", "hi", "sb", "lo_open", "hi_open")]
ref = np.array([float(r["log_prob"]) for r in rows])
nat = [np.asarray(v) for v in nb._lineup_raw(*[jnp.asarray(c) for c in cols])]
fixture_rel = float(np.max(np.abs(nat[0] - ref) / np.maximum(1.0, np.abs(ref))))
pj = [np.asarray(v) for v in nb._lineup_cell_jax(*[jnp.asarray(c) for c in cols[:7]], jnp.asarray(cols[7] > 0.5),
                                                 jnp.asarray(cols[8] > 0.5))]
jax_vs_native = float(max(np.max(np.abs(a - b) / np.maximum(1.0, np.abs(b))) for a, b in zip(pj, nat)))

rng = np.random.default_rng(3)
K = 6
N = 400
y = rng.integers(1, K + 1, N)
kind = np.where(y <= 2, 2, rng.integers(0, 2, N))
slot = rng.integers(0, 3, N)
size = rng.choice([1, 3, 6, 9], N)
kind = np.where(slot == 0, 2, kind)
kind = np.where((size == 1) & (kind == 1), 0, kind)
args = [jnp.asarray(rng.normal(1.3, 0.4, N)), jnp.asarray(np.exp(rng.normal(0.1, 0.2, N))),
        jnp.asarray(rng.normal(0.2, 0.3, N)), jnp.asarray(np.exp(rng.normal(-0.1, 0.2, N))),
        jnp.asarray(np.exp(rng.normal(-1.0, 0.4, N))), jnp.asarray(rng.normal(0.6, 0.3, N)),
        jnp.asarray(rng.normal(-0.6, 0.3, (N, K - 2)))]
f = nb._make_fused_lineup_loglik(K, False)
yy, kk, ss, zz = (jnp.asarray(v, dtype=jnp.int32) for v in (y, kind, slot, size))
run = lambda *a: f(yy, kk, ss, zz.astype(jnp.float64), *a)
tan = [jnp.asarray(rng.normal(size=a.shape)) for a in args]
_, jv = jax.jvp(run, args, tan)
h = 1e-6
fd = (np.asarray(run(*[a + h * t for a, t in zip(args, tan)])) - np.asarray(run(*[a - h * t for a, t in zip(args, tan)]))) / (2 * h)
jvp_vs_fd = float(np.max(np.abs(np.asarray(jv) - fd) / np.maximum(1.0, np.abs(fd))))

B = [jnp.stack([c, c * 1.0]) for c in (jnp.asarray(cols[j]) for j in range(9))]
vm = [np.asarray(v) for v in jax.vmap(nb._lineup_raw)(*B)]
vmap_vs_rows = float(max(np.max(np.abs(vm[j][i] - nat[j])) for i in range(2) for j in range(6)))

c32 = [np.asarray(c, dtype=np.float32) for c in cols]
lp32 = np.asarray(nb._lineup_raw(*[jnp.asarray(c) for c in c32])[0])
lp64 = np.asarray(nb._lineup_raw(*[jnp.asarray(c.astype(np.float64)) for c in c32])[0])
ok = lp64 > np.log(1e-30)
f32_rel = float(np.max(np.abs(lp32[ok] - lp64[ok]) / np.maximum(1.0, np.abs(lp64[ok]))))

print("RESULT " + json.dumps({"ffi_loaded": bool(nb._BVN_FFI), "fixture_rel": fixture_rel, "jax_vs_native": jax_vs_native,
                              "jvp_vs_fd": jvp_vs_fd, "vmap_vs_rows": vmap_vs_rows, "f32_rel": f32_rel}))
