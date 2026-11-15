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
import tmap

rows = list(csv.DictReader(open(sys.argv[3])))
worst_fx = 0.0
sets = {}
for r in rows:
    sets.setdefault(r["set_id"], []).append(r)
for sid, rs in sets.items():
    r0 = rs[0]
    K = 2 * int(r0["K"])
    d = float(r0["d"])
    tau = np.array([float(r0[f"tau{j}"]) for j in range(1, K)]) + d / 2
    mid = (K - 2) // 2
    up = np.diff(tau[mid:])
    dn = np.diff(tau[:mid + 1])[::-1]
    gaps = np.log(np.concatenate([up, dn]))
    n = len(rs)
    f = nb._make_fused_metad_loglik(K, False, 0.5, r0["anchor"] == "relative")
    ll = f(jnp.asarray([int(r["cell"]) for r in rs], dtype=jnp.int32), jnp.full(n, d),
           jnp.full(n, float(r0["M2"])), jnp.full(n, float(r0["M1"])),
           jnp.asarray([float(r["stimulus"]) for r in rs]), jnp.full(n, tau[mid]),
           jnp.asarray(np.tile(gaps, (n, 1))))
    ref = np.array([float(r["log_prob"]) for r in rs])
    worst_fx = max(worst_fx, float(np.max(np.abs(np.asarray(ll) - ref) / np.maximum(1.0, np.abs(ref)))))

y = np.array([1, 2, 3, 4, 5, 6, 6, 1])
rows_ok = True
for shared in (False, True):
    cfg = {"family": "metadt", "y": y.tolist(), "K": 6, "metad_shared": shared, "is_old": [0, 1] * 4}
    for prm, want in (("dprime", None), ("criterion", None), ("mratio", None if shared else y > 3),
                      ("mratio_N", None if shared else y <= 3)):
        got = tmap._param_rows(cfg, prm, len(y))
        rows_ok &= (got is None) if want is None else (got is not None and bool(np.array_equal(got, want)))
print("RESULT " + json.dumps({"ffi_loaded": nb._bvn_lib is not None, "fixture_rel": worst_fx,
                              "tmap_rows": bool(rows_ok)}))
