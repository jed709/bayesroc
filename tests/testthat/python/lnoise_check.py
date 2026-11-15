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
col = lambda k: np.array([float(r[k]) for r in rows])
a, Dn, Df, s = col("a"), col("Dn"), col("Df"), col("s")
ref = [np.log(col("P")), col("da"), col("dln"), col("dlf"), col("ds")]
ln = np.where(Dn == 0, -np.inf, np.log(np.where(Dn == 0, 1.0, Dn)))
lf = np.where(np.isinf(Df), np.inf, np.log(np.where(np.isinf(Df), 1.0, Df)))
args = [jnp.asarray(v) for v in (a, ln, lf, s)]
nat = [np.asarray(v) for v in nb._lnoise_raw(*args)]


def worst(x, y):
    return float(max(np.max(np.abs(p - q) / np.maximum(1.0, np.abs(q))) for p, q in zip(x, y)))


print("RESULT " + json.dumps({"ffi_loaded": nb._bvn_lib is not None, "native_vs_ref": worst(nat, ref),
                              "finite": bool(all(np.all(np.isfinite(v)) for v in nat))}))
