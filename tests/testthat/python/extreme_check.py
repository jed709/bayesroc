import glob
import json
import os
import sys

# Every configuration in a directory (written by test-extreme-inputs.R) keeps its per-trial log-likelihood a
# log-probability and its gradient finite at points s = 1 and 3 units from a default fit's start in the
# sampler's coordinates (through the transport map, in the map's correlation order as a fit evaluates the
# model). A point is left out when a parameter is already non-finite before the likelihood runs or a
# random-effect correlation is near singular (smallest eigenvalue below 1e-9 in float64, 1e-7 in float32):
# the transport map's derivative through such a correlation (condition number past about 1e9, |corr| near
# 0.99997) is not finite, which a sampler meets as a divergence. The native path
# saves each configuration's anchors and start; the f32 path (a float32 process) reuses them.
# Arguments: the backend directory, the configuration directory, the path ("native" or "f32").

backend, directory, path = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, backend)
import numpy as np
import jax
import jax.numpy as jnp
from numpyro.infer.util import constrain_fn
import numpyro_backend as nb
import worker

singular = 1e-7 if path == "f32" else 1e-9


def corr_of(x):
    x = np.asarray(x, dtype=np.float64)
    D = int(round((1 + np.sqrt(1 + 8 * x.size)) / 2))
    t = np.zeros((D, D))
    t[np.tril_indices(D, -1)] = np.tanh(x)
    L = np.zeros((D, D))
    for i in range(D):
        rest = 1.0
        for j in range(i):
            L[i, j] = t[i, j] * np.sqrt(rest)
            rest *= 1 - t[i, j] ** 2
        L[i, i] = np.sqrt(rest)
    return L @ L.T


out = {}
for file in sorted(glob.glob(os.path.join(directory, "*.json"))):
    name = os.path.splitext(os.path.basename(file))[0]
    with open(file) as fh:
        config = json.load(fh)["config"]
    worker._convert_matrices(config)
    saved = os.path.join(directory, name + ".npz")
    if path == "f32":
        d = np.load(saved)
        coords = nb.NutsCoords(config, 1, 1, False, False, "laplace", 0.0, transport=d["aux"].size > 0,
                               aux=d["aux"] if d["aux"].size else None, announce=False)
        coords.start = d["start"]
    else:
        coords = nb.NutsCoords(config, 1, 1, False, False, "laplace", 0.0, transport=True, announce=False)
        np.savez(saved, aux=np.zeros(0) if coords.aux_flat is None else np.asarray(coords.aux_flat),
                 start=np.asarray(coords.start))
    if coords.tm is None:
        cfg, to_model = config, lambda v: v
    else:
        cfg = coords.tm.with_order(config, coords.aux, "_corr_perm")
        to_model = lambda v: coords.tm.forward(v, coords.aux, coords.vg.trim)[0]
    sites = lambda v: constrain_fn(nb.sdt_model, (cfg,), {}, coords.unflatten(to_model(v)), return_deterministic=True)
    log_lik = jax.jit(lambda v: sites(v)["log_lik"])
    gradient = jax.jit(jax.grad(lambda v: jnp.sum(sites(v)["log_lik"])))
    finite = jax.jit(lambda v: jnp.all(jnp.asarray([jnp.all(jnp.isfinite(x)) for k, x in sites(v).items()
                                                    if k != "log_lik" and jnp.issubdtype(x.dtype, jnp.floating)])))
    corr_keys = [k for k in coords.keys if k.startswith("L_corr")]
    start = jnp.asarray(np.asarray(coords.start)[0])
    rng = np.random.default_rng(15)
    failures, checked = [], 0
    for scale in (1.0, 3.0):
        for point in range(8):
            v = start + scale * jnp.asarray(rng.standard_normal(start.shape), dtype=start.dtype)
            vn = np.asarray(v, dtype=np.float64)
            eig = min([1.0] + [float(np.linalg.eigvalsh(corr_of(vn[coords.off[k][0]])).min()) for k in corr_keys])
            if not bool(finite(v)) or eig < singular:
                continue
            checked += 1
            ll = np.asarray(log_lik(v))
            g = np.asarray(gradient(v))
            bad = []
            if np.any(np.isnan(ll)) or np.any(ll == np.inf):
                bad.append("NaN or +inf log-likelihood")
            if np.nanmax(np.where(np.isfinite(ll), ll, -np.inf)) > 1e-7:
                bad.append(f"log-likelihood {np.nanmax(ll):.3g} > 0")
            if not np.all(np.isfinite(g)):
                bad.append(f"{int(np.sum(~np.isfinite(g)))} non-finite gradient components")
            if bad:
                failures.append(f"s = {scale:g}, point {point}: " + "; ".join(bad))
    out[name] = {"checked": checked, "failures": failures}
    print(f"{path} {name}: {checked} points checked, {len(failures)} failing", flush=True)

print("RESULT " + json.dumps(out))
