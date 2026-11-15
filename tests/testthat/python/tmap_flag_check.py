"""The transport map with a super-block's candidate blocks mapped (flag 0) and non-centred
(flag 1): the map's log-determinant term against the numerical log|det| of its Jacobian,
forward-inverse round trips, and the mapped gradient against a finite difference.
Usage: tmap_flag_check.py <backend dir> <ffi lib> <config.json>"""
import json
import os
import sys

sys.path.insert(0, sys.argv[1])
os.environ["BAYESROC_BVN_FFI"] = sys.argv[2]
os.environ["BAYESROC_ENABLE_X64"] = "1"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"
import numpy as np
import jax
import jax.numpy as jnp
import numpyro_backend as nb
import drghmc as dr

cfg = json.load(open(sys.argv[3]))
cfg = cfg.get("config", cfg)
dr._convert_matrices(cfg)
ctrl = dr._merge_control(dict(transport=True))
vg, unflatten, keys, off, D, post = dr._build(nb.sdt_model, cfg)
tm = dr._transport(nb.sdt_model, cfg, ctrl, off, unflatten)
vg_s = dr._sampler_vg(vg, tm)
OFF = vg.OFF0
rng = np.random.default_rng(0)
th = rng.normal(size=(1, D)) * 0.3
aux0 = tm.anchors_chains(vg, th, OFF, parallel=False)
res = {"ffi_loaded": nb._bvn_lib is not None, "candidates": int(sum(sb.cand is not None for sb in tm.sbs)),
       "kept_dims": [int(d) for d in tm.kept_dims]}
e = rng.normal(size=D)
e /= np.linalg.norm(e)
h = 1e-5
for flag in (0.0, 1.0):
    aux = tm.with_flags(aux0, [flag] * len(tm.sbs))
    P = tuple(p[0] for p in tm.parts(aux))
    tv = tm.cheap_of(aux)
    tht = tm.inverse_chains(th, aux, vg.trim)
    res[f"roundtrip_{int(flag)}"] = float(np.max(np.abs(tm.forward_chains(tht, aux, vg.trim) - th)))
    v = jnp.asarray(tht[0])
    _, ld = jax.jit(lambda v: tv.forward(v, P, vg.trim))(v)
    _, pull = jax.vjp(lambda v: tv.forward(v, P, vg.trim)[0], v)
    pj = jax.jit(lambda ct: pull(ct)[0])
    eye = np.eye(D)
    J = np.stack([np.asarray(pj(jnp.asarray(eye[i]))) for i in range(D)])
    sign, logdet = np.linalg.slogdet(J)
    res[f"logdet_err_{int(flag)}"] = float(abs(logdet + float(ld)))
    res[f"jacobian_sign_{int(flag)}"] = float(sign)
    f = jax.jit(lambda v: tv.wrap(tm.pot_off)(v, OFF, P))
    U, g = f(v)
    fd = (float(f(v + h * e)[0]) - float(f(v - h * e)[0])) / (2 * h)
    res[f"grad_err_{int(flag)}"] = float(abs(fd - float(jnp.dot(g, e))) / max(1.0, abs(fd)))
print("RESULT " + json.dumps(res))
