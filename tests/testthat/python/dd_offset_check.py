import json
import os
import sys
from decimal import Decimal, getcontext

sys.path.insert(0, sys.argv[1])
os.environ["BAYESROC_BVN_FFI"] = sys.argv[2]
os.environ["BAYESROC_ENABLE_X64"] = "1"
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
import numpyro_backend as nb

getcontext().prec = 60
rng = np.random.default_rng(31)
N, G, K = 321, 4, 5
gm = nb._make_batch_gap_mask_np(G, K, 2, 2)
mid = rng.normal(0.0, 1.0, N)
gaps = rng.normal(-0.5, 1.0, (N, G))
tiny = rng.uniform(size=(N, G)) < 0.25
gaps[tiny] = rng.uniform(-30.0, -12.0, tiny.sum())
k = rng.integers(0, K, N)
mu1, s1 = rng.normal(0.0, 1.0, N), rng.uniform(0.4, 2.5, N)
s2 = rng.uniform(0.4, 2.5, N)
rho = rng.choice([-1.0, 1.0], N) * rng.uniform(0.05, 0.95, N)
c_R = rng.normal(0.5, 1.0, N)


def gap_value(g, softplus):
    e = Decimal(float(g)).exp()
    return (1 + e).ln() if softplus else e


def criterion(i, softplus):
    c = Decimal(float(mid[i]))
    for g in range(G):
        if gm[g, k[i]] != 0:
            c += int(gm[g, k[i]]) * gap_value(gaps[i, g], softplus)
    return c


def setup(softplus):
    crit = np.array([float(criterion(i, softplus)) for i in range(N)])
    z1 = (crit - mu1) / s1
    mu2 = -(rho * s2) * z1 * (1.0 + 1e-12 * rng.uniform(-1.0, 1.0, N))
    c_K = crit - c_R + 1e-11 * rng.uniform(-1.0, 1.0, N)
    return mu2, c_K


def reference(crossing, softplus, mu2, c_K):
    out, scale = [], []
    for i in range(N):
        c = criterion(i, softplus)
        if crossing:
            z1 = (c - Decimal(float(mu1[i]))) / Decimal(float(s1[i]))
            zc = -Decimal(float(mu2[i])) / (Decimal(float(rho[i])) * Decimal(float(s2[i])) + Decimal(1e-30))
            out.append(zc - z1)
            scale.append(abs(zc) + abs(z1))
        else:
            u = Decimal(float(c_R[i])) + Decimal(float(c_K[i]))
            out.append(u - c)
            scale.append(abs(u) + abs(c))
    return out, scale


worst, same = {}, True
for softplus in (False, True):
    mu2, c_K = setup(softplus)
    for crossing in (True, False):
        ref, scale = reference(crossing, softplus, mu2, c_K)
        full = None
        f = jax.jit(lambda a, m, gp, kk, crossing=crossing, softplus=softplus:
                    nb._dd_offset_native(crossing, softplus, a, m, gp, gm, kk))
        for L in (321, 1, 8, 16, 17, 64):
            a = (mu1, s1, mu2, s2, rho) if crossing else (c_R, c_K)
            got = np.asarray(f(tuple(jnp.asarray(x[:L]) for x in a), jnp.asarray(mid[:L]), jnp.asarray(gaps[:L]),
                               jnp.asarray(k[:L])))
            if full is None:
                full = got
            same = same and bool(np.array_equal(got, full[:L]))
            ulps = [abs(Decimal(float(got[i])) - ref[i]) /
                    max(Decimal(float(np.spacing(abs(float(ref[i]))))), Decimal("1e-27") * scale[i])
                    for i in range(L)]
            key = f"{'crossing' if crossing else 'criterion'}_{'softplus' if softplus else 'exp'}_n{L}"
            worst[key] = float(max(ulps))
print("RESULT " + json.dumps({"dd_offset_max_ulps": max(worst.values()), "by_case": worst, "same_at_every_length": same,
                              "ffi_loaded": nb._bvn_lib is not None}))
