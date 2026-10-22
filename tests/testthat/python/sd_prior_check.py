import json
import os
import sys

sys.path.insert(0, sys.argv[1])
os.environ["BAYESROC_BVN_FFI"] = sys.argv[2]
os.environ["BAYESROC_ENABLE_X64"] = "1"
import jax
import jax.numpy as jnp
import numpyro_backend as nb

x = jnp.linspace(1e-6, 400.0, 4_000_001)
out = {}
for s in sys.argv[3].split(";"):
    base = nb.parse_prior(s)
    d = nb._positive_dist(base)
    lp = d.log_prob(x)
    p = jnp.where(jnp.isfinite(lp), jnp.exp(lp), 0.0)
    a, b = jnp.asarray(0.3), jnp.asarray(0.9)
    smp = d.sample(jax.random.PRNGKey(0), (2000,))
    out[s] = dict(mass=float(jnp.trapezoid(p, x)), nan=bool(jnp.isnan(lp).any()),
                  shape=float((d.log_prob(a) - d.log_prob(b)) - (base.log_prob(a) - base.log_prob(b))),
                  positive=bool((smp > 0).all()))
print("RESULT " + json.dumps(out))
