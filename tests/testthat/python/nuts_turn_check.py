import json
import sys

sys.path.insert(0, sys.argv[1])
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from numpyro.infer.hmc import hmc as hmc_stock
from numpyro.infer.util import ParamInfo
from nuts_turn import hmc as hmc_checked

D = 2000
pot = lambda z: 0.5 * jnp.sum(z ** 2)
keys = jax.random.split(jax.random.PRNGKey(1), 4)


def steps(hmc, eps, n=40):
    init_kernel, sample_kernel = hmc(potential_fn=pot, algo="NUTS")

    def run(key):
        z0 = jax.random.normal(key, (D,))
        st = init_kernel(ParamInfo(z0, pot(z0), z0), num_warmup=0, step_size=eps, adapt_step_size=False,
                         adapt_mass_matrix=False, max_tree_depth=10, rng_key=key)
        _, out = jax.lax.scan(lambda s, _: (lambda t: (t, (t.num_steps, t.z)))(sample_kernel(s)), st, None, length=n)
        return out

    ns, z = jax.jit(jax.vmap(run))(keys)
    return float(np.mean(ns)), np.asarray(z)


res = {}
res["stock_resonant"], _ = steps(hmc_stock, 0.1)
res["checked_resonant"], _ = steps(hmc_checked, 0.1)
s1, z1 = steps(hmc_stock, 0.3)
s2, z2 = steps(hmc_checked, 0.3)
res["max_draw_diff_off_resonance"] = float(np.max(np.abs(z1 - z2)))
res["steps_off_resonance"] = [s1, s2]
print("RESULT " + json.dumps(res))
