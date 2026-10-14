"""NUTS with Stan's U-turn checks across subtree boundaries (Stan 2.23).

numpyro's tree builder tests only the outer ends of each merged subtree. On a nearly isotropic
target, at step sizes where one oscillation period is close to a power-of-two number of leapfrog
steps, the summed momentum over a whole period is near zero, those ends never register a U-turn,
and trajectories run to the maximum depth. Every merge of two subtrees A (built first) and B is
here also tested on A plus B's first leaf and on A's last leaf plus B, as Stan does. The checks
are symmetric functions of the merged subtree, so the transition stays reversible."""
import types

import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax import random

from numpyro.infer import hmc_util as _hu
from numpyro.infer import hmc as _hmc_mod
from numpyro.util import cond, identity, while_loop


def _turn(inverse_mass_matrix, r_a, r_b, r_sum, unravel_fn):
    return _hu._is_turning(inverse_mass_matrix, unravel_fn(r_a), unravel_fn(r_b), unravel_fn(r_sum))


def _iterative_turning(inverse_mass_matrix, r, r_sum, r_ckpts, r_sum_ckpts, rp_ckpts, idx_min, idx_max,
                       unravel_fn):
    def _body_fn(state):
        i, _ = state
        base = r_sum_ckpts[i] - r_ckpts[i]
        turning = _turn(inverse_mass_matrix, r_ckpts[i], r, r_sum - base, unravel_fn)
        j = jnp.minimum(i + 1, r_ckpts.shape[0] - 1)
        mid = r_sum_ckpts[j] - r_ckpts[j]
        inner = i < idx_max
        a_ext = _turn(inverse_mass_matrix, r_ckpts[i], r_ckpts[j], mid - base + r_ckpts[j], unravel_fn)
        b_ext = _turn(inverse_mass_matrix, rp_ckpts[j], r, r_sum - mid + rp_ckpts[j], unravel_fn)
        return i - 1, turning | (inner & (a_ext | b_ext))

    _, turning = while_loop(lambda it: (it[0] >= idx_min) & ~it[1], _body_fn, (idx_max, False))
    return turning


def _iterative_build_subtree(prototype_tree, vv_update, kinetic_fn, inverse_mass_matrix, step_size, going_right,
                             rng_key, energy_current, max_delta_energy, r_ckpts, r_sum_ckpts):
    max_num_proposals = 2 ** prototype_tree.depth
    rp_ckpts = jnp.zeros_like(r_ckpts)

    def _cond_fn(state):
        tree, turning = state[0], state[1]
        return (tree.num_proposals < max_num_proposals) & ~turning & ~tree.diverging

    def _body_fn(state):
        current_tree, _, r_ckpts, r_sum_ckpts, rp_ckpts, rng_key = state
        rng_key, transition_rng_key = random.split(rng_key)
        z, r_prev, z_grad = _hu._get_leaf(current_tree, going_right)
        new_leaf = _hu._build_basetree(vv_update, kinetic_fn, z, r_prev, z_grad, inverse_mass_matrix, step_size,
                                       going_right, energy_current, max_delta_energy)
        new_tree = cond(current_tree.num_proposals == 0, new_leaf, identity,
                        (current_tree, new_leaf, inverse_mass_matrix, going_right, transition_rng_key),
                        lambda x: _hu._combine_tree(*x, False))
        leaf_idx = current_tree.num_proposals
        ckpt_idx_min, ckpt_idx_max = _hu._leaf_idx_to_ckpt_idxs(leaf_idx)
        r, unravel_fn = ravel_pytree(new_leaf.r_right)
        r_sum, _ = ravel_pytree(new_tree.r_sum)
        rp, _ = ravel_pytree(r_prev)
        r_ckpts, r_sum_ckpts, rp_ckpts = cond(
            leaf_idx % 2 == 0, (r_ckpts, r_sum_ckpts, rp_ckpts),
            lambda x: (x[0].at[ckpt_idx_max].set(r), x[1].at[ckpt_idx_max].set(r_sum),
                       x[2].at[ckpt_idx_max].set(rp)),
            (r_ckpts, r_sum_ckpts, rp_ckpts), identity)
        turning = _iterative_turning(inverse_mass_matrix, r, r_sum, r_ckpts, r_sum_ckpts, rp_ckpts,
                                     ckpt_idx_min, ckpt_idx_max, unravel_fn)
        return new_tree, turning, r_ckpts, r_sum_ckpts, rp_ckpts, rng_key

    basetree = prototype_tree._replace(num_proposals=0)
    tree, turning, _, _, _, _ = while_loop(_cond_fn, _body_fn,
                                           (basetree, False, r_ckpts, r_sum_ckpts, rp_ckpts, rng_key))
    return tree._replace(depth=prototype_tree.depth, turning=turning)


def _double_tree(current_tree, vv_update, kinetic_fn, inverse_mass_matrix, step_size, going_right, rng_key,
                 energy_current, max_delta_energy, r_ckpts, r_sum_ckpts):
    key, transition_key = random.split(rng_key)
    new_tree = _iterative_build_subtree(current_tree, vv_update, kinetic_fn, inverse_mass_matrix, step_size,
                                        going_right, key, energy_current, max_delta_energy, r_ckpts, r_sum_ckpts)
    tree = _hu._combine_tree(current_tree, new_tree, inverse_mass_matrix, going_right, transition_key, True)
    left, right = cond(going_right, (current_tree, new_tree), identity, (new_tree, current_tree), identity)
    ll, unravel_fn = ravel_pytree(left.r_left)
    lr, rl, rr = (ravel_pytree(x)[0] for x in (left.r_right, right.r_left, right.r_right))
    ls, rs = ravel_pytree(left.r_sum)[0], ravel_pytree(right.r_sum)[0]
    ext = (_turn(inverse_mass_matrix, ll, rl, ls + rl, unravel_fn)
           | _turn(inverse_mass_matrix, lr, rr, rs + lr, unravel_fn))
    return tree._replace(turning=tree.turning | ext)


def _with_globals(fn, **over):
    f = types.FunctionType(fn.__code__, {**fn.__globals__, **over}, fn.__name__, fn.__defaults__, fn.__closure__)
    f.__kwdefaults__ = fn.__kwdefaults__
    return f


build_tree = _with_globals(_hu.build_tree, _double_tree=_double_tree)
hmc = _with_globals(_hmc_mod.hmc, build_tree=build_tree)
