"""Module in infancy state."""

from trigonometric_utils import *
from matrix_utils import *
from functools import partial
from jax import jit, random


def splits(n_angles, n_moving_angles):
    return [i*n_moving_angles for i in range(-(-n_angles//n_moving_angles))]


def partial_update_angles(F, angles, s, n_moving_angles):
    updated_angles = min_angles(F, angles, s, s+n_moving_angles)
    angles = jnp.concatenate([angles[:s], updated_angles, angles[s+n_moving_angles:]])
    return angles


@partial(jit, static_argnums=(0, ))
def staircase_update(f, angles):

    def body(i, angs):
        a_i_min = min_angle(lambda a: f(angs.at[i].set(a)))
        return angs.at[i].set(a_i_min)

    return lax.fori_loop(0, len(angles), body, angles)


def staircase_min(f, n_angles, initial_angles=None, n_iterations=100, keep_history=False):
    if initial_angles is None:
        initial_angles = random.uniform(random.PRNGKey(0), minval=0, maxval=2 * jnp.pi, shape=(n_angles,))

    angles = initial_angles
    angles_history = [angles]

    for _ in range(n_iterations):
        angles = staircase_update(f, angles)
        if keep_history:
            angles_history.append(angles)

    angles_history = jnp.array(angles_history)
    if keep_history:
        loss_history = vmap(jit(f))(angles_history)
        return angles_history, loss_history

    return angles


def staircase_learn(u_func, u_target, n_angles, **kwargs):
    return staircase_min(lambda angles: disc2(u_func(angles), u_target), n_angles, **kwargs)
