"""Module in infancy state."""

from trigonometric_utils import *
from matrix_utils import *
from functools import partial
from jax import jit, random


@partial(jit, static_argnums=(0, ))
def angle_by_angle_update(f, angles):
    """Updates all angles to optimal values one-by-one.

    Args:
        f: function periodic in each argument with period 2pi.
        angles: initial set of parameters.
    Returns:
        updated angles.
    """

    def body(i, angs):
        """Sets angle number i to its optimal value."""
        a_i_min = min_angle(lambda a: f(angs.at[i].set(a)))
        return angs.at[i].set(a_i_min)

    return lax.fori_loop(0, len(angles), body, angles)


def angle_by_angle_learn(cost_function,
                         num_angles,
                         initial_angles=None,
                         num_iterations=5000,
                         target_disc=1e-7
                         ):

    if initial_angles is None:
        initial_angles = random.uniform(random.PRNGKey(0), minval=0, maxval=2 * jnp.pi, shape=(num_angles,))

    jit_cost = jit(cost_function)  # Can potentially slow down the code.

    angles = initial_angles
    angles_history = [angles]
    loss_history = [jit_cost(angles)]

    for _ in range(num_iterations):
        angles = angle_by_angle_update(cost_function, angles)
        cost = jit_cost(angles)

        angles_history.append(angles)
        loss_history.append(cost)

        if cost < target_disc:
            break

    return angles_history, loss_history


# def splits(n_angles, n_moving_angles):
#     return [i*n_moving_angles for i in range(-(-n_angles//n_moving_angles))]
#
#
# def partial_update_angles(F, angles, s, n_moving_angles):
#     updated_angles = min_angles(F, angles, s, s+n_moving_angles)
#     angles = jnp.concatenate([angles[:s], updated_angles, angles[s+n_moving_angles:]])
#     return angles
