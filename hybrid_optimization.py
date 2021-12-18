"""Module in infancy."""

import jax.numpy as jnp
from angle_by_angle_optimization import staircase_learn
from gd_optimization import gradient_descent_learn


def hybrid_learn(u_func, u_target, n_angles, n_stairs=10, **kwargs):
    gd_kwargs = kwargs.copy()
    sc_kwargs = kwargs
    sc_kwargs.update({'n_iterations': n_stairs})

    sc_angles_history, sc_loss_history = staircase_learn(u_func, u_target, n_angles, **sc_kwargs)

    gd_init_angles = sc_angles_history[-1]
    gd_kwargs.update({'initial_angles': gd_init_angles})
    gd_kwargs.pop('keep_history', None)
    gd_angles_history, gd_loss_history = gradient_descent_learn(u_func, u_target, n_angles, **gd_kwargs)

    return jnp.concatenate([sc_angles_history, gd_angles_history]), jnp.concatenate([sc_loss_history, gd_loss_history])