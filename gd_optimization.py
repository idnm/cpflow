"""Module in infancy."""

from functools import partial
import optax
from jax import jit, random, value_and_grad

from matrix_utils import *


@partial(jit, static_argnums=(0, 1,))
def gradient_descent_update(loss_and_grad, opt, opt_state, angles):
    loss, grads = loss_and_grad(angles)
    updates, opt_state = opt.update(grads, opt_state)
    angles = optax.apply_updates(angles, updates)
    return angles, opt_state, loss


def gradient_descent_learn(cost_func, num_angles,
                           initial_angles=None,
                           learning_rate=0.01,
                           num_iterations=5000,
                           target_disc=1e-10):
    
    if initial_angles is None:
        key = random.PRNGKey(0)
        initial_angles = random.uniform(key, shape=(num_angles,), minval=0, maxval=2 * jnp.pi)

    if len(initial_angles.shape) == 1:
        all_angles = [initial_angles]
    elif len(initial_angles.shape) > 2:
        print('initial angles must be a list or an array of lists, got shape {}'.format(initial_angles.shape))
        return
    else:
        all_angles = initial_angles

    loss_and_grad = value_and_grad(cost_func)
    opt = optax.adam(learning_rate)

    all_angles_and_loss_histories = []
    for angles in all_angles:
        opt_state = opt.init(angles)
        angles_history = []
        loss_history = []
        for _ in range(num_iterations):
            angles, opt_state, loss = gradient_descent_update(loss_and_grad, opt, opt_state, angles)
            angles_history.append(angles)
            loss_history.append(loss)
            if loss < target_disc:
                break
        all_angles_and_loss_histories.append([angles_history, loss_history])

    if len(initial_angles.shape) == 1:
        return all_angles_and_loss_histories[0]
    else:
        return all_angles_and_loss_histories
