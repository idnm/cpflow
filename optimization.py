"""Module in infancy."""

from functools import partial

import jax
import optax
from jax import jit, random, value_and_grad, hessian, jvp, grad

from matrix_utils import *
from trigonometric_utils import *


def random_angles(num_angles, key=None):
    if key is None:
        key = random.PRNGKey(0)
    return random.uniform(key, (num_angles, ), minval=0, maxval=2*jnp.pi)


@partial(jit, static_argnums=(0, 1, 4))
def optax_update_step(loss_and_grad, opt, opt_state, params, preconditioner_func):
    if preconditioner_func is None:
        preconditioner_func = lambda x, y: y

    loss, grads = loss_and_grad(params)
    grads = preconditioner_func(params, grads)
    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


def optax_minimize(cost_func,
                   num_params,
                   opt,
                   preconditioner_func=None,
                   initial_params=None,
                   num_iterations=5000,
                   target_disc=1e-7):
    
    if initial_params is None:
        initial_params = random_angles(num_params)

    loss_and_grad = value_and_grad(cost_func)
    opt_state = opt.init(initial_params)

    params = initial_params
    params_history = []
    loss_history = []
    for _ in range(num_iterations):
        params, opt_state, loss = optax_update_step(loss_and_grad, opt, opt_state, params, preconditioner_func)
        params_history.append(params)
        loss_history.append(loss)
        if loss < target_disc:
            break

    return params_history, loss_history


@partial(jit, static_argnums=(0, 1))
def gradient_descent_update_step(cost_func, preconditioner_func, params, learning_rate):

    loss_and_grad = value_and_grad(cost_func)
    loss, grads = loss_and_grad(params)
    new_params = params - learning_rate*preconditioner_func(params, grads)
    new_loss = cost_func(new_params)
    return new_params, new_loss


def plain_hessian_preconditioner(cost_func, tikhonov_delta=1e-4):

    def preconditioner(params, grads):
        reg_hess = hessian(cost_func)(params) + tikhonov_delta*jnp.identity(len(params))
        return jnp.linalg.inv(reg_hess) @ grads

    return preconditioner


def sparse_hessian_preconditioner(cost_func, tikhonov_delta=1e-4):

    def hvp(f, primals, tangents):
        return jvp(grad(f), (primals,), (tangents,))[1]

    def preconditioner(params, grads):
        sol = jax.scipy.sparse.linalg.cg(lambda x: hvp(cost_func, params, x)+tikhonov_delta*x, grads)[0]
        return sol

    return preconditioner


def plain_natural_preconditioner(u_func, tikhonov_delta=1e-4):
    def preconditioner(params, grads):
        g = fubini_study(u_func, params)+tikhonov_delta*jnp.identity(len(grads))
        return jnp.linalg.inv(g) @ grads

    return preconditioner


def gradient_descent_minimize(cost_func,
                              num_params,
                              preconditioner_func=None,
                              learning_rate=0.1,
                              initial_params=None,
                              num_iterations=5000,
                              target_loss=1e-7):
    if initial_params is None:
        initial_params = random_angles(num_params)
    if preconditioner_func is None:
        preconditioner_func = lambda x, y: y

    params = initial_params
    params_history = []
    loss_history = []
    for _ in range(num_iterations):
        params, loss = gradient_descent_update_step(cost_func, preconditioner_func, params, learning_rate)
        params_history.append(params)
        loss_history.append(loss)
        if loss < target_loss:
            break

    return params_history, loss_history


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


def unitary_learn(u_func,
                  u_target,
                  num_params,
                  method,
                  learning_rate,
                  cp_mask=None,
                  cp_penalty_func=None,
                  **kwargs):

    disc_func = lambda angs: disc2(u_func(angs), u_target)
    loss_func = disc_func
    if cp_mask is not None:
        penalty_func = lambda angs: cp_penalty_func(angs*cp_mask)
        loss_func = lambda angs: disc_func(angs) + penalty_func(angs)

    natural_preconditioner = plain_natural_preconditioner(u_func)
    hessian_preconditioner = plain_hessian_preconditioner(u_func)

    if method == 'angle by angle':
        if cp_mask is not None:
            print('Warning: cp penalty data is ignored.')
        angles_history, loss_history = angle_by_angle_learn(disc_func, num_params, **kwargs)

    elif method == 'adam':
        opt = optax.adam(learning_rate)
        angles_history, loss_history = optax_minimize(loss_func, num_params, opt, **kwargs)

    elif method == 'natural gd':
        angles_history, loss_history = gradient_descent_minimize(loss_func,
                                                                 num_params,
                                                                 learning_rate=learning_rate,
                                                                 preconditioner_func=natural_preconditioner,
                                                                 **kwargs)

    elif method == 'natural adam':
        angles_history, loss_history = optax_minimize(loss_func,
                                                      num_params,
                                                      optax.adam(learning_rate),
                                                      preconditioner_func=natural_preconditioner,
                                                      **kwargs)

    elif method == 'hessian':
        angles_history, loss_history = gradient_descent_minimize(loss_func,
                                                                 num_params,
                                                                 learning_rate=learning_rate,
                                                                 preconditioner_func=hessian_preconditioner,
                                                                 **kwargs)

    else:
        print('Method {} not supported'.format(method))

    if cp_mask is None:
        return jnp.array(angles_history), jnp.array(loss_history)
    else:
        angles_history = jnp.array(angles_history)
        disc_history = vmap(jit(disc_func))(angles_history)
        penalty_history = vmap(jit(penalty_func))(angles_history)
        return angles_history, jnp.array(loss_history), disc_history, penalty_history
