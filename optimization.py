"""Module in infancy."""

from functools import partial

import jax
import optax
from jax import jit, random, value_and_grad, hessian, jvp, grad

from matrix_utils import *
from trigonometric_utils import *
from penalty import *


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
                   target_loss=1e-7):
    
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
        if loss < target_loss:
            break

    return params_history, loss_history


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


@partial(jit, static_argnums=(0, 1))
def gradient_descent_update_step(cost_func, preconditioner_func, params, learning_rate):

    loss_and_grad = value_and_grad(cost_func)
    loss, grads = loss_and_grad(params)
    new_params = params - learning_rate*preconditioner_func(params, grads)
    new_loss = cost_func(new_params)
    return new_params, new_loss


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


def angle_by_angle_minimize(cost_function,
                            num_angles,
                            initial_angles=None,
                            num_iterations=5000,
                            target_loss=1e-7
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

        if cost < target_loss:
            break

    return angles_history, loss_history


def mynimize(loss_func,
             num_params,
             method='adam',
             learning_rate=0.1,
             u_func=None,
             target_loss=1e-7,
             **kwargs):

    kwargs['target_loss'] = target_loss

    natural_preconditioner = plain_natural_preconditioner(u_func)
    hessian_preconditioner = plain_hessian_preconditioner(u_func)

    if method == 'angle by angle':
        if u_func is not None:
            print('Warning, method aba does not use preconditioner, why u_func is provided?')
        angles_history, loss_history = angle_by_angle_minimize(loss_func, num_params, **kwargs)

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

    return {'params': angles_history, 'loss': loss_history}


def mynimize_regularized(regularization_func,
                         loss_func,
                         num_params,
                         method='adam',
                         learning_rate=0.1,
                         target_loss=1e-7,
                         u_func=None,
                         **kwargs):
    regloss_func = lambda params: loss_func(params) + regularization_func(params)
    learning_res = mynimize(regloss_func,
                            num_params,
                            method=method,
                            learning_rate=learning_rate,
                            u_func=u_func,
                            target_loss=target_loss,
                            **kwargs)

    params_history, regloss_history = learning_res['params'], learning_res['loss']

    params_history = jnp.array(params_history)
    loss_history = vmap(jit(loss_func))(params_history)
    reg_history = vmap(jit(regularization_func))(params_history)

    return {'params': params_history, 'regloss': jnp.array(regloss_history), 'loss': loss_history, 'reg': reg_history}


def mynimize_repeated(loss_func,
                      num_params,
                      method='adam',
                      learning_rate=0.1,
                      target_loss=1e-7,
                      target_reg=None,
                      u_func=None,
                      initial_params_batch=None,
                      num_repeats=1,
                      regularization_func=None,
                      **kwargs):

    if initial_params_batch is None:
        key = random.PRNGKey(0)
        initial_params_batch = []
        for _ in range(num_repeats):
            key, subkey = random.split(key)
            initial_params_batch.append(random_angles(num_params, key=subkey))
        if num_repeats == 1:
            input_is_vector = False
        else:
            input_is_vector = True

    else:
        if num_repeats != 1:
            print('Warning, initial conditions provided and number of repeats will be ignored.')

        initial_params_shape = jnp.array(initial_params_batch).shape
        if len(initial_params_shape) == 1:
            initial_params_batch = [initial_params_batch]
            input_is_vector = False
        elif len(initial_params_shape) == 2:
            input_is_vector = True
        else:
            print('Warning: initial parameters must be either 1d or 2d array (multiple initial conditions)')

    if regularization_func is None:
        minimize_procedure = mynimize
    else:
        minimize_procedure = partial(mynimize_regularized, regularization_func)

    result_history = []
    best_params_history = []
    success_history = []

    for initial_params in initial_params_batch:
        learn_result = minimize_procedure(loss_func,
                                         num_params,
                                         method=method,
                                         learning_rate=learning_rate,
                                         target_loss=target_loss,
                                         initial_params=initial_params,
                                         u_func=u_func,
                                         **kwargs)

        best_state = jnp.argmin(learn_result['regloss'])
        success = learn_result['loss'][best_state] < target_loss

        if target_reg is not None:
            reg_success = learn_result['reg'][best_state] < target_reg
            success = success and reg_success

        result_history.append(learn_result)
        best_params_history.append(learn_result['params'][best_state])
        success_history.append(success)

    if input_is_vector:
        return result_history, best_params_history, success_history
    else:
        return result_history[0], best_params_history[0], success_history[0]


def unitary_learn(u_func,
                  u_target,
                  num_params,
                  method='adam',
                  learning_rate=0.1,
                  target_loss=1e-7,
                  disc_func=None,
                  regularization_options=None,
                  initial_angles=None,
                  num_repeats=1,
                  **kwargs):

    if disc_func is None:
        disc_func = lambda angs: disc2(u_func(angs), u_target)

    if regularization_options is not None:
        regularization_func, target_reg = construct_penalty_function(regularization_options)
    else:
        regularization_func = lambda x: 0
        target_reg = None

    return mynimize_repeated(disc_func,
                             num_params,
                             method=method,
                             learning_rate=learning_rate,
                             u_func=u_func,
                             num_repeats=num_repeats,
                             initial_params_batch=initial_angles,
                             regularization_func=regularization_func,
                             target_loss=target_loss,
                             target_reg=target_reg,
                             **kwargs)

