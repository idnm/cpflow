"""Routines for efficient multi-start optimization."""

from functools import partial

import jax
import optax
from jax import jit, value_and_grad, hessian, jvp, grad

from cpflow.matrix_utils import *
from cpflow.penalty import *
from cpflow.trigonometric_utils import *


@partial(jit, static_argnums=(0, 1, 4))
def optax_update_step(loss_and_grad_func, opt, opt_state, params, preconditioner_func):

    if preconditioner_func is None:
        preconditioner_func = lambda x, y: y

    loss, grads = loss_and_grad_func(params)
    grads = preconditioner_func(params, grads)
    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss


def optax_minimize(loss_func,
                   num_params,
                   opt,
                   preconditioner_func=None,
                   loss_is_loss_and_grad=False,
                   initial_params=None,
                   num_iterations=5000,
                   keep_history=True,
                   target_loss=1e-7):

    if target_loss != 1e-7:
        print('Warning: target loss not yet supported.')  # Probably need to switch to lax.while_loop ?

    if initial_params is None:
        initial_params = random_angles(num_params)

    if loss_is_loss_and_grad:
        initial_loss, _ = loss_func(initial_params)
    else:
        initial_loss = loss_func(initial_params)

    opt_state = opt.init(initial_params)
    loss_and_grad_func = value_and_grad(loss_func)

    def iteration_with_history(i, params_loss_and_opt_state):
        params_history, loss_history, opt_state = params_loss_and_opt_state
        params = params_history[i]
        if loss_is_loss_and_grad:
            params, opt_state, loss = optax_update_step(loss_func, opt, opt_state, params, preconditioner_func)
        else:
            params, opt_state, loss = optax_update_step(loss_and_grad_func, opt, opt_state, params, preconditioner_func)
        return [params_history.at[i+1].set(params), loss_history.at[i].set(loss), opt_state]

    def iteration_without_history(i, params_loss_and_opt_state):
        params, best_params, previous_loss, best_loss, opt_state = params_loss_and_opt_state

        if loss_is_loss_and_grad:
            new_params, opt_state, loss = optax_update_step(loss_func, opt, opt_state, params, preconditioner_func)
        else:
            new_params, opt_state, loss = optax_update_step(loss_and_grad_func, opt, opt_state, params, preconditioner_func)

        # If new loss is lower than best loss, update the latter.
        best_loss, best_params = lax.cond(loss < best_loss,
                                          lambda x: [loss, params],
                                          lambda x: [best_loss, best_params],
                                          None)

        return new_params, best_params, loss, best_loss, opt_state

    if keep_history:
        inititial_params_history = jnp.zeros((num_iterations, len(initial_params)))
        inititial_params_history = inititial_params_history.at[0].set(initial_params)

        initial_loss_history = jnp.zeros((num_iterations, )).at[0].set(initial_loss)

        params_history, loss_history, opt_state = lax.fori_loop(0, num_iterations, iteration_with_history,
                                                  [inititial_params_history, initial_loss_history, opt_state])

        return params_history, loss_history

    else:
        initial_best_params = initial_params
        initial_best_loss = initial_loss
        params, best_params, loss, best_loss, opt_state = lax.fori_loop(0, num_iterations, iteration_without_history,
                                                                        (initial_params, initial_best_params, initial_loss, initial_best_loss, opt_state))

        return jnp.array([initial_params, best_params]), jnp.array([initial_loss, best_loss])


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
             opt_instance=None,
             u_func=None,
             loss_is_loss_and_grad=False,
             target_loss=1e-7,
             keep_history=True,
             **kwargs):

    kwargs['target_loss'] = target_loss

    natural_preconditioner = plain_natural_preconditioner(u_func)
    hessian_preconditioner = plain_hessian_preconditioner(u_func)

    if method == 'angle by angle':
        if u_func is not None:
            print('Warning, method aba does not use preconditioner, why u_func is provided?')
        angles_history, loss_history = angle_by_angle_minimize(loss_func, num_params, **kwargs)

    elif method == 'adam':
        if opt_instance is None:
            opt_instance = optax.adam(learning_rate)
        angles_history, loss_history = optax_minimize(loss_func,
                                        num_params,
                                        opt_instance,
                                        loss_is_loss_and_grad=loss_is_loss_and_grad,
                                        keep_history=keep_history,
                                        **kwargs)

    elif method == 'natural adam':
        if opt_instance is None:
            opt_instance = optax.adam(learning_rate)
        angles_history, loss_history = optax_minimize(loss_func,
                                                      num_params,
                                                      opt_instance,
                                                      preconditioner_func=natural_preconditioner,
                                                      loss_is_loss_and_grad=loss_is_loss_and_grad,
                                                      **kwargs)

    elif method == 'natural gd':
        angles_history, loss_history = gradient_descent_minimize(loss_func,
                                                                 num_params,
                                                                 learning_rate=learning_rate,
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

    return angles_history, loss_history


def mynimize_repeated(loss_func,
                      num_params,
                      method='adam',
                      learning_rate=0.1,
                      target_loss=1e-7,
                      u_func=None,
                      initial_params_batch=None,
                      num_repeats=1,
                      regularization_func=None,
                      keep_history=True,
                      compute_losses=True,
                      **kwargs):

    """Runs minimization routine multiple times efficiently parallelizing computations.

    Args:
        loss_func: function to minimize.
        num_params: number of parameters in the funciton.
        learning_rate: learning rate to use in gradient-based optimizers.
        method: which minimization procedure to use.
        target_loss: stop learning if loss reaches this value.
        u_func: function returning unitary matrix, used for constructing preconditioner for the natural gradient.
        initial_params_batch: batch of initial parameters of shape (num_params) or (n, num_params).
        num_repeats: how many times to repeat minimization starting from different initial conditions.
        If initial_params_batch is specified num_repeats becomes irrelevant. Otherwise random initial conditions
        are used for each minimization.
        regularization_func: if provided, the actual function to minimize is loss_func+regularization_func.

    Returns: single result or list of results (depending on batch size). Each result is a dict containing
     'params': history of params during learning.
     'regloss': history of regularized loss.
     'loss': history of (unregularized) loss.
     'reg': history of regularization function = regloss-loss.
    """

    # If initial parameters are not provided generate random ones.
    # input_is_vector variable keeps track of whether to output single result or list of results.
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

    # If regularization function is provided add it to the basic loss. Otherwise just rename basic loss to regloss.
    if regularization_func is None:
        regloss_func = loss_func
    else:
        regloss_func = lambda params: loss_func(params) + regularization_func(params)

    # This is a workaround to avoid repeated compilations of a single update step.
    # Maybe no longer needed with vmap. Need to check.
    if method in ['adam', 'natural adam']:
        loss_is_loss_and_grad = True
        regloss_func = value_and_grad(regloss_func)

        opt = optax.adam(learning_rate)
    else:
        loss_is_loss_and_grad = False
        opt = None

    def mynimize_particular(initial_params):

        return mynimize(regloss_func,
                        num_params,
                        method=method,
                        learning_rate=learning_rate,
                        opt_instance=opt,
                        target_loss=target_loss,
                        initial_params=initial_params,
                        u_func=u_func,
                        loss_is_loss_and_grad=loss_is_loss_and_grad,
                        keep_history=keep_history,
                        **kwargs)

    if input_is_vector:
        batch_params_history, batch_regloss_history = jit(vmap(mynimize_particular))(jnp.array(initial_params_batch))
        results = [{'params': p, 'loss': l} for p, l in zip(batch_params_history, batch_regloss_history)]

        if compute_losses:
            if regularization_func is not None:
                batch_reg_history = jit(vmap(vmap(regularization_func)))(batch_params_history)
                batch_loss_history = batch_regloss_history-batch_reg_history
                results = [{'params': p, 'loss': l, 'reg': r, 'regloss': rl} for p, l, r, rl in
                           zip(batch_params_history, batch_loss_history, batch_reg_history, batch_regloss_history)]
        return results

    else:
        params_history, regloss_history = mynimize_particular(initial_params_batch[0])
        result = {'params': params_history, 'loss': regloss_history}
        if compute_losses:
            if regularization_func is not None:
                reg_history = jit(vmap(regularization_func))(params_history)
                loss_history = regloss_history - reg_history
                result = {'params': params_history, 'loss': loss_history, 'reg': reg_history, 'regloss': regloss_history}

        return result


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
                  keep_history=True,
                  **kwargs):

    if disc_func == 'swap':
        num_qubits = int(jnp.log2(u_target.shape[0]))
        loss_func = lambda angs: disc2_swap(u_func(angs), u_target, num_qubits)
    else:
        loss_func = lambda angs: cost_HST(u_func(angs), u_target)

    if regularization_options is not None:
        regularization_func = construct_penalty_function(regularization_options)
    else:
        regularization_func = lambda x: 0

    return mynimize_repeated(loss_func,
                             num_params,
                             method=method,
                             learning_rate=learning_rate,
                             u_func=u_func,
                             num_repeats=num_repeats,
                             initial_params_batch=initial_angles,
                             regularization_func=regularization_func,
                             target_loss=target_loss,
                             keep_history=keep_history,
                             **kwargs)


# def adaptive_decompose(u_target,
#                        layer,
#                        target_num_gates,
#                        accepted_num_gates,
#                        regularization_options=None,
#                        disc_func=None):
#
#     reg_options = {'r': 0.005,
#                    ''}