"""Building, employing, and projecting CP templates."""

from functools import partial

import numpy as np
from jax import random, jit, ops, vmap

from cpflow.optimization import mynimize
from cpflow.penalty import *
from cpflow.trigonometric_utils import random_angles


def random_cp_angles(num_angles, cp_mask, cp_dist='uniform', key=random.PRNGKey(0)):
    """Randomly initializes angles.

    Args:
        num_angles: total number of angles.
        cp_mask: mask specifying positions of cp angles.
        cp_dist: type of distribution to use.
            Non-cp angles are always initialized uniformly. CP angles can be initialized as
            'uniform': uniformly in (0, 2pi).
            'normal': normally peaked at 0.
            '0': to zero values.
        key: random key.

    Returns:
        array of length num_angles corresponding to all angles.

    """

    key, subkey = random.split(key)
    rnd_angles = random_angles(num_angles, key=subkey)

    if cp_dist == 'uniform':
        return rnd_angles
    elif cp_dist == '0':
        return rnd_angles * (1 - cp_mask)
    elif cp_dist == 'normal':
        key, subkey = random.split(key)
        return rnd_angles * (1 - cp_mask) + 1.5 * random.normal(subkey, shape=(num_angles,)) * cp_mask
    else:
        print('cp_dist', cp_dist, 'not supported')


@partial(jit, static_argnums=(1,))
def cz_value(a, threshold=1e-2):
    """Returns 0 if CP-angle is near zero, 1 if it is near pi and 2 else."""
    t = threshold
    a = a % (2 * jnp.pi)
    return jnp.piecewise(a, [a < t, jnp.abs(a - 2 * jnp.pi) < t, jnp.abs(a - jnp.pi) < t], [0, 0, 1, 2])
    # if a < t or jnp.abs(a - 2 * jnp.pi) < t:
    #     return 0
    # elif jnp.abs(a - jnp.pi) < t:
    #     return 1
    # else:
    #     return 2


def count_cz(angles, threshold=0.2):
    """
    Args:
        angles: angles corresponding to cp gates.
        threshold: to judge if angle is close to 0 or pi or not.
    returns:
        The number of CZ gates in the circuit, omitting CP gates with angles below the threshold."""
    cz = vmap(lambda a: cz_value(a, threshold=threshold))(angles).sum()
    return int(cz)


def project_cp_angle(a, threshold=0.2):
    a = a % (2 * jnp.pi)
    if jnp.abs(a - jnp.pi) < threshold:
        return jnp.pi
    elif jnp.abs(a) < threshold or jnp.abs(a - 2 * jnp.pi) < threshold:
        return 0
    else:
        return a


def insert_params(params, insertion_params, insertion_indices, jax_numpy=True):
    """Replaces params array at positions specified by indices by insertion_params.
    Example: params=[0,1,2,3], insertion_params=[-1,-2,-4], indices=[0,2,4] gives [-1,  0, -2,  1, -4,  2,  3]
    params and insertion_params must be jnp.arrays, indices must be list."""

    total_params = len(params) + len(insertion_params)
    params_indices = [i for i in range(total_params) if i not in insertion_indices]
    if jax_numpy:
        res = jnp.zeros(total_params)

        res = res.at[jnp.array(params_indices)].set(params)
        res = res.at[jnp.array(insertion_indices)].set(insertion_params)
        return res
    else:
        res = np.zeros(total_params)
        res[params_indices] = params
        res[insertion_indices] = insertion_params
        return jnp.array(res)


def constrained_function(f, fixed_params, indices, jax_numpy=True):
    """Function with part of parameters fixed.

    Example f=f(x,y,z), fixed_params=[1,10], indices=[0,2] gives g(y)=f(1,y,10) """

    def cf(free_params):
        return f(insert_params(free_params, fixed_params, indices, jax_numpy=jax_numpy))

    return cf


def convert_cp_to_cz(anz, angles, threshold=0.2):
    """Takes cp ansatz and converts it to a cz/mixed cp-cz ansatz by rounding off angles in CP gates close to Id or CZ.

    Args:
        anz: ansatz with block type 'cz'.
        angles: all angles in the ansatz.
        threshold: threshold value for rounding.

    Returns: tuple (circ, u, angs)
         circ: circuit with CP gates below threshold projected to Id or CZ gates.
         Projected angles are not included as parameter anymore.
         u: unitary matrix of the new circuit.
         free_angles: original angles except projected angles. Typically len(free_angles) < len(angles) .
    """

    mask = anz.cp_mask
    cp_indices = jnp.where(mask == 1)[0]

    # print('old version', angles[mask == 1])
    cp_angles = angles[jnp.where(mask == 1)]

    projected_cp_angles = jnp.array([project_cp_angle(a, threshold) for a in cp_angles])
    projected_mask = (projected_cp_angles == 0) + (projected_cp_angles == jnp.pi)
    projected_cp_angles = projected_cp_angles[projected_mask]
    projected_indices = [int(i) for i in cp_indices[projected_mask]]

    free_angles = jnp.array([a for i, a in enumerate(angles) if i not in projected_indices])

    return [constrained_function(anz.circuit, projected_cp_angles, projected_indices),
            constrained_function(anz.unitary, projected_cp_angles, projected_indices),
            free_angles]


def evaluate_cp_result(res, cp_mask, threshold=0.2):
    """Find best cz count, fidelity and angles from learning history.

    Args:
        res: dict with histories of regularized loss 'regloss', discrepancy 'loss', angles 'params'.
        cp_mask: mask specifying cp angles in the parameters.
        threshold: threshold value for projecting cp angles.

    Returns: (cz, loss, angles)
        cz: number of cz gates in the projected circuit at the lowest value of 'regloss'.
        loss: discrepancy at the lowest value of 'regloss'.
        angles: angles at the lowest value of 'regloss'.
    """

    best_i = jnp.argmin(res['regloss'])

    loss = res['loss'][best_i]
    angles = res['params'][best_i]
    cz = count_cz(angles * cp_mask, threshold=threshold)

    return cz, loss, angles


def filter_cp_results(
        res_list,
        cp_mask,
        threshold_cz_count,
        threshold_loss,
        threshold_cp=0.2,
        disable_tqdm=False):

    """ Select learning histories that have cz count and discrepancy below threshold values.

    Args:
        res_list: list of learning results.
        cp_mask: mask specifying cp angles in the ansatz.
        threshold_cz_count: max number of cz gates to accept.
        threshold_loss: max discrepancy with the target unitary to accept.
        threshold_cp: threshold value for projecting cp angles.
        disable_tqdm: whether display progress bar or not.
    Returns: OUTDATED
        list of tuples with data for selected results (cz, loss, i):
        cz: number of cz gates in the result.
        loss: discrepancy of the result.
        i: index of the result. res_list[i] is the result for which cz and loss are computed.
    """

    selected_results = []
    # for i, res in tqdm(enumerate(res_list), disable=disable_tqdm):
    for i, res in enumerate(res_list):
        cz, loss, angles = evaluate_cp_result(res, cp_mask, threshold=threshold_cp)
        cz_success = cz <= threshold_cz_count
        loss_success = loss <= threshold_loss
        if cz_success and loss_success:
            selected_results.append([cz, res])

    selected_results.sort(key=lambda x: x[0])

    return selected_results


def verify_cp_result(res, anz, unitary_loss_func, options, keep_history=False):
    """ Takes a cp ansatz, projects it to cz/mixed ansatz and verifies if nearly-exact compilation is possible.

    Args:
        res: dict specifying learning history.
        u_target: target unitary.
        anz: cp ansatz that produced the learning history.
        disc_func: which discrepancy function to use in learning.
        target_loss: loss that a projected ansatz should achieve.
        threshold: threshold for projection of cp angles.
    Returns: (success, circ, u, best_angless):
        success: whether threshold loss is achieved or not.
        circ: circuit of the projected ansatz.
        u: unitary of the projected ansatz.
        best_angles: best angles learned by projected ansatz.

    Note: it would be better to return some ansatz instead of separate circ/unitary,
    but mixed ansatz is not yet implemented.
    """

    num_cz_gates, loss, angles = evaluate_cp_result(res, anz.cp_mask, threshold=options.threshold_cp)
    circ, u, free_angles = convert_cp_to_cz(anz, angles, threshold=options.threshold_cp)

    refined_result = mynimize(
        lambda angs: unitary_loss_func(u(angs)),
        anz.num_angles,
        method=options.method,
        learning_rate=options.learning_rate_at_verification,
        num_iterations=options.num_gd_iterations_at_verification,
        u_func=anz.unitary,
        keep_history=keep_history,
        initial_params=free_angles
    )

    angles_history, loss_history = refined_result
    best_i = jnp.argmin(loss_history)
    best_angs = angles_history[best_i]
    best_loss = loss_history[best_i]

    if not keep_history:
        return best_loss <= options.target_loss, num_cz_gates, circ, u, best_angs
    else:
        return best_loss <= options.target_loss, num_cz_gates, circ, u, best_angs, angles_history, loss_history
