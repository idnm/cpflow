import jax.numpy as jnp
from jax import vmap
from functools import partial


def a_t(t):
    """Maximum angle `a_t` for which discrepancy between cp(a) gate and Identity matrix
    is below threshold `t` """
    return jnp.arccos((8*(t**2)-16*t+3)/3)


def cp_penalty_trig(a, height):
    """Old version of penalty function. Probably not helpful."""
    h = height
    res = (1-2*h)*jnp.cos(2*a)-2*jnp.cos(a)+1+2*h
    res = res/4
    return res


def line(x, x0, y0, x1, y1):
    return (y1-y0)/(x1-x0)*x+(x0*y1-x1*y0)/(x0-x1)


@partial(vmap, in_axes=(0, None, None, None))
def cp_penalty_linear(a, ymax, xmax, plato):
    """Piecewise linear penalty function"""
    a = a % (2 * jnp.pi)

    segments = [a <= plato,
                (plato < a) & (a <= xmax),
                (xmax < a) & (a <= jnp.pi - plato),
                (jnp.pi - plato < a) & (a <= jnp.pi + plato),
                (jnp.pi + plato < a) & (a <= 2 * jnp.pi - xmax),
                (2 * jnp.pi - xmax < a) & (a <= 2 * jnp.pi - plato),
                (2 * jnp.pi - plato < a) & (a <= 2 * jnp.pi)
                ]

    functions = [line(a, 0, 0, plato, 0),
                 line(a, plato, 0, xmax, ymax),
                 line(a, xmax, ymax, jnp.pi - plato, 1),
                 line(a, jnp.pi - plato, 1, jnp.pi + plato, 1),
                 line(a, jnp.pi + plato, 1, 2 * jnp.pi - xmax, ymax),
                 line(a, 2 * jnp.pi - xmax, ymax, 2 * jnp.pi - plato, 0),
                 line(a, 2 * jnp.pi - plato, 0, 0, 0)
                 ]

    return jnp.piecewise(a, segments, functions)


def cz_value(a, threshold=1e-1):
    """Return 0 if CP-angle is near zero, 1 if it is near pi and 2 else."""
    t = threshold
    a = a % (2*jnp.pi)
    return jnp.piecewise(a,
                         [a < t, jnp.abs(a-2*jnp.pi) < t, jnp.abs(a-jnp.pi) < t],
                         [0, 0, 1, 2])


def count_cz(angles, threshold=1e-1):
    """Count the number of CZ gate in the circuit, omitting CP gates with angles below the threshold."""
    return sum([cz_value(a, threshold=threshold) for a in angles])


@vmap
def cp_penalty_L1(a):
    """L1 penalty"""
    return jnp.abs(a)


def penalty(angles, options):
    # array of 0 and 1 specifying which angles are angles of cp gates and must be penalized.
    penalized_angles = jnp.array(options['angles'])
    penalty_function = options['function']
    h = options['height']  # Height of the regularization function.
    reg = options['regularization']  # How much weight to put on regularization term.

    if penalty_function == 'trig':
        return reg * cp_penalty_trig(angles * penalized_angles, h).sum()
    elif penalty_function == 'linear':
        t = options['threshold']
        return reg * cp_penalty_linear(angles * penalized_angles, h, t).sum()


def construct_penalty_function(penalty_options):
    cp_mask = penalty_options['cp_mask']
    r = penalty_options['r']

    if penalty_options['function'] == 'linear':
        ymax = penalty_options['ymax']
        xmax = penalty_options['xmax']
        plato = penalty_options['plato']
        # target_num_gates = penalty_options['num_gates']
        # angle_tolerance = penalty_options['angle_tolerance']

        penalty_func = lambda angs: r * cp_penalty_linear(angs*cp_mask, ymax, xmax, plato).sum()

        # tolerated_identity_penalty = penalty_func(angle_tolerance * cp_mask)/sum(cp_mask)
        # tolerated_cz_penalty = penalty_func((jnp.pi+angle_tolerance) * cp_mask)/sum(cp_mask)
        #
        # target_reg = target_num_gates*tolerated_cz_penalty+(sum(cp_mask)-target_num_gates)*tolerated_identity_penalty

    elif penalty_options['function'] == 'L1':
        penalty_func = lambda angs: r * cp_penalty_L1(angs*cp_mask).sum()
        # target_reg = None

    else:
        print('penalty function not supported')
        print(penalty_options['func'])

    return penalty_func
