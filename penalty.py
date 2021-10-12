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


@partial(vmap, in_axes=(0, None, None))
def cp_penalty_linear(a, height, threshold):
    """Piecewise linear penalty function"""
    a = a % (2*jnp.pi)
    at = a_t(threshold)
    segments = [a < at, (at < a) & (a < jnp.pi),
                (jnp.pi < a) & (a < 2*jnp.pi-at),
                (2*jnp.pi-at < a) & (a < 2*jnp.pi)]
    functions = [line(a, 0, 0, at, height),
                 line(a, at, height, jnp.pi, 1),
                 line(a, jnp.pi, 1, 2*jnp.pi-at, height),
                 line(a, 2*jnp.pi-at, height, 2*jnp.pi, 0)]

    return jnp.piecewise(a, segments, functions)


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

### To get a feel for what a_t is doing run:

# t=1e-1
# asweep = jnp.linspace(0, 2*jnp.pi, 100)
# asweep_t = asweep[jnp.abs(asweep)<a_t(t)]
#
# plt.plot(asweep, [disc(cp_mat(a), jnp.identity(4)) for a in asweep])
# plt.plot(asweep, [t for _ in asweep])
#
# plt.plot(asweep_t,[disc(cp_mat(a), jnp.identity(4)) for a in asweep_t], 'bo')

# t=1e-1
# asweep = jnp.linspace(0, 2*jnp.pi, 100)
# asweep_t = asweep[jnp.abs(asweep-jnp.pi)<a_t(t)]
#
# plt.plot(asweep, [disc(cp_mat(a).reshape(4,4), cp_mat(jnp.pi)) for a in asweep])
# plt.plot(asweep, [t for _ in asweep])
#
# plt.plot(asweep_t,[disc(cp_mat(a), cp_mat(jnp.pi)) for a in asweep_t], 'bo')