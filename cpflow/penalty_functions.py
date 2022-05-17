"""Penalty function and regularization."""

import jax.numpy as jnp
from cpflow.trigonometric_utils import bracket_angle


def cp_penalty_trig(a, height):
    """Old version of penalty function. Probably not helpful."""
    h = height
    res = (1-2*h)*jnp.cos(2*a)-2*jnp.cos(a)+1+2*h
    res = res/4
    return res


def line(x, x0, y0, x1, y1):
    return (y1-y0)/(x1-x0)*x+(x0*y1-x1*y0)/(x0-x1)


def cp_penalty_linear(a, xmax, ymax, plato_0, plato_1, plato_2):
    a = a % (2 * jnp.pi)

    def left(a):
        """Piecewise linear penalty function"""

        segments = [a <= plato_0,
                    (plato_0 < a) & (a <= xmax - plato_2),
                    (xmax - plato_2 < a) & (a <= xmax + plato_2),
                    (xmax + plato_2 < a) & (a <= jnp.pi - plato_1),
                    (jnp.pi - plato_1 < a) & (a <= jnp.pi),
                    jnp.pi < a  # Workaround for bug with vmap
                    ]

        functions = [line(a, 0, 0, plato_0, 0),
                     line(a, plato_0, 0, xmax - plato_2, ymax),
                     line(a, xmax - plato_2, ymax, xmax + plato_2, ymax),
                     line(a, xmax + plato_2, ymax, jnp.pi - plato_1, 1),
                     line(a, jnp.pi - plato_1, 1, jnp.pi, 1),
                     1]

        return jnp.piecewise(a, segments, functions)

    return left(a)


def penalty_linear(a, xmax, ymax, plato_0, plato_1, plato_2):
    a = a % (2 * jnp.pi)

    segments = [a <= plato_0,
                (plato_0 < a) & (a <= (xmax - plato_2)),
                ((xmax - plato_2) < a) & (a <= (xmax + plato_2)),
                ((xmax + plato_2) < a) & (a <= (jnp.pi - plato_1)),
                ((jnp.pi - plato_1) < a) & (a <= (jnp.pi+plato_1)),
                ((jnp.pi+plato_1) < a) & (a <= (jnp.pi+xmax-plato_2)),
                ((jnp.pi+xmax-plato_2) < a) & (a <= (jnp.pi + xmax + plato_2)),
                ((jnp.pi + xmax + plato_2) < a) & (a <= (2 * jnp.pi - plato_0)),
                ((2*jnp.pi-plato_0) < a) & (a <= 2*jnp.pi),
                (2*jnp.pi < a) & (a <= 3*jnp.pi)  # Workaround for bug with vmap
                ]

    functions = [line(a, 0, 0, plato_0, 0),
                 line(a, plato_0, 0, xmax - plato_2, ymax),
                 line(a, xmax - plato_2, ymax, xmax + plato_2, ymax),
                 line(a, xmax + plato_2, ymax, jnp.pi - plato_1, 1),
                 line(a, jnp.pi - plato_1, 1, jnp.pi+plato_1, 1),
                 line(a, jnp.pi+plato_1, 1, jnp.pi+xmax-plato_2, ymax),
                 line(a, jnp.pi + xmax - plato_2, ymax, jnp.pi + xmax + plato_2, ymax),
                 line(a, jnp.pi + xmax + plato_2, ymax, 2*jnp.pi - plato_0, 0),
                 line(a, 2 * jnp.pi - plato_0, 0, 2*jnp.pi, 0),
                 1,
                 ]

    return jnp.piecewise(a, segments, functions)


def penalty_L1(a):
    """L1 penalty"""
    a = bracket_angle(a)
    return jnp.abs(a)


def default_cp_regularization_function(a):
    ymax = 2
    xmax = jnp.pi / 2
    plato_0 = 0.05
    plato_1 = 0.05
    plato_2 = 0.05

    return cp_penalty_linear(a, xmax, ymax, plato_0, plato_1, plato_2)





