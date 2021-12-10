"""Module in infancy state."""

from jax import lax, vmap
import jax.numpy as jnp


def min_angle(F):
    """Maximum angle of a periodic function F = F_0 cos x + F_1 sin x+F_const"""

    # Need to fix numerical instability near a =0 !

    F_0 = F(0)
    F_1 = F(jnp.pi / 2)
    F_2 = F(jnp.pi)

    F_const = (F_0 + F_2) / 2
    a = F_0 - F_const
    b = F_1 - F_const

    res = lax.cond(a == 0,
                   lambda _: -jnp.pi/2 * jnp.sign(b),
                   lambda _: jnp.arctan(b / a) + jnp.pi * jnp.heaviside(a, 0.5),
                   operand=None)

    return res


def min_angles(F, angles, s0, s1):
    def one_min_angle(i):
        return min_angle(lambda a: F(angles.at[i].set(a)))

    return vmap(one_min_angle)(jnp.arange(s0, s1))