import jax.numpy as jnp


# Matrix representations of CNOT, CZ and single-qubit rotations.

cx_mat = jnp.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 0]])

cz_mat = jnp.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, -1]])

x_mat = jnp.array([[0, 1],
                   [1, 0]])

y_mat = jnp.array([[0, -1j],
                   [1j, 0]], dtype=jnp.complex64)

z_mat = jnp.array([[1, 0],
                   [0, -1]])


def rx_mat(a):
    return jnp.cos(a / 2) * jnp.identity(2) - 1j * x_mat * jnp.sin(a / 2)


def ry_mat(a):
    return jnp.cos(a / 2) * jnp.identity(2) - 1j * y_mat * jnp.sin(a / 2)


def rz_mat(a):
    return jnp.cos(a / 2) * jnp.identity(2) - 1j * z_mat * jnp.sin(a / 2)
