import jax.numpy as jnp


def theoretical_lower_bound(n):
    """ Minimum number of CNOT gates to decompose arbitrary n-qubit unitary."""

    return int((4**n-3*n-1)/4 + 1)


def trace_prod(u, v):
    """Scalar product of two matrices defined by Tr(U^dagger, V)."""

    # Note that computing full matrix product is unnecessary.
    # Element-wise product is sufficient.

    return (u.conj() * v).sum()


def disc(u, u_target):
    """A measure of discrepancy between two unitary matrices proportional to the trace product.

    Normalized so that disc(u,u)=0, disc(u,v)=1 when u,v are orthogonal in the trace product.
    """
    n = u_target.shape[0]
    return 1 - jnp.abs(trace_prod(u, u_target)) / n


def disc2(u, u_target):
    """Discrepancy between two unitary matrices proportional to the trace product squared.

    Normalized so that disc(u,u)=0, disc(u,v)=1 when u,v are orthogonal in the trace product.
    """

    n = u_target.shape[0]
    return 1 - jnp.abs((u * u_target.conj()).sum()) ** 2 / n ** 2

