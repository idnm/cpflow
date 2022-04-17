"""Matrix manipulations."""

from itertools import permutations

import jax.numpy as jnp
from jax import jacfwd
from qiskit.circuit.library import Permutation
from qiskit.quantum_info import Operator


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


def cost_HST(u, u_target):
    """Discrepancy between two unitary matrices proportional to the trace product squared.

    Normalized so that disc(u,u)=0, disc(u,v)=1 when u,v are orthogonal in the trace product.
    """

    n = u_target.shape[0]
    return 1 - jnp.abs((u * u_target.conj()).sum()) ** 2 / n ** 2


def disc2_swap(u, u_target, num_qubits):

    p_matrices = permutation_matrices(num_qubits)

    return jnp.product(jnp.array([cost_HST(m @ u, u_target) for m in p_matrices]))


def permutation_matrices(n):
    return [Operator(Permutation(n, p)).data for p in permutations(list(range(n)))]


def fubini_study(u_func, x, relative_coeff=1):
    u = u_func(x)
    u_norm2 = jnp.abs(trace_prod(u, u))
    u_jac = jacfwd(u_func)(x)

    dudu = jnp.tensordot(u_jac, u_jac.conj(), axes=[[0, 1], [0, 1]])
    udu = jnp.tensordot(u_jac, u.conj(), axes=[[0, 1], [0, 1]])

    Gij = dudu/u_norm2 - relative_coeff*jnp.outer(udu.conj(), udu)/u_norm2**2
    gij = jnp.real(Gij)

    return gij


def reorder_wires(wires, num_qubits):
    """Example: wires = [1, 3], num_qubits = 5, returns [1, 3, 0, 2, 4]"""
    all_wires = list(range(num_qubits))
    new_wires_order = wires + [w for w in all_wires if w not in wires]
    return new_wires_order


def move_wires_up(u, num_qubits, wires):
    """Transpose wires in tensor so that those specified appear first (at the top)."""

    u = u.reshape([2] * (2 * num_qubits))
    transposition_input_legs = reorder_wires(wires, num_qubits)
    transposition_output_legs = [w + num_qubits for w in transposition_input_legs]

    transposition = transposition_input_legs + transposition_output_legs

    return jnp.transpose(u, axes=transposition).reshape(2 ** num_qubits, 2 ** num_qubits)


def shifting_matrix(n):
    """Matrix that shifts order of basis elements i.e. 0->1, 1->2, 2->0 and m=((0,1,0),(0,0,1),(1,0,0))."""
    m = jnp.zeros((n, n))
    for i in range(n):
        m = m.at[(i, (i + 1) % n)].set(1)
    return m


def shift_matrix(u):
    """For diagonal matrix this shifts the eigenvalues, example: diag(1,2,3,4) -> diag(2,3,4,1)."""
    k = u.shape[0]
    x = shifting_matrix(k)
    return x @ u @ jnp.linalg.inv(x)


def shift_block_diagonal_matrix(u, m):
    """For matrix which is a block diagonal matrix with equal sized blocks m x m this shifts the order of the blocks.

    To understand what the function is doing run

    u = jnp.array([[0,1,0,0,0,0],[2,3,0,0,0,0],[0,0,4,5,0,0],[0,0,6,7,0,0],[0,0,0,0,8,9],[0,0,0,0,10,11]])
    shift_block_diagonal_matrix(u, 2)

    """
    k = int(u.shape[0] / m)
    x = jnp.kron(shifting_matrix(k), jnp.identity(m))
    return x @ u @ jnp.linalg.inv(x)


def block_diagonal_split(u, num_qubits, n):
    """Splits matrix into block-diagonal with blocks of size n x n, shifted block diagonal and
    off-block-diagonal. To get an idea of what the function is doing run

    u  = jnp.arange(2**6).reshape(2**3,2**3)
    block_diagonal_split(u, 1)
    block_diagonal_split(u, 2)

    """
    identity_dim = num_qubits - n
    block_diagonal_mask = jnp.kron(jnp.identity(2 ** identity_dim), jnp.ones((2 ** n, 2 ** n)))
    block_diagonal_mask_complement = 1 - block_diagonal_mask

    u_diag = block_diagonal_mask * u
    u_off_diag = block_diagonal_mask_complement * u

    return u_diag, shift_block_diagonal_matrix(u_diag, 2 ** n), u_off_diag


def tensor_identity_loss_frobenius(u, num_qubits, wires):
    """If the tensor corresponding to matrix u applies no gates to qubits specified by `wires` the function returns 0.
    Otherwise it's positive. """
    u = move_wires_up(u, num_qubits, wires)

    block_size = num_qubits - len(wires)
    u_diag, u_diag_shifted, u_off_diag = block_diagonal_split(u, num_qubits, block_size)

    loss_off_diag = (jnp.abs(u_off_diag) ** 2).sum()
    loss_diag = (jnp.abs(u_diag - u_diag_shifted) ** 2).sum()

    return loss_diag + loss_off_diag


def tensor_identity_loss(u, num_qubits, wires):
    """If the tensor corresponding to matrix u applies no gates to qubits specified by `wires` the function returns 0.
    Otherwise it's positive.


    The idea behind this implementation is as follows. First we move all `wires` up, meaning that u should take the form
    I x V . That means, u is block diagonal with identical blocks equal to V. To test if this is the case we
    1) Split u into diagonal and off-diagonal blocks. If any off-diagonal are non-zero the factorization I x V does not hold.
    2) To check if diagonal blocks are all identical we can create a matrix u_diag_shifted where diagonal blocks are shifted,
    and take a row-wise multiplication with original u. Because u is unitary this row-wise multiplication will give
    maximum value only when all rows are equal to each other, implying that all diagonl blocks are also equal.

    """
    u = move_wires_up(u, num_qubits, wires)

    block_size = num_qubits - len(wires)
    u_diag, u_diag_shifted, u_off_diag = block_diagonal_split(u, num_qubits, block_size)

    scalar_product_matrix = u_diag * u_diag_shifted.conj()
    scalar_product_vector = scalar_product_matrix.sum(axis=1)
    scalar_product_total = jnp.abs(scalar_product_vector.sum())

    loss_off_diag = (jnp.abs(u_off_diag) ** 2).sum()
    loss_diag = 1 - scalar_product_total / 2 ** num_qubits

    return loss_diag + loss_off_diag


def tensor_diagonal_loss(u, num_qubits, wires):
    """If the tensor corresponding to matrix u only applies a diagonal gate qubits specified by `wires` the function returns 0.
    Otherwise it's positive.

    The implementation is similar to the tensor_identity_loss but instead of summing over all row-wise scalar products we sum
    all absolute values of row-wise scalar products. This accounts for possible phases introduced by the diagonal gate.
    """

    u = move_wires_up(u, num_qubits, wires)

    block_size = num_qubits - len(wires)
    u_diag, u_diag_shifted, u_off_diag = block_diagonal_split(u, num_qubits, block_size)

    loss_off_diag = (jnp.abs(u_off_diag) ** 2).sum()

    scalar_product_matrix = u_diag * u_diag_shifted.conj()
    scalar_product_vector = scalar_product_matrix.sum(axis=1)
    scalar_product_vector_up_to_phases = jnp.abs(scalar_product_vector)
    scalar_product_total = (scalar_product_vector_up_to_phases ** 2).sum()

    loss_diag = 1 - scalar_product_total / 2 ** num_qubits

    return loss_diag + loss_off_diag


def disc_modulo_identity(u_target, u, num_qubits, wires):
    """ Returns zero if `u` as quantum circuit is equivalent to `u_target` up to a transformation acting as identity on `wires`.
    Otherwise returns a positive number that quantifies the deviation from the scenario."""

    return tensor_identity_loss((u @ u_target).conj().T, num_qubits, wires)


def disc_modulo_diagonal(u_target, u, num_qubits, wires):
    """ Returns zero if `u` as quantum circuit is equivalent to `u_target` times a diagonal transofmation followed by
    arbitrary transormations not touching `wires`. Otherwise returns a positive number that quantifies the deviation from the scenario."""

    return tensor_diagonal_loss((u @ u_target).conj().T, num_qubits, wires)


# To get an idea of how tensor losses work try to run something like

# n = 6
#
# u_rnd = unitary_group.rvs(4, random_state=0)
# u_diag = jnp.diag(jnp.exp(1j * jnp.arange(1, 2 ** n + 1, dtype=jnp.complex64))).reshape([2] * (2 * n))
#
# u = jnp.identity(2 ** n).reshape([2] * (2 * n))
# u = apply_gate_to_tensor(u_diag, u, list(range(n)))
#
# u = apply_gate_to_tensor(u_rnd.reshape(2, 2, 2, 2), u, [2, 4])
# u = apply_gate_to_tensor(x_mat, u, [1]).reshape(2 ** n, 2 ** n)
#
# wires = [0, 3, 5]
# print(tensor_identity_loss_frobenius(u, wires))
# print(tensor_identity_loss(u, wires))
# print(tensor_diagonal_loss(u, wires))

