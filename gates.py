""""Matrix representations of some quantum gates."""

import jax.numpy as jnp
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

# Single-qubit pauli gates.

x_mat = jnp.array([[0, 1],
                   [1, 0]])

y_mat = jnp.array([[0, -1j],
                   [1j, 0]], dtype=jnp.complex64)

z_mat = jnp.array([[1, 0],
                   [0, -1]])


# Single-qubit rotation gates.

def rotation_matrix(mat, a):
    return jnp.cos(a / 2) * jnp.identity(2) - 1j * mat * jnp.sin(a / 2)


def rx_mat(a):
    return rotation_matrix(x_mat, a)


def ry_mat(a):
    return rotation_matrix(y_mat, a)


def rz_mat(a):
    return rotation_matrix(z_mat, a)

# Two-qubit gates.


cx_mat = jnp.array([[1, 0, 0, 0],  # cx is the same as CNOT
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 0]])

cz_mat = jnp.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, -1]])


def cp_mat(a):
    """Controlled-phase gate. For a=0, 2pi it is identity, for a=pi it is CZ."""

    phase_gate = jnp.array([[1, 0], [0, jnp.exp(1j*a)]])
    control0 = jnp.kron(jnp.array([[1, 0], [0, 0]]), jnp.identity(2))
    control1 = jnp.kron(jnp.array([[0, 0], [0, 1]]), phase_gate)

    return control0+control1


# Toffoli gates from qiskit
qc = QuantumCircuit(3)
qc.ccx(0, 1, 2)
u_toff3 = Operator(qc.reverse_bits()).data

qc = QuantumCircuit(4)
qc.mct([0, 1, 2], 3)
u_toff4 = Operator(qc.reverse_bits()).data

qc = QuantumCircuit(5)
qc.mct([0, 1, 2, 3], 4)
u_toff5 = Operator(qc.reverse_bits()).data
