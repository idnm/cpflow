"""Assembling circuits and unitaries from building blocks."""

from cpflow.gates import *
from cpflow.matrix_utils import cost_HST


def gate_transposition(placement):
    """Determine transposition associated with initial placement of gate."""

    position_index = [(placement[i], i) for i in range(len(placement))]
    position_index.sort()
    transposition = [i for _, i in position_index]
    return transposition


def transposition(n_qubits, placement):
    """Return a transposition that relabels tensor axes correctly.
    Example (from the figure above): n=6, placement=[1, 3] gives [2, 0, 3, 1, 4, 5].
    Twisted: n=6, placement=[3, 1] gives [2, 1, 3, 0, 4, 5]."""

    gate_width = len(placement)

    t = list(range(gate_width, n_qubits))

    for position, insertion in zip(sorted(placement), gate_transposition(placement)):
        t.insert(position, insertion)

    return t


def apply_gate_to_tensor(gate, tensor, placement):
    """Append `gate` to `tensor` along legs specified by `placement`. Transpose the output axes properly."""

    gate_width = int(len(gate.shape) / 2)
    tensor_width = int(len(tensor.shape) / 2)

    # contraction axes for `tensor` are input axes (=last half of all axes)
    gate_contraction_axes = list(range(gate_width, 2 * gate_width))

    contraction = jnp.tensordot(gate, tensor, axes=[gate_contraction_axes, placement])

    # input(=last half) indices are intact
    t = transposition(tensor_width, placement) + list(range(tensor_width, 2 * tensor_width))

    return jnp.transpose(contraction, axes=t)


def qiskit_circ_to_jax_unitary(qc):
    num_qubits = len(qc.qubits)

    qc_angles = [gate.params[0] for gate, _, _ in qc.data if gate.name in ['rx', 'ry', 'rz']]
    wires = [qregs[0]._index for gate, qregs, _ in qc.data if gate.name in ['rx', 'ry', 'rz']]

    def u(angles):

        u0 = jnp.identity(2 ** num_qubits).reshape([2] * (2 * num_qubits))
        i = 0
        for gate, qargs, cargs in qc.data:
            if gate.name == 'cz':
                qbit0, qbit1 = qargs
                u0 = apply_gate_to_tensor(cz_mat.reshape(2, 2, 2, 2), u0, [qbit0._index, qbit1._index])
            else:
                if gate.name == 'rx':
                    mat = rx_mat
                elif gate.name == 'rz':
                    mat = rz_mat
                elif gate.name == 'ry':
                    mat = ry_mat
                else:
                    raise TypeError(f"Gate `{gate.name}` not in ['rx', 'ry', 'rz'].")

                qbit = qargs[0]
                u0 = apply_gate_to_tensor(mat(angles[i]), u0, [qbit._index])
                i += 1

        return u0.reshape(2 ** num_qubits, 2 ** num_qubits)

    cost = cost_HST(u(qc_angles), Operator(qc.reverse_bits()).data)
    assert cost < 1e-5, f'Error in converting from qiskit to jax: HST distance {cost} too high.'

    return u, qc_angles, wires
