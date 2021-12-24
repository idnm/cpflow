import jax.numpy as jnp


def gate_transposition(placement):
    """Determine transposition associated with initial placement of gate."""

    position_index = [(placement[i], i) for i in range(len(placement))]
    position_index.sort()
    transposition = [i for _, i in position_index]
    return transposition


def transposition(n_qubits, placement):
    """Return a transposition that relabels tensor axes correctly.
    Example (from the figure above): n=6, placement=[1, 3] gives [2, 0, 3, 1, 4, 5].
    Twiseted: n=6, placement=[3, 1] gives [2, 1, 3, 0, 4, 5]."""

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

