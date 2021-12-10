import jax.numpy as jnp


def transposition(n, placement):
    w = len(placement)
    t = list(range(w, n))

    for i, p in enumerate(placement):
        t.insert(p, i)

    return t


def apply_gate_to_tensor(gate, tensor, placement):
    gate_width = int(len(gate.shape) / 2)
    tensor_width = int(len(tensor.shape) / 2)
    gate_contraction_axes = list(range(gate_width, 2 * gate_width))

    contraction = jnp.tensordot(gate, tensor, axes=[gate_contraction_axes, placement])
    t = transposition(tensor_width, placement) + list(range(tensor_width, 2 * tensor_width))  # last indices are intact

    return jnp.transpose(contraction, axes=t)

