import numpy as np
import jax.numpy as jnp

from jax import random, value_and_grad, jit, lax
import optax

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from functools import partial

from gates import *


# Class for a single building block
class EntanglingBlock:
    def __init__(self, gate_name, angles):
        self.gate_name = gate_name
        self.angles = angles

    def circuit(self):

        angles = np.array(self.angles)  # convert from JAX array to numpy array if applicable.

        qc = QuantumCircuit(2)

        # Apply entangling gate
        if self.gate_name == 'cx':
            qc.cx(0, 1)
        elif self.gate_name == 'cz':
            qc.cz(0, 1)
        elif self.gate_name == 'cp':
            qc.cp(angles[4], 0, 1)
        else:
            print("Gate '{}' not yet supported'".format(self.gate_name))

        # Apply single-qubit gates.
        qc.ry(angles[0], 0)
        qc.rx(angles[1], 0)
        qc.ry(angles[2], 1)
        qc.rx(angles[3], 1)

        return qc

    def unitary(self):
        if self.gate_name == 'cx':
            entangling_matrix = cx_mat
        elif self.gate_name == 'cz':
            entangling_matrix = cz_mat
        elif self.gate_name == 'cp':
            entangling_matrix = cp_mat(self.angles[4])
        else:
            print("Gate '{}' not yet supported'".format(self.gate_name))

        x_rotations = jnp.kron(rx_mat(self.angles[1]), rx_mat(self.angles[3]))
        y_rotations = jnp.kron(ry_mat(self.angles[0]), ry_mat(self.angles[2]))

        return x_rotations @ y_rotations @ entangling_matrix


def disc(u, u_target):
    """Discrepancy between two unitary matrices."""
    n = u_target.shape[0]
    return 1 - jnp.abs((u.conj() * u_target).sum()) / n


@partial(jit, static_argnums=(0, 1,))
def unitary_update(loss_and_grad, opt, opt_state, angles):
    loss, grads = loss_and_grad(angles)
    updates, opt_state = opt.update(grads, opt_state)
    angles = optax.apply_updates(angles, updates)
    return angles, opt_state, loss


def unitary_learn(u_func, u_target, n_angles,
                  init_angles=None,
                  learning_rate=0.01, num_iterations=5000,
                  target_disc=1e-10):

    if init_angles is None:
        key = random.PRNGKey(0)
        angles = random.uniform(key, shape=(n_angles,), minval=0, maxval=2 * jnp.pi)
    else:
        angles = init_angles

    loss_func = lambda angles: disc(u_func(angles), u_target)

    loss_and_grad = value_and_grad(loss_func)

    opt = optax.adam(learning_rate)
    opt_state = opt.init(angles)

    def update(angles, opt_state):
        return unitary_update(loss_and_grad, opt, opt_state, angles)

    angles_history = []
    loss_history = []
    for _ in range(num_iterations):
        angles, opt_state, loss = update(angles, opt_state)
        angles_history.append(angles)
        loss_history.append(loss)
        if loss < target_disc:
            break

    return angles_history, loss_history


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


def split_angles(angles, num_qubits, layer_len, num_layers):
    surface_angles = angles[:3 * num_qubits].reshape(num_qubits, 3)
    block_angles = angles[3 * num_qubits:].reshape(-1, 4)
    layers_angles = block_angles[:layer_len * num_layers].reshape(num_layers, layer_len, 4)
    free_block_angles = block_angles[layer_len * num_layers:]

    return surface_angles, block_angles, layers_angles, free_block_angles


def build_unitary(num_qubits, block_type, angles, placements):

    layer, num_layers = placements['layers']
    free_placements = placements['free']

    layer_depth = len(layer)

    surface_angles, _, layers_angles, free_block_angles = split_angles(angles, num_qubits,
                                                                       len(layer),
                                                                       num_layers)

    u = jnp.identity(2 ** num_qubits).reshape([2] * num_qubits * 2)

    # Initial round of single-qubit gates
    for i, a in enumerate(surface_angles):
        gate = rz_mat(a[2]) @ rx_mat(a[1]) @ rz_mat(a[0])
        u = apply_gate_to_tensor(gate, u, [i])

    # Sequence of layers wrapped in fori_loop.
    layers_angles = layers_angles.reshape(num_layers, layer_depth, 4)

    def apply_layer(i, u, layer, layers_angles):
        angles = layers_angles[i]

        for a, p in zip(angles, layer):
            gate = EntanglingBlock(block_type, a).unitary().reshape(2, 2, 2, 2)
            u = apply_gate_to_tensor(gate, u, p)

        return u

    if num_layers > 0:
        u = lax.fori_loop(0, num_layers, lambda i, u: apply_layer(i, u, layer, layers_angles), u)

    # Add remainder(free) blocks.
    for a, p in zip(free_block_angles, free_placements):
        gate = EntanglingBlock(block_type, a).unitary().reshape(2, 2, 2, 2)
        u = apply_gate_to_tensor(gate, u, p)

    return u.reshape(2 ** num_qubits, 2 ** num_qubits)


class Ansatz:

    def __init__(self, num_qubits, block_type, placements):

        self.num_qubits = num_qubits
        self.block_type = block_type
        self.placements = placements

        self.layer, self.num_layers = placements['layers']
        self.free_placements = placements['free']
        self.all_placements = self.layer * self.num_layers + self.free_placements

        self.num_angles = 3 * num_qubits + 4 * len(self.all_placements)

        self.unitary = lambda angles: build_unitary(self.num_qubits, self.block_type, angles, self.placements)

    def circuit(self, angles=None):
        if angles is None:
            angles = np.array([Parameter('a{}'.format(i)) for i in range(self.num_angles)])

        surface_angles, block_angles, _, _ = split_angles(angles, self.num_qubits,
                                                          len(self.layer), self.num_layers)

        qc = QuantumCircuit(self.num_qubits)

        # Initial rounf of single-qubit gates
        for n, a in enumerate(surface_angles):
            qc.rz(a[0], n)
            qc.rx(a[1], n)
            qc.rz(a[2], n)

        # Entangling gates accoring to placements
        for a, p in zip(block_angles, self.all_placements):
            qc_block = EntanglingBlock(self.block_type, a).circuit()
            qc = qc.compose(qc_block, p)

        return qc

    def learn(self, u_target, **kwargs):
        u_func = self.unitary
        return unitary_learn(u_func, u_target, self.num_angles, **kwargs)


def sequ_layer(num_qubits):
    return [[i, j] for i in range(num_qubits) for j in range(i + 1, num_qubits)]


def fill_layers(layer, depth):
    num_complete_layers = depth // len(layer)
    complete_layers = [layer, num_complete_layers]
    incomplete_layer = layer[:depth % len(layer)]

    return {'layers': complete_layers, 'free': incomplete_layer}
