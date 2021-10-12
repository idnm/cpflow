import numpy as np
import jax.numpy as jnp

from jax import random, value_and_grad, jit, lax
import optax

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from functools import partial

from gates import *
from penalty import *


# Class for a single building block
class EntanglingBlock:
    def __init__(self, gate_name, angles):
        self.gate_name = gate_name
        self.angles = angles

    @staticmethod
    def n_angles(gate_name):
        if gate_name == 'cp':
            return 5
        else:
            return 4

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
                  penalty_options=None,
                  learning_rate=0.01, num_iterations=5000,
                  target_disc=1e-10):

    if init_angles is None:
        key = random.PRNGKey(0)
        angles = random.uniform(key, shape=(n_angles,), minval=0, maxval=2 * jnp.pi)
    else:
        angles = init_angles

    def loss_func(angles):
        loss0 = disc(u_func(angles), u_target)
        reg = 0
        if penalty_options is not None:
            reg = penalty_func(penalty_options)(angles)
        return loss0 + reg

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


def split_angles(angles, num_qubits, block_type, layer_len, num_layers):

    n_block_angles = EntanglingBlock.n_angles(block_type)

    surface_angles = angles[:3 * num_qubits].reshape(num_qubits, 3)
    block_angles = angles[3 * num_qubits:].reshape(-1, n_block_angles)
    layers_angles = block_angles[:layer_len * num_layers].reshape(num_layers, layer_len, n_block_angles)
    free_block_angles = block_angles[layer_len * num_layers:]

    return {'surface angles': surface_angles,
            'block angles': block_angles,
            'layers angles': layers_angles,
            'free block angles': free_block_angles}


def control_angles(angles, num_qubits, block_type):
    assert block_type == 'cp', 'other block types not supported'
    n_block_angles = EntanglingBlock.n_angles('cp')
    block_angles = angles[3 * num_qubits:].reshape(-1, n_block_angles)
    # Last angle in each block is the control angle
    return [a[-1] for a in block_angles]


def build_unitary(num_qubits, block_type, placements, angles):

    layer, num_layers = placements['layers']
    free_placements = placements['free']

    layer_depth = len(layer)

    angles_dict = split_angles(angles, num_qubits, block_type, len(layer), num_layers)

    surface_angles = angles_dict['surface angles']
    layers_angles = angles_dict['layers angles']
    free_block_angles = angles_dict['free block angles']

    u = jnp.identity(2 ** num_qubits).reshape([2] * num_qubits * 2)

    # Initial round of single-qubit gates
    for i, a in enumerate(surface_angles):
        gate = rz_mat(a[2]) @ rx_mat(a[1]) @ rz_mat(a[0])
        u = apply_gate_to_tensor(gate, u, [i])

    # Sequence of layers wrapped in fori_loop.
    layers_angles = layers_angles.reshape(num_layers, layer_depth, EntanglingBlock.n_angles(block_type))

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

        placements.setdefault('layers', [[], 0])
        placements.setdefault('free', [])
        self.placements = placements

        self.layer, self.num_layers = placements['layers']
        self.free_placements = placements['free']

        self.all_placements = self.layer * self.num_layers + self.free_placements

        n_block_angles = EntanglingBlock.n_angles(block_type)
        self.num_angles = 3 * num_qubits + n_block_angles * len(self.all_placements)

        self.unitary = lambda angles: build_unitary(self.num_qubits, self.block_type, self.placements, angles)

    def circuit(self, angles):
        if angles is None:
            angles = np.array([Parameter('a{}'.format(i)) for i in range(self.num_angles)])
        angles_dict = split_angles(angles, self.num_qubits, self.block_type, len(self.layer), self.num_layers)

        surface_angles = angles_dict['surface angles']
        block_angles = angles_dict['block angles']

        qc = QuantumCircuit(self.num_qubits)

        # Initial round of single-qubit gates
        for n, a in enumerate(surface_angles):
            qc.rz(a[0], n)
            qc.rx(a[1], n)
            qc.rz(a[2], n)

        # Entangling gates according to placements
        for a, p in zip(block_angles, self.all_placements):
            qc_block = EntanglingBlock(self.block_type, a).circuit()
            qc = qc.compose(qc_block, p)

        return qc

    def learn(self, u_target, **kwargs):
        u_func = self.unitary
        return unitary_learn(u_func, u_target, self.num_angles, **kwargs)

