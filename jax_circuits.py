import numpy as np
import jax.numpy as jnp

from jax import random, value_and_grad, jit, lax, custom_jvp
import optax

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from functools import partial

from gates import *
from circuit_assemebly import *
from gd_optimization import *


class EntanglingBlock:
    """Two-qubit entangling block.

    Methods:
        circuit: gives an equivalent `qiskit` circuit.
        unitary: gives `jax.numpy` unitary matrix of the circuit.
        num_angles: total number of angles (parameters) in a block.
    """

    def __init__(self, gate_name, angles):
        self.gate_name = gate_name
        self.angles = angles

    @staticmethod
    def num_angles(gate_name):
        if gate_name == 'cp':
            return 5
        else:
            return 4

    def circuit(self):
        """Quantum circuit in `qiskit` corresponding to our block."""

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
        """2x2 unitary of the block."""

        if self.gate_name == 'cx':
            entangling_matrix = cx_mat
        elif self.gate_name == 'cz':
            entangling_matrix = cz_mat
        elif self.gate_name == 'cp':
            entangling_matrix = cp_mat(self.angles[4])
        else:
            raise Exception("Gate '{}' not yet supported'".format(self.gate_name))

        x_rotations = jnp.kron(rx_mat(self.angles[1]), rx_mat(self.angles[3]))
        y_rotations = jnp.kron(ry_mat(self.angles[0]), ry_mat(self.angles[2]))

        return x_rotations @ y_rotations @ entangling_matrix


def split_angles(angles, n_qubits, n_block_angles, layer_len, n_layers):

    surface_angles = angles[:3 * n_qubits].reshape(n_qubits, 3)
    block_angles = angles[3 * n_qubits:].reshape(-1, n_block_angles)
    layers_angles = block_angles[:layer_len * n_layers].reshape(n_layers, layer_len, n_block_angles)
    free_block_angles = block_angles[layer_len * n_layers:]

    return {'surface angles': surface_angles,
            'block angles': block_angles,
            'layers angles': layers_angles,
            'free block angles': free_block_angles}


def build_unitary(n_qubits, block_type, placements, angles):
    layer, n_layers = placements['layers']
    free_placements = placements['free']

    layer_depth = len(layer)
    n_block_angles = EntanglingBlock.num_angles(block_type)

    angles_dict = split_angles(angles, n_qubits, n_block_angles, len(layer), n_layers)

    surface_angles = angles_dict['surface angles']
    layers_angles = angles_dict['layers angles']
    free_block_angles = angles_dict['free block angles']

    u = jnp.identity(2 ** n_qubits).reshape([2] * n_qubits * 2)

    # Initial round of single-qubit gates
    for i, a in enumerate(surface_angles):
        gate = rz_mat(a[2]) @ rx_mat(a[1]) @ rz_mat(a[0])
        u = apply_gate_to_tensor(gate, u, [i])

    # Sequence of layers wrapped in fori_loop.
    layers_angles = layers_angles.reshape(n_layers, layer_depth, n_block_angles)

    def apply_layer(i, u, layer, layers_angles):
        angles = layers_angles[i]

        for a, p in zip(angles, layer):
            gate = EntanglingBlock(block_type, a).unitary().reshape(2, 2, 2, 2)
            u = apply_gate_to_tensor(gate, u, p)

        return u

    if n_layers > 0:
        u = lax.fori_loop(0, n_layers, lambda i, u: apply_layer(i, u, layer, layers_angles), u)

    # Add remainder(free) blocks.
    for a, p in zip(free_block_angles, free_placements):
        gate = EntanglingBlock(block_type, a).unitary().reshape(2, 2, 2, 2)
        u = apply_gate_to_tensor(gate, u, p)

    return u.reshape(2 ** n_qubits, 2 ** n_qubits)


class Ansatz:

    def __init__(self, n_qubits, block_type, placements):

        self.n_qubits = n_qubits
        self.block_type = block_type

        placements.setdefault('layers', [[], 0])
        placements.setdefault('free', [])
        self.placements = placements

        self.layer, self.n_layers = placements['layers']
        self.free_placements = placements['free']

        self.all_placements = self.layer * self.n_layers + self.free_placements

        n_block_angles = EntanglingBlock.num_angles(block_type)
        self.n_angles = 3 * n_qubits + n_block_angles * len(self.all_placements)

        self.unitary = lambda angles: build_unitary(self.n_qubits, self.block_type, self.placements, angles)

    def circuit(self, angles=None):
        if angles is None:
            angles = np.array([Parameter('a{}'.format(i)) for i in range(self.n_angles)])

        num_block_angles = EntanglingBlock.num_angles(self.block_type)
        angles_dict = split_angles(angles, self.n_qubits, num_block_angles, len(self.layer), self.n_layers)

        surface_angles = angles_dict['surface angles']
        block_angles = angles_dict['block angles']

        qc = QuantumCircuit(self.n_qubits)

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
        return gradient_descent_learn(u_func, u_target, self.n_angles, **kwargs)


# def learn_disc(u_func, u_target, n_angles, n_iterations=100, n_evaluations=10):
#     @jit
#     def u_disc(angles):
#         return disc2(u_func(angles), u_target)
#
#     def one_estimation(k):
#         initial_angles = random.uniform(random.PRNGKey(k), shape=(n_angles, ), minval=0, maxval=2*jnp.pi)
#         angles = staircase_min(u_disc, n_angles, initial_angles=initial_angles, n_iterations=n_iterations)
#         return angles
#
#     estimations = vmap(one_estimation)(jnp.arange(n_evaluations))
#     discs = vmap(u_disc)(estimations)
#
#     min_disc = jnp.min(discs)
#     minimizing_angles = estimations[jnp.argmin(discs)]
#
#     return minimizing_angles, min_disc
#
#
# def unitary_fitness(u_func, u_target, n_angles, n_gates, **kwargs):
#     _, disc = learn_disc(u_func, u_target, n_angles, **kwargs)
#     return 1-disc + 1/(n_gates+1)
#

# @partial(jit, static_argnums=(1, 2, 3, ))
# def update_angles(angles, f, n_angles, n_moving_angles):
#     s = jnp.array(splits(n_angles, n_moving_angles))
#
#     def body(i, angles):
#         return partial_update_angles(f, angles, s[i], n_moving_angles)
#
#     for i in range(len(s)):
#         angles = body(i, angles)
#
#     return angles
#
#
# def staircase_min_batch(f, n_angles, initial_angles=None, n_iterations=100, n_moving_angles=1):
#     if initial_angles is None:
#         initial_angles = random.uniform(random.PRNGKey(0), minval=0, maxval=2 * jnp.pi, shape=(n_angles,))
#     angles = initial_angles
#     angles_history = [angles]
#
#     for _ in range(n_iterations):
#         angles = update_angles(angles, f, n_angles, n_moving_angles)
#         angles_history.append(angles)
#     return angles_history
