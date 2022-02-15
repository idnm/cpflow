import pickle
import time
import os

import dill
import numpy as np
import jax.numpy as jnp

from jax import random, value_and_grad, jit, lax, custom_jvp
import optax

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from functools import partial

from tqdm import tqdm

from cp_utils import random_cp_angles, filter_cp_results, refine_cp_result
from gates import *
from circuit_assemebly import *
from optimization import *
from penalty import *
from topology import fill_layers

from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from hyperopt.pyll import scope


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


def split_angles(angles, num_qubits, num_block_angles, layer_len=0, num_layers=0):

    surface_angles = angles[:3 * num_qubits].reshape(num_qubits, 3)
    block_angles = angles[3 * num_qubits:].reshape(-1, num_block_angles)
    if num_layers is None:
        layers_angles = []
    else:
        layers_angles = block_angles[:layer_len * num_layers].reshape(num_layers, layer_len, num_block_angles)
    free_block_angles = block_angles[layer_len * num_layers:]
    if num_block_angles == 5:  # CP blocks
        cp_angles = [b[-1] for b in block_angles]
    else:
        cp_angles = []

    return {'surface angles': surface_angles,
            'block angles': block_angles,
            'layers angles': layers_angles,
            'free block angles': free_block_angles,
            'cp angles': cp_angles}


def build_unitary(num_qubits, block_type, placements, angles):
    layer, num_layers = placements['layers']
    free_placements = placements['free']

    layer_depth = len(layer)
    num_block_angles = EntanglingBlock.num_angles(block_type)

    angles_dict = split_angles(angles, num_qubits, num_block_angles, len(layer), num_layers)

    surface_angles = angles_dict['surface angles']
    layers_angles = angles_dict['layers angles']
    free_block_angles = angles_dict['free block angles']

    u = jnp.identity(2 ** num_qubits).reshape([2] * num_qubits * 2)

    # Initial round of single-qubit gates
    for i, a in enumerate(surface_angles):
        gate = rz_mat(a[2]) @ rx_mat(a[1]) @ rz_mat(a[0])
        u = apply_gate_to_tensor(gate, u, [i])

    # Sequence of layers wrapped in fori_loop.
    layers_angles = layers_angles.reshape(num_layers, layer_depth, num_block_angles)

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
        self.num_blocks = len(self.all_placements)

        num_block_angles = EntanglingBlock.num_angles(block_type)
        self.num_angles = 3 * num_qubits + num_block_angles * len(self.all_placements)

        if self.block_type == 'cp':
            sample_angles = jnp.arange(self.num_angles)
            cp_angles = split_angles(sample_angles, self.num_qubits, 5)['cp angles']
            self.cp_mask = jnp.array([1 if a in cp_angles else 0 for a in sample_angles])

        self.unitary = lambda angles: build_unitary(self.num_qubits, self.block_type, self.placements, angles)

    def circuit(self, angles=None):
        if angles is None:
            angles = np.array([Parameter('a{}'.format(i)) for i in range(self.num_angles)])

        num_block_angles = EntanglingBlock.num_angles(self.block_type)
        angles_dict = split_angles(angles, self.num_qubits, num_block_angles, len(self.layer), self.num_layers)

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

    def learn(self,
              u_target,
              method='adam',
              learning_rate=0.1,
              target_loss=1e-7,
              keep_history=True,
              **kwargs):

        return unitary_learn(self.unitary,
                             u_target,
                             self.num_angles,
                             method=method,
                             learning_rate=learning_rate,
                             target_loss=target_loss,
                             keep_history=keep_history,
                             **kwargs)


def adaptive_decompose(u_target,
                       layer,
                       hyperopt_options,
                       disc_func=disc2,
                       regularization_options=None,
                       save_to=None,
                       overwrite_existing=False):
    if save_to is None:
        print("Warning: results will not be saved because save_to not provided.")
    else:
        if not os.path.exists(save_to):
            os.makedirs(save_to)

        trials_path = save_to + 'trials.pickle'
        decompositions_path = save_to + 'decompositions.pickle'

    # Number of qubits is the maximum index in the coupling map, plus 1.
    num_qubits = max([item for sublist in layer for item in sublist]) + 1

    default_regularizatoin_options = {
        'r': 0.00055,
        'function': 'linear',
        'ymax': 2,
        'xmax': jnp.pi / 2,
        'plato': 0.05
    }

    default_hyperopt_options = {
        'r_initial': 0.00055,
        'r_variance': 0.5,
        'cp_dist': 'uniform',
        'entry_loss': 1e-3,
        'target_loss': 1e-6,
        'threshold_cp': 0.2,
        'batch_size': 1000,
        'num_gd_iterations': 2000,
        'threshold_num_gates': theoretical_lower_bound(num_qubits),
        'accepted_num_gates': None,
        'max_num_cp_gates': None,
        'max_evals': 100,
        'stop_if_target_reached': True
    }

    # Update default options with those provided with the function call.
    if regularization_options:
        regularization_options = dict(default_regularizatoin_options, **regularization_options)
    else:
        regularization_options = default_regularizatoin_options

    hyperopt_options = dict(default_hyperopt_options, **hyperopt_options)
    if hyperopt_options['accepted_num_gates'] is None:
        raise Exception('Accepted number of gates must be provided.')
    if hyperopt_options['max_num_cp_gates'] is None:
        print(
            'Warning: max_num_cp_gates not provided. Defaulting to the theoretical lower bound. This is likely to significantly slow down the search.')
        hyperopt_options.update({'max_num_cp_gates': theoretical_lower_bound(num_qubits)})

    def objective(search_params):

        r, num_cp_gates = search_params

        anz = Ansatz(num_qubits, 'cp', placements=fill_layers(layer, num_cp_gates))

        regularization_options.update({'r': r, 'cp_mask': anz.cp_mask})

        # My attepmt to generate random seed from time stamp.
        angles_random_seed = int(float(str(time.time())[::-1]))
        key = random.PRNGKey(angles_random_seed)

        key, *subkeys = random.split(key, num=hyperopt_options['batch_size'] + 1)
        initial_angles_array = jnp.array(
            [random_cp_angles(anz.num_angles, anz.cp_mask, cp_dist=hyperopt_options['cp_dist'], key=k)
             for k in subkeys])

        raw_results = anz.learn(
            u_target,
            regularization_options=regularization_options,
            initial_angles=initial_angles_array,
            num_iterations=hyperopt_options['num_gd_iterations'],
            disc_func=disc_func,
            keep_history=False)

        successful_results = filter_cp_results(
            raw_results,
            anz.cp_mask,
            hyperopt_options['threshold_num_gates'],
            hyperopt_options['entry_loss'],
            threshold_cp=hyperopt_options['threshold_cp'],
            disable_tqdm=True,
        )

        cz_counts = [cz for cz, *_ in successful_results]

        score = (2 ** (-(jnp.array(cz_counts, dtype=jnp.float32) - hyperopt_options['target_num_gates']))).sum() / hyperopt_options['batch_size']

        #         current_best_cz = min(scoreboard.keys())
        #         better_than_existing_results = [raw_results[i] for cz, loss, i in successful_results if cz<current_best_cz]

        return {
            'loss': -score,
            'status': STATUS_OK,
            'angles_random_seed': angles_random_seed,
            'cz_counts': cz_counts,
            'num_gd_iterations': hyperopt_options['num_gd_iterations'],
            'entry_loss': hyperopt_options['entry_loss'],
            'threshold_cp': hyperopt_options['threshold_cp'],
            'attachments': {'decompositions': pickle.dumps(successful_results)}
        }

    space = [
        hp.lognormal('r', jnp.log(hyperopt_options['r_initial']), hyperopt_options['r_variance']),
        scope.int(
            hp.quniform('num_cp_gates', hyperopt_options['target_num_gates'], hyperopt_options['max_num_cp_gates'], 1))
    ]

    trials = Trials()
    decompositions = []
    if save_to and not overwrite_existing:
        try:
            with open(trials_path, 'rb') as f:
                trials = pickle.load(f)
            print('Found existing trials, resuming from here.')
        except FileNotFoundError:
            pass

        try:
            with open(decompositions_path, 'rb') as f:
                decompositions = dill.load(f)
        except FileNotFoundError:
            pass

    if decompositions:
        cz_list = [cz for cz, *_ in decompositions]
        best_cz = min(cz_list)
        scoreboard = {cz: cz_list.count(cz) for cz in set(cz_list)}
    else:
        scoreboard = {hyperopt_options['accepted_num_gates'] + 1: 0}

    for n in tqdm(range(hyperopt_options['max_evals'] // hyperopt_options['evals_between_verification']),
                  desc='Epochs'):

        print('\n')

        best = fmin(
            objective,
            space=space,
            algo=tpe.suggest,
            max_evals=hyperopt_options['evals_between_verification'] + len(trials.trials),
            trials=trials)

        if save_to:
            with open(trials_path, 'wb') as f:
                pickle.dump(trials, f)

        current_best_cz = min(scoreboard.keys())
        results_to_verify = []
        for trial in trials.trials[-hyperopt_options['evals_between_verification']:]:
            msg = trials.trial_attachments(trial)['decompositions']
            successful_results = pickle.loads(msg)
            num_cp_gates = int(trial['misc']['vals']['num_cp_gates'][0])
            prospective_results = [[num_cp_gates, r] for cz, r in successful_results if cz < current_best_cz]
            num_equivalent_results = sum([cz == current_best_cz for cz, r in successful_results])
            results_to_verify.extend(prospective_results)

        if len(results_to_verify):
            print(
                f'\nFound {len(results_to_verify)} decompositions potentially better than current best count {current_best_cz}, verifying...')
        else:
            print(
                f'\nFound no better decompositions. Found {num_equivalent_results} decompositions with the current best count {current_best_cz}.')

        for num_cp_gates, res in prospective_results:
            anz = Ansatz(num_qubits, 'cp', placements=fill_layers(layer, num_cp_gates))

            success, cz, circ, u, best_angs = refine_cp_result(res,
                                                               u_target,
                                                               anz,
                                                               target_loss=hyperopt_options['target_loss'],
                                                               disc_func=disc_func)

            if success:
                print(f'\nFound new decomposition with {cz} gates.\n')

                decompositions.append([cz, circ, u, best_angs])
                scoreboard.update({cz: 1})

                if save_to:
                    with open(decompositions_path, 'wb') as f:
                        dill.dump(decompositions, f)

                break
        else:
            if prospective_results:
                print('\nNo decomposition passed.\n')

        if hyperopt_options['stop_if_target_reached'] and min(scoreboard.keys()) <= hyperopt_options['target_num_gates']:
            print('\nTarget number of gates reached.')
            break

    return trials, decompositions
