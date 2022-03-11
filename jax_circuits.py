import pickle
import time
import os
from pprint import pprint
from dataclasses import dataclass, asdict

import dill
import numpy as np
import matplotlib.pyplot as plt
from qiskit.circuit import Parameter

from tqdm import tqdm

from cp_utils import random_cp_angles, filter_cp_results, refine_cp_result, verify_cp_result
from gates import *
from circuit_assemebly import *
from optimization import *
from penalty import *
from topology import *

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
            parametrized = True
        else:
            parametrized = False

        num_block_angles = EntanglingBlock.num_angles(self.block_type)
        angles_dict = split_angles(angles, self.num_qubits, num_block_angles, len(self.layer), self.num_layers)

        surface_angles = angles_dict['surface angles']
        block_angles = angles_dict['block angles']

        qc = QuantumCircuit(self.num_qubits)

        # Initial round of single-qubit gates
        for n, a in enumerate(surface_angles):
            if not parametrized:
                a = list(map(float, a))

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


class Decomposition:
    def __init__(self, unitary_func, circuit_func, angles, num_cz_gates):
        self.unitary_func = unitary_func
        self.circuit_func = circuit_func
        self.angles = angles
        self.num_cz_gates = num_cz_gates
        self.circuit = self.circuit_func(self.angles)

    def __repr__(self):
        return f'Decomposition with {self.num_cz_gates} cz gates.'


@dataclass
class RegularizationOptions:
    function: str = 'linear'
    ymax: float = 2
    xmax: float = jnp.pi / 2
    plato_0: float = 0.05
    plato_1: float = 0.05
    plato_2: float = 0.05


@dataclass
class BasicOptions:
    num_samples: int = 100
    method: str = 'adam'
    learning_rate: float = 0.1
    num_gd_iterations: int = 2000
    cp_distribution: str = 'uniform'
    entry_loss: float = 1e-3
    target_loss: float = 1e-6
    threshold_cp: float = 0.2
    learning_rate_at_verification: float = 0.01
    num_gd_iterations_at_verification: int = 5000


@dataclass
class StaticOptions(BasicOptions):
    num_cp_gates: int = -1
    r: float = 0.00055
    accepted_num_cz_gates: int = -1

    def __post_init__(self):
        if self.num_cp_gates == -1:
            raise TypeError("Missing required argument 'num_cp_gates'")
        if self.accepted_num_cz_gates == -1:
            raise TypeError("Missing required argument 'accepted_num_cz_gates'")


@dataclass
class AdaptiveOptions(BasicOptions):
    min_num_cp_gates: int = -1
    max_num_cp_gates: int = -1
    r_mean: float = 0.00055
    r_variance: float = 0.5
    max_evals: int = 100
    target_num_cz_gates: int = 0
    stop_if_target_reached: bool = False
    random_seed: int = 0

    def __post_init__(self):
        if self.min_num_cp_gates == -1:
            raise TypeError("Missing required argument 'min_num_cp_gates'")
        if self.max_num_cp_gates == -1:
            raise TypeError("Missing required argument 'max_num_cp_gates'")

    def get_static(self, num_cp_gates, r):
        default_static_dict = asdict(BasicOptions())
        options_dict = asdict(self)
        basic_dict = {key: value for key, value in options_dict.items() if key in default_static_dict}
        basic_dict['num_cp_gates'] = num_cp_gates
        basic_dict['r'] = r
        basic_dict['accepted_num_cz_gates'] = None
        return StaticOptions(**basic_dict)


class Decompose:

    def __init__(self, layer, unitary_loss_func=None, cp_regularization_func=None, u_target=None):
        self.u_target = u_target
        if unitary_loss_func is not None:
            self.unitary_loss_func = unitary_loss_func
        else:
            assert self.u_target is not None, 'Neither unitary loss function nor target unitary is provided.'
            self.unitary_loss_func = lambda u: disc2(u, self.u_target)

        if cp_regularization_func:
            self.cp_regularization_func = cp_regularization_func
        else:
            self.cp_regularization_func = make_regularization_function(RegularizationOptions)
        self.layer = layer
        self.num_qubits = num_qubits_from_layer(self.layer)

    @staticmethod
    def generate_initial_angles(key, num_angles, cp_mask, cp_dist='uniform', batch_size=1):
        key, *subkeys = random.split(key, num=batch_size + 1)
        initial_angles_array = jnp.array(
            [random_cp_angles(num_angles, cp_mask, cp_dist=cp_dist, key=k)
             for k in subkeys])

        return initial_angles_array

    @staticmethod
    def save_to_paths(save_to):
        if save_to is None:
            print("Warning: results will not be saved since 'save_to' is not provided.")
            return None, None
        if not os.path.exists(save_to):
            os.makedirs(save_to)

        trials_path = save_to + 'trials'
        decompositions_path = save_to + 'decompositions'

        return trials_path, decompositions_path

    @staticmethod
    def save_trials(save_to, trials):
        if save_to is None:
            return
        trials_path, decompositions_path = Decompose.save_to_paths(save_to)

        with open(trials_path, 'wb') as f:
            dill.dump(trials, f)

    @staticmethod
    def save_decompositions(save_to, overwrite_existing_decompositions, decompositions):
        if save_to is None or decompositions is None:
            return

        trials_path, decompositions_path = Decompose.save_to_paths(save_to)
        existing_trials, existing_decompositions = Decompose.load_trials_and_decompositions(save_to)

        if overwrite_existing_decompositions:
            decompositions_to_save = decompositions
        else:
            decompositions_to_save = existing_decompositions
            decompositions_to_save.extend(decompositions)

        with open(decompositions_path, 'wb') as f:
            dill.dump(decompositions_to_save, f)

    @staticmethod
    def load_trials_and_decompositions(save_to):
        if save_to is None:
            trials, decompositions = [], []
        else:
            trials_path, decompositions_path = Decompose.save_to_paths(save_to)
            try:
                with open(trials_path, 'rb') as f:
                    trials = dill.load(f)
            except FileNotFoundError:
                trials = Trials()

            try:
                with open(decompositions_path, 'rb') as f:
                    decompositions = dill.load(f)
            except FileNotFoundError:
                decompositions = []

        return trials, decompositions

    @staticmethod
    def plot_raw(res):
        plt.plot(res['regloss'], label='regloss')
        plt.plot(res['loss'], label='loss')
        plt.plot(res['reg'], label='reg')
        plt.yscale('log')
        plt.legend()

    def generate_raw(self, options, key=random.PRNGKey(0), initial_angles_array=None, keep_history=False):

        # options = Decompose.updated_options(Decompose.default_static_options, options)
        anz = Ansatz(self.num_qubits, 'cp', fill_layers(self.layer, options.num_cp_gates))
        loss_func = lambda angles: self.unitary_loss_func(anz.unitary(angles))

        def regularization_func(angs):
            return options.r*vmap(self.cp_regularization_func)(angs*anz.cp_mask).sum()

        if initial_angles_array is None:
            initial_angles_array = Decompose.generate_initial_angles(
                key,
                anz.num_angles,
                anz.cp_mask,
                cp_dist=options.cp_distribution,
                batch_size=options.num_samples)

        raw_results = mynimize_repeated(
            loss_func,
            anz.num_angles,
            method=options.method,
            learning_rate=options.learning_rate,
            num_iterations=options.num_gd_iterations,
            initial_params_batch=initial_angles_array,
            regularization_func=regularization_func,
            u_func=anz.unitary,  # For 'adam' this won't be used, but it will e.g. for 'natural adam'.
            keep_history=keep_history
        )

        return raw_results

    def evaluate_raw(self, raw_results, options, disable_tqdm=False):

        # options = Decompose.updated_options(Decompose.default_static_options, options)
        anz = Ansatz(self.num_qubits, 'cp', fill_layers(self.layer, options.num_cp_gates))

        below_entry_loss_results = filter_cp_results(
            raw_results,
            anz.cp_mask,
            float('inf'),  # At this stage we only filter by convergence, not by the number of gates.
            options.entry_loss,
            threshold_cp=options.threshold_cp,
            disable_tqdm=disable_tqdm
        )

        return below_entry_loss_results

    def static(self, options, key=random.PRNGKey(0), save_to=None, overwrite_existing_decompositions=False):

        _, decompositions_path = Decompose.save_to_paths(save_to)
        _, existing_decompositions = Decompose.load_trials_and_decompositions(save_to)

        if existing_decompositions and overwrite_existing_decompositions:
            print("Warning: existing decompositions will be overwritten.")

        print('\nStarting decomposition routine with the following options:\n')
        print(options)

        print('\nComputing raw results...')
        raw_results = Decompose.generate_raw(self,
                                             options,
                                             key=key)

        print('\nSelecting prospective results...')
        raw = Decompose.evaluate_raw(self, raw_results, options)
        prospective_results = raw
        prospective_results = [res for res in prospective_results if res[0] <= options.accepted_num_cz_gates]
        successful_results = []

        if prospective_results:
            print(f'\nFound {len(prospective_results)}. Verifying...')

            anz = Ansatz(self.num_qubits, 'cp', fill_layers(self.layer, options.num_cp_gates))
            for num_cz_gates, res in tqdm(prospective_results):

                success, num_cz_gates, circ, u, best_angs = verify_cp_result(
                    res,
                    anz,
                    self.unitary_loss_func,
                    options,
                    keep_history=False)

                if success:
                    new_decomposition = Decomposition(u, circ, best_angs, num_cz_gates)
                    successful_results.append(new_decomposition)

            if successful_results:
                print(f'\n{len(successful_results)} successful. cz counts are:')
                print(sorted([d.num_cz_gates for d in successful_results]))
                Decompose.save_decompositions(save_to, overwrite_existing_decompositions, successful_results)
            else:
                print('\nAll prospective results failed.\n')

        else:
            print('No results passed.')

        return successful_results

    def adaptive(self,
                 options,
                 save_to=None,
                 overwrite_existing_trials=False,
                 overwrite_existing_decompositions=False):

        def objective_from_cz_distribution(random_seed, search_params):

            # angles_random_seed = int(float(str(time.time())[::-1]))
            num_cp_gates, r = search_params

            static_options = options.get_static(num_cp_gates, r)

            raw_results = Decompose.generate_raw(
                self,
                static_options,
                key=random.PRNGKey(random_seed),
                )

            evaluated_results = Decompose.evaluate_raw(
                self,
                raw_results,
                static_options,
                disable_tqdm=True,
            )

            cz_counts = [res[0] for res in evaluated_results]
            score = (2 ** (-(jnp.array(cz_counts, dtype=jnp.float32) - options.target_num_cz_gates))).sum() / \
                    options.num_samples

            return {
                'loss': -score,
                'status': STATUS_OK,
                'angles_random_seed': random_seed,
                'cz_counts': cz_counts,
                'layer': self.layer,
                'unitary_loss_func': self.unitary_loss_func,
                'static_options': static_options,
                'attachments': {'prospective_decompositions': pickle.dumps(evaluated_results)}
            }

        print('\nStarting decomposition routine with the following options:\n')
        print(options)
        print('\n')

        # Defining the hyperparameter search space.
        space = [
            scope.int(
                hp.quniform('num_cp_gates', options.min_num_cp_gates, options.max_num_cp_gates,
                            1)),
            hp.lognormal('r', jnp.log(options.r_mean), options.r_variance)
        ]

        # Loading existing trials and decompositions.
        existing_trials, existing_decompositions = Decompose.load_trials_and_decompositions(save_to)
        if existing_trials and not overwrite_existing_trials:
            print('\nFound existing trials, resuming from here.')
            trials = existing_trials
        else:
            trials = Trials()

        # Creating scoreboard variable that keeps track of the best current cz count.
        if existing_decompositions:
            scoreboard = set([d.num_cz_gates for d in existing_decompositions])
            scoreboard = sorted(list(scoreboard))
        else:
            scoreboard = [theoretical_lower_bound(self.num_qubits)]

        decompositions = []
        key = random.PRNGKey(options.random_seed)

        for _ in tqdm(range(options.max_evals), desc='Epochs'):

            key, subkey = random.split(key)
            seed_angles, seed_hyperopt = random.randint(subkey, (2,),  minval=0, maxval=10e6)

            best = fmin(
                partial(objective_from_cz_distribution, seed_angles),
                space=space,
                algo=tpe.suggest,
                max_evals=len(trials.trials)+1,
                trials=trials,
                rstate=np.random.default_rng(int(seed_hyperopt)))

            Decompose.save_trials(save_to, trials)

            current_best_cz = scoreboard[0]
            results_to_verify = []

            trial = trials.trials[-1]
            msg = trials.trial_attachments(trial)['prospective_decompositions']
            successful_results = pickle.loads(msg)
            num_cp_gates = int(trial['misc']['vals']['num_cp_gates'][0])
            prospective_results = [[num_cp_gates, res] for cz, res in successful_results if cz < current_best_cz]
            num_equivalent_results = sum([cz == current_best_cz for cz, res in successful_results])
            results_to_verify.extend(prospective_results)

            if len(results_to_verify):
                print(
                    f'\nFound {len(results_to_verify)} decompositions potentially improving the current best count {current_best_cz}, verifying...')
            else:
                print(
                    f'\nFound no better decompositions. Found {num_equivalent_results} decompositions with the current best count {current_best_cz}.')

            for num_cp_gates, res in prospective_results:
                anz = Ansatz(self.num_qubits, 'cp', placements=fill_layers(self.layer, num_cp_gates))

                print('verifying with options')
                print(options.get_static(None, None))
                success, num_cz_gates, circ, u, best_angs = verify_cp_result(
                    res,
                    anz,
                    self.unitary_loss_func,
                    options.get_static(None, None))

                if success:
                    print(f'\nFound new decomposition with {num_cz_gates} gates.\n')

                    scoreboard.insert(0, num_cz_gates)
                    new_decomposition = Decomposition(u, circ, best_angs, num_cz_gates)
                    Decompose.save_decompositions(save_to, overwrite_existing_decompositions, [new_decomposition])

                    break
            else:
                if prospective_results:
                    print('\nNo decomposition passed.\n')

            if options.stop_if_target_reached and scoreboard[0] <= options.target_num_cz_gates:
                print('\nTarget number of gates reached.')
                break

        return decompositions, trials, best

    # @staticmethod
    # def reconstruct_trial(trial):
    #
    #     options = trial['result']['static_options']
    #     layer = trial['result']['layer']
    #     unitary_loss_func = trial['result']['unitary_loss_func']
    #
    #     num_qubits = num_qubits_from_layer(layer)
    #     options = Decompose.updated_options(Decompose.default_static_options, options)
    #
    #     angles_random_seed = trial['result']['angles_random_seed']
    #     num_cp_gates = int(trial['misc']['vals']['num_cp_gates'][0])
    #     r = trial['misc']['vals']['r'][0]
    #
    #     anz = Ansatz(num_qubits, 'cp', fill_layers(layer, num_cp_gates))
    #
    #     initial_angles = Decompose.generate_initial_angles(
    #         random.PRNGKey(angles_random_seed),
    #         anz.num_angles,
    #         anz.cp_mask,
    #         cp_dist=options['cp_dist'],
    #         batch_size=options['batch_size'])
    #
    #     d = Decompose(layer, unitary_loss_func=unitary_loss_func)
    #     raw_results = d.generate_raw(
    #         num_cp_gates,
    #         r,
    #         initial_angles_array=initial_angles,
    #         options=options,
    #         keep_history=True)
    #
    #     below_entry_loss_results = d.evaluate_raw(
    #         raw_results,
    #         num_cp_gates,
    #         options=options,
    #         disable_tqdm=False)
    #
    #     return below_entry_loss_results

    # @staticmethod
    # def reconstruct_verification(trials, i_trial, i_sample):
    #     trial = trials.trials[i_trial]
    #     msg = trials.trial_attachments(trial)['prospective_decompositions']
    #     prospective_results = pickle.loads(msg)
    #     cz, res = prospective_results[i_sample]
    #
    #     options = trial['result']['static_options']
    #     layer = trial['result']['layer']
    #     unitary_loss_func = trial['result']['unitary_loss_func']
    #
    #     num_qubits = num_qubits_from_layer(layer)
    #     num_cp_gates = int(trial['misc']['vals']['num_cp_gates'][0])
    #
    #     anz = Ansatz(num_qubits, 'cp', fill_layers(layer, num_cp_gates))
    #
    #     print('vefigying with options')
    #     print(options)
    #     success, num_cz_gates, circ, u, best_angs, angles_history, loss_history = verify_cp_result(
    #         res, anz, unitary_loss_func, **options, keep_history=True)
    #
    #     return {'success': success, 'num_cz_gates': num_cz_gates, 'circ': circ, 'unitary': u, 'best_angles': best_angs,
    #             'angles_history': angles_history, 'loss_history': loss_history}