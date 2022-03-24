import pickle
import os
from dataclasses import dataclass, asdict

import dill
import numpy as np
import matplotlib.pyplot as plt

from qiskit import transpile
from qiskit.circuit import Parameter

from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from hyperopt.pyll import scope

from tqdm.auto import tqdm

from cp_utils import random_cp_angles, filter_cp_results, verify_cp_result
from circuit_assembly import *
from optimization import *
from penalty import *
from topology import *
from exact_decompositions import make_exact, cp_to_cz_circuit


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
        qc.rx(angles[0], 0)
        qc.rz(angles[1], 0)
        qc.rx(angles[2], 1)
        qc.rz(angles[3], 1)

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

        x_rotations = jnp.kron(rx_mat(self.angles[0]), rx_mat(self.angles[2]))
        z_rotations = jnp.kron(rz_mat(self.angles[1]), rz_mat(self.angles[3]))

        return z_rotations @ x_rotations @ entangling_matrix


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

    @staticmethod
    def _gate_filter(gate_names, d):
        gate, qargs, cargs = d
        if gate.name in gate_names:
            return True
        else:
            return False

    @staticmethod
    def _gate_count(gate_name, circuit):
        ops = circuit.count_ops()
        if gate_name in ops:
            return ops[gate_name]
        else:
            return 0

    def __init__(self, unitary_loss_func, circuit, label='', type='Approximate'):

        self.unitary_loss_func = unitary_loss_func
        self.circuit = circuit
        self.unitary = Operator(self.circuit.reverse_bits()).data
        self.label = label

        self.loss = self.unitary_loss_func(self.unitary)
        self.type = type

        self.cz_count = Decomposition._gate_count('cz', self.circuit)
        self.cz_depth = self.circuit.depth(partial(Decomposition._gate_filter, ['cz']))

        self.t_count = None
        self.t_depth = None

        self._cp_data = None
        self._static_options = None
        self._adaptive_options = None
        self._decomposer = None

    @classmethod
    def from_cp_circuit(cls, unitary_loss_func, u_func, circ_func, angles, label):
        circuit = cp_to_cz_circuit(circ_func(angles))
        circuit = transpile(circuit, basis_gates=['cz', 'rz', 'rx', 'id'], optimization_level=3)

        d = cls(unitary_loss_func, circuit, label=label)
        d._cp_data = [u_func, circ_func, angles]

        return d

    def make_exact(self, cp_threshold=0.01, reduce_threshold=1e-5, recursion_degree=0, recursion_depth=5):

        try:
            exact_qc = make_exact(
                self.circuit,
                self.unitary_loss_func,
                cp_threshold=cp_threshold,
                reduce_threshold=reduce_threshold,
                recursion_degree=recursion_degree,
                recursion_depth=recursion_depth)

            self.circuit = exact_qc
            self.type = 'Exact'

            self.t_count = Decomposition._gate_count('t', self.circuit)+Decomposition._gate_count('tdg', self.circuit)
            self.t_depth = self.circuit.depth(partial(Decomposition._gate_filter, ['t', 'tdg']))

            return 'Successful'
        except ValueError as e:
            print(e)
            return 'Failed'

    def __repr__(self):
        if self.type == 'Approximate':
            return f"< {self.label}| {self.type} | cz count: {self.cz_count} | cz depth: {self.cz_count} | loss: {self.loss} >"
        elif self.type == 'Exact':
            return f"< {self.label}| {self.type} | cz count: {self.cz_count} | cz depth: {self.cz_count} | t count: {self.t_count} | t depth: {self.t_depth} >"


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
    random_seed: int = 0
    stop_if_target_reached: bool = False
    keep_logs: bool = False

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


@dataclass
class Results:
    loss_function: callable
    layer: list
    label: str = ''
    trials: Trials = None
    decompositions: tuple = ()
    save_to: str = ''

    def __post_init__(self):
        if self.save_to == '':
            self.save_to = f'results/{self.label}'

    def save(self):
        os.makedirs(os.path.dirname(self.save_to), exist_ok=True)
        with open(self.save_to, 'wb') as f:
            dill.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            results = dill.load(f)

        return results

    def plot_trials(self):
        trials = self.trials
        options_list = [res['static_options'] for res in trials.results]
        num_list = [o.num_cp_gates for o in options_list]
        r_list = [o.r for o in options_list]
        loss_list = [res['loss'] for res in trials.results]

        plt.scatter(num_list, r_list, c=jnp.exp(jnp.array(loss_list, dtype=jnp.float32)), cmap='gray')
        plt.ylabel('r')
        plt.xlabel('num_cp_gates')
        plt.title('score')


class Decompose:
    """Class for efficient automated decomposing of unitary matrices into CZ+1q gates.


    """
    def __init__(self, layer, unitary_loss_func=None, target_unitary=None, label=None, cp_regularization_func=None):

        self.layer = layer
        self.num_qubits = num_qubits_from_layer(self.layer)

        self.target_unitary = target_unitary
        if unitary_loss_func is not None:
            self.unitary_loss_func = unitary_loss_func
        else:
            assert self.target_unitary is not None, 'Neither unitary loss function nor target unitary is provided.'
            assert self.target_unitary.shape == (2**self.num_qubits, 2**self.num_qubits), 'Number of qubits in target unitary and layer do not match.'
            self.unitary_loss_func = lambda u: disc2(u, self.target_unitary)

        self.label = label
        if cp_regularization_func:
            self.cp_regularization_func = cp_regularization_func
        else:
            self.cp_regularization_func = make_regularization_function(RegularizationOptions)

    @staticmethod
    def generate_initial_angles(key, num_angles, cp_mask, cp_dist='uniform', batch_size=1):
        key, *subkeys = random.split(key, num=batch_size + 1)
        initial_angles_array = jnp.array(
            [random_cp_angles(num_angles, cp_mask, cp_dist=cp_dist, key=k)
             for k in subkeys])

        return initial_angles_array

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

    def _initialize_results(self, save_results, save_to):
        results = Results(self.unitary_loss_func, self.layer, label=self.label)
        if save_results:
            assert self.label or save_to, 'To save results either `label` or `save_to` must be provided.'
            if save_to:
                results.save_to = save_to

            # Try to load existing results.
            try:
                results = Results.load(results.save_to)
            except FileNotFoundError:
                pass

        return results

    def _make_decomposition(self, u_func, circ_func, best_angs, static_options=None, adaptive_options=None):
        d = Decomposition.from_cp_circuit(self.unitary_loss_func, u_func, circ_func, best_angs, self.label)
        d._static_options = static_options
        d._adaptive_options = adaptive_options
        d._decomposer = self
        return d

    def static(self, options, key=random.PRNGKey(0), save_results=True, save_to=''):

        results = Decompose._initialize_results(self, save_results, save_to)

        print('\nStarting decomposition routine with the following options:')
        print('\n', options)

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
                    new_decomposition = Decompose._make_decomposition(self, u, circ, best_angs, static_options=options)
                    successful_results.append(new_decomposition)

            if successful_results:
                print(f'\n{len(successful_results)} successful. cz counts are:')
                print(sorted([d.cz_count for d in successful_results]))
                results.decompositions = list(results.decompositions)+successful_results
                if save_results:
                    results.save()

            else:
                print('\nAll prospective results failed.')

        else:
            print('\nNo results passed.')

        return results

    def adaptive(self,
                 options,
                 save_results=True,
                 save_to=''):

        def objective_from_cz_distribution(random_seed, search_params):
            """Evalueates quality of the hyperparameter configuration.

            Params:
                random_seed: seed to be used for generating random initial angles and setting hyperopt state.
                search_params: params subject to hyperopt optimization."""

            num_cp_gates, r = search_params
            tqdm.write(f'\nnum_cp_gates: {num_cp_gates}, r: {r}')
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

            # Score is defined as the weighted sum of cz counts of all successful results.
            # For convenience it is normalized on the number of samples and presented in log scale.

            score = 2 ** (-jnp.array(cz_counts, dtype=jnp.float32))
            score = (score.sum() / options.num_samples)
            score = jnp.log2(score)

            tqdm.write(f'score: {-score}, cz counts of prospective results: {cz_counts}')

            return_dict = {
                'loss': -score,
                'status': STATUS_OK,
                'random_seed': random_seed,
                'cz_counts': cz_counts,
                'num_cp_gates': num_cp_gates,
                'r': r,
                'layer': self.layer,
                'prospective_decompositions': evaluated_results}

            if options.keep_logs:
                return_dict['attachments'] = {
                    'prospective_decompositions': dill.dumps(evaluated_results),
                    'static_options': dill.dumps(static_options),
                    'unitary_loss_func': dill.dumps(self.unitary_loss_func)}

            return return_dict

        print('\nStarting decomposition routine with the following options:')
        print('\n', options)

        # Defining the hyperparameter search space.
        space = [
            scope.int(
                hp.quniform('num_cp_gates', options.min_num_cp_gates, options.max_num_cp_gates,
                            1)),
            hp.lognormal('r', jnp.log(options.r_mean), options.r_variance)
        ]

        # Loading existing trials and decompositions.
        results = Decompose._initialize_results(self, save_results, save_to)

        if results.trials is not None:
            print('\nFound existing trials, resuming from here.')
            trials = results.trials
            random_seed = trials.results[-1]['random_seed']
            num_existing_trials = len(trials.results)
        else:
            trials = Trials()
            random_seed = options.random_seed
            num_existing_trials = 0

        # Creating scoreboard variable that keeps track of the best current cz count.
        if results.decompositions:
            scoreboard = set([d.cz_count for d in results.decompositions])
            scoreboard = sorted(list(scoreboard))
        else:
            scoreboard = [theoretical_lower_bound(self.num_qubits)]

        if num_existing_trials >= options.max_evals:
            print(f'Maximum number of evaluations reached.')

        for i in tqdm(range(num_existing_trials, options.max_evals), desc='Evaluations'):

            tqdm.write('\n' + '-'*42)
            tqdm.write(f'iteration {i}/{options.max_evals}')

            _, subkey = random.split(random.PRNGKey(random_seed))
            random_seed = int(subkey[1])

            fmin(
                partial(objective_from_cz_distribution, random_seed),
                space=space,
                algo=tpe.suggest,
                max_evals=len(trials.trials)+1,
                trials=trials,
                rstate=np.random.default_rng(int(random_seed)),
                verbose=False,
                show_progressbar=False
            )

            results.trials = trials
            results.save()

            current_best_cz = scoreboard[0]

            last_result = trials.results[-1]
            num_cp_gates = last_result['num_cp_gates']
            r = last_result['r']
            successful_results = last_result['prospective_decompositions']
            if not options.keep_logs:
                last_result.pop('prospective_decompositions')

            results_to_verify = [[num_cp_gates, res] for cz, res in successful_results if cz < current_best_cz]

            if len(results_to_verify):
                tqdm.write(
                    f'\nFound {len(results_to_verify)} decompositions potentially improving the current best count {current_best_cz}, verifying...')
            else:
                tqdm.write(
                    f'\nFound no decompositions potentially improving the current best count {current_best_cz}.')

            for num_cp_gates, res in results_to_verify:
                anz = Ansatz(self.num_qubits, 'cp', placements=fill_layers(self.layer, num_cp_gates))

                success, num_cz_gates, circ, u, best_angs = verify_cp_result(
                    res,
                    anz,
                    self.unitary_loss_func,
                    options.get_static(None, None))

                if success:
                    tqdm.write(f'\nFound a new decomposition with {num_cz_gates} gates.')

                    scoreboard.insert(0, num_cz_gates)
                    new_decomposition = Decompose._make_decomposition(
                        self,
                        u,
                        circ,
                        best_angs,
                        adaptive_options=options,
                        static_options=options.get_static(num_cp_gates, r))
                    results.decompositions = list(results.decompositions) + [new_decomposition]
                    results.save()
                    break
            else:
                if results_to_verify:
                    tqdm.write('\nNone of prospective decompositions passed.')

            if options.stop_if_target_reached and scoreboard[0] <= options.target_num_cz_gates:
                tqdm.write('\nTarget number of gates reached.')
                break

        return results

