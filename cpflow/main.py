"""Main classes and routines exposed to the user."""


import os
from dataclasses import dataclass, asdict

import dill
import matplotlib.pyplot as plt
import numpy as np
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from hyperopt.pyll import scope
from qiskit.circuit import Parameter
from tqdm.auto import tqdm

from cpflow.circuit_assembly import *
from cpflow.regularization import *
from cpflow.exact_decompositions import *
from cpflow.optimization import *
from cpflow.penalty_functions import *
from cpflow.topology import *


class EntanglingBlock:
    """Two-qubit entangling block.

    Methods:
        circuit: gives an equivalent `qiskit` circuit.
        unitary: gives `jax.numpy` unitary matrix of the circuit.
        num_angles: total number of angles (parameters) in a block.
    """

    def __init__(self, entangling_gate_name, rotation_gates):
        self.entangling_gate_name = entangling_gate_name
        self.rotation_gates = rotation_gates

        self.entangling_gate = Gate.from_name(self.entangling_gate_name)
        self.num_entangling_angles = self.entangling_gate.num_angles
        self.num_angles = self.num_entangling_angles+2*len(rotation_gates)

    @staticmethod
    def split_angles(num_entangling_angles, angles, as_numpy=False):
        entangling_angles = angles[:num_entangling_angles]
        up_angles = angles[num_entangling_angles::2]
        down_angles = angles[num_entangling_angles+1::2]

        if as_numpy:
            entangling_angles = np.array(entangling_angles)
            up_angles = np.array(up_angles)
            down_angles = np.array(down_angles)

        return entangling_angles, up_angles, down_angles


    @staticmethod
    def apply_entangling_gate(gate_func, angles, to=None):
        if len(angles) == 0:
            if to == 'qs':
                return gate_func()
            else:
                return gate_func
        elif len(angles) == 1:
            return gate_func(angles[0])
        else:
            return gate_func(angles)

    def circuit(self, angles=None):
        """Quantum circuit in `qiskit` corresponding to our block."""

        if angles is None:
            angles = [Parameter(f'a{i}') for i in range(self.num_angles)]
        else:
            assert len(angles) == self.num_angles, f'Provided angles do not match required number {self.num_angles}.'

        entangling_angles, up_angles, down_angles = EntanglingBlock.split_angles(
            self.num_entangling_angles,
            angles,
            as_numpy=True)

        qc = QuantumCircuit(2)

        # Apply entangling gate
        gate_instance = EntanglingBlock.apply_entangling_gate(
            self.entangling_gate.qiskit_gate,
            entangling_angles,
            to='qs')

        qc.append(gate_instance, [0, 1])

        # Apply single-qubit gates.
        for xyz, angle0, angle1 in zip(self.rotation_gates, up_angles, down_angles):
            gate = Gate.from_name('r'+xyz)
            qc.append(gate.qiskit_gate(angle0), [0])
            qc.append(gate.qiskit_gate(angle1), [1])

        return qc

    def unitary(self, angles):
        """2x2 unitary of the block."""

        entangling_angles, up_angles, down_angles = EntanglingBlock.split_angles(
            self.num_entangling_angles,
            angles)

        entagling_matrix = EntanglingBlock.apply_entangling_gate(
            self.entangling_gate.jax_matrix,
            entangling_angles
        )

        u = entagling_matrix
        for xyz, angle0, angle1 in zip(self.rotation_gates, up_angles, down_angles):
            gate = Gate.from_name('r'+xyz)
            u = jnp.kron(gate.jax_matrix(angle0), gate.jax_matrix(angle1)) @ u

        return u


def split_angles(angles, num_qubits, num_block_angles, num_entangling_angles, layer_len=0, num_layers=0):

    surface_angles = angles[:3 * num_qubits].reshape(num_qubits, 3)
    block_angles = angles[3 * num_qubits:].reshape(-1, num_block_angles)
    if num_layers is None:
        layers_angles = []
    else:
        layers_angles = block_angles[:layer_len * num_layers].reshape(num_layers, layer_len, num_block_angles)
    free_block_angles = block_angles[layer_len * num_layers:]

    entangling_angles = [b[:num_entangling_angles] for b in block_angles]

    return {'surface angles': surface_angles,
            'block angles': block_angles,
            'layers angles': layers_angles,
            'free block angles': free_block_angles,
            'entangling angles': entangling_angles}


def build_unitary(num_qubits, entangling_gate_name, rotation_gates, placements, angles):
    layer, num_layers = placements['layers']
    free_placements = placements['free']

    layer_depth = len(layer)
    num_block_angles = EntanglingBlock(entangling_gate_name, rotation_gates).num_angles
    num_entangling_angles = EntanglingBlock(entangling_gate_name, rotation_gates).num_entangling_angles

    angles_dict = split_angles(angles, num_qubits, num_block_angles, num_entangling_angles, len(layer), num_layers)

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
            gate = EntanglingBlock(entangling_gate_name, rotation_gates).unitary(a).reshape(2, 2, 2, 2)
            u = apply_gate_to_tensor(gate, u, p)

        return u

    if num_layers > 0:
        u = lax.fori_loop(0, num_layers, lambda i, u: apply_layer(i, u, layer, layers_angles), u)

    # Add remainder(free) blocks.
    for a, p in zip(free_block_angles, free_placements):
        gate = EntanglingBlock(entangling_gate_name, rotation_gates).unitary(a).reshape(2, 2, 2, 2)
        u = apply_gate_to_tensor(gate, u, p)

    return u.reshape(2 ** num_qubits, 2 ** num_qubits)


class Ansatz:
    """Building and training template circuits.

    Attributes:
        num_qubits (int): number of qubits.
        entangling_gate_name (str): type of the entagling gate in the ansatz, either 'cx', 'cz' or 'cp'.
        rotation_gates (str): type of 1q gates in the entagling blocks, e.g. 'xz' or 'xyz', can be an arbitrary string of basis letters.
        placements (dict): specifies where entangling gates are placed.

    Methods:
        circuit (angles): qiskit circuit of the ansatz. For unspecified angles returns parametrized circuit.
        learn (array): run optimization routine to minimize Hilbert-Schmidt distance to the array.
    """
    def __init__(self, num_qubits, entangling_gate_name, placements, rotation_gates='xyz'):

        self.num_qubits = num_qubits
        self.entangling_gate_name = entangling_gate_name
        self.rotation_gates = rotation_gates

        placements.setdefault('layers', [[], 0])
        placements.setdefault('free', [])
        self.placements = placements

        self.layer, self.num_layers = placements['layers']
        self.free_placements = placements['free']

        self.all_placements = self.layer * self.num_layers + self.free_placements
        self.num_blocks = len(self.all_placements)

        self.num_block_angles = EntanglingBlock(entangling_gate_name, rotation_gates).num_angles
        self.num_angles = 3 * num_qubits + self.num_block_angles * len(self.all_placements)

        sample_angles = jnp.arange(self.num_angles)
        num_entangling_angles_in_block = EntanglingBlock(self.entangling_gate_name, self.rotation_gates).num_entangling_angles
        entangling_angles = split_angles(sample_angles, self.num_qubits, self.num_block_angles, num_entangling_angles_in_block)['entangling angles']
        self.entangling_mask = jnp.array([1 if a in entangling_angles else 0 for a in sample_angles])

        self.unitary = lambda angles: build_unitary(
            self.num_qubits,
            self.entangling_gate_name,
            self.rotation_gates,
            self.placements,
            angles)

    def circuit(self, angles=None):
        if angles is None:
            angles = np.array([Parameter(f"a_{ {i} }") for i in range(self.num_angles)])
            parametrized = True
        else:
            parametrized = False
        angles_dict = split_angles(angles, self.num_qubits, self.num_block_angles, len(self.layer), self.num_layers)

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
            qc_block = EntanglingBlock(self.entangling_gate_name, self.rotation_gates).circuit(a)
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
    """Storing and processing decompositions.

    Attributes:
        unitary_loss_func (callable): function of a unitary that was minimized.
        circuit (qiskit circuit): circuit of the decomposition.
        unitary (array): unitary matrix of the decomposition.
        label: name of the decomposition.
        loss: value of the `unitary_loss_func`.
        type: 'Approximate` for raw decompositions, sometimes can be impoved to `Rational` or `Clifford+T` by the `refine` method.
        gate_count_2q: total number of CZ gates.
        gate_depth_2q: circuit depth with respect to CZ gates.
        t_count: for Clifford+T `type` total number of T and T dagger gates.
        t_depth: for Clifford+T `type` circuit depth with respect to T and T dagger gates.

    Methods:
        refine(): attempts to simplify 1q angles in the circuit, represent them as rational multiples of pi and/or translate to the Clifford+T basis.
    """

    def __init__(self, unitary_loss_func, circuit, entangling_gate_name, label='', type='Approximate'):

        self.unitary_loss_func = unitary_loss_func
        self.circuit = circuit
        self.entangling_gate_name = entangling_gate_name
        self.unitary = Operator(self.circuit.reverse_bits()).data
        self.label = label

        self.loss = self.unitary_loss_func(self.unitary)
        self.type = type

        self.gate_count_2q = gates_count([entangling_gate_name], self.circuit)
        self.gate_depth_2q = gates_depth([entangling_gate_name], self.circuit)

        self.t_count = None
        self.t_depth = None

        self._cp_data = None
        self._static_options = None
        self._adaptive_options = None
        self._decomposer = None

    @classmethod
    def _from_verified_result(cls, unitary_loss_func, u_func, circ_func, angles, entangling_gate_name, regularization_options, label):
        qc = circ_func(angles)
        qc = refine_circuit(qc, entangling_gate_name, regularization_options)
        d = cls(unitary_loss_func, qc, entangling_gate_name, label=label)
        d._cp_data = [u_func, circ_func, angles]

        return d

    def refine(
            self,
            max_denominator=32,
            angle_threshold=0.01,
            cp_threshold=0.01,
            reduce_threshold=1e-5,
            recursion_degree=0,
            recursion_depth=5):

        qc, refine_type, t_count, t_depth = refine(
            self.circuit,
            self.unitary_loss_func,
            max_denominator=max_denominator,
            angle_threshold=angle_threshold,
            cp_threshold=cp_threshold,
            reduce_threshold=reduce_threshold,
            recursion_degree=recursion_degree,
            recursion_depth=recursion_depth)

        self.type = refine_type
        self.circuit = qc

        if refine_type == 'Clifford+T':
            self.t_count = t_count
            self.t_depth = t_depth

        return f'Refined to {refine_type}'

    def __repr__(self):
        description = f"< {self.label}| {self.type} | loss: {self.loss}  | 2q count: {self.gate_count_2q} | 2q depth: {self.gate_depth_2q}  >"
        if self.type == 'Clifford+T':
            description = description[:-1]+f'| T count: {self.t_count} | T depth: {self.t_depth} >'
        return description


@dataclass
class RegularizationOptions:
    regularization_func: callable
    count_func: callable
    angle_projector: callable
    gate_decomposer: callable
    angle_threshold: float = 0.2


regularization_options_cp = RegularizationOptions(
    default_cp_regularization_function,
    cz_value,
    project_cp_angle,
    cp_to_cz_gate)

regularization_options_parametric = RegularizationOptions(
    penalty_L1,
    parametric_value,
    project_parametric_angle,
    parametric_2q_decomposer)



@dataclass
class BasicOptions:
    """Dataclass keeping options needed for both static and adaptive synthesis.

    Attributes:
        num_samples (int): number of initial conditions in multi-start optimization.
        method (str): optimization method, currently only 'adam' is well tested.
        learning_rate (str): learning rate for the optimizer.
        num_gd_iterations (int): number of optimizer updates at the raw sampling stage.
        entangling_angles_distribution (str): 'uniform' for uniform initialization of CP angles, '0' for zero initialization.
        entry_loss (float): acceptable loss to deem a CP template as prospective.
        target_loss (float): threshold value of loss to consider a CZ circuit to be a valid decomposition.
        learning_rate_at_verification (float): learning rate to use at the verification of CZ circuits.
        num_gd_iterations_at_verification (int): number of optimizer updates at the verification of CZ circuits.
        random_seed (int): seed controlling sampling of the initial angles and hyperparameters (in adaptive routine).
        rotation_gates (str): type of 1q gates to use in the templates, e.g. 'xz' or 'xyz'.
    """
    num_samples: int = 100
    method: str = 'adam'
    learning_rate: float = 0.1
    num_gd_iterations: int = 2000
    entangling_angles_distribution: str = 'uniform'
    entry_loss: float = 1e-3
    target_loss: float = 1e-6
    learning_rate_at_verification: float = 0.01
    num_gd_iterations_at_verification: int = 5000
    random_seed: int = 0
    rotation_gates: str = 'xyz'


@dataclass
class StaticOptions(BasicOptions):
    """ Options for static synthesis.

    Attributes:
        num_entangling_blocks (int): total number of CP gates in the template circuit.
        r (float): regularization weight.
        accepted_num_2q_gates (int): verify prospective decompositions if their CZ count is below.

    """
    num_entangling_blocks: int = -1
    r: float = 0.00055
    accepted_num_2q_gates: int = -1

    def __post_init__(self):
        if self.num_entangling_blocks == -1:
            raise TypeError("Missing required argument 'num_entangling_blocks'")
        if self.accepted_num_2q_gates == -1:
            raise TypeError("Missing required argument 'accepted_num_2q_gates'")


@dataclass
class AdaptiveOptions(BasicOptions):
    """ Options for adaptive synthesis.

    min_num_entangling_blocks (int): lower bound on the number of CP gates in templates.
    max_num_entangling_blocks (int): upper bound on the number of CP gates in templates.
    r_mean (flaot): mean value for the regularization weight.
    r_variance (float): variance for the regularization weight (lognormal distribution).
    max_evals (int): how many hyperparameter configurations to evaluate.
    target_num_2q_gates (int): desired number of CZ gates in the final  decomposition.
    stop_if_target_reached (bool): continue or stop if `target_num_2q_gates` has been reached.
    keep_logs (bool): whether to keep extensive logs with raw evaluations.
    """
    min_num_entangling_blocks: int = -1
    max_num_entangling_blocks: int = -1
    r_mean: float = 0.00055
    r_variance: float = 0.5
    max_evals: int = 100
    target_num_2q_gates: int = 0
    stop_if_target_reached: bool = False
    keep_logs: bool = False

    def __post_init__(self):
        if self.min_num_entangling_blocks == -1:
            raise TypeError("Missing required argument 'min_num_entangling_blocks'")
        if self.max_num_entangling_blocks == -1:
            raise TypeError("Missing required argument 'max_num_entangling_blocks'")

    def get_static(self, num_2q_gates, r):
        default_static_dict = asdict(BasicOptions())
        options_dict = asdict(self)
        basic_dict = {key: value for key, value in options_dict.items() if key in default_static_dict}
        basic_dict['num_entangling_blocks'] = num_2q_gates
        basic_dict['r'] = r
        basic_dict['accepted_num_2q_gates'] = None
        return StaticOptions(**basic_dict)


@dataclass
class Results:
    """Store and manipulate results of static and adaptive routines.

    Attributes:
        loss_function (callable): Function of a unitary matrix that was minimized.
        layer (list): List specifying connectivity of the qubits.
        label (str): Reminder about what was decomposed, e.g. '3q Toffoli on linear topology'.
        trials : hyperopt trials instance (if results include adaptive stage).
        decompositions (tuple): all decompositions, in order that they were found.
        save_to (str): Where to save the results. Default is 'results/label'

    Methods:
        save(): save results to location specified by `save_to`.
        load(path): load results from the path (staticmethod).
        best_hyperparameters(): list all pairs [num_entangling_blocks, r] tried in adaptive decomposition in order of increasing score.
        plot_trials(): visualize trials on a 2d scatter plot.
    """

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

    def best_hyperparameters(self):
        """Returns list of pairs [num_entangling_blocks, r] ordered by increasing score."""

        results = self.trials.results
        results = sorted(results, key=lambda res: res['loss'])
        hyperparams = [[res['num_entangling_blocks'], res['r']] for res in results]
        return hyperparams

    def plot_trials(self):
        results = self.trials.results

        num_list = jnp.array([res['num_entangling_blocks'] for res in results])
        r_list = jnp.array([res['r'] for res in results])
        loss_list = jnp.array([res['loss'] for res in results])

        finite_num_list = num_list[loss_list < jnp.inf]
        finite_r_list = r_list[loss_list < jnp.inf]
        finite_loss_list = loss_list[loss_list < jnp.inf]

        inf_num_list = num_list[loss_list >= jnp.inf]
        inf_r_list = r_list[loss_list >= jnp.inf]

        n_best, r_best = self.best_hyperparameters()[0]

        plt.scatter(finite_num_list, finite_r_list, c=finite_loss_list, cmap='jet', edgecolors='black')
        plt.colorbar()
        plt.scatter(inf_num_list, inf_r_list, marker='x', color='red')
        plt.scatter([n_best], [r_best], marker='*', facecolors='gold', edgecolors='black', s=[250])

        plt.xlabel('Number of CP gates')
        plt.ylabel('r: regularization weight')
        plt.title('Score')


class Synthesize:
    """Automated synthesis of unitary matrices into CZ+1q gates.

    Attributes:
        layer: sequence of pairs corresponding to connections between qubits, ex: [[0,1], [1,2], [0,2]].
        num_qubits: number of qubits in the circuit to synthesize.
        unitary_loss_function: function of a unitary matrix that is to be minimized.
        target_unitary: if provided `unitary_loss_function` is set to Hilbert-Schmidt distance to the `target_unitary`.
        target_state: if provided `unitary_loss_function` is set to 1-overlap squared with the `target_state`.
        label:  reminder about what is synthesised, e.g. '3q Toffoli on linear topology'.
        regularization_func: function determining penalty cost for cp-angles.

    Methods:
        static(options): synthesis with a fixed CP template and regularization weight.
        adaptive(options): synthesis with CP template length and regularization weight optimized by hyperopt.
    """

    def __init__(
            self,
            entanglging_gate_name,
            layer,
            unitary_loss_func=None,
            target_unitary=None,
            label=None,
            regularization_options=regularization_options_cp):

        self.entanglging_gate_name = entanglging_gate_name
        self.layer = layer
        self.num_qubits = num_qubits_from_layer(self.layer)

        self.target_unitary = target_unitary
        if unitary_loss_func is not None:
            self.unitary_loss_func = unitary_loss_func
        else:
            assert self.target_unitary is not None, 'Neither unitary loss function nor target unitary is provided.'
            assert self.target_unitary.shape == (2**self.num_qubits, 2**self.num_qubits), 'Number of qubits in target unitary and layer do not match.'
            self.unitary_loss_func = lambda u: cost_HST(u, self.target_unitary)

        self.label = label
        self.regularization_options = regularization_options

    @staticmethod
    def _generate_initial_angles(key, num_angles, cp_mask, cp_dist='uniform', batch_size=1):
        key, *subkeys = random.split(key, num=batch_size + 1)
        initial_angles_array = jnp.array(
            [random_cp_angles(num_angles, cp_mask, cp_dist=cp_dist, key=k)
             for k in subkeys])

        return initial_angles_array

    @staticmethod
    def _plot_raw(res):
        plt.plot(res['regloss'], label='regloss')
        plt.plot(res['loss'], label='loss')
        plt.plot(res['reg'], label='reg')
        plt.yscale('log')
        plt.legend()

    def _generate_ansatz(self, static_options):
        anz = Ansatz(
            self.num_qubits,
            self.entanglging_gate_name,
            fill_layers(self.layer, static_options.num_entangling_blocks),
            static_options.rotation_gates)
        return anz

    def _generate_raw(self, static_options, initial_angles_array=None, keep_history=False):

        anz = Synthesize._generate_ansatz(self, static_options)
        loss_func = lambda angles: self.unitary_loss_func(anz.unitary(angles))

        def regulator(angs):
            reg_weight = static_options.r
            reg_func = self.regularization_options.regularization_func
            reg_angles = angs * anz.entangling_mask
            return reg_weight * vmap(reg_func)(reg_angles).sum()

        key = random.PRNGKey(static_options.random_seed)
        if initial_angles_array is None:
            initial_angles_array = Synthesize._generate_initial_angles(
                key,
                anz.num_angles,
                anz.entangling_mask,
                cp_dist=static_options.entangling_angles_distribution,
                batch_size=static_options.num_samples)

        raw_results = mynimize_repeated(
            loss_func,
            anz.num_angles,
            method=static_options.method,
            learning_rate=static_options.learning_rate,
            num_iterations=static_options.num_gd_iterations,
            initial_params_batch=initial_angles_array,
            regularization_func=regulator,
            u_func=anz.unitary,  # For 'adam' this won't be used, but it will e.g. for 'natural adam'.
            keep_history=keep_history
        )

        return raw_results

    def _evaluate_raw(self, raw_results, static_options):

        anz = Synthesize._generate_ansatz(self, static_options)

        below_entry_loss_results = filter_raw_results(
            raw_results,
            anz.entangling_mask,
            self.regularization_options.count_func,
            float('inf'),  # At this stage we only filter by convergence, not by the number of gates.
            static_options.entry_loss,
            self.regularization_options.angle_threshold
        )

        return below_entry_loss_results

    def _initialize_results(self, save_results, save_to):
        results = Results(self.unitary_loss_func, self.layer, label=self.label)
        if save_results:
            assert self.label or save_to, \
                'To save results on a disk either `label` or `save_to` must be provided. ' \
                'If you insist on not saving the results call the decomposition routine with `save_results=False` flag.'
            if save_to:
                results.save_to = save_to

            # Try to load existing results.
            try:
                results = Results.load(results.save_to)
            except FileNotFoundError:
                pass

        return results

    def _make_decomposition(self, u_func, circ_func, best_angs, static_options=None, adaptive_options=None, circuit=None):
        if circuit is None:
            circuit = Decomposition._from_verified_result(
                self.unitary_loss_func,
                u_func,
                circ_func,
                best_angs,
                self.entanglging_gate_name,
                self.regularization_options,
                self.label,
                )

        d = circuit
        d._static_options = static_options
        d._adaptive_options = adaptive_options
        d._decomposer = self
        return d

    def _verify_result(self, res, static_options, adaptive_options=None):
        anz = Synthesize._generate_ansatz(self, static_options)
        success, num_2q_gates, circ, u, best_angles = verify_result(
            res,
            anz,
            self.unitary_loss_func,
            static_options,
            self.regularization_options,
            keep_history=False)

        if success:
            new_decomposition = Synthesize._make_decomposition(self, u, circ, best_angles,
                                                               static_options=static_options,
                                                               adaptive_options=adaptive_options)
        else:
            new_decomposition = None

        return success, new_decomposition

    def _print_decomposer(self):
        print(f"\nStarting decomposer '{self.label}' with entangling gate '{self.entanglging_gate_name}' and layer {self.layer} ")

    def static(self, static_options, save_results=True, save_to=''):
        """ Unitary synthesis with a fixed CP template and regularization weight.

        Args:
            static_options: instance of `StaticOptions`.
            save_results: whether to save results on disk or not.
            save_to: if provided overwrites default saving path.

        Returns:
            results: instance of `Results` class.
        """

        Synthesize._print_decomposer(self)
        results = Synthesize._initialize_results(self, save_results, save_to)

        print('\nStarting decomposition routine with the following options:')
        print('\n', static_options)

        print('\nComputing raw results...')
        raw_results = Synthesize._generate_raw(self, static_options)

        print('\nSelecting prospective results...')
        raw = Synthesize._evaluate_raw(self, raw_results, static_options)
        prospective_results = raw
        prospective_results = [res for res in prospective_results if res[0] <= static_options.accepted_num_2q_gates]

        successful_results = []
        if prospective_results:
            print(f'\nFound {len(prospective_results)}. Verifying...')

            for num_2q_gates, res in tqdm(prospective_results):
                success, new_decomposition = Synthesize._verify_result(self, res, static_options)
                if success:
                    successful_results.append(new_decomposition)

            if successful_results:
                successful_results = sorted(successful_results, key=lambda d: d.gate_count_2q)
                print(f'\n{len(successful_results)} successful. 2q counts are:')
                print([d.gate_count_2q for d in successful_results])
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
        """Unitary synthesis with the length of CP template and regularization weight optimized by hyperopt.

        Args:
            options: instance of `AdaptiveOptions` class.
            save_results: whether to save results on disk or not.
            save_to: if provided overwrites default saving path.
        Returns:
            results: instance of `Results` class.
        """

        def objective_from_cz_distribution(random_seed, search_params):
            """Evalueates quality of the hyperparameter configuration.

            Params:
                random_seed: seed to be used for generating random initial angles and setting hyperopt state.
                search_params: params subject to hyperopt optimization."""

            num_2q_blocks, r = search_params
            tqdm.write(f'\nnum_entangling_blocks: {num_2q_blocks}, r: {r}')
            static_options = options.get_static(num_2q_blocks, r)
            static_options.random_seed = random_seed

            raw_results = Synthesize._generate_raw(self, static_options)

            evaluated_results = Synthesize._evaluate_raw(
                self,
                raw_results,
                static_options
            )

            counts_2q_gates = [res[0] for res in evaluated_results]

            # Score is defined as the weighted sum of cz counts of all successful results.
            # For convenience it is normalized on the number of samples and presented in log scale.

            score = 2 ** (-jnp.array(counts_2q_gates, dtype=jnp.float32))
            score = (score.sum() / options.num_samples)
            score = jnp.log2(score)

            tqdm.write(f'score: {-score}, cz counts of prospective results: {counts_2q_gates}')

            return_dict = {
                'loss': -score,
                'status': STATUS_OK,
                'random_seed': random_seed,
                'cz_counts': counts_2q_gates,
                'num_entangling_blocks': num_2q_blocks,
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
                hp.quniform('num_entangling_blocks', options.min_num_entangling_blocks, options.max_num_entangling_blocks,
                            1)),
            hp.lognormal('r', jnp.log(options.r_mean), options.r_variance)
        ]

        # Loading existing trials and decompositions.
        results = Synthesize._initialize_results(self, save_results, save_to)

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
            scoreboard = set([d.gate_count_2q for d in results.decompositions])
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

            current_best_2q_count = scoreboard[0]

            last_result = trials.results[-1]
            num_2q_gates = last_result['num_entangling_blocks']
            r = last_result['r']
            successful_results = last_result['prospective_decompositions']
            if not options.keep_logs:
                last_result.pop('prospective_decompositions')

            results_to_verify = [[num_2q_gates, res] for count, res in successful_results if count < current_best_2q_count]

            if len(results_to_verify):
                tqdm.write(
                    f'\nFound {len(results_to_verify)} decompositions potentially improving the current best count {current_best_2q_count}, verifying...')
            else:
                tqdm.write(
                    f'\nFound no decompositions potentially improving the current best count {current_best_2q_count}.')

            for num_2q_gates, res in results_to_verify:
                success, new_decomposition = Synthesize._verify_result(
                    self,
                    res,
                    options.get_static(num_2q_gates, r),
                    adaptive_options=options)
                if success:
                    successful_results.append(new_decomposition)


                if success:
                    tqdm.write(f'\nFound a new decomposition with {new_decomposition.gate_count_2q} gates.')
                    scoreboard.insert(0, new_decomposition.gate_count_2q)
                    results.decompositions = list(results.decompositions) + [new_decomposition]
                    results.save()
                    break
            else:
                if results_to_verify:
                    tqdm.write('\nNone of prospective decompositions passed.')

            if options.stop_if_target_reached and scoreboard[0] <= options.target_num_2q_gates:
                tqdm.write('\nTarget number of gates reached.')
                break

        return results

