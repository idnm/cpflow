"""Procedures to refine approximate circuits."""

import math
from fractions import Fraction
from functools import partial

from jax import jit
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import *
from qiskit.quantum_info import OneQubitEulerDecomposer
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Optimize1qGates

try:
    from qiskit.transpiler.passes import SolovayKitaevDecomposition
    skd_available = True
except ImportError:
    skd_available = False
    print('WARNING: the SolovayKitaevDecomposition plugin from dev version of qiskit is not installed.')
    print('To intall it run "pip install git+https://github.com/LNoorl/qiskit-terra@d2e0dc1185ccc3b0c9957e3d7d9bc610dede29d4" ')
    print('You might also need to install the Rust compiler https://www.rust-lang.org/tools/install .')

from cpflow.circuit_assembly import qiskit_circ_to_jax_unitary
from cpflow.cp_utils import constrained_function
from cpflow.matrix_utils import *
from cpflow.optimization import mynimize_repeated
from cpflow.trigonometric_utils import *


def check_approximation(circuit, new_circuit, loss=1e-5):
    l = cost_HST(Operator(circuit).data, Operator(new_circuit).data)
    if not l < loss:
        raise ValueError(f'Difference {l} between modified and original circuit is above threshold {loss}.')


def check_loss(circuit, unitary_loss_func, threshold_loss=1e-5):
    loss = unitary_loss_func(Operator(circuit.reverse_bits()).data)
    if not loss < threshold_loss:
        raise ValueError(f'Circuit loss {loss} is above threshold {threshold_loss}.')


def cp_to_cz_circuit(circuit, cp_threshold=0.2):
    new_data = []
    for gate, qargs, cargs in circuit.data:
        if gate.name == 'cp':
            new_gate = cp_to_cz_gate(gate, cp_threshold)
        else:
            new_gate = gate

        new_data.append((new_gate, qargs, cargs))

    new_circuit = circuit.copy()
    new_circuit.data = new_data
    new_circuit = new_circuit.decompose(gates_to_decompose=['id', 'cp_trans'])

    check_approximation(circuit, new_circuit, loss=1e-5)

    return new_circuit


def cp_to_cz_gate(gate, cp_threshold):
    cp_angle = gate.params[0]
    if jnp.abs(cp_angle) <= cp_threshold:
        qc = QuantumCircuit(2)
        gate = qc.to_gate(label='id')
    elif jnp.abs(cp_angle - jnp.pi) <= cp_threshold:
        gate = CZGate()
    else:
        qc = QuantumCircuit(2)
        qc.cp(cp_angle, 0, 1)
        qc = transpile(qc, basis_gates=['cz', 'rz', 'rx'], optimization_level=3)
        gate = qc.to_gate(label='cp_trans')

    return gate


def reduce_all_1q_angles(loss_func, initial_angles, wires, threshold=1e-5):
    if len(initial_angles) == 0:
        return initial_angles

    new_angles = reduce_first_1q_angle(loss_func, initial_angles, wires, threshold)
    new_loss_func = constrained_function(loss_func, new_angles[:1], [0], jax_numpy=False)

    return jnp.concatenate([new_angles[:1], reduce_all_1q_angles(new_loss_func, new_angles[1:], wires[1:], threshold=threshold)])


def reduce_first_1q_angle(loss_func, angles, wires, threshold):
    new_angles = angles
    if loss_func(angles.at[0].set(0)) < threshold:
        new_angles = new_angles.at[0].set(0)
        return new_angles

    else:
        for i in range(1, len(angles)):
            can_reduce, new_angles = can_reduce_two_angles(loss_func, angles, 0, i, wires[0], wires[i], threshold)
            if can_reduce:
                return new_angles

    return angles


def can_reduce_two_angles(loss_func, angles, i, j, wi, wj, threshold):
    if wi != wj:
        return False, angles

    for sign in [-1, 1]:
        new_angles = angles
        new_angles = new_angles.at[j].set(angles[j] + sign * angles[i])
        new_angles = new_angles.at[i].set(0)
        if loss_func(new_angles) < threshold:
            return True, new_angles
    else:
        return False, angles


def replace_angles_in_circuit(qc, angles):
    new_qc = qc.copy()
    angles = angles.copy()

    new_data = []
    i = 0
    for gate, qregs, cregs in new_qc.data:
        if gate.name in ['rx', 'ry', 'rz']:
            gate.params = [angles[i]]
            i += 1
        new_data.append((gate, qregs, cregs))

    new_qc.data = new_data

    return new_qc


def singeq_to_U(gate):
    qc = QuantumCircuit(1)
    qc.append(gate, [0])

    qc = transpile(qc, basis_gates=['u'])

    return qc.to_gate(label='uu')


def convert_to_U(circuit):
    qc = circuit.copy()
    new_data = []
    for gate, qargs, cargs in qc.data:
        if len(qargs) == 1:
            new_gate = singeq_to_U(gate)
        else:
            new_gate = gate
        new_data.append((new_gate, qargs, cargs))

    qc.data = new_data
    qc = qc.decompose('uu')

    pass_ = Optimize1qGates(basis=['u'])
    pm = PassManager(pass_)
    qc = pm.run(qc)

    check_approximation(circuit, qc)
    return qc


def U_to_ZXZ(gate):
    qc = QuantumCircuit(1)
    qc.append(gate, [0])

    euler = OneQubitEulerDecomposer(basis='ZXZ')
    x1, z2, z1 = euler.angles(Operator(qc).data)

    qc = QuantumCircuit(1)
    qc.rz(z1, 0)
    qc.rx(x1, 0)
    qc.rz(z2, 0)

    return qc.to_gate(label='zxz')


def convert_to_ZXZ(circuit):
    qc = convert_to_U(circuit)
    new_data = []
    for gate, qargs, cargs in qc.data:
        if gate.name == 'u':
            new_gate = U_to_ZXZ(gate)
        else:
            new_gate = gate
        new_data.append((new_gate, qargs, cargs))

    qc.data = new_data
    check_approximation(circuit, qc)
    return qc.decompose('zxz')


def reduce_angles(circuit, unitary_loss_func, reduce_threshold=1e-5, cp_threshold=0.01):

    qc = circuit.copy()
    qc = cp_to_cz_circuit(qc, cp_threshold=cp_threshold)

    qc = convert_to_ZXZ(qc)

    u, angles, wires = qiskit_circ_to_jax_unitary(qc)
    loss_f = lambda angs: unitary_loss_func(u(angs))
    loss_f = jit(loss_f)

    reduced_angs = reduce_all_1q_angles(loss_f, jnp.array(angles), wires, threshold=reduce_threshold)
    qc = replace_angles_in_circuit(qc, vmap(bracket_angle)(reduced_angs))

    check_loss(qc, unitary_loss_func, threshold_loss=reduce_threshold)

    return qc


def rationalize_all_rgates(circuit, max_denominator=32, angle_threshold=1e-3):
    new_data = []
    for gate, qargs, cargs in circuit.data:
        if gate.name in ['rx', 'ry', 'rz']:
            new_gate = rationalize_rgate(gate, max_denominator, angle_threshold)
        else:
            new_gate = gate
        new_data.append((new_gate, qargs, cargs))

    new_circuit = circuit.copy()
    new_circuit.data = new_data

    check_approximation(circuit, new_circuit)

    return new_circuit


def all_rgates_are_rational(circuit, power):
    """Test if all rotation angles in the circuit are of the form pi*n/k where n is integer and k = 2**power"""

    for gate, qargs, cargs in circuit.data:
        if gate.name in ['rx', 'ry', 'rz']:
            if not angle_is_rational(gate.params[0], power):
                return False

    return True


def angle_is_rational(a, power):
    f = Fraction(a/jnp.pi).limit_denominator(2**power)
    if jnp.abs(jnp.pi*f - a) < 1e-6 and math.log2(f.denominator).is_integer():
        return True
    else:
        return False


def rationalize_rgate(gate, max_denominator, angle_threshold):
    angle = gate.params[0]
    rational_angle = jnp.pi * Fraction.from_float(angle / jnp.pi).limit_denominator(max_denominator)
    if jnp.abs(rational_angle - angle) < angle_threshold:
        new_angle = rational_angle
    else:
        new_angle = angle

    new_gate = gate.copy()
    new_gate.params = [new_angle]
    return new_gate


def solovay_kitaev(circuit, recursion_degree=0, recursion_depth=5):
    qc = circuit.copy()
    basis_gates = [TGate(), TdgGate(), SGate(), SdgGate(), HGate()]
    skd = SolovayKitaevDecomposition(recursion_degree=recursion_degree, basis_gates=basis_gates, depth=recursion_depth)
    qc = skd(qc)

    check_approximation(qc, circuit)

    return qc


def gate_filter(gate_names, d):
    gate, qargs, cargs = d
    if gate.name in gate_names:
        return True
    else:
        return False


def gates_count(gate_names, circuit):
    ops = circuit.count_ops()
    total = 0
    for gate_name in gate_names:
        if gate_name in ops:
            total += ops[gate_name]
    return total


def gates_depth(gate_names, circuit):
    return circuit.depth(filter_function=partial(gate_filter, gate_names))


def refine(
        circuit,
        unitary_loss_func,
        max_denominator=32,
        angle_threshold=1e-3,
        cp_threshold=0.01,
        reduce_threshold=1e-5,
        recursion_degree=0,
        recursion_depth=5,
        verbose=False):

    qc = circuit.copy()
    refine_type = 'Approximate'
    t_count = None
    t_depth = None

    try:
        qc = reduce_angles(qc, unitary_loss_func, reduce_threshold=reduce_threshold, cp_threshold=cp_threshold)
        qc = remove_zero_rgates(qc)
        refine_type = 'Approximate'
    except ValueError as e:
        if verbose:
            print(e)
        return qc, refine_type, t_count, t_depth

    try:
        qc = rationalize_all_rgates(qc, max_denominator=max_denominator, angle_threshold=angle_threshold)
        qc = remove_zero_rgates(qc)
        if all_rgates_are_rational(qc, int(jnp.log2(max_denominator))):
            refine_type = 'Rational'
    except ValueError as e:
        if verbose:
            print(e)
        return qc, refine_type, t_count, t_depth

    if skd_available:
        try:
            qc_sk = solovay_kitaev(qc, recursion_degree=recursion_degree, recursion_depth=recursion_depth)
            t_count = gates_count(['t', 'tdg'], qc_sk)
            t_depth = gates_depth(['t', 'tdg'], qc_sk)

            qc = reduce_angles(qc_sk, unitary_loss_func, reduce_threshold=reduce_threshold, cp_threshold=cp_threshold)
            qc = rationalize_all_rgates(qc, max_denominator=max_denominator, angle_threshold=angle_threshold)
            qc = remove_zero_rgates(qc)

            refine_type = 'Clifford+T'
        except ValueError as e:
            if verbose:
                print(e)
            return qc, refine_type, t_count, t_depth

    return qc, refine_type, t_count, t_depth


def lasso_angles(loss_function, angles, eps=1e-5, threshold_loss=1e-6):

    penatly_f = lambda angs: eps * (jnp.abs(vmap(bracket_angle)(angs))).sum()

    res = mynimize_repeated(
        loss_function,
        len(angles),
        regularization_func=penatly_f,
        num_repeats=1,
        method='adam',
        learning_rate=0.01,
        initial_params_batch=angles,
        num_iterations=10000)

    best_i = jnp.argmin(res['regloss'])
    best_angs = res['params'][best_i]
    assert res['loss'][best_i] <= threshold_loss, 'L1 regularization was not successful.'

    return best_angs


def project_circuit(circuit, threshold):
    """Replaces gates with parameters numerically close to refernce values by these values."""
    new_data = circuit.data.copy()
    projected_data = [(project_gate(gate, threshold), qargs, cargs) for gate, qargs, cargs in new_data]
    new_data = []
    for gate, qargs, cargs in projected_data:
        if type(gate) is list:
            for g in gate:
                new_data.append((g, qargs, cargs))
        else:
            new_data.append((gate, qargs, cargs))

    new_circuit = circuit.copy()
    new_circuit.data = new_data

    check_approximation(circuit, new_circuit)

    return new_circuit


rx_projections = {
    0: IGate(),
    jnp.pi: XGate(),
    -jnp.pi: XGate(),
    jnp.pi / 2: [HGate(), SGate(), HGate()],
    -jnp.pi / 2: [HGate(), SGate().inverse(), HGate()],
    jnp.pi / 4: [HGate(), TGate(), HGate()],
    -jnp.pi / 4: [HGate(), TGate().inverse(), HGate()],
    3 * jnp.pi / 4: [XGate(), HGate(), TGate().inverse(), HGate()],
    -3 * jnp.pi / 4: [XGate(), HGate(), TGate(), HGate()]}

rz_projections = {
    0: IGate(),
    jnp.pi: ZGate(),
    -jnp.pi: ZGate(),
    jnp.pi / 2: SGate(),
    -jnp.pi / 2: SGate().inverse(),
    jnp.pi / 4: TGate(),
    -jnp.pi / 4: TGate().inverse()
}


def project_gate(gate, threshold):
    """Projects 'rx' or 'ry' if their parameters are below `threshold` distance away from predefined reference values."""

    if gate.name == 'rx':
        projections = rx_projections
    elif gate.name == 'rz':
        projections = rz_projections
    else:
        return gate

    angle = gate.params[0]
    for special_angle, special_gate in projections.items():
        if jnp.abs(angle - special_angle) < threshold:
            return special_gate

    return gate


def remove_zero_rgates(circuit):
    """Removes rotation gates with zero angles."""
    new_circuit = circuit.copy()
    new_data = []
    for gate, qregs, cregs in circuit.data:
        if gate.name in ['rx', 'ry', 'rz'] and jnp.abs(gate.params[0]) < 1e-5:
            qc = QuantumCircuit(1)
            new_gate = qc.to_gate(label='id')
        else:
            new_gate = gate
        new_data.append((new_gate, qregs, cregs))

    new_circuit.data = new_data
    new_circuit = new_circuit.decompose('id')

    check_approximation(circuit, new_circuit)

    return new_circuit


def move_all_rgates(circuit):
    """Moves all rotations gates as far to the right as possible."""

    new_circuit = circuit.copy()
    new_circuit_data = new_circuit.data
    for qubit in circuit.qubits:
        new_circuit_data = move_all_rgates_along_wire(new_circuit_data, qubit)

    new_circuit.data = new_circuit_data
    check_approximation(circuit, new_circuit)

    return new_circuit


def move_all_rgates_along_wire(data, qubit):
    """Moves all rotation gates at a given wire as far to the right as possible."""

    if not contains_rgate_at_wire(data, qubit):
        return data

    data = move_last_rgate_along_wire(data, qubit)

    i = get_last_rgate_index(data, qubit)

    return move_all_rgates_along_wire(data[:i], qubit) + data[i:]


def move_last_rgate_along_wire(data, qubit):
    """Moves last rotation gate at a given wire as far to the right as possible."""
    i = get_last_rgate_index(data, qubit)

    return data[:i] + move_single_rgate_along_wire(data[i:], qubit)


def move_single_rgate_along_wire(data, qubit):
    """Given a wire starting with a rotation gate moves this gate as far to the right as possible."""
    if len(data) == 1:
        return data

    move_successful, new_data = move_rgate_along_wire_once(data)
    if move_successful:
        return [new_data[0]] + move_single_rgate_along_wire(new_data[1:], qubit)
    else:
        return data


def move_rgate_along_wire_once(data):
    """Given a wire starting from a rotation gate attempts to commute this gate past the next one."""

    r_gate, r_qargs, r_cargs = data[0]
    next_gate, next_qargs, next_cargs = data[1]

    move_successful = True

    if r_gate.name == 'rz':
        if r_qargs != next_qargs or next_gate.name in ['id', 'z', 's', 't', 'sdg', 'tdg']:
            new_r_gate = r_gate
        elif next_gate.name == 'x':
            new_r_gate = r_gate
            new_r_gate.params = [-r_gate.params[0]]  # Commutation with X gate flips the sign in RZ gate.
        elif next_gate.name == 'h':  # Commutation with H changes Z to X
            new_r_gate = RXGate(r_gate.params[0])
        else:
            move_successful = False

    elif r_gate.name == 'rx':
        if r_qargs[0] not in next_qargs or next_gate.name in ['id', 'x']:
            new_r_gate = r_gate
        elif r_qargs == next_qargs:
            if next_gate.name == 'z':
                new_r_gate = r_gate
                new_r_gate.params = [-r_gate.params[0]]  # Commutation with Z gate flips the sign in RX gate.
            elif next_gate.name == 'h':  # Commutation with H changes X to Z
                new_r_gate = RZGate(r_gate.params[0])
            elif next_gate.name == 's':
                new_r_gate = RYGate(r_gate.params[0])  # XS=-SY
            elif next_gate.name == 'sdg':
                new_r_gate = RYGate(-r_gate.params[0])  # XS^*=S^*Y
            else:
                move_successful = False
        else:
            move_successful = False

    elif r_gate.name == 'ry':
        if r_qargs[0] not in next_qargs or next_gate.name == 'id':
            new_r_gate = r_gate
        elif r_qargs == next_qargs:
            if next_gate.name in ['x', 'z', 'h']:
                new_r_gate = r_gate
                new_r_gate.params = [-r_gate.params[0]]  # YZ=-ZY, YX=-XY, YH=-HY
            elif next_gate.name == 's':
                new_r_gate = RXGate(-r_gate.params[0])
            elif next_gate.name == 'sdg':
                new_r_gate = RXGate(r_gate.params[0])  # YS^*=-S^*X
            else:
                move_successful = False
        else:
            move_successful = False

    if move_successful:
        data01 = [data[1], (new_r_gate, r_qargs, r_cargs)]
    else:
        data01 = [data[0], data[1]]

    return move_successful, data01 + data[2:]


def merge_all_rgates(circuit):
    """Merges all adjacent 'rz' gates."""

    new_circuit = circuit.copy()
    new_data = new_circuit.data
    for qubit in circuit.qubits:
        new_data = merge_rgates_in_data(new_data, qubit)

    new_circuit.data = new_data

    check_approximation(circuit, new_circuit)

    return new_circuit


def merge_rgates_in_data(data, qubit):
    i = i_of_rgate_followed_by_same_rgate(data, qubit)
    if i is None:
        return data

    data0, data1 = data[:i], data[i:]
    r_gate, qargs, cargs = data1[0]
    next_r_gate, next_qargs, next_cargs = data1[1]

    r_gate_angle = r_gate.params[0]
    next_r_gate_angle = next_r_gate.params[0]

    new_r_gate = r_gate
    new_r_gate.params = [bracket_angle(r_gate_angle + next_r_gate_angle)] 

    data1 = [(new_r_gate, qargs, cargs)] + data1[2:]

    return data0 + merge_rgates_in_data(data1, qubit)


def get_indices_of_rgates_at_wire(data, qubit):
    i_list = []
    for i, (gate, qargs, cargs) in enumerate(data):
        if gate.name in ['rx', 'ry', 'rz']:
            if qubit == qargs[0]:
                i_list.append(i)
    return i_list


def contains_rgate_at_wire(data, qubit):
    return bool(get_indices_of_rgates_at_wire(data, qubit))


def get_last_rgate_index(data, qubit):
    return get_indices_of_rgates_at_wire(data, qubit)[-1]


def i_of_rgate_followed_by_same_rgate(data, qubit):
    all_gate_indices = get_indices_of_rgates_at_wire(data, qubit)
    for i, (gate, qargs, cargs) in enumerate(data[:-1]):
        next_gate, next_qargs, next_cargs = data[i+1]
        if i in all_gate_indices and i+1 in all_gate_indices:
            if gate.name == next_gate.name:
                return i

    return None



