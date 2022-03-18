from matrix_utils import *
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from qiskit.circuit.library import *
from trigonometric_utils import *
from optimization import mynimize_repeated


def check_approximation(circuit, new_circuit, loss=1e-4):
    assert disc2(Operator(circuit).data,
                 Operator(new_circuit).data) < loss, 'Difference between projected and original circuit too large.'
    print('ver succ')


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

    check_approximation(circuit, new_circuit)

    return new_circuit


def cp_to_cz_gate(gate, cp_threshold):
    cp_angle = gate.params[0]
    if jnp.abs(cp_angle) < cp_threshold:
        qc = QuantumCircuit(2)
        qc.i([0, 1])
        gate = qc.to_gate(label='id')
    elif jnp.abs(cp_angle - jnp.pi) < cp_threshold:
        gate = CZGate()
    else:
        qc = QuantumCircuit(2)
        qc.cp(cp_angle, 0, 1)
        qc = transpile(qc, basis_gates=['cz', 'rz', 'rx'], optimization_level=3)
        gate = qc.to_gate(label='cp_trans')

    return gate


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



