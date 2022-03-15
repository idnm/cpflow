from matrix_utils import *
from qiskit.quantum_info import Operator
from qiskit.circuit.library import *


def check_approximation(circuit, new_circuit, loss=1e-3):
    assert disc2(Operator(circuit).data,
                 Operator(new_circuit).data) < loss, 'Difference between projected and original circuit too large.'
    print('ver succ')


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


def move_all_rz(circuit):
    """Moves all rz gates as far to the right as possible."""

    new_circuit = circuit.copy()
    new_circuit_data = new_circuit.data
    for qubit in circuit.qubits:
        new_circuit_data = move_all_rz_along_wire(new_circuit_data, qubit)

    new_circuit.data = new_circuit_data
    check_approximation(circuit, new_circuit)

    return new_circuit


def move_all_rz_along_wire(data, qubit):
    """Moves all rz gates at a given wire as far to the right as possible."""

    if not contains_rz_at_wire(data, qubit):
        return data

    data = move_last_rz_along_wire(data, qubit)
    i = get_last_rz_index(data, qubit)

    return move_all_rz_along_wire(data[:i], qubit) + data[i:]


def move_last_rz_along_wire(data, qubit):
    """Moves last rz gate at a given wire as far to the right as possible."""
    i = get_last_rz_index(data, qubit)

    return data[:i] + move_single_rz_along_wire(data[i:], qubit)


def move_single_rz_along_wire(data, qubit):
    """Given a wire starting with an rz gate moves this gate as far to the right as possible."""
    if len(data) == 1:
        return data

    move_successful, new_data = move_rz_along_wire_once(data)
    if move_successful:
        return [new_data[0]] + move_single_rz_along_wire(new_data[1:], qubit)
    else:
        return data


def move_rz_along_wire_once(data):
    """Given a wire starting from an rz gate attempts to commute this gate past the next one."""


    rz_gate, rz_qargs, rz_cargs = data[0]
    next_gate, next_qargs, next_cargs = data[1]

    if rz_qargs != next_qargs or next_gate.name == 'id':
        move_successful = True
        data01 = [data[1], data[0]]
    elif next_gate.name == 'x':
        move_successful = True
        rz_gate.params = [-rz_gate.params[0]]  # Commutation with X gate flips the sign in RZ gate.
        data01 = [data[1], data[0]]
    else:
        move_successful = False
        data01 = [data[0], data[1]]
    return move_successful, data01 + data[2:]


def get_indices_rz_at_wire(data, qubit):
    i_list = []
    for i, (gate, qargs, cargs) in enumerate(data):
        if gate.name == 'rz':
            if qubit == qargs[0]:
                i_list.append(i)
    return i_list


def contains_rz_at_wire(data, qubit):
    return bool(get_indices_rz_at_wire(data, qubit))


def get_last_rz_index(data, qubit):
    return get_indices_rz_at_wire(data, qubit)[-1]

