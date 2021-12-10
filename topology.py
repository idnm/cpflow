def sequ_layer(num_qubits):
    return [[i, j] for i in range(num_qubits) for j in range(i + 1, num_qubits)]


def chain_layer(num_qubits):
    return [[i, i+1] for i in range(num_qubits-1)]


def fill_layers(layer, depth):
    num_complete_layers = depth // len(layer)
    complete_layers = [layer, num_complete_layers]
    incomplete_layer = layer[:depth % len(layer)]

    return {'layers': complete_layers, 'free': incomplete_layer}
