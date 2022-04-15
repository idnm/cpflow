"""Constructing layers complying with topological restrictions."""

import jax.numpy as jnp
from jax import random


def connected_layer(num_qubits):
    return [[i, j] for i in range(num_qubits) for j in range(i + 1, num_qubits)]


def chain_layer(num_qubits):
    return [[i, i+1] for i in range(num_qubits-1)]


def fill_layers(layer, depth):
    num_complete_layers = depth // len(layer)
    complete_layers = [layer, num_complete_layers]
    incomplete_layer = layer[:depth % len(layer)]

    return {'layers': complete_layers, 'free': incomplete_layer}


def random_placements(num_qubits, num_gates, coupling_map=None, key=random.PRNGKey(0)):
    placements = []
    for _ in range(num_gates):
        key, subkey = random.split(key)
        placements.append(random_placement(num_qubits, coupling_map=coupling_map, key=subkey))
    return placements


def random_placement(num_qubits, coupling_map=None, key=random.PRNGKey(0)):
    i, j = random.choice(key, jnp.arange(num_qubits), (2,), replace=False)
    return [i, j]


def num_qubits_from_layer(layer):
    # Number of qubits is the maximum index in the coupling map, plus 1.
    return max([item for sublist in layer for item in sublist]) + 1


