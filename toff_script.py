from cpflow import gates
from cpflow.main import *
from cpflow.topology import *
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.circuit import Parameter

import jax.numpy as jnp


u_target = u_toff4
layer = connected_layer(4)

decomposer = Synthesize('rzx', layer, target_unitary=u_target, label='toff4_conn_rzx')
decomposer.regularization_options = regularization_options_parametric

# static_options = StaticOptions(num_entangling_blocks=15, accepted_num_2q_gates=7, num_samples=200)
adaptive_options = AdaptiveOptions(min_num_entangling_blocks=10, max_num_entangling_blocks=30)

# results = decomposer.static(static_options) # Should take from one to five minutes.
results = decomposer.adaptive(adaptive_options)