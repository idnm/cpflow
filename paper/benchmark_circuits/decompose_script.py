import sys
sys.path.append('/home/idnm/Programming projects/jax_circuits/')

from jax_circuits import *

options = AdaptiveOptions(
    min_num_cp_gates=20,
    max_num_cp_gates=60,
    num_samples=1000,
    max_evals=50)

for qasm in os.listdir('Table 1'):
    qc = QuantumCircuit.from_qasm_file(qasm)
    u_target = Operator(qc.reverse_bits()).data

    layer = connected_layer(5)
    decomposer = Synthesize(layer, target_unitary=u_target, label=f'res_{qasm[:-5]}')
    results = decomposer.adaptive(options)

for qasm in os.listdir('Table 3'):
    qc = QuantumCircuit.from_qasm_file(qasm)
    u_target = Operator(qc.reverse_bits()).data

    layer = chain_layer(5)
    decomposer = Synthesize(layer, target_unitary=u_target, label=f'res_{qasm[:-5]}')
    results = decomposer.adaptive(options)

for qasm in os.listdir('Table 4'):
    qc = QuantumCircuit.from_qasm_file(qasm)
    u_target = Operator(qc.reverse_bits()).data

    layer = connected_layer(5)
    decomposer = Synthesize(layer, target_unitary=u_target, label=f'res_{qasm[:-5]}')
    results = decomposer.adaptive(options)
