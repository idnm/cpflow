from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[5];\ncx q[2],q[1];\nt q[1];\ncx q[2],q[0];\nt q[0];\nh q[2];\nt q[2];\nt q[3];\ncx q[0],q[3];\ncx q[2],q[0];\ntdg q[0];\ncx q[3],q[2];\nt q[2];\ncx q[3],q[0];\ntdg q[0];\ncx q[2],q[0];\ntdg q[3];\ncx q[3],q[2];\ncx q[0],q[3];\nh q[0];\nt q[0];\nh q[2];\nh q[2];\nt q[2];\nt q[3];\nt q[4];\ncx q[4],q[1];\ncx q[0],q[4];\ncx q[1],q[0];\nt q[0];\ntdg q[4];\ncx q[1],q[4];\ntdg q[1];\ntdg q[4];\ncx q[0],q[4];\ncx q[1],q[0];\nh q[0];\nt q[0];\ncx q[0],q[3];\ncx q[2],q[0];\ntdg q[0];\ncx q[3],q[2];\nt q[2];\ncx q[3],q[0];\ntdg q[0];\ncx q[2],q[0];\ntdg q[3];\ncx q[3],q[2];\ncx q[0],q[3];\nh q[0];\nt q[0];\nh q[2];\nh q[2];\nt q[2];\ncx q[4],q[1];\nt q[1];\nt q[4];\ncx q[4],q[1];\ncx q[0],q[4];\ncx q[1],q[0];\nt q[0];\ntdg q[4];\ncx q[1],q[4];\ntdg q[1];\ntdg q[4];\ncx q[0],q[4];\ncx q[1],q[0];\nh q[0];\nt q[0];\ncx q[4],q[1];\ncx q[4],q[3];\nt q[3];\ncx q[3],q[0];\ncx q[2],q[3];\ncx q[0],q[2];\nt q[2];\ntdg q[3];\ncx q[0],q[3];\ntdg q[0];\ntdg q[3];\ncx q[2],q[3];\ncx q[0],q[2];\nh q[2];\nx q[2];\ncx q[3],q[0];\n'

qc = QuantumCircuit.from_qasm_str(qasm)
u_target = Operator(qc.reverse_bits()).data

import sys
sys.path.append('/home/rqc-qit-0/nnemkov/jc_module')
from cpflow.jax_circuits import *

layer = chain_layer(5)
decomposer = Synthesize(layer, target_unitary=u_target)
options = AdaptiveOptions(
    min_num_cp_gates=20,
    max_num_cp_gates=60,
    num_samples=1000,
    max_evals=100
)

results = decomposer.adaptive(options, save_to='data/alu-v0_26/')