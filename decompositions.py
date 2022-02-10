from jax_circuits import*
from topology import *
from penalty import *
from cp_utils import *
import matplotlib.pyplot as plt
from scipy.stats import unitary_group
from qiskit import transpile
from qiskit.quantum_info import Operator
import pickle
import time


qc = QuantumCircuit(4)
qc.mct([0, 1, 2], 3)
u_toff4 = Operator(qc.reverse_bits()).data


anz = Ansatz(4, 'cp', fill_layers(sequ_layer(4), 24))

reg_options = {'r': 0.001,
               'function': 'linear',
               'ymax': 2,
               'xmax': jnp.pi/2,
               'plato': 0.05,
               'num_gates': 15,
               'cp_mask': anz.cp_mask}

key = random.PRNGKey(131)

num_samples = 400

start_time = time.time()

successful_results, failed_results = cp_decompose(u_toff4,
                                               anz,
                                               regularization_options=reg_options,
                                               num_samples=num_samples,
                                               key=key,
                                               disc_func=disc2,
                                               cp_dist='0',
                                               save_to='toff4_connected')

print("time:", time.time() - start_time)