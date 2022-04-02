import sys
sys.path.append('/home/rqc-qit-0/nnemkov/jc_module')
# sys.path.append('/home/idnm/Programming projects/jax_circuits/')
from jax_circuits import *

unitary_loss_f = lambda u: disc_modulo_diagonal(u_toff4, u, 4, [3])
decomposer = Synthesize(connected_layer(4), unitary_loss_func=unitary_loss_f, label='relative_toff4_3q_conn')
options = StaticOptions(num_cp_gates=14, accepted_num_cz_gates=6, num_samples=500)

results = decomposer.static(options)