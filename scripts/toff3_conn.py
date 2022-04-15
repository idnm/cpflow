import sys
sys.path.append('/home/idnm/Programming projects/jax_circuits')

from cpflow.cpflow import *
from cpflow.matrix_utils import *

layer = [[0,1]]
u_target = jnp.kron(y_mat, x_mat)

decomposer = Synthesize(layer, target_unitary=u_target, label='2qtest_ad')

adaptive_options = AdaptiveOptions(
    num_samples=3,
    min_num_cp_gates=1,
    max_num_cp_gates=5,
    max_evals=3)

results = decomposer.adaptive(adaptive_options)
