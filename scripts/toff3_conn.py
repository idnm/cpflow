import sys
sys.path.append('/home/idnm/Programming projects/jax_circuits')

from jax_circuits import *
from cp_utils import *
from topology import *
from matrix_utils import *
from scipy.stats import unitary_group


layer = [[0,1]]
u_target = jnp.kron(y_mat, x_mat)

decomposer = Decompose(layer, target_unitary=u_target, label='2qtest_ad')

adaptive_options = AdaptiveOptions(
    num_samples=3,
    min_num_cp_gates=1,
    max_num_cp_gates=5,
    max_evals=3)

results = decomposer.adaptive(adaptive_options)
