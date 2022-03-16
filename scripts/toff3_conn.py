import sys
sys.path.append('/home/idnm/Programming projects/jax_circuits')

from jax_circuits import *
from cp_utils import *
from topology import *
from matrix_utils import *
from scipy.stats import unitary_group


layer = sequ_layer(3)
decomposer = Decompose(layer, target_unitary=u_toff3)

static_options = {'batch_size': 100, 'accepted_num_gates': 4}

adaptive_options = {'max_evals':5,
                   'max_num_cp_gates':12,
                   'min_num_cp_gates':2,
                   'target_num_gates': 5,
                   'evals_between_verification':1}
                   
adaptive_results = decomposer.adaptive(
    static_options=static_options,
    adaptive_options=adaptive_options,
    save_to='data/toff3_conn/',
    overwrite_existing_trials=False,
    overwrite_existing_decompositions=False)

