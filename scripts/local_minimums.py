from scipy.stats import unitary_group

import sys
# sys.path.append('/home/rqc-qit-0/nnemkov/jc_module')
sys.path.append('/home/idnm/Programming projects/jax_circuits/')
from cpflow.main import *


def success_ratio(num_qubits, num_cz_gates, target_type, num_samples, random_seed):
    anz = Ansatz(num_qubits, 'cz', fill_layers(connected_layer(num_qubits), num_cz_gates))

    if target_type == 'random_unitary':
        u_target = unitary_group.rvs(2 ** num_qubits, random_state=random_seed)
    elif target_type == 'random_self':
        angles_target = random_angles(anz.num_angles, key=random.PRNGKey(random_seed))
        u_target = anz.unitary(angles_target)
    else:
        raise TypeError

    results = anz.learn(u_target, num_repeats=num_samples, keep_history=False)
    best_losses = [jnp.min(r['loss']) for r in results]

    if target_type == 'random_unitary':
        best_loss = min(best_losses)
        successes = [loss <= (best_loss + 1e-4) for loss in best_losses]
    elif target_type == 'random_self':
        successes = [loss < 1e-4 for loss in best_losses]

    return {
        'u_target': u_target,
        'target_type': target_type,
        'random_seed': random_seed,
        'num_cz_gates': num_cz_gates,
        'num_qubits': num_qubits,
        'losses': best_losses,
        'success_ratio': sum(successes) / len(successes)}


def success_chart(num_qubits, range_cz_gates, target_type, num_samples, random_seed, save_to):
    results_list = []
    for num_cz_gates in tqdm(range_cz_gates):
        results_list.append(success_ratio(num_qubits, num_cz_gates, target_type, num_samples, random_seed))

    os.makedirs(os.path.dirname(save_to), exist_ok=True)
    with open(save_to, 'wb') as f:
        pickle.dump(results_list, f)

    success_list = [res['success_ratio'] for res in results_list]

    return success_list


num_qubits = 2
target_type = 'random_unitary'
num_samples = 100
num_targets = 5
for random_seed in range(num_targets):
    save_to = f'results/local_minimums/{num_qubits}q_{target_type}_rs{random_seed}'
    success_chart(
        num_qubits,
        range(theoretical_lower_bound(num_qubits)+1),
        target_type,
        num_samples,
        random_seed,
        save_to)
