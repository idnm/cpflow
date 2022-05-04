## CPFlow
CPFlow uses variational synthesis to find quantum circuits consisting of CNOTs+arbitrary 1q gates that simultaneously
- Minimize a given loss function L(U), where U is the unitary of the circuit.
- Do it with as few CNOT gates as possible.

Indirectly the circuits can also be optimized for CNOT depth and even T count or T depth in some cases. Typical loss functions L(U) correspond to a compilation and a state preparation problem, but arbitrary well-defined loss functions can be handled as well. The cornerstone objective is to obtain as short circtuis as possible, possibly at the cost of a longer search time.

CPFlow implements the synthesis algorithms  described in https://arxiv.org/abs/2205.01121. It is distributed under the MIT licence.

## Installation
`CPFlow` is available via `pip`.  It is highly recommended to install the package in a new virtual environment.

```sh
pip install cpflow
```

A feature that allows to decompose sythesized circuits into Clifford+T basis requires yet experimental `qiskit` branch that can be installed through

```sh
pip install git+https://github.com/LNoorl/qiskit-terra@d2e0dc1185ccc3b0c9957e3d7d9bc610dede29d4
```

## Basic example
Decomposing the CCZ gate with linear qubit connectivity 0-1-2. Can be executed in python console but intended for use with Jupyter notebooks.

```python
import numpy as np
from cpflow import *

u_target = np.diag([1, 1, 1, 1, 1, 1, 1, -1])  # CCZ gate
layer = [[0, 1], [1, 2]]  # Linear connectivity
decomposer = Synthesize(layer, target_unitary=u_target, label='ccz_chain')
options = StaticOptions(num_cp_gates=12, accepted_num_cz_gates=10, num_samples=10)

results = decomposer.static(options) # Should take from one to five minutes.

d = results.decompositions[3]  # This turned out to be the best decomposition for refinement.
d.refine()
print(d)
d.circuit.draw()
```
Output:

```sh
< ccz_chain| Clifford+T | loss: 0.0  | CZ count: 8 | CZ depth: 8  | T count: 7 | T depth: 5 >
```
![image](https://user-images.githubusercontent.com/13020565/165085291-f566108b-66bf-4dc8-a9c9-dcd771ea64b8.png)

## More features
For further examples we encourage to explore a [tutorial notebook](https://github.com/idnm/cpflow/blob/master/tutorial/CPFlow_tutorial.ipynb) interactively. For motivation and background see the original paper https://arxiv.org/abs/2205.01121.
