# CPFlow
`CPFlow` package is an implementation of the synthesis algorithms for quantum circuits described in ...
## Installation
`CPFlow` is available via `pip`.  It is highly recommended to install the package in a new virtual environment.
```sh
pip install cpflow
```
A feature that allows to decompose sythesized circuits into Clifford+T basis requires yet experimental `qiskit` branch that can be installed through
```sh
pip install git+https://github.com/LNoorl/qiskit-terra@d2e0dc1185ccc3b0c9957e3d7d9bc610dede29d4
```
# Basic example
Decomposing the CCZ gate with linear qubit connectivity 0-1-2. Can be run in python console but indeded for use with Jupyter notebooks.

```python
import numpy as np
from cpflow import *

u_target = np.diag([1, 1, 1, 1, 1, 1, 1, -1])  # CCZ gate
layer = [[0, 1], [1, 2]]  # Linear connectivity
decomposer = Synthesize(layer, target_unitary=u_target, label='ccz_chain')
options = StaticOptions(num_cp_gates=12, accepted_num_cz_gates=10, num_samples=10)

results = decomposer.static(options)

d = results.decompositions[3]  # This turned out to be the best decomposition.
d.refine()
print(d)
d.circuit.draw()
```
Output

```python
< ccz_chain| Rational | loss: 1.1920928955078125e-07  | CZ count: 8 | CZ depth: 8  >
```
figure!

# More features
For further examples see link_to_notebook. For motivation and background see the original paper link_to_paper.

# LICENSE
MIT
