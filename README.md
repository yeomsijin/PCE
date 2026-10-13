# PCE: persistence entropy excess

Research implementation of **own-own PCE** for H0 and H1 persistence diagrams,
and the PRL main-text neighborhood convention for topological causality (TC).
The current estimator definition was adopted on 2026-10-12.

## Installation

Python 3.9 or newer:

```sh
python -m pip install .
```

Core dependencies are NumPy, SciPy and Ripser. PyTorch is not required.

## Current PCE API

```python
import numpy as np
from tpce import own_own_costs, direction

# H0: finite lifetimes from each space's OWN neighborhood.
x = np.array([0.1, 0.3, 0.6])
y = np.array([0.2, 0.4])
cost, sigma, metadata = own_own_costs(x, y)
score = direction(cost[0], cost[1])
print(cost, sigma, score)

# H1 uses (birth, lifetime), NOT (birth, death).
x1 = np.array([[0.1, 0.3], [0.2, 0.6]])
y1 = np.array([[0.15, 0.4]])
cost1, widths1, metadata1 = own_own_costs(x1, y1)
```

Both diagrams must use the same reference distance scale. Neighborhood choice,
embedding, normalization and reference scale are upstream choices; this PD API
does not silently choose them. Construct each space's own neighborhood with the
anchor excluded. The active H0 backend `tpce.h0_float64.lifetimes` computes
Euclidean minimum spanning tree edges in float64. `tpce.persistence.diagrams`
uses Ripser for H0/H1 and defaults to an H1 minimum lifetime of `1e-10`.

## Defaults and definition

| Setting | Current own-own PCE default |
| --- | --- |
| Bandwidth method | `pair_persistence_variance` |
| Explicit bandwidth | `None` (estimated from the pair) |
| Response scale eta | `5.0` |
| H0 input | 1D nonnegative finite lifetimes |
| H1 input | `(bars, 2)` nonnegative finite birth/lifetime coordinates |
| Mixture | equal mass per nonempty PD; lifetime weights within each PD |
| H0 width | `sigma² = 2 Var_Q(lifetime)` |
| H1 widths | `sigma_axis² = 2 Var_Q(axis)` for birth and lifetime |
| Constant H1 axes | exact Gaussian factor 1; no bandwidth floor |
| Zero-lifetime bars | ignored |
| Undefined direction | NaN; no denominator floor |

For a nonempty target A, `p_i = lifetime_i / sum(lifetimes)` and
`E(A,B) = sum_i p_i (F_B(u_i) - F_A(u_i))² / (2 eta²)`.
Each response sums a Gaussian kernel weighted by `lifetime / (1 + lifetime)`.
`own_own_costs` returns `[E(X,Y), E(Y,X)]` using the same pair bandwidth.
The score is `(E(Y,X) - E(X,Y)) / (E(Y,X) + E(X,Y))`; positive is the X→Y
convention, not a guarantee of causal identification.

The induced distribution includes an extra unexplained event. Do not normalize
it only over target bars. Empty targets use the existing mass-coarsening rule.
Both empty H1 PDs have zero costs and undefined direction. Unidentifiable H0
adaptive widths raise `ValueError`; no arbitrary floor is inserted. Fixed widths
require an explicit `bandwidth_method="fixed"` and positive `bandwidth` (two
widths for H1). Aggregate costs over times first, then form the ratio; missing
costs remain undefined rather than silently disappearing from the aggregate.

## TC and historical compatibility

`tpce.estimate_tc` consumes two time-aligned reconstructed spaces. Its defaults
are `k=20`, `theiler_full_window=0`, `batch=64` and 99% PCA energy. For causal
i→j, choose j's OWN neighbors and map their same-time coordinates forward to i.
Outputs carry `tc_main_text_source_neighbors_v1`. The interpretation that
conflicting supplement labels are a typo is a working assumption, not author
confirmation.

`TPCEConfig`, `estimate`, `prepare`, `calibrate` and `python -m tpce` retain the
**historical time-induced TPCE** estimator. They are not the current own-own
PCE entry point. Historical defaults are m=5, delay=17, k=20, KDE-CDF
normalization, full Theiler window `(m-1)*delay`, 64 calibration anchors and
256 scale samples. Historical H1 calibration remains unchanged.

## Tests

The standalone release includes portable PCE and bandwidth tests:

```sh
python -m unittest discover -s tests -v
```

The full research workspace additionally contains TC integration tests,
historical reference fixtures and FPRM experiments. Those are not bundled with
the standalone release. PRL snapshots, observations and experiment results are
excluded from the installable package and release bundle.
