# bdsurv

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Survival-conditioned simulation of subcritical linear birth-death processes.
Given a per-individual birth rate `lambda`, death rate `mu` (with
`lambda < mu`), and a terminal time `T`, both simulators here produce a tree
of individuals conditioned on the population surviving to `T` — a rare event
in the subcritical regime, since the unconditioned process goes extinct with
probability 1.

Two independent methods, compared head-to-head in `validation/`:

| Method                     | Function                                             | Mechanism                                            |
| -------------------------- | ----------------------------------------------------- | ----------------------------------------------------- |
| Contour process            | `simulate_birth_death_process_from_contour_process`    | Exact — the spine excursion is constructed to reach `T` |
| Multilevel splitting       | `simulate_subcritical_survival_splitting`               | Fixed-factor (RESTART) enrichment at level boundaries  |

Both return a `SimulatedTree`: a plain dataclass wrapping the individual-level
records table plus the rates and terminal time used to generate it. The
package has exactly two runtime dependencies, `numpy` and `pandas`.

See `docs/contour_process.md` and `docs/splitting.md` for the mathematical
derivation of each method.

## Installation

```sh
pip install bdsurv
```

or, for development:

```sh
git clone https://github.com/jaycrick/bdsurv.git
cd bdsurv
uv sync --all-groups
```

## Quick Start

```python
from bdsurv import (
    SplittingConfig,
    simulate_birth_death_process_from_contour_process,
    simulate_subcritical_survival_splitting,
)

# Exact: one conditioned tree per call.
tree = simulate_birth_death_process_from_contour_process(
    birth_rate=0.9, death_rate=1.0, terminal_time=50.0, random_seed=0
)
print(tree.records.head())

# Approximate: a survival-probability estimate plus a batch of conditioned
# trajectories, via multilevel splitting.
config = SplittingConfig(
    birth_rate=0.9, death_rate=1.0, terminal_time=50.0, n_particles=500
)
result = simulate_subcritical_survival_splitting(config)
print(result.survival_probability_estimate)
```

## Development

Run tests:

```sh
uv run pytest
```

Lint and type-check:

```sh
uv run ruff check src tests
uv run ruff format --check src tests
uv run basedpyright src/
```

Run the simulator comparison (speed and correctness, contour process vs.
splitting) as a Quarto notebook:

```sh
uv run quarto render validation/simulator_comparison.qmd
```
