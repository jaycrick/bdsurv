# bdsurv

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Simulation of linear birth-death processes, with a focus on survival-conditioned
processes.

In the subcritical regime (`lambda < mu`, per-individual birth rate `lambda`,
death rate `mu`) the unconditioned process goes extinct with probability 1, so
surviving to a terminal time `T` is a rare event.
Two independent simulators produce a tree conditioned on that event.
Both accept any positive rates; splitting also covers supercritical processes,
which still go extinct with probability `mu / lambda`.
Alongside them are forward simulators for processes whose rates change over
time, in any regime.

## Survival-conditioned simulation

Compared head-to-head in
[`bdpinf`'s validation notebooks](https://github.com/jaycrick/bdpinf/tree/main/qmd_scripts/validation/survival_conditioned_simulation).

| Method                     | Function                                             | Mechanism                                            |
| -------------------------- | ----------------------------------------------------- | ----------------------------------------------------- |
| Contour process            | `simulate_birth_death_process_from_contour_process`    | Exact — the spine excursion is constructed to reach `T` |
| Multilevel splitting       | `simulate_subcritical_survival_splitting`               | Adaptive-grid (RESTART) enrichment at level boundaries |

Both return a `SimulatedTree`: a plain dataclass wrapping the individual-level
records table plus the rates and terminal time used to generate it.

## Forward simulation of general processes

These are not conditioned on survival, and the rates may vary by epoch.
They return a `SimulatedProcess`: the records table plus the epoch rates and
end times.

| Function                              | Rates                    | Mechanism                                                   |
| ------------------------------------- | ------------------------ | ----------------------------------------------------------- |
| `simulate_birth_death_process`        | piecewise constant       | Gillespie event loop in `numpy`                             |
| `simulate_birth_death_lifespans`      | piecewise constant       | JIT-compiled (JAX) loop; each lifespan drawn at birth       |
| `simulate_birth_death_linear`         | piecewise linear         | Exact inverse of the cumulative hazard within each segment  |
| `sample_conditioned_survival_process` | piecewise constant       | Era-by-era filtering and cloning on per-era survivor counts |

`simulate_birth_death_lifespans_arrays` returns the raw per-individual arrays,
and `sample_conditioned_survival_per_posterior` runs the era-conditioned
sampler once per posterior draw.

The runtime dependencies are `numpy`, `pandas` and `jax`.
`import bdsurv` enables JAX 64-bit mode and defaults `JAX_PLATFORMS` to `cpu`.

See `docs/` for the mathematical derivation of each method:
`contour_process.md`, `splitting.md`, `lifespans.md` and `era_conditioning.md`.
Wall-clock comparisons are in `benchmarks/simulator_speed.qmd`.

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

# Approximate: a batch of trajectories that survive to the terminal time,
# via multilevel splitting.
config = SplittingConfig(
    birth_rate=0.9, death_rate=1.0, terminal_time=50.0, n_particles=500
)
result = simulate_subcritical_survival_splitting(config)
print(len(result.conditioned_trajectories))
```

Forward simulation with rates that change at the epoch end times:

```python
from bdsurv import simulate_birth_death_process

process = simulate_birth_death_process(
    birth_rates=[0.5, 0.3], death_rates=[0.3, 0.4], times=[5.0, 10.0], random_seed=0
)
print(process.records.head())
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

Speed benchmarks of every simulator live in `benchmarks/simulator_speed.qmd`
(see `benchmarks/README.md`).
The correctness comparison against theoretical age densities lives in `bdpinf`
(`qmd_scripts/validation/`), because it needs `bdpinf`'s age laws.

## Related: fitting and age sampling

`bdsurv` only simulates the process.
Inference and the age laws live in [`bdpinf`](https://github.com/jaycrick/bdpinf):
`bdpinf.inference.fit` maximises the exact likelihood of observed ages for piecewise-constant
or piecewise-linear birth and death rates, and `bdpinf.simulation.simulate_ages_from_rates`
and `simulate_bdp_ages_from_distributions` draw ages directly from the exact age laws.
Those need `bdpinf`'s JAX age densities, so they stay there; `bdpinf` depends on `bdsurv`, not the reverse.
