# bdsurv

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Simulation of linear birth-death processes, with a focus on survival-conditioned
processes.

In the subcritical regime (`lambda < mu`, per-individual birth rate `lambda`,
death rate `mu`) the unconditioned process goes extinct with probability 1, so
surviving to a terminal time `T` is a rare event.
Two independent simulators produce a tree conditioned on that event.
Both accept rates that are finite and non-negative; splitting also covers
supercritical processes, which still go extinct with probability `mu / lambda`.
Alongside them is a forward simulator for processes whose rates change over
time, in any regime.

## Rates and results

Every simulator takes each rate as a `RateInput`: a float, which is constant on
`[0, T]`, or a rate object, which is piecewise constant (`PiecewiseConstant`) or
piecewise linear (`PiecewiseLinear`) in time with epochs ending at the given
times (an Arley process, for example, has `mu(t) = m t`).
Rate objects must share one end time, which is the terminal time `T`, so
`terminal_time` is required when both rates are floats and otherwise may be
omitted.
A rate can also be a `Hazard`: `PowerHazard` (with `PowerHazard.weibull` and `PowerHazard.rayleigh`),
`GompertzHazard`, `LogLogisticHazard`, `LomaxHazard`, `QuadraticHazard`, or a `PiecewiseHazard`
that composes them over epochs in local time.
Event times are drawn by exact left-truncated inversion of the cumulative hazard, in closed form where
one exists and by Brent's method otherwise (`docs/hazards.md`).
A parametric hazard does not fix an end time, so `terminal_time` is required unless a `PiecewiseHazard`
or a rate object supplies it; `sample_residual_lifetimes` samples a `Hazard` directly.
`ConstantRateLike` and `LinearRateLike` are the protocols any object with the
same attributes satisfies.

Every simulator returns a `SimulatedTree`: a frozen dataclass of the
individual-level `records` table, `birth_rate`, `death_rate` and
`terminal_time`.
The rates of a tree are always rate objects: a float is stored as a one-epoch
`PiecewiseConstant`.

## Survival-conditioned simulation

Compared head-to-head in
[`bdpinf`'s validation notebooks](https://github.com/jaycrick/bdpinf/tree/main/qmd_scripts/validation/survival_conditioned_simulation).

| Method               | Function                         | Mechanism                                               |
| -------------------- | -------------------------------- | ------------------------------------------------------- |
| Contour process      | `simulate_contour_given_survival` | Exact: the spine excursion is constructed to reach `T`   |
| Multilevel splitting | `simulate_survival_splitting`     | Adaptive-grid (RESTART) enrichment at level boundaries   |

Splitting is configured by a keyword-only `SplittingConfig`.
Its level grid, replicate counts and restart cap come from
`select_level_times_for_band`, `compute_level_replicates` and
`default_max_restarts`, which take the same `RateInput` rates.
The jumps of the contour process and the survival probability that drives the
splitting levels are computed from the rates by exact hazard inversion and
quadrature (`docs/contour_process.md`, `docs/splitting.md`).
That survival probability is public as `log_survival_probability`.

Two further exact contour simulators condition on the population alive at `T`
(`docs/contour_conditioned.md`):

| Function                            | Conditioning                         |
| ----------------------------------- | ------------------------------------ |
| `simulate_contour_given_population` | exactly `n` individuals alive at `T` |
| `simulate_contour_given_ages`       | survivors at `T` with the given ages |

## Forward simulation of general processes

These are not conditioned on survival, and the rates may vary over time.

| Function                              | Rates                             | Mechanism                                                   |
| ------------------------------------- | --------------------------------- | ----------------------------------------------------------- |
| `simulate_birth_death`                | any `RateInput`                   | Exact inverse of the cumulative hazard within each segment  |
| `simulate_birth_death_lifespans`      | piecewise constant, as arrays     | JIT-compiled (JAX) loop; each lifespan drawn at birth       |
| `sample_conditioned_survival_process` | piecewise constant, as arrays     | Era-by-era filtering and cloning on per-era survivor counts |

`simulate_birth_death_lifespans_arrays` returns the raw per-individual arrays,
and `sample_conditioned_survival_per_posterior` runs the era-conditioned
sampler once per posterior draw.
The three functions that take per-epoch arrays require strictly positive rates.

The runtime dependencies are `numpy`, `scipy`, `pandas` and `jax`.
`import bdsurv` enables JAX 64-bit mode and defaults `JAX_PLATFORMS` to `cpu`.

See `docs/` for the mathematical derivation of each method:
`contour_process.md`, `splitting.md`, `hazards.md`, `lifespans.md` and `era_conditioning.md`.
The works the methods rest on are listed in `docs/references.md`.
Wall-clock comparisons are in `benchmarks/simulator_speed.qmd`,
`benchmarks/method_comparison.qmd` and, for Arley processes matched to the
constant-rate cases, `benchmarks/arley_comparison.qmd`.

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
    simulate_contour_given_survival,
    simulate_survival_splitting,
)

# Exact: one conditioned tree per call.
tree = simulate_contour_given_survival(
    birth_rate=0.9, death_rate=1.0, terminal_time=50.0, random_seed=0
)
print(tree.records.head())

# Approximate: a batch of trajectories that survive to the terminal time,
# via multilevel splitting.
config = SplittingConfig(
    birth_rate=0.9, death_rate=1.0, terminal_time=50.0, n_particles=500
)
result = simulate_survival_splitting(config)
print(len(result.conditioned_trajectories))
```

Rates that vary linearly in time (here birth 1, death `0.1 t`, over `[0, 20]`);
the terminal time is the rates' end time:

```python
import numpy as np

from bdsurv import PiecewiseConstant, PiecewiseLinear, simulate_contour_given_survival

birth = PiecewiseConstant(np.array([1.0]), np.array([20.0]))
death = PiecewiseLinear(np.array([0.0]), np.array([2.0]), np.array([20.0]))
tree = simulate_contour_given_survival(birth, death, random_seed=0)
```

Forward simulation with rates that change at the epoch end times:

```python
from bdsurv import PiecewiseConstant, simulate_birth_death

epoch_ends = [5.0, 10.0]
process = simulate_birth_death(
    PiecewiseConstant([0.5, 0.3], epoch_ends),
    PiecewiseConstant([0.3, 0.4], epoch_ends),
    random_seed=0,
)
print(process.records.head())
```

## Development

Run tests:

```sh
uv run pytest
```

This runs the fast set (tests marked `slow` are deselected).
Run the full suite with coverage, as CI does:

```sh
uv run pytest -m "slow or not slow" --cov=src --cov-report=term-missing --cov-fail-under=86
```

Lint and type-check:

```sh
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
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
