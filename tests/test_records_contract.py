"""One contract for the records table of every simulator that returns one.

Every public simulator is run on small cases across the subcritical, critical,
supercritical and Arley regimes (the lifespan simulator takes per-epoch arrays,
so it runs the float regimes only).  The first three regimes are run with float
rates and again as one-epoch rate-likes.  Each run must satisfy the invariants of
``_helpers.assert_valid_records``, be identical for a repeated seed, and run
with ``random_seed=None``.

The expected column list and dtypes are data, one entry per simulator, because
the simulators do not all produce the same schema today:

- ``simulate_birth_death_lifespans`` adds a ``birth_epoch`` column.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
from _helpers import assert_valid_records
from pandas.testing import assert_frame_equal

from bdsurv import (
    ConstantRateLike,
    LinearRateLike,
    PiecewiseConstant,
    PiecewiseLinear,
    SimulatedTree,
    SplittingConfig,
    simulate_birth_death,
    simulate_birth_death_lifespans,
    simulate_contour_given_ages,
    simulate_contour_given_population,
    simulate_contour_given_survival,
    simulate_survival_splitting,
)

T = 3.0
SEEDS = tuple(range(8))

BASE_DTYPES = {
    "individual_id": "int64",
    "parent_id": "float64",
    "creation_time": "float64",
    "death_time": "float64",
    "is_spine": "bool",
    "survived": "bool",
    "age": "float64",
    "lifespan": "float64",
}
LIFESPAN_DTYPES = {**BASE_DTYPES, "birth_epoch": "int64"}

RateLike = ConstantRateLike | LinearRateLike
# (records, terminal time) of every tree a call returns.
Runs = list[tuple[pd.DataFrame, float]]

# The regimes with float rates, the same ones as rate-likes (suffix ``_epochs``),
# and the time-varying Arley regime.
FLOAT_REGIMES = {
    "subcritical": (0.5, 1.0),
    "critical": (1.0, 1.0),
    "supercritical": (1.3, 0.8),
}
ARLEY_RATES: tuple[RateLike, RateLike] = (
    PiecewiseConstant(np.array([1.0]), np.array([T])),
    PiecewiseLinear(np.array([0.0]), np.array([0.8 * T]), np.array([T])),
)
REGIMES: dict[str, tuple[float | RateLike, float | RateLike]] = (
    dict(FLOAT_REGIMES)
    | {
        f"{name}_epochs": (
            PiecewiseConstant(np.array([b]), np.array([T])),
            PiecewiseConstant(np.array([d]), np.array([T])),
        )
        for name, (b, d) in FLOAT_REGIMES.items()
    }
    | {"arley": ARLEY_RATES}
)


@dataclass(frozen=True)
class Simulator:
    """A public simulator, the regimes it accepts and its records schema."""

    name: str
    regimes: tuple[str, ...]
    dtypes: dict[str, str]
    run: Callable[[str, int | None], Runs]
    # Whether the simulator can return an extinct process.
    can_go_extinct: bool = False

    def schema_for(self) -> dict[str, str]:
        """Return the expected dtypes of this simulator's records, as data."""
        return dict(self.dtypes)


def _tree_run(tree: SimulatedTree) -> Runs:
    return [(tree.records, tree.terminal_time)]


def _forward(regime: str, seed: int | None) -> Runs:
    birth, death = REGIMES[regime]
    return _tree_run(simulate_birth_death(birth, death, T, random_seed=seed))


def _lifespans(regime: str, seed: int | None) -> Runs:
    birth, death = FLOAT_REGIMES[regime]
    return _tree_run(simulate_birth_death_lifespans(birth, death, T, random_seed=seed))


def _contour(regime: str, seed: int | None) -> Runs:
    birth, death = REGIMES[regime]
    return _tree_run(simulate_contour_given_survival(birth, death, T, random_seed=seed))


def _given_population(regime: str, seed: int | None) -> Runs:
    birth, death = REGIMES[regime]
    return _tree_run(
        simulate_contour_given_population(
            birth, death, 3, terminal_time=T, random_seed=seed
        )
    )


def _given_ages(regime: str, seed: int | None) -> Runs:
    birth, death = REGIMES[regime]
    return _tree_run(
        simulate_contour_given_ages(
            birth, death, [0.7, 1.9], terminal_time=T, random_seed=seed
        )
    )


def _splitting(regime: str, seed: int | None) -> Runs:
    birth, death = REGIMES[regime]
    result = simulate_survival_splitting(
        SplittingConfig(
            birth_rate=birth,
            death_rate=death,
            terminal_time=T,
            n_particles=30,
            random_seed=seed,
            n_conditioned_trajectories=3,
            max_restarts=200,
        )
    )
    return [
        (t.process.records, t.process.terminal_time)
        for t in result.conditioned_trajectories
    ]


def _sim(
    name: str,
    regimes: tuple[str, ...],
    run: Callable[[str, int | None], Runs],
    *,
    forward: bool = False,
) -> Simulator:
    """Build a simulator entry; ``forward`` ones can go extinct."""
    return Simulator(
        name,
        regimes,
        LIFESPAN_DTYPES if name == "simulate_birth_death_lifespans" else BASE_DTYPES,
        run,
        can_go_extinct=forward,
    )


FLOAT = tuple(FLOAT_REGIMES)
ALL = tuple(REGIMES)

SIMULATORS = [
    _sim("simulate_birth_death", ALL, _forward, forward=True),
    _sim("simulate_birth_death_lifespans", FLOAT, _lifespans, forward=True),
    _sim("simulate_contour_given_survival", ALL, _contour),
    _sim("simulate_contour_given_population", ALL, _given_population),
    _sim("simulate_contour_given_ages", ALL, _given_ages),
    _sim("simulate_survival_splitting", ALL, _splitting),
]

CASES = [
    pytest.param(sim, regime, id=f"{sim.name}-{regime}")
    for sim in SIMULATORS
    for regime in sim.regimes
]


def _check(sim: Simulator, runs: Runs) -> None:
    assert runs, f"{sim.name} returned no records"
    for records, terminal_time in runs:
        assert_valid_records(records, terminal_time, expected_dtypes=sim.schema_for())


@pytest.mark.parametrize(("sim", "regime"), CASES)
def test_records_satisfy_contract(sim: Simulator, regime: str) -> None:
    """Every seed's records satisfy every invariant and the simulator's schema."""
    for seed in SEEDS:
        _check(sim, sim.run(regime, seed))


@pytest.mark.parametrize(("sim", "regime"), CASES)
def test_same_seed_gives_identical_records(sim: Simulator, regime: str) -> None:
    """The same seed gives identical records."""
    first = sim.run(regime, 11)
    second = sim.run(regime, 11)
    assert len(first) == len(second), f"{sim.name} returned different counts"
    for (a, _), (b, _) in zip(first, second, strict=True):
        assert_frame_equal(a, b)


@pytest.mark.parametrize(("sim", "regime"), CASES)
def test_runs_without_a_seed(sim: Simulator, regime: str) -> None:
    """``random_seed=None`` runs and satisfies the contract."""
    _check(sim, sim.run(regime, None))


@pytest.mark.parametrize(
    "sim", [s for s in SIMULATORS if s.can_go_extinct], ids=lambda s: s.name
)
def test_forward_seeds_cover_extinct_and_surviving_processes(sim: Simulator) -> None:
    """The spine rule is checked both with and without a survivor."""
    outcomes = {
        bool(sim.run("critical", seed)[0][0]["survived"].any()) for seed in SEEDS
    }
    assert outcomes == {True, False}, (
        f"{sim.name}: critical seeds {SEEDS} should include both an extinct and a "
        f"surviving process, got survivor outcomes {sorted(outcomes)}"
    )
