"""One contract for the records table of every simulator that returns one.

Every public simulator is run on small cases across the subcritical, critical,
supercritical and Arley regimes (the last only where the simulator accepts
time-varying rates).  Each run must satisfy the invariants of
``_helpers.assert_valid_records``, be identical for a repeated seed, and run
with ``random_seed=None``.

The expected column list and dtypes are data, one entry per simulator, because
the simulators do not all produce the same schema today:

- ``simulate_birth_death_lifespans`` adds a ``birth_epoch`` column.
- ``parent_id`` is ``float64`` (NaN for the root) in every simulator, except in
  a frame holding only the root, where all of them give ``object`` (``None``).
- ``death_time`` is ``float64``, except that the two splitting simulators give
  ``object`` (all ``None``) when no individual in the tree has died.
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
    LinearSplittingConfig,
    PiecewiseConstant,
    PiecewiseLinear,
    SplittingConfig,
    simulate_birth_death_lifespans,
    simulate_birth_death_linear,
    simulate_birth_death_process,
    simulate_birth_death_process_from_contour_process,
    simulate_contour_given_ages,
    simulate_contour_given_population,
    simulate_linear_rate_contour_process,
    simulate_linear_survival_splitting,
    simulate_subcritical_survival_splitting,
)
from bdsurv.linear_rates import ConstantRateLike, LinearRateLike

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
# (records, terminal time) of every tree or process a call returns.
Runs = list[tuple[pd.DataFrame, float]]

CONSTANT_REGIMES = {
    "subcritical": (0.5, 1.0),
    "critical": (1.0, 1.0),
    "supercritical": (1.3, 0.8),
}
ARLEY_RATES: tuple[RateLike, RateLike] = (
    PiecewiseConstant(np.array([1.0]), np.array([T])),
    PiecewiseLinear(np.array([0.0]), np.array([0.8 * T]), np.array([T])),
)
LINEAR_REGIMES: dict[str, tuple[RateLike, RateLike]] = {
    name: (
        PiecewiseConstant(np.array([b]), np.array([T])),
        PiecewiseConstant(np.array([d]), np.array([T])),
    )
    for name, (b, d) in CONSTANT_REGIMES.items()
} | {"arley": ARLEY_RATES}


@dataclass(frozen=True)
class Simulator:
    """A public simulator, the regimes it accepts and its records schema."""

    name: str
    regimes: tuple[str, ...]
    dtypes: dict[str, str]
    run: Callable[[str, int | None], Runs]
    # Whether a frame in which nobody died has an ``object`` ``death_time``.
    no_deaths_death_time_is_object: bool = False
    # Whether the simulator can return an extinct process.
    can_go_extinct: bool = False

    def schema_for(self, records: pd.DataFrame) -> dict[str, str]:
        """Return the expected dtypes for these records, as data."""
        dtypes = dict(self.dtypes)
        if len(records) == 1:  # a root-only frame has an all-None parent_id
            dtypes["parent_id"] = "object"
        if self.no_deaths_death_time_is_object and records["survived"].all():
            dtypes["death_time"] = "object"
        return dtypes


def _process(regime: str, seed: int | None) -> Runs:
    birth, death = CONSTANT_REGIMES[regime]
    process = simulate_birth_death_process(birth, death, T, random_seed=seed)
    return [(process.records, float(process.epoch_end_times[-1]))]


def _lifespans(regime: str, seed: int | None) -> Runs:
    birth, death = CONSTANT_REGIMES[regime]
    process = simulate_birth_death_lifespans(birth, death, T, random_seed=seed)
    return [(process.records, float(process.epoch_end_times[-1]))]


def _linear(regime: str, seed: int | None) -> Runs:
    birth, death = LINEAR_REGIMES[regime]
    process = simulate_birth_death_linear(birth, death, random_seed=seed)
    return [(process.records, float(process.epoch_end_times[-1]))]


def _contour(regime: str, seed: int | None) -> Runs:
    birth, death = CONSTANT_REGIMES[regime]
    tree = simulate_birth_death_process_from_contour_process(
        birth, death, T, random_seed=seed
    )
    return [(tree.records, tree.terminal_time)]


def _linear_contour(regime: str, seed: int | None) -> Runs:
    birth, death = LINEAR_REGIMES[regime]
    tree = simulate_linear_rate_contour_process(birth, death, random_seed=seed)
    return [(tree.records, tree.terminal_time)]


def _conditioned_rates(regime: str) -> tuple[float | RateLike, float | RateLike]:
    return CONSTANT_REGIMES.get(regime) or ARLEY_RATES


def _given_population(regime: str, seed: int | None) -> Runs:
    birth, death = _conditioned_rates(regime)
    tree = simulate_contour_given_population(
        birth, death, 3, terminal_time=T, random_seed=seed
    )
    return [(tree.records, tree.terminal_time)]


def _given_ages(regime: str, seed: int | None) -> Runs:
    birth, death = _conditioned_rates(regime)
    tree = simulate_contour_given_ages(
        birth, death, [0.7, 1.9], terminal_time=T, random_seed=seed
    )
    return [(tree.records, tree.terminal_time)]


def _splitting(regime: str, seed: int | None) -> Runs:
    birth, death = CONSTANT_REGIMES[regime]
    result = simulate_subcritical_survival_splitting(
        SplittingConfig(
            birth,
            death,
            T,
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


def _linear_splitting(regime: str, seed: int | None) -> Runs:
    birth, death = LINEAR_REGIMES[regime]
    result = simulate_linear_survival_splitting(
        LinearSplittingConfig(
            birth,
            death,
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
    splitting: bool = False,
) -> Simulator:
    """Build a simulator entry; ``forward`` ones can go extinct."""
    return Simulator(
        name,
        regimes,
        LIFESPAN_DTYPES if name == "simulate_birth_death_lifespans" else BASE_DTYPES,
        run,
        no_deaths_death_time_is_object=splitting,
        can_go_extinct=forward,
    )


CONSTANT = tuple(CONSTANT_REGIMES)
WITH_ARLEY = (*CONSTANT, "arley")

SIMULATORS = [
    _sim(
        "simulate_birth_death_process",
        CONSTANT,
        _process,
        forward=True,
    ),
    _sim(
        "simulate_birth_death_linear",
        WITH_ARLEY,
        _linear,
        forward=True,
    ),
    _sim(
        "simulate_birth_death_lifespans",
        CONSTANT,
        _lifespans,
        forward=True,
    ),
    _sim("simulate_birth_death_process_from_contour_process", CONSTANT, _contour),
    _sim("simulate_linear_rate_contour_process", WITH_ARLEY, _linear_contour),
    _sim("simulate_contour_given_population", WITH_ARLEY, _given_population),
    _sim("simulate_contour_given_ages", WITH_ARLEY, _given_ages),
    _sim(
        "simulate_subcritical_survival_splitting",
        CONSTANT,
        _splitting,
        splitting=True,
    ),
    _sim(
        "simulate_linear_survival_splitting",
        WITH_ARLEY,
        _linear_splitting,
        splitting=True,
    ),
]

CASES = [
    pytest.param(sim, regime, id=f"{sim.name}-{regime}")
    for sim in SIMULATORS
    for regime in sim.regimes
]


def _check(sim: Simulator, runs: Runs) -> None:
    assert runs, f"{sim.name} returned no records"
    for records, terminal_time in runs:
        assert_valid_records(
            records, terminal_time, expected_dtypes=sim.schema_for(records)
        )


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
