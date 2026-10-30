"""Splitting-based rare-event simulation for survival-conditioned birth-death trees.

This module provides continuous-time simulation utilities for sampling
birth-death branching-process trajectories conditioned on survival up to a large
terminal time.  Any positive rates are accepted: for subcritical processes
survival is rare, for supercritical ones it is likely but not certain.

Implemented methods:
- Multilevel splitting with exact Gillespie trajectory evolution, an adaptive
  level grid, per-level replicate counts and capped restarts

The method keeps the particles that survive to the terminal time and exposes
level diagnostics and particle ancestry metadata; it does not estimate the
survival probability.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from bdsurv.tree import SimulatedTree

MIN_LEVEL_TIMES = 2
_BAND_LOWER: float = np.e**-2.0  # lowest allowed Pr(t_i) / Pr(t_{i-1})
_BAND_UPPER: float = 0.5  # highest allowed Pr(t_i) / Pr(t_{i-1})
_LOG_BAND_LOWER: float = -2.0
_LOG_BAND_UPPER: float = math.log(0.5)
_TARGET_LEVEL_RATIO: float = 1.0 / 3.0  # per-level survival ratio of the start grid
_MAX_GRID_POINTS: int = 100_000  # defensive cap on the refine step
_MIN_REPLICATES: int = 2
_CRITICAL_TOLERANCE: float = 1e-10  # |lambda - mu| * t below this is critical
_REPLICATE_TOLERANCE: float = 1e-9  # so Pr(t_1) = 1/2 gives exactly R_1 = 2
_MAX_LOG_REPLICATES: float = 30.0  # refuse R_i beyond ~1e13
DEFAULT_RESTART_SUCCESS_PROBABILITY: float = 0.999


@dataclass(frozen=True)
class SplittingConfig:
    """Configuration for multilevel splitting.

    Parameters
    ----------
    birth_rate : float
        Per-individual birth rate used in the birth-death dynamics.
    death_rate : float
        Per-individual death rate used in the birth-death dynamics.
    terminal_time : float
        Target time horizon T for conditioning on survival.
    n_particles : int
        Cap on the ensemble size after each enrichment step.  The initial
        ensemble size is ``R_1`` (see ``compute_level_replicates``), which may
        be smaller than this cap.  After enrichment, if the ensemble would
        exceed ``n_particles`` it is subsampled uniformly back down to
        ``n_particles``.
    level_times : list[float] | None, optional
        Level boundary times (including 0 and `terminal_time`).  If ``None``
        the starting grid is ``linspace(0, terminal_time, K + 1)`` with ``K`` from
        ``n_levels``.
        Unless ``replicates`` is also given, this grid is only the *starting*
        grid and is adjusted by ``select_level_times_for_band``.
    n_levels : int | None, optional
        Number of intervals ``K`` in the starting grid when ``level_times`` is not
        provided.  If ``None`` (the default), ``K`` is the integer for which
        ``Pr(T) ** (1 / K)`` is closest to 1/3, and at least 1.
    random_seed : int | None, optional
        Seed for random number generation to allow reproducible runs.
    n_conditioned_trajectories : int | None, optional
        Maximum number of conditioned trajectories to return for downstream
        analysis. If ``None``, all surviving terminal trajectories are
        returned.
    replicates : list[int] | None, optional
        Manual replicate counts, one per interval of ``level_times``.  Requires
        ``level_times``.  When both are given they are used exactly as supplied,
        with no band adjustment and no computed replicates.
    max_restarts : int | None, optional
        Maximum number of attempts (each one a fresh run from time 0) made
        while no particle survives to the terminal time.  If ``None``, it is
        computed by ``default_max_restarts``.  When survival to ``T`` is likely
        (``Pr(T) > 0.5``) trees are simulated one at a time, and ``None`` means
        keep going until one survives.

    """

    birth_rate: float
    death_rate: float
    terminal_time: float
    n_particles: int
    level_times: list[float] | None = None
    n_levels: int | None = None
    random_seed: int | None = None
    n_conditioned_trajectories: int | None = None
    replicates: list[int] | None = None
    max_restarts: int | None = None


@dataclass(frozen=True)
class LevelDiagnostic:
    """Per-level diagnostics for splitting evolution.

    Parameters
    ----------
    level_index : int
        Index of the level interval (1-based with 0 reserved for t0).
    time : float
        Level boundary time corresponding to this diagnostic.
    survivor_count : int
        Number of particles surviving at the level boundary.
    mean_population : float
        Ensemble mean population size at the level boundary.
    entering_count : int | None, optional
        Number of particles that entered this level interval.

    """

    level_index: int
    time: float
    survivor_count: int
    mean_population: float
    entering_count: int | None = None


@dataclass(frozen=True)
class ConditionedTrajectory:
    """A sampled conditioned population path and associated tree record.

    Parameters
    ----------
    particle_id : int
        Identifier for the particle that produced this conditioned trajectory.
    times : np.ndarray
        Monotone array of observation times for the population trace.
    populations : np.ndarray
        Population sizes corresponding to ``times``.
    process : SimulatedTree
        Individual-level birth-death tree reconstructed from particle
        ancestry records.

    """

    particle_id: int
    times: np.ndarray
    populations: np.ndarray
    process: SimulatedTree


@dataclass(frozen=True)
class SplittingSimulationResult:
    """Output of multilevel splitting simulation.

    Parameters
    ----------
    config : SplittingConfig
        The configuration used to run the simulation.
    level_times : np.ndarray
        Array of level boundary times including time 0 and terminal time.
    level_diagnostics : list[LevelDiagnostic]
        Per-level diagnostic objects describing survival and population
        statistics.
    level_mean_populations : np.ndarray
        Ensemble mean population at each level boundary.
    final_survivor_count : int
        Number of surviving records/lineages at the terminal time.
    conditioned_trajectories : list[ConditionedTrajectory]
        Trajectories of the particles that survived to the terminal time.
    particle_genealogy : pd.DataFrame
        Tabular parent-child lineage records produced during resampling (from
        the last attempt).
    replicates : np.ndarray
        Replicate count ``R_i`` for each level interval.
    initial_particle_count : int
        Number of particles at the start of each attempt, ``min(R_1, cap)``.
    n_attempts : int
        Number of attempts made (1 + number of restarts used).
    max_restarts : int | None
        Maximum number of attempts allowed, or ``None`` if unbounded.

    """

    config: SplittingConfig
    level_times: np.ndarray
    level_diagnostics: list[LevelDiagnostic]
    level_mean_populations: np.ndarray
    final_survivor_count: int
    conditioned_trajectories: list[ConditionedTrajectory]
    particle_genealogy: pd.DataFrame
    replicates: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int64),
    )
    initial_particle_count: int = 1
    n_attempts: int = 1
    max_restarts: int | None = 1


@dataclass
class _ParticleState:
    """Mutable per-particle state for exact event simulation.

    Parameters
    ----------
    particle_id : int
        Unique identifier assigned to the particle instance.
    current_time : float
        Current simulated time for this particle.
    next_individual_id : int
        Next available individual identifier for descendants.
    alive_ids : list[int]
        List of individual ids currently alive within the particle.
    records : dict[int, dict[str, Any]]
        Mapping from individual id to a record dict containing creation and
        death metadata.
    trace_times : list[float]
        Times at which the particle trace population was recorded.
    trace_populations : list[int]
        Population sizes corresponding to ``trace_times``.
    rng : np.random.Generator
        Random number generator instance used for stochastic events.

    """

    particle_id: int
    current_time: float
    next_individual_id: int
    alive_ids: list[int]
    records: dict[int, dict[str, Any]]
    trace_times: list[float]
    trace_populations: list[int]
    rng: np.random.Generator


def _validate_common_inputs(
    *,
    birth_rate: float,
    death_rate: float,
    terminal_time: float,
    n_particles: int,
) -> None:
    if birth_rate <= 0.0 or death_rate <= 0.0:
        msg = "birth_rate and death_rate must be positive"
        raise ValueError(msg)
    if terminal_time <= 0.0:
        msg = "terminal_time must be positive"
        raise ValueError(msg)
    if n_particles <= 0:
        msg = "n_particles must be positive"
        raise ValueError(msg)


def _build_level_times(
    *,
    terminal_time: float,
    level_times: list[float] | None,
    n_levels: int | None,
) -> np.ndarray:
    if level_times is None:
        if n_levels is None or n_levels <= 0:
            msg = "n_levels must be positive when level_times is not provided"
            raise ValueError(msg)
        levels = np.linspace(0.0, terminal_time, num=n_levels + 1, dtype=np.float64)
    else:
        levels = np.asarray(level_times, dtype=np.float64)

    if levels.ndim != 1:
        msg = "level_times must be one-dimensional"
        raise ValueError(msg)
    if levels.size < MIN_LEVEL_TIMES:
        msg = "At least two level times are required"
        raise ValueError(msg)
    if not np.isclose(levels[0], 0.0):
        msg = "First level time must be 0"
        raise ValueError(msg)
    if np.any(np.diff(levels) <= 0.0):
        msg = "level_times must be strictly increasing"
        raise ValueError(msg)
    if not np.isclose(levels[-1], terminal_time):
        msg = "Last level time must equal terminal_time"
        raise ValueError(msg)
    return levels


def _make_child_rng(
    *,
    seed_rng: np.random.Generator,
) -> np.random.Generator:
    child_seed = int(seed_rng.integers(0, 2**31 - 1))
    return np.random.default_rng(child_seed)


def _log_survival_probability(
    *,
    birth_rate: float,
    death_rate: float,
    time: float,
) -> float:
    """Return ``log Pr(Z_t > 0)`` from one individual, stable for large ``t``."""
    if time <= 0.0:
        return 0.0
    growth_rate = birth_rate - death_rate
    x = growth_rate * time
    # log of ``num`` in ``pr_extinction``; survival is 1 / (1 + num).
    if abs(x) < _CRITICAL_TOLERANCE:
        log_num = math.log(death_rate) + math.log(time)
    elif x < 0.0:
        # kernel = expm1(|x|) / |g|, with log(expm1(y)) = y + log(1 - exp(-y))
        log_num = (
            math.log(death_rate)
            + (-x)
            + math.log(-math.expm1(x))
            - math.log(-growth_rate)
        )
    else:
        log_num = (
            math.log(death_rate) + math.log(-math.expm1(-x)) - math.log(growth_rate)
        )
    return -float(np.logaddexp(0.0, log_num))


def _initial_level_count(log_pr_terminal: float) -> int:
    """Return ``K`` with ``Pr(T) ** (1 / K)`` closest to 1/3, and at least 1."""
    return max(1, round(log_pr_terminal / math.log(_TARGET_LEVEL_RATIO)))


def select_level_times_for_band(
    *,
    birth_rate: float,
    death_rate: float,
    terminal_time: float,
    n_levels: int | None = None,
    initial_level_times: list[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Adjust a starting level grid so each survival ratio lies in the band.

    With ``Pr(t) = P(Z_t > 0)`` from a single individual, the ratio over
    interval ``i`` is ``r_i = Pr(t_i) / Pr(t_{i-1})``.  The band is
    ``[e^{-2}, 0.5]``.

    1. **Refine.** Every interval with ``r_i < e^{-2}`` gets its midpoint added,
       and the ratios are recomputed.  This repeats until no ratio is below the
       band.  A new interval whose ratio is now above 0.5 is kept.
    2. **Coarsen.** Sweep left to right.  For each interval with ``r_i > 0.5``
       try removing its right end point ``t_i``; the merged interval
       ``(t_{i-1}, t_{i+1})`` has ratio ``Pr(t_{i+1}) / Pr(t_{i-1})``, and the
       removal is kept if that is at least ``e^{-2}``.  After a removal the new
       interval at the same position is examined next.  Sweeps repeat until a
       sweep removes nothing.

    No time is protected in either step, so points added while refining may
    be removed when coarsening.  Only ``0`` and ``T`` stay, so a last interval
    whose ratio exceeds 0.5 is left as it is.

    Parameters
    ----------
    birth_rate : float
        Per-individual birth rate.
    death_rate : float
        Per-individual death rate.
    terminal_time : float
        Terminal time T.
    n_levels : int | None, optional
        Number of intervals ``K`` in the starting grid
        ``linspace(0, T, K + 1)``.  If ``None``, ``K`` is chosen so that
        ``Pr(T) ** (1 / K)`` is about 1/3.  Ignored if ``initial_level_times``
        is given.
    initial_level_times : list[float] | np.ndarray | None, optional
        Starting grid, including ``0`` and ``terminal_time``.

    Returns
    -------
    np.ndarray
        Strictly increasing level boundary array ``[0, ..., T]``.

    Raises
    ------
    ValueError
        If the starting grid is invalid or the refine step exceeds
        ``_MAX_GRID_POINTS`` points.

    """
    if n_levels is None and initial_level_times is None:
        n_levels = _initial_level_count(
            _log_survival_probability(
                birth_rate=birth_rate, death_rate=death_rate, time=terminal_time
            )
        )
    grid = _build_level_times(
        terminal_time=terminal_time,
        level_times=None if initial_level_times is None else list(initial_level_times),
        n_levels=n_levels,
    )
    times = [float(t) for t in grid]

    def log_pr(t: float) -> float:
        return _log_survival_probability(
            birth_rate=birth_rate, death_rate=death_rate, time=t
        )

    # --- Refine: bisect intervals whose ratio is too small ---
    log_prs = [log_pr(t) for t in times]
    while True:
        new_times = [times[0]]
        new_log_prs = [log_prs[0]]
        changed = False
        for i in range(1, len(times)):
            if log_prs[i] - log_prs[i - 1] < _LOG_BAND_LOWER:
                mid = 0.5 * (times[i - 1] + times[i])
                if times[i - 1] < mid < times[i]:
                    new_times.append(mid)
                    new_log_prs.append(log_pr(mid))
                    changed = True
            new_times.append(times[i])
            new_log_prs.append(log_prs[i])
        times, log_prs = new_times, new_log_prs
        if len(times) > _MAX_GRID_POINTS:
            msg = f"Level grid refinement exceeded {_MAX_GRID_POINTS} points."
            raise ValueError(msg)
        if not changed:
            break

    # --- Coarsen: drop points whose ratio is too large, keeping the band ---
    changed = True
    while changed:
        changed = False
        i = 1
        while i < len(times) - 1:  # interior points only: 0 and T stay
            too_large = log_prs[i] - log_prs[i - 1] > _LOG_BAND_UPPER
            merged_in_band = log_prs[i + 1] - log_prs[i - 1] >= _LOG_BAND_LOWER
            if too_large and merged_in_band:
                del times[i], log_prs[i]
                changed = True  # re-examine the new interval at position i
            else:
                i += 1

    return np.asarray(times, dtype=np.float64)


def compute_level_replicates(
    *,
    birth_rate: float,
    death_rate: float,
    level_times: list[float] | np.ndarray,
) -> np.ndarray:
    """Return the replicate count ``R_i`` for each level interval.

    ``R_i`` is the number of copies made at ``t_{i-1}`` of each surviving line
    (``R_1`` is the initial ensemble size).  It is the smallest integer
    ``>= 2`` such that ``R_1 * ... * R_i * Pr(t_i) >= 1``, i.e. the expected
    number of lines alive at ``t_i`` is at least one.

    Parameters
    ----------
    birth_rate : float
        Per-individual birth rate.
    death_rate : float
        Per-individual death rate.
    level_times : list[float] | np.ndarray
        Level boundary times ``0 = t_0 < ... < t_K``.

    Returns
    -------
    np.ndarray
        Integer array of length ``K``.

    Raises
    ------
    ValueError
        If some ``R_i`` would be astronomically large (a grid far outside the
        band).

    """
    times = np.asarray(level_times, dtype=np.float64)
    replicates = np.zeros(times.size - 1, dtype=np.int64)
    log_cumulative = 0.0
    for i in range(1, times.size):
        log_pr = _log_survival_probability(
            birth_rate=birth_rate, death_rate=death_rate, time=float(times[i])
        )
        needed = -(log_pr + log_cumulative)
        if needed > _MAX_LOG_REPLICATES:
            msg = f"Replicate count for interval {i} is too large; refine level_times."
            raise ValueError(msg)
        count = max(
            _MIN_REPLICATES,
            math.ceil(math.exp(needed) - _REPLICATE_TOLERANCE),
        )
        replicates[i - 1] = count
        log_cumulative += math.log(count)
    return replicates


def default_max_restarts(
    *,
    birth_rate: float,
    death_rate: float,
    level_times: list[float] | np.ndarray,
    replicates: list[int] | np.ndarray,
    target_success: float = DEFAULT_RESTART_SUCCESS_PROBABILITY,
) -> int:
    """Return the number of attempts so that at least one succeeds w.p. ``target``.

    An attempt fails if no line survives to ``T``.  Its failure probability
    ``p_fail`` is approximated as a Galton-Watson process in which each line
    survives interval ``i`` independently with probability ``r_i`` and every
    survivor is replaced by ``R_i`` copies.  Working backwards with
    ``F_{K+1} = 0``::

        F_i = (1 - r_i + r_i * F_{i+1}) ** R_i,    p_fail = F_1.

    The result is the smallest ``n >= 1`` with ``1 - p_fail ** n >= target``.
    The approximation ignores the ensemble cap.  Clones of one parent also share
    its starting state, which makes their survival positively correlated, so the
    real failure probability can be higher than ``p_fail`` (and sometimes
    lower); pass ``SplittingConfig.max_restarts`` when a run must not give up
    early.

    Parameters
    ----------
    birth_rate : float
        Per-individual birth rate.
    death_rate : float
        Per-individual death rate.
    level_times : list[float] | np.ndarray
        Level boundary times ``0 = t_0 < ... < t_K``.
    replicates : list[int] | np.ndarray
        Replicate counts, one per interval.
    target_success : float, optional
        Required probability that at least one attempt survives.

    Returns
    -------
    int
        Maximum number of attempts.

    """
    if not 0.0 < target_success < 1.0:
        msg = "target_success must lie strictly between 0 and 1"
        raise ValueError(msg)
    times = np.asarray(level_times, dtype=np.float64)
    reps = np.asarray(replicates, dtype=np.int64)
    log_prs = [
        _log_survival_probability(
            birth_rate=birth_rate, death_rate=death_rate, time=float(t)
        )
        for t in times
    ]
    failure = 0.0
    for i in range(reps.size, 0, -1):
        ratio = math.exp(log_prs[i] - log_prs[i - 1])
        failure = (1.0 - ratio + ratio * failure) ** int(reps[i - 1])
    if failure <= 0.0:
        return 1
    if failure >= 1.0:
        msg = "Attempts cannot succeed under the Galton-Watson approximation."
        raise ValueError(msg)
    return max(1, math.ceil(math.log(1.0 - target_success) / math.log(failure)))


def _initialize_particle(
    *,
    particle_id: int,
    seed_rng: np.random.Generator,
) -> _ParticleState:
    root_record = {
        "individual_id": 0,
        "parent_id": None,
        "creation_time": 0.0,
        "death_time": None,
        "is_spine": False,
    }
    return _ParticleState(
        particle_id=particle_id,
        current_time=0.0,
        next_individual_id=1,
        alive_ids=[0],
        records={0: root_record},
        trace_times=[0.0],
        trace_populations=[1],
        rng=_make_child_rng(seed_rng=seed_rng),
    )


def _append_trace_point(
    *,
    state: _ParticleState,
    time_point: float,
) -> None:
    if time_point < state.trace_times[-1]:
        msg = "time_point must be monotone in particle trace"
        raise RuntimeError(msg)
    if np.isclose(time_point, state.trace_times[-1]):
        state.trace_populations[-1] = len(state.alive_ids)
        return
    state.trace_times.append(time_point)
    state.trace_populations.append(len(state.alive_ids))


def _simulate_to_level(
    *,
    state: _ParticleState,
    birth_rate: float,
    death_rate: float,
    target_time: float,
) -> None:
    if target_time < state.current_time:
        msg = "target_time cannot be less than current_time"
        raise ValueError(msg)

    birth_probability = birth_rate / (birth_rate + death_rate)

    while state.current_time < target_time:
        if len(state.alive_ids) == 0:
            state.current_time = target_time
            _append_trace_point(state=state, time_point=target_time)
            return

        total_rate = len(state.alive_ids) * (birth_rate + death_rate)
        dt = float(state.rng.exponential(1.0 / total_rate))
        event_time = state.current_time + dt

        if event_time >= target_time:
            state.current_time = target_time
            _append_trace_point(state=state, time_point=target_time)
            return

        state.current_time = event_time
        if state.rng.uniform() < birth_probability:
            child_id = state.next_individual_id
            state.next_individual_id += 1
            # Insert at a random position >= 1 so index-0 lineage is never
            # displaced — matching a Gillespie birth-death simulator's
            # convention. Parent is the left neighbour at insert_idx - 1.
            if len(state.alive_ids) == 1:
                state.alive_ids.append(child_id)
                parent_id = state.alive_ids[0]
            else:
                insert_idx = int(state.rng.integers(1, len(state.alive_ids) + 1))
                state.alive_ids.insert(insert_idx, child_id)
                parent_id = state.alive_ids[insert_idx - 1]
            state.records[child_id] = {
                "individual_id": child_id,
                "parent_id": parent_id,
                "creation_time": event_time,
                "death_time": None,
                "is_spine": False,
            }
        else:
            death_idx = int(state.rng.integers(0, len(state.alive_ids)))
            dead_id = state.alive_ids.pop(death_idx)
            state.records[dead_id]["death_time"] = event_time

        _append_trace_point(state=state, time_point=event_time)


def _clone_particle(
    *,
    donor: _ParticleState,
    particle_id: int,
    seed_rng: np.random.Generator,
) -> _ParticleState:
    return _ParticleState(
        particle_id=particle_id,
        current_time=float(donor.current_time),
        next_individual_id=int(donor.next_individual_id),
        alive_ids=donor.alive_ids.copy(),
        records={key: value.copy() for key, value in donor.records.items()},
        trace_times=donor.trace_times.copy(),
        trace_populations=donor.trace_populations.copy(),
        rng=_make_child_rng(seed_rng=seed_rng),
    )


def _particle_to_tree(
    *,
    state: _ParticleState,
    terminal_time: float,
    birth_rate: float,
    death_rate: float,
) -> SimulatedTree:
    records_df = pd.DataFrame(state.records.values())
    if records_df.empty:
        records_df = pd.DataFrame(
            columns=[
                "individual_id",
                "parent_id",
                "creation_time",
                "death_time",
                "is_spine",
            ],
        )

    records_df["is_spine"] = False
    if state.alive_ids:
        spine_id = state.alive_ids[0]
        records_df.loc[records_df["individual_id"] == spine_id, "is_spine"] = True

    records_df["survived"] = records_df["death_time"].isna()
    records_df["age"] = np.where(
        records_df["survived"],
        terminal_time - records_df["creation_time"].to_numpy(dtype=np.float64),
        np.nan,
    )
    records_df["lifespan"] = np.where(
        ~records_df["survived"],
        records_df["death_time"].to_numpy(dtype=np.float64)
        - records_df["creation_time"].to_numpy(dtype=np.float64),
        np.nan,
    )

    return SimulatedTree(
        records=records_df,
        birth_rate=float(birth_rate),
        death_rate=float(death_rate),
        terminal_time=float(terminal_time),
    )


def _build_conditioned_trajectories(
    *,
    particles: list[_ParticleState],
    config: Any,
    seed_rng: np.random.Generator,
) -> list[ConditionedTrajectory]:
    survivors = [particle for particle in particles if len(particle.alive_ids) > 0]
    if not survivors:
        return []

    if config.n_conditioned_trajectories is None:
        n_take = len(survivors)
    else:
        if config.n_conditioned_trajectories <= 0:
            return []
        n_take = min(config.n_conditioned_trajectories, len(survivors))

    chosen_indices = seed_rng.choice(len(survivors), size=n_take, replace=False)
    trajectories: list[ConditionedTrajectory] = []
    for raw_idx in np.atleast_1d(chosen_indices):
        particle = survivors[int(raw_idx)]
        process = _particle_to_tree(
            state=particle,
            terminal_time=float(config.terminal_time),
            birth_rate=float(config.birth_rate),
            death_rate=float(config.death_rate),
        )
        trajectories.append(
            ConditionedTrajectory(
                particle_id=particle.particle_id,
                times=np.asarray(particle.trace_times, dtype=np.float64),
                populations=np.asarray(particle.trace_populations, dtype=np.int64),
                process=process,
            ),
        )
    return trajectories


def _enrich_survivors(
    *,
    survivors: np.ndarray,
    particles: list[_ParticleState],
    splitting_factor: int,
    n_particles_cap: int,
    seed_rng: np.random.Generator,
    particle_id_counter: int,
    level_index: int,
    target_time: float,
    lineage_rows: list[dict[str, Any]],
) -> tuple[list[_ParticleState], int]:
    """Clone survivors R-fold; subsample to cap if needed.

    Parameters
    ----------
    survivors : np.ndarray
        Indices into ``particles`` of survivors at the current level.
    particles : list[_ParticleState]
        Full particle list at the current level boundary.
    splitting_factor : int
        Number of copies made of each survivor.
    n_particles_cap : int
        Maximum ensemble size; subsampled to this if R * survivors > cap.
    seed_rng : np.random.Generator
        Master RNG used to draw child seeds.
    particle_id_counter : int
        Next available particle identifier.
    level_index : int
        Current level index (used in lineage records).
    target_time : float
        Current level boundary time (used in lineage records).
    lineage_rows : list[dict[str, Any]]
        Appended in-place with parent-child lineage entries.

    Returns
    -------
    tuple[list[_ParticleState], int]
        The enriched particle list and the updated ``particle_id_counter``.

    """
    target_count = splitting_factor * int(survivors.size)
    if target_count <= n_particles_cap:
        selected_slots = np.repeat(survivors, splitting_factor)
    else:
        selected_slots = seed_rng.choice(survivors, size=n_particles_cap, replace=True)

    next_particles: list[_ParticleState] = []
    for child_slot, parent_slot_raw in enumerate(np.atleast_1d(selected_slots)):
        parent_slot = int(parent_slot_raw)
        parent_particle = particles[parent_slot]
        child_particle = _clone_particle(
            donor=parent_particle,
            particle_id=particle_id_counter,
            seed_rng=seed_rng,
        )
        particle_id_counter += 1
        next_particles.append(child_particle)
        lineage_rows.append(
            {
                "level_index": level_index,
                "time": target_time,
                "child_slot": child_slot,
                "parent_slot": parent_slot,
                "child_particle_id": child_particle.particle_id,
                "parent_particle_id": parent_particle.particle_id,
            },
        )
    return next_particles, particle_id_counter


@dataclass
class _AttemptOutcome:
    """Everything one splitting attempt produces."""

    particles: list[_ParticleState]
    diagnostics: list[LevelDiagnostic]
    mean_populations: list[float]
    lineage_rows: list[dict[str, Any]]
    n_initial: int


def _run_attempt(
    *,
    config: SplittingConfig,
    levels: np.ndarray,
    replicates: np.ndarray,
    seed_rng: np.random.Generator,
) -> _AttemptOutcome:
    """Run one pass of multilevel splitting from time 0 to the terminal time."""
    n_particles_cap = int(config.n_particles)
    n_initial = min(int(replicates[0]), n_particles_cap)
    particles = [
        _initialize_particle(particle_id=i, seed_rng=seed_rng) for i in range(n_initial)
    ]

    diagnostics: list[LevelDiagnostic] = []
    mean_populations: list[float] = [1.0]
    lineage_rows: list[dict[str, Any]] = []
    particle_id_counter = n_initial

    for level_index in range(1, len(levels)):
        target_time = float(levels[level_index])
        n_entering = len(particles)

        for particle in particles:
            _simulate_to_level(
                state=particle,
                birth_rate=float(config.birth_rate),
                death_rate=float(config.death_rate),
                target_time=target_time,
            )

        populations_at_level = np.asarray(
            [len(particle.alive_ids) for particle in particles],
            dtype=np.int64,
        )
        survivors = np.flatnonzero(populations_at_level > 0)
        survivor_count = int(survivors.size)

        diagnostics.append(
            LevelDiagnostic(
                level_index=level_index,
                time=target_time,
                survivor_count=survivor_count,
                mean_population=float(np.mean(populations_at_level)),
                entering_count=n_entering,
            ),
        )
        mean_populations.append(float(np.mean(populations_at_level)))

        if survivor_count == 0:
            particles = []
            break

        # Enrich for the next interval; nothing to enrich after the last one.
        if level_index < len(levels) - 1:
            particles, particle_id_counter = _enrich_survivors(
                survivors=survivors,
                particles=particles,
                splitting_factor=int(replicates[level_index]),
                n_particles_cap=n_particles_cap,
                seed_rng=seed_rng,
                particle_id_counter=particle_id_counter,
                level_index=level_index,
                target_time=target_time,
                lineage_rows=lineage_rows,
            )

    return _AttemptOutcome(
        particles=particles,
        diagnostics=diagnostics,
        mean_populations=mean_populations,
        lineage_rows=lineage_rows,
        n_initial=n_initial,
    )


def _resolve_levels_and_replicates(
    config: SplittingConfig,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return ``(levels, replicates, direct)``.

    ``direct`` is true when survival to ``T`` is likely (``Pr(T) > 0.5``): the
    grid is just ``[0, T]`` with one replicate, so each attempt is a plain
    forward simulation that is kept if it survives.
    """
    birth_rate = float(config.birth_rate)
    death_rate = float(config.death_rate)
    terminal_time = float(config.terminal_time)

    if config.replicates is not None:
        if config.level_times is None:
            msg = "replicates requires level_times"
            raise ValueError(msg)
        levels = _build_level_times(
            terminal_time=terminal_time,
            level_times=config.level_times,
            n_levels=config.n_levels,
        )
        replicates = np.asarray(config.replicates, dtype=np.int64)
        if replicates.shape != (levels.size - 1,):
            msg = (
                f"replicates must have one entry per level interval "
                f"({levels.size - 1}), got {replicates.size}."
            )
            raise ValueError(msg)
        if np.any(replicates < 1):
            msg = "replicates must all be >= 1"
            raise ValueError(msg)
        return levels, replicates, False

    log_pr_terminal = _log_survival_probability(
        birth_rate=birth_rate, death_rate=death_rate, time=terminal_time
    )
    if log_pr_terminal > _LOG_BAND_UPPER:
        return (
            np.array([0.0, terminal_time], dtype=np.float64),
            np.ones(1, dtype=np.int64),
            True,
        )

    levels = select_level_times_for_band(
        birth_rate=birth_rate,
        death_rate=death_rate,
        terminal_time=terminal_time,
        n_levels=config.n_levels,
        initial_level_times=config.level_times,
    )
    replicates = compute_level_replicates(
        birth_rate=birth_rate, death_rate=death_rate, level_times=levels
    )
    return levels, replicates, False


def simulate_subcritical_survival_splitting(
    config: SplittingConfig,
) -> SplittingSimulationResult:
    """Sample survival-conditioned trajectories by multilevel (RESTART) splitting.

    Accepts any positive rates.  The name is historical: the method was
    written for the subcritical case, where survival is rare, but supercritical
    and critical processes work too (survival is then likely but not certain).

    Algorithm
    ---------
    **Levels.** Starting from ``config.level_times`` (or an equally spaced grid
    of ``K`` intervals with ``Pr(T) ** (1 / K)`` about 1/3),
    ``select_level_times_for_band`` adjusts the grid so each ratio
    ``Pr(t_i) / Pr(t_{i-1})`` lies in ``[e^{-2}, 0.5]``.

    **Direct case.** If ``Pr(T) > 0.5`` (always so when ``death_rate /
    birth_rate < 0.5``) there is nothing to split.  Trees are simulated one at
    a time, with grid ``[0, T]`` and one replicate, until one survives; unless
    ``config.max_restarts`` is given the number of attempts is unbounded.

    **Replicates.** ``compute_level_replicates`` gives each interval the
    smallest ``R_i >= 2`` keeping the expected number of live lines at
    least one.  ``R_1`` lines are started at time 0, and at ``t_{i-1}`` every
    survivor is copied ``R_i`` times.  If ``config.level_times`` and
    ``config.replicates`` are both given, they are used as supplied.

    **Each level k** is processed in three steps:

    1. **Evolve.** Every particle advances from ``t_{k-1}`` to ``t_k`` under
       exact Gillespie dynamics.
    2. **Cull.** Particles with no live individual at ``t_k`` are dropped.
    3. **Enrich.** Each survivor is copied ``R_{k+1}`` times (not after the
       last level).  If this exceeds ``config.n_particles`` the ensemble is
       subsampled uniformly with replacement to ``config.n_particles``.

    The particles alive at ``T`` are kept as the conditioned samples; no
    survival probability is estimated.  Clones of one survivor share their
    history up to the level where they were split, so the kept trajectories
    are not independent.

    **Restarts.** If no particle survives to ``T`` the attempt is restarted
    from time 0, up to ``max_restarts`` attempts in total (default from
    ``default_max_restarts``).  The first successful attempt is returned and
    failed attempts are discarded.  If every attempt fails a warning is issued
    and no trajectories are returned.

    Parameters
    ----------
    config : SplittingConfig
        Simulation parameters.  See ``SplittingConfig`` for field descriptions.

    Returns
    -------
    SplittingSimulationResult
        Contains:

        - ``level_mean_populations`` - mean population at each level boundary
          of the last attempt (including t_0, where it equals 1).
        - ``level_times`` - the boundary times used (truncated if the last
          attempt went extinct).
        - ``conditioned_trajectories`` - surviving terminal particles as
          ``ConditionedTrajectory`` objects, each carrying a full
          ``SimulatedTree`` with ``is_spine`` flags set.
        - ``particle_genealogy`` - parent-child particle lineage records of the
          last attempt.
        - ``level_diagnostics`` - per-level diagnostic records of the last
          attempt.
        - ``replicates`` - the ``R_i`` used.
        - ``initial_particle_count`` - ``min(R_1, config.n_particles)``.
        - ``n_attempts`` and ``max_restarts`` (``None`` if unbounded).

    Raises
    ------
    ValueError
        If a rate is not positive, ``n_particles < 1``, ``terminal_time <= 0``,
        ``max_restarts < 1``, or ``replicates`` is invalid or given without
        ``level_times``.

    """
    _validate_common_inputs(
        birth_rate=float(config.birth_rate),
        death_rate=float(config.death_rate),
        terminal_time=float(config.terminal_time),
        n_particles=int(config.n_particles),
    )

    levels, replicates, direct = _resolve_levels_and_replicates(config)

    max_restarts: int | None
    if config.max_restarts is None:
        max_restarts = (
            None
            if direct
            else default_max_restarts(
                birth_rate=float(config.birth_rate),
                death_rate=float(config.death_rate),
                level_times=levels,
                replicates=replicates,
            )
        )
    else:
        max_restarts = int(config.max_restarts)
        if max_restarts < 1:
            msg = f"max_restarts must be >= 1, got {config.max_restarts}."
            raise ValueError(msg)

    seed_rng = np.random.default_rng(config.random_seed)

    n_attempts = 0
    outcome: _AttemptOutcome
    while True:
        outcome = _run_attempt(
            config=config,
            levels=levels,
            replicates=replicates,
            seed_rng=seed_rng,
        )
        n_attempts += 1
        if outcome.particles or (
            max_restarts is not None and n_attempts >= max_restarts
        ):
            break

    if not outcome.particles:
        warnings.warn(
            f"No particle survived to the terminal time in {n_attempts} "
            "attempt(s); returning an empty result.",
            RuntimeWarning,
            stacklevel=2,
        )

    conditioned_trajectories = _build_conditioned_trajectories(
        particles=outcome.particles,
        config=config,
        seed_rng=seed_rng,
    )

    final_survivor_count = int(sum(len(p.alive_ids) > 0 for p in outcome.particles))
    particle_genealogy = pd.DataFrame(outcome.lineage_rows)
    if particle_genealogy.empty:
        particle_genealogy = pd.DataFrame(
            columns=[
                "level_index",
                "time",
                "child_slot",
                "parent_slot",
                "child_particle_id",
                "parent_particle_id",
            ],
        )

    return SplittingSimulationResult(
        config=config,
        level_times=levels[: len(outcome.diagnostics) + 1],
        level_diagnostics=outcome.diagnostics,
        level_mean_populations=np.asarray(outcome.mean_populations, dtype=np.float64),
        final_survivor_count=final_survivor_count,
        conditioned_trajectories=conditioned_trajectories,
        particle_genealogy=particle_genealogy,
        replicates=replicates,
        initial_particle_count=outcome.n_initial,
        n_attempts=n_attempts,
        max_restarts=max_restarts,
    )
