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
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from bdsurv._forward import ForwardState, Segment, advance
from bdsurv._hazard import (
    close,
    log_survival_probability,
    log_survival_probability_constant,
)
from bdsurv._seeding import bump, root_rng, spawn
from bdsurv.rates import (
    ConstantRateLike,
    LinearRateLike,
    check_positive_rates,
    check_strictly_increasing,
    check_terminal_time,
    merged_segments,
    resolve_rates,
)
from bdsurv.tree import SimulatedTree

MIN_LEVEL_TIMES = 2
# The band of allowed ratios Pr(t_i) / Pr(t_{i-1}), as logs and as ratios.
_LOG_BAND_LOWER: float = -2.0
_LOG_BAND_UPPER: float = math.log(0.5)
_BAND_LOWER: float = math.exp(_LOG_BAND_LOWER)
_BAND_UPPER: float = math.exp(_LOG_BAND_UPPER)
_TARGET_LEVEL_RATIO: float = 1.0 / 3.0  # per-level survival ratio of the start grid
_MAX_GRID_POINTS: int = 100_000  # defensive cap on the refine step
_MIN_REPLICATES: int = 2
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
class LinearSplittingConfig:
    """Configuration for multilevel splitting with time-varying linear rates.

    The fields are those of :class:`SplittingConfig`, except that the rates
    are piecewise constant or piecewise linear in time over ``[0, T]`` and the
    terminal time ``T`` is their common end time (``terminal_time``).

    Parameters
    ----------
    birth_rate, death_rate : ConstantRateLike | LinearRateLike
        Per-individual rates with a common end time.

    """

    birth_rate: ConstantRateLike | LinearRateLike
    death_rate: ConstantRateLike | LinearRateLike
    n_particles: int
    level_times: list[float] | None = None
    n_levels: int | None = None
    random_seed: int | None = None
    n_conditioned_trajectories: int | None = None
    replicates: list[int] | None = None
    max_restarts: int | None = None

    @property
    def terminal_time(self) -> float:
        """The common end time of the rates."""
        return float(np.atleast_1d(self.birth_rate.epoch_end_times)[-1])


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
    n_dropped_by_cap : int, optional
        Clones discarded when enriching the survivors of this level, because
        ``replicates * survivors`` exceeded ``n_particles``.  Zero when the cap
        did not bind and after the last level, which is not enriched.

    """

    level_index: int
    time: float
    survivor_count: int
    mean_population: float
    entering_count: int | None = None
    n_dropped_by_cap: int = 0


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
    config : SplittingConfig | LinearSplittingConfig
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
    n_dropped_by_cap : int
        Clones discarded by the ``n_particles`` cap, summed over all levels and
        all attempts (``level_diagnostics`` covers the last attempt only).
    n_capped_levels : int
        Number of (attempt, level) enrichments where the cap bound.

    """

    config: SplittingConfig | LinearSplittingConfig
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
    n_dropped_by_cap: int = 0
    n_capped_levels: int = 0


@dataclass
class _ParticleState:
    """Mutable per-particle state: the forward population plus its trace.

    Parameters
    ----------
    particle_id : int
        Unique identifier assigned to the particle instance.
    forward : ForwardState
        The population (alive individuals and every individual's record).
    trace_times : list[float]
        Times at which the particle trace population was recorded.
    trace_populations : list[int]
        Population sizes corresponding to ``trace_times``.
    rng : np.random.Generator
        Random number generator instance used for stochastic events.

    """

    particle_id: int
    forward: ForwardState
    trace_times: list[float]
    trace_populations: list[int]
    rng: np.random.Generator


def _check_n_particles(n_particles: int) -> None:
    if n_particles <= 0:
        msg = "n_particles must be positive"
        raise ValueError(msg)


def _as_level_array(level_times: list[float] | np.ndarray) -> np.ndarray:
    """Return ``level_times`` as an array, checked to be a finite increasing grid."""
    levels = np.asarray(level_times, dtype=np.float64)
    if levels.ndim != 1:
        msg = "level_times must be one-dimensional"
        raise ValueError(msg)
    if levels.size < MIN_LEVEL_TIMES:
        msg = "At least two level times are required"
        raise ValueError(msg)
    if not np.all(np.isfinite(levels)):
        msg = "level_times must be finite"
        raise ValueError(msg)
    check_strictly_increasing("level_times", levels)
    return levels


def _check_replicates(
    replicates: list[int] | np.ndarray, n_intervals: int
) -> np.ndarray:
    """Return ``replicates`` as integers, one at least 1 per level interval."""
    counts = np.asarray(replicates, dtype=np.int64)
    if counts.shape != (n_intervals,):
        msg = (
            f"replicates must have one entry per level interval "
            f"({n_intervals}), got {counts.size}."
        )
        raise ValueError(msg)
    if np.any(counts < 1):
        msg = "replicates must all be >= 1"
        raise ValueError(msg)
    return counts


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
        levels = level_times
    levels = _as_level_array(levels)
    if not np.isclose(levels[0], 0.0):
        msg = "First level time must be 0"
        raise ValueError(msg)
    if not np.isclose(levels[-1], terminal_time):
        msg = "Last level time must equal terminal_time"
        raise ValueError(msg)
    return levels


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
        If a rate or ``terminal_time`` is not finite and positive, the starting
        grid is invalid, or the refine step exceeds ``_MAX_GRID_POINTS`` points.

    """
    check_positive_rates(birth_rate, death_rate)
    check_terminal_time(terminal_time)
    return _select_level_times(
        log_pr=lambda t: log_survival_probability_constant(
            birth_rate=birth_rate, death_rate=death_rate, time=t
        ),
        terminal_time=terminal_time,
        n_levels=n_levels,
        initial_level_times=initial_level_times,
    )


def _select_level_times(
    *,
    log_pr: Callable[[float], float],
    terminal_time: float,
    n_levels: int | None,
    initial_level_times: list[float] | np.ndarray | None,
) -> np.ndarray:
    """Implement :func:`select_level_times_for_band` for any ``log Pr(t)``."""
    if n_levels is None and initial_level_times is None:
        n_levels = _initial_level_count(log_pr(terminal_time))
    grid = _build_level_times(
        terminal_time=terminal_time,
        level_times=None if initial_level_times is None else list(initial_level_times),
        n_levels=n_levels,
    )
    times = [float(t) for t in grid]
    times, log_prs = _refine_grid(times, log_pr)
    return np.asarray(_coarsen_grid(times, log_prs), dtype=np.float64)


def _refine_grid(
    times: list[float], log_pr: Callable[[float], float]
) -> tuple[list[float], list[float]]:
    """Bisect intervals whose ratio is below the band until none is."""
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
            return times, log_prs


def _coarsen_grid(times: list[float], log_prs: list[float]) -> list[float]:
    """Drop points whose ratio is too large while the merged ratio stays in band."""
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
    return times


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
        If a rate is not finite and positive, ``level_times`` is not a finite
        strictly increasing grid, or some ``R_i`` would be astronomically large
        (a grid far outside the band).

    """
    check_positive_rates(birth_rate, death_rate)
    return _level_replicates(
        log_pr=lambda t: log_survival_probability_constant(
            birth_rate=birth_rate, death_rate=death_rate, time=t
        ),
        level_times=level_times,
    )


def _level_replicates(
    *,
    log_pr: Callable[[float], float],
    level_times: list[float] | np.ndarray,
) -> np.ndarray:
    """Implement :func:`compute_level_replicates` for any ``log Pr(t)``."""
    times = _as_level_array(level_times)
    replicates = np.zeros(times.size - 1, dtype=np.int64)
    log_cumulative = 0.0
    for i in range(1, times.size):
        needed = -(log_pr(float(times[i])) + log_cumulative)
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

    Raises
    ------
    ValueError
        If a rate is not finite and positive, ``level_times`` is not a finite
        strictly increasing grid, ``replicates`` is not one integer ``>= 1`` per
        interval, or ``target_success`` is outside ``(0, 1)``.

    """
    check_positive_rates(birth_rate, death_rate)
    return _max_restarts(
        log_pr=lambda t: log_survival_probability_constant(
            birth_rate=birth_rate, death_rate=death_rate, time=t
        ),
        level_times=level_times,
        replicates=replicates,
        target_success=target_success,
    )


def _max_restarts(
    *,
    log_pr: Callable[[float], float],
    level_times: list[float] | np.ndarray,
    replicates: list[int] | np.ndarray,
    target_success: float,
) -> int:
    """Implement :func:`default_max_restarts` for any ``log Pr(t)``."""
    if not 0.0 < target_success < 1.0:
        msg = "target_success must lie strictly between 0 and 1"
        raise ValueError(msg)
    times = _as_level_array(level_times)
    reps = _check_replicates(replicates, times.size - 1)
    log_prs = [log_pr(float(t)) for t in times]
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
    return _ParticleState(
        particle_id=particle_id,
        forward=ForwardState.root(),
        trace_times=[0.0],
        trace_populations=[1],
        rng=spawn(seed_rng, "dynamics"),
    )


def _append_trace_point(
    *,
    state: _ParticleState,
    time_point: float,
) -> None:
    if time_point < state.trace_times[-1]:
        msg = "time_point must be monotone in particle trace"
        raise RuntimeError(msg)
    if close(time_point, state.trace_times[-1]):
        state.trace_populations[-1] = len(state.forward.alive_ids)
        return
    state.trace_times.append(time_point)
    state.trace_populations.append(len(state.forward.alive_ids))


def _advance_particle(
    particle: _ParticleState, segments: list[Segment], target_time: float
) -> None:
    """Advance a particle to ``target_time``, tracing its population."""
    advance(
        particle.forward,
        segments,
        target_time,
        particle.rng,
        on_event=lambda _state, event_time: _append_trace_point(
            state=particle, time_point=event_time
        ),
    )
    _append_trace_point(state=particle, time_point=target_time)


def _clone_particle(
    *,
    donor: _ParticleState,
    particle_id: int,
    seed_rng: np.random.Generator,
) -> _ParticleState:
    bump("clones")
    bump("records_copied", donor.forward.n_individuals)
    return _ParticleState(
        particle_id=particle_id,
        forward=donor.forward.copy(),
        trace_times=donor.trace_times.copy(),
        trace_populations=donor.trace_populations.copy(),
        rng=spawn(seed_rng, "dynamics"),
    )


def _particle_to_tree(
    *,
    state: _ParticleState,
    terminal_time: float,
    birth_rate: float,
    death_rate: float,
    birth_rate_function: ConstantRateLike | LinearRateLike | None = None,
    death_rate_function: ConstantRateLike | LinearRateLike | None = None,
) -> SimulatedTree:
    return SimulatedTree(
        records=state.forward.to_records(terminal_time),
        birth_rate=float(birth_rate),
        death_rate=float(death_rate),
        terminal_time=float(terminal_time),
        birth_rate_function=birth_rate_function,
        death_rate_function=death_rate_function,
    )


@dataclass(frozen=True)
class _Dynamics:
    """What distinguishes constant-rate from linear-rate splitting."""

    terminal_time: float
    log_pr: Callable[[float], float]
    segments: list[Segment]
    tree_rates: dict[str, Any]


def _build_conditioned_trajectories(
    *,
    particles: list[_ParticleState],
    config: Any,
    seed_rng: np.random.Generator,
    dynamics: _Dynamics,
) -> list[ConditionedTrajectory]:
    survivors = [
        particle for particle in particles if len(particle.forward.alive_ids) > 0
    ]
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
            terminal_time=dynamics.terminal_time,
            **dynamics.tree_rates,
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


class _Level(NamedTuple):
    """A level boundary: its 1-based index and its time."""

    number: int
    time: float


def _enrich_survivors(
    *,
    particles: list[_ParticleState],
    splitting_factor: int,
    n_particles_cap: int,
    seed_rng: np.random.Generator,
    particle_id_counter: int,
    level: _Level,
) -> tuple[list[_ParticleState], int, int, list[dict[str, Any]]]:
    """Clone survivors R-fold; subsample to cap if needed.

    Parameters
    ----------
    particles : list[_ParticleState]
        Full particle list at the current level boundary; those with a live
        individual are the survivors.
    splitting_factor : int
        Number of copies made of each survivor.
    n_particles_cap : int
        Maximum ensemble size; subsampled to this if R * survivors > cap.
    seed_rng : np.random.Generator
        Master RNG used to draw child seeds.
    particle_id_counter : int
        Next available particle identifier.
    level : _Level
        Current level (index and time, used in lineage records).

    Returns
    -------
    tuple[list[_ParticleState], int, int, list[dict[str, Any]]]
        The enriched particle list, the updated ``particle_id_counter``, the
        number of clones dropped by the cap (0 if it did not bind) and the
        parent-child lineage rows, one per clone.

    """
    survivors = np.flatnonzero([len(p.forward.alive_ids) > 0 for p in particles])
    lineage_rows: list[dict[str, Any]] = []
    target_count = splitting_factor * int(survivors.size)
    n_dropped = max(0, target_count - n_particles_cap)
    if n_dropped == 0:
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
                "level_index": level.number,
                "time": level.time,
                "child_slot": child_slot,
                "parent_slot": parent_slot,
                "child_particle_id": child_particle.particle_id,
                "parent_particle_id": parent_particle.particle_id,
            },
        )
    return next_particles, particle_id_counter, n_dropped, lineage_rows


@dataclass
class _AttemptOutcome:
    """Everything one splitting attempt produces."""

    particles: list[_ParticleState]
    diagnostics: list[LevelDiagnostic]
    mean_populations: list[float]
    lineage_rows: list[dict[str, Any]]
    n_initial: int
    n_dropped_by_cap: int = 0
    n_capped_levels: int = 0


def _run_attempt(
    *,
    config: SplittingConfig | LinearSplittingConfig,
    levels: np.ndarray,
    replicates: np.ndarray,
    seed_rng: np.random.Generator,
    dynamics: _Dynamics,
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
    n_dropped_by_cap = 0
    n_capped_levels = 0

    for level_index in range(1, len(levels)):
        target_time = float(levels[level_index])
        n_entering = len(particles)
        bump("particle_steps", n_entering)

        for particle in particles:
            _advance_particle(particle, dynamics.segments, target_time)

        populations_at_level = np.asarray(
            [len(particle.forward.alive_ids) for particle in particles],
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
            particles, particle_id_counter, n_dropped, new_rows = _enrich_survivors(
                particles=particles,
                splitting_factor=int(replicates[level_index]),
                n_particles_cap=n_particles_cap,
                seed_rng=seed_rng,
                particle_id_counter=particle_id_counter,
                level=_Level(level_index, target_time),
            )
            lineage_rows.extend(new_rows)
            if n_dropped:
                diagnostics[-1] = replace(diagnostics[-1], n_dropped_by_cap=n_dropped)
                n_dropped_by_cap += n_dropped
                n_capped_levels += 1
                bump("dropped_by_cap", n_dropped)
                bump("capped_levels")

    return _AttemptOutcome(
        particles=particles,
        diagnostics=diagnostics,
        mean_populations=mean_populations,
        lineage_rows=lineage_rows,
        n_initial=n_initial,
        n_dropped_by_cap=n_dropped_by_cap,
        n_capped_levels=n_capped_levels,
    )


def _resolve_levels_and_replicates(
    config: SplittingConfig | LinearSplittingConfig,
    dynamics: _Dynamics,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return ``(levels, replicates, direct)``.

    ``direct`` is true when survival to ``T`` is likely (``Pr(T) > 0.5``): the
    grid is just ``[0, T]`` with one replicate, so each attempt is a plain
    forward simulation that is kept if it survives.
    """
    terminal_time = dynamics.terminal_time

    if config.replicates is not None:
        if config.level_times is None:
            msg = "replicates requires level_times"
            raise ValueError(msg)
        levels = _build_level_times(
            terminal_time=terminal_time,
            level_times=config.level_times,
            n_levels=config.n_levels,
        )
        replicates = _check_replicates(config.replicates, levels.size - 1)
        return levels, replicates, False

    log_pr_terminal = dynamics.log_pr(terminal_time)
    if log_pr_terminal > _LOG_BAND_UPPER:
        return (
            np.array([0.0, terminal_time], dtype=np.float64),
            np.ones(1, dtype=np.int64),
            True,
        )

    levels = _select_level_times(
        log_pr=dynamics.log_pr,
        terminal_time=terminal_time,
        n_levels=config.n_levels,
        initial_level_times=config.level_times,
    )
    replicates = _level_replicates(log_pr=dynamics.log_pr, level_times=levels)
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
    birth_rate, death_rate = float(config.birth_rate), float(config.death_rate)
    check_positive_rates(birth_rate, death_rate)
    check_terminal_time(float(config.terminal_time))
    _check_n_particles(int(config.n_particles))
    terminal_time = float(config.terminal_time)
    dynamics = _Dynamics(
        terminal_time=terminal_time,
        log_pr=lambda t: log_survival_probability_constant(
            birth_rate=birth_rate, death_rate=death_rate, time=t
        ),
        segments=merged_segments(
            *resolve_rates(birth_rate, death_rate, terminal_time)[:2]
        ),
        tree_rates={"birth_rate": birth_rate, "death_rate": death_rate},
    )
    return _run_splitting(config, dynamics)


def simulate_linear_survival_splitting(
    config: LinearSplittingConfig,
) -> SplittingSimulationResult:
    r"""Sample survival-conditioned trajectories by splitting, with linear rates.

    The algorithm is that of :func:`simulate_subcritical_survival_splitting`,
    with two changes for rates that vary in time.

    - **Survival probability.** A single individual alive at 0 has
      $\Pr(Z_t > 0) = 1 / (1 + I(t))$ with
      $I(t) = \int_0^t \mu(s) e^{-\Gamma(s)} ds$ and
      $\Gamma(s) = \int_0^s (\lambda - \mu)$, evaluated by panelled
      Gauss-Legendre quadrature in log space.  It drives the level grid, the
      replicate counts and the default restart cap.
    - **Dynamics.** Each particle is evolved by exact event simulation: within
      every linear segment the next event time inverts the cumulative total
      hazard, and the event is a birth with probability
      $\lambda(t) / (\lambda(t) + \mu(t))$.

    Parameters
    ----------
    config : LinearSplittingConfig
        Simulation parameters; ``terminal_time`` is the rates' end time.

    Returns
    -------
    SplittingSimulationResult
        As for the constant-rate version.  The trees carry NaN scalar rates
        and the rates themselves in ``birth_rate_function`` and
        ``death_rate_function``.

    Raises
    ------
    ValueError
        If the rates are invalid or differ in end time, ``n_particles < 1``,
        ``max_restarts < 1``, or ``replicates`` is invalid or given without
        ``level_times``.

    """
    birth, death, terminal_time = resolve_rates(
        config.birth_rate, config.death_rate, None
    )
    _check_n_particles(config.n_particles)
    dynamics = _Dynamics(
        terminal_time=terminal_time,
        log_pr=lambda t: log_survival_probability(birth, death, t),
        segments=merged_segments(birth, death),
        tree_rates={
            "birth_rate": math.nan,
            "death_rate": math.nan,
            "birth_rate_function": config.birth_rate,
            "death_rate_function": config.death_rate,
        },
    )
    return _run_splitting(config, dynamics)


def _run_splitting(
    config: SplittingConfig | LinearSplittingConfig, dynamics: _Dynamics
) -> SplittingSimulationResult:
    """Run levels, restarts and trajectory extraction for either dynamics."""
    levels, replicates, direct = _resolve_levels_and_replicates(config, dynamics)

    max_restarts: int | None
    if config.max_restarts is None:
        max_restarts = (
            None
            if direct
            else _max_restarts(
                log_pr=dynamics.log_pr,
                level_times=levels,
                replicates=replicates,
                target_success=DEFAULT_RESTART_SUCCESS_PROBABILITY,
            )
        )
    else:
        max_restarts = int(config.max_restarts)
        if max_restarts < 1:
            msg = f"max_restarts must be >= 1, got {config.max_restarts}."
            raise ValueError(msg)

    seed_rng = root_rng(config.random_seed, "seeding")

    n_attempts = 0
    total_dropped = 0
    total_capped_levels = 0
    outcome: _AttemptOutcome
    while True:
        outcome = _run_attempt(
            config=config,
            levels=levels,
            replicates=replicates,
            seed_rng=seed_rng,
            dynamics=dynamics,
        )
        n_attempts += 1
        total_dropped += outcome.n_dropped_by_cap
        total_capped_levels += outcome.n_capped_levels
        if outcome.particles or (
            max_restarts is not None and n_attempts >= max_restarts
        ):
            break

    bump("attempts", n_attempts)
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
        dynamics=dynamics,
    )

    final_survivor_count = int(
        sum(len(p.forward.alive_ids) > 0 for p in outcome.particles)
    )
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
        n_dropped_by_cap=total_dropped,
        n_capped_levels=total_capped_levels,
    )
