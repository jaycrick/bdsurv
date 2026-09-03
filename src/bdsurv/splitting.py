"""Splitting-based rare-event simulation for subcritical birth-death processes.

This module provides continuous-time simulation utilities for sampling
subcritical birth-death branching-process trajectories conditioned on survival
up to a large terminal time.

Implemented methods:
- Multilevel splitting with exact Gillespie trajectory evolution

The method exposes level diagnostics, conditioned trajectory samples, and
particle ancestry metadata suitable for downstream analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from bdsurv._extinction import pr_extinction
from bdsurv.tree import SimulatedTree

MIN_LEVEL_TIMES = 2
_BAND_LOWER: float = np.e**-2.0  # ≈ 0.135; at this pr_ext, R = ceil(e^2) = 8
_BAND_UPPER: float = 0.5  # at this pr_ext, R = ceil(2) = 2
_MAX_BAND_LEVELS: int = 100_000  # defensive cap on the equal-split search


@dataclass(frozen=True)
class SplittingConfig:
    """Configuration for multilevel splitting in a subcritical regime.

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
        ensemble size is ``ceil(1 / pr_ext)`` (the splitting factor R), which
        may be smaller than this cap.  After R-fold duplication of survivors,
        if the ensemble would exceed ``n_particles`` it is subsampled uniformly
        back down to ``n_particles``.
    level_times : list[float] | None, optional
        Explicit level boundary times (including 0 and `terminal_time`). If
        ``None``, levels are automatically chosen so that the per-increment
        extinction probability lies in ``[e^{-2}, 0.5]``; in that case
        ``n_levels`` is ignored.
    n_levels : int, optional
        Number of level intervals to create when ``level_times`` is not
        provided.  Ignored when ``level_times`` is ``None`` (levels are
        auto-selected from the band constraint).  Defaults to 10.
    random_seed : int | None, optional
        Seed for random number generation to allow reproducible runs.
    n_conditioned_trajectories : int | None, optional
        Maximum number of conditioned trajectories to return for downstream
        analysis. If ``None``, all surviving terminal trajectories are
        returned.
    splitting_factor : int | None, optional
        Optional manual override for the enrichment factor R. When provided
        it is used directly instead of ``ceil(1/pr_ext)``. When both
        ``level_times`` and ``splitting_factor`` are supplied, the band
        validation on the first increment is bypassed, so a predetermined
        out-of-band grid can be used without error.

    """

    birth_rate: float
    death_rate: float
    terminal_time: float
    n_particles: int
    level_times: list[float] | None = None
    n_levels: int = 10
    random_seed: int | None = None
    n_conditioned_trajectories: int | None = None
    splitting_factor: int | None = None


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
    conditional_survival_estimate : float
        Empirical conditional survival probability for this level interval,
        computed as ``survivor_count / entering_count``.
    cumulative_survival_estimate : float
        Cumulative product of conditional survival estimates up to this
        level, estimating overall survival to the current time.
    mean_population : float
        Ensemble mean population size at the level boundary.
    entering_count : int | None, optional
        Number of particles that entered this level interval.

    """

    level_index: int
    time: float
    survivor_count: int
    conditional_survival_estimate: float
    cumulative_survival_estimate: float
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
    survival_probability_estimate : float
        Estimated probability of survival to the terminal time (product of
        conditional estimates across levels).
    level_diagnostics : list[LevelDiagnostic]
        Per-level diagnostic objects describing survival and population
        statistics.
    level_survival_probabilities : np.ndarray
        Empirical conditional survival probabilities for each level
        interval (length = len(level_times) - 1).
    level_mean_populations : np.ndarray
        Ensemble mean population at each level boundary.
    final_survivor_count : int
        Number of surviving records/lineages at the terminal time.
    conditioned_trajectories : list[ConditionedTrajectory]
        Sampled conditioned trajectories selected from surviving particles.
    particle_genealogy : pd.DataFrame
        Tabular parent-child lineage records produced during resampling.

    """

    config: SplittingConfig
    level_times: np.ndarray
    survival_probability_estimate: float
    level_diagnostics: list[LevelDiagnostic]
    level_survival_probabilities: np.ndarray
    level_mean_populations: np.ndarray
    final_survivor_count: int
    conditioned_trajectories: list[ConditionedTrajectory]
    particle_genealogy: pd.DataFrame
    splitting_factor: int = 1
    initial_particle_count: int = 1


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
    if birth_rate >= death_rate:
        msg = "Subcritical regime required: birth_rate must be < death_rate"
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
    n_levels: int,
) -> np.ndarray:
    if level_times is None:
        if n_levels <= 0:
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


def _calculate_extinction_probability(
    *,
    birth_rate: float,
    death_rate: float,
    interval_time: float,
) -> float:
    probability = pr_extinction(birth_rate, death_rate, interval_time)
    return float(np.clip(probability, 0.0, 1.0))


def _adaptive_particle_count(
    *,
    birth_rate: float,
    death_rate: float,
    interval_time: float,
) -> int:
    extinction_probability = _calculate_extinction_probability(
        birth_rate=birth_rate,
        death_rate=death_rate,
        interval_time=interval_time,
    )
    if extinction_probability <= 0.0:
        return 1
    return int(np.ceil(1.0 / extinction_probability))


def select_level_times_for_band(
    *,
    birth_rate: float,
    death_rate: float,
    terminal_time: float,
) -> np.ndarray:
    """Auto-select an equal-spaced level grid balancing enrichment against extinction.

    Enumerates equal-split counts ``n = 1, 2, ...``.  Each gives an increment
    ``tau = T / n`` with single-particle extinction probability
    ``pr = pr_ext(tau)`` (monotonically decreasing in ``n``) and splitting
    factor ``R = ceil(1 / pr)``.  Among the splits whose ``pr`` lies in the band
    ``[_BAND_LOWER, _BAND_UPPER]`` (equivalently ``R in [2, 8]``), it selects the
    ``n`` minimising ``|(R * pr)^n - 1|`` — i.e. the grid whose cumulative
    R-fold enrichment across all ``n`` levels most nearly offsets the
    cumulative die-off. The chosen factor is constant across the whole
    process because the grid is equal-spaced.

    Parameters
    ----------
    birth_rate : float
        Per-individual birth rate.
    death_rate : float
        Per-individual death rate (must exceed ``birth_rate``).
    terminal_time : float
        Terminal time T.

    Returns
    -------
    np.ndarray
        Strictly increasing level boundary array ``[0, ..., T]`` of length
        ``n_levels + 1``, where ``n_levels >= 1`` and the per-increment
        extinction probability lies in ``[_BAND_LOWER, _BAND_UPPER]``.

    Raises
    ------
    ValueError
        If no equal-split grid with per-increment ``pr_ext`` in the band can be
        found (e.g. the process is not subcritical, or ``terminal_time`` is so
        short that even a single-level split falls outside the band).

    """
    best_score = np.inf
    best_n = 0
    n = 1
    while n <= _MAX_BAND_LEVELS:
        pr = _calculate_extinction_probability(
            birth_rate=birth_rate,
            death_rate=death_rate,
            interval_time=terminal_time / n,
        )
        if pr < _BAND_LOWER:
            # pr_ext is monotone decreasing in n; every larger n is also below the band
            break
        if pr <= _BAND_UPPER:
            splitting_factor = int(np.ceil(1.0 / pr))
            score = abs((splitting_factor * pr) ** n - 1.0)
            if score < best_score:
                best_score = score
                best_n = n
        n += 1

    if best_n == 0:
        msg = (
            "Cannot find an equal-split grid with per-increment pr_ext in the "
            f"band [{_BAND_LOWER:.4f}, {_BAND_UPPER:.4f}] for the given "
            "birth_rate, death_rate, and terminal_time. Check that the process "
            "is subcritical and terminal_time is large enough that extinction "
            "over the whole horizon is at least e^{-2}."
        )
        raise ValueError(msg)

    return np.linspace(0.0, terminal_time, num=best_n + 1, dtype=np.float64)


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
        R = ceil(1 / pr_ext); each survivor is cloned this many times.
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


def simulate_subcritical_survival_splitting(
    config: SplittingConfig,
) -> SplittingSimulationResult:
    """Run fixed-factor enrichment (RESTART) splitting to estimate subcritical survival.

    Requires a strictly subcritical process (``birth_rate < death_rate``).

    Algorithm
    ---------
    **Band constraint.** The level grid is chosen so that the single-particle
    extinction probability over each increment satisfies
    ``pr_ext in [e^{-2}, 0.5]``.  This keeps the splitting factor
    ``R = ceil(1 / pr_ext)`` in the range ``[2, 8]``, balancing variance
    reduction against computational cost.

    **Initialisation.** When ``config.level_times`` is ``None``, the grid is
    auto-selected by enumeration over equal-split counts
    (``select_level_times_for_band``); otherwise the caller-supplied grid is
    used and its first increment is validated against the band. The initial
    ensemble size is ``N_0 = min(R, config.n_particles)``.

    **Each level k** is processed in three steps:

    1. **Evolve.** Every particle advances from ``t_{k-1}`` to ``t_k`` under
       exact Gillespie dynamics.
    2. **Estimate.** The empirical conditional survival fraction is
       ``p_k = S_k / N_{k-1}``, where ``S_k`` is the survivor count and
       ``N_{k-1}`` is the entering particle count (not the cap).
    3. **Enrich.** Each surviving particle is duplicated R times (R-fold
       enrichment).  If ``R * S_k > config.n_particles`` the enlarged ensemble
       is subsampled uniformly with replacement to ``config.n_particles``; this
       caps cost while keeping the estimator unbiased.

    The final survival estimate is the product
    ``P(survival to T) ~= p_1 * p_2 * ... * p_K``.

    If all particles go extinct at any level, the algorithm stops early
    (``survival_probability_estimate`` will be zero) and the returned
    ``level_times`` array will be shorter than the full K+1 entries.

    Parameters
    ----------
    config : SplittingConfig
        Simulation parameters.  See ``SplittingConfig`` for field descriptions.
        ``n_particles`` acts as a cap on ensemble size; ``n_levels`` is ignored
        when ``level_times`` is ``None``.

    Returns
    -------
    SplittingSimulationResult
        Contains:

        - ``survival_probability_estimate`` - product of the per-level
          empirical fractions.
        - ``level_survival_probabilities`` - the individual ``p_k`` values.
        - ``level_mean_populations`` - mean population at each level boundary
          (including t_0, where it equals 1).
        - ``level_times`` - the boundary times actually used (may be truncated
          on early extinction).
        - ``conditioned_trajectories`` - surviving terminal particles as
          ``ConditionedTrajectory`` objects, each carrying a full
          ``SimulatedTree`` with ``is_spine`` flags set.
        - ``particle_genealogy`` - DataFrame of parent-child particle lineage
          records at each enrichment step.
        - ``level_diagnostics`` - per-level diagnostic records including
          ``entering_count`` (particles entering that level).
        - ``splitting_factor`` - the R value derived from the first increment.
        - ``initial_particle_count`` - ``N_0 = min(R, config.n_particles)``.

    Raises
    ------
    ValueError
        If ``birth_rate >= death_rate`` (not subcritical), ``n_particles < 1``,
        ``terminal_time <= 0``, or if ``level_times`` is supplied without a
        ``splitting_factor`` override but its first increment violates the band
        constraint ``[e^{-2}, 0.5]``, or if ``splitting_factor < 1``.

    """
    _validate_common_inputs(
        birth_rate=float(config.birth_rate),
        death_rate=float(config.death_rate),
        terminal_time=float(config.terminal_time),
        n_particles=int(config.n_particles),
    )

    # --- Level selection ---
    if config.level_times is None:
        levels = select_level_times_for_band(
            birth_rate=float(config.birth_rate),
            death_rate=float(config.death_rate),
            terminal_time=float(config.terminal_time),
        )
    else:
        levels = _build_level_times(
            terminal_time=float(config.terminal_time),
            level_times=config.level_times,
            n_levels=int(config.n_levels),
        )
        # When the caller supplies both an explicit grid and an explicit
        # splitting_factor, they have predetermined the configuration and
        # the band check is bypassed.  Otherwise validate that the first
        # increment falls within [e^{-2}, 0.5] to ensure R in [2, 8].
        if config.splitting_factor is None:
            first_increment = float(levels[1]) - float(levels[0])
            first_pr_ext = _calculate_extinction_probability(
                birth_rate=float(config.birth_rate),
                death_rate=float(config.death_rate),
                interval_time=first_increment,
            )
            if not (_BAND_LOWER <= first_pr_ext <= _BAND_UPPER):
                msg = (
                    f"First increment extinction probability {first_pr_ext:.4f} "
                    "is outside "
                    f"the required band [{_BAND_LOWER:.4f}, {_BAND_UPPER:.4f}]. "
                    "Adjust level_times, leave level_times=None for auto-selection, "
                    "or supply splitting_factor to bypass this check."
                )
                raise ValueError(msg)

    # --- Splitting factor from first increment (or manual override) ---
    if config.splitting_factor is not None:
        if int(config.splitting_factor) < 1:
            msg = f"splitting_factor must be >= 1, got {config.splitting_factor}."
            raise ValueError(msg)
        splitting_factor = int(config.splitting_factor)
    else:
        first_increment_time = float(levels[1]) - float(levels[0])
        splitting_factor = _adaptive_particle_count(
            birth_rate=float(config.birth_rate),
            death_rate=float(config.death_rate),
            interval_time=first_increment_time,
        )
    n_particles_cap = int(config.n_particles)
    n_initial = min(splitting_factor, n_particles_cap)

    seed_rng = np.random.default_rng(config.random_seed)
    particles = [
        _initialize_particle(particle_id=i, seed_rng=seed_rng) for i in range(n_initial)
    ]

    diagnostics: list[LevelDiagnostic] = []
    level_probs: list[float] = []
    mean_populations: list[float] = [1.0]
    lineage_rows: list[dict[str, Any]] = []
    particle_id_counter = n_initial
    cumulative = 1.0

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
        # Correct denominator: entering count, not cap
        conditional_prob = survivor_count / float(n_entering)
        cumulative *= conditional_prob

        diagnostics.append(
            LevelDiagnostic(
                level_index=level_index,
                time=target_time,
                survivor_count=survivor_count,
                conditional_survival_estimate=conditional_prob,
                cumulative_survival_estimate=cumulative,
                mean_population=float(np.mean(populations_at_level)),
                entering_count=n_entering,
            ),
        )
        level_probs.append(conditional_prob)
        mean_populations.append(float(np.mean(populations_at_level)))

        if survivor_count == 0:
            particles = []
            break

        # --- R-fold enrichment with cap ---
        particles, particle_id_counter = _enrich_survivors(
            survivors=survivors,
            particles=particles,
            splitting_factor=splitting_factor,
            n_particles_cap=n_particles_cap,
            seed_rng=seed_rng,
            particle_id_counter=particle_id_counter,
            level_index=level_index,
            target_time=target_time,
            lineage_rows=lineage_rows,
        )

    conditioned_trajectories = _build_conditioned_trajectories(
        particles=particles,
        config=config,
        seed_rng=seed_rng,
    )

    final_survivor_count = int(sum(len(p.alive_ids) > 0 for p in particles))
    particle_genealogy = pd.DataFrame(lineage_rows)
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
        level_times=levels[: len(level_probs) + 1],
        survival_probability_estimate=float(cumulative),
        level_diagnostics=diagnostics,
        level_survival_probabilities=np.asarray(level_probs, dtype=np.float64),
        level_mean_populations=np.asarray(mean_populations, dtype=np.float64),
        final_survivor_count=final_survivor_count,
        conditioned_trajectories=conditioned_trajectories,
        particle_genealogy=particle_genealogy,
        splitting_factor=splitting_factor,
        initial_particle_count=n_initial,
    )
