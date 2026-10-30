"""Conditioned survival sampling for multi-epoch birth-death processes.

Estimates the proportion of ever-created individuals currently alive
(surviving to the process terminal time) by conditioning forward simulations
on observed per-era survivor counts via a multilevel-splitting / sequential-ABC
scheme.

Algorithm (user's specification)
---------------------------------
For eras 0 … K-2 (the last era is left unconditioned):

1. Simulate ``n_initial`` independent era-0 particles from a single-root
   initialisation.
2. **Filter**: keep particles whose count of *born-in-era-0 individuals that
   survive to the terminal time* lies within ``target_ranges[0]``.
3. **Split**: clone each passing particle ``split_factor`` times (each clone
   receives a fresh PRNGKey; JAX arrays are immutable so clones are O(1)).
4. Simulate era 1 for each clone → filter on ``target_ranges[1]`` → split.
5. Repeat for eras 2 … K-2.
6. Simulate the final era (K-1) **unconditioned**.
7. Collect matched trajectories.  If fewer than ``n_samples`` are collected,
   restart from step 1 (bounded by ``max_attempts``).

Per-era match criterion
-----------------------
``#{i : birth_epoch[i] == k  and  death_time[i] > terminal_time}``

i.e. individuals born in era *k* whose pre-sampled death time exceeds the
terminal time (they survive to the present day).  This is computed from the
flat lifespan buffers the moment era *k* ends, so no additional simulation
is required.

Engine
------
The JAX lifespan-at-birth engine from
:func:`~bdsurv.lifespans.simulate_birth_death_lifespans_arrays`
is the simulation substrate.  A new single-era JIT core
(:func:`_simulate_era_core`) seeds from an arbitrary alive-population state
and advances the event loop to the era boundary, sampling new births' death
times over the **full** piecewise hazard so survival-to-present is always
well-defined.

Public API
----------
- :func:`build_target_ranges` — construct per-era ``[lo, hi]`` bounds from
  observed counts (optionally widened by a tolerance).
- :func:`sample_conditioned_survival_process` — draw ``n_samples`` matched
  trajectories for a fixed set of rates.
- :func:`sample_conditioned_survival_per_posterior` — one matched trajectory
  per posterior draw; returns a tidy ``pd.DataFrame``.
- :class:`MatchedSurvivalSample` — per-trajectory result dataclass.
- :class:`ConditionedSurvivalResult` — collection of samples plus diagnostics.
"""

# JAX's jit/while_loop wrappers are untyped, which strict mode reports as
# unknown.
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import functools
import warnings
from dataclasses import dataclass, field
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import lax
from numpy.typing import NDArray

from bdsurv.lifespans import (
    _sample_piecewise_exponential_death_time,  # pyright: ignore[reportPrivateUsage]
)

# ---------------------------------------------------------------------------
# JAX state type for the per-era while_loop
# ---------------------------------------------------------------------------


class _EraLoopState(NamedTuple):
    """State carried through the within-era ``jax.lax.while_loop``.

    Mirrors the relevant fields of ``_LifespanLoopState`` from
    ``lifespans.py``, but without ``epoch_idx`` and genealogy fields —
    those are not needed for the conditioning algorithm.
    """

    key: jax.Array  # PRNGKey
    current_time: jax.Array  # float64 scalar
    n_alive: jax.Array  # int32 scalar — individuals alive at current_time
    n_born: jax.Array  # int32 scalar — total ever born (next free slot)
    overflow: jax.Array  # bool scalar
    birth_time_buf: jax.Array  # float64 [max_population]
    death_time_buf: jax.Array  # float64 [max_population]  (inf = survives)
    birth_epoch_buf: jax.Array  # int32 [max_population]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EraConditioningDiagnostic:
    """Per-era filter diagnostics from one outer-loop iteration.

    Attributes
    ----------
    era_index : int
        Zero-based era index.
    n_entering : int
        Particles entering this era (before the filter).
    n_passing_filter : int
        Particles passing the survivor-count filter.
    acceptance_rate : float
        ``n_passing_filter / n_entering`` (0.0 if ``n_entering == 0``).
    """

    era_index: int
    n_entering: int
    n_passing_filter: int
    acceptance_rate: float


@dataclass(frozen=True)
class MatchedSurvivalSample:
    """One conditioned trajectory from the multilevel-splitting sampler.

    Attributes
    ----------
    n_born : int
        Total number of individuals ever created (including the root).
    n_alive_present : int
        Individuals alive at the terminal time (death_time > terminal_time).
    proportion_alive : float
        ``n_alive_present / n_born``.  Zero when ``n_born == 0``.
    per_era_ever_created : ndarray, shape [n_epochs]
        Count of individuals born in each era.
    per_era_survivors : ndarray, shape [n_epochs]
        Count of individuals born in each era that survive to terminal time.
    """

    n_born: int
    n_alive_present: int
    proportion_alive: float
    per_era_ever_created: NDArray[np.intp]
    per_era_survivors: NDArray[np.intp]


@dataclass(frozen=True)
class ConditionedSurvivalResult:
    """Output of :func:`sample_conditioned_survival_process`.

    Attributes
    ----------
    samples : list[MatchedSurvivalSample]
        Matched trajectories (length == ``n_samples`` when successful).
    diagnostics : list[EraConditioningDiagnostic]
        Per-era filter diagnostics from the **last** outer-loop iteration.
    n_outer_iterations : int
        Number of outer-loop restarts (≥ 1).
    n_total_era0_particles : int
        Total era-0 particles simulated across all outer-loop iterations.
    """

    samples: list[MatchedSurvivalSample] = field(default_factory=list)
    diagnostics: list[EraConditioningDiagnostic] = field(default_factory=list)
    n_outer_iterations: int = 0
    n_total_era0_particles: int = 0


# ---------------------------------------------------------------------------
# JIT-compiled single-era event loop
# ---------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnums=(6,))
def _simulate_era_core(
    birth_rate_k: jax.Array,  # float64 scalar — birth rate for this era
    death_rates: jax.Array,  # float64 [n_epochs] — FULL hazard grid
    epoch_end_times: jax.Array,  # float64 [n_epochs] — FULL epoch ends
    era_idx: jax.Array,  # int32 scalar — era tag written to birth_epoch_buf
    segment_end: jax.Array,  # float64 scalar — this era's end time
    state: _EraLoopState,  # initial carry (from previous era or seed)
    max_population: int,  # STATIC — buffer capacity; determines JIT variants
) -> _EraLoopState:
    """Advance the birth-death event loop to ``segment_end``.

    Uses a ``jax.lax.while_loop`` to process births and pre-scheduled deaths
    within a single era.  The birth rate is fixed at ``birth_rate_k`` for the
    duration.  New births receive their death time sampled over the **full**
    ``epoch_end_times`` hazard grid via
    :func:`~bdsurv.lifespans._sample_piecewise_exponential_death_time`,
    so survival-to-present is always well-defined regardless of the current
    era.

    Parameters
    ----------
    birth_rate_k : jax.Array
        Per-individual birth rate λ for the current era (float64 scalar).
    death_rates : jax.Array
        Full per-epoch death rates μ, shape ``[n_epochs]``.
    epoch_end_times : jax.Array
        Full epoch end times, shape ``[n_epochs]``; last entry = terminal time.
    era_idx : jax.Array
        Current era index (int32 scalar); written into ``birth_epoch_buf`` for
        every individual born in this era.
    segment_end : jax.Array
        End time of the current era (float64 scalar).
    state : _EraLoopState
        Carry-in from the previous era (or the seeded initial state for era 0).
    max_population : int
        **Static** buffer capacity.  Determines the compiled kernel variant.

    Returns
    -------
    _EraLoopState
        Updated loop state at ``current_time == segment_end``.  ``overflow``
        is set if the buffer would have been exceeded.
    """
    terminal_time: jax.Array = epoch_end_times[-1]

    def _cond(s: _EraLoopState) -> jax.Array:
        return (s.current_time < segment_end) & ~s.overflow

    def _body(s: _EraLoopState) -> _EraLoopState:
        key, k_birth_dt, k_death_time = jax.random.split(s.key, 3)

        # ---- Next birth time (linear-rate exponential inter-arrival) ----
        total_birth_rate = s.n_alive.astype(jnp.float64) * birth_rate_k
        dt_birth = jax.random.exponential(k_birth_dt, dtype=jnp.float64) / jnp.where(
            total_birth_rate > 0.0, total_birth_rate, jnp.float64(1.0)
        )
        next_birth = jnp.where(
            total_birth_rate > 0.0,
            s.current_time + dt_birth,
            jnp.inf,
        )

        # ---- Next pre-scheduled death (smallest future death in buffer) ----
        indices = jnp.arange(max_population)
        future_death_mask = (
            (indices < s.n_born)
            & jnp.isfinite(s.death_time_buf)
            & (s.death_time_buf > s.current_time)
        )
        next_death = jnp.min(jnp.where(future_death_mask, s.death_time_buf, jnp.inf))

        # ---- Choose next event ----
        next_event = jnp.minimum(jnp.minimum(next_birth, next_death), segment_end)

        is_epoch = next_event >= segment_end - jnp.float64(1e-12)
        is_death = (~is_epoch) & (next_death <= next_birth)
        is_birth = (~is_epoch) & (~is_death)

        new_time = jnp.where(next_event >= segment_end, segment_end, next_event)

        # ---- Birth event ----
        can_do_birth = is_birth & (s.n_born < max_population)
        new_overflow = s.overflow | (is_birth & (s.n_born >= max_population))

        new_indiv_death = _sample_piecewise_exponential_death_time(
            birth_time=next_event,
            death_rates=death_rates,
            epoch_end_times=epoch_end_times,
            terminal_time=terminal_time,
            key=k_death_time,
        )

        safe_write_idx = jnp.minimum(s.n_born, max_population - 1)
        new_birth_time_buf = jnp.where(
            can_do_birth,
            s.birth_time_buf.at[safe_write_idx].set(next_event),
            s.birth_time_buf,
        )
        new_death_time_buf = jnp.where(
            can_do_birth,
            s.death_time_buf.at[safe_write_idx].set(new_indiv_death),
            s.death_time_buf,
        )
        new_birth_epoch_buf = jnp.where(
            can_do_birth,
            s.birth_epoch_buf.at[safe_write_idx].set(era_idx),
            s.birth_epoch_buf,
        )
        new_n_born = jnp.where(can_do_birth, s.n_born + 1, s.n_born)

        # ---- Update n_alive ----
        new_n_alive = jnp.where(
            is_birth & can_do_birth,
            s.n_alive + 1,
            jnp.where(is_death, jnp.maximum(s.n_alive - 1, 0), s.n_alive),
        )

        return _EraLoopState(
            key=key,
            current_time=new_time,
            n_alive=new_n_alive,
            n_born=new_n_born,
            overflow=new_overflow,
            birth_time_buf=new_birth_time_buf,
            death_time_buf=new_death_time_buf,
            birth_epoch_buf=new_birth_epoch_buf,
        )

    return lax.while_loop(_cond, _body, state)


# ---------------------------------------------------------------------------
# Host-side helper functions
# ---------------------------------------------------------------------------


def _make_initial_particle(
    death_rates: jax.Array,
    epoch_end_times: jax.Array,
    key: jax.Array,
    max_population: int,
) -> _EraLoopState:
    """Initialise one era-0 particle from a single root born at t=0."""
    terminal_time: jax.Array = epoch_end_times[-1]
    key, k_root = jax.random.split(key)
    root_death = _sample_piecewise_exponential_death_time(
        jnp.float64(0.0), death_rates, epoch_end_times, terminal_time, k_root
    )
    birth_time_buf = (
        jnp.full(max_population, jnp.inf, dtype=jnp.float64).at[0].set(jnp.float64(0.0))
    )
    death_time_buf = (
        jnp.full(max_population, jnp.inf, dtype=jnp.float64).at[0].set(root_death)
    )
    birth_epoch_buf = jnp.zeros(max_population, dtype=jnp.int32)
    return _EraLoopState(
        key=key,
        current_time=jnp.float64(0.0),
        n_alive=jnp.int32(1),
        n_born=jnp.int32(1),
        overflow=jnp.bool_(False),
        birth_time_buf=birth_time_buf,
        death_time_buf=death_time_buf,
        birth_epoch_buf=birth_epoch_buf,
    )


def _count_era_survivors(state: _EraLoopState, era_k: int, terminal_time: float) -> int:
    """Count individuals born in era *k* that survive to *terminal_time*.

    Materialises only the first ``n_born`` entries of the birth-epoch and
    death-time buffers (NumPy conversion is O(n_born)).
    """
    n = int(state.n_born)
    be = np.asarray(state.birth_epoch_buf[:n])
    dt = np.asarray(state.death_time_buf[:n])
    return int(np.sum((be == era_k) & (dt > terminal_time)))


def _clone_particle(state: _EraLoopState, new_key: jax.Array) -> _EraLoopState:
    """Clone a particle by assigning a fresh PRNGKey.

    JAX arrays are immutable, so the buffer references are shared O(1).
    :func:`_simulate_era_core` creates new arrays on every write, so there
    is no aliasing hazard.
    """
    return state._replace(key=new_key)


def _extract_sample(
    state: _EraLoopState, n_epochs: int, terminal_time: float
) -> MatchedSurvivalSample:
    """Extract a :class:`MatchedSurvivalSample` from a completed trajectory."""
    n = int(state.n_born)
    be = np.asarray(state.birth_epoch_buf[:n])
    dt = np.asarray(state.death_time_buf[:n])
    survived = dt > terminal_time

    per_era_ever_created = np.array(
        [int(np.sum(be == k)) for k in range(n_epochs)], dtype=np.intp
    )
    per_era_survivors = np.array(
        [int(np.sum((be == k) & survived)) for k in range(n_epochs)], dtype=np.intp
    )
    n_alive_present = int(np.sum(survived))
    proportion_alive = n_alive_present / n if n > 0 else 0.0

    return MatchedSurvivalSample(
        n_born=n,
        n_alive_present=n_alive_present,
        proportion_alive=proportion_alive,
        per_era_ever_created=per_era_ever_created,
        per_era_survivors=per_era_survivors,
    )


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def build_target_ranges(
    observed_counts: NDArray[np.intp] | list[int],
    *,
    tolerance: int | float = 0,
) -> NDArray[np.intp]:
    """Build inclusive ``[lo, hi]`` survivor-count ranges for each era.

    Parameters
    ----------
    observed_counts : array-like of int, length ``n_eras``
        Observed per-era survivor counts.  The **last era is left
        unconditioned** — its row is set to ``[0, 2^31 - 1]`` regardless of
        ``tolerance``.
    tolerance : int or float, optional
        Widens the acceptance window symmetrically.

        - ``int`` (default 0): absolute ``± tolerance`` counts.  Zero means
          exact matching (``lo = hi = observed``).
        - ``float`` in ``(0, 1)``: relative — ``±round(observed * tolerance)``
          counts.

    Returns
    -------
    ndarray, shape ``[n_eras, 2]``, dtype int
        Inclusive ``[lo, hi]`` bounds per era.

    Examples
    --------
    >>> build_target_ranges([73, 209, 282, 2345, 2384, 540])
    array([[  73,   73],
           [ 209,  209],
           [ 282,  282],
           [2345, 2345],
           [2384, 2384],
           [   0, 2147483647]])

    >>> build_target_ranges([73, 209], tolerance=5)
    array([[ 68,  78],
           [  0, 2147483647]])
    """
    counts = np.asarray(observed_counts, dtype=np.intp)
    n_eras = len(counts)
    ranges = np.zeros((n_eras, 2), dtype=np.intp)

    for k, c in enumerate(counts[:-1]):
        if isinstance(tolerance, float) and 0.0 < tolerance < 1.0:
            delta = max(0, round(int(c) * tolerance))
        else:
            delta = int(tolerance)
        ranges[k, 0] = max(0, int(c) - delta)
        ranges[k, 1] = int(c) + delta

    # Last era always passes
    ranges[-1, 0] = 0
    ranges[-1, 1] = 2**31 - 1

    return ranges


# ---------------------------------------------------------------------------
# Per-process conditioned sampler
# ---------------------------------------------------------------------------


def sample_conditioned_survival_process(
    birth_rates: NDArray[np.float64] | list[float],
    death_rates: NDArray[np.float64] | list[float],
    epoch_end_times: NDArray[np.float64] | list[float],
    target_ranges: NDArray[np.intp],
    *,
    n_samples: int = 1,
    split_factor: int,
    n_initial: int,
    max_attempts: int | None = None,
    random_seed: int | None = None,
    max_population: int = 200_000,
) -> ConditionedSurvivalResult:
    """Sample *n_samples* matched trajectories for fixed rates via multilevel splitting.

    The algorithm conditions forward simulations on observed per-era survivor
    counts by filtering and cloning particles at each era boundary.  The last
    era is run unconditioned so as not to over-constrain the sampler.

    Parameters
    ----------
    birth_rates : array-like, shape ``[n_epochs]``
        Per-individual birth rates λ, one per epoch.
    death_rates : array-like, shape ``[n_epochs]``
        Per-individual death rates μ, one per epoch.
    epoch_end_times : array-like, shape ``[n_epochs]``
        Strictly increasing era end times; the last entry is the terminal time
        *T* (e.g. ``1900.0`` for a process starting at 0 CE lasting to 1900 years
        from the start).
    target_ranges : ndarray, shape ``[n_epochs, 2]``
        Inclusive ``[lo, hi]`` survivor-count bounds per era.  The last row is
        ignored (last era is unconditioned).  Typically produced by
        :func:`build_target_ranges`.
    n_samples : int, optional
        Number of matched trajectories to collect.  Default 1.
    split_factor : int
        Number of clones made from each passing particle after each era filter.
    n_initial : int
        Number of independent era-0 particles simulated per outer-loop restart.
    max_attempts : int, optional
        Maximum total number of era-0 particles ever simulated (across all
        restarts).  Defaults to ``n_initial * 1000``.
    random_seed : int, optional
        Integer seed for reproducibility.
    max_population : int, optional
        Buffer capacity (individuals per particle).  Triggers a JIT recompile
        if a new value is first used.  Default 200 000.

    Returns
    -------
    ConditionedSurvivalResult
        Contains the matched samples, per-era filter diagnostics, and counts
        of restarts and total era-0 particles simulated.

    Warns
    -----
    RuntimeWarning
        If ``max_attempts`` is reached before ``n_samples`` are collected.
    """
    br = np.asarray(birth_rates, dtype=np.float64)
    dr = np.asarray(death_rates, dtype=np.float64)
    et = np.asarray(epoch_end_times, dtype=np.float64)

    n_epochs: int = len(br)
    if not (len(dr) == len(et) == n_epochs):
        msg = "birth_rates, death_rates, and epoch_end_times must have the same length"
        raise ValueError(msg)
    if n_epochs < 2:
        msg = "At least 2 epochs are required for conditioned sampling"
        raise ValueError(msg)

    br_jax = jnp.array(br, dtype=jnp.float64)
    dr_jax = jnp.array(dr, dtype=jnp.float64)
    et_jax = jnp.array(et, dtype=jnp.float64)

    terminal_time = float(et[-1])
    n_conditioned = n_epochs - 1  # eras 0 … n_epochs-2 are filtered

    if max_attempts is None:
        max_attempts = n_initial * 1000

    rng = np.random.default_rng(random_seed)

    samples_collected: list[MatchedSurvivalSample] = []
    last_diagnostics: list[EraConditioningDiagnostic] = []
    n_outer = 0
    total_era0 = 0

    while len(samples_collected) < n_samples:
        if total_era0 >= max_attempts:
            warnings.warn(
                f"sample_conditioned_survival_process: reached max_attempts="
                f"{max_attempts} with only {len(samples_collected)}/{n_samples} "
                "samples collected.  Returning partial results.",
                RuntimeWarning,
                stacklevel=2,
            )
            break

        n_outer += 1
        diagnostics: list[EraConditioningDiagnostic] = []

        # ---- Era-0: seed n_initial independent particles ----
        init_keys = jax.random.split(
            jax.random.key(int(rng.integers(0, 2**31))), n_initial
        )
        particles: list[_EraLoopState] = [
            _make_initial_particle(dr_jax, et_jax, k, max_population) for k in init_keys
        ]
        total_era0 += n_initial

        # ---- Conditioned eras 0 … n_conditioned-1 ----
        restart = False
        for era_k in range(n_conditioned):
            segment_end = et_jax[era_k]
            br_k = br_jax[era_k]
            era_idx_jax = jnp.int32(era_k)

            # Simulate era_k for every particle
            simulated: list[_EraLoopState] = [
                _simulate_era_core(
                    br_k, dr_jax, et_jax, era_idx_jax, segment_end, p, max_population
                )
                for p in particles
            ]

            # Filter: keep particles within target_ranges[era_k]
            lo = int(target_ranges[era_k, 0])
            hi = int(target_ranges[era_k, 1])
            passing: list[_EraLoopState] = [
                p
                for p in simulated
                if not bool(p.overflow)
                and lo <= _count_era_survivors(p, era_k, terminal_time) <= hi
            ]

            diag = EraConditioningDiagnostic(
                era_index=era_k,
                n_entering=len(simulated),
                n_passing_filter=len(passing),
                acceptance_rate=(len(passing) / len(simulated) if simulated else 0.0),
            )
            diagnostics.append(diag)

            if not passing:
                restart = True
                break

            # Split: clone each passing particle split_factor times
            split_keys = jax.random.split(
                jax.random.key(int(rng.integers(0, 2**31))),
                len(passing) * split_factor,
            )
            particles = [
                _clone_particle(p, split_keys[i * split_factor + j])
                for i, p in enumerate(passing)
                for j in range(split_factor)
            ]

        if restart:
            last_diagnostics = diagnostics
            continue

        # ---- Final (unconditioned) era ----
        final_era = n_epochs - 1
        segment_end_final = et_jax[final_era]
        br_final = br_jax[final_era]
        era_idx_final = jnp.int32(final_era)

        for p in particles:
            final_state = _simulate_era_core(
                br_final,
                dr_jax,
                et_jax,
                era_idx_final,
                segment_end_final,
                p,
                max_population,
            )
            if bool(final_state.overflow):
                continue
            sample = _extract_sample(final_state, n_epochs, terminal_time)
            samples_collected.append(sample)
            if len(samples_collected) >= n_samples:
                break

        last_diagnostics = diagnostics

    return ConditionedSurvivalResult(
        samples=samples_collected,
        diagnostics=last_diagnostics,
        n_outer_iterations=n_outer,
        n_total_era0_particles=total_era0,
    )


# ---------------------------------------------------------------------------
# Posterior-driven wrapper
# ---------------------------------------------------------------------------


def sample_conditioned_survival_per_posterior(
    posterior_birth_rates: NDArray[np.float64],
    posterior_death_rates: NDArray[np.float64],
    epoch_end_times: NDArray[np.float64] | list[float],
    target_ranges: NDArray[np.intp],
    *,
    split_factor: int,
    n_initial: int,
    max_attempts: int | None = None,
    random_seed: int | None = None,
    max_population: int = 200_000,
) -> pd.DataFrame:
    """Run the conditioned sampler once per posterior draw.

    For each row in ``posterior_birth_rates`` / ``posterior_death_rates``,
    calls :func:`sample_conditioned_survival_process` with ``n_samples=1``.
    Draws that exhaust ``max_attempts`` are flagged (``matched=False``) and
    skipped rather than raising, so the DataFrame always has one row per draw.

    Parameters
    ----------
    posterior_birth_rates : ndarray, shape ``[n_draws, n_epochs]``
        Posterior birth-rate samples (flat over MCMC chains).
    posterior_death_rates : ndarray, shape ``[n_draws, n_epochs]``
        Posterior death-rate samples (flat over MCMC chains).
    epoch_end_times : array-like, shape ``[n_epochs]``
        Era end times (process time); last entry is the terminal time.
    target_ranges : ndarray, shape ``[n_epochs, 2]``
        Inclusive per-era bounds; see :func:`build_target_ranges`.
    split_factor : int
        Clone factor applied after each era filter.
    n_initial : int
        Era-0 particles per outer-loop restart.
    max_attempts : int, optional
        Maximum era-0 particles per posterior draw.  Defaults to
        ``n_initial * 1000``.
    random_seed : int, optional
        Master seed for reproducibility.
    max_population : int, optional
        Buffer capacity per particle.  Default 200 000.

    Returns
    -------
    pd.DataFrame
        One row per posterior draw.  Columns:

        ``draw``
            Zero-based draw index.
        ``matched``
            ``True`` when a trajectory was successfully sampled.
        ``n_born``, ``n_alive_present``, ``proportion_alive``
            Trajectory-level statistics (``NaN`` when ``matched=False``).
        ``era_{k}_ever_created``, ``era_{k}_survivors``
            Per-era counts (``k`` in ``0 … n_epochs-1``; ``NaN`` when
            ``matched=False``).
        ``n_outer_iterations``
            Restarts needed for this draw (0 when ``matched=False``).
    """
    br_arr = np.asarray(posterior_birth_rates, dtype=np.float64)
    dr_arr = np.asarray(posterior_death_rates, dtype=np.float64)

    if br_arr.ndim != 2 or dr_arr.ndim != 2:
        msg = "posterior_birth_rates and posterior_death_rates must be 2-D arrays"
        raise ValueError(msg)
    if br_arr.shape != dr_arr.shape:
        msg = (
            f"posterior_birth_rates shape {br_arr.shape} != "
            f"posterior_death_rates shape {dr_arr.shape}"
        )
        raise ValueError(msg)

    n_draws, n_epochs = br_arr.shape
    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, object]] = []

    for i in range(n_draws):
        seed_i = int(rng.integers(0, 2**31))
        try:
            result = sample_conditioned_survival_process(
                br_arr[i],
                dr_arr[i],
                epoch_end_times,
                target_ranges,
                n_samples=1,
                split_factor=split_factor,
                n_initial=n_initial,
                max_attempts=max_attempts,
                random_seed=seed_i,
                max_population=max_population,
            )
        except Exception as exc:
            warnings.warn(
                f"sample_conditioned_survival_per_posterior: draw {i} raised "
                f"{type(exc).__name__}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            result = ConditionedSurvivalResult()

        if result.samples:
            s = result.samples[0]
            row: dict[str, object] = {
                "draw": i,
                "matched": True,
                "n_born": s.n_born,
                "n_alive_present": s.n_alive_present,
                "proportion_alive": s.proportion_alive,
                "n_outer_iterations": result.n_outer_iterations,
            }
            for k in range(n_epochs):
                row[f"era_{k}_ever_created"] = int(s.per_era_ever_created[k])
                row[f"era_{k}_survivors"] = int(s.per_era_survivors[k])
        else:
            row = {
                "draw": i,
                "matched": False,
                "n_born": float("nan"),
                "n_alive_present": float("nan"),
                "proportion_alive": float("nan"),
                "n_outer_iterations": 0,
            }
            for k in range(n_epochs):
                row[f"era_{k}_ever_created"] = float("nan")
                row[f"era_{k}_survivors"] = float("nan")

        rows.append(row)

    return pd.DataFrame(rows)
