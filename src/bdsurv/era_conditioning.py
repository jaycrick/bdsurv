"""Conditioned survival sampling for multi-epoch birth-death processes.

Estimates the proportion of ever-created individuals currently alive
(surviving to the process terminal time) by conditioning forward simulations
on observed per-era survivor counts via a multilevel-splitting / sequential-ABC
scheme.

Algorithm
---------
For eras 0 ... K-2 (the last era is left unconditioned):

1. Simulate ``n_initial`` independent era-0 particles, each a single root
   individual born at time 0.
2. **Filter**: keep particles whose count of *born-in-era-0 individuals that
   survive to the terminal time* lies within ``target_ranges[0]``.
3. **Split**: clone each passing particle ``split_factor`` times (each clone
   receives a fresh PRNG key; JAX arrays are immutable so clones are O(1)).
4. Simulate era 1 for each clone, filter on ``target_ranges[1]``, split.
5. Repeat for eras 2 ... K-2.
6. Simulate the final era (K-1) **unconditioned**.
7. Collect the resulting trajectories.  If fewer than ``n_samples`` are
   collected, or no particle passed a filter, restart from step 1 (bounded by
   ``max_attempts`` era-0 particles in total).

Per-era match criterion
-----------------------
``#{i : birth_epoch[i] == k  and  death_time[i] > terminal_time}``

i.e. individuals born in era *k* whose pre-sampled death time exceeds the
terminal time (they survive to the present day).  This is computed from the
flat lifespan buffers the moment era *k* ends, so no additional simulation
is required.

Engine
------
The particles are advanced by the one JAX event kernel of the package,
:func:`bdsurv.lifespans.advance_era`, starting from
:func:`bdsurv.lifespans.initial_state`.  The kernel advances an arbitrary
alive-population state to the era boundary and samples every new birth's death
time over the **full** piecewise hazard, so survival-to-present is always
well-defined.

Public API
----------
- :func:`build_target_ranges`: construct per-era ``[lo, hi]`` bounds from
  observed counts (optionally widened by a tolerance).
- :func:`sample_conditioned_survival_process`: draw ``n_samples`` matched
  trajectories for a fixed set of rates.
- :func:`sample_conditioned_survival_per_posterior`: one matched trajectory
  per posterior draw; returns a tidy ``pd.DataFrame``.
- :class:`MatchedSurvivalSample`: per-trajectory result dataclass.
- :class:`ConditionedSurvivalResult`: collection of samples plus diagnostics.
"""

# JAX's jit/while_loop wrappers are untyped, which strict mode reports as
# unknown.
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from bdsurv._seeding import child_seed, jax_key, root_rng
from bdsurv.lifespans import LoopState, advance_era, initial_state
from bdsurv.rates import (
    check_epoch_ends,
    check_positive_rates,
    epoch_rates,
)

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
# Host-side helper functions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _EraGrid:
    """The rates and sizes shared by every particle and era of one sampler call."""

    birth_rates: jax.Array  # float64 [n_epochs]
    death_rates: jax.Array  # float64 [n_epochs]
    epoch_end_times: jax.Array  # float64 [n_epochs]
    max_population: int

    @property
    def n_epochs(self) -> int:
        return int(self.epoch_end_times.shape[0])

    @property
    def terminal_time(self) -> float:
        return float(self.epoch_end_times[-1])

    def advance(self, era_k: int, state: LoopState) -> LoopState:
        """Advance ``state`` through era ``era_k``."""
        return advance_era(
            self.birth_rates[era_k],
            self.death_rates,
            self.epoch_end_times,
            jnp.int32(era_k),
            self.epoch_end_times[era_k],
            state,
            self.max_population,
        )


def _count_era_survivors(state: LoopState, era_k: int, terminal_time: float) -> int:
    """Count individuals born in era *k* that survive to *terminal_time*.

    Materialises only the first ``n_born`` entries of the birth-epoch and
    death-time buffers (NumPy conversion is O(n_born)).
    """
    n = int(state.n_born)
    be = np.asarray(state.birth_epoch_buf[:n])
    dt = np.asarray(state.death_time_buf[:n])
    return int(np.sum((be == era_k) & (dt > terminal_time)))


def _clone_particle(state: LoopState, new_key: jax.Array) -> LoopState:
    """Clone a particle by assigning a fresh PRNGKey.

    JAX arrays are immutable, so the buffer references are shared O(1).
    :func:`~bdsurv.lifespans.advance_era` creates new arrays on every write, so there
    is no aliasing hazard.
    """
    return state._replace(key=new_key)


def _extract_sample(
    state: LoopState, n_epochs: int, terminal_time: float
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

    Raises
    ------
    ValueError
        If ``observed_counts`` is empty or has a negative entry, or
        ``tolerance`` is negative.

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
    if n_eras < 1:
        msg = "observed_counts must have at least one entry"
        raise ValueError(msg)
    if np.any(counts < 0):
        msg = "observed_counts must be non-negative"
        raise ValueError(msg)
    if not tolerance >= 0:
        msg = "tolerance must be non-negative"
        raise ValueError(msg)
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
    ranges[-1, 1] = int(np.iinfo(np.int32).max)

    return ranges


def _run_conditioned_eras(
    grid: _EraGrid,
    target_ranges: NDArray[np.intp],
    rng: np.random.Generator,
    *,
    n_initial: int,
    split_factor: int,
) -> tuple[list[LoopState] | None, list[EraConditioningDiagnostic]]:
    """Run eras ``0 ... K-2`` for one outer iteration.

    Seeds ``n_initial`` root particles, then for each conditioned era simulates
    every particle through it, keeps those that did not overflow and whose era
    survivor count lies in the era's target range, records the diagnostic, and
    clones each passing particle ``split_factor`` times with fresh keys.

    Returns
    -------
    particles, diagnostics
        The particles entering the final era, or ``None`` if no particle passed
        some filter (a restart is needed); and the diagnostics recorded so far.
    """
    init_keys = jax.random.split(jax_key(rng), n_initial)
    particles: list[LoopState] = [
        initial_state(grid.death_rates, grid.epoch_end_times, k, grid.max_population)
        for k in init_keys
    ]
    diagnostics: list[EraConditioningDiagnostic] = []

    for era_k in range(grid.n_epochs - 1):
        simulated = [grid.advance(era_k, p) for p in particles]

        lo = int(target_ranges[era_k, 0])
        hi = int(target_ranges[era_k, 1])
        passing = [
            p
            for p in simulated
            if not bool(p.overflow)
            and lo <= _count_era_survivors(p, era_k, grid.terminal_time) <= hi
        ]
        diagnostics.append(
            EraConditioningDiagnostic(
                era_index=era_k,
                n_entering=len(simulated),
                n_passing_filter=len(passing),
                acceptance_rate=(len(passing) / len(simulated) if simulated else 0.0),
            )
        )
        if not passing:
            return None, diagnostics

        split_keys = jax.random.split(jax_key(rng), len(passing) * split_factor)
        particles = [
            _clone_particle(p, split_keys[i * split_factor + j])
            for i, p in enumerate(passing)
            for j in range(split_factor)
        ]
    return particles, diagnostics


def _run_final_era(
    grid: _EraGrid, particles: list[LoopState], n_wanted: int
) -> list[MatchedSurvivalSample]:
    """Run the last, unconditioned era and collect up to ``n_wanted`` samples.

    Particles that overflow are dropped.  Stops as soon as ``n_wanted`` samples
    are collected.
    """
    samples: list[MatchedSurvivalSample] = []
    for p in particles:
        final_state = grid.advance(grid.n_epochs - 1, p)
        if bool(final_state.overflow):
            continue
        samples.append(_extract_sample(final_state, grid.n_epochs, grid.terminal_time))
        if len(samples) >= n_wanted:
            break
    return samples


# ---------------------------------------------------------------------------
# Per-process conditioned sampler
# ---------------------------------------------------------------------------


def _check_sampler_arguments(
    target_ranges: NDArray[np.intp],
    n_epochs: int,
    *,
    split_factor: int,
    n_initial: int,
) -> None:
    """Check the arguments of the conditioned samplers that do not depend on rates."""
    for name, value in (("split_factor", split_factor), ("n_initial", n_initial)):
        if value < 1:
            msg = f"{name} must be a positive integer"
            raise ValueError(msg)
    if np.shape(target_ranges) != (n_epochs, 2):
        msg = (
            f"target_ranges must have shape (n_epochs, 2) = ({n_epochs}, 2), "
            f"got {np.shape(target_ranges)}"
        )
        raise ValueError(msg)


def sample_conditioned_survival_process(  # noqa: PLR0913  # public signature
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

    Raises
    ------
    ValueError
        If the three rate and time arguments differ in length, a rate is not
        finite and positive, the end times are not finite, positive and strictly
        increasing, there are fewer than 2 epochs, ``target_ranges`` does not
        have shape ``(n_epochs, 2)``, or ``n_samples``, ``split_factor`` or
        ``n_initial`` is less than 1.

    Warns
    -----
    RuntimeWarning
        If ``max_attempts`` is reached before ``n_samples`` are collected.
    """
    br, dr, et = epoch_rates(
        birth_rates, death_rates, epoch_end_times, times_name="epoch_end_times"
    )

    n_epochs: int = len(br)
    if n_epochs < 2:
        msg = "At least 2 epochs are required for conditioned sampling"
        raise ValueError(msg)
    _check_sampler_arguments(
        target_ranges, n_epochs, split_factor=split_factor, n_initial=n_initial
    )
    if n_samples < 1:
        msg = "n_samples must be a positive integer"
        raise ValueError(msg)

    grid = _EraGrid(
        birth_rates=jnp.array(br, dtype=jnp.float64),
        death_rates=jnp.array(dr, dtype=jnp.float64),
        epoch_end_times=jnp.array(et, dtype=jnp.float64),
        max_population=max_population,
    )
    if max_attempts is None:
        max_attempts = n_initial * 1000

    rng = root_rng(random_seed, "seeding")

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
        total_era0 += n_initial
        particles, last_diagnostics = _run_conditioned_eras(
            grid, target_ranges, rng, n_initial=n_initial, split_factor=split_factor
        )
        if particles is None:
            continue
        samples_collected += _run_final_era(
            grid, particles, n_samples - len(samples_collected)
        )

    return ConditionedSurvivalResult(
        samples=samples_collected,
        diagnostics=last_diagnostics,
        n_outer_iterations=n_outer,
        n_total_era0_particles=total_era0,
    )


# ---------------------------------------------------------------------------
# Posterior-driven wrapper
# ---------------------------------------------------------------------------


def sample_conditioned_survival_per_posterior(  # noqa: PLR0913  # public signature
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
    A draw that exhausts ``max_attempts`` is not an error: the sampler warns
    (``RuntimeWarning``) and returns no sample, and the draw is flagged
    ``matched=False``, so the DataFrame always has one row per draw.  Any other
    exception propagates.

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

    Raises
    ------
    ValueError
        If the posterior arrays are not 2-D arrays of the same shape, a
        posterior rate is not finite and positive, ``epoch_end_times`` is
        invalid or not of length ``n_epochs``, ``target_ranges`` does not have
        shape ``(n_epochs, 2)``, ``split_factor`` or ``n_initial`` is less
        than 1, or there are fewer than 2 epochs.  All but the last are checked
        once, before any draw.
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
    check_positive_rates(
        br_arr, dr_arr, label="posterior_birth_rates and posterior_death_rates"
    )
    check_epoch_ends("epoch_end_times", np.asarray(epoch_end_times, dtype=np.float64))
    if len(epoch_end_times) != n_epochs:
        msg = (
            f"epoch_end_times must have the same length as the {n_epochs} "
            "epochs of the posterior draws"
        )
        raise ValueError(msg)
    _check_sampler_arguments(
        target_ranges, n_epochs, split_factor=split_factor, n_initial=n_initial
    )
    rng = root_rng(random_seed, "seeding")
    rows: list[dict[str, object]] = []

    for i in range(n_draws):
        result = sample_conditioned_survival_process(
            br_arr[i],
            dr_arr[i],
            epoch_end_times,
            target_ranges,
            n_samples=1,
            split_factor=split_factor,
            n_initial=n_initial,
            max_attempts=max_attempts,
            random_seed=child_seed(rng),
            max_population=max_population,
        )

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
