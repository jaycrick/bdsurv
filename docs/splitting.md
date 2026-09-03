# Multilevel splitting (RESTART) simulation

## Overview

This document describes the multilevel-splitting simulator in
`src/bdsurv/splitting.py`, which estimates `P(survival to T)` for a strictly
subcritical process (`birth_rate < death_rate`) and returns a batch of
conditioned trajectories.

Unlike the contour-process method, this is an **approximate**, simulation-based
estimator: it evolves an ensemble of particles under exact Gillespie dynamics
and periodically duplicates ("enriches") survivors to counteract the
population's tendency to die out, rather than constructing a tree that is
guaranteed to survive by design.

## Why splitting is needed

In the subcritical regime the unconditioned process goes extinct with
probability 1, and `P(survival to T)` can be extremely small — naive Monte
Carlo (simulate many independent runs, keep only the survivors) wastes almost
all its samples on trajectories that die out early. Splitting concentrates
computational effort on the surviving branch of the ensemble instead.

## Algorithm: fixed-factor enrichment (RESTART)

1. **Level grid.** Choose boundary times `0 = t_0 < t_1 < ... < t_K = T`. By
   default (`select_level_times_for_band`) these are equally spaced, chosen so
   the single-particle extinction probability over the first increment,
   `pr_ext(t_1 - t_0)`, falls in the band `[e^{-2}, 0.5] ≈ [0.135, 0.5]`. This
   keeps the splitting factor `R = ceil(1 / pr_ext)` in `[2, 8]` — enough
   enrichment to counteract die-off without an unbounded ensemble.
2. **Initialise.** Start with `N_0 = min(R, n_particles)` independent
   particles, each a single individual at time 0.
3. **Each level `k = 1, ..., K`:**
   - **Evolve.** Every particle advances from `t_{k-1}` to `t_k` under exact
     Gillespie dynamics (`_simulate_to_level`).
   - **Estimate.** The empirical conditional survival fraction is
     `p_k = S_k / N_{k-1}`, where `S_k` is the number of particles with at
     least one live individual at `t_k`, and `N_{k-1}` is the number that
     *entered* the level (not the cap).
   - **Enrich.** Each surviving particle is cloned `R` times. If
     `R * S_k > n_particles`, the enlarged ensemble is subsampled uniformly
     with replacement back down to `n_particles`; this bounds cost while
     keeping the estimator unbiased.
4. **Combine.** The overall survival estimate is the product of the
   per-level conditional fractions, `P(survival to T) ~= p_1 * p_2 * ... * p_K`.

If every particle goes extinct at some level, the loop stops early: the
estimate is 0 and `level_times` is truncated to the levels actually reached.

## Level-grid selection in detail

`select_level_times_for_band` enumerates equal-split counts `n = 1, 2, ...`.
Each gives an increment `tau = T / n` with single-particle extinction
probability `pr = pr_ext(tau)` — monotonically decreasing in `n` — and
splitting factor `R = ceil(1 / pr)`. Among the splits whose `pr` falls in the
band, it picks the `n` minimising `|(R * pr)^n - 1|`: the grid whose
*cumulative* enrichment across all `n` levels most nearly offsets the
cumulative die-off, not merely the coarsest grid that satisfies the band
constraint on its own.

A caller may instead supply an explicit `level_times` grid (validated against
the band on its first increment) and/or an explicit `splitting_factor`
override; supplying both together bypasses the band check entirely, letting a
predetermined out-of-band grid be used deliberately.

## Conditioned trajectories and genealogy

Every particle carries its own individual-level event records
(`_ParticleState.records`) and a population trace (`trace_times`,
`trace_populations`). At the end of the run, up to
`config.n_conditioned_trajectories` surviving particles are converted to
`SimulatedTree` objects (`_particle_to_tree`) — the spine is the individual
at index 0 of `alive_ids`, matching the head-of-alive-list convention used
elsewhere. Parent-child lineage across enrichment steps is recorded in
`particle_genealogy` (one row per clone, per level).

## Reproducibility and RNG

A single master `np.random.default_rng(config.random_seed)` seeds every
particle's own child `Generator` (`_make_child_rng`), including clones
created during enrichment, so a run is fully reproducible from
`config.random_seed` alone.

## Usage

```python
from bdsurv import SplittingConfig, simulate_subcritical_survival_splitting

config = SplittingConfig(
    birth_rate=0.9, death_rate=1.0, terminal_time=100.0, n_particles=1000
)
result = simulate_subcritical_survival_splitting(config)
print(result.survival_probability_estimate)
```

See the function docstring in `src/bdsurv/splitting.py` for the full
parameter and return-value reference.
