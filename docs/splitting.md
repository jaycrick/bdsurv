# Multilevel splitting (RESTART) simulation

## Overview

This document describes the multilevel-splitting simulator in
`src/bdsurv/splitting.py`, which samples trajectories of a birth-death process
(finite, non-negative rates, floats or functions of time) that survive to time `T`. It keeps the
particles alive at `T` and does not estimate the survival probability.

Unlike the contour-process method, this is an **approximate**, simulation-based
sampler: it evolves an ensemble of particles under exact Gillespie dynamics
and periodically duplicates ("enriches") survivors to counteract the
population's tendency to die out, rather than constructing a tree that is
guaranteed to survive by design.

## Why splitting is needed

In the subcritical regime the unconditioned process goes extinct with
probability 1, and `P(survival to T)` can be extremely small — naive Monte
Carlo (simulate many independent runs, keep only the survivors) wastes almost
all its samples on trajectories that die out early. Splitting concentrates
computational effort on the surviving branch of the ensemble instead.

The supercritical regime (`birth_rate > death_rate`) also needs it, since the
process still goes extinct with probability `death_rate / birth_rate`. There
`Pr(t)` levels off at `1 - death_rate / birth_rate`, so the grid collapses
towards a single level and the first replicate count is small (for
`birth_rate = 1`, `death_rate = 0.8` it is about 5, and a run starts with only a
handful of particles).

## Algorithm: multilevel splitting (RESTART)

Write `Pr(t) = P(Z_t > 0)` for a single individual started at time 0, and
`r_i = Pr(t_i) / Pr(t_{i-1})` for the survival ratio of interval `i`.

1. **Level grid.** Choose boundary times `0 = t_0 < t_1 < ... < t_K = T`
   (`select_level_times_for_band`), adjusted so every `r_i` lies in
   `[e^{-2}, 0.5]`.
2. **Replicates.** `compute_level_replicates` gives each interval a count `R_i`.
3. **Initialise.** Start `min(R_1, n_particles)` independent particles, each a
   single individual at time 0.
4. **Each level `k = 1, ..., K`:**
   - **Evolve.** Every particle advances from `t_{k-1}` to `t_k` under exact
     Gillespie dynamics (`_advance_particle`).
   - **Cull.** Particles with no live individual at `t_k` are dropped.
   - **Enrich.** Each survivor is cloned `R_{k+1}` times (none after the last
     level). If that exceeds `n_particles`, the ensemble is subsampled uniformly
     with replacement back down to `n_particles`.
5. **Keep.** The particles alive at `T` are the output. Clones of one survivor
   share history up to the level where they were split, so the kept
   trajectories are not independent.

If every particle goes extinct at some level the attempt fails, and is
restarted (see below).

## Level-grid selection

The starting grid is the caller's `level_times`, or `linspace(0, T, K + 1)`. With
`n_levels=None` (the default) `K` is the integer for which `Pr(T)^(1/K)` is
closest to 1/3 (at least 1), so each starting interval keeps about a third of
the lines; for example `Pr(T) = 1e-9` gives `K = 19`. The grid is then adjusted
as follows:

- **Refine.** Each interval with `r_i < e^{-2}` gets its midpoint added and the
  ratios are recomputed, until none is below the band. A new interval that is
  then above `0.5` is kept.
- **Coarsen.** Sweep left to right. For each interval with `r_i > 0.5`, try
  removing its right end point `t_i`; the merged interval `(t_{i-1}, t_{i+1})`
  has ratio `Pr(t_{i+1}) / Pr(t_{i-1})`, and the removal is kept if that is at
  least `e^{-2}`. After a removal the new interval at the same position is
  examined next. Sweeps repeat until a sweep removes nothing.

No time is protected in either step, so a point added while refining can be
removed when coarsening. Only `t_0` and `T` stay, so a last interval whose ratio
exceeds `0.5` is left as it is.

## Replicates

`R_i` is the smallest integer `>= 2` with `R_1 * R_2 * ... * R_i * Pr(t_i) >= 1`,
so the expected number of live lines at every level is at least one. For
example `Pr(t_1) = 1/2` gives `R_1 = 2`, and `R_2` is the smallest integer
`>= 2` with `R_1 * R_2 * Pr(t_2) >= 1`.

## Restarts

An attempt fails when no particle survives to `T`. It is then restarted from
time 0, up to `max_restarts` attempts in total. By default
`default_max_restarts` chooses the smallest `n` with `1 - p_fail^n >= 0.999`,
where `p_fail = F_1` comes from the backward recursion
`F_{K+1} = 0`, `F_i = (1 - r_i + r_i F_{i+1})^{R_i}` (each line survives interval
`i` independently with probability `r_i`; the cap is ignored). Clones of one
parent share its starting state, so their survival is positively correlated and
the true failure probability can be higher (or sometimes lower) than `p_fail`.
Pass `max_restarts` in `SplittingConfig` to override it. If every attempt fails
a `RuntimeWarning` is issued and no trajectories are returned.

## The easy case

If `Pr(T) > 0.5` there is nothing to split. This always holds when
`death_rate / birth_rate < 0.5`, because `Pr(T) >= 1 - death_rate / birth_rate`,
and also for short horizons at other rates. The simulator then uses the grid
`[0, T]` with one replicate and simulates trees one at a time until one
survives. With `max_restarts=None` the number of attempts is unbounded (at most
two are needed on average); an explicit `max_restarts` still caps it.

The first successful attempt is returned; failed attempts are discarded.
Diagnostics, trajectories and genealogy come from the last attempt.

## Overrides

Supplying `level_times` alone makes it the starting grid. Supplying
`level_times` together with `replicates` (one count per interval) uses both
exactly as given, with no band adjustment.

## Conditioned trajectories and genealogy

Every particle carries its own individual-level event records
(`_ParticleState.forward`) and a population trace (`trace_times`,
`trace_populations`). At the end of the run, up to
`config.n_conditioned_trajectories` surviving particles are converted to
`SimulatedTree` objects (`_particle_to_tree`) — the spine is the individual
at index 0 of `alive_ids`, matching the head-of-alive-list convention used
elsewhere. Parent-child lineage across enrichment steps is recorded in
`particle_genealogy` (one row per clone, per level).

## Linear (time-varying) rates

`simulate_survival_splitting(SplittingConfig(...))` runs the same algorithm for
piecewise-constant or piecewise-linear rate objects, as well as for floats.
Two ingredients change; the levels, replicates, enrichment and restarts are
shared with the constant-rate code.

- **Survival probability.** For one individual at time 0,
  $\Pr(Z_t > 0) = 1 / (1 + I(t))$ with
  $I(t) = \int_0^t \mu(s) e^{-\Gamma(s)}\,ds$ and
  $\Gamma(s) = \int_0^s (\lambda - \mu)$ (`bdpinf`'s $\mathcal W$
  formulation).
  $I$ is computed in log space by Gauss-Legendre quadrature on panels short
  enough that the exponent changes by at most a few units, so cases with
  $1/\Pr$ of order $10^{300}$ stay accurate.
  It drives the level grid, `R_i` and the restart cap; for constant rates the
  closed form is used instead.
- **Dynamics.** Within a segment where both rates are linear, the next event
  among $n$ individuals solves $a s + b s^2/2 = E$ with
  $a = n(\lambda + \mu)$, $b = n(\lambda' + \mu')$.
  An event beyond the segment end is discarded and redrawn from the boundary
  (exact, by independent increments).
  A birth has probability $\lambda(t) / (\lambda(t) + \mu(t))$.

## Reproducibility and RNG

A single master `np.random.default_rng(config.random_seed)` seeds every
particle's own child `Generator` (`_seeding.spawn`), including clones
created during enrichment, so a run is fully reproducible from
`config.random_seed` alone.

## Usage

```python
from bdsurv import SplittingConfig, simulate_survival_splitting

config = SplittingConfig(
    birth_rate=0.9, death_rate=1.0, terminal_time=100.0, n_particles=1000
)
result = simulate_survival_splitting(config)
print(len(result.conditioned_trajectories))
```

See the function docstring in `src/bdsurv/splitting.py` for the full
parameter and return-value reference.
