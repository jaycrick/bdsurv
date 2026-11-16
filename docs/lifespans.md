# Lifespan-at-birth simulation

## Overview

`src/bdsurv/lifespans.py` simulates a linear birth-death process whose rates are
piecewise constant over epochs.
It returns the same records as `simulate_birth_death` with
`PiecewiseConstant` rates (`src/bdsurv/forward.py`), but the numeric core runs in
JIT-compiled JAX: one `jax.lax.while_loop` per epoch, driven by a host loop over
the epochs (`advance_era`).

The process is not conditioned on survival.
For survival-conditioned subcritical trees use the contour-process or splitting
simulators.

## Algorithm

Each individual's whole lifespan is decided the moment it is born.

1. **Death time at birth.** For an individual born at time $b$, draw
   $E \sim \mathrm{Exp}(1)$ and find the first time the cumulative death hazard
   $\int_b^t \mu(s)\,ds$ reaches $E$.
   The hazard is piecewise constant, so this is a scan over the epoch
   boundaries (`sample_piecewise_exponential_death_time`).
   If the hazard up to the terminal time $T$ never reaches $E$, the death time
   is $\infty$ and the individual survives.
2. **Next birth.** With $n$ individuals alive in an epoch with birth rate
   $\lambda$, the next birth is at $t + \mathrm{Exp}(1) / (n\lambda)$.
3. **Next event.** The loop takes the earliest of the next birth, the earliest
   pending death in the buffer, and the epoch end.
   Crossing an epoch end only advances the epoch index.
   Redrawing the birth waiting time at the boundary is exact because the
   exponential is memoryless.
4. **Buffers.** Birth times, death times, birth epochs, insertion uniforms and
   the population just before each birth live in fixed-size arrays of length
   `max_population`.
   The loop sets an overflow flag instead of writing past the end, and the
   Python wrapper raises `RuntimeError`.

## Genealogy

The JIT core stores, for each individual, a uniform $u$ and the population
size $n$ just before its birth.
`_reconstruct_genealogy` replays the events in time order on the host,
maintaining the ordered list of living individuals.
A newborn is inserted at position $1 + \lfloor u n \rfloor$ (clamped to $n$)
and its parent is the individual in the preceding position,
the same convention as `simulate_birth_death`.
The spine is the first living individual at the terminal time.

## Entry points

- `simulate_birth_death_lifespans_arrays` returns a `LifespanSimulationArrays`
  tuple (`birth_times`, `death_times`, `birth_epochs`, `survived`, `n_born`)
  and skips the genealogy replay.
  Use it when only per-epoch counts are needed, as in era-conditioned sampling.
- `simulate_birth_death_lifespans` returns a `SimulatedTree` with a
  `birth_epoch` column in `records`; its `birth_rate` and `death_rate` are
  `PiecewiseConstant` objects holding the epoch rates and end times.
  `ensure_non_extinct=True` redraws, up to `max_retries` times, until at least
  one individual survives.

## Notes

- `max_population` is a static argument of the compiled kernel, so using a new
  value for the first time recompiles it.
- The first call of a session includes compilation; time later calls to
  measure steady-state cost (see `benchmarks/simulator_speed.qmd`).
- `import bdsurv` enables JAX 64-bit mode and defaults `JAX_PLATFORMS=cpu`
  unless the variable is already set.
