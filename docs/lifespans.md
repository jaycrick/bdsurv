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
3. **Next event.** Within one call of `advance_era` the loop takes the earliest of
   the next birth, the earliest pending death in the buffer, and the epoch end,
   and the call returns when it reaches the epoch end.
   The host loop then starts the next epoch from the returned state.
   The birth waiting time is redrawn at every iteration, including the first
   one after an epoch end, which is exact because the exponential is memoryless.
4. **Buffers.** Birth times, death times, birth epochs, insertion uniforms and
   the population just before each birth live in fixed-size arrays of length
   `max_population`.
   The loop sets an overflow flag instead of writing past the end, and the
   Python wrapper raises `RuntimeError`.

## Forward simulation and the other simulators

`simulate_birth_death` (`src/bdsurv/forward.py`, built on the kernel in
`src/bdsurv/_forward.py`) is the exact event-by-event simulation of
Gillespie (1977) (see [References](#references)): the next event of the whole
population is drawn from its total hazard, and is a birth or a death with
probability proportional to the rates.
The extension to rates that vary linearly within a segment is made here, by
inverting the cumulative hazard of the segment; it is not part of that reference.

Within a segment on which both rates are linear, with $n$ individuals alive at
$t_0$ and $s = t - t_0$, the total hazard is $a + b s$ with
$a = n(\lambda(t_0) + \mu(t_0))$ and $b = n(\lambda' + \mu')$, where
$\lambda'$ and $\mu'$ are the slopes of the rates.
Its integral is $H(s) = a s + b s^2 / 2$, and the next event offset solves
$H(s) = E$ with $E \sim \mathrm{Exp}(1)$:
$$
s = \frac{2E}{a + \sqrt{a^2 + 2 b E}},
$$
which is $E / a$ when $b = 0$.
An event past the end of the segment is discarded and redrawn from the boundary,
which is exact because the event process has independent increments.
The event is a birth with probability $\lambda(t) / (\lambda(t) + \mu(t))$.

The lifespan-at-birth engine of this page simulates the same process for piecewise
constant rates, but draws each individual's death time when it is born instead
of a single total hazard for the population.

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

## References

- Gillespie (1977): exact event-by-event simulation.

The full entry is in [`docs/references.md`](references.md).
