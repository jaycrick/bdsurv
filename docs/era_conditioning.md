# Era-conditioned survival sampling

## Overview

`src/bdsurv/era_conditioning.py` samples trajectories of a multi-epoch
birth-death process conditioned on the number of individuals born in each era
who survive to the terminal time $T$.
The motivating question is the proportion of ever-created individuals who are
alive at $T$, given observed per-era survivor counts.

It is a multilevel-splitting / sequential-ABC scheme: simulate one era at a
time, keep particles whose survivor count matches the target, and clone the
ones that pass.
Splitting in general goes back to Kahn and Harris (1951)
(see [References](#references)).
It uses the lifespan-at-birth engine from `bdsurv.lifespans`, so each
individual's death time over the **full** piecewise hazard is known at birth
and "survives to $T$" is well defined while the first eras are still running.

This is separate from `simulate_survival_splitting`, which
estimates the probability of surviving a single-epoch subcritical process and
returns a tree.

## Algorithm

For eras $0, \dots, K-2$ (the last era is left unconditioned):

1. Start `n_initial` independent particles, each a single root individual.
2. Simulate the current era for every particle (`advance_era`).
3. **Filter.** Count individuals born in this era whose death time exceeds $T$.
   Keep the particle if the count lies in `target_ranges[k]`.
   Particles that overflow `max_population` are dropped.
4. **Split.** Clone each passing particle `split_factor` times.
   Each clone gets a fresh PRNG key; the arrays are immutable, so cloning is
   cheap.
5. Move to the next era.

Simulate era $K-1$ unconditioned for the remaining particles and collect the
matched trajectories.
If fewer than `n_samples` are collected, restart from step 1, up to
`max_attempts` era-0 particles in total (a `RuntimeWarning` is raised if the
budget runs out).

## API

- `build_target_ranges(observed_counts, tolerance=...)` builds inclusive `[lo, hi]` ranges
  from observed per-era counts and leaves the last era unconditioned.
- `sample_conditioned_survival_process` draws `n_samples` matched trajectories
  for fixed rates and returns a `ConditionedSurvivalResult`: the
  `MatchedSurvivalSample`s, per-era `EraConditioningDiagnostic`s from the last
  outer iteration, and restart and particle counts.
- `sample_conditioned_survival_per_posterior` runs the sampler once per row of
  posterior rate draws and returns a tidy `pd.DataFrame`.

Each `MatchedSurvivalSample` holds `n_born`, `n_alive_present`,
`proportion_alive`, and per-era ever-created and surviving counts.

## Caveats

- Matched trajectories that share an ancestor particle are correlated.
- Tight `target_ranges` with a small `n_initial` make restarts frequent;
  inspect `diagnostics` for the per-era acceptance rate.
- `max_population` is the per-particle buffer size and, as in
  `bdsurv.lifespans`, a new value recompiles the kernel.

## References

- Kahn and Harris (1951): multilevel splitting, cited for splitting in general only.

The full entry is in [`docs/references.md`](references.md).
