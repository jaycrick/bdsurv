# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.2.0]

### Added

- Forward simulators moved from `bdpinf`, returning the new `SimulatedProcess`
  result type (records plus epoch rates and end times):
  `simulate_birth_death_process` (numpy Gillespie, piecewise-constant rates),
  `simulate_birth_death_lifespans` and `simulate_birth_death_lifespans_arrays`
  (JIT-compiled lifespan-at-birth engine), and `simulate_birth_death_linear`
  (exact, piecewise-linear rates).
- Era-conditioned survival sampling (`sample_conditioned_survival_process`,
  `sample_conditioned_survival_per_posterior`, `build_target_ranges`).
- `PiecewiseConstant` and `PiecewiseLinear` rate classes, and the
  `ConstantRateLike` / `LinearRateLike` protocols that `bdpinf`'s rate objects
  also satisfy.
- `benchmarks/simulator_speed.qmd`: wall-clock comparison of every simulator.
- `docs/lifespans.md` and `docs/era_conditioning.md`.

### Changed

- `jax` is now a runtime dependency. `import bdsurv` enables JAX 64-bit mode and
  defaults `JAX_PLATFORMS` to `cpu` when unset.
- The `death_time` column of the records returned by the new simulators is
  always `float` (`NaN` for survivors).

### Documentation

- README points to `bdpinf` for fitting (`bdpinf.inference.fit`) and for exact age sampling under
  time-varying rates (`bdpinf.simulation.simulate_ages_from_rates`).

### Changed

- The validation notebooks (`simulator_comparison`, `subcritical_splitting`) moved to
  `bdpinf` (`qmd_scripts/validation/survival_conditioned_simulation/`), which already
  depends on `bdsurv`; `bdsurv` no longer references `bdpinf` (removed the `paper`
  dependency group and the `bdpinf` path source).

### Added

- Initial extraction from `bdpinf`: `simulate_birth_death_process_from_contour_process`
  (exact contour-process reconstruction) and `simulate_subcritical_survival_splitting`
  (fixed-factor RESTART multilevel splitting), both returning the new pure
  `numpy`/`pandas` `SimulatedTree` result type.
- Closed-form single-epoch extinction probability (`bdsurv._extinction.pr_extinction`),
  replacing the JAX-based multi-epoch kernel used internally by `bdpinf`.
