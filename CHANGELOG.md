# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `simulate_linear_rate_contour_process` and `simulate_linear_survival_splitting`
  (with `LinearSplittingConfig`): the contour process and multilevel splitting for
  birth and death rates that are piecewise constant or piecewise linear in time.
  The maths (hazard inversion, survival probability `1 / (1 + I(T))`, mean
  cumulative individuals) is in `bdsurv._linear_math`, ported from `bdpinf`.
- `SimulatedTree.birth_rate_function` / `death_rate_function`: the rates of a
  linear-rate tree (its scalar `birth_rate` / `death_rate` are NaN).
- `scripts/bench_arley_comparison.py` and `benchmarks/arley_comparison.qmd`:
  contour against splitting on Arley processes (`lambda = 1`, `mu(t) = m t`)
  whose slopes match the constant-rate benchmark cells.
- `scripts/bench_method_comparison_local.sh` takes `SCRIPT=` to run either
  benchmark.
- `bdsurv.count_draws`: a context manager that counts the random numbers the
  conditioned simulators draw (by role and kind), plus events, clones, records
  copied, attempts, contour steps and spine resets. Counted runs give identical
  trees. `scripts/count_draws.py` runs it over the benchmark cells with the
  benchmark seeds.
- `scripts/bench_arley_comparison.slurm`: SLURM array job for the Arley benchmark.

### Changed

- Speed-ups that leave every seeded result unchanged: `numpy.isclose` replaced by
  an inline comparison in the contour and trace code; splitting particles share
  immutable record dicts (copy-on-write), so cloning is one shallow copy; scalar
  hazard inversion in pure Python. Splitting runs 3.6-5x and the linear contour
  about 15x faster at the Arley cells measured.

## [0.3.0] - 2026-11-02

### Added

- `compute_level_replicates`: per-level replicate counts `R_i`, the smallest
  integer `>= 2` keeping the expected number of live lines at least one.
- `default_max_restarts`: number of attempts so at least one survives with
  probability 0.999, from a Galton-Watson approximation.
- `SplittingConfig.max_restarts` and `SplittingSimulationResult.n_attempts` /
  `max_restarts`: a failed run (no survivor at `T`) restarts from time 0.

- `scripts/bench_method_comparison.py` and `.slurm`: parallel (SLURM array)
  benchmark of the contour process against splitting over a (mu, T) grid, whose
  merged results `benchmarks/method_comparison.qmd` reads.
- `_quarto.yml`: the repository is now a Quarto website project (render with
  `uv run quarto render`); new `bench` dependency group.

### Changed

- Splitting accepts any positive rates (supercritical and critical as well as
  subcritical). When `Pr(T) > 0.5` (always for `death_rate / birth_rate < 0.5`)
  it simulates trees one at a time until one survives, with unbounded restarts
  unless `max_restarts` is given; `SplittingSimulationResult.max_restarts` is
  then `None`.
- The default number of starting levels is chosen so `Pr(T)^(1/K)` is about 1/3;
  `n_levels` now defaults to `None`.
- The coarsening step of `select_level_times_for_band` tries removing each
  `t_i` whose ratio exceeds 1/2 in a left-to-right sweep (repeated until stable),
  keeping the removal if the merged ratio stays at least `e^-2`; no time is
  protected, so refined midpoints can be removed again.
- **Breaking:** `select_level_times_for_band` now refines/coarsens a starting grid
  (`n_levels` or `initial_level_times`) so each ratio `Pr(t_i)/Pr(t_{i-1})` lies in
  `[e^-2, 0.5]`, instead of searching over equal splits.
- **Breaking:** `SplittingConfig.splitting_factor` is replaced by
  `replicates` (one count per interval, requires `level_times`), and
  `SplittingSimulationResult.splitting_factor` by `replicates`.
- **Breaking:** the splitting simulator no longer estimates the survival
  probability. `survival_probability_estimate`, `level_survival_probabilities`
  and the per-level survival estimates on `LevelDiagnostic` are removed; the
  result keeps the particles that survive to `T`.
- A user-supplied `level_times` is now a starting grid unless `replicates` is also
  given. Survivors are no longer enriched after the final level.

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
