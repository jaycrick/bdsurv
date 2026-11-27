# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.9.0] - 2026-11-26

### Added

- `LogisticHazard(L, k, c)` (hazard $L\,\sigma(k(t-c))$, closed-form forward and backward inversion) in `bdsurv.hazards`, exported from `bdsurv`.
  Cumulative hazards use stable forms (`softplus` differences).
- `LogisticHazard` accepts `k < 0` (a falling, defective logistic with `cumulative_limit` $(L/|k|)\,\mathrm{softplus}(|k|c)$ and closed-form forward and backward inversion).
- `SumHazard(parts)`, with `SumHazard.with_floor(floor, hazard)` and `LogisticHazard.with_floor(floor, L, k, c)`: hazards added by superposition, sampled as the minimum (forward) or maximum (backward) of one closed-form draw per part, exact in law.
  New `Hazard.n_draws` and split methods (`forward_time_split`, `backward_time_split`, `forward_offsets_split`).
  Hazards with `n_draws > 1` draw several exponentials per event in `sample_residual_lifetimes`, `simulate_birth_death`-type forward runs and the contour steps; seeded output of every other hazard and rate is unchanged.
- Scalar `math`-only `forward_time`/`backward_time` fast path for `LogisticHazard` and `GompertzHazard` (and so for `SumHazard` floors): faster simulator per-event sampling, same formulas as the vectorised offsets (agreement to round-off).
  `QuadraticHazard`/`CubicHazard.check_nonnegative` now uses plain float arithmetic, with unchanged behaviour.

### Changed

- `QuadraticHazard` and `CubicHazard` now invert by closed form plus Newton (with a gated algebraic root and a Brent fallback for flagged elements) in `forward_offsets`, `backward_offsets`, `forward_time` and `backward_time`, instead of Brent throughout.
  This is faster; the law is unchanged, but seeded results of these hazards (and of `PiecewiseHazard.from_polynomials`) change at round-off.
  Rate-based streams are byte-identical.
  Brent stays available through `forward_offsets_numerical` and `backward_offsets_numerical`.

## [0.8.0] - 2026-11-20

### Added

- `CubicHazard(a, b, c, d)` in `bdsurv.hazards` (exported from `bdsurv`): hazard $a + bt + ct^2 + dt^3$, inverted by Brent's method, with exact non-negativity checks.
- `PiecewiseHazard.from_polynomials(coefficients, epoch_end_times)` builds a piecewise hazard from local-time polynomial coefficients (one to four per epoch), using `QuadraticHazard` or `CubicHazard` pieces.

## [0.7.0] - 2026-11-20

### Added

- `log_mean_cumulative_individuals_curve(birth_rate, death_rate, times, terminal_time=None)` is exported from `bdsurv` (it lives in `bdsurv.survival`) and returns `log_mean_cumulative_individuals` at every time in `times` (any order, duplicates allowed, output aligned with the input) from one panelled quadrature pass over `[0, max(times)]` with each time an edge, instead of one integral per time.
  Each element agrees with the scalar function to rounding; it accepts the same rates, including a `Hazard`.

## [0.6.0] - 2026-11-19

### Added

- `log_mean_cumulative_individuals(birth_rate, death_rate, t, terminal_time=None)` is exported from `bdsurv` (it lives in `bdsurv.survival`) and returns the log of the expected number of individuals ever born by `t`, the root included, for the process not conditioned on survival.
- Hazard families `PowerHazard` (with `PowerHazard.weibull` and `PowerHazard.rayleigh`), `GompertzHazard`, `LogLogisticHazard`, `LomaxHazard` and `QuadraticHazard`, and `PiecewiseHazard`, which composes them over epochs in local time, all subclasses of the new `Hazard` base class.
  Residual times are exact left-truncated inversions of the cumulative hazard: closed forms for all but `QuadraticHazard`, which uses Brent's method; `HazardInversionError` reports a failed inversion.
  See `docs/hazards.md`.
- `sample_residual_lifetimes(hazard, start_times, random_seed=None, end_time=inf)` draws the left-truncated time to the next event from each start time.
- A `Hazard` is accepted wherever a `RateInput` is: `simulate_birth_death`, `simulate_contour_given_survival`, `simulate_contour_given_population`, `simulate_contour_given_ages`, `simulate_survival_splitting`, the level helpers, `log_survival_probability` and `log_mean_cumulative_individuals`.
  `terminal_time` is required unless a rate fixes the end time (a `PiecewiseHazard` or a rate object).
  The forward simulators use competing risks for these rates, which draws two exponentials per event, so `count_draws` totals differ from the constant and linear path, whose seeded output is unchanged.

### Changed

- `scipy` is now a runtime dependency (it was a development dependency).
- Splitting records clones dropped by the `n_particles` cap, and the merged draw counts report `capped_levels` and `dropped_by_cap`.
- The repository is public; install with `bdsurv @ git+https://github.com/jaycrick/bdsurv@v0.6.0`.
- Dependencies upgraded (`uv lock --upgrade`).

### Added (repository)

- The contour workshop talk, `talks/contour_workshop/` (build with `talks/contour_workshop/build.sh`).

## [0.5.0] - 2026-11-16

This release unifies the public API: the pairs of functions and config classes
that differed only in whether rates were floats or rate objects are merged, and
the two result types become one.
There are no aliases for the old names.

### Changed (breaking)

| Old | New |
| --- | --- |
| `simulate_birth_death_process(birth_rates, death_rates, times, ...)`, `simulate_birth_death_linear(birth, death, ...)` | `simulate_birth_death(birth_rate, death_rate, terminal_time=None, random_seed=None, *, ensure_non_extinct=False, max_retries=100)` with `RateInput` rates |
| `simulate_birth_death_process_from_contour_process`, `simulate_linear_rate_contour_process` | `simulate_contour_given_survival(birth_rate, death_rate, terminal_time=None, random_seed=None)` |
| `SplittingConfig`, `LinearSplittingConfig` | one `SplittingConfig(birth_rate, death_rate, n_particles, terminal_time=None, ...)`, keyword-only |
| `simulate_subcritical_survival_splitting`, `simulate_linear_survival_splitting` | `simulate_survival_splitting(config)` |
| `select_level_times_for_band`, `compute_level_replicates`, `default_max_restarts` (float rates) | same names, `RateInput` rates, and an optional `terminal_time` |
| `SimulatedTree` (NaN rates and `*_function` fields), `SimulatedProcess` | one `SimulatedTree(records, birth_rate, death_rate, terminal_time)` whose rates are always rate objects (a float input is stored as a one-epoch `PiecewiseConstant`) |
| `bdsurv._extinction.pr_extinction(birth_rate, death_rate, terminal_time)` | public `bdsurv.log_survival_probability(birth_rate, death_rate, t, terminal_time=None)`; the extinction probability is `-math.expm1(...)` of it |
| modules `bdsurv.linear_rates`, `bdsurv.gillespie` | `bdsurv.rates` (the rate types), `bdsurv.forward` (`simulate_birth_death`) |

Further changes of the same kind:

- **Rates** follow one rule in `simulate_birth_death`, `simulate_contour_given_survival`,
  `simulate_contour_given_population`, `simulate_contour_given_ages`,
  `simulate_survival_splitting`, the three level helpers and `log_survival_probability`:
  each rate is a float or a rate object, finite and non-negative (zero is allowed),
  and `terminal_time` is required when both rates are floats and otherwise must be
  `None` or equal to the rate objects' common end time.
  A float rate of exactly zero used to be rejected by the float functions and is now
  accepted.
  `simulate_contour_given_survival` raises `ValueError` when the death rate is
  identically zero, for float rates as before for rate objects.
  `simulate_birth_death_lifespans`, `simulate_birth_death_lifespans_arrays`,
  `sample_conditioned_survival_process` and
  `sample_conditioned_survival_per_posterior` keep their per-epoch array arguments and
  still require strictly positive rates.
- **Keyword-only options.** `simulate_contour_given_population(birth_rate, death_rate,
  n_alive, terminal_time=None, random_seed=None, *, direction="auto")` and
  `simulate_contour_given_ages(birth_rate, death_rate, ages, terminal_time=None,
  random_seed=None, *, spine_index=None, direction="auto")`: `direction` and
  `spine_index` can no longer be passed positionally, and `random_seed` moved before
  `spine_index` in the ages function.
- **`SplittingConfig` is keyword-only** (`@dataclass(frozen=True, kw_only=True)`) and
  its fields are `birth_rate`, `death_rate`, `n_particles`, `terminal_time=None`, then
  `level_times`, `n_levels`, `random_seed`, `n_conditioned_trajectories`, `replicates`,
  `max_restarts`.
  The field order changed, so positional construction would swap `terminal_time` and
  `n_particles`; it now raises `TypeError`.
- **One result type.** Every simulator returns `SimulatedTree`, including
  `simulate_birth_death_lifespans` (whose records keep the `birth_epoch` column).
  Its `birth_rate` and `death_rate` are rate objects: a rate object given by the caller
  is stored as given, a float as `PiecewiseConstant([value], [terminal_time])`, and the
  per-epoch arrays of the lifespan simulator as `PiecewiseConstant(values,
  epoch_end_times)`.
  What `SimulatedProcess` carried is recovered as `tree.birth_rate.epoch_end_times`,
  `tree.birth_rate.values`, `tree.death_rate.values` and `tree.terminal_time`
  (`start_time` was always 0 and `initial_population` always 1).
  `simulate_birth_death` with rate objects no longer returns the merged segment
  boundaries as `epoch_end_times`; the rates themselves are stored.
  `birth_rate_function` and `death_rate_function` are removed.

### Changed

- Seeded output differs from 0.4.0 for every simulator: child generators are now
  spawned with `SeedSequence.spawn` (`bdsurv._seeding`).
- `parent_id` and `death_time` are always `float64` (NaN for none) in every records
  table, and every simulator builds its table through `bdsurv.tree.build_records`.
- Invalid inputs such as NaN or infinite rates and times now raise `ValueError`
  instead of being accepted or hanging.
- `sample_conditioned_survival_per_posterior` validates its arguments up front and no
  longer turns exceptions into warnings.
- An inconsistent lifespan event stream raises instead of producing wrong genealogy.
- Python 3.13 is the minimum version.
- Plain `pytest` skips the slow statistical tests (`pytest -m slow` runs them).

### Added

- `scripts/figure_export.py`: `save_chart` (Altair via vl-convert) writes a figure and
  records it in a per-source manifest; `prune` (and `scripts/prune_figures.sh`) lists or
  deletes figures no manifest lists. The benchmark notebooks now save each chart they
  display as a PDF in `figures/benchmarks/` (git-ignored); `vl-convert-python` joins the
  `bench` dependency group.
- `benchmarks/method_comparison.qmd`: a talk-sized random-draws-per-tree chart over death
  rate at `T = 60` (saved as `mc_draws_by_mu_T60_talk.pdf`), with the splitting-to-contour
  draw ratios at the largest `mu`.

### Removed

- `SimulatedProcess`, `LinearSplittingConfig`, and the functions in the table above
  under their old names.
- Modules `bdsurv.gillespie`, `bdsurv.linear_rates` and `bdsurv._extinction`.

### Migrating `bdpinf`

- `simulate_birth_death_process(b, d, times, ...)`: write
  `simulate_birth_death(PiecewiseConstant(b, times), PiecewiseConstant(d, times), ...)`;
  for scalar rates, `simulate_birth_death(b, d, terminal_time, ...)`.
  The seeded output is that of the old function under the same seed.
- `simulate_birth_death_lifespans(...)`: unchanged call; it returns `SimulatedTree`, so
  replace `process.epoch_end_times` by `tree.birth_rate.epoch_end_times` (or
  `tree.terminal_time`), `process.birth_rates` by `tree.birth_rate.values` and
  `process.death_rates` by `tree.death_rate.values`.
- `simulate_birth_death_linear(birth, death, ...)`: write
  `simulate_birth_death(birth, death, ...)`.
- `SplittingConfig(birth_rate, death_rate, terminal_time, n_particles, ...)`: write
  `SplittingConfig(birth_rate=..., death_rate=..., terminal_time=..., n_particles=...,
  ...)` with keywords only.
  `LinearSplittingConfig(birth, death, n_particles=..., ...)` becomes
  `SplittingConfig(birth_rate=birth, death_rate=death, n_particles=..., ...)`, with no
  `terminal_time`.
  `config.terminal_time` is `None` for rate objects; read `result.level_times[-1]` or a
  tree's `terminal_time` instead.
- Contour entry points: `simulate_birth_death_process_from_contour_process(b, d, T, ...)`
  and `simulate_linear_rate_contour_process(birth, death, ...)` become
  `simulate_contour_given_survival(b, d, T, ...)` and
  `simulate_contour_given_survival(birth, death, ...)`.
  `simulate_contour_given_population` and `simulate_contour_given_ages` keep their
  names; pass `direction=` and `spine_index=` by keyword.
- Splitting entry points: `simulate_subcritical_survival_splitting(config)` and
  `simulate_linear_survival_splitting(config)` become `simulate_survival_splitting(config)`.
- `SplittingSimulationResult`: unchanged apart from `config`, which is typed
  `SplittingConfig`.
  `select_level_times_for_band`, `compute_level_replicates` and `default_max_restarts`
  keep their names and keyword arguments and also take rate objects.
- `SimulatedTree`: read `tree.birth_rate` and `tree.death_rate` as rate objects instead
  of floats (for a float input, `tree.birth_rate.values[0]`), and drop any use of
  `birth_rate_function` and `death_rate_function` or of NaN rates.
- `SimulatedProcess`: use `SimulatedTree`, which has `records`, `birth_rate`,
  `death_rate` and `terminal_time`; see the lifespan item above for the removed fields.
- `bdsurv._extinction.pr_extinction(b, d, T)`: write
  `-math.expm1(bdsurv.log_survival_probability(b, d, T))`.
  `log_survival_probability` also takes rate objects, and its result is the log of the
  survival probability.
- `build_target_ranges`: unchanged.
- `ConstantRateLike`, `LinearRateLike` (and `PiecewiseConstant`, `PiecewiseLinear`):
  import them from `bdsurv` or `bdsurv.rates`, no longer from `bdsurv.linear_rates`.
- Submodules: `bdsurv.splitting`, `bdsurv.lifespans`, `bdsurv.tree` and
  `bdsurv.era_conditioning` keep their paths and names, apart from the changes above;
  `bdsurv.linear_rates` is replaced by `bdsurv.rates` and `bdsurv.gillespie` by
  `bdsurv.forward`.

## [0.4.0] - 2026-11-11

### Added

- `simulate_contour_given_population`: exact contour simulator for a birth-death
  tree conditioned on exactly `n` individuals alive at `T`, for scalar or
  piecewise constant / linear rates (`docs/contour_conditioned.md`).
- `simulate_contour_given_ages`: exact contour simulator conditioned on the ages
  of the survivors at `T`, with the spine sampled from its exact weights or given.
- `LevelDiagnostic.n_dropped_by_cap` and `SplittingSimulationResult.n_dropped_by_cap` /
  `n_capped_levels`: how many clones the `n_particles` cap discarded (per level, and
  summed over levels and attempts). `count_draws` reports them as `dropped_by_cap`
  and `capped_levels`.
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
- `bench_arley_comparison.py --grid` and `scripts/bench_arley_grid.slurm`: every matched
  slope over the whole `T` grid. `benchmarks/arley_comparison.qmd` gains two tabbed
  sections from it, cost per tree over time by death-rate slope (with a graph of the birth
  and death rates over time in each tab) and over slope by time; charts in
  `scripts/arley_charts.py`.
- `scripts/bench_arley_comparison.slurm` and `scripts/count_draws.slurm`: SLURM array
  jobs (with a `resubmit` mode that re-runs only the tasks without output) for the
  Arley benchmark and the draw counts.

### Changed

- Benchmark grids: the constant-rate benchmark now covers 18 death rates (0.6 to 20)
  and 21 rounded horizons (1 to 100), up from 9 rates (0.6 to 10) and 13 unrounded
  horizons. Seeds now come from the `mu` and `T` values rather than their position in
  the grid, so grid edits no longer reshuffle them. Default block size 5 and unit
  budget 1800 s; 100 tasks.
- Arley benchmark: the slope-by-time grid is 11 round slopes (0.06 to 0.4), each paired
  with the constant rate that has the same mean death rate over `[0, 100]`
  (`mu = 50 m`, 3 to 20). The cumulative-matched cells move to `T` = 20, 40 and 100; the
  survival-matched cells come from the grid. Unit budget 7200 s and 32G per task (three m = 0.06 blocks were OOM-killed at 16G).
- `scripts/count_draws.py` counts draws over the whole constant-rate grid and the whole
  Arley grid (default 50 replicates), with a `cell_set` column and a `--quick` mode;
  `scripts/count_draws.slurm` runs 250 tasks with 32G and 6 h, sized from the first cluster run. The cost charts on the
  benchmark pages now show the mean with a 95% confidence interval. The method comparison
  and Arley pages gain random-draws-per-tree sections over time (by death rate or slope)
  and over death rate or slope (by time). Tabs are generated from the data.
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
