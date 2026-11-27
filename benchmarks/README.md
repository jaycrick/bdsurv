# Benchmarks

The repository is a [Quarto](https://quarto.org) website project (`_quarto.yml`
at the root).
Render everything, or one notebook, with

```sh
uv sync --all-groups
uv run quarto render
uv run quarto render benchmarks/method_comparison.qmd
```

Quarto, Jupyter and Altair live in the `bench` dependency group; they are tools
you run, not dependencies of the package.
Always use `uv run quarto`, since a bare `quarto` uses the system Python.

### Figures

Executing a notebook also saves every Altair chart it displays as a PDF in the
git-ignored `figures/benchmarks/` (stems start `mc_`, `arley_`, `ccv_` and `hf_`;
`simulator_speed.qmd` has no charts), through `scripts/figure_export.py`
(the vl-convert engine, in the `bench` group).
Each file is recorded in `figures/benchmarks/.figure_manifest/<notebook>.txt`, which is
rewritten on each execution, so it lists exactly the figures of that notebook's last run.
Figures left over from renamed or removed charts are reported, and with `--apply`
deleted, by

```sh
scripts/prune_figures.sh            # dry run
scripts/prune_figures.sh --apply    # delete stale figures and orphan manifests
```

Files in a directory with no manifest are only touched with `--include-unmanaged`.

## `simulator_speed.qmd`

Times every `bdsurv` simulator once, on a laptop-sized workload.
Exact-versus-theory validation of the simulated ages, and the timing of sampling
directly from the age distributions, live in `bdpinf` (`qmd_scripts/validation/`).

## `hazard_families.qmd`

Shows the cost of sampling event times by inverting the cumulative hazard, for
hazards of different shapes (constant, linear, piecewise-linear, logistic,
floor-logistic, quadratic and cubic) along several code paths, one call per event and
one vectorised call.
It also compares the polynomial closed-form inversion with Brent against `n_alive`.
Regenerate the data with `uv run python scripts/bench_hazard_families.py` (about one
hour; `--quick` for a smoke run).
Data: `benchmarks/data/hazard_families.csv` and `hazard_families_environment.json`;
the notebook only reads them.
Figure stem: `hf_`.

## `contour_conditioned_validation.qmd`

Statistical validation of `simulate_contour_given_population` and
`simulate_contour_given_ages` (`docs/contour_conditioned.md`).
It draws its own samples with fixed seeds (2000 per comparison, about four minutes)
and needs no data files:
forward simulation with rejection, the spine and non-spine age laws,
top against bottom generation, composition of the two simulators and the spine choice,
with every p-value collected at the end.

## `arley_comparison.qmd`

The same comparison for Arley processes (`lambda = 1`, `mu(t) = m t`) with the linear-rate
simulators (see `scripts/bench_arley_comparison.py`).
The default run is three cells matched to constant-rate cells on mean cumulative
individuals (`mu` = 0.6, 0.8 and 1 at `T` = 20, 40 and 100).
Cells whose Arley trees would be too large (over 1e5 individuals ever born) are listed
but not run.
Data: `benchmarks/data/arley_comparison.csv`, `arley_comparison_units.csv`,
`arley_slopes.csv` and `arley_environment.json`.
Produce it locally with
`SCRIPT=scripts/bench_arley_comparison.py scripts/bench_method_comparison_local.sh`,
or on a cluster with `scripts/bench_arley_comparison.slurm submit`.

The two tabbed sections (cost per tree over time for each slope, with a graph of the birth
and death rates, and over slope at each time) need every slope run over the whole `T`
grid: `bench_arley_comparison.py --grid`, or on a cluster `scripts/bench_arley_grid.slurm submit`
(400 tasks, 100 replicates, 7200 s block budget, 32G; `resubmit` to re-run failed tasks, with
`MEM=64G` if the m = 0.06 blocks at T = 100 are killed again; about 55-60 CPU-hours).
The slopes are `GRID_SLOPES` (0.06, 0.07, 0.08, 0.1, 0.12, 0.14, 0.17, 0.2, 0.25, 0.3, 0.4).
Each is paired with the constant rate `mu = 50 m` (3 to 20), which has the same mean
death rate over `[0, 100]` and so nearly the same survival probability at `T = 100`
(within 0.2%).
The slowest slope is 0.06: below about 0.055 the mean individuals ever born exceeds the
1e5 size cap before the process dies back.
It writes `benchmarks/data/arley_grid.csv`, `arley_grid_units.csv`, `arley_grid_cells.csv` and
`arley_grid_environment.json`; cells above the size cap are listed in
`arley_grid_cells.csv` with `feasible = False` and not run (none at the current slopes).
Until those files exist the two sections show a "no grid data" note;
`uv run python scripts/bench_arley_comparison.py --grid --quick` makes a tiny draft.

Both comparisons also have an untimed *draw count* (`scripts/count_draws.py`, writing
`benchmarks/data/draws.csv`): it re-runs the benchmark seeds inside `bdsurv.count_draws`
and records the random numbers each method draws, plus events, clones and records
copied.
It covers every feasible cell of the constant-rate grid, every cell of the Arley
slope-by-time grid, and the cumulative-matched Arley cells (the `model` and `cell_set`
columns tell them apart), and feeds the random-draws sections of both pages.
Counts are deterministic and counted runs give identical trees, so this reproduces
the timed runs without a wrapper in them.
On a cluster use `scripts/count_draws.slurm submit` (250 tasks, 50 replicates, 32G, 6 h:
59,600 runs and about 40 CPU-hours; then `resubmit` re-runs any task that wrote nothing),
which merges into `benchmarks/data/draws.csv`.
It does not depend on the timed jobs, so submit all three together.
Locally, `--quick --replicates 2` makes a tiny draft; the full set is too big to run
serially, so shard it with `--shard i/N`, then `--merge`.

Run both timed benchmarks on the same code version before comparing them: the
speed-ups to the shared splitting and contour code (inline `isclose`, copy-on-write
particle records) leave every seeded result unchanged but cut run times several-fold,
and `arley_comparison.qmd` reads `method_comparison.csv` for the constant-rate side.

## `method_comparison.qmd`

Compares the contour process with multilevel splitting over a grid of 18 death rates
`mu` (0.6 to 20, with `lambda = 1`) and 21 horizons `T` (1 to 100, rounded).
It shows one tab per `mu` with cost per tree against `T` (log-log axes,
with a clickable legend), one tab per `T` against `mu`, a splitting-versus-contour ratio
map, random draws per tree on the same two axes, and a Kolmogorov–Smirnov check of
the simulated population at `T` against its exact geometric law.
The cost and draw charts show the mean with a 95% confidence interval.

The page reads `benchmarks/data/method_comparison.csv` (raw timings, one row per
call or run), `method_comparison_units.csv` and `environment.json`.
If `data/method_comparison.csv` is missing it runs a tiny draft grid so the page
still renders.
No results are committed at the moment: the earlier laptop run used code that has
since been sped up (see below), so it was removed. Produce fresh data with
`scripts/bench_method_comparison_local.sh` or on a cluster (below), and commit
`benchmarks/data/`. A cluster run gives the same simulated trees; only the
timings differ.

### Producing the data

The work is a list of independent *units* (a `(mu, T)` cell, a method and a block
of replicates).
Seeds depend only on the cell, method and replicate, so the simulated output is
the same however the units are split up; only the timings differ.

```sh
# Quick check on a laptop (tiny grid, a few seconds):
uv run python scripts/bench_method_comparison.py --quick

# Everything in one process, 10 replicates:
uv run python scripts/bench_method_comparison.py --replicates 10

# Everything, 100 replicates, in parallel on this machine (cores - 2 shards):
scripts/bench_method_comparison_local.sh

# By hand: shard i of N, then merge into benchmarks/data/:
uv run python scripts/bench_method_comparison.py --shard 0/200
uv run python scripts/bench_method_comparison.py --merge
```

A cell `(mu, T)` is skipped when the expected tree size given survival,
`exp((lambda - mu) T) / Pr(T)`, exceeds `MAX_EXPECTED_POPULATION` (1e5), which
trims the longest supercritical horizons.
`--unit-seconds` stops starting replicates in a block once it has used that many
seconds; blocks cut short are flagged in `method_comparison_units.csv`.

### On an HPC cluster (SLURM)

```sh
scripts/bench_method_comparison.slurm submit
SHARDS=400 REPLICATES=200 scripts/bench_method_comparison.slurm submit
```

`submit` syncs the environment once, submits an array job (one task per shard,
one CPU each) and a merge job that depends on the whole array.
Shards go to `benchmarks/_shards/` (git-ignored), logs to `logs/`, and the merged
files to `benchmarks/data/`.
Edit `--account` in the script for your allocation; it is set to `punim1793`.
The 100-replicate cluster run on the 18 by 21 grid up to `mu` = 20 took 17 CPU-hours (the
worst cell, `mu` = 20 at `T` = 100, is about 100 s per splitting run), and the busiest of
200 tasks took under half an hour, so the script now defaults to 100 tasks.
Raise `REPLICATES` for tighter bands.
Timings from different nodes scatter, so the notebook lists the hosts used and
shows percentile bands.
