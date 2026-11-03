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

## `simulator_speed.qmd`

Times every `bdsurv` simulator once, on a laptop-sized workload.
Exact-versus-theory validation of the simulated ages, and the timing of sampling
directly from the age distributions, live in `bdpinf` (`qmd_scripts/validation/`).

## `arley_comparison.qmd`

The same comparison for Arley processes (`lambda = 1`, `mu(t) = m t`) with the linear-rate
simulators, each slope `m` matched to a constant-rate cell (see
`scripts/bench_arley_comparison.py`).
Cells whose Arley trees would be too large (`mu` = 1.2, 1.5, 2) are listed but not run.
Data: `benchmarks/data/arley_comparison.csv`, `arley_comparison_units.csv`,
`arley_slopes.csv` and `arley_environment.json`.
Produce it locally with
`SCRIPT=scripts/bench_arley_comparison.py scripts/bench_method_comparison_local.sh`,
or on a cluster with `scripts/bench_arley_comparison.slurm submit`.

Both comparisons also have an untimed *draw count* (`scripts/count_draws.py`, writing
`benchmarks/data/draws.csv`): it re-runs the benchmark seeds inside `bdsurv.count_draws`
and records the random numbers each method draws, plus events, clones and records
copied.
Counts are deterministic and counted runs give identical trees, so this reproduces
the timed runs without a wrapper in them.
Shard it with `--shard i/N` (one replicate of all cells takes about 8 minutes serially),
then `--merge`.

Run both timed benchmarks on the same code version before comparing them: the
speed-ups to the shared splitting and contour code (inline `isclose`, copy-on-write
particle records) leave every seeded result unchanged but cut run times several-fold,
and `arley_comparison.qmd` reads `method_comparison.csv` for the constant-rate side.

## `method_comparison.qmd`

Compares the contour process with multilevel splitting over a grid of death rates
`mu` (0.6 to 10, with `lambda = 1`) and horizons `T` (1 to 100).
It shows one tab per `mu` with cost per tree against `T` (log-log axes,
with a clickable legend), a splitting-versus-contour ratio map, and a Kolmogorov–Smirnov check of
the simulated population at `T` against its exact geometric law.

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
An earlier 100-replicate laptop run (before the speed-ups) took about 4.7 CPU-hours
in total (most cells take milliseconds), so 200 tasks should finish quickly.
Raise `REPLICATES` for tighter bands.
Timings from different nodes scatter, so the notebook lists the hosts used and
shows percentile bands.
