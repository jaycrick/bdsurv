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

## `method_comparison.qmd`

Compares the contour process with multilevel splitting over a grid of death rates
`mu` (0.6 to 10, with `lambda = 1`) and horizons `T` (1 to 100).
It shows one tab per `mu` with cost per tree against `T` (log x-axis, linear
y-axis), a splitting-versus-contour ratio map, and a Kolmogorov–Smirnov check of
the simulated population at `T` against its exact geometric law.

The page reads `benchmarks/data/method_comparison.csv` (raw timings, one row per
call or run), `method_comparison_units.csv` and `environment.json`.
If `data/method_comparison.csv` is missing it runs a tiny draft grid so the page
still renders.
The data currently committed is a preliminary 10-replicate run on a laptop
(`--replicates 10 --block-size 5 --unit-seconds 120`, 10 local shards).
Replace it with the output of a cluster run before quoting numbers.

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
A 10-replicate run on a laptop took about 21 CPU-minutes, so 100 replicates should
take roughly 3.5 CPU-hours in total (most cells take milliseconds, the heaviest few
take minutes) and 200 tasks should finish quickly.
Raise `REPLICATES` for tighter bands.
Timings from different nodes scatter, so the notebook lists the hosts used and
shows percentile bands.
