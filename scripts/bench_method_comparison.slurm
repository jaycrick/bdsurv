#!/bin/bash
# SLURM array job: benchmark the contour process against multilevel splitting on
# the (mu, T) grid in scripts/bench_method_comparison.py.
#
# The work is a list of independent units (cell, method, block of replicates).
# Array task i of SHARDS runs the units whose index is i modulo SHARDS and
# writes benchmarks/_shards/{rows,units,env}_<i>.*.  A second job, dependent on
# the whole array, merges the shards into benchmarks/data/ (committed, and read
# by benchmarks/method_comparison.qmd).
#
# Usage (from the repo root, on an HPC login node):
#   scripts/bench_method_comparison.slurm submit
#   SHARDS=400 REPLICATES=200 scripts/bench_method_comparison.slurm submit
#
# Settings (environment variables, all tasks must agree):
#   SHARDS        number of array tasks (default 200)
#   REPLICATES    timed replicates per method and (mu, T) cell (default 100)
#   UNIT_SECONDS  stop starting replicates in a block after this many seconds
#                 (default 900); blocks cut short are flagged in the units CSV
#
# Sizing: a 100-replicate local run, before the splitting speed-ups, took about
# 4.7 CPU-hours in total (longest block about 12 minutes); expect less now.  A single splitting run costs about 40 s at its worst
# grid cell (mu = 10, T = 100) and a contour tree about 13 s at (mu = 0.8,
# T = 50); most cells take milliseconds.  Units are ordered heaviest first and dealt out
# round-robin, so tasks finish at similar times.  Adjust --time if you raise
# REPLICATES or shrink SHARDS.  Timings are single-threaded, so one CPU per task.
# Check the cluster's limit on array size and running jobs before submitting.
#
# Logs land in repo-root logs/ (gitignored); submit from the repo root.
#SBATCH --job-name=bench-method-comparison
#SBATCH --account=punim1793
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=06:00:00
#SBATCH --array=0-199
#SBATCH --output=logs/slurm-bench-method-comparison-%A_%a-%N.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

SHARDS="${SHARDS:-200}"
REPLICATES="${REPLICATES:-100}"
UNIT_SECONDS="${UNIT_SECONDS:-900}"

if [[ "${1:-}" == "submit" ]]; then
    mkdir -p logs
    # Sync once here so the array tasks do not race on the environment.
    uv sync --frozen --no-dev
    rm -f benchmarks/_shards/rows_*.csv benchmarks/_shards/units_*.csv \
        benchmarks/_shards/env_*.json
    export SHARDS REPLICATES UNIT_SECONDS
    array_id=$(sbatch --parsable --array=0-$((SHARDS - 1)) "$0")
    sbatch --dependency=afterok:"${array_id}" --time=00:20:00 \
        --job-name=bench-method-comparison-merge \
        --output=logs/slurm-bench-method-comparison-merge-%j-%N.log \
        --wrap="cd ${SLURM_SUBMIT_DIR:-$(pwd)} && uv run --frozen --no-sync --no-dev python scripts/bench_method_comparison.py --merge"
    exit 0
fi

# One thread per task: the timings are of single-threaded numpy code.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

uv run --frozen --no-sync --no-dev python scripts/bench_method_comparison.py \
    --replicates "${REPLICATES}" --unit-seconds "${UNIT_SECONDS}" \
    --shard "${SLURM_ARRAY_TASK_ID}/${SHARDS}"
