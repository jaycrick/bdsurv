#!/bin/bash
# SLURM array job: benchmark the contour process against multilevel splitting on
# the cumulative-matched Arley cells in scripts/bench_arley_comparison.py.
#
# The work is a list of independent units (cell, method, block of replicates).
# Array task i of SHARDS runs the units whose index is i modulo SHARDS and
# writes benchmarks/_shards_arley/arley_{rows,units,env}_<i>.*.  A second job,
# dependent on the whole array, merges the shards into benchmarks/data/
# (committed, and read by benchmarks/arley_comparison.qmd).
#
# Usage (from the repo root, on an HPC login node):
#   scripts/bench_arley_comparison.slurm submit
#   SHARDS=20 BLOCK_SIZE=5 scripts/bench_arley_comparison.slurm submit
#   scripts/bench_arley_comparison.slurm resubmit
#
# `resubmit` keeps the shard files already written and re-runs only the array
# tasks that have none (killed for memory or time), then merges all of them.  Use
# the same SHARDS, REPLICATES and BLOCK_SIZE as the original `submit`.
#
# Settings (environment variables, all tasks must agree):
#   SHARDS        number of array tasks (default 40)
#   REPLICATES    timed replicates per method and cell (default 100)
#   BLOCK_SIZE    replicates per unit (default 5); smaller blocks lose less work
#                 when the time budget cuts one short, and parallelise further
#   MEM           memory per task (default 16G)
#   UNIT_SECONDS  stop starting replicates in a block after this many seconds
#                 (default 1800); blocks cut short are flagged in the units CSV
#
# Sizing: 3 cumulative-matched cells (mu = 0.6 at T = 20, 0.8 at T = 40, 1 at
# T = 100), all under the size cap: 3 x 2 methods x (REPLICATES / BLOCK_SIZE) =
# 120 units at the defaults.  The Spartan run took 0.9 CPU-hours in total; the
# longest block was 226 s (splitting at mu = 1, T = 100) and every block finished.
# With 120 shards the busiest task took 4 minutes, so 40 shards and a 1 hour limit
# are plenty.  Memory: a single splitting run peaked at 0.8-1.9 GB locally and earlier
# tasks were OOM-killed at 4G, so the default stays 16G; `resubmit` re-runs failed tasks.
# The survival-matched cells (mu = 1.2 to 10) of the earlier version now come from
# the slope-by-time grid (scripts/bench_arley_grid.slurm).
# Units are ordered heaviest first and dealt out round-robin, so tasks finish at
# similar times.  Timings are single-threaded, so one CPU per task.  Check the
# cluster's limit on array size and running jobs before submitting.
#
# Logs land in repo-root logs/ (gitignored); submit from the repo root.
#SBATCH --job-name=bench-arley-comparison
#SBATCH --account=punim1793
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-39
#SBATCH --output=logs/slurm-bench-arley-comparison-%A_%a-%N.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

SHARDS="${SHARDS:-40}"
REPLICATES="${REPLICATES:-100}"
BLOCK_SIZE="${BLOCK_SIZE:-5}"
UNIT_SECONDS="${UNIT_SECONDS:-1800}"
SHARDS_DIR=benchmarks/_shards_arley
MEM="${MEM:-16G}"

if [[ "${1:-}" == "submit" || "${1:-}" == "resubmit" ]]; then
    mkdir -p logs
    # Sync once here so the array tasks do not race on the environment.
    uv sync --frozen --no-dev
    if [[ "$1" == "submit" ]]; then
        rm -f "${SHARDS_DIR}"/arley_rows_*.csv "${SHARDS_DIR}"/arley_units_*.csv \
            "${SHARDS_DIR}"/arley_env_*.json
        array="0-$((SHARDS - 1))"
    else
        # Only the tasks that wrote no rows file.
        array=""
        for ((i = 0; i < SHARDS; i++)); do
            [[ -f "${SHARDS_DIR}/arley_rows_${i}.csv" ]] || array="${array:+${array},}${i}"
        done
        if [[ -z "${array}" ]]; then
            echo "Every shard already has results; nothing to resubmit."
            exit 0
        fi
        echo "Resubmitting array tasks: ${array}"
    fi
    export SHARDS REPLICATES BLOCK_SIZE UNIT_SECONDS MEM
    array_id=$(sbatch --parsable --array="${array}" --mem-per-cpu="${MEM}" "$0")
    sbatch --dependency=afterok:"${array_id}" --time=00:20:00 \
        --job-name=bench-arley-comparison-merge \
        --output=logs/slurm-bench-arley-comparison-merge-%j-%N.log \
        --wrap="cd ${SLURM_SUBMIT_DIR:-$(pwd)} && uv run --frozen --no-sync --no-dev python scripts/bench_arley_comparison.py --replicates ${REPLICATES} --block-size ${BLOCK_SIZE} --shards-dir ${SHARDS_DIR} --merge"
    exit 0
fi

# One thread per task: the timings are of single-threaded numpy code.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

uv run --frozen --no-sync --no-dev python scripts/bench_arley_comparison.py \
    --replicates "${REPLICATES}" --block-size "${BLOCK_SIZE}" \
    --unit-seconds "${UNIT_SECONDS}" --shards-dir "${SHARDS_DIR}" \
    --shard "${SLURM_ARRAY_TASK_ID}/${SHARDS}"
