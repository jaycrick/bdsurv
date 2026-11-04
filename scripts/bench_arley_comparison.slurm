#!/bin/bash
# SLURM array job: benchmark the contour process against multilevel splitting on
# the matched Arley cells in scripts/bench_arley_comparison.py.
#
# The work is a list of independent units (cell, method, block of replicates).
# Array task i of SHARDS runs the units whose index is i modulo SHARDS and
# writes benchmarks/_shards_arley/arley_{rows,units,env}_<i>.*.  A second job,
# dependent on the whole array, merges the shards into benchmarks/data/
# (committed, and read by benchmarks/arley_comparison.qmd).
#
# Usage (from the repo root, on an HPC login node):
#   scripts/bench_arley_comparison.slurm submit
#   SHARDS=240 BLOCK_SIZE=5 scripts/bench_arley_comparison.slurm submit
#   scripts/bench_arley_comparison.slurm resubmit
#
# `resubmit` keeps the shard files already written and re-runs only the array
# tasks that have none (killed for memory or time), then merges all of them.  Use
# the same SHARDS, REPLICATES and BLOCK_SIZE as the original `submit`.
#
# Settings (environment variables, all tasks must agree):
#   SHARDS        number of array tasks (default 120)
#   REPLICATES    timed replicates per method and cell (default 100)
#   BLOCK_SIZE    replicates per unit (default 5); smaller blocks lose less work
#                 when the time budget cuts one short, and parallelise further
#   MEM           memory per task (default 16G)
#   UNIT_SECONDS  stop starting replicates in a block after this many seconds
#                 (default 1800); blocks cut short are flagged in the units CSV
#
# Sizing, from timings of the optimised code on one machine (3 trees or one run
# per cell; 1000 particles):
#   - 6 of the 9 matched cells are run (mu = 1.2, 1.5 and 2 are skipped as too
#     large, see MAX_LOG10_CUMULATIVE): 6 x 2 methods x (REPLICATES / BLOCK_SIZE)
#     = 240 units at the defaults.
#   - splitting per run: mu = 0.6 under 1 s; 0.8 about 14 s; 1, 5 and 10 about
#     25-35 s; mu = 3 about 75 s (the heaviest).  A block of 5 at mu = 3 is about
#     6-7 minutes, and run lengths vary a lot with the number of restarts.
#   - contour per tree: 0.1-0.3 s everywhere (mu = 0.8 and 3 the slowest).
#   - total about 5 CPU-hours.  Memory is the catch: a single splitting run peaked
#     at 0.8-1.9 GB locally, but on the cluster the tasks holding the mu = 1 and
#     mu = 3 splitting blocks were OOM-killed at 4G (4.2 GB resident), so the
#     default is now 16G; `resubmit` re-runs just those tasks.
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
#SBATCH --time=03:00:00
#SBATCH --array=0-119
#SBATCH --output=logs/slurm-bench-arley-comparison-%A_%a-%N.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

SHARDS="${SHARDS:-120}"
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
