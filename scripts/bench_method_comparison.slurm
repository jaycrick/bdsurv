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
#   scripts/bench_method_comparison.slurm resubmit   # re-run only tasks with no output
#
# Settings (environment variables, all tasks must agree):
#   SHARDS        number of array tasks (default 200)
#   REPLICATES    timed replicates per method and (mu, T) cell (default 100)
#   BLOCK_SIZE    replicates per unit (default 5)
#   UNIT_SECONDS  stop starting replicates in a block after this many seconds
#                 (default 1800); blocks cut short are flagged in the units CSV
#   MEM           memory per task (default 8G)
#
# Sizing, from the 100-replicate Spartan run on the earlier 9 x 13 grid (about 2.1
# CPU-hours of splitting, 3 CPU-minutes of contour) extrapolated to the 18 x 21
# grid (mu 0.6 to 20, T 1 to 100; 357 feasible cells): expect about 18-20 CPU-hours
# at 100 replicates, nearly all of it splitting at large mu and long T.  The worst
# cell is splitting at mu = 20, T = 100 at about 100 s per run (mu = 10 took 25 s),
# so a block of 5 is about 500 s, inside the 1800 s unit budget; the contour
# process is milliseconds except near-critical long horizons (about 1 s at
# mu = 0.8, T = 46).  Units are ordered heaviest first and dealt out round-robin,
# so tasks finish at similar times (the busiest task takes well under 2 hours).
# Timings are single-threaded, so one CPU per task.  Check the cluster's limit on
# array size and running jobs before submitting.
#
# Logs land in repo-root logs/ (gitignored); submit from the repo root.
#SBATCH --job-name=bench-method-comparison
#SBATCH --account=punim1793
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=06:00:00
#SBATCH --array=0-199
#SBATCH --output=logs/slurm-bench-method-comparison-%A_%a-%N.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

SHARDS="${SHARDS:-200}"
REPLICATES="${REPLICATES:-100}"
BLOCK_SIZE="${BLOCK_SIZE:-5}"
UNIT_SECONDS="${UNIT_SECONDS:-1800}"
MEM="${MEM:-8G}"
SHARDS_DIR=benchmarks/_shards

if [[ "${1:-}" == "submit" || "${1:-}" == "resubmit" ]]; then
    mkdir -p logs
    # Sync once here so the array tasks do not race on the environment.
    uv sync --frozen --no-dev
    if [[ "$1" == "submit" ]]; then
        rm -f "${SHARDS_DIR}"/rows_*.csv "${SHARDS_DIR}"/units_*.csv \
            "${SHARDS_DIR}"/env_*.json
        array="0-$((SHARDS - 1))"
    else
        # Only the tasks that wrote no rows file.
        array=""
        for ((i = 0; i < SHARDS; i++)); do
            [[ -f "${SHARDS_DIR}/rows_${i}.csv" ]] || array="${array:+${array},}${i}"
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
    --replicates "${REPLICATES}" --block-size "${BLOCK_SIZE}" \
    --unit-seconds "${UNIT_SECONDS}" \
    --shard "${SLURM_ARRAY_TASK_ID}/${SHARDS}"
