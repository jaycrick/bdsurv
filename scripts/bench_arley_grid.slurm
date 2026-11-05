#!/bin/bash
# SLURM array job: benchmark the contour process against multilevel splitting on
# every matched Arley slope over the whole terminal-time grid (T = 1 ... 100),
# `scripts/bench_arley_comparison.py --grid`.  It gives cost against T for each
# slope and against slope at each T, which benchmarks/arley_comparison.qmd shows.
#
# The work is a list of independent units (cell, method, block of replicates).
# Array task i of SHARDS runs the units whose index is i modulo SHARDS and
# writes benchmarks/_shards_arley_grid/arley_grid_{rows,units,env}_<i>.*.  A
# second job, dependent on the whole array, merges the shards into
# benchmarks/data/arley_grid*.csv (committed, and read by the notebook).
#
# Usage (from the repo root, on an HPC login node):
#   scripts/bench_arley_grid.slurm submit
#   scripts/bench_arley_grid.slurm resubmit   # re-run only tasks with no output
#
# Settings (environment variables, all tasks must agree):
#   SHARDS        number of array tasks (default 400)
#   REPLICATES    timed replicates per method and cell (default 100)
#   BLOCK_SIZE    replicates per unit (default 5)
#   MEM           memory per task (default 32G; the m = 0.06 blocks at T = 100 needed
#                 more than 16G)
#   UNIT_SECONDS  stop starting replicates in a block after this many seconds
#                 (default 7200); blocks cut short are flagged in the units CSV
#
# Sizing: all 231 (slope, T) cells are run (11 slopes, 0.06 to 0.4, by 21 times;
# every one is under the MAX_LOG10_CUMULATIVE cap), so 231 x 2 methods x 20 blocks
# = 9240 units at the defaults, about 23 per task, dealt out heaviest first.  The
# first Spartan run took 51 CPU-hours over the 397 tasks that finished; the busiest
# took 1.7 hours and the median 1 minute.  The cost is dominated by splitting at
# m = 0.06 and 0.07 at T = 60 to 100: a block of 5 took up to 4414 s, which went
# over the old 3600 s budget (one block lost its last replicates), so the budget is
# now 7200 s.  Memory is the catch: three tasks were OOM-killed at 16G on blocks of
# m = 0.06, T = 100 (their peak was above 16.7 GB; the other blocks of that cell fit),
# so the default is 32G.  If tasks are still killed, `resubmit` them with a larger
# value, e.g. `MEM=64G scripts/bench_arley_grid.slurm resubmit`.
# Timings are single-threaded, so one CPU per task.  Check the cluster's limit on
# array size and running jobs before submitting.
#
# Logs land in repo-root logs/ (gitignored); submit from the repo root.
#SBATCH --job-name=bench-arley-grid
#SBATCH --account=punim1793
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-399
#SBATCH --output=logs/slurm-bench-arley-grid-%A_%a-%N.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

SHARDS="${SHARDS:-400}"
REPLICATES="${REPLICATES:-100}"
BLOCK_SIZE="${BLOCK_SIZE:-5}"
UNIT_SECONDS="${UNIT_SECONDS:-7200}"
SHARDS_DIR=benchmarks/_shards_arley_grid
MEM="${MEM:-32G}"

if [[ "${1:-}" == "submit" || "${1:-}" == "resubmit" ]]; then
    mkdir -p logs
    # Sync once here so the array tasks do not race on the environment.
    uv sync --frozen --no-dev
    if [[ "$1" == "submit" ]]; then
        rm -f "${SHARDS_DIR}"/arley_grid_rows_*.csv "${SHARDS_DIR}"/arley_grid_units_*.csv \
            "${SHARDS_DIR}"/arley_grid_env_*.json
        array="0-$((SHARDS - 1))"
    else
        # Only the tasks that wrote no rows file.
        array=""
        for ((i = 0; i < SHARDS; i++)); do
            [[ -f "${SHARDS_DIR}/arley_grid_rows_${i}.csv" ]] || array="${array:+${array},}${i}"
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
        --job-name=bench-arley-grid-merge \
        --output=logs/slurm-bench-arley-grid-merge-%j-%N.log \
        --wrap="cd ${SLURM_SUBMIT_DIR:-$(pwd)} && uv run --frozen --no-sync --no-dev python scripts/bench_arley_comparison.py --grid --replicates ${REPLICATES} --block-size ${BLOCK_SIZE} --shards-dir ${SHARDS_DIR} --merge"
    exit 0
fi

# One thread per task: the timings are of single-threaded numpy code.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

uv run --frozen --no-sync --no-dev python scripts/bench_arley_comparison.py --grid \
    --replicates "${REPLICATES}" --block-size "${BLOCK_SIZE}" \
    --unit-seconds "${UNIT_SECONDS}" --shards-dir "${SHARDS_DIR}" \
    --shard "${SLURM_ARRAY_TASK_ID}/${SHARDS}"
