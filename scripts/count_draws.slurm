#!/bin/bash
# SLURM array job: count the random numbers each method draws, on the matched
# benchmark cells (scripts/count_draws.py).
#
# The counted runs reuse the benchmark seeds, so they reproduce exactly what the
# timed runs drew; they are untimed (the counting wrapper adds overhead).  Array
# task i of SHARDS counts the runs whose index is i modulo SHARDS and writes
# benchmarks/_shards_draws/draws_rows_<i>.csv.  A second job, dependent on the
# whole array, merges the shards into benchmarks/data/draws.csv (committed, and
# read by benchmarks/arley_comparison.qmd).
#
# Usage (from the repo root, on an HPC login node):
#   scripts/count_draws.slurm submit
#   scripts/count_draws.slurm resubmit    # re-run only tasks with no output
#   SHARDS=200 REPLICATES=10 scripts/count_draws.slurm submit
#
# Settings (environment variables, all tasks must agree):
#   SHARDS        number of array tasks (default 100)
#   REPLICATES    counted runs per model, method and cell (default 20)
#   MEM           memory per task (default 16G; splitting at mu = 1 and 3 needed
#                 more than 4G in the timed benchmark)
#
# Sizing: 600 runs at the defaults (6 Arley cells and 9 constant-rate cells, two
# methods, 20 replicates).  Runs are dealt out heaviest first, round-robin.  The
# heaviest are Arley splitting at mu = 3 (several minutes each on the cluster)
# and mu = 1, 5 and 10; contour runs take seconds.  There is no per-run time
# budget, so give --time room for a task that draws the slowest few runs.
# Check the cluster's limit on array size before submitting.
#
# Logs land in repo-root logs/ (gitignored); submit from the repo root.
#SBATCH --job-name=count-draws
#SBATCH --account=punim1793
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-99
#SBATCH --output=logs/slurm-count-draws-%A_%a-%N.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

SHARDS="${SHARDS:-100}"
REPLICATES="${REPLICATES:-20}"
MEM="${MEM:-16G}"
SHARDS_DIR=benchmarks/_shards_draws

if [[ "${1:-}" == "submit" || "${1:-}" == "resubmit" ]]; then
    mkdir -p logs
    # Sync once here so the array tasks do not race on the environment.
    uv sync --frozen --no-dev
    if [[ "$1" == "submit" ]]; then
        rm -f "${SHARDS_DIR}"/draws_rows_*.csv
        array="0-$((SHARDS - 1))"
    else
        # Only the tasks that wrote no rows file.
        array=""
        for ((i = 0; i < SHARDS; i++)); do
            [[ -f "${SHARDS_DIR}/draws_rows_${i}.csv" ]] || array="${array:+${array},}${i}"
        done
        if [[ -z "${array}" ]]; then
            echo "Every shard already has results; nothing to resubmit."
            exit 0
        fi
        echo "Resubmitting array tasks: ${array}"
    fi
    export SHARDS REPLICATES MEM
    array_id=$(sbatch --parsable --array="${array}" --mem-per-cpu="${MEM}" "$0")
    sbatch --dependency=afterok:"${array_id}" --time=00:20:00 \
        --job-name=count-draws-merge \
        --output=logs/slurm-count-draws-merge-%j-%N.log \
        --wrap="cd ${SLURM_SUBMIT_DIR:-$(pwd)} && uv run --frozen --no-sync --no-dev python scripts/count_draws.py --shards-dir ${SHARDS_DIR} --merge"
    exit 0
fi

# One thread per task.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

uv run --frozen --no-sync --no-dev python scripts/count_draws.py \
    --replicates "${REPLICATES}" --shards-dir "${SHARDS_DIR}" \
    --shard "${SLURM_ARRAY_TASK_ID}/${SHARDS}"
