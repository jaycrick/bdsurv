#!/bin/bash
# SLURM array job: count the random numbers each method draws, on the matched
# benchmark cells (scripts/count_draws.py).
#
# The counted runs reuse the benchmark seeds, so they reproduce exactly what the
# timed runs drew; they are untimed (the counting wrapper adds overhead).  They
# cover the whole constant-rate grid (mu 0.6 to 20 by T 1 to 100), the whole Arley
# slope-by-time grid (slopes 0.06 to 0.4) and the cumulative-matched Arley cells,
# which feed the random-draw sections of benchmarks/method_comparison.qmd and
# benchmarks/arley_comparison.qmd.  Array task i of SHARDS counts the runs whose
# index is i modulo SHARDS and writes benchmarks/_shards_draws/draws_rows_<i>.csv.
# A second job, dependent on the whole array, merges the shards into
# benchmarks/data/draws.csv (committed).  It does not depend on the timed jobs, so
# submit it at the same time.
#
# Usage (from the repo root, on an HPC login node):
#   scripts/count_draws.slurm submit
#   scripts/count_draws.slurm resubmit    # re-run only tasks with no output
#   SHARDS=1000 REPLICATES=100 scripts/count_draws.slurm submit
#
# Settings (environment variables, all tasks must agree):
#   SHARDS        number of array tasks (default 500)
#   REPLICATES    counted runs per model, method and cell (default 50)
#   MEM           memory per task (default 32G; splitting at mid-range mu needed
#                 more than 4G in the timed benchmark, so this is generous)
#
# Sizing: about 59,600 runs at the defaults (36,200 constant-rate, 23,100 Arley
# grid and 300 Arley matched, two methods, 50 replicates), about 60 CPU-hours in
# total with the counting overhead (about 1.25x a timed run).  Runs are dealt out
# heaviest first, round-robin, so each task gets at most about one of the slowest:
# Arley splitting at m = 0.06, T = 100 (about 11 minutes each, more on a slow
# node), so typical tasks take 10-20 minutes and the busiest an hour or two.  There
# is no per-run time budget, so every requested run completes; the 12 hour limit is
# far beyond what any task needs.  `resubmit` re-runs only tasks with no output.
# Check the cluster's limit on array size before submitting.
#
# Logs land in repo-root logs/ (gitignored); submit from the repo root.
#SBATCH --job-name=count-draws
#SBATCH --account=punim1793
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-499
#SBATCH --output=logs/slurm-count-draws-%A_%a-%N.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

SHARDS="${SHARDS:-500}"
REPLICATES="${REPLICATES:-50}"
MEM="${MEM:-32G}"
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
