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
#   SHARDS=360 BLOCK_SIZE=5 scripts/bench_arley_comparison.slurm submit
#
# Settings (environment variables, all tasks must agree):
#   SHARDS        number of array tasks (default 180)
#   REPLICATES    timed replicates per method and cell (default 100)
#   BLOCK_SIZE    replicates per unit (default 5); smaller blocks lose less work
#                 when the time budget cuts one short, and parallelise further
#   UNIT_SECONDS  stop starting replicates in a block after this many seconds
#                 (default 2400); blocks cut short are flagged in the units CSV
#
# Sizing (PROVISIONAL, from the first 30 minutes of a local run; update once
# that run has finished):
#   - 9 cells x 2 methods x (REPLICATES / BLOCK_SIZE) units: 360 at the defaults.
#   - splitting at mu = 10 (m = 0.2) takes about 60-90 s per run and at mu = 5
#     (m = 0.1) over 90 s, so a block of 5 can need 7-10 minutes or more; the
#     contour process at mu = 3 (m = 0.06) takes 2-5 s per tree.
#   - the other cells (mu = 0.6, 0.8, 1, 1.2, 1.5, 2, 3 splitting) were not yet
#     measured; mu = 1 splitting needs several hundred levels.
#   - a splitting shard held about 1 GB after 30 minutes and copies every
#     particle's records at each level, so memory is set well above the
#     constant-rate benchmark's 4G per CPU.
# Units are ordered heaviest first and dealt out round-robin, so tasks finish at
# similar times.  Timings are single-threaded, so one CPU per task.  Check the
# cluster's limit on array size and running jobs before submitting.
#
# Logs land in repo-root logs/ (gitignored); submit from the repo root.
#SBATCH --job-name=bench-arley-comparison
#SBATCH --account=punim1793
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=12G
#SBATCH --time=04:00:00
#SBATCH --array=0-179
#SBATCH --output=logs/slurm-bench-arley-comparison-%A_%a-%N.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

SHARDS="${SHARDS:-180}"
REPLICATES="${REPLICATES:-100}"
BLOCK_SIZE="${BLOCK_SIZE:-5}"
UNIT_SECONDS="${UNIT_SECONDS:-2400}"
SHARDS_DIR=benchmarks/_shards_arley

if [[ "${1:-}" == "submit" ]]; then
    mkdir -p logs
    # Sync once here so the array tasks do not race on the environment.
    uv sync --frozen --no-dev
    rm -f "${SHARDS_DIR}"/arley_rows_*.csv "${SHARDS_DIR}"/arley_units_*.csv \
        "${SHARDS_DIR}"/arley_env_*.json
    export SHARDS REPLICATES BLOCK_SIZE UNIT_SECONDS
    array_id=$(sbatch --parsable --array=0-$((SHARDS - 1)) "$0")
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
