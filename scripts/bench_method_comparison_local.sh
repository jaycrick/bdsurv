#!/bin/bash
# Run the method-comparison benchmark on this machine: N shard processes in
# parallel, then a merge into benchmarks/data/.  The local counterpart of
# scripts/bench_method_comparison.slurm.
#
# Usage (from anywhere):
#   scripts/bench_method_comparison_local.sh
#   SHARDS=6 REPLICATES=50 scripts/bench_method_comparison_local.sh
#
# Settings (environment variables):
#   SHARDS        parallel processes (default: cores - 2, at least 1)
#   REPLICATES    timed replicates per method and (mu, T) cell (default 100)
#   UNIT_SECONDS  stop starting replicates in a block after this many seconds
#                 (default 900); blocks cut short are flagged in the units CSV
#
# Leave a few cores free and keep the machine otherwise idle: the timings are
# wall-clock, so a busy or throttled CPU adds noise.  Per-shard logs go to
# logs/local-shard-<i>.log (git-ignored).

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

cores=$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu)
default_shards=$((cores > 2 ? cores - 2 : 1))
SHARDS="${SHARDS:-${default_shards}}"
REPLICATES="${REPLICATES:-100}"
UNIT_SECONDS="${UNIT_SECONDS:-900}"

# Stop the machine sleeping while the run is going (macOS only).
if command -v caffeinate >/dev/null 2>&1; then
    caffeinate -i -w $$ &
fi

mkdir -p logs
uv sync --frozen --all-groups
rm -rf benchmarks/_shards

echo "Running ${SHARDS} shards, ${REPLICATES} replicates per cell, on ${cores} cores"
pids=()
for ((i = 0; i < SHARDS; i++)); do
    uv run --no-sync python scripts/bench_method_comparison.py \
        --replicates "${REPLICATES}" --unit-seconds "${UNIT_SECONDS}" \
        --shard "${i}/${SHARDS}" >"logs/local-shard-${i}.log" 2>&1 &
    pids+=($!)
done

failed=0
for pid in "${pids[@]}"; do
    wait "${pid}" || failed=1
done
if [[ "${failed}" -ne 0 ]]; then
    echo "A shard failed; see logs/local-shard-*.log. Not merging." >&2
    exit 1
fi

uv run --no-sync python scripts/bench_method_comparison.py \
    --replicates "${REPLICATES}" --merge
echo "Done. Render with: uv run quarto render"
