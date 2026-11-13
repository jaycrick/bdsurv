#!/usr/bin/env bash
# Report (dry run) or delete (--apply) figures under figures/ that no manifest lists.
# Arguments are passed through to `figure_export.py prune`, e.g.
#   scripts/prune_figures.sh --apply
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run python scripts/figure_export.py prune figures --project-root . "$@"
