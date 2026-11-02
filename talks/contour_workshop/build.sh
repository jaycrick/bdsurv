#!/usr/bin/env bash
# Build the contour workshop talk: compile the TikZ figures to SVG, then render the slides.
#
# Usage:
#   talks/contour_workshop/build.sh                  # figures, then slides
#   talks/contour_workshop/build.sh --figures-only
#   talks/contour_workshop/build.sh --render-only
#
# Needs pdflatex (with tikz and standalone) and pdftocairo for the figures,
# and uv and quarto for the slides.
set -euo pipefail

# figure name : number of stages (stage k is compiled with \def\stage{k})
FIGURES=("tree_contour:5" "exchanged_pair:2")

here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"

do_figures=1
do_render=1
case "${1:-}" in
  "") ;;
  --figures-only) do_render=0 ;;
  --render-only) do_figures=0 ;;
  *) echo "unknown option: $1" >&2; exit 2 ;;
esac

build_figures() {
  for tool in pdflatex pdftocairo; do
    command -v "$tool" >/dev/null || { echo "missing $tool" >&2; exit 1; }
  done
  local work
  work="$(mktemp -d)"
  trap 'rm -rf "$work"' RETURN
  for entry in "${FIGURES[@]}"; do
    local name="${entry%%:*}" stages="${entry##*:}"
    for k in $(seq 1 "$stages"); do
      echo "figure $name stage $k"
      (cd "$here/figures" && pdflatex -interaction=nonstopmode -halt-on-error \
        -output-directory "$work" -jobname "${name}_${k}" \
        "\\def\\stage{$k}\\input{$name.tex}" >"$work/${name}_${k}.out" 2>&1) \
        || { tail -30 "$work/${name}_${k}.out" >&2; exit 1; }
      pdftocairo -svg "$work/${name}_${k}.pdf" "$here/figures/${name}_${k}.svg"
    done
  done
}

render_slides() {
  (cd "$root" && uv run quarto render talks/contour_workshop/slides.qmd)
  echo "slides: $here/slides.html"
}

((do_figures)) && build_figures
((do_render)) && render_slides
exit 0
