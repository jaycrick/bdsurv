"""Save Altair charts and keep a manifest of the figures each source produced.

``save_chart`` writes a chart with the vl-convert engine and records the file in
``<output_dir>/.figure_manifest/<source-slug>.txt``.
The first line of a manifest is ``# source: <source path>``, then one file name per
line.
The first write per (directory, source) in a process truncates the manifest and
later writes append, so a manifest describes the last execution of its source.

The source is the Quarto document being executed (from ``QUARTO_DOCUMENT_PATH`` and
``QUARTO_DOCUMENT_FILE``, relative to ``QUARTO_PROJECT_ROOT``), else the running
script, else ``interactive``.

``prune`` removes figures no manifest lists::

    python scripts/figure_export.py prune figures [--apply] [--include-unmanaged]

This is a standalone copy of the exporter in ``bdpinf.visualization`` (``bdsurv``
cannot import ``bdpinf``); keep the manifest format in step with it.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MANIFEST_DIR = ".figure_manifest"
FIGURE_SUFFIXES = (".pdf", ".svg", ".png")
INTERACTIVE = "interactive"
_HEADER = "# source: "

# (output directory, source) pairs already written in this process.
_written: dict[tuple[Path, str], set[str]] = {}


def current_source() -> str:
    """Return the Quarto document, else the running script, else ``interactive``."""
    doc_file = os.environ.get("QUARTO_DOCUMENT_FILE")
    if doc_file:
        doc = Path(os.environ.get("QUARTO_DOCUMENT_PATH", "")) / doc_file
        root = os.environ.get("QUARTO_PROJECT_ROOT")
        if root and doc.is_absolute():
            try:
                return doc.relative_to(root).as_posix()
            except ValueError:
                pass
        return doc.as_posix()
    script = getattr(sys.modules.get("__main__"), "__file__", None)
    if script:
        path = Path(script).resolve()
        try:
            return path.relative_to(Path.cwd().resolve()).as_posix()
        except ValueError:
            return path.as_posix()
    return INTERACTIVE


def source_slug(source: str) -> str:
    """Return a file-name-safe slug for a source path."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", source.replace("/", "__")).strip("_")


def _record(output_dir: Path, filename: str, source: str) -> None:
    manifest_dir = output_dir / MANIFEST_DIR
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = manifest_dir / f"{source_slug(source)}.txt"
    key = (output_dir.resolve(), source)
    if key not in _written:
        _written[key] = set()
        manifest.write_text(f"{_HEADER}{source}\n")
    if filename not in _written[key]:
        _written[key].add(filename)
        with manifest.open("a") as handle:
            handle.write(f"{filename}\n")


def save_chart(
    chart: Any,
    filename: str,
    *,
    output_dir: Path | str,
    width: int | None = None,
    height: int | None = None,
    fmt: str = "pdf",
) -> Path:
    """Save ``chart`` as ``<output_dir>/<filename>.<fmt>`` and record it.

    Parameters
    ----------
    chart
        An Altair chart.
    filename
        File stem (no suffix).
    output_dir
        Directory to write into; created if missing.
    width, height
        Optional chart size in pixels, applied with ``properties`` before export.
    fmt
        File format: ``pdf``, ``svg`` or ``png``.

    Returns
    -------
    pathlib.Path
        The written file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if width is not None or height is not None:
        size = {k: v for k, v in (("width", width), ("height", height)) if v}
        chart = chart.properties(**size)
    path = out / f"{filename}.{fmt}"
    chart.save(path, engine="vl-convert")
    _record(out, path.name, current_source())
    return path


# --- prune ---------------------------------------------------------------------------


@dataclass
class PruneReport:
    """What ``prune`` found under a root."""

    stale: list[Path] = field(default_factory=list)
    orphan_manifests: list[Path] = field(default_factory=list)
    unmanaged_dirs: dict[Path, list[Path]] = field(default_factory=dict)
    kept: int = 0


def read_manifest(path: Path) -> tuple[str, list[str]]:
    """Return ``(source, filenames)`` from a manifest file."""
    lines = path.read_text().splitlines()
    source = ""
    names: list[str] = []
    for line in lines:
        if line.startswith(_HEADER):
            source = line[len(_HEADER) :].strip()
        elif line.strip():
            names.append(line.strip())
    return source, names


def _source_exists(source: str, project_root: Path) -> bool:
    if source == INTERACTIVE:
        return True
    if not source:
        return False
    return (project_root / source).exists()


def scan(root: Path, project_root: Path) -> PruneReport:
    """Classify the figure files under ``root`` without changing anything."""
    report = PruneReport()
    for directory, subdirs, files in os.walk(root):
        subdirs[:] = [d for d in subdirs if d != MANIFEST_DIR]
        here = Path(directory)
        figures = sorted(here / f for f in files if f.endswith(FIGURE_SUFFIXES))
        manifest_dir = here / MANIFEST_DIR
        if not manifest_dir.is_dir():
            if figures:
                report.unmanaged_dirs[here] = figures
            continue
        listed: set[str] = set()
        for manifest in sorted(manifest_dir.glob("*.txt")):
            source, names = read_manifest(manifest)
            if _source_exists(source, project_root):
                listed.update(names)
            else:
                report.orphan_manifests.append(manifest)
        for figure in figures:
            if figure.name in listed:
                report.kept += 1
            else:
                report.stale.append(figure)
    return report


def prune(
    root: Path,
    project_root: Path,
    *,
    apply: bool = False,
    include_unmanaged: bool = False,
) -> PruneReport:
    """Report (and with ``apply`` delete) figures no manifest lists."""
    report = scan(root, project_root)
    if apply:
        doomed = list(report.stale) + list(report.orphan_manifests)
        if include_unmanaged:
            for figures in report.unmanaged_dirs.values():
                doomed.extend(figures)
        for path in doomed:
            path.unlink()
        for manifest in report.orphan_manifests:
            try:
                manifest.parent.rmdir()
            except OSError:
                pass
    return report


def format_report(
    report: PruneReport, *, apply: bool, include_unmanaged: bool, root: Path
) -> str:
    """Return a readable summary of a prune run."""

    def rel(path: Path) -> str:
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            return path.as_posix()

    verb = "Deleted" if apply else "Would delete"
    unmanaged_n = sum(len(v) for v in report.unmanaged_dirs.values())
    lines = [f"{'APPLY' if apply else 'DRY RUN'}: figure root {root}"]
    lines.append(f"Kept (listed in a manifest): {report.kept}")
    lines.append(f"{verb} {len(report.stale)} stale figure(s):")
    lines.extend(f"  {rel(p)}" for p in report.stale)
    lines.append(
        f"{verb} {len(report.orphan_manifests)} orphan manifest(s) "
        "(source no longer exists):"
    )
    lines.extend(f"  {rel(p)}" for p in report.orphan_manifests)
    action = verb if include_unmanaged else "Left alone"
    lines.append(
        f"{action} {unmanaged_n} figure(s) in {len(report.unmanaged_dirs)} "
        "unmanaged director(ies) (no manifest):"
    )
    for directory, figures in report.unmanaged_dirs.items():
        lines.append(f"  {rel(directory)}/ ({len(figures)} file(s))")
    if not apply:
        lines.append("Dry run: nothing deleted. Pass --apply to delete.")
    if report.unmanaged_dirs and not include_unmanaged:
        lines.append("Pass --include-unmanaged to remove unmanaged figures too.")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line interface."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prune", help="remove figures no manifest lists")
    p.add_argument("root", type=Path)
    p.add_argument("--project-root", type=Path, default=Path.cwd())
    p.add_argument("--apply", action="store_true", help="delete (default: dry run)")
    p.add_argument("--include-unmanaged", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if not root.is_dir():
        print(f"No figure root at {root}; nothing to prune.")
        return 0
    report = prune(
        root,
        args.project_root.resolve(),
        apply=args.apply,
        include_unmanaged=args.include_unmanaged,
    )
    print(
        format_report(
            report,
            apply=args.apply,
            include_unmanaged=args.include_unmanaged,
            root=root,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
