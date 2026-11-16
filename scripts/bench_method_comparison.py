r"""Benchmark the contour process against multilevel splitting on a (mu, T) grid.

Birth rate is fixed at ``lambda = 1``.
For every death rate ``mu`` and horizon ``T`` the script times independent
replicates of

``contour``
    ``simulate_contour_given_survival``: one exact tree per
    call.
``splitting``
    ``simulate_survival_splitting``: one run returns several kept
    trees, which are correlated because they share ancestry.

Each replicate is one row of raw data, so the notebook can show medians and
spreads rather than a single timing.
A cell ``(mu, T)`` is skipped when the expected tree size given survival,
``exp((lambda - mu) T) / Pr(T)``, exceeds ``MAX_EXPECTED_POPULATION``
(supercritical trees grow exponentially).

The work is split into *units* (cell, method, block of replicates) that are
independent, so it parallelises as a SLURM array job: shard ``i/N`` runs the
units whose index is ``i`` modulo ``N``.
Seeds depend only on ``(mu, T, method, replicate)``, so the simulated output
(not the timings) is identical however the units are sharded.

Usage
-----
    uv run python scripts/bench_method_comparison.py --quick
    uv run python scripts/bench_method_comparison.py --shard 3/200
    uv run python scripts/bench_method_comparison.py --merge
    scripts/bench_method_comparison.slurm submit
"""

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Loaded by path from a notebook, ``scripts/`` is not on the import path yet.
_SCRIPTS = str(Path(__file__).resolve().parent)
if _SCRIPTS not in sys.path:
    sys.path.append(_SCRIPTS)

import _bench_common as common
import pandas as pd

from bdsurv import log_survival_probability

LAMBDA = 1.0
MUS = (
    0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0,
    2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0,
)  # fmt: skip
TS = (
    1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0,
    12.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0,
)  # fmt: skip
MAX_EXPECTED_POPULATION = 1e5
BASE_SEED = 20260101
DEFAULT_REPLICATES = 100
DEFAULT_BLOCK_SIZE = 5
# Splitting at mu = 20, T = 100 takes ~100 s per run on the cluster, so a block of
# 5 replicates needs ~500 s; 1800 s leaves room for slow nodes.
DEFAULT_UNIT_SECONDS = 1800.0

QUICK_MUS = (0.6, 1.5, 3.0)
QUICK_TS = (1.0, 3.0, 10.0)
QUICK_REPLICATES = 6
QUICK_BLOCK_SIZE = 3

# Shared with the other benchmark scripts, which read them from this module.
METHODS = common.METHODS
N_PARTICLES = common.N_PARTICLES
MAX_RESTARTS = common.MAX_RESTARTS
ROOT = common.ROOT
environment_record = common.environment_record
_parse_shard = common.parse_shard  # the name the tests use
shard_units = common.shard

DEFAULT_SHARDS_DIR = ROOT / "benchmarks" / "_shards"
DEFAULT_DATA_DIR = ROOT / "benchmarks" / "data"
ROWS_FILE = "method_comparison.csv"
UNITS_FILE = "method_comparison_units.csv"
ENV_FILE = "environment.json"


@dataclass(frozen=True)
class Unit:
    """A block of replicates for one method in one ``(mu, T)`` cell."""

    method: str
    mu: float
    time: float
    mu_index: int
    time_index: int
    first_replicate: int
    n_replicates: int


def expected_population(mu: float, terminal_time: float) -> float:
    """Return the expected tree size at ``T`` given survival.

    ``E[Z_T | Z_T > 0] = exp((lambda - mu) T) / Pr(T)``.
    """
    log_pr = log_survival_probability(
        LAMBDA, mu, terminal_time, terminal_time=terminal_time
    )
    return math.exp((LAMBDA - mu) * terminal_time - log_pr)


def cell_is_feasible(mu: float, terminal_time: float) -> bool:
    """Return whether the expected tree size is within ``MAX_EXPECTED_POPULATION``."""
    return expected_population(mu, terminal_time) <= MAX_EXPECTED_POPULATION


def build_units(
    mus: tuple[float, ...],
    ts: tuple[float, ...],
    replicates: int,
    block_size: int,
) -> list[Unit]:
    """List every unit, heaviest first so round-robin sharding balances load."""
    units: list[Unit] = []
    for mu_index, mu in enumerate(mus):
        for time_index, terminal_time in enumerate(ts):
            if not cell_is_feasible(mu, terminal_time):
                continue
            for method in METHODS:
                units.extend(
                    Unit(
                        method=method,
                        mu=mu,
                        time=terminal_time,
                        mu_index=mu_index,
                        time_index=time_index,
                        first_replicate=first,
                        n_replicates=min(block_size, replicates - first),
                    )
                    for first in range(0, replicates, block_size)
                )
    units.sort(key=lambda u: (-u.time, -u.mu, u.method, u.first_replicate))
    return units


def replicate_seed(unit: Unit, replicate: int) -> int:
    """Return a seed depending only on the cell values, method and replicate.

    The cell enters by value (``mu`` and ``T`` in thousandths), not by position in
    the grid, so adding grid points never changes the seeds of existing cells.
    """
    return common.replicate_seed(
        BASE_SEED,
        [round(unit.mu * 1000), round(unit.time * 1000)],
        unit.method,
        replicate,
    )


def run_replicate(unit: Unit, seed: int) -> dict[str, Any]:
    """Run one replicate of the unit's method on its cell and return the row columns."""
    return common.run_conditioned(unit.method, LAMBDA, unit.mu, unit.time, seed)


def run_unit(
    unit: Unit, unit_seconds: float, host: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run one unit; stop starting replicates once ``unit_seconds`` is used up."""
    return common.run_unit(
        {"method": unit.method, "mu": unit.mu, "T": unit.time},
        range(unit.first_replicate, unit.first_replicate + unit.n_replicates),
        lambda replicate: replicate_seed(unit, replicate),
        lambda seed: run_replicate(unit, seed),
        unit_seconds,
        host,
    )


def _describe(unit: Unit, _rows: list[dict[str, Any]], summary: Any) -> str:
    return (
        f"{unit.method} mu={unit.mu:g} T={unit.time:g} "
        f"block={unit.first_replicate} {summary['status']} {summary['seconds']:.2f}s"
    )


def run_shard(
    units: list[Unit], shard: int, n_shards: int, out_dir: Path, unit_seconds: float
) -> None:
    """Run this shard's units and write its rows, unit summaries and environment."""
    common.run_shard(
        units,
        shard,
        n_shards,
        out_dir,
        lambda unit, host: run_unit(unit, unit_seconds, host),
        common.ShardFiles("", _describe),
    )


def merge_shards(shards_dir: Path, data_dir: Path) -> pd.DataFrame:
    """Concatenate shard files into the committed data files and return the rows."""
    spec = common.MergeSpec(
        "",
        ["method", "mu", "T", "replicate"],
        ROWS_FILE,
        UNITS_FILE,
        ENV_FILE,
        {"lambda": LAMBDA, "max_expected_population": MAX_EXPECTED_POPULATION},
    )
    return common.merge_shards(shards_dir, data_dir, spec)


def main(argv: list[str] | None = None) -> None:
    """Run one shard, merge shard files, or (default) do both in one process."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--shard", type=common.parse_shard, help="run shard i/N only")
    parser.add_argument("--merge", action="store_true", help="merge shard files")
    parser.add_argument("--quick", action="store_true", help="tiny grid for a draft")
    parser.add_argument("--replicates", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--unit-seconds", type=float, default=DEFAULT_UNIT_SECONDS)
    parser.add_argument("--shards-dir", type=Path, default=DEFAULT_SHARDS_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args(argv)

    mus, ts = (QUICK_MUS, QUICK_TS) if args.quick else (MUS, TS)
    replicates = args.replicates or (
        QUICK_REPLICATES if args.quick else DEFAULT_REPLICATES
    )
    block_size = args.block_size or (
        QUICK_BLOCK_SIZE if args.quick else DEFAULT_BLOCK_SIZE
    )

    if args.merge:
        rows = merge_shards(args.shards_dir, args.data_dir)
        print(f"Merged {len(rows)} rows into {args.data_dir / ROWS_FILE}")
        return

    units = build_units(mus, ts, replicates, block_size)
    if args.shard is not None:
        shard, n_shards = args.shard
        run_shard(units, shard, n_shards, args.shards_dir, args.unit_seconds)
        return

    for stale in args.shards_dir.glob("*_0.*"):
        stale.unlink()
    run_shard(units, 0, 1, args.shards_dir, args.unit_seconds)
    rows = merge_shards(args.shards_dir, args.data_dir)
    print(f"Wrote {len(rows)} rows to {args.data_dir / ROWS_FILE}")


if __name__ == "__main__":
    main()
