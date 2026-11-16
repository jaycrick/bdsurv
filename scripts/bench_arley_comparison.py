r"""Benchmark contour against splitting on Arley processes matched to constant ones.

The birth rate is ``lambda = 1`` and the death rate is linear from zero,
``mu(t) = m t`` (an Arley process).

Two runs, both with the constant-rate process of ``bench_method_comparison.py``
as the reference:

the slope-by-time grid (``--grid``)
    Each of ``GRID_SLOPES`` is run over the whole constant-rate ``T`` grid (files
    ``arley_grid*.csv``).
    A slope ``m`` is paired with the constant rate ``mu = m * SURVIVAL_T / 2``, the
    same *mean death rate over* ``[0, 100]``.
    At ``T = 100`` this also matches ``1 / Pr(Z_T > 0)`` to the constant-rate value
    to within 0.2% (the largest gap is at ``m = 0.06``, 0.1%).
    The slopes step through round values from 0.06 (``mu = 3``) to 0.4
    (``mu = 20``); slower slopes grow too large before they die back.
the matched cells (the default)
    ``CUMULATIVE_CELLS``: ``mu`` (0.6, 0.8, 1) at ``T`` = 20, 40, 100, with the
    slope ``m`` solving *mean cumulative individuals* equality,
    ``1 + int_0^T exp(tau - m tau^2 / 2) d tau
    = (mu - e^{(1 - mu) T}) / (mu - 1)``  (``1 + T`` at ``mu = 1``).
    ``m`` is the exact root, not rounded.

Cells whose Arley trees would be too large (mean individuals ever born above
``10^MAX_LOG10_CUMULATIVE``) are listed in the slope table with
``feasible = False`` but not run.

All the maths is in ``bdsurv._hazard`` (public entry point
``bdsurv.log_survival_probability``).
The work is split into units and sharded exactly as in
``bench_method_comparison.py``; this script reuses that module's constants,
sharding and environment record.

Usage
-----
    uv run python scripts/bench_arley_comparison.py --slopes
    uv run python scripts/bench_arley_comparison.py --grid --slopes
    uv run python scripts/bench_arley_comparison.py --quick
    uv run python scripts/bench_arley_comparison.py --shard 3/40
    uv run python scripts/bench_arley_comparison.py --merge
"""

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import _bench_common as common
import bench_method_comparison as base
import numpy as np
import pandas as pd

from bdsurv import (
    PiecewiseConstant,
    PiecewiseLinear,
    RateInput,
    log_survival_probability,
)
from bdsurv import _hazard as lm
from bdsurv.rates import Rate, as_rate

LAMBDA = base.LAMBDA
METHODS = base.METHODS
BASE_SEED = base.BASE_SEED + 1
# (mu, T) cells matched on mean cumulative individuals.  T = 40 rather than 50 for
# mu = 0.8, because the matched Arley tree at T = 50 is over the size cap.
CUMULATIVE_CELLS = ((0.6, 20.0), (0.8, 40.0), (1.0, 100.0))
# Round slopes for the slope-by-time grid, paired with the constant rate
# mu = m * SURVIVAL_T / 2 (the same mean death rate over [0, SURVIVAL_T]).
GRID_SLOPES = (0.06, 0.07, 0.08, 0.1, 0.12, 0.14, 0.17, 0.2, 0.25, 0.3, 0.4)
SURVIVAL_T = 100.0
# Cells whose Arley process has more than 10^5 mean individuals ever born are
# skipped, as the constant-rate benchmark skips cells with a mean population
# above 1e5 (``MAX_EXPECTED_POPULATION``).  An Arley process is supercritical
# until t = 1 / m, so a slow slope grows enormous before it dies back, e.g. about
# 10^9.9 individuals at m = 0.025.  A conditioned tree has about 10^5
# individuals at 10^4.6 (m = 0.06), which already costs ~2 GB and minutes per run.
MAX_LOG10_CUMULATIVE = 5.0
QUICK_MUS = (0.6, 1.0)  # matched cells; the grid uses QUICK_SLOPES
QUICK_SLOPES = (0.1, 0.2)
QUICK_REPLICATES = 3
QUICK_BLOCK_SIZE = 3

ROOT = base.ROOT
ROWS_FILE = "arley_comparison.csv"
UNITS_FILE = "arley_comparison_units.csv"
SLOPES_FILE = "arley_slopes.csv"
ENV_FILE = "arley_environment.json"
QUICK_GRID_MAX_T = 10.0


class Files(NamedTuple):
    """Names of the files, shard directory and seed base of one benchmark mode."""

    prefix: str  # shard files are <prefix>_{rows,units,env}_<i>.*
    rows: str
    units: str
    cells: str
    env: str
    shards_dir: Path
    seed_base: int


# The three cumulative-matched (mu, T) cells.
MATCHED = Files(
    "arley",
    ROWS_FILE,
    UNITS_FILE,
    SLOPES_FILE,
    ENV_FILE,
    ROOT / "benchmarks" / "_shards_arley",
    BASE_SEED,
)
# Every grid slope over the whole T grid (``--grid``).
GRID = Files(
    "arley_grid",
    "arley_grid.csv",
    "arley_grid_units.csv",
    "arley_grid_cells.csv",
    "arley_grid_environment.json",
    ROOT / "benchmarks" / "_shards_arley_grid",
    BASE_SEED + 1,
)


@dataclass(frozen=True)
class Cell:
    """One matched comparison: constant ``mu`` against Arley slope ``slope``."""

    mu: float
    time: float
    slope: float
    matched_on: str  # "cumulative" or "survival"
    constant_value: float  # log10 of the matched quantity, constant rates
    arley_value: float  # log10 of the matched quantity, Arley rates
    log10_mean_cumulative: float  # log10 mean individuals ever born, Arley rates

    @property
    def feasible(self) -> bool:
        """Whether trees are small enough to simulate (see ``MAX_LOG10_CUMULATIVE``)."""
        return self.log10_mean_cumulative <= MAX_LOG10_CUMULATIVE


@dataclass(frozen=True)
class Unit:
    """A block of replicates for one method in one cell."""

    method: str
    cell: Cell
    cell_index: int
    first_replicate: int
    n_replicates: int
    seed_base: int = BASE_SEED


def arley_rates(slope: float, terminal_time: float) -> tuple[Any, Any]:
    """Return ``(birth, death)`` for ``lambda = 1`` and ``mu(t) = slope * t``."""
    return (
        PiecewiseConstant(np.array([LAMBDA]), np.array([terminal_time])),
        PiecewiseLinear(
            np.array([0.0]),
            np.array([slope * terminal_time]),
            np.array([terminal_time]),
        ),
    )


def _arley(slope: float, terminal_time: float) -> tuple[Rate, Rate]:
    birth, death = arley_rates(slope, terminal_time)
    return as_rate(birth), as_rate(death)


def log_constant_cumulative(mu: float, terminal_time: float) -> float:
    """Return ``ln`` of the mean cumulative individuals, constant rates."""
    growth = LAMBDA - mu
    if abs(growth) < 1e-12:
        return math.log(1.0 + LAMBDA * terminal_time)
    # (mu - lambda n) / (mu - lambda), n = exp(growth T); sign-safe in logs.
    log_n = growth * terminal_time
    if growth > 0:
        return log_n + math.log((LAMBDA - mu * math.exp(-log_n)) / (LAMBDA - mu))
    return math.log((mu - LAMBDA * math.exp(log_n)) / (mu - LAMBDA))


def match_cumulative_slope(mu: float, terminal_time: float) -> float:
    """Return the slope whose Arley mean cumulative equals the constant-rate one."""
    target = log_constant_cumulative(mu, terminal_time)

    def gap(slope: float) -> float:
        birth, death = _arley(slope, terminal_time)
        return lm.log_mean_cumulative_individuals(birth, death, terminal_time) - target

    low, high = 1e-6, 10.0  # the mean cumulative falls as the slope rises
    for _ in range(200):
        mid = math.sqrt(low * high)
        if gap(mid) > 0.0:
            low = mid
        else:
            high = mid
    return math.sqrt(low * high)


def log_inverse_survival(birth: RateInput, death: RateInput, t: float) -> float:
    """Return ``ln(1 / Pr(Z_t > 0))``."""
    return -log_survival_probability(birth, death, t)


def _log10_cumulative(slope: float, terminal_time: float) -> float:
    birth, death = _arley(slope, terminal_time)
    return lm.log_mean_cumulative_individuals(birth, death, terminal_time) / math.log(
        10
    )


def build_cells() -> list[Cell]:
    """Return the cumulative-matched cells, with the matched quantity on both sides."""
    cells: list[Cell] = []
    for mu, terminal_time in CUMULATIVE_CELLS:
        slope = match_cumulative_slope(mu, terminal_time)
        birth, death = _arley(slope, terminal_time)
        cells.append(
            Cell(
                mu=mu,
                time=terminal_time,
                slope=slope,
                matched_on="cumulative",
                constant_value=log_constant_cumulative(mu, terminal_time)
                / math.log(10),
                arley_value=lm.log_mean_cumulative_individuals(
                    birth, death, terminal_time
                )
                / math.log(10),
                log10_mean_cumulative=_log10_cumulative(slope, terminal_time),
            )
        )
    return cells


def build_grid_cells() -> list[Cell]:
    """Return every grid slope at every ``T`` of the constant-rate grid.

    ``mu`` is the constant rate with the same mean death rate over
    ``[0, SURVIVAL_T]`` as the slope.
    Cells that are too large to simulate stay in the list (``feasible`` is false)
    so cell indices do not depend on the cap.
    """
    cells: list[Cell] = []
    for slope in GRID_SLOPES:
        mu = round(slope * SURVIVAL_T / 2, 6)
        for terminal_time in base.TS:
            birth, death = arley_rates(slope, terminal_time)
            cells.append(
                Cell(
                    mu=mu,
                    time=float(terminal_time),
                    slope=slope,
                    matched_on="mean-rate",
                    constant_value=math.nan,
                    arley_value=log_inverse_survival(birth, death, terminal_time)
                    / math.log(10),
                    log10_mean_cumulative=_log10_cumulative(slope, terminal_time),
                )
            )
    return cells


def slopes_frame(cells: list[Cell]) -> pd.DataFrame:
    """Tabulate the matched cells (quantities are log10)."""
    return pd.DataFrame(
        [
            {
                "mu": c.mu,
                "T": c.time,
                "slope": c.slope,
                "matched_on": c.matched_on,
                "log10_constant": c.constant_value,
                "log10_arley": c.arley_value,
                "log10_mean_cumulative": c.log10_mean_cumulative,
                "feasible": c.feasible,
            }
            for c in cells
        ]
    )


def build_units(
    cells: list[Cell],
    replicates: int,
    block_size: int,
    seed_base: int = BASE_SEED,
) -> list[Unit]:
    """List every unit, heaviest first so round-robin sharding balances load."""
    units: list[Unit] = []
    for cell_index, cell in enumerate(cells):
        if not cell.feasible:  # keep cell_index, so seeds do not depend on the skips
            continue
        for method in METHODS:
            units.extend(
                Unit(
                    method=method,
                    cell=cell,
                    cell_index=cell_index,
                    first_replicate=first,
                    n_replicates=min(block_size, replicates - first),
                    seed_base=seed_base,
                )
                for first in range(0, replicates, block_size)
            )
    units.sort(key=lambda u: (-u.cell.arley_value, u.method, u.first_replicate))
    return units


def replicate_seed(unit: Unit, replicate: int) -> int:
    """Return a seed depending only on the cell values, method and replicate.

    The cell enters by slope (in ten-thousandths) and ``T`` (in thousandths), so
    changing the grid never changes the seeds of the cells that remain.
    """
    return common.replicate_seed(
        unit.seed_base,
        [round(unit.cell.slope * 10_000), round(unit.cell.time * 1000)],
        unit.method,
        replicate,
    )


def run_replicate(unit: Unit, seed: int) -> dict[str, Any]:
    """Run one replicate of the unit's method on its Arley cell; return its row."""
    birth, death = arley_rates(unit.cell.slope, unit.cell.time)
    return common.run_conditioned(unit.method, birth, death, None, seed)


def run_unit(
    unit: Unit, unit_seconds: float, host: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run one unit; stop starting replicates once ``unit_seconds`` is used up."""
    return common.run_unit(
        {
            "method": unit.method,
            "mu": unit.cell.mu,
            "T": unit.cell.time,
            "slope": unit.cell.slope,
        },
        range(unit.first_replicate, unit.first_replicate + unit.n_replicates),
        lambda replicate: replicate_seed(unit, replicate),
        lambda seed: run_replicate(unit, seed),
        unit_seconds,
        host,
    )


def _describe(unit: Unit, _rows: list[dict[str, Any]], summary: Any) -> str:
    return (
        f"{unit.method} mu={unit.cell.mu:g} T={unit.cell.time:g} "
        f"m={unit.cell.slope:.4g} block={unit.first_replicate} "
        f"{summary['status']} {summary['seconds']:.2f}s"
    )


def run_shard(
    units: list[Unit],
    shard: int,
    n_shards: int,
    out_dir: Path,
    unit_seconds: float,
    files: Files = MATCHED,
) -> None:
    """Run this shard's units and write its rows, unit summaries and environment."""
    common.run_shard(
        units,
        shard,
        n_shards,
        out_dir,
        lambda unit, host: run_unit(unit, unit_seconds, host),
        common.ShardFiles(f"{files.prefix}_", _describe),
    )


def merge_shards(
    shards_dir: Path, data_dir: Path, cells: list[Cell], files: Files = MATCHED
) -> pd.DataFrame:
    """Concatenate shard files into the committed data files and return the rows."""
    spec = common.MergeSpec(
        f"{files.prefix}_",
        ["method", "mu", "T", "replicate"],
        files.rows,
        files.units,
        files.env,
        {"lambda": LAMBDA},
    )
    rows = common.merge_shards(shards_dir, data_dir, spec)
    slopes_frame(cells).to_csv(data_dir / files.cells, index=False)
    return rows


def main(argv: list[str] | None = None) -> None:
    """Print the slope table, run one shard, merge shard files, or do both."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--slopes", action="store_true", help="print the cells")
    parser.add_argument(
        "--grid",
        action="store_true",
        help="every grid slope over the whole T grid, not just the matched cells",
    )
    parser.add_argument("--shard", type=common.parse_shard, help="run shard i/N only")
    parser.add_argument("--merge", action="store_true", help="merge shard files")
    parser.add_argument("--quick", action="store_true", help="tiny grid for a draft")
    parser.add_argument("--replicates", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--unit-seconds", type=float, default=base.DEFAULT_UNIT_SECONDS)
    parser.add_argument("--shards-dir", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=base.DEFAULT_DATA_DIR)
    args = parser.parse_args(argv)

    files = GRID if args.grid else MATCHED
    shards_dir = args.shards_dir or files.shards_dir
    cells = build_grid_cells() if args.grid else build_cells()
    if args.slopes:
        print(slopes_frame(cells).to_string(index=False))
        return
    if args.quick:
        if args.grid:
            cells = [
                c
                for c in cells
                if c.slope in QUICK_SLOPES and c.time <= QUICK_GRID_MAX_T
            ]
        else:
            cells = [c for c in cells if c.mu in QUICK_MUS]
    replicates = args.replicates or (
        QUICK_REPLICATES if args.quick else base.DEFAULT_REPLICATES
    )
    block_size = args.block_size or (
        QUICK_BLOCK_SIZE if args.quick else base.DEFAULT_BLOCK_SIZE
    )

    if args.merge:
        rows = merge_shards(shards_dir, args.data_dir, cells, files)
        print(f"Merged {len(rows)} rows into {args.data_dir / files.rows}")
        return

    units = build_units(cells, replicates, block_size, files.seed_base)
    if args.shard is not None:
        shard, n_shards = args.shard
        run_shard(units, shard, n_shards, shards_dir, args.unit_seconds, files)
        return

    for stale in shards_dir.glob(f"{files.prefix}_*_0.*"):
        stale.unlink()
    run_shard(units, 0, 1, shards_dir, args.unit_seconds, files)
    rows = merge_shards(shards_dir, args.data_dir, cells, files)
    print(f"Wrote {len(rows)} rows to {args.data_dir / files.rows}")


if __name__ == "__main__":
    main()
