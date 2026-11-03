r"""Benchmark contour against splitting on Arley processes matched to constant ones.

The birth rate is ``lambda = 1`` and the death rate is linear from zero,
``mu(t) = m t`` (an Arley process).
Each slope ``m`` is chosen so that the Arley process matches the constant-rate
process with death rate ``mu`` used in ``bench_method_comparison.py``:

supercritical-side and critical ``mu`` (0.6, 0.8, 1) at ``T`` = 21.544, 46.416, 100
    ``m`` solves *mean cumulative individuals* equality,
    ``1 + int_0^T exp(tau - m tau^2 / 2) d tau
    = (mu - e^{(1 - mu) T}) / (mu - 1)``  (``1 + T`` at ``mu = 1``).
    ``m`` is the exact root, not rounded.
subcritical ``mu`` (1.2, 1.5, 2, 3, 5, 10) at ``T = 100``
    ``m`` is the candidate slope whose ``1 / Pr(Z_T > 0)`` is closest, in
    absolute distance, to the constant-rate value.
    Candidates step by 0.01 below 0.1, by 0.1 from 0.1 to below 2, and by 1
    from 2.
    ``mu = 1.2`` is pinned to ``m = 0.025`` (``SLOPE_OVERRIDES``), the closest
    value to its exact match of about 0.0243 on a 0.005 grid.

All the maths is in ``bdsurv._linear_math`` (ported from ``bdpinf``).
The work is split into units and sharded exactly as in
``bench_method_comparison.py``; this script reuses that module's constants,
sharding and environment record.

Usage
-----
    uv run python scripts/bench_arley_comparison.py --slopes
    uv run python scripts/bench_arley_comparison.py --quick
    uv run python scripts/bench_arley_comparison.py --shard 3/40
    uv run python scripts/bench_arley_comparison.py --merge
"""

import argparse
import json
import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bench_method_comparison as base
import numpy as np
import pandas as pd

from bdsurv import (
    LinearSplittingConfig,
    PiecewiseConstant,
    PiecewiseLinear,
    simulate_linear_rate_contour_process,
    simulate_linear_survival_splitting,
)
from bdsurv import _linear_math as lm
from bdsurv.splitting import _log_survival_probability

LAMBDA = base.LAMBDA
METHODS = base.METHODS
BASE_SEED = base.BASE_SEED + 1
# (mu, T) cells matched on mean cumulative individuals.
CUMULATIVE_CELLS = ((0.6, 21.544), (0.8, 46.416), (1.0, 100.0))
# mu values matched on survival probability, all at SURVIVAL_T.
SURVIVAL_MUS = (1.2, 1.5, 2.0, 3.0, 5.0, 10.0)
SURVIVAL_T = 100.0
SLOPE_OVERRIDES = {1.2: 0.025}
QUICK_MUS = (0.6, 3.0)
QUICK_REPLICATES = 4
QUICK_BLOCK_SIZE = 2

ROOT = base.ROOT
ROWS_FILE = "arley_comparison.csv"
UNITS_FILE = "arley_comparison_units.csv"
SLOPES_FILE = "arley_slopes.csv"
ENV_FILE = "arley_environment.json"


@dataclass(frozen=True)
class Cell:
    """One matched comparison: constant ``mu`` against Arley slope ``slope``."""

    mu: float
    time: float
    slope: float
    matched_on: str  # "cumulative" or "survival"
    constant_value: float  # log10 of the matched quantity, constant rates
    arley_value: float  # log10 of the matched quantity, Arley rates


@dataclass(frozen=True)
class Unit:
    """A block of replicates for one method in one cell."""

    method: str
    cell: Cell
    cell_index: int
    first_replicate: int
    n_replicates: int


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


def _arley(slope: float, terminal_time: float) -> tuple[lm.Rate, lm.Rate]:
    birth, death = arley_rates(slope, terminal_time)
    return lm.as_rate(birth), lm.as_rate(death)


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


def candidate_slopes() -> list[float]:
    """Slopes on the 0.01 / 0.1 / 1 grid, finer at small slopes."""
    fine = [round(0.01 * k, 2) for k in range(1, 10)]
    medium = [round(0.1 * k, 1) for k in range(1, 20)]
    coarse = [float(k) for k in range(2, 31)]
    return fine + medium + coarse


def _log_odds_distance(log_a: float, log_b: float) -> float:
    """Return ``ln |exp(log_a) - exp(log_b)|`` without overflow."""
    high, low = max(log_a, log_b), min(log_a, log_b)
    if high == low:
        return -math.inf
    return high + math.log1p(-math.exp(low - high))


def log_inverse_survival(birth: lm.Rate, death: lm.Rate, t: float) -> float:
    """Return ``ln(1 / Pr(Z_t > 0))``."""
    return -lm.log_survival_probability(birth, death, t)


def match_survival_slope(mu: float, terminal_time: float) -> float:
    """Return the candidate slope with ``1 / Pr(survival)`` closest to constant."""
    if mu in SLOPE_OVERRIDES:
        return SLOPE_OVERRIDES[mu]
    target = -_log_survival_probability(
        birth_rate=LAMBDA, death_rate=mu, time=terminal_time
    )

    def distance(slope: float) -> float:
        birth, death = _arley(slope, terminal_time)
        return _log_odds_distance(
            log_inverse_survival(birth, death, terminal_time), target
        )

    return min(candidate_slopes(), key=distance)


def build_cells() -> list[Cell]:
    """Return every matched cell, with the matched quantity on both sides."""
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
            )
        )
    for mu in SURVIVAL_MUS:
        slope = match_survival_slope(mu, SURVIVAL_T)
        birth, death = _arley(slope, SURVIVAL_T)
        cells.append(
            Cell(
                mu=mu,
                time=SURVIVAL_T,
                slope=slope,
                matched_on="survival",
                constant_value=-_log_survival_probability(
                    birth_rate=LAMBDA, death_rate=mu, time=SURVIVAL_T
                )
                / math.log(10),
                arley_value=log_inverse_survival(birth, death, SURVIVAL_T)
                / math.log(10),
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
            }
            for c in cells
        ]
    )


def build_units(cells: list[Cell], replicates: int, block_size: int) -> list[base.Unit]:
    """List every unit, heaviest first so round-robin sharding balances load."""
    units: list[Unit] = []
    for cell_index, cell in enumerate(cells):
        for method in METHODS:
            for first in range(0, replicates, block_size):
                units.append(
                    Unit(
                        method=method,
                        cell=cell,
                        cell_index=cell_index,
                        first_replicate=first,
                        n_replicates=min(block_size, replicates - first),
                    )
                )
    units.sort(key=lambda u: (-u.cell.arley_value, u.method, u.first_replicate))
    return units


def replicate_seed(unit: Unit, replicate: int) -> int:
    """Return a seed depending only on the cell, method and replicate."""
    sequence = np.random.SeedSequence(
        [BASE_SEED, unit.cell_index, METHODS.index(unit.method), replicate]
    )
    return int(sequence.generate_state(1)[0] % (2**31 - 1))


def _run_contour(unit: Unit, seed: int) -> dict[str, Any]:
    birth, death = arley_rates(unit.cell.slope, unit.cell.time)
    start = time.perf_counter()
    tree = simulate_linear_rate_contour_process(birth, death, random_seed=seed)
    seconds = time.perf_counter() - start
    return {
        "seconds": seconds,
        "n_trees": 1,
        "n_attempts": np.nan,
        "levels": np.nan,
        "pop_at_T": int(tree.records["survived"].sum()),
        "n_individuals": len(tree.records),
    }


def _run_splitting(unit: Unit, seed: int) -> dict[str, Any]:
    birth, death = arley_rates(unit.cell.slope, unit.cell.time)
    config = LinearSplittingConfig(
        birth_rate=birth,
        death_rate=death,
        n_particles=base.N_PARTICLES,
        random_seed=seed,
        max_restarts=base.MAX_RESTARTS,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        start = time.perf_counter()
        result = simulate_linear_survival_splitting(config)
        seconds = time.perf_counter() - start
    kept = result.conditioned_trajectories
    row: dict[str, Any] = {
        "seconds": seconds,
        "n_trees": len(kept),
        "n_attempts": result.n_attempts,
        "levels": len(result.replicates),
        "pop_at_T": np.nan,
        "n_individuals": np.nan,
    }
    if kept:
        chosen = kept[int(np.random.default_rng(seed).integers(len(kept)))]
        records = chosen.process.records
        row["pop_at_T"] = int(records["survived"].sum())
        row["n_individuals"] = len(records)
    return row


def run_unit(
    unit: Unit, unit_seconds: float, host: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run one unit; stop starting replicates once ``unit_seconds`` is used up."""
    runner = _run_contour if unit.method == "contour" else _run_splitting
    rows: list[dict[str, Any]] = []
    spent = 0.0
    status = "ok"
    identity = {"method": unit.method, "mu": unit.cell.mu, "T": unit.cell.time}
    for replicate in range(
        unit.first_replicate, unit.first_replicate + unit.n_replicates
    ):
        if spent > unit_seconds:
            status = "budget"
            break
        seed = replicate_seed(unit, replicate)
        row = runner(unit, seed)
        spent += float(row["seconds"])
        rows.append(
            {
                **identity,
                "slope": unit.cell.slope,
                "replicate": replicate,
                "seed": seed,
                "host": host,
                **row,
            }
        )
    summary = {
        **identity,
        "slope": unit.cell.slope,
        "first_replicate": unit.first_replicate,
        "replicates_requested": unit.n_replicates,
        "replicates_done": len(rows),
        "seconds": spent,
        "status": status,
        "host": host,
    }
    return rows, summary


def run_shard(
    units: list[Unit], shard: int, n_shards: int, out_dir: Path, unit_seconds: float
) -> None:
    """Run this shard's units and write its rows, unit summaries and environment."""
    out_dir.mkdir(parents=True, exist_ok=True)
    env = base.environment_record()
    mine = [unit for index, unit in enumerate(units) if index % n_shards == shard]
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for position, unit in enumerate(mine, start=1):
        unit_rows, summary = run_unit(unit, unit_seconds, env["host"])
        rows += unit_rows
        summaries.append(summary)
        print(
            f"[{shard}/{n_shards}] unit {position}/{len(mine)} "
            f"{unit.method} mu={unit.cell.mu:g} T={unit.cell.time:g} "
            f"m={unit.cell.slope:.4g} block={unit.first_replicate} "
            f"{summary['status']} {summary['seconds']:.2f}s",
            flush=True,
        )
    pd.DataFrame(rows).to_csv(out_dir / f"arley_rows_{shard}.csv", index=False)
    pd.DataFrame(summaries).to_csv(out_dir / f"arley_units_{shard}.csv", index=False)
    (out_dir / f"arley_env_{shard}.json").write_text(json.dumps(env, indent=2) + "\n")


def merge_shards(shards_dir: Path, data_dir: Path, cells: list[Cell]) -> pd.DataFrame:
    """Concatenate shard files into the committed data files and return the rows."""
    row_files = sorted(shards_dir.glob("arley_rows_*.csv"))
    if not row_files:
        msg = f"No shard files found in {shards_dir}"
        raise FileNotFoundError(msg)
    rows = pd.concat([pd.read_csv(f) for f in row_files], ignore_index=True)
    rows = rows.sort_values(["method", "mu", "T", "replicate"]).reset_index(drop=True)
    units = pd.concat(
        [pd.read_csv(f) for f in sorted(shards_dir.glob("arley_units_*.csv"))],
        ignore_index=True,
    )
    envs = [
        json.loads(f.read_text()) for f in sorted(shards_dir.glob("arley_env_*.json"))
    ]
    data_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(data_dir / ROWS_FILE, index=False)
    units.to_csv(data_dir / UNITS_FILE, index=False)
    slopes_frame(cells).to_csv(data_dir / SLOPES_FILE, index=False)
    summary = {
        "lambda": LAMBDA,
        "n_shards": len(envs),
        "hosts": sorted({e["host"] for e in envs}),
        "processors": sorted({e["processor"] for e in envs}),
        "python": sorted({e["python"] for e in envs}),
        "numpy": sorted({e["numpy"] for e in envs}),
        "bdsurv": sorted({e["bdsurv"] for e in envs}),
        "git_sha": sorted({e["git_sha"] for e in envs}),
    }
    (data_dir / ENV_FILE).write_text(json.dumps(summary, indent=2) + "\n")
    return rows


def main(argv: list[str] | None = None) -> None:
    """Print the slope table, run one shard, merge shard files, or do both."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--slopes", action="store_true", help="print matched slopes")
    parser.add_argument("--shard", type=base._parse_shard, help="run shard i/N only")  # pyright: ignore[reportPrivateUsage]
    parser.add_argument("--merge", action="store_true", help="merge shard files")
    parser.add_argument("--quick", action="store_true", help="tiny grid for a draft")
    parser.add_argument("--replicates", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--unit-seconds", type=float, default=base.DEFAULT_UNIT_SECONDS)
    parser.add_argument("--shards-dir", type=Path, default=base.DEFAULT_SHARDS_DIR)
    parser.add_argument("--data-dir", type=Path, default=base.DEFAULT_DATA_DIR)
    args = parser.parse_args(argv)

    cells = build_cells()
    if args.slopes:
        print(slopes_frame(cells).to_string(index=False))
        return
    if args.quick:
        cells = [c for c in cells if c.mu in QUICK_MUS]
    replicates = args.replicates or (
        QUICK_REPLICATES if args.quick else base.DEFAULT_REPLICATES
    )
    block_size = args.block_size or (
        QUICK_BLOCK_SIZE if args.quick else base.DEFAULT_BLOCK_SIZE
    )

    if args.merge:
        rows = merge_shards(args.shards_dir, args.data_dir, cells)
        print(f"Merged {len(rows)} rows into {args.data_dir / ROWS_FILE}")
        return

    units = build_units(cells, replicates, block_size)
    if args.shard is not None:
        shard, n_shards = args.shard
        run_shard(units, shard, n_shards, args.shards_dir, args.unit_seconds)
        return

    for stale in args.shards_dir.glob("arley_*_0.*"):
        stale.unlink()
    run_shard(units, 0, 1, args.shards_dir, args.unit_seconds)
    rows = merge_shards(args.shards_dir, args.data_dir, cells)
    print(f"Wrote {len(rows)} rows to {args.data_dir / ROWS_FILE}")


if __name__ == "__main__":
    main()
