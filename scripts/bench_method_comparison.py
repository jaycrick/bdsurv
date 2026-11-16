r"""Benchmark the contour process against multilevel splitting on a (mu, T) grid.

Birth rate is fixed at ``lambda = 1``.
For every death rate ``mu`` and horizon ``T`` the script times independent
replicates of

``contour``
    ``simulate_birth_death_process_from_contour_process``: one exact tree per
    call.
``splitting``
    ``simulate_subcritical_survival_splitting``: one run returns several kept
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
import json
import math
import os
import platform
import socket
import subprocess
import time
import warnings
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

# Timings should be single-threaded and CPU-only, whatever the node offers.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pandas as pd

from bdsurv import (
    SplittingConfig,
    simulate_birth_death_process_from_contour_process,
    simulate_subcritical_survival_splitting,
)
from bdsurv._hazard import log_survival_probability_constant

LAMBDA = 1.0
MUS = (
    0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0,
    2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0,
)  # fmt: skip
TS = (
    1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0,
    12.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0,
)  # fmt: skip
METHODS = ("contour", "splitting")
MAX_EXPECTED_POPULATION = 1e5
N_PARTICLES = 1000
# Far above ``default_max_restarts``, whose independence assumption can be
# optimistic (clones of one parent share a state); the unit budget bounds time.
MAX_RESTARTS = 5000
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

ROOT = Path(__file__).resolve().parent.parent
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
    log_pr = log_survival_probability_constant(
        birth_rate=LAMBDA, death_rate=mu, time=terminal_time
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
                for first in range(0, replicates, block_size):
                    units.append(
                        Unit(
                            method=method,
                            mu=mu,
                            time=terminal_time,
                            mu_index=mu_index,
                            time_index=time_index,
                            first_replicate=first,
                            n_replicates=min(block_size, replicates - first),
                        )
                    )
    units.sort(key=lambda u: (-u.time, -u.mu, u.method, u.first_replicate))
    return units


def shard_units(units: list[Unit], shard: int, n_shards: int) -> list[Unit]:
    """Return the units belonging to shard ``shard`` of ``n_shards``."""
    return [unit for index, unit in enumerate(units) if index % n_shards == shard]


def replicate_seed(unit: Unit, replicate: int) -> int:
    """Return a seed depending only on the cell values, method and replicate.

    The cell enters by value (``mu`` and ``T`` in thousandths), not by position in
    the grid, so adding grid points never changes the seeds of existing cells.
    """
    sequence = np.random.SeedSequence(
        [
            BASE_SEED,
            round(unit.mu * 1000),
            round(unit.time * 1000),
            METHODS.index(unit.method),
            replicate,
        ]
    )
    return int(sequence.generate_state(1)[0] % (2**31 - 1))


def _run_contour(unit: Unit, seed: int) -> dict[str, Any]:
    start = time.perf_counter()
    tree = simulate_birth_death_process_from_contour_process(
        birth_rate=LAMBDA,
        death_rate=unit.mu,
        terminal_time=unit.time,
        random_seed=seed,
    )
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
    config = SplittingConfig(
        birth_rate=LAMBDA,
        death_rate=unit.mu,
        terminal_time=unit.time,
        n_particles=N_PARTICLES,
        random_seed=seed,
        max_restarts=MAX_RESTARTS,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        start = time.perf_counter()
        result = simulate_subcritical_survival_splitting(config)
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
        # One random tree per run, so these draws are independent across runs.
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
                "method": unit.method,
                "mu": unit.mu,
                "T": unit.time,
                "replicate": replicate,
                "seed": seed,
                "host": host,
                **row,
            }
        )
    summary = {
        "method": unit.method,
        "mu": unit.mu,
        "T": unit.time,
        "first_replicate": unit.first_replicate,
        "replicates_requested": unit.n_replicates,
        "replicates_done": len(rows),
        "seconds": spent,
        "status": status,
        "host": host,
    }
    return rows, summary


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=ROOT,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def environment_record() -> dict[str, Any]:
    """Describe the machine and package versions running this shard."""
    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "bdsurv": _package_version("bdsurv"),
        "git_sha": _git_sha(),
        "slurm_job": os.environ.get("SLURM_ARRAY_JOB_ID", ""),
    }


def run_shard(
    units: list[Unit], shard: int, n_shards: int, out_dir: Path, unit_seconds: float
) -> None:
    """Run this shard's units and write its rows, unit summaries and environment."""
    out_dir.mkdir(parents=True, exist_ok=True)
    env = environment_record()
    mine = shard_units(units, shard, n_shards)
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for position, unit in enumerate(mine, start=1):
        unit_rows, summary = run_unit(unit, unit_seconds, env["host"])
        rows += unit_rows
        summaries.append(summary)
        print(
            f"[{shard}/{n_shards}] unit {position}/{len(mine)} "
            f"{unit.method} mu={unit.mu:g} T={unit.time:g} "
            f"block={unit.first_replicate} {summary['status']} "
            f"{summary['seconds']:.2f}s",
            flush=True,
        )
    pd.DataFrame(rows).to_csv(out_dir / f"rows_{shard}.csv", index=False)
    pd.DataFrame(summaries).to_csv(out_dir / f"units_{shard}.csv", index=False)
    (out_dir / f"env_{shard}.json").write_text(json.dumps(env, indent=2) + "\n")


def merge_shards(shards_dir: Path, data_dir: Path) -> pd.DataFrame:
    """Concatenate shard files into the committed data files and return the rows."""
    row_files = sorted(shards_dir.glob("rows_*.csv"))
    if not row_files:
        msg = f"No shard files found in {shards_dir}"
        raise FileNotFoundError(msg)
    rows = pd.concat([pd.read_csv(f) for f in row_files], ignore_index=True)
    rows = rows.sort_values(["method", "mu", "T", "replicate"]).reset_index(drop=True)
    units = pd.concat(
        [pd.read_csv(f) for f in sorted(shards_dir.glob("units_*.csv"))],
        ignore_index=True,
    )
    envs = [json.loads(f.read_text()) for f in sorted(shards_dir.glob("env_*.json"))]
    data_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(data_dir / ROWS_FILE, index=False)
    units.to_csv(data_dir / UNITS_FILE, index=False)
    summary = {
        "lambda": LAMBDA,
        "max_expected_population": MAX_EXPECTED_POPULATION,
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


def _parse_shard(text: str) -> tuple[int, int]:
    shard, _, total = text.partition("/")
    index, count = int(shard), int(total)
    if count < 1 or not 0 <= index < count:
        msg = f"--shard must look like i/N with 0 <= i < N, got {text!r}"
        raise argparse.ArgumentTypeError(msg)
    return index, count


def main(argv: list[str] | None = None) -> None:
    """Run one shard, merge shard files, or (default) do both in one process."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--shard", type=_parse_shard, help="run shard i/N only")
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
