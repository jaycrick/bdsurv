r"""Count the random numbers each method draws, on the matched benchmark cells.

For every cell of ``bench_arley_comparison`` the script re-runs the contour
process and splitting, with the *same seeds* as the timed benchmarks, inside
``bdsurv.count_draws``, for both the constant death rate ``mu`` and the matched
Arley death rate.
Counted runs produce the same trees as uncounted ones and the counts are
deterministic, so this untimed pass reproduces exactly what the timed runs
drew without putting a counting wrapper in them.

Each row of the output has the draws by role (``contour``, ``dynamics``,
``seeding``) and kind, and the work counters: ``events``, ``clones``,
``records_copied``, ``attempts``, ``particle_steps``, ``contour_steps`` and
``spine_resets``.
Draws measure the stochastic work; ``records_copied`` and ``events`` track the
bookkeeping a draw count does not see.

The work is sharded like the timed benchmarks: shard ``i/N`` takes the runs
whose index is ``i`` modulo ``N``.

Usage
-----
    uv run python scripts/count_draws.py --replicates 5          # all in one go
    uv run python scripts/count_draws.py --shard 3/20 --replicates 20
    uv run python scripts/count_draws.py --merge
"""

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import bench_arley_comparison as arley
import bench_method_comparison as base
import pandas as pd

from bdsurv import count_draws

DEFAULT_REPLICATES = 20
DEFAULT_SHARDS_DIR = base.ROOT / "benchmarks" / "_shards_draws"
DEFAULT_DATA_DIR = base.DEFAULT_DATA_DIR
OUT_FILE = "draws.csv"


@dataclass(frozen=True)
class Run:
    """One counted run: a model, method, cell and replicate."""

    model: str  # "constant" or "arley"
    method: str
    mu: float
    time: float
    slope: float  # nan for the constant model
    replicate: int
    seed: int
    weight: float  # heavier first, so round-robin sharding balances load


def build_runs(replicates: int) -> list[Run]:
    """List every counted run on the feasible cells, heaviest cells first."""
    runs: list[Run] = []
    for cell_index, cell in enumerate(arley.build_cells()):
        if cell.feasible:
            for method in arley.METHODS:
                unit = arley.Unit(method, cell, cell_index, 0, replicates)
                runs += [
                    Run(
                        "arley",
                        method,
                        cell.mu,
                        cell.time,
                        cell.slope,
                        r,
                        arley.replicate_seed(unit, r),
                        cell.arley_value,
                    )
                    for r in range(replicates)
                ]
        if base.cell_is_feasible(cell.mu, cell.time):
            for method in base.METHODS:
                unit = base.Unit(
                    method,
                    cell.mu,
                    cell.time,
                    base.MUS.index(cell.mu),
                    base.TS.index(cell.time),
                    0,
                    replicates,
                )
                runs += [
                    Run(
                        "constant",
                        method,
                        cell.mu,
                        cell.time,
                        float("nan"),
                        r,
                        base.replicate_seed(unit, r),
                        cell.constant_value,
                    )
                    for r in range(replicates)
                ]
    runs.sort(key=lambda run: (-run.weight, run.model, run.method, run.replicate))
    return runs


def count_run(run: Run) -> dict[str, Any]:
    """Run once under the counter and return identifying fields, sizes and counts."""
    if run.model == "arley":
        cell = next(
            c
            for c in arley.build_cells()
            if c.mu == run.mu and c.time == run.time and c.feasible
        )
        unit: Any = arley.Unit(run.method, cell, 0, run.replicate, 1)
        runner = arley._run_contour if run.method == "contour" else arley._run_splitting  # pyright: ignore[reportPrivateUsage]
    else:
        unit = base.Unit(run.method, run.mu, run.time, 0, 0, run.replicate, 1)
        runner = base._run_contour if run.method == "contour" else base._run_splitting  # pyright: ignore[reportPrivateUsage]
    start = time.perf_counter()
    with count_draws() as counts:
        result = runner(unit, run.seed)
    return {
        "model": run.model,
        "method": run.method,
        "mu": run.mu,
        "T": run.time,
        "slope": run.slope,
        "replicate": run.replicate,
        "seed": run.seed,
        "seconds_counted": time.perf_counter() - start,  # wrapper adds overhead
        "n_trees": result["n_trees"],
        "n_individuals": result["n_individuals"],
        **counts.as_row(),
    }


def run_shard(runs: list[Run], shard: int, n_shards: int, out_dir: Path) -> None:
    """Count this shard's runs and write them."""
    out_dir.mkdir(parents=True, exist_ok=True)
    mine = [run for index, run in enumerate(runs) if index % n_shards == shard]
    rows: list[dict[str, Any]] = []
    for position, run in enumerate(mine, start=1):
        rows.append(count_run(run))
        print(
            f"[{shard}/{n_shards}] {position}/{len(mine)} {run.model} {run.method} "
            f"mu={run.mu:g} T={run.time:g} rep={run.replicate} "
            f"draws={rows[-1]['draws_total']}",
            flush=True,
        )
    pd.DataFrame(rows).to_csv(out_dir / f"draws_rows_{shard}.csv", index=False)


def merge_shards(shards_dir: Path, data_dir: Path) -> pd.DataFrame:
    """Concatenate shard files into ``draws.csv`` and return the rows."""
    files = sorted(shards_dir.glob("draws_rows_*.csv"))
    if not files:
        msg = f"No shard files found in {shards_dir}"
        raise FileNotFoundError(msg)
    rows = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    rows = rows.sort_values(["model", "method", "mu", "replicate"])
    data_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(data_dir / OUT_FILE, index=False)
    return rows


def main(argv: list[str] | None = None) -> None:
    """Count draws for one shard, merge shard files, or (default) do both."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--shard", type=base._parse_shard, help="run shard i/N only")  # pyright: ignore[reportPrivateUsage]
    parser.add_argument("--merge", action="store_true", help="merge shard files")
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--shards-dir", type=Path, default=DEFAULT_SHARDS_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args(argv)

    if args.merge:
        rows = merge_shards(args.shards_dir, args.data_dir)
        print(f"Merged {len(rows)} rows into {args.data_dir / OUT_FILE}")
        return
    runs = build_runs(args.replicates)
    if args.shard is not None:
        shard, n_shards = args.shard
        run_shard(runs, shard, n_shards, args.shards_dir)
        return
    for stale in args.shards_dir.glob("draws_rows_0.csv"):
        stale.unlink()
    run_shard(runs, 0, 1, args.shards_dir)
    rows = merge_shards(args.shards_dir, args.data_dir)
    print(f"Wrote {len(rows)} rows to {args.data_dir / OUT_FILE}")


if __name__ == "__main__":
    main()
