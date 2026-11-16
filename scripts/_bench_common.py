"""Machinery shared by the benchmark scripts.

``bench_method_comparison.py``, ``bench_arley_comparison.py`` and
``count_draws.py`` split their work into independent items, deal them out to
shards by index modulo the number of shards, write each shard's files, and merge
the shard files into the committed data files.
This module holds that machinery once, together with the single replicate runner
that both timed benchmarks call and the seed rule they share.

It needs only ``bdsurv``, NumPy and pandas, so a cluster node running the
benchmarks does not need the plotting dependencies.
Each script keeps what is specific to it: its grid, its units, its seed keys and
its file names.
"""

import argparse
import json
import os
import platform
import socket
import subprocess
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
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
    RateInput,
    SplittingConfig,
    simulate_contour_given_survival,
    simulate_survival_splitting,
)

ROOT = Path(__file__).resolve().parent.parent
METHODS = ("contour", "splitting")
N_PARTICLES = 1000
# Far above ``default_max_restarts``, whose independence assumption can be
# optimistic (clones of one parent share a state); the unit budget bounds time.
MAX_RESTARTS = 5000

type Row = dict[str, Any]


def replicate_seed(
    base: int, key_ints: Sequence[int], method: str, replicate: int
) -> int:
    """Return a seed that depends only on the cell keys, the method and the replicate.

    The cell enters by value, not by position in a grid, so adding grid points
    never changes the seeds of the cells that remain.

    Parameters
    ----------
    base
        Seed base of the benchmark.
    key_ints
        The cell values already rounded to integers (for example ``mu`` and ``T``
        in thousandths); each benchmark chooses its own rounding.
    method
        One of ``METHODS``.
    replicate
        Replicate number within the cell.

    Returns
    -------
    int
        A seed in ``[0, 2**31 - 2]``.
    """
    sequence = np.random.SeedSequence(
        [base, *key_ints, METHODS.index(method), replicate]
    )
    return int(sequence.generate_state(1)[0] % (2**31 - 1))


def run_conditioned(
    method: str,
    birth: RateInput,
    death: RateInput,
    terminal_time: float | None,
    seed: int,
) -> Row:
    """Run one replicate of a method and return its timing and size columns.

    ``"contour"`` draws one exact tree with ``simulate_contour_given_survival``.
    ``"splitting"`` runs ``simulate_survival_splitting`` with ``N_PARTICLES``
    particles and ``MAX_RESTARTS`` restarts, and measures one tree chosen at
    random from those it kept, using ``numpy.random.default_rng(seed)``, so the
    sizes are independent across runs.

    Parameters
    ----------
    method
        ``"contour"`` or ``"splitting"``.
    birth, death
        Rates: floats (with ``terminal_time``) or rate objects.
    terminal_time
        The horizon ``T``; ``None`` when the rates are rate objects.
    seed
        Random seed of the run.

    Returns
    -------
    dict
        ``seconds``, ``n_trees``, ``n_attempts``, ``levels``, ``pop_at_T`` and
        ``n_individuals`` (NaN where the method has no such quantity).

    Raises
    ------
    ValueError
        If ``method`` is not in ``METHODS``.
    """
    if method == "contour":
        start = time.perf_counter()
        tree = simulate_contour_given_survival(
            birth, death, terminal_time=terminal_time, random_seed=seed
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
    if method != "splitting":
        msg = f"method must be one of {METHODS}, got {method!r}"
        raise ValueError(msg)
    config = SplittingConfig(
        birth_rate=birth,
        death_rate=death,
        n_particles=N_PARTICLES,
        terminal_time=terminal_time,
        random_seed=seed,
        max_restarts=MAX_RESTARTS,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        start = time.perf_counter()
        result = simulate_survival_splitting(config)
        seconds = time.perf_counter() - start
    kept = result.conditioned_trajectories
    row: Row = {
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
    identity: Mapping[str, Any],
    replicates: range,
    seed_for: Callable[[int], int],
    run: Callable[[int], Row],
    unit_seconds: float,
    host: str,
) -> tuple[list[Row], Row]:
    """Run one block of replicates; stop starting replicates once the budget is spent.

    Parameters
    ----------
    identity
        Columns that identify the unit (``method``, ``mu``, ``T`` and so on),
        written first in every row and in the summary.
    replicates
        The replicate numbers of the block.
    seed_for
        Maps a replicate number to its seed.
    run
        Runs one replicate from its seed and returns a row with a ``seconds`` entry.
    unit_seconds
        Time budget: no replicate starts once the time spent exceeds it.
    host
        Host name written to every row and to the summary.

    Returns
    -------
    rows : list of dict
        One row per replicate run.
    summary : dict
        One row describing the block, with ``status`` ``"ok"`` or ``"budget"``.
    """
    rows: list[Row] = []
    spent = 0.0
    status = "ok"
    for replicate in replicates:
        if spent > unit_seconds:
            status = "budget"
            break
        seed = seed_for(replicate)
        row = run(seed)
        spent += float(row["seconds"])
        rows.append(
            {**identity, "replicate": replicate, "seed": seed, "host": host, **row}
        )
    summary = {
        **identity,
        "first_replicate": replicates.start,
        "replicates_requested": len(replicates),
        "replicates_done": len(rows),
        "seconds": spent,
        "status": status,
        "host": host,
    }
    return rows, summary


def shard[T](items: Sequence[T], shard_index: int, n_shards: int) -> list[T]:
    """Return the items whose index is ``shard_index`` modulo ``n_shards``."""
    return [item for index, item in enumerate(items) if index % n_shards == shard_index]


def parse_shard(text: str) -> tuple[int, int]:
    """Parse the ``--shard i/N`` option into ``(i, N)``.

    Raises
    ------
    argparse.ArgumentTypeError
        If ``text`` is not ``i/N`` with ``0 <= i < N``.
    """
    index_text, _, total = text.partition("/")
    index, count = int(index_text), int(total)
    if count < 1 or not 0 <= index < count:
        msg = f"--shard must look like i/N with 0 <= i < N, got {text!r}"
        raise argparse.ArgumentTypeError(msg)
    return index, count


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


def environment_record() -> Row:
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


@dataclass(frozen=True)
class ShardFiles:
    """How one script names and reports its shard files.

    Parameters
    ----------
    prefix
        Shard files are ``<prefix>rows_<i>.csv`` and, when ``summaries`` is true,
        ``<prefix>units_<i>.csv`` and ``<prefix>env_<i>.json``.
    describe
        Text of the progress line for one item, from the item, its rows and its
        summary.
    summaries
        Whether the shard also writes unit summaries and the environment record.
    """

    prefix: str
    describe: Callable[[Any, list[Row], Row | None], str]
    summaries: bool = True


def run_shard[T](
    items: Sequence[T],
    shard_index: int,
    n_shards: int,
    out_dir: Path,
    work: Callable[[T, str], tuple[list[Row], Row | None]],
    files: ShardFiles,
) -> None:
    """Run this shard's items and write its rows, unit summaries and environment.

    Parameters
    ----------
    items
        Every item of the run, in the order that balances load across shards.
    shard_index, n_shards
        This shard takes the items whose index is ``shard_index`` modulo ``n_shards``.
    out_dir
        Directory for the shard files; created if missing.
    work
        Runs one item given the host name and returns its rows and, if the
        script keeps unit summaries, its summary row.
    files
        File naming and progress reporting.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    env = environment_record() if files.summaries else None
    host = env["host"] if env is not None else ""
    mine = shard(items, shard_index, n_shards)
    rows: list[Row] = []
    summaries: list[Row] = []
    for position, item in enumerate(mine, start=1):
        item_rows, summary = work(item, host)
        rows += item_rows
        if summary is not None:
            summaries.append(summary)
        print(
            f"[{shard_index}/{n_shards}] {position}/{len(mine)} "
            f"{files.describe(item, item_rows, summary)}",
            flush=True,
        )
    prefix = files.prefix
    pd.DataFrame(rows).to_csv(out_dir / f"{prefix}rows_{shard_index}.csv", index=False)
    if env is not None:
        pd.DataFrame(summaries).to_csv(
            out_dir / f"{prefix}units_{shard_index}.csv", index=False
        )
        (out_dir / f"{prefix}env_{shard_index}.json").write_text(
            json.dumps(env, indent=2) + "\n"
        )


@dataclass(frozen=True)
class MergeSpec:
    """The shard files a script merges and the data files it writes.

    Parameters
    ----------
    prefix
        Prefix of the shard files, as in ``ShardFiles``.
    sort_keys
        Columns that order the merged rows.
    rows_file
        Name of the merged rows file in the data directory.
    units_file, env_file
        Names of the merged unit summaries and environment summary; ``None`` for a
        script that writes rows only.
    header
        Entries written first in the environment summary (the benchmark's own
        constants).
    """

    prefix: str
    sort_keys: Sequence[str]
    rows_file: str
    units_file: str | None = None
    env_file: str | None = None
    header: Mapping[str, Any] = field(default_factory=dict)


def merge_shards(shards_dir: Path, data_dir: Path, spec: MergeSpec) -> pd.DataFrame:
    """Concatenate shard files into the committed data files and return the rows.

    Parameters
    ----------
    shards_dir
        Directory holding the shard files.
    data_dir
        Directory for the merged files; created if missing.
    spec
        Which files to read and write.

    Returns
    -------
    pandas.DataFrame
        The merged rows, sorted by ``spec.sort_keys``.

    Raises
    ------
    FileNotFoundError
        If ``shards_dir`` holds no rows file.
    """
    prefix = spec.prefix
    row_files = sorted(shards_dir.glob(f"{prefix}rows_*.csv"))
    if not row_files:
        msg = f"No shard files found in {shards_dir}"
        raise FileNotFoundError(msg)
    rows = pd.concat([pd.read_csv(f) for f in row_files], ignore_index=True)
    rows = rows.sort_values(list(spec.sort_keys)).reset_index(drop=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(data_dir / spec.rows_file, index=False)
    if spec.units_file is not None and spec.env_file is not None:
        units = pd.concat(
            [pd.read_csv(f) for f in sorted(shards_dir.glob(f"{prefix}units_*.csv"))],
            ignore_index=True,
        )
        units.to_csv(data_dir / spec.units_file, index=False)
        envs = [
            json.loads(f.read_text())
            for f in sorted(shards_dir.glob(f"{prefix}env_*.json"))
        ]
        summary = {
            **spec.header,
            "n_shards": len(envs),
            "hosts": sorted({e["host"] for e in envs}),
            "processors": sorted({e["processor"] for e in envs}),
            "python": sorted({e["python"] for e in envs}),
            "numpy": sorted({e["numpy"] for e in envs}),
            "bdsurv": sorted({e["bdsurv"] for e in envs}),
            "git_sha": sorted({e["git_sha"] for e in envs}),
        }
        (data_dir / spec.env_file).write_text(json.dumps(summary, indent=2) + "\n")
    return rows
