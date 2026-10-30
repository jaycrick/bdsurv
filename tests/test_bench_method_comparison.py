"""Tests for the method-comparison benchmark engine."""

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPT = (
    Path(__file__).resolve().parent.parent / "scripts" / "bench_method_comparison.py"
)
_spec = importlib.util.spec_from_file_location("bench_method_comparison", _SCRIPT)
assert _spec is not None
assert _spec.loader is not None
bench = importlib.util.module_from_spec(_spec)
sys.modules["bench_method_comparison"] = bench
_spec.loader.exec_module(bench)

DETERMINISTIC_COLUMNS = [
    "method",
    "mu",
    "T",
    "replicate",
    "seed",
    "n_trees",
    "n_attempts",
    "levels",
    "pop_at_T",
    "n_individuals",
]


class TestFeasibleCells:
    """Cells whose trees would be too large are skipped."""

    def test_critical_expected_population(self) -> None:
        # lambda = mu: E[Z_T | Z_T > 0] = 1 / Pr(T) = 1 + T.
        assert math.isclose(bench.expected_population(1.0, 50.0), 51.0)

    def test_supercritical_long_horizon_is_skipped(self) -> None:
        assert not bench.cell_is_feasible(0.6, 100.0)
        assert bench.cell_is_feasible(0.6, 10.0)

    def test_subcritical_cells_are_kept(self) -> None:
        assert bench.cell_is_feasible(10.0, 100.0)
        assert bench.cell_is_feasible(1.0, 100.0)

    def test_every_unit_is_in_a_feasible_cell(self) -> None:
        units = bench.build_units(bench.MUS, bench.TS, 4, 2)
        assert units
        assert all(bench.cell_is_feasible(u.mu, u.time) for u in units)


class TestUnitsAndShards:
    """Units cover every replicate once and shard without overlap."""

    def test_replicates_covered_exactly_once(self) -> None:
        units = bench.build_units((1.5,), (1.0, 3.0), 7, 3)
        for method in bench.METHODS:
            for t in (1.0, 3.0):
                covered = sorted(
                    r
                    for u in units
                    if u.method == method and u.time == t
                    for r in range(
                        u.first_replicate, u.first_replicate + u.n_replicates
                    )
                )
                assert covered == list(range(7))

    def test_shards_partition_units(self) -> None:
        units = bench.build_units(bench.MUS, bench.TS, 6, 3)
        n_shards = 7
        pieces = [bench.shard_units(units, i, n_shards) for i in range(n_shards)]
        flat = [u for piece in pieces for u in piece]
        assert len(flat) == len(units)
        assert set(flat) == set(units)

    def test_heaviest_units_come_first(self) -> None:
        units = bench.build_units(bench.MUS, bench.TS, 2, 2)
        assert units[0].time == max(bench.TS)

    def test_seed_depends_only_on_cell_method_and_replicate(self) -> None:
        a = bench.Unit("contour", 1.5, 3.0, 2, 1, 0, 2)
        same = bench.Unit("contour", 1.5, 3.0, 2, 1, 4, 2)
        other_method = bench.Unit("splitting", 1.5, 3.0, 2, 1, 0, 2)
        assert bench.replicate_seed(a, 5) == bench.replicate_seed(same, 5)
        assert bench.replicate_seed(a, 5) != bench.replicate_seed(a, 6)
        assert bench.replicate_seed(a, 5) != bench.replicate_seed(other_method, 5)

    def test_parse_shard(self) -> None:
        assert bench._parse_shard("3/10") == (3, 10)
        with pytest.raises(argparse.ArgumentTypeError):
            bench._parse_shard("10/10")


class TestRunUnit:
    """One unit produces raw rows and a summary."""

    @pytest.mark.parametrize("method", ["contour", "splitting"])
    def test_rows_and_summary(self, method: str) -> None:
        unit = bench.Unit(method, 1.5, 3.0, 0, 0, 0, 2)
        rows, summary = bench.run_unit(unit, 100.0, "testhost")
        assert len(rows) == 2
        assert summary["replicates_done"] == 2
        assert summary["status"] == "ok"
        for row in rows:
            assert row["seconds"] > 0
            assert row["n_trees"] >= 1
            assert row["pop_at_T"] >= 1
            assert row["host"] == "testhost"

    def test_budget_stops_the_block(self) -> None:
        unit = bench.Unit("contour", 1.5, 3.0, 0, 0, 0, 5)
        rows, summary = bench.run_unit(unit, -1.0, "testhost")
        assert rows == []
        assert summary["status"] == "budget"
        assert summary["replicates_done"] == 0


class TestSharding:
    """Sharded and merged output matches an unsharded run."""

    def test_merge_of_shards_equals_unsharded(self, tmp_path: Path) -> None:
        whole = tmp_path / "whole"
        split = tmp_path / "split"
        args = ["--quick", "--replicates", "2", "--block-size", "1"]
        bench.main(
            [*args, "--shards-dir", str(whole / "s"), "--data-dir", str(whole / "d")]
        )
        for shard in range(3):
            bench.main(
                [*args, "--shard", f"{shard}/3", "--shards-dir", str(split / "s")]
            )
        bench.main(
            [
                *args,
                "--merge",
                "--shards-dir",
                str(split / "s"),
                "--data-dir",
                str(split / "d"),
            ]
        )

        a = pd.read_csv(whole / "d" / bench.ROWS_FILE)
        b = pd.read_csv(split / "d" / bench.ROWS_FILE)
        assert len(a) == len(b) > 0
        pd.testing.assert_frame_equal(
            a[DETERMINISTIC_COLUMNS], b[DETERMINISTIC_COLUMNS]
        )
        assert (split / "d" / bench.ENV_FILE).exists()
        assert not np.isnan(b["seconds"]).any()

    def test_merge_without_shards_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            bench.merge_shards(tmp_path / "empty", tmp_path / "data")
