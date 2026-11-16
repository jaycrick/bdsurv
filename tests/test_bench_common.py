"""Tests for the helpers shared by the benchmark and chart scripts."""

import argparse
import json
from pathlib import Path
from typing import Any

import _bench_common as common
import numpy as np
import pandas as pd
import pytest


class TestReplicateSeed:
    """The seed depends on the base, the cell keys, the method and the replicate."""

    def test_matches_the_seed_sequence_rule(self) -> None:
        expected = int(
            np.random.SeedSequence([7, 1500, 3000, 1, 4]).generate_state(1)[0]
            % (2**31 - 1)
        )
        assert common.replicate_seed(7, [1500, 3000], "splitting", 4) == expected

    def test_each_argument_matters(self) -> None:
        seed = common.replicate_seed(7, [1500, 3000], "contour", 4)
        assert seed == common.replicate_seed(7, [1500, 3000], "contour", 4)
        assert seed != common.replicate_seed(8, [1500, 3000], "contour", 4)
        assert seed != common.replicate_seed(7, [1500, 3001], "contour", 4)
        assert seed != common.replicate_seed(7, [1500, 3000], "splitting", 4)
        assert seed != common.replicate_seed(7, [1500, 3000], "contour", 5)

    def test_unknown_method_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="not in tuple"):
            common.replicate_seed(7, [1], "gillespie", 0)


class TestShard:
    """Sharding deals items out by index modulo the number of shards."""

    def test_modulo_selection(self) -> None:
        items = list("abcdefg")
        assert common.shard(items, 0, 3) == ["a", "d", "g"]
        assert common.shard(items, 2, 3) == ["c", "f"]

    def test_shards_partition_the_items(self) -> None:
        items = list(range(23))
        pieces = [common.shard(items, i, 5) for i in range(5)]
        assert sorted(x for piece in pieces for x in piece) == items

    def test_parse_shard(self) -> None:
        assert common.parse_shard("3/10") == (3, 10)
        for bad in ("10/10", "0/0", "-1/4"):
            with pytest.raises(argparse.ArgumentTypeError):
                common.parse_shard(bad)


class TestRunConditioned:
    """One replicate of either method gives the same columns."""

    COLUMNS = (
        "seconds",
        "n_trees",
        "n_attempts",
        "levels",
        "pop_at_T",
        "n_individuals",
    )

    @pytest.mark.parametrize("method", ["contour", "splitting"])
    def test_columns_and_seed_determinism(self, method: str) -> None:
        a = common.run_conditioned(method, 1.0, 1.5, 3.0, 11)
        b = common.run_conditioned(method, 1.0, 1.5, 3.0, 11)
        assert tuple(a) == self.COLUMNS
        assert a["seconds"] > 0
        assert a["pop_at_T"] >= 1
        for column in self.COLUMNS[1:]:
            assert np.array_equal(a[column], b[column], equal_nan=True)

    def test_contour_has_one_tree_and_no_attempts(self) -> None:
        row = common.run_conditioned("contour", 1.0, 1.5, 3.0, 11)
        assert row["n_trees"] == 1
        assert np.isnan(row["n_attempts"])
        assert np.isnan(row["levels"])

    def test_splitting_reports_levels(self) -> None:
        row = common.run_conditioned("splitting", 1.0, 1.5, 3.0, 11)
        assert row["n_trees"] >= 1
        assert row["levels"] >= 1
        assert row["n_attempts"] >= 1

    def test_unknown_method_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="method must be one of"):
            common.run_conditioned("gillespie", 1.0, 1.5, 3.0, 11)


class TestRunUnit:
    """The generic block runner keeps identity columns first and honours the budget."""

    @staticmethod
    def _run(seed: int) -> dict[str, Any]:
        return {"seconds": 0.5, "value": seed * 2}

    def test_rows_and_summary(self) -> None:
        rows, summary = common.run_unit(
            {"method": "contour", "mu": 1.5},
            range(4, 7),
            lambda replicate: 100 + replicate,
            self._run,
            10.0,
            "h",
        )
        assert [r["replicate"] for r in rows] == [4, 5, 6]
        assert list(rows[0]) == [
            "method",
            "mu",
            "replicate",
            "seed",
            "host",
            "seconds",
            "value",
        ]
        assert [r["seed"] for r in rows] == [104, 105, 106]
        assert summary == {
            "method": "contour",
            "mu": 1.5,
            "first_replicate": 4,
            "replicates_requested": 3,
            "replicates_done": 3,
            "seconds": 1.5,
            "status": "ok",
            "host": "h",
        }

    def test_budget_stops_the_block_after_the_budget_is_exceeded(self) -> None:
        # Each replicate costs 0.5 s; a replicate starts while spent <= budget.
        rows, summary = common.run_unit(
            {"method": "contour"}, range(5), lambda r: r, self._run, 0.9, "h"
        )
        assert len(rows) == 2
        assert summary["status"] == "budget"
        assert summary["replicates_done"] == 2
        assert summary["replicates_requested"] == 5


def _describe(item: int, rows: list[dict[str, Any]], summary: Any) -> str:
    return f"item {item} rows={len(rows)} summary={summary is not None}"


def _work(item: int, host: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [{"key": item, "part": p, "host": host} for p in range(2)]
    return rows, {"key": item, "status": "ok", "host": host}


class TestShardFilesAndMerge:
    """Shard files and merged files for a script with summaries and one without."""

    def test_round_trip_with_summaries(self, tmp_path: Path) -> None:
        shards, data = tmp_path / "s", tmp_path / "d"
        files = common.ShardFiles("x_", _describe)
        for index in range(2):
            common.run_shard([5, 4, 3, 2, 1], index, 2, shards, _work, files)
        assert sorted(p.name for p in shards.iterdir()) == [
            "x_env_0.json",
            "x_env_1.json",
            "x_rows_0.csv",
            "x_rows_1.csv",
            "x_units_0.csv",
            "x_units_1.csv",
        ]
        spec = common.MergeSpec(
            "x_", ["key", "part"], "rows.csv", "units.csv", "env.json", {"lambda": 1.0}
        )
        rows = common.merge_shards(shards, data, spec)
        assert list(rows["key"]) == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
        assert list(pd.read_csv(data / "rows.csv").columns) == ["key", "part", "host"]
        assert len(pd.read_csv(data / "units.csv")) == 5
        summary = json.loads((data / "env.json").read_text())
        assert list(summary)[:2] == ["lambda", "n_shards"]
        assert summary["n_shards"] == 2

    def test_rows_only(self, tmp_path: Path) -> None:
        shards, data = tmp_path / "s", tmp_path / "d"
        files = common.ShardFiles("y_", _describe, summaries=False)
        common.run_shard(
            [2, 1], 0, 1, shards, lambda item, _host: ([{"key": item}], None), files
        )
        assert [p.name for p in shards.iterdir()] == ["y_rows_0.csv"]
        rows = common.merge_shards(
            shards, data, common.MergeSpec("y_", ["key"], "y.csv")
        )
        assert list(rows["key"]) == [1, 2]
        assert [p.name for p in data.iterdir()] == ["y.csv"]

    def test_merge_without_shards_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No shard files"):
            common.merge_shards(
                tmp_path / "none", tmp_path / "d", common.MergeSpec("", ["a"], "r.csv")
            )

    def test_prefixes_do_not_mix(self, tmp_path: Path) -> None:
        shards = tmp_path / "s"
        for prefix in ("a_", "a_grid_"):
            common.run_shard(
                [1], 0, 1, shards, _work, common.ShardFiles(prefix, _describe)
            )
        rows = common.merge_shards(
            shards, tmp_path / "d", common.MergeSpec("a_", ["key"], "r.csv")
        )
        assert len(rows) == 2


class TestEnvironmentRecord:
    """The environment record names the machine and the package versions."""

    def test_keys(self) -> None:
        record = common.environment_record()
        assert {"host", "python", "numpy", "pandas", "bdsurv", "git_sha"} <= set(record)


class TestChartsCommon:
    """The shared chart code gives each caller its own grouping and options."""

    @staticmethod
    def _rows(group_extra: bool) -> pd.DataFrame:
        records = [
            {
                "method": method,
                "mu": 3.0,
                "slope": 0.06,
                "T": t,
                "replicate": r,
                "seconds": 0.1 * (r + 1) * t,
                "draws_total": 10 * (r + 1),
                "n_trees": 1 if method == "contour" else 5,
            }
            for method in ("contour", "splitting")
            for t in (1.0, 10.0)
            for r in range(4)
        ]
        frame = pd.DataFrame(records)
        return frame if group_extra else frame.drop(columns="slope")

    def test_band_groups_by_the_given_columns(self) -> None:
        charts = pytest.importorskip("_charts_common")
        rows = self._rows(group_extra=True)
        out = charts.band(rows, "seconds", "label", ["mu", "slope", "T"])
        assert list(out.columns[:3]) == ["mu", "slope", "T"]
        assert len(out) == 2
        assert set(out["replicates"]) == {8}
        out = charts.band(rows, "seconds", "label", ["mu", "T"])
        assert "slope" not in out.columns
        assert len(out) == 2

    def test_curves_have_the_three_labels(self) -> None:
        charts = pytest.importorskip("_charts_common")
        for group, extra in ((["mu", "T"], False), (["mu", "slope", "T"], True)):
            rows = self._rows(group_extra=extra)
            for curves in (
                charts.curves_from_rows(rows, group),
                charts.curves_from_draws(rows, group),
            ):
                assert set(curves["method"]) == {
                    charts.LABEL_CONTOUR,
                    charts.LABEL_RUN,
                    charts.LABEL_KEPT,
                }

    def test_cost_chart_options_and_overrides(self) -> None:
        charts = pytest.importorskip("_charts_common")
        curves = charts.curves_from_rows(self._rows(group_extra=False), ["mu", "T"])
        x = charts.x_by_t([1.0, 10.0])
        spec = charts.cost_chart(curves, x, statistic="mean", show_band=False).to_dict()
        assert len(spec["layer"]) == 2
        options = charts.CostChartOptions(mark_critical=True, lam=1.0, width=300)
        spec = charts.cost_chart(curves, x, options, height=120).to_dict()
        assert len(spec["layer"]) == 4
        assert spec["width"] == 300
        assert spec["height"] == 120

    def test_cost_chart_rejects_unknown_options(self) -> None:
        charts = pytest.importorskip("_charts_common")
        curves = charts.curves_from_rows(self._rows(group_extra=False), ["mu", "T"])
        with pytest.raises(TypeError):
            charts.cost_chart(curves, charts.x_by_t([1.0]), colour="red")
        with pytest.raises(ValueError, match="lam"):
            charts.cost_chart(curves, charts.x_by_t([1.0]), mark_critical=True)

    def test_theme_is_registered_once(self) -> None:
        alt = pytest.importorskip("altair")
        charts = pytest.importorskip("_charts_common")
        charts.register_theme()
        charts.register_theme()
        assert alt.theme.names().count("bdsurv") == 1
