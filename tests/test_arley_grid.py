"""Tests for the Arley benchmark's slope-by-time grid mode and its charts."""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


arley = _load("bench_arley_comparison")


@pytest.fixture(scope="module")
def matched_cells():
    """Return the cumulative-matched cells (the slopes take a few seconds to compute)."""
    return arley.build_cells()


@pytest.fixture(scope="module")
def grid_cells():
    """Return every grid slope over the whole T grid."""
    return arley.build_grid_cells()


class TestGridCells:
    def test_every_slope_over_the_whole_grid(self, grid_cells) -> None:
        cells = grid_cells
        assert len(cells) == len(arley.GRID_SLOPES) * len(arley.base.TS)
        assert {c.matched_on for c in cells} == {"mean-rate"}
        assert {round(c.time, 3) for c in cells} == set(arley.base.TS)
        assert {c.slope for c in cells} == set(arley.GRID_SLOPES)

    def test_mu_is_the_constant_rate_with_the_same_mean_death_rate(
        self, grid_cells
    ) -> None:
        # mean of m t over [0, SURVIVAL_T] is m * SURVIVAL_T / 2.
        for c in grid_cells:
            assert c.mu == pytest.approx(c.slope * arley.SURVIVAL_T / 2)

    def test_survival_matches_the_constant_rate_at_the_matching_time(
        self, grid_cells
    ) -> None:
        from bdsurv.splitting import _log_survival_probability

        at_t = [c for c in grid_cells if c.time == arley.SURVIVAL_T]
        assert len(at_t) == len(arley.GRID_SLOPES)
        for c in at_t:
            constant = -_log_survival_probability(
                birth_rate=arley.LAMBDA, death_rate=c.mu, time=c.time
            ) / math.log(10)
            assert c.arley_value == pytest.approx(constant, rel=3e-3)

    def test_every_grid_cell_is_under_the_size_cap(self, grid_cells) -> None:
        assert all(c.feasible for c in grid_cells)
        assert max(c.log10_mean_cumulative for c in grid_cells) < 5.0

    def test_grid_slopes_are_in_the_grid_and_cover_mu_up_to_20(
        self, grid_cells
    ) -> None:
        assert max(c.mu for c in grid_cells) == pytest.approx(20.0)
        assert min(c.mu for c in grid_cells) == pytest.approx(3.0)

    def test_matched_cells_are_cumulative_and_feasible(self, matched_cells) -> None:
        assert {(c.mu, c.time) for c in matched_cells} == set(arley.CUMULATIVE_CELLS)
        assert {c.matched_on for c in matched_cells} == {"cumulative"}
        assert all(c.feasible for c in matched_cells)
        assert all(c.time in arley.base.TS for c in matched_cells)

    def test_units_skip_infeasible_cells_but_keep_indices(self, grid_cells) -> None:
        cells = grid_cells
        units = arley.build_units(cells, 4, 2, arley.GRID.seed_base)
        assert len(units) == len(cells) * 2 * 2
        assert all(u.cell.feasible for u in units)
        index_of = {id(c): i for i, c in enumerate(cells)}
        assert all(index_of[id(u.cell)] == u.cell_index for u in units)


class TestSeeds:
    def test_grid_seeds_are_deterministic_and_differ_from_matched(
        self, matched_cells
    ) -> None:
        cell = matched_cells[0]
        matched = arley.Unit("contour", cell, 0, 0, 1)
        grid = arley.Unit("contour", cell, 0, 0, 1, arley.GRID.seed_base)
        assert arley.replicate_seed(matched, 0) == arley.replicate_seed(matched, 0)
        assert arley.replicate_seed(grid, 0) == arley.replicate_seed(grid, 0)
        assert arley.replicate_seed(matched, 0) != arley.replicate_seed(grid, 0)

    def test_seed_depends_on_cell_values_not_position(self, matched_cells) -> None:
        cell = matched_cells[0]
        unit = arley.Unit("splitting", cell, 3, 0, 1)
        moved = arley.Unit("splitting", cell, 99, 0, 1)
        expected = int(
            np.random.SeedSequence(
                [
                    arley.BASE_SEED,
                    round(cell.slope * 10_000),
                    round(cell.time * 1000),
                    1,
                    7,
                ]
            ).generate_state(1)[0]
            % (2**31 - 1)
        )
        assert arley.replicate_seed(unit, 7) == expected
        assert arley.replicate_seed(moved, 7) == expected


class TestQuickGridRun:
    def test_writes_the_grid_files(self, tmp_path: Path) -> None:
        arley.main(
            [
                "--grid",
                "--quick",
                "--shards-dir",
                str(tmp_path / "shards"),
                "--data-dir",
                str(tmp_path / "data"),
            ]
        )
        data = tmp_path / "data"
        for name in (
            arley.GRID.rows,
            arley.GRID.units,
            arley.GRID.cells,
            arley.GRID.env,
        ):
            assert (data / name).exists()
        rows = pd.read_csv(data / arley.GRID.rows)
        assert set(rows["slope"]) == set(arley.QUICK_SLOPES)
        assert rows["T"].max() <= arley.QUICK_GRID_MAX_T
        assert set(rows["method"]) == {"contour", "splitting"}
        assert not (data / arley.ROWS_FILE).exists()  # the matched files are separate


def _rows() -> pd.DataFrame:
    records = []
    for method in ("contour", "splitting"):
        for t in (1.0, 10.0):
            for r in range(4):
                records.append(
                    {
                        "method": method,
                        "mu": 3.0,
                        "slope": 0.06,
                        "T": t,
                        "replicate": r,
                        "seconds": 0.1 * (r + 1) * t,
                        "n_trees": 1 if method == "contour" else 5,
                        "n_attempts": np.nan if method == "contour" else 2,
                        "levels": np.nan if method == "contour" else 3,
                    }
                )
    return pd.DataFrame(records)


class TestCharts:
    def test_curves_and_diagnostics(self) -> None:
        pytest.importorskip("altair")
        charts = _load("arley_charts")
        curves = charts.curves_from_rows(_rows())
        assert set(curves["method"]) == {
            charts.LABEL_CONTOUR,
            charts.LABEL_RUN,
            charts.LABEL_KEPT,
        }
        assert set(curves["replicates"]) == {4}
        run = curves[(curves["method"] == charts.LABEL_RUN) & (curves["T"] == 1.0)]
        kept = curves[(curves["method"] == charts.LABEL_KEPT) & (curves["T"] == 1.0)]
        assert float(kept["median"].iloc[0]) == pytest.approx(
            float(run["median"].iloc[0]) / 5
        )
        diag = charts.diagnostics_from_rows(_rows())
        assert list(diag.columns) == ["mu", "slope", "T", "trees", "attempts", "levels"]
        assert set(diag["attempts"]) == {2.0}

    def test_cost_charts(self) -> None:
        pytest.importorskip("altair")
        charts = _load("arley_charts")
        curves = charts.curves_from_rows(_rows())
        by_t = charts.cost_chart(curves, charts.x_by_t([1.0, 10.0])).to_dict()
        assert len(by_t["layer"]) == 3
        by_slope = charts.cost_chart(curves, charts.x_by_slope([0.06])).to_dict()
        assert by_slope["layer"][0]["encoding"]["x"]["field"] == "slope"
        assert by_slope["layer"][0]["encoding"]["x"]["scale"]["type"] == "log"
        diag = charts.diagnostic_chart(charts.diagnostics_from_rows(_rows())).to_dict()
        assert len(diag["hconcat"]) == 3

    @pytest.mark.parametrize("slope", [0.025, 0.06, 0.2])
    def test_rate_chart_marks_the_crossing(self, slope: float) -> None:
        pytest.importorskip("altair")
        charts = _load("arley_charts")
        spec = charts.rate_chart(slope).to_dict()
        data = spec["datasets"]
        frames = [pd.DataFrame(v) for v in data.values()]
        lines = next(f for f in frames if "rate" in f.columns)
        death = lines[lines["series"].str.contains("death")]
        assert death["rate"].max() == pytest.approx(slope * 100.0)
        assert set(lines[lines["series"].str.contains("birth")]["rate"]) == {1.0}
        crossing = next(f for f in frames if "label" in f.columns)
        assert float(crossing["t"].iloc[0]) == pytest.approx(1.0 / slope)
        first = spec["layer"][0]["encoding"]
        assert first["y"]["scale"]["domain"] == [0, 3.0]
        assert spec["layer"][0]["mark"]["clip"] is True

    def test_no_crossing_marker_beyond_the_plot(self) -> None:
        pytest.importorskip("altair")
        charts = _load("arley_charts")
        assert math.isclose(1.0 / 0.005, 200.0)
        spec = charts.rate_chart(0.005).to_dict()
        assert len(spec["layer"]) == 1
