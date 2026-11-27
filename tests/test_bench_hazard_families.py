"""Tests for the hazard-family benchmark harness (quick mode)."""

import json
import math
from pathlib import Path

import bench_hazard_families as bench
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def quick_out(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run the quick benchmark once into a temporary directory."""
    out = tmp_path_factory.mktemp("bench_hazard_families")
    bench.main(["--quick", "--out", str(out)])
    return out


@pytest.fixture(scope="module")
def rows(quick_out: Path) -> pd.DataFrame:
    """Read the quick benchmark rows."""
    return pd.read_csv(quick_out / bench.CSV_NAME)


class TestQuickRun:
    def test_schema(self, rows: pd.DataFrame) -> None:
        assert list(rows.columns[:7]) == [
            "family",
            "path",
            "scenario",
            "n",
            "measure",
            "repeat",
            "seconds",
        ]
        assert set(rows.columns) == set(bench.ALL_COLUMNS)

    def test_families_and_paths(self, rows: pd.DataFrame) -> None:
        pairs = set(zip(rows["family"], rows["path"], strict=True))
        assert pairs == {
            ("constant", "hazard-direct"),
            ("constant", "rate"),
            ("linear", "hazard-brent"),
            ("linear", "rate"),
            ("pw-linear", "rate"),
            ("pw-linear", "hazard-piecewise"),
            ("logistic", "hazard-direct"),
            ("floor-logistic", "split"),
            ("floor-logistic-fall", "split"),
            *(
                (name, path)
                for name in bench.POLY_NAMES
                for path in ("hazard-brent", "closed-forced", "closed-gated")
            ),
        }

    def test_measures_and_scenarios(self, rows: pd.DataFrame) -> None:
        assert set(rows["measure"]) == {
            *bench.QUICK_MEASURES,
            "end_to_end",
            "evals_per_root",
            "no_event_fraction",
            "mean_residual",
            "regime_newton",
            "regime_algebraic",
            "regime_brent",
            "fallback_fraction",
        }
        timed = rows[rows["measure"].isin(bench.QUICK_MEASURES)]
        assert set(timed["scenario"]) == set(bench.QUICK_SCENARIOS)
        batch = timed[timed["measure"] != "between_scalar"]
        assert set(batch["n"]) == set(bench.QUICK_SIZES)

    def test_timings_positive_and_repeated(self, rows: pd.DataFrame) -> None:
        timed = rows[rows["measure"].isin(bench.QUICK_MEASURES)]
        assert (timed["seconds"] > 0).all()
        assert set(timed["repeat"]) == set(range(bench.QUICK_REPEATS))

    def test_end_to_end_rows_per_seed(self, rows: pd.DataFrame) -> None:
        e2e = rows[rows["measure"] == "end_to_end"]
        assert (e2e["n"] == bench.QUICK_E2E_SEEDS).all()
        # 9 original families plus the four polynomial hazard-brent ones.
        assert len(e2e) == 13 * bench.QUICK_E2E_SEEDS
        assert (e2e["n_alive"] == 1).all()
        assert not e2e["path"].str.startswith("closed").any()

    def test_no_events_are_rare(self, rows: pd.DataFrame) -> None:
        frac = rows[rows["measure"] == "no_event_fraction"]["seconds"]
        assert (frac == 0.0).all()

    def test_closed_form_needs_no_evaluations(self, rows: pd.DataFrame) -> None:
        evals = rows[rows["measure"] == "evals_per_root"]
        direct = evals[evals["path"] == "hazard-direct"]
        brent = evals[evals["path"] == "hazard-brent"]
        split = evals[evals["path"] == "split"]
        assert (direct["seconds"] == 0.0).all()
        assert (split["seconds"] == 0.0).all()
        assert (split["note"] == "split closed form").all()
        assert (brent["seconds"] > 1.0).all()
        closed = evals[evals["path"].str.startswith("closed")]
        assert closed["seconds"].isna().all()

    def test_matched_means_in_s1(self, rows: pd.DataFrame) -> None:
        mean = rows[(rows["measure"] == "mean_residual") & (rows["scenario"] == "S1")]
        mean = mean[mean["n_alive"] == 1]
        by = mean.set_index(["family", "path"])["seconds"]
        assert math.isclose(
            by["pw-linear", "rate"], by["pw-linear", "hazard-piecewise"]
        )

    def test_n_alive_axis(self, rows: pd.DataFrame) -> None:
        axis = rows[rows["n_alive"] != 1]
        assert set(axis["n_alive"]) == set(bench.N_ALIVE_AXIS) - {1}
        on_axis = set(zip(axis["family"], axis["path"], strict=True))
        assert on_axis == {
            *((n, p) for n in bench.POLY_NAMES for p in bench.CLOSED_METHODS),
            *((n, "hazard-brent") for n in bench.POLY_NAMES),
        }
        # Only the measures that depend on n_alive are run on the axis.
        assert set(axis["measure"]) >= {"invert", "full_batch"}
        assert not set(axis["measure"]) & {"hazard", "cumulative", "between_scalar"}
        other = rows[~rows["family"].isin(bench.POLY_NAMES)]
        assert (other["n_alive"] == 1).all()

    def test_regime_counts_recorded(self, rows: pd.DataFrame) -> None:
        reg = rows[rows["measure"].str.startswith("regime_")]
        assert set(reg["path"]) == set(bench.CLOSED_METHODS)
        cells = reg.groupby(["family", "path", "n_alive", "scenario", "n"], sort=False)[
            "seconds"
        ].sum()
        assert (cells == bench.QUICK_SIZES[0]).all()
        fb = rows[rows["measure"] == "fallback_fraction"]
        # Gated Newton/Cardano never falls back on the benchmark parameter sets.
        assert (fb[fb["path"] == "closed-gated"]["seconds"] == 0.0).all()
        # Forced Cardano/Ferrari never uses Newton-from-truncation, except that
        # the linear hazard (c = d = 0) is always the exact closed form.
        forced = reg[
            (reg["path"] == "closed-forced") & (reg["family"] != "quadratic-c0")
        ]
        assert (forced[forced["measure"] == "regime_newton"]["seconds"] == 0.0).all()

    def test_closed_paths_agree_in_law_with_brent(self, rows: pd.DataFrame) -> None:
        mean = rows[(rows["measure"] == "mean_residual") & (rows["scenario"] == "S1")]
        mean = mean[mean["n_alive"] == 1].set_index(["family", "path"])["seconds"]
        for name in bench.POLY_NAMES:
            brent = mean[name, "hazard-brent"]
            for path in bench.CLOSED_METHODS:
                # 1000 draws: Monte Carlo error only (the E streams differ).
                assert math.isclose(mean[name, path], brent, rel_tol=0.2)

    def test_environment_json(self, quick_out: Path) -> None:
        env = json.loads((quick_out / bench.ENV_NAME).read_text())
        for key in ("python", "numpy", "bdsurv", "git_sha"):
            assert key in env
        assert "host" not in env
        assert "slurm_job" not in env
        assert env["quick"] is True
        assert env["T"] == bench.T


class TestHelpers:
    def test_split_families(self) -> None:
        fams = {(f.name, f.path): f for f in bench.build_families()}
        for key in (("floor-logistic", "split"), ("floor-logistic-fall", "split")):
            assert fams[key].n_draws == 2
            assert not fams[key].loopish
        assert fams["linear", "hazard-brent"].n_draws == 1
        assert fams["linear", "hazard-brent"].loopish
        rise = fams["floor-logistic", "split"].obj
        fall = fams["floor-logistic-fall", "split"].obj
        assert float(rise.hazard(60.0)) == pytest.approx(1.0)
        assert float(fall.hazard(60.0)) == pytest.approx(0.2)
        assert float(fall.hazard(0.0)) > float(fall.hazard(20.0))

    def test_split_ops_run(self) -> None:
        fam = next(
            f
            for f in bench.build_families()
            if (f.name, f.path) == ("floor-logistic", "split")
        )
        ops = bench.operations(fam)
        batch = bench.Batch("S2", 50)
        for name in ("invert", "individual", "full_batch"):
            out = np.asarray(ops[name](batch))
            assert out.shape == (50,)
            assert np.all(out >= 0)

    def test_closed_ops_run_and_agree(self) -> None:
        fams = {
            (f.name, f.path, f.n_alive): f
            for f in bench.with_n_alive(bench.build_families())
        }
        batch = bench.Batch("S2", 60)
        for name in bench.POLY_NAMES:
            for n_alive in bench.N_ALIVE_AXIS:
                ref = bench.operations(fams[name, "hazard-brent", n_alive])
                for path in bench.CLOSED_METHODS:
                    ops = bench.operations(fams[name, path, n_alive])
                    np.testing.assert_allclose(
                        ops["invert"](batch), ref["invert"](batch), rtol=1e-6
                    )
                    individual = np.asarray(ops["individual"](batch))
                    assert individual.shape == (60,)
                    full = np.asarray(ops["full_batch"](batch))
                    assert full.shape == (60,)
                    assert np.all(full[np.isfinite(full)] >= 0.0)

    def test_n_alive_axis_expansion(self) -> None:
        fams = bench.with_n_alive(bench.build_families())
        keyed = [(f.name, f.path, f.n_alive) for f in fams]
        assert len(keyed) == len(set(keyed))
        assert ("logistic", "hazard-direct", 1) in keyed
        assert ("logistic", "hazard-direct", 100) not in keyed
        assert ("cubic", "closed-gated", 10_000) in keyed
        assert ("floor-logistic", "split", 100) not in keyed

    def test_runtime_trim_plan(self) -> None:
        fams = {(f.name, f.path): f for f in bench.build_families()}
        brent = fams["linear", "hazard-brent"]
        split = fams["floor-logistic", "split"]
        assert bench.cell_plan(brent, "S1", 1_000, "invert", 7) == (7, False)
        assert bench.cell_plan(brent, "S1", 100_000, "invert", 7) == (3, False)
        assert bench.cell_plan(brent, "S1", 1_000_000, "invert", 7) is None
        assert bench.cell_plan(brent, "S2", 1_000_000, "individual", 7) is None
        assert bench.cell_plan(brent, "S2", 1_000_000, "full_batch", 7) == (1, True)
        assert bench.cell_plan(split, "S1", 1_000_000, "invert", 7) == (7, False)

    def test_matched_check_passes(self) -> None:
        bench.check_matched()

    def test_scenarios_are_reproducible(self) -> None:
        a = bench.scenario_starts("S2", 100)
        b = bench.scenario_starts("S2", 100)
        assert np.array_equal(a, b)
        assert a.min() >= 0.0
        assert a.max() <= 2 * bench.C
        assert np.all(bench.scenario_starts("S1", 5) == 0.0)
        assert np.all(bench.scenario_starts("S4", 5) == bench.C + 20.0)

    def test_budget_stops_after_first_repeat(self) -> None:
        calls: list[int] = []
        times, _, note = bench.timed_cell(
            lambda: calls.append(1), lambda: None, 5, 0.0, None
        )
        assert len(times) == 1
        assert "skipped" in note

    def test_projection_over_budget_runs_once(self) -> None:
        times, _, note = bench.timed_cell(lambda: None, lambda: None, 5, 10.0, 99.0)
        assert len(times) == 1
        assert "projected" in note

    def test_summary_is_one_row_per_cell(self, rows: pd.DataFrame) -> None:
        summary = bench.summarise(rows)
        keys = ["family", "path", "n_alive", "scenario", "n", "measure"]
        assert not summary.duplicated(keys).any()
