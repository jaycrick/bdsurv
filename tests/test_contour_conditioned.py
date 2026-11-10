"""Test contour simulation conditioned on the population alive at ``T``."""

from collections.abc import Iterator
from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd
import pytest

from bdsurv import _linear_math as lm
from bdsurv.contour import _seeded_rng, _validate_contour_process_list
from bdsurv.contour_conditioned import (
    _join_pieces,
    _log_survival_pair,
    _population_contour,
    _prefer_top,
    _resolve_rates,
    _Walker,
    simulate_contour_given_population,
)
from bdsurv.linear_rates import PiecewiseConstant, PiecewiseLinear

T = 4.0
N_SEEDS = 200
DIRECTIONS = ("top", "bottom")
POPULATIONS = (1, 2, 5)


def _const(value: float, end: float = T) -> PiecewiseConstant:
    return PiecewiseConstant(np.array([value]), np.array([end]))


def _line(start: float, end_value: float, end: float = T) -> PiecewiseLinear:
    return PiecewiseLinear(np.array([start]), np.array([end_value]), np.array([end]))


ARLEY = (_const(1.0), _line(0.0, 0.8 * T))
VARYING_BIRTH = (_line(2.0, 0.5), _const(1.0))

REGIMES: dict[str, tuple[Any, Any]] = {
    "subcritical": (0.8, 1.1),
    "critical": (1.0, 1.0),
    "supercritical": (1.2, 0.7),
    "arley": ARLEY,
    "varying_birth": VARYING_BIRTH,
}
regimes = pytest.mark.parametrize("regime", list(REGIMES))
directions = pytest.mark.parametrize("direction", DIRECTIONS)


def _terminal_time(birth: Any) -> float | None:
    return T if isinstance(birth, float) else None


def _tree(regime: str, n: int, seed: int, direction: str = "auto") -> Any:
    birth, death = REGIMES[regime]
    return simulate_contour_given_population(
        birth,
        death,
        n,
        terminal_time=_terminal_time(birth),
        random_seed=seed,
        direction=direction,  # type: ignore[arg-type]
    )


def _contour(regime: str, n: int, seed: int, direction: str) -> list[float]:
    birth, death = REGIMES[regime]
    rates = _resolve_rates(birth, death, _terminal_time(birth))
    walker = _Walker(rates.birth, rates.death, rates.terminal_time, _seeded_rng(seed))
    return _population_contour(walker, n, prefer_top=direction == "top")


class _FakeRng:
    """Yield the given Exp(1) draws, then fixed positive ones."""

    def __init__(self, draws: list[float]) -> None:
        self._draws: Iterator[float] = iter(draws)

    def exponential(self) -> float:
        return next(self._draws, 0.5)


def _walker(rng: Any, birth: float = 1.0, death: float = 1.0) -> _Walker:
    return _Walker(lm.as_rate(_const(birth)), lm.as_rate(_const(death)), T, rng)


class TestJoinPieces:
    def test_example(self) -> None:
        first = [0.0, 3.5, 3.0, 4.0]  # reversed (4, 3, 3.5, 0)
        tt = [[4.0, 2.5, 4.0]]
        last = [4.0, 2.0, 3.3, 0.0]
        assert _join_pieces(first, tt, last) == [
            0.0, 3.5, 3.0, 4.0, 2.5, 4.0, 2.0, 3.3, 0.0,
        ]  # fmt: skip

    def test_no_excursions(self) -> None:
        assert _join_pieces([0.0, 4.0], [], [4.0, 1.0, 2.0, 0.0]) == [
            0.0, 4.0, 1.0, 2.0, 0.0,
        ]  # fmt: skip


class TestContourValidity:
    @regimes
    @directions
    @pytest.mark.parametrize("n", POPULATIONS)
    def test_contour_structure(self, regime: str, direction: str, n: int) -> None:
        for seed in range(N_SEEDS):
            contour = _contour(regime, n, seed, direction)
            _validate_contour_process_list(contour)
            assert all(x != y for x, y in pairwise(contour))
            assert contour[0] == 0.0
            assert contour[-1] == 0.0
            assert all(0.0 <= x <= T for x in contour)
            assert contour.count(T) == n
            assert 0.0 not in contour[1:-1]


def _check_tree(tree: Any, n: int) -> None:
    records: pd.DataFrame = tree.records
    assert tree.terminal_time == T
    assert int(records["survived"].sum()) == n
    spine = records[records["is_spine"]]
    assert len(spine) == 1
    assert bool(spine["survived"].iloc[0])
    assert records["individual_id"].is_unique
    assert int(records["parent_id"].isna().sum()) == 1
    by_id = records.set_index("individual_id")
    for row in records.itertuples():
        if not pd.isna(row.death_time):
            assert row.death_time > row.creation_time
        if pd.isna(row.parent_id):
            continue
        parent = by_id.loc[int(row.parent_id)]
        end = T if parent["survived"] else parent["death_time"]
        assert parent["creation_time"] < row.creation_time < end


class TestTreeValidity:
    @regimes
    @directions
    @pytest.mark.parametrize("n", POPULATIONS)
    def test_tree_structure(self, regime: str, direction: str, n: int) -> None:
        for seed in range(N_SEEDS):
            _check_tree(_tree(regime, n, seed, direction), n)

    @regimes
    @pytest.mark.parametrize("n", POPULATIONS)
    def test_auto_direction(self, regime: str, n: int) -> None:
        for seed in range(50):
            _check_tree(_tree(regime, n, seed), n)

    def test_rates_stored(self) -> None:
        tree = _tree("subcritical", 2, 0)
        assert (tree.birth_rate, tree.death_rate) == (0.8, 1.1)
        assert tree.birth_rate_function is None
        assert tree.death_rate_function is None
        tree = _tree("arley", 2, 0)
        assert np.isnan(tree.birth_rate)
        assert np.isnan(tree.death_rate)
        assert tree.birth_rate_function is ARLEY[0]
        assert tree.death_rate_function is ARLEY[1]
        assert tree.terminal_time == T

    def test_terminal_time_given_for_rate_likes(self) -> None:
        tree = simulate_contour_given_population(*ARLEY, 2, terminal_time=T)
        assert tree.terminal_time == T

    def test_direction_rule(self) -> None:
        birth, death = (lm.as_rate(r) for r in ARLEY)
        fwd, rev = _log_survival_pair(birth, death, T)
        assert (rev > fwd) == _prefer_top(birth, death, T, "auto")
        growth = float(lm.cumulative(birth, T) - lm.cumulative(death, T))
        assert rev - fwd == pytest.approx(-growth)
        critical = (lm.as_rate(_const(1.0)),) * 2
        assert not _prefer_top(*critical, T, "auto")  # tie -> bottom

    def test_zero_death_rate(self) -> None:
        for direction in DIRECTIONS:
            tree = simulate_contour_given_population(
                1.0,
                0.0,
                3,
                T,
                random_seed=1,
                direction=direction,  # type: ignore[arg-type]
            )
            _check_tree(tree, 3)

    def test_zero_birth_rate_single(self) -> None:
        tree = simulate_contour_given_population(0.0, 1.0, 1, T, random_seed=1)
        _check_tree(tree, 1)
        assert len(tree.records) == 1


class TestSeeding:
    @regimes
    def test_same_seed_same_records(self, regime: str) -> None:
        a, b = _tree(regime, 3, 7), _tree(regime, 3, 7)
        pd.testing.assert_frame_equal(a.records, b.records)

    @regimes
    def test_different_seeds_differ(self, regime: str) -> None:
        records = [_tree(regime, 3, seed).records for seed in range(5)]
        assert any(not records[0].equals(r) for r in records[1:])


class TestSteps:
    def test_down_never_returns_start(self) -> None:
        walker = _walker(_FakeRng([0.0, 0.3]))
        assert walker.down(2.0) == pytest.approx(1.7)

    def test_up_never_returns_start(self) -> None:
        walker = _walker(_FakeRng([0.0, 0.3]))
        assert walker.up(1.0) == pytest.approx(1.3)

    def test_snapping(self) -> None:
        assert _walker(_FakeRng([2.0 - 1e-9])).down(2.0) == 0.0
        assert _walker(_FakeRng([100.0])).down(2.0) == 0.0
        assert _walker(_FakeRng([2.0 - 1e-9])).up(2.0) == T
        assert _walker(_FakeRng([100.0])).up(2.0) == T

    def test_real_rng_steps_move(self) -> None:
        walker = _walker(np.random.default_rng(0))
        for _ in range(500):
            assert 0.0 <= walker.down(1.0) < 1.0
            assert 1.0 < walker.up(1.0) <= T

    def test_run_endpoints(self) -> None:
        walker = _walker(np.random.default_rng(1))
        for _ in range(100):
            top = walker.top_excursion()
            assert top[0] == T
            assert top[-1] in (0.0, T)
            bottom = walker.bottom_excursion()
            assert bottom[0] == 0.0
            assert bottom[-1] in (0.0, T)


class TestInputErrors:
    @pytest.mark.parametrize("n", [0, -1])
    def test_n_alive(self, n: int) -> None:
        with pytest.raises(ValueError, match="n_alive"):
            simulate_contour_given_population(1.0, 1.0, n, T)

    @pytest.mark.parametrize("terminal_time", [0.0, -1.0])
    def test_non_positive_terminal_time(self, terminal_time: float) -> None:
        with pytest.raises(ValueError, match="terminal_time must be positive"):
            simulate_contour_given_population(1.0, 1.0, 1, terminal_time)
        with pytest.raises(ValueError, match="terminal_time must be positive"):
            simulate_contour_given_population(*ARLEY, 1, terminal_time)

    def test_missing_terminal_time(self) -> None:
        with pytest.raises(ValueError, match="terminal_time is required"):
            simulate_contour_given_population(1.0, 1.0, 1)

    def test_mismatched_terminal_time(self) -> None:
        with pytest.raises(ValueError, match="terminal_time must equal"):
            simulate_contour_given_population(*ARLEY, 1, terminal_time=T + 1.0)

    def test_mismatched_end_times(self) -> None:
        with pytest.raises(ValueError, match="same end time"):
            simulate_contour_given_population(_const(1.0, T), _const(1.0, T + 1.0), 1)

    @pytest.mark.parametrize("birth", [0.0, _const(0.0)])
    def test_zero_birth_rate(self, birth: Any) -> None:
        tt = T if isinstance(birth, float) else None
        with pytest.raises(ValueError, match="birth_rate must not be identically"):
            simulate_contour_given_population(birth, _const(1.0), 2, tt)

    def test_bad_direction(self) -> None:
        with pytest.raises(ValueError, match="direction"):
            simulate_contour_given_population(
                1.0,
                1.0,
                1,
                T,
                direction="sideways",  # type: ignore[arg-type]
            )

    def test_negative_rate(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            simulate_contour_given_population(-1.0, 1.0, 1, T)
