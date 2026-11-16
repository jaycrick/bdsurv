"""Test contour-process simulation and reconstruction."""

from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
import pytest

from bdsurv import PiecewiseConstant, PiecewiseLinear, count_draws
from bdsurv._seeding import root_rng
from bdsurv.contour import (
    Walker,
    _survival_contour,
    contour_process_list_to_tree,
    join_pieces,
    sample_until,
    simulate_birth_death_process_from_contour_process,
    validate_contour_process_list,
)
from bdsurv.rates import resolve_rates
from bdsurv.tree import SimulatedTree

T = 4.0


class _FakeRng:
    """Yield the given Exp(1) draws, then fixed positive ones."""

    def __init__(self, draws: list[float]) -> None:
        self._draws: Iterator[float] = iter(draws)

    def exponential(self) -> float:
        return next(self._draws, 0.5)


def _walker(rng: Any, rates: tuple[Any, Any] = (1.0, 1.0)) -> Walker:
    birth, death, horizon = resolve_rates(*rates, T)
    return Walker(birth, death, horizon, rng)


def _arley_walker(seed: int) -> Walker:
    birth = PiecewiseConstant(np.array([1.0]), np.array([T]))
    death = PiecewiseLinear(np.array([0.0]), np.array([0.8 * T]), np.array([T]))
    b, d, horizon = resolve_rates(birth, death, None)
    return Walker(b, d, horizon, np.random.default_rng(seed))


class TestContourProcessSimulation:
    """Test contour-process simulation and reconstruction."""

    def test_contour_process_concatenation_example(self) -> None:
        # Old test of ``_build_contour_process_list``: the same example, with
        # the spine piece already oriented from 0 to T.
        contour_process_list = join_pieces(
            [0.0, 3.0, 2.0, 5.0], [[5.0, 2.0, 5.0]], [5.0, 3.0, 4.0, 0.0]
        )

        assert contour_process_list == [0.0, 3.0, 2.0, 5.0, 2.0, 5.0, 3.0, 4.0, 0.0]

    @pytest.mark.parametrize(
        "contour_process_list",
        [
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 3.0, 2.0],
            [0.0, 3.0, 2.0],
            [0.0, 3.0, 3.0, 0.0, 0.0],
            [0.0, 3.0, 2.0, 1.0, 0.0],
            [0.0, 3.0, 4.0, 2.0, 0.0],
        ],
    )
    def test_contour_process_validation_rejects_invalid_examples(
        self,
        contour_process_list: list[float],
    ) -> None:
        with pytest.raises(ValueError, match="contour_process_list"):
            validate_contour_process_list(contour_process_list)

    def test_contour_process_validation_accepts_valid_examples(self) -> None:
        validate_contour_process_list([0.0, 3.0, 2.0, 5.0, 2.0, 5.0, 3.0, 4.0, 0.0])
        validate_contour_process_list([0.0, 1.0, 0.0])

    def test_tree_reconstruction_from_contour_list(self) -> None:
        contour_process_list = [0.0, 3.0, 2.0, 5.0, 2.0, 5.0, 3.0, 4.0, 0.0]

        tree = contour_process_list_to_tree(
            contour_process_list,
            birth_rate=0.5,
            death_rate=0.2,
        )

        assert isinstance(tree, SimulatedTree)
        records = tree.records.sort_values("individual_id").reset_index(drop=True)

        assert pd.isna(records.loc[0, "parent_id"])
        assert records.loc[0, "individual_id"] == 0
        assert records.loc[0, "creation_time"] == 0.0
        assert records.loc[0, "death_time"] == 3.0
        assert records.loc[1, "parent_id"] == 0
        assert records.loc[2, "parent_id"] == 0
        assert records.loc[3, "parent_id"] == 2

    def test_simulation_is_deterministic_for_fixed_seed(self) -> None:
        tree_1 = simulate_birth_death_process_from_contour_process(
            birth_rate=0.5,
            death_rate=0.2,
            terminal_time=5.0,
            random_seed=123,
        )
        tree_2 = simulate_birth_death_process_from_contour_process(
            birth_rate=0.5,
            death_rate=0.2,
            terminal_time=5.0,
            random_seed=123,
        )

        assert tree_1.records.equals(tree_2.records)
        assert tree_1.birth_rate == tree_2.birth_rate
        assert tree_1.death_rate == tree_2.death_rate
        assert tree_1.terminal_time == tree_2.terminal_time

    def test_generated_contour_process_satisfies_validation_and_reconstruction(
        self,
    ) -> None:
        walker = _walker(np.random.default_rng(2026), (0.8, 1.1))
        contour_process_list = _survival_contour(walker)

        validate_contour_process_list(contour_process_list)

        tree = contour_process_list_to_tree(
            contour_process_list,
            birth_rate=0.8,
            death_rate=1.1,
        )
        assert tree.records["individual_id"].is_unique
        assert tree.records["parent_id"].isna().sum() == 1
        assert tree.terminal_time == 4.0
        assert all(0.0 <= ct <= 4.0 for ct in tree.records["creation_time"])

    def test_rejects_non_positive_rates_and_terminal_time(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            simulate_birth_death_process_from_contour_process(
                birth_rate=0.0, death_rate=0.2, terminal_time=1.0
            )
        with pytest.raises(ValueError, match="must be positive"):
            simulate_birth_death_process_from_contour_process(
                birth_rate=0.5, death_rate=-0.2, terminal_time=1.0
            )
        with pytest.raises(ValueError, match="must be positive"):
            simulate_birth_death_process_from_contour_process(
                birth_rate=0.5, death_rate=0.2, terminal_time=0.0
            )

    def test_exactly_one_spine_when_survivors_exist(self) -> None:
        tree = simulate_birth_death_process_from_contour_process(
            birth_rate=0.9,
            death_rate=1.0,
            terminal_time=10.0,
            random_seed=7,
        )
        spine_rows = tree.records[tree.records["is_spine"]]
        assert len(spine_rows) == 1
        assert bool(spine_rows.iloc[0]["survived"])


class TestSurvivalContour:
    @pytest.mark.parametrize("seed", range(5))
    @pytest.mark.parametrize("rates", ["constant", "arley"])
    def test_contour_matches_decoded_tree(self, rates: str, seed: int) -> None:
        if rates == "constant":
            walker = _walker(np.random.default_rng(seed), (0.9, 1.0))
        else:
            walker = _arley_walker(seed)
        contour = _survival_contour(walker)
        assert contour[0] == 0.0
        assert contour[-1] == 0.0, "the contour must end at 0"
        validate_contour_process_list(contour)
        tree = contour_process_list_to_tree(contour, birth_rate=1.0, death_rate=1.0)
        survivors = int(tree.records["survived"].sum())
        assert contour.count(T) == survivors, (
            f"{contour.count(T)} visits to T but {survivors} survivors"
        )

    def test_spine_excursion_returning_to_t_is_rejected(self) -> None:
        # Spine: down to 3 then up to T (rejected); down to 0 (kept).
        # Then the last piece goes straight down to 0.
        walker = _walker(_FakeRng([1.0, 5.0, 100.0, 100.0]))
        with count_draws() as counts:
            contour = _survival_contour(walker)
        assert contour == [0.0, T, 0.0]
        assert counts.extras["spine_resets"] == 1, "one spine excursion was rejected"
        assert counts.extras["contour_steps"] == 4, "2 + 1 + 1 steps were taken"

    def test_further_visits_to_t_are_kept(self) -> None:
        walker = _walker(_FakeRng([100.0, 1.0, 5.0, 100.0]))
        with count_draws() as counts:
            contour = _survival_contour(walker)
        assert contour == [0.0, T, 3.0, T, 0.0]
        assert counts.extras["spine_resets"] == 0, "no spine excursion was rejected"
        assert counts.extras["contour_steps"] == 4

    def test_contour_steps_equals_steps_taken(self) -> None:
        # Each step draws one exponential (a retry has probability zero), so
        # the step counter equals the number of draws.
        with count_draws() as counts:
            walker = _walker(root_rng(3, "contour"), (0.9, 1.0))
            contour = _survival_contour(walker)
        steps = counts.extras["contour_steps"]
        assert steps == counts.by_kind("exponential"), (
            f"{steps} steps but {counts.by_kind('exponential')} draws"
        )
        assert steps >= len(contour) - 1, "the kept pieces alone have this many steps"


class TestSampleUntil:
    def test_returns_first_accepted_value(self) -> None:
        values = iter([1, 3, 4, 6, 8])
        calls = 0

        def make() -> int:
            nonlocal calls
            calls += 1
            return next(values)

        assert sample_until(make, lambda v: v % 2 == 0) == 4
        assert calls == 3, "make must be called until the first accepted value"

    def test_accepts_immediately(self) -> None:
        calls = 0

        def make() -> int:
            nonlocal calls
            calls += 1
            return 7

        assert sample_until(make, lambda _: True) == 7
        assert calls == 1, "make must be called once when the first value is accepted"
