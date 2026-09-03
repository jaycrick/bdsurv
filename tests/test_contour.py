"""Test contour-process simulation and reconstruction."""

import numpy as np
import pandas as pd
import pytest

from bdsurv.contour import (
    _build_contour_process_list,
    _contour_process_list_to_tree,
    _generate_non_spine_component,
    _generate_spine_component,
    _validate_contour_process_list,
    simulate_birth_death_process_from_contour_process,
)
from bdsurv.tree import SimulatedTree


class TestContourProcessSimulation:
    """Test contour-process simulation and reconstruction."""

    def test_contour_process_concatenation_example(self) -> None:
        non_spine_component = [5.0, 2.0, 5.0, 3.0, 4.0, 0.0]
        spine_component = [5.0, 2.0, 3.0, 0.0]

        contour_process_list = _build_contour_process_list(
            spine_component=spine_component,
            non_spine_component=non_spine_component,
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
        with pytest.raises(ValueError):
            _validate_contour_process_list(contour_process_list)

    def test_contour_process_validation_accepts_valid_examples(self) -> None:
        _validate_contour_process_list([0.0, 3.0, 2.0, 5.0, 2.0, 5.0, 3.0, 4.0, 0.0])
        _validate_contour_process_list([0.0, 1.0, 0.0])

    def test_tree_reconstruction_from_contour_list(self) -> None:
        contour_process_list = [0.0, 3.0, 2.0, 5.0, 2.0, 5.0, 3.0, 4.0, 0.0]

        tree = _contour_process_list_to_tree(
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
        rng = np.random.default_rng(2026)
        non_spine_component = _generate_non_spine_component(
            birth_rate=0.8,
            death_rate=1.1,
            terminal_time=4.0,
            rng=rng,
        )
        spine_component = _generate_spine_component(
            birth_rate=0.8,
            death_rate=1.1,
            terminal_time=4.0,
            rng=rng,
        )
        contour_process_list = _build_contour_process_list(
            spine_component=spine_component,
            non_spine_component=non_spine_component,
        )

        _validate_contour_process_list(contour_process_list)

        tree = _contour_process_list_to_tree(
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
