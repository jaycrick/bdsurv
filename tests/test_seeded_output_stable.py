"""Pin seeded outputs, so performance work cannot silently change the random stream.

The expected values were recorded before the speed-ups (inline ``isclose``,
copy-on-write particle records, scalar hazard inversion), which must not alter
any seeded result.
"""

import numpy as np
import pytest

from bdsurv import (
    LinearSplittingConfig,
    PiecewiseConstant,
    PiecewiseLinear,
    SplittingConfig,
    sample_conditioned_survival_process,
    simulate_birth_death_lifespans,
    simulate_birth_death_linear,
    simulate_birth_death_process,
    simulate_birth_death_process_from_contour_process,
    simulate_contour_given_ages,
    simulate_contour_given_population,
    simulate_linear_rate_contour_process,
    simulate_linear_survival_splitting,
    simulate_subcritical_survival_splitting,
)

BIRTH = PiecewiseConstant(np.array([1.0]), np.array([6.0]))
DEATH = PiecewiseLinear(np.array([0.0]), np.array([3.0]), np.array([6.0]))


def _summary(records) -> list[float]:
    return [
        len(records),
        int(records["survived"].sum()),
        float(records["creation_time"].sum()),
        float(records["death_time"].sum()),
        float(records["is_spine"].sum()),
    ]


def _check(records, expected: list[float]) -> None:
    got = _summary(records)
    assert got[:2] == expected[:2]
    assert got[4] == expected[4]
    assert got[2:4] == pytest.approx(expected[2:4], rel=1e-12)


def _parent_id_sum(records) -> int:
    """Sum of the parent ids, which pins the genealogy as well as the times."""
    return int(records["parent_id"].astype(float).sum())


def _check_with_genealogy(records, expected: list[float], parent_id_sum: int) -> None:
    _check(records, expected)
    assert _parent_id_sum(records) == parent_id_sum


class TestSeededOutputs:
    @pytest.mark.parametrize(
        ("seed", "expected"),
        [
            (1, [169, 4, 1219.1315482308883, 1333.5796881079948, 1.0]),
            (2, [98, 24, 1047.767986043143, 776.7077183066651, 1.0]),
        ],
    )
    def test_constant_contour(self, seed: int, expected: list[float]) -> None:
        tree = simulate_birth_death_process_from_contour_process(
            0.9, 1.0, 15.0, random_seed=seed
        )
        _check(tree.records, expected)

    @pytest.mark.parametrize(
        ("seed", "expected"),
        [
            (1, [35, 2, 94.77814507692689, 115.229770030539, 1.0]),
            (2, [69, 7, 255.3754298349648, 261.45144086413035, 1.0]),
        ],
    )
    def test_linear_contour(self, seed: int, expected: list[float]) -> None:
        tree = simulate_linear_rate_contour_process(BIRTH, DEATH, random_seed=seed)
        _check(tree.records, expected)

    def test_constant_splitting(self) -> None:
        result = simulate_subcritical_survival_splitting(
            SplittingConfig(
                1.0, 2.0, 8.0, n_particles=200, random_seed=1, max_restarts=500
            )
        )
        assert [
            result.n_attempts,
            len(result.replicates),
            result.final_survivor_count,
        ] == [
            5,
            8,
            3,
        ]
        _check(
            result.conditioned_trajectories[0].process.records,
            [28, 1, 121.11721121713245, 132.32771273325469, 1.0],
        )

    def test_linear_splitting(self) -> None:
        result = simulate_linear_survival_splitting(
            LinearSplittingConfig(
                BIRTH, DEATH, n_particles=200, random_seed=2, max_restarts=500
            )
        )
        assert [
            result.n_attempts,
            len(result.replicates),
            result.final_survivor_count,
        ] == [
            5,
            3,
            3,
        ]
        _check(
            result.conditioned_trajectories[0].process.records,
            [26, 1, 61.04204663194287, 74.9515368674886, 1.0],
        )


# Pins for the simulators that had none.  The values were recorded from the code
# as it was before the codebase review, so that its refactor can be checked.

FORWARD_BIRTH = np.array([1.2, 0.8])
FORWARD_DEATH = np.array([0.4, 0.5])
FORWARD_TIMES = np.array([2.0, 4.0])
LINEAR_BIRTH = PiecewiseLinear(
    np.array([1.0, 0.8]), np.array([0.8, 0.2]), np.array([3.0, 6.0])
)
LINEAR_DEATH = PiecewiseLinear(np.array([0.3]), np.array([1.1]), np.array([6.0]))


class TestSeededForwardOutputs:
    @pytest.mark.parametrize(
        ("seed", "expected", "parent_id_sum"),
        [
            (1, [16, 7, 30.737271931419592, 23.477966775788325, 1.0], 68),
            (2, [12, 7, 27.57141350097904, 12.361988080781476, 1.0], 38),
        ],
    )
    def test_gillespie(
        self, seed: int, expected: list[float], parent_id_sum: int
    ) -> None:
        process = simulate_birth_death_process(
            FORWARD_BIRTH, FORWARD_DEATH, FORWARD_TIMES, random_seed=seed
        )
        _check_with_genealogy(process.records, expected, parent_id_sum)

    @pytest.mark.parametrize(
        ("seed", "expected", "parent_id_sum", "birth_epoch_sum"),
        [
            # Seed 1 goes extinct, so this pins the no-spine case too.
            (1, [5, 0, 3.270825749036394, 6.783972028255517, 0.0], 2, 0),
            (2, [8, 3, 11.490354257091717, 9.452960978580496, 1.0], 9, 1),
        ],
    )
    def test_lifespans(
        self,
        seed: int,
        expected: list[float],
        parent_id_sum: int,
        birth_epoch_sum: int,
    ) -> None:
        process = simulate_birth_death_lifespans(
            FORWARD_BIRTH, FORWARD_DEATH, FORWARD_TIMES, random_seed=seed
        )
        _check_with_genealogy(process.records, expected, parent_id_sum)
        assert int(process.records["birth_epoch"].sum()) == birth_epoch_sum

    @pytest.mark.parametrize(
        ("seed", "expected", "parent_id_sum"),
        [
            (1, [18, 2, 45.09248583023237, 63.024201157719936, 1.0], 88),
            (2, [19, 3, 60.73872011813883, 69.69757615203609, 1.0], 85),
        ],
    )
    def test_linear(self, seed: int, expected: list[float], parent_id_sum: int) -> None:
        process = simulate_birth_death_linear(
            LINEAR_BIRTH, LINEAR_DEATH, random_seed=seed
        )
        _check_with_genealogy(process.records, expected, parent_id_sum)


class TestSeededConditionedContourOutputs:
    @pytest.mark.parametrize(
        ("seed", "expected", "parent_id_sum"),
        [
            (1, [36, 3, 100.62110826343981, 115.229770030539, 1.0], 536),
            (2, [17, 3, 42.839345506614464, 39.87996018421805, 1.0], 79),
        ],
    )
    def test_given_population(
        self, seed: int, expected: list[float], parent_id_sum: int
    ) -> None:
        tree = simulate_contour_given_population(BIRTH, DEATH, 3, random_seed=seed)
        _check_with_genealogy(tree.records, expected, parent_id_sum)

    @pytest.mark.parametrize(
        ("seed", "expected", "parent_id_sum"),
        [
            (1, [22, 3, 48.293128295198066, 53.1805059002459, 1.0], 171),
            (2, [76, 3, 182.90537902967893, 230.03008820965204, 1.0], 2551),
        ],
    )
    def test_given_ages(
        self, seed: int, expected: list[float], parent_id_sum: int
    ) -> None:
        tree = simulate_contour_given_ages(
            BIRTH, DEATH, [1.5, 3.0, 4.5], random_seed=seed
        )
        _check_with_genealogy(tree.records, expected, parent_id_sum)


class TestSeededEraConditioningOutputs:
    @pytest.mark.parametrize(
        ("seed", "samples", "n_filter_passing"),
        [
            # (n_born, n_alive_present, per_era_survivors) of each sample.
            (1, [(4, 4, [1, 3]), (5, 4, [1, 3])], 4),
            (2, [(4, 1, [1, 0]), (11, 6, [1, 5])], 5),
        ],
    )
    def test_conditioned_survival_process(
        self,
        seed: int,
        samples: list[tuple[int, int, list[int]]],
        n_filter_passing: int,
    ) -> None:
        result = sample_conditioned_survival_process(
            [0.6, 0.5],
            [0.2, 0.3],
            [3.0, 6.0],
            np.array([[1, 6], [0, 2**31 - 1]]),
            n_samples=2,
            split_factor=3,
            n_initial=8,
            random_seed=seed,
            max_population=2000,
        )
        assert result.n_outer_iterations == 1
        assert result.n_total_era0_particles == 8
        assert [
            (s.n_born, s.n_alive_present, s.per_era_survivors.tolist())
            for s in result.samples
        ] == samples
        assert [(d.n_entering, d.n_passing_filter) for d in result.diagnostics] == [
            (8, n_filter_passing)
        ]
