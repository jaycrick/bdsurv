"""Pin seeded outputs, so performance work cannot silently change the random stream.

The expected values were recorded before the speed-ups (inline ``isclose``,
copy-on-write particle records, scalar hazard inversion), which must not alter
any seeded result.
"""

import numpy as np
import pytest

from bdsurv import (
    PiecewiseConstant,
    PiecewiseLinear,
    SplittingConfig,
    sample_conditioned_survival_process,
    simulate_birth_death,
    simulate_birth_death_lifespans,
    simulate_contour_given_ages,
    simulate_contour_given_population,
    simulate_contour_given_survival,
    simulate_survival_splitting,
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
            (1, [42, 2, 285.22691705762065, 301.159843050245, 1.0]),
            (2, [107, 8, 858.6619580099759, 838.9802443762617, 1.0]),
        ],
    )
    def test_constant_contour(self, seed: int, expected: list[float]) -> None:
        tree = simulate_contour_given_survival(0.9, 1.0, 15.0, random_seed=seed)
        _check(tree.records, expected)

    @pytest.mark.parametrize(
        ("seed", "expected"),
        [
            (1, [20, 3, 63.61825352429334, 64.86647536153421, 1.0]),
            (2, [29, 1, 90.45824745145643, 106.67347497445284, 1.0]),
        ],
    )
    def test_linear_contour(self, seed: int, expected: list[float]) -> None:
        tree = simulate_contour_given_survival(BIRTH, DEATH, random_seed=seed)
        _check(tree.records, expected)

    def test_constant_splitting(self) -> None:
        result = simulate_survival_splitting(
            SplittingConfig(
                birth_rate=1.0,
                death_rate=2.0,
                terminal_time=8.0,
                n_particles=200,
                random_seed=1,
                max_restarts=500,
            )
        )
        assert [
            result.n_attempts,
            len(result.replicates),
            result.final_survivor_count,
        ] == [
            1,
            8,
            12,
        ]
        _check(
            result.conditioned_trajectories[0].process.records,
            [19, 1, 77.14673158264206, 83.79439924882608, 1.0],
        )

    def test_linear_splitting(self) -> None:
        result = simulate_survival_splitting(
            SplittingConfig(
                birth_rate=BIRTH,
                death_rate=DEATH,
                n_particles=200,
                random_seed=2,
                max_restarts=500,
            )
        )
        assert [
            result.n_attempts,
            len(result.replicates),
            result.final_survivor_count,
        ] == [
            3,
            3,
            6,
        ]
        _check(
            result.conditioned_trajectories[0].process.records,
            [34, 1, 79.25211379095654, 101.20468192704998, 1.0],
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
            (1, [7, 2, 12.965153575756315, 12.393864627083243, 1.0], 7),
            (2, [6, 0, 11.116018231543297, 17.866882013197998, 0.0], 7),
        ],
    )
    def test_forward_epochs(
        self, seed: int, expected: list[float], parent_id_sum: int
    ) -> None:
        process = simulate_birth_death(
            PiecewiseConstant(FORWARD_BIRTH, FORWARD_TIMES),
            PiecewiseConstant(FORWARD_DEATH, FORWARD_TIMES),
            random_seed=seed,
        )
        _check_with_genealogy(process.records, expected, parent_id_sum)

    @pytest.mark.parametrize(
        ("seed", "expected", "parent_id_sum", "birth_epoch_sum"),
        [
            (1, [11, 7, 24.141982974716907, 11.006915536528556, 1.0], 37, 6),
            (2, [6, 1, 6.559275103268879, 8.213494626179076, 1.0], 7, 1),
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
            (1, [9, 0, 21.636116034544536, 36.92982001999652, 0.0], 13),
            (2, [9, 1, 24.269645880678187, 33.589325318372644, 1.0], 8),
        ],
    )
    def test_linear(self, seed: int, expected: list[float], parent_id_sum: int) -> None:
        process = simulate_birth_death(LINEAR_BIRTH, LINEAR_DEATH, random_seed=seed)
        _check_with_genealogy(process.records, expected, parent_id_sum)


class TestSeededConditionedContourOutputs:
    @pytest.mark.parametrize(
        ("seed", "expected", "parent_id_sum"),
        [
            (1, [20, 3, 63.61825352429334, 64.86647536153421, 1.0], 129),
            (2, [35, 3, 122.32761498642452, 128.75144934188975, 1.0], 447),
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
            (1, [22, 3, 64.5884897373746, 67.97628203765409, 1.0], 158),
            (2, [23, 3, 61.83765057964648, 66.82969077054324, 1.0], 153),
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
            (1, [(9, 6, [2, 4]), (9, 6, [2, 4])], 8),
            (2, [(15, 11, [3, 8]), (23, 15, [3, 12])], 3),
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
