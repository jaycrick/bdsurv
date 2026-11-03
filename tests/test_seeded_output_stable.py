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
    simulate_birth_death_process_from_contour_process,
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
