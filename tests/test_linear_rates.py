"""Test the inverse-CDF simulator for linear birth/death rates."""

import math

import numpy as np
import pytest
from _helpers import n_individuals
from scipy import stats

from bdsurv import PiecewiseConstant, PiecewiseLinear, simulate_birth_death_linear
from bdsurv.linear_rates import _next_event_offset


def _linear(start: list[float], end: list[float], times: list[float]):
    return PiecewiseLinear(np.array(start), np.array(end), np.array(times))


def _integral(rate: PiecewiseLinear) -> float:
    """Exact integral of a piecewise-linear rate over ``[0, T]``."""
    ends = np.asarray(rate.epoch_end_times)
    starts = np.concatenate([[0.0], ends[:-1]])
    mean = 0.5 * (np.asarray(rate.start_values) + np.asarray(rate.end_values))
    return float(np.sum(mean * (ends - starts)))


def _mean_size(birth, death, n_runs: int, seed: int = 0) -> tuple[float, float]:
    sizes = np.array(
        [
            n_individuals(
                simulate_birth_death_linear(birth, death, random_seed=seed + i)
            )
            for i in range(n_runs)
        ],
        dtype=float,
    )
    return sizes.mean(), sizes.std(ddof=1) / math.sqrt(n_runs)


class TestSimulateBirthDeathLinear:
    def test_offset_constant_hazard(self) -> None:
        assert _next_event_offset(2.0, 0.0, 1.5) == pytest.approx(0.75)

    @pytest.mark.parametrize("b", [0.7, 1e-9, -0.2])
    def test_offset_inverts_cumulative_hazard(self, b: float) -> None:
        a, e = 1.3, 0.9
        s = _next_event_offset(a, b, e)
        assert a * s + 0.5 * b * s * s == pytest.approx(e)

    def test_offset_matches_closed_form(self) -> None:
        a, b, e = 1.3, 0.7, 0.9
        expected = (-a + math.sqrt(a * a + 2 * b * e)) / b
        assert _next_event_offset(a, b, e) == pytest.approx(expected)

    def test_offset_infinite_when_hazard_exhausted(self) -> None:
        assert _next_event_offset(1.0, -1.0, 5.0) == math.inf

    def test_death_time_distribution(self) -> None:
        birth = _linear([0.0], [0.0], [6.0])
        death = _linear([0.2], [1.4], [6.0])
        times = []
        for i in range(1500):
            proc = simulate_birth_death_linear(birth, death, random_seed=i)
            dead = proc.records.loc[0, "death_time"]
            times.append(6.0 if np.isnan(dead) else dead)

        def cdf(t: float) -> float:
            return 1.0 - math.exp(-(0.2 * t + 0.5 * 0.2 * t * t))

        observed = np.array([t for t in times if t < 6.0])
        # Condition on dying before T: compare to the truncated CDF.
        trunc = stats.kstest(
            observed, lambda x: np.array([cdf(v) / cdf(6.0) for v in x])
        )
        assert trunc.pvalue > 0.001

    @pytest.mark.parametrize(
        ("birth", "death"),
        [
            (_linear([0.3], [0.8], [3.0]), _linear([0.4], [0.4], [3.0])),
            (_linear([0.9], [0.2], [3.0]), _linear([0.1], [0.5], [3.0])),
            (
                _linear([0.4, 0.8], [0.4, 0.2], [1.5, 3.0]),
                _linear([0.2], [0.6], [3.0]),
            ),
        ],
    )
    def test_mean_size_matches_theory(self, birth, death) -> None:
        expected = math.exp(_integral(birth) - _integral(death))
        mean, se = _mean_size(birth, death, 2000)
        assert abs(mean - expected) < 4 * se + 0.02

    def test_constant_rates_match_theory(self) -> None:
        birth = PiecewiseConstant([0.5], [3.0])
        death = PiecewiseConstant([0.3], [3.0])
        mean, se = _mean_size(birth, death, 2000)
        assert abs(mean - math.exp(0.2 * 3.0)) < 4 * se + 0.02

    def test_reproducible(self) -> None:
        birth = _linear([0.5], [0.9], [3.0])
        death = _linear([0.3], [0.3], [3.0])
        a = simulate_birth_death_linear(birth, death, random_seed=7)
        b = simulate_birth_death_linear(birth, death, random_seed=7)
        assert a.records.equals(b.records)

    def test_ensure_non_extinct(self) -> None:
        birth = _linear([0.3], [0.3], [3.0])
        death = _linear([0.6], [0.6], [3.0])
        proc = simulate_birth_death_linear(
            birth, death, random_seed=1, ensure_non_extinct=True
        )
        assert n_individuals(proc) >= 1
        assert proc.records["is_spine"].sum() == 1

    def test_mismatched_end_times_raise(self) -> None:
        with pytest.raises(ValueError, match="same end time"):
            simulate_birth_death_linear(
                _linear([0.5], [0.5], [3.0]), _linear([0.5], [0.5], [4.0])
            )

    def test_negative_retries_raise(self) -> None:
        rate = _linear([0.5], [0.5], [3.0])
        with pytest.raises(ValueError, match="max_retries"):
            simulate_birth_death_linear(
                rate, rate, ensure_non_extinct=True, max_retries=-1
            )
