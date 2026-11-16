"""Test the public survival probability against analytic and numerical values."""

import math

import numpy as np
import pytest
from scipy import integrate

import bdsurv
from bdsurv import (
    PiecewiseConstant,
    PiecewiseLinear,
    log_mean_cumulative_individuals,
    log_survival_probability,
)


def _extinction(birth_rate: float, death_rate: float, terminal_time: float) -> float:
    """Extinction probability by ``terminal_time`` from the survival log."""
    return -math.expm1(log_survival_probability(birth_rate, death_rate, terminal_time))


class TestExtinctionProbability:
    """Validate ``-expm1(log_survival_probability(...))`` against analytic values."""

    def test_pure_death_process_matches_exponential_cdf(self) -> None:
        # With birth_rate -> 0, a single individual's extinction probability by
        # T is exactly 1 - exp(-death_rate * T) (simple exponential survival).
        death_rate = 0.7
        terminal_time = 3.0
        expected = 1.0 - math.exp(-death_rate * terminal_time)
        actual = _extinction(1e-12, death_rate, terminal_time)
        assert actual == pytest.approx(expected, rel=1e-6)

    def test_probability_is_between_zero_and_one(self) -> None:
        for birth_rate, death_rate, terminal_time in [
            (0.1, 1.0, 5.0),
            (0.9, 1.0, 50.0),
            (1.0, 10.0, 20.0),  # deeply subcritical: pr saturates to 1.0 in float64
            (5.0, 0.5, 1.0),  # supercritical: still well-defined
        ]:
            pr = _extinction(birth_rate, death_rate, terminal_time)
            assert 0.0 <= pr <= 1.0

    def test_monotone_increasing_in_terminal_time(self) -> None:
        birth_rate, death_rate = 0.8, 1.0
        times = [1.0, 5.0, 20.0, 100.0]
        probs = [_extinction(birth_rate, death_rate, t) for t in times]
        assert probs == sorted(probs)

    def test_critical_case_matches_known_formula(self) -> None:
        # At lambda == mu, P(extinction by T) = (mu*T) / (1 + mu*T).
        rate = 0.6
        terminal_time = 4.0
        expected = (rate * terminal_time) / (1.0 + rate * terminal_time)
        actual = _extinction(rate, rate, terminal_time)
        assert actual == pytest.approx(expected, rel=1e-9)

    def test_continuous_across_critical_threshold(self) -> None:
        # The critical-case branch (|x| < 1e-10) must not create a
        # discontinuity relative to the general formula just outside it.
        death_rate = 1.0
        terminal_time = 10.0
        just_below = _extinction(death_rate - 1e-8, death_rate, terminal_time)
        at_critical = _extinction(death_rate, death_rate, terminal_time)
        just_above = _extinction(death_rate + 1e-8, death_rate, terminal_time)
        assert just_below == pytest.approx(at_critical, abs=1e-6)
        assert just_above == pytest.approx(at_critical, abs=1e-6)


class TestRateForms:
    def test_float_and_piecewise_constant_agree(self) -> None:
        birth, death, horizon = 0.9, 1.3, 7.0
        as_floats = log_survival_probability(birth, death, 5.0, terminal_time=horizon)
        as_epochs = log_survival_probability(
            PiecewiseConstant([birth], [horizon]),
            PiecewiseConstant([death], [horizon]),
            5.0,
        )
        assert as_epochs == pytest.approx(as_floats, rel=1e-9)

    def test_float_defaults_the_horizon_to_t(self) -> None:
        assert log_survival_probability(0.9, 1.3, 5.0) == log_survival_probability(
            0.9, 1.3, 5.0, terminal_time=5.0
        )

    def test_float_with_a_rate_like(self) -> None:
        death = PiecewiseConstant([1.3], [7.0])
        mixed = log_survival_probability(0.9, death, 7.0)
        floats = log_survival_probability(0.9, 1.3, 7.0)
        assert mixed == pytest.approx(floats, rel=1e-9)

    def test_piecewise_linear_matches_quadrature_of_the_formula(self) -> None:
        birth = PiecewiseLinear([1.0, 0.8], [0.8, 0.2], [3.0, 6.0])
        death = PiecewiseLinear([0.3], [1.1], [6.0])

        def lam(s: float) -> float:
            return float(np.interp(s, [0.0, 3.0, 6.0], [1.0, 0.8, 0.2]))

        # The first epoch is linear from 1.0 to 0.8, the second from 0.8 to 0.2.
        def mu(s: float) -> float:
            return 0.3 + (1.1 - 0.3) * s / 6.0

        def gamma(s: float) -> float:
            return integrate.quad(lambda x: lam(x) - mu(x), 0.0, s, points=[3.0])[0]

        t = 5.0
        i_t, _ = integrate.quad(
            lambda s: mu(s) * math.exp(-gamma(s)), 0.0, t, points=[3.0]
        )
        expected = -math.log1p(i_t)
        actual = log_survival_probability(birth, death, t)
        assert actual == pytest.approx(expected, rel=1e-7)

    def test_zero_rates_are_allowed(self) -> None:
        assert log_survival_probability(0.0, 0.5, 2.0) == pytest.approx(
            -math.log1p(math.expm1(1.0)), rel=1e-9
        )
        assert log_survival_probability(1.0, 0.0, 2.0) == pytest.approx(0.0, abs=1e-12)


class TestRange:
    def test_t_zero_is_certain_survival_for_rate_likes(self) -> None:
        rate = PiecewiseConstant([1.0], [3.0])
        assert log_survival_probability(rate, rate, 0.0) == 0.0

    def test_t_zero_is_certain_survival_for_floats(self) -> None:
        assert log_survival_probability(1.0, 1.5, 0.0) == 0.0
        assert log_survival_probability(1.0, 1.5, 0.0, terminal_time=4.0) == 0.0
        assert log_survival_probability(0.0, 0.0, 0.0) == 0.0

    @pytest.mark.parametrize("bad", [-1.0, math.nan, math.inf])
    def test_t_zero_still_validates_float_rates(self, bad: float) -> None:
        with pytest.raises(ValueError, match="rate values"):
            log_survival_probability(bad, 1.5, 0.0)
        with pytest.raises(ValueError, match="rate values"):
            log_survival_probability(1.0, bad, 0.0)

    @pytest.mark.parametrize("t", [-0.1, math.nan, math.inf])
    def test_bad_t_raises_naming_t(self, t: float) -> None:
        with pytest.raises(ValueError, match=r"t must"):
            log_survival_probability(1.0, 1.5, t)
        rate = PiecewiseConstant([1.0], [3.0])
        with pytest.raises(ValueError, match=r"t must"):
            log_survival_probability(rate, rate, t)

    def test_t_beyond_the_horizon_raises(self) -> None:
        rate = PiecewiseConstant([1.0], [3.0])
        with pytest.raises(ValueError, match="t must lie in"):
            log_survival_probability(rate, rate, 3.5)

    def test_t_beyond_an_explicit_terminal_time_raises(self) -> None:
        with pytest.raises(ValueError, match="t must lie in"):
            log_survival_probability(1.0, 2.0, 5.0, terminal_time=4.0)

    def test_terminal_time_must_match_the_rate_likes(self) -> None:
        rate = PiecewiseConstant([1.0], [3.0])
        with pytest.raises(ValueError, match="terminal_time"):
            log_survival_probability(rate, rate, 2.0, terminal_time=4.0)


class TestMeanCumulativeIndividuals:
    @pytest.mark.parametrize(
        ("lam", "mu", "t"), [(1.3, 0.4, 2.5), (0.5, 1.2, 4.0), (0.9, 0.9, 3.0)]
    )
    def test_constant_rates_match_the_closed_form(
        self, lam: float, mu: float, t: float
    ) -> None:
        if lam == mu:
            expected = math.log(1.0 + lam * t)
        else:
            expected = math.log(1.0 + lam * math.expm1((lam - mu) * t) / (lam - mu))
        actual = log_mean_cumulative_individuals(lam, mu, t)
        assert actual == pytest.approx(expected, rel=1e-9), "closed form mismatch"

    def test_zero_birth_rate_gives_exactly_zero(self) -> None:
        assert log_mean_cumulative_individuals(0.0, 0.7, 3.0) == 0.0, (
            "only the root is ever born"
        )

    def test_float_and_piecewise_constant_agree(self) -> None:
        birth, death, t = 1.1, 0.6, 2.0
        as_float = log_mean_cumulative_individuals(birth, death, t)
        as_epoch = log_mean_cumulative_individuals(
            PiecewiseConstant([birth], [t]), PiecewiseConstant([death], [t]), t
        )
        assert as_epoch == pytest.approx(as_float, rel=1e-12), "forms disagree"

    def test_piecewise_linear_matches_quadrature_of_the_formula(self) -> None:
        birth = PiecewiseLinear([1.0, 0.8], [0.8, 0.2], [3.0, 6.0])
        death = PiecewiseLinear([0.3], [1.1], [6.0])

        def lam(s: float) -> float:
            return float(np.interp(s, [0.0, 3.0, 6.0], [1.0, 0.8, 0.2]))

        def mu(s: float) -> float:
            return 0.3 + (1.1 - 0.3) * s / 6.0

        def gamma(s: float) -> float:
            return integrate.quad(lambda x: lam(x) - mu(x), 0.0, s, points=[3.0])[0]

        t = 5.0
        integral, _ = integrate.quad(
            lambda s: lam(s) * math.exp(gamma(s)), 0.0, t, points=[3.0]
        )
        expected = math.log1p(integral)
        actual = log_mean_cumulative_individuals(birth, death, t)
        assert actual == pytest.approx(expected, rel=1e-8), "quadrature mismatch"

    def test_t_zero_returns_exactly_zero(self) -> None:
        assert log_mean_cumulative_individuals(1.0, 1.5, 0.0) == 0.0, "floats"
        rate = PiecewiseConstant([1.0], [3.0])
        assert log_mean_cumulative_individuals(rate, rate, 0.0) == 0.0, "rate-likes"

    @pytest.mark.parametrize("t", [-0.1, math.nan, math.inf])
    def test_bad_t_raises_naming_t(self, t: float) -> None:
        with pytest.raises(ValueError, match=r"t must"):
            log_mean_cumulative_individuals(1.0, 1.5, t)

    def test_t_beyond_the_horizon_raises(self) -> None:
        rate = PiecewiseConstant([1.0], [3.0])
        with pytest.raises(ValueError, match="t must lie in"):
            log_mean_cumulative_individuals(rate, rate, 3.5)
        with pytest.raises(ValueError, match="t must lie in"):
            log_mean_cumulative_individuals(1.0, 2.0, 5.0, terminal_time=4.0)

    def test_is_exported_from_the_top_level_package(self) -> None:
        assert bdsurv.log_mean_cumulative_individuals is log_mean_cumulative_individuals
        assert "log_mean_cumulative_individuals" in bdsurv.__all__, "not in __all__"
