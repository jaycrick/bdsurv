"""Test the closed-form single-epoch extinction probability."""

import math

import pytest

from bdsurv._extinction import pr_extinction


class TestPrExtinction:
    """Validate ``pr_extinction`` against known analytic values."""

    def test_pure_death_process_matches_exponential_cdf(self) -> None:
        # With birth_rate -> 0, a single individual's extinction probability by
        # T is exactly 1 - exp(-death_rate * T) (simple exponential survival).
        death_rate = 0.7
        terminal_time = 3.0
        expected = 1.0 - math.exp(-death_rate * terminal_time)
        actual = pr_extinction(1e-12, death_rate, terminal_time)
        assert actual == pytest.approx(expected, rel=1e-6)

    def test_probability_is_between_zero_and_one(self) -> None:
        for birth_rate, death_rate, terminal_time in [
            (0.1, 1.0, 5.0),
            (0.9, 1.0, 50.0),
            (1.0, 10.0, 20.0),  # deeply subcritical: pr saturates to 1.0 in float64
            (5.0, 0.5, 1.0),  # supercritical: still well-defined
        ]:
            pr = pr_extinction(birth_rate, death_rate, terminal_time)
            assert 0.0 <= pr <= 1.0

    def test_monotone_increasing_in_terminal_time(self) -> None:
        birth_rate, death_rate = 0.8, 1.0
        times = [1.0, 5.0, 20.0, 100.0]
        probs = [pr_extinction(birth_rate, death_rate, t) for t in times]
        assert probs == sorted(probs)

    def test_critical_case_matches_known_formula(self) -> None:
        # At lambda == mu, P(extinction by T) = (mu*T) / (1 + mu*T).
        rate = 0.6
        terminal_time = 4.0
        expected = (rate * terminal_time) / (1.0 + rate * terminal_time)
        actual = pr_extinction(rate, rate, terminal_time)
        assert actual == pytest.approx(expected, rel=1e-9)

    def test_continuous_across_critical_threshold(self) -> None:
        # The critical-case branch (|x| < 1e-10) must not create a
        # discontinuity relative to the general formula just outside it.
        death_rate = 1.0
        terminal_time = 10.0
        just_below = pr_extinction(death_rate - 1e-8, death_rate, terminal_time)
        at_critical = pr_extinction(death_rate, death_rate, terminal_time)
        just_above = pr_extinction(death_rate + 1e-8, death_rate, terminal_time)
        assert just_below == pytest.approx(at_critical, abs=1e-6)
        assert just_above == pytest.approx(at_critical, abs=1e-6)
