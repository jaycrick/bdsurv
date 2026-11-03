"""Test survival-conditioned simulation with linear-in-time rates."""

import math

import numpy as np
import pytest
from scipy import integrate, stats

from bdsurv import (
    LinearSplittingConfig,
    PiecewiseConstant,
    PiecewiseLinear,
    simulate_birth_death_linear,
    simulate_linear_rate_contour_process,
    simulate_linear_survival_splitting,
)
from bdsurv import _linear_math as lm
from bdsurv.splitting import _log_survival_probability

T = 4.0


def _const(value: float, end: float = T) -> PiecewiseConstant:
    return PiecewiseConstant(np.array([value]), np.array([end]))


def _line(start: float, end_value: float, end: float = T) -> PiecewiseLinear:
    return PiecewiseLinear(np.array([start]), np.array([end_value]), np.array([end]))


ARLEY = (_const(1.0), _line(0.0, 0.8 * T))
VARYING_BIRTH = (_line(2.0, 0.5), _const(1.0))


class TestMaths:
    @pytest.mark.parametrize(
        ("lam", "mu", "t"),
        [(1.0, 0.6, 21.5), (1.0, 1.0, 100.0), (1.0, 3.0, 50.0), (1.0, 10.0, 100.0)],
    )
    def test_matches_constant_rate_closed_forms(
        self, lam: float, mu: float, t: float
    ) -> None:
        birth = lm.as_rate(_const(lam, t))
        death = lm.as_rate(_const(mu, t))
        assert lm.log_survival_probability(birth, death, t) == pytest.approx(
            _log_survival_probability(birth_rate=lam, death_rate=mu, time=t)
        )
        growth = lam - mu
        expected = (
            1.0 + lam * t
            if abs(growth) < 1e-12
            else (mu - lam * math.exp(growth * t)) / (mu - lam)
        )
        assert lm.mean_cumulative_individuals(birth, death, t) == pytest.approx(
            expected
        )

    def test_arley_matches_quadrature(self) -> None:
        slope = 0.3
        birth = lm.as_rate(_const(1.0, 10.0))
        death = lm.as_rate(_line(0.0, slope * 10.0, 10.0))
        cumulative, _ = integrate.quad(lambda s: math.exp(s - slope * s * s / 2), 0, 10)
        assert lm.mean_cumulative_individuals(birth, death, 10.0) == pytest.approx(
            1.0 + cumulative
        )
        i_of_t, _ = integrate.quad(
            lambda s: slope * s * math.exp(slope * s * s / 2 - s), 0, 10
        )
        assert lm.log_survival_probability(birth, death, 10.0) == pytest.approx(
            -math.log1p(i_of_t)
        )

    def test_survival_probability_edge_cases(self) -> None:
        birth, death = (lm.as_rate(r) for r in ARLEY)
        assert lm.log_survival_probability(birth, death, 0.0) == 0.0
        zero = lm.as_rate(_const(0.0))
        assert lm.log_survival_probability(birth, zero, T) == 0.0

    def test_piecewise_rate_validation(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            lm.as_rate(_const(-1.0))
        with pytest.raises(ValueError, match="finite"):
            lm.as_rate(_const(math.inf))
        with pytest.raises(ValueError, match="increasing"):
            lm.as_rate(PiecewiseConstant(np.array([1.0, 1.0]), np.array([2.0, 1.0])))
        with pytest.raises(ValueError, match="same shape"):
            lm.as_rate(PiecewiseConstant(np.array([1.0]), np.array([1.0, 2.0])))

    def test_hazard_inverses_round_trip(self) -> None:
        rate = lm.as_rate(
            PiecewiseLinear(
                np.array([0.5, 1.0]), np.array([1.5, 0.2]), np.array([2.0, 5.0])
            )
        )
        for start, hazard in [(0.3, 0.7), (1.9, 1.2), (0.0, 3.0)]:
            s = lm.forward_hazard_time(rate, start, hazard, cap=5.0)
            assert s < 5.0
            got = lm.cumulative(rate, s) - lm.cumulative(rate, start)
            assert got == pytest.approx(hazard)
        for start, hazard in [(4.0, 0.7), (4.9, 2.5), (1.0, 0.4)]:
            s = lm.backward_hazard_time(rate, start, hazard)
            assert s > 0.0
            got = lm.cumulative(rate, start) - lm.cumulative(rate, s)
            assert got == pytest.approx(hazard)

    def test_hazard_exhausted(self) -> None:
        rate = lm.as_rate(_const(1.0, 5.0))
        assert lm.forward_hazard_time(rate, 4.0, 3.0, cap=5.0) == 5.0
        assert lm.backward_hazard_time(rate, 1.0, 3.0) == 0.0
        assert lm.forward_hazard_time(rate, 5.0, 1.0, cap=5.0) == 5.0
        assert lm.value(rate, 2.0) == 1.0


def _check_tree(tree) -> None:
    records = tree.records
    assert records["survived"].sum() >= 1
    assert records["is_spine"].sum() == 1
    assert bool(records.loc[records["is_spine"], "survived"].all())
    created = records.set_index("individual_id")["creation_time"]
    children = records.dropna(subset=["parent_id"])
    parent_times = created.loc[children["parent_id"].astype(int)].to_numpy()
    assert np.all(parent_times <= children["creation_time"].to_numpy())
    assert np.all(records["creation_time"] <= tree.terminal_time)


class TestContour:
    @pytest.mark.parametrize("rates", [ARLEY, VARYING_BIRTH])
    def test_structure(self, rates) -> None:
        for seed in range(20):
            tree = simulate_linear_rate_contour_process(*rates, random_seed=seed)
            assert tree.terminal_time == T
            assert math.isnan(tree.birth_rate)
            assert tree.birth_rate_function is rates[0]
            _check_tree(tree)

    def test_reproducible(self) -> None:
        a = simulate_linear_rate_contour_process(*ARLEY, random_seed=3)
        b = simulate_linear_rate_contour_process(*ARLEY, random_seed=3)
        assert a.records.equals(b.records)

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="same end time"):
            simulate_linear_rate_contour_process(_const(1.0, 2.0), _const(1.0, 3.0))
        with pytest.raises(ValueError, match="identically zero"):
            simulate_linear_rate_contour_process(_const(1.0), _const(0.0))

    def test_zero_birth_rate_gives_single_individual(self) -> None:
        tree = simulate_linear_rate_contour_process(
            _const(0.0), _const(0.2), random_seed=1
        )
        assert len(tree.records) == 1


class TestSplitting:
    @pytest.mark.parametrize("rates", [ARLEY, VARYING_BIRTH])
    def test_structure(self, rates) -> None:
        config = LinearSplittingConfig(
            *rates, n_particles=200, random_seed=1, max_restarts=500
        )
        assert config.terminal_time == T
        result = simulate_linear_survival_splitting(config)
        assert result.conditioned_trajectories
        for trajectory in result.conditioned_trajectories:
            _check_tree(trajectory.process)
            assert trajectory.times[-1] == T

    def test_levels_for_hard_case(self) -> None:
        birth, death = _const(1.0, 10.0), _line(0.0, 6.0, 10.0)
        config = LinearSplittingConfig(
            birth, death, n_particles=500, random_seed=2, max_restarts=2000
        )
        result = simulate_linear_survival_splitting(config)
        assert len(result.replicates) > 1
        assert result.final_survivor_count > 0
        assert np.isclose(result.level_times[-1], 10.0)

    def test_multi_epoch_rates(self) -> None:
        birth = PiecewiseConstant(np.array([1.0, 0.5]), np.array([1.5, T]))
        death = PiecewiseLinear(np.array([0.2]), np.array([1.0]), np.array([T]))
        result = simulate_linear_survival_splitting(
            LinearSplittingConfig(birth, death, n_particles=100, random_seed=4)
        )
        assert result.conditioned_trajectories

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="same end time"):
            simulate_linear_survival_splitting(
                LinearSplittingConfig(_const(1.0, 2.0), _const(1.0, 3.0), 10)
            )
        with pytest.raises(ValueError, match="n_particles"):
            simulate_linear_survival_splitting(LinearSplittingConfig(*ARLEY, 0))
        with pytest.raises(ValueError, match="max_restarts"):
            simulate_linear_survival_splitting(
                LinearSplittingConfig(*ARLEY, 10, max_restarts=0)
            )

    def test_manual_levels_and_replicates(self) -> None:
        result = simulate_linear_survival_splitting(
            LinearSplittingConfig(
                *ARLEY,
                n_particles=200,
                level_times=[0.0, 2.0, T],
                replicates=[4, 3],
                random_seed=5,
                max_restarts=200,
            )
        )
        assert list(result.replicates) == [4, 3]


def _reference_sizes(rates, n: int, seed: int) -> list[int]:
    sizes: list[int] = []
    i = 0
    while len(sizes) < n:
        process = simulate_birth_death_linear(*rates, random_seed=seed + i)
        i += 1
        survivors = int(process.records["survived"].sum())
        if survivors:
            sizes.append(survivors)
    return sizes


@pytest.mark.slow
class TestAgainstForwardSimulation:
    """Population at T given survival matches rejection-sampled forward runs."""

    @pytest.mark.parametrize("rates", [ARLEY, VARYING_BIRTH])
    def test_contour(self, rates) -> None:
        reference = _reference_sizes(rates, 3000, 10**6)
        sizes = [
            int(
                simulate_linear_rate_contour_process(*rates, random_seed=i)
                .records["survived"]
                .sum()
            )
            for i in range(3000)
        ]
        assert stats.ks_2samp(sizes, reference).pvalue > 0.001

    @pytest.mark.parametrize("rates", [ARLEY, VARYING_BIRTH])
    def test_splitting(self, rates) -> None:
        reference = _reference_sizes(rates, 1500, 2 * 10**6)
        sizes = []
        for i in range(300):
            result = simulate_linear_survival_splitting(
                LinearSplittingConfig(
                    *rates,
                    n_particles=1000,
                    random_seed=i,
                    n_conditioned_trajectories=1,
                    max_restarts=500,
                )
            )
            sizes.append(
                int(
                    result.conditioned_trajectories[0].process.records["survived"].sum()
                )
            )
        assert stats.ks_2samp(sizes, reference).pvalue > 0.001

    def test_constant_rates_through_linear_path(self) -> None:
        rates = (_const(1.0), _const(1.5))
        reference = _reference_sizes(rates, 3000, 3 * 10**6)
        sizes = [
            int(
                simulate_linear_rate_contour_process(*rates, random_seed=i)
                .records["survived"]
                .sum()
            )
            for i in range(3000)
        ]
        assert stats.ks_2samp(sizes, reference).pvalue > 0.001
