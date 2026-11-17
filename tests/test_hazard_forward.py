"""Test the forward simulators with Hazard rates (competing-risks path)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from _helpers import assert_valid_records
from scipy import stats

import bdsurv
from bdsurv import SplittingConfig, simulate_birth_death, simulate_survival_splitting
from bdsurv._forward import RatePair, build_dynamics, simulate_forward
from bdsurv.hazards import GompertzHazard, LomaxHazard, PowerHazard
from bdsurv.rates import resolve_rates

DTYPES = {
    "individual_id": "int64",
    "parent_id": "float64",
    "creation_time": "float64",
    "death_time": "float64",
    "is_spine": "bool",
    "survived": "bool",
    "age": "float64",
    "lifespan": "float64",
}
N_SEEDS = 2000


def _final_sizes(birth, death, t: float, n: int = N_SEEDS) -> np.ndarray:
    return np.array(
        [
            simulate_birth_death(birth, death, t, random_seed=s)
            .records["survived"]
            .sum()
            for s in range(n)
        ],
        dtype=float,
    )


def test_build_dynamics_dispatch() -> None:
    """Test build dynamics dispatch."""
    birth, death, horizon = resolve_rates(1.0, 0.5, 2.0)
    assert isinstance(build_dynamics(birth, death, horizon), list)
    birth, death, horizon = resolve_rates(GompertzHazard(1.0, 0.0), 0.5, 2.0)
    assert isinstance(build_dynamics(birth, death, horizon), RatePair)
    _, dynamics = simulate_forward(
        birth,
        death,
        horizon,
        1,
        ensure_non_extinct=False,
        max_retries=0,
        name="test",
    )
    assert isinstance(dynamics, RatePair)


def test_weibull_birth_gompertz_death_runs_and_is_reproducible() -> None:
    """Test weibull birth gompertz death runs and is reproducible."""
    birth, death = PowerHazard.weibull(1.5, 1.0), GompertzHazard(0.3, 0.5)
    for seed in range(10):
        a = simulate_birth_death(birth, death, 3.0, random_seed=seed)
        b = simulate_birth_death(birth, death, 3.0, random_seed=seed)
        assert a.records.equals(b.records)
        assert a.birth_rate is birth
        assert a.death_rate is death
        assert a.terminal_time == 3.0
        assert_valid_records(a.records, 3.0, expected_dtypes=DTYPES)


def test_ensure_non_extinct_with_hazards() -> None:
    """Test ensure non extinct with hazards."""
    birth, death = PowerHazard.weibull(1.5, 1.0), GompertzHazard(0.3, 0.5)
    tree = simulate_birth_death(
        birth, death, 3.0, random_seed=4, ensure_non_extinct=True
    )
    assert tree.records["survived"].sum() >= 1


def test_generic_path_matches_linear_path_in_distribution() -> None:
    """Test generic path matches linear path in distribution."""
    """Gompertz with b = 0 (generic path) agrees with float rates (segment path)."""
    lam, mu, t = 1.0, 0.6, 2.0
    generic = _final_sizes(GompertzHazard(lam, 0.0), GompertzHazard(mu, 0.0), t)
    linear = _final_sizes(lam, mu, t)
    se = math.sqrt(generic.var(ddof=1) / N_SEEDS + linear.var(ddof=1) / N_SEEDS)
    assert abs(generic.mean() - linear.mean()) < 5 * se
    p = (generic > 0).mean(), (linear > 0).mean()
    survival = math.exp(bdsurv.log_survival_probability(lam, mu, t, terminal_time=t))
    for frac in p:
        assert abs(frac - survival) < 5 * math.sqrt(survival * (1 - survival) / N_SEEDS)
    # E[N_T] = exp((lam - mu) t) for constant rates
    expected = math.exp((lam - mu) * t)
    for sizes in (generic, linear):
        assert abs(sizes.mean() - expected) < 5 * sizes.std(ddof=1) / math.sqrt(N_SEEDS)


def test_time_varying_survival_matches_theory() -> None:
    """Test time varying survival matches theory."""
    birth, death, t = PowerHazard.weibull(2.0, 1.5), LomaxHazard(1.5, 1.0), 3.0
    sizes = _final_sizes(birth, death, t)
    survival = math.exp(
        bdsurv.log_survival_probability(birth, death, t, terminal_time=t)
    )
    frac = (sizes > 0).mean()
    assert abs(frac - survival) < 5 * math.sqrt(survival * (1 - survival) / N_SEEDS)


def test_splitting_accepts_hazards() -> None:
    """Test splitting accepts hazards."""
    birth, death = PowerHazard.weibull(2.0, 1.5), LomaxHazard(1.5, 1.0)
    result = simulate_survival_splitting(
        SplittingConfig(
            birth_rate=birth,
            death_rate=death,
            terminal_time=3.0,
            n_particles=100,
            random_seed=2,
            n_conditioned_trajectories=2,
            max_restarts=200,
        )
    )
    assert result.conditioned_trajectories
    for trajectory in result.conditioned_trajectories:
        tree = trajectory.process
        assert tree.birth_rate is birth
        assert tree.death_rate is death
        assert tree.terminal_time == 3.0
        assert tree.records["survived"].sum() >= 1


@pytest.mark.slow
def test_contour_given_survival_matches_rejection_sampled_forward() -> None:
    """Test contour given survival matches rejection sampled forward."""
    birth, death, t = PowerHazard.weibull(1.5, 1.0), GompertzHazard(0.3, 0.5), 3.0
    n = 3000
    reference = [
        int(
            simulate_birth_death(
                birth, death, t, random_seed=10**6 + i, ensure_non_extinct=True
            )
            .records["survived"]
            .sum()
        )
        for i in range(n)
    ]
    sizes = [
        int(
            bdsurv.simulate_contour_given_survival(birth, death, t, random_seed=i)
            .records["survived"]
            .sum()
        )
        for i in range(n)
    ]
    pvalue = stats.ks_2samp(sizes, reference).pvalue
    print(f"KS p-value {pvalue:.4g}")
    assert pvalue > 0.001, (
        f"two-sample KS test of population sizes against the rejection-sampled "
        f"reference: p-value {pvalue:.3g} <= threshold 0.001"
    )
