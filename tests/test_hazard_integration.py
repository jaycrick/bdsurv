"""Test Hazard rates through resolve_rates, the generic quadrature and the contour."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import integrate

import bdsurv
from bdsurv._hazard import log_mean_cumulative_individuals as _log_mean
from bdsurv.hazards import (
    GompertzHazard,
    LogLogisticHazard,
    LomaxHazard,
    PiecewiseHazard,
    PowerHazard,
    QuadraticHazard,
)
from bdsurv.rates import (
    PiecewiseConstant,
    PiecewiseLinear,
    as_rate,
    as_rate_like,
    resolve_rates,
)
from bdsurv.survival import log_survival_probability

PAIRS = {
    "weibull-gompertz": (
        PowerHazard.weibull(shape=0.5, scale=2.0),
        GompertzHazard(0.5, 0.3),
        3.0,
    ),
    "lomax-rayleigh": (LomaxHazard(2.0, 3.0), PowerHazard.rayleigh(0.3), 4.0),
    "loglogistic-quadratic": (
        LogLogisticHazard(0.7, 2.0),
        QuadraticHazard(0.2, 0.1, 0.05),
        3.0,
    ),
}


def _oracle_i(birth, death, t: float) -> float:
    pts = sorted({*birth.singular_points, *death.singular_points} - {0.0, t})

    def gamma(s: float) -> float:
        return float(birth.cumulative_hazard(s) - death.cumulative_hazard(s))

    value, _ = integrate.quad(
        lambda s: float(death.hazard(s)) * math.exp(-gamma(s)),
        0.0,
        t,
        points=pts or None,
        limit=200,
        epsabs=0.0,
        epsrel=1e-12,
    )
    return value


@pytest.mark.parametrize("name", list(PAIRS))
def test_survival_matches_quad_oracle(name: str) -> None:
    """Test survival matches quad oracle."""
    birth, death, t = PAIRS[name]
    got = log_survival_probability(birth, death, t, terminal_time=t)
    want = -math.log1p(_oracle_i(birth, death, t))
    assert got == pytest.approx(want, rel=1e-8)


def test_mean_born_with_singular_weight_matches_quad() -> None:
    """Test mean born with singular weight matches quad."""
    birth, death, t = PAIRS["weibull-gompertz"]

    def f(s: float) -> float:
        g = float(birth.cumulative_hazard(s) - death.cumulative_hazard(s))
        return float(birth.hazard(s)) * math.exp(g)

    value, _ = integrate.quad(f, 0.0, t, limit=200, epsabs=0.0, epsrel=1e-12)
    got = _log_mean(birth, death, t)
    # Loosened from 1e-8: the s**-0.5 singularity leaves ~1.4e-8 in the
    # innermost graded panel (40 levels).
    assert got == pytest.approx(math.log1p(value), rel=1e-7)


def test_piecewise_quadratic_matches_piecewise_linear() -> None:
    """Test piecewise quadratic matches piecewise linear."""
    ends = [1.0, 2.5, 4.0]
    starts_b, ends_b = [0.5, 1.0, 0.8], [1.0, 0.6, 1.5]
    starts_d, ends_d = [0.4, 0.9, 0.3], [0.7, 0.2, 1.1]

    def quad_pieces(v0, v1):
        lengths = np.diff([0.0, *ends])
        return PiecewiseHazard(
            [
                QuadraticHazard(a, (b - a) / ln, 0.0)
                for a, b, ln in zip(v0, v1, lengths, strict=True)
            ],
            ends,
        )

    hb, hd = quad_pieces(starts_b, ends_b), quad_pieces(starts_d, ends_d)
    rb, rd = (
        PiecewiseLinear(starts_b, ends_b, ends),
        PiecewiseLinear(starts_d, ends_d, ends),
    )
    for t in (0.7, 2.0, 4.0):
        assert log_survival_probability(hb, hd, t, 4.0) == pytest.approx(
            log_survival_probability(rb, rd, t), rel=1e-10
        )
        assert _log_mean(hb, hd, t) == pytest.approx(
            _log_mean(as_rate(rb), as_rate(rd), t), rel=1e-10
        )


def test_piecewise_gompertz_matches_piecewise_constant() -> None:
    """Test piecewise gompertz matches piecewise constant."""
    ends = [1.0, 2.5, 4.0]
    vb, vd = [0.5, 1.2, 0.8], [0.4, 0.9, 1.3]
    hb = PiecewiseHazard([GompertzHazard(v, 0.0) for v in vb], ends)
    hd = PiecewiseHazard([GompertzHazard(v, 0.0) for v in vd], ends)
    rb, rd = PiecewiseConstant(vb, ends), PiecewiseConstant(vd, ends)
    for t in (0.7, 2.0, 4.0):
        assert log_survival_probability(hb, hd, t, 4.0) == pytest.approx(
            log_survival_probability(rb, rd, t), rel=1e-10
        )
        assert _log_mean(hb, hd, t) == pytest.approx(
            _log_mean(as_rate(rb), as_rate(rd), t), rel=1e-10
        )


def test_resolve_rates_passes_hazard_through() -> None:
    """Test resolve rates passes hazard through."""
    birth = PowerHazard.weibull(1.5, 2.0)
    birth_r, death_r, horizon = resolve_rates(birth, 0.7, 3.0)
    assert birth_r is birth
    assert horizon == 3.0
    assert as_rate_like(birth, 3.0) is birth
    assert death_r.hazard(1.0) == pytest.approx(0.7)


def test_hazard_with_rate_like_death_takes_horizon_from_it() -> None:
    """Test hazard with rate like death takes horizon from it."""
    death = PiecewiseLinear([0.2, 0.3], [0.4, 0.1], [1.0, 2.0])
    _, _, horizon = resolve_rates(GompertzHazard(0.5, 0.1), death, None)
    assert horizon == 2.0


def test_horizon_mismatch_with_piecewise_hazard_raises() -> None:
    """Test horizon mismatch with piecewise hazard raises."""
    pw = PiecewiseHazard([GompertzHazard(0.5, 0.0)], [2.0])
    with pytest.raises(ValueError, match="same end time"):
        resolve_rates(pw, PiecewiseConstant([1.0], [3.0]), None)
    with pytest.raises(ValueError, match="terminal_time must equal"):
        resolve_rates(pw, 1.0, 3.0)
    assert resolve_rates(pw, 1.0, None)[2] == 2.0


def test_negative_quadratic_raises() -> None:
    """Test negative quadratic raises."""
    bad = QuadraticHazard(0.1, -1.0, 0.0)  # negative past t = 0.1
    with pytest.raises(ValueError, match="negative"):
        resolve_rates(bad, 1.0, 3.0)


def test_missing_terminal_time_raises() -> None:
    """Test missing terminal time raises."""
    with pytest.raises(ValueError, match="neither rate fixes an end time"):
        resolve_rates(GompertzHazard(0.5, 0.1), 1.0, None)


def test_float_rates_still_need_terminal_time() -> None:
    """Test float rates still need terminal time."""
    with pytest.raises(ValueError, match="terminal_time is required"):
        resolve_rates(1.0, 1.0, None)


def test_contour_given_survival_with_hazards() -> None:
    """Test contour given survival with hazards."""
    birth, death = PowerHazard.weibull(0.5, 2.0), GompertzHazard(0.4, 0.2)
    a = bdsurv.simulate_contour_given_survival(birth, death, 3.0, random_seed=11)
    b = bdsurv.simulate_contour_given_survival(birth, death, 3.0, random_seed=11)
    assert a.birth_rate is birth
    assert a.death_rate is death
    assert a.terminal_time == 3.0
    assert a.records.equals(b.records)
    assert a.records["is_spine"].sum() == 1
    assert a.records.loc[a.records["is_spine"], "survived"].all()


def test_contour_given_population_with_hazards() -> None:
    """Test contour given population with hazards."""
    birth, death = LomaxHazard(2.0, 3.0), GompertzHazard(0.4, 0.2)
    a = bdsurv.simulate_contour_given_population(birth, death, 3, 2.0, random_seed=5)
    b = bdsurv.simulate_contour_given_population(birth, death, 3, 2.0, random_seed=5)
    assert a.records.equals(b.records)
    assert a.records["survived"].sum() == 3


def test_contour_given_ages_with_hazards() -> None:
    """Test contour given ages with hazards."""
    birth, death = LogLogisticHazard(1.5, 2.0), QuadraticHazard(0.2, 0.0, 0.05)
    ages = [0.5, 1.0, 1.5]
    kw = {"terminal_time": 3.0, "random_seed": 7}
    a = bdsurv.simulate_contour_given_ages(birth, death, ages, **kw)
    b = bdsurv.simulate_contour_given_ages(birth, death, ages, **kw)
    assert a.records.equals(b.records)
    survivors = a.records[a.records["survived"]]
    assert sorted(survivors["age"].round(9)) == pytest.approx(sorted(ages))


def test_hazard_rates_equal_to_constant_rates_agree_in_distribution() -> None:
    """Test hazard rates equal to constant rates agree in distribution."""
    lam, mu, t = 1.0, 0.6, 2.0
    n_seeds = 300

    def sizes(birth, death) -> np.ndarray:
        return np.array(
            [
                bdsurv.simulate_contour_given_survival(birth, death, t, random_seed=s)
                .records["survived"]
                .sum()
                for s in range(n_seeds)
            ],
            dtype=float,
        )

    h = sizes(GompertzHazard(lam, 0.0), GompertzHazard(mu, 0.0))
    f = sizes(lam, mu)
    se = math.sqrt(h.var(ddof=1) / n_seeds + f.var(ddof=1) / n_seeds)
    assert abs(h.mean() - f.mean()) < 5 * se
