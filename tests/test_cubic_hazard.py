"""Test CubicHazard, PiecewiseHazard.from_polynomials and cubic-rate simulation."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate

import bdsurv
from bdsurv.hazards import CubicHazard, PiecewiseHazard, QuadraticHazard
from bdsurv.rates import PiecewiseLinear
from bdsurv.survival import log_survival_probability

PARAMS = [
    (1.0, 2.0, 3.0, 4.0),
    (0.5, -1.0, 1.0, 0.3),
    (0.0, 0.0, 0.0, 1.0),
    (2.0, 1.0, -0.5, 0.05),
]


@pytest.mark.parametrize("params", PARAMS)
def test_cumulative_hazard_matches_quadrature(params) -> None:
    """Test cumulative hazard and cumulative_between against quad."""
    h = CubicHazard(*params)
    for s, e in [(0.0, 1.5), (0.3, 2.0), (1.0, 1.0)]:
        want, _ = integrate.quad(lambda t: float(h.hazard(t)), s, e, epsabs=0.0)
        assert float(h.cumulative_between(s, e)) == pytest.approx(want, rel=1e-10)
        diff = float(h.cumulative_hazard(e) - h.cumulative_hazard(s))
        assert diff == pytest.approx(want, rel=1e-10)


def test_inversion_round_trip() -> None:
    """Test H(t0 + R) - H(t0) = E for forward and backward offsets."""
    h = CubicHazard(0.4, 0.3, 0.1, 0.02)
    t0 = np.array([0.0, 0.5, 2.0])
    e = np.array([0.1, 1.0, 3.0])
    r = h.forward_offsets(t0, e)
    got = h.cumulative_hazard(t0 + r) - h.cumulative_hazard(t0)
    assert got == pytest.approx(e, rel=1e-9)
    x = np.array([1.0, 2.0, 3.0])
    d = h.backward_offsets(x, np.array([0.2, 0.5, 1.0]))
    back = h.cumulative_hazard(x) - h.cumulative_hazard(x - d)
    assert back == pytest.approx([0.2, 0.5, 1.0], rel=1e-9)


def test_validation_and_repr() -> None:
    """Test parameter validation and repr."""
    with pytest.raises(ValueError, match="non-negative"):
        CubicHazard(-0.1, 0.0, 0.0, 0.0)
    for bad in (np.nan, np.inf):
        with pytest.raises(ValueError, match="finite"):
            CubicHazard(1.0, 0.0, 0.0, bad)
    assert repr(CubicHazard(1.0, 2.0, 3.0, 4.0)) == (
        "CubicHazard(a=1.0, b=2.0, c=3.0, d=4.0)"
    )
    assert not CubicHazard.has_direct_sampler


def test_check_nonnegative() -> None:
    """Test interior dips are caught and positive minima pass."""
    # h = 1 - 3t + 2t^2 ... cubic with negative local minimum near t = 1.5.
    dip = CubicHazard(1.0, -4.0, 2.5, 0.0)
    cubic_dip = CubicHazard(1.0, -3.0, 1.5, 0.2)
    for hz in (dip, cubic_dip):
        with pytest.raises(ValueError, match="negative"):
            hz.check_nonnegative(5.0)
    CubicHazard(1.0, -1.0, 0.5, 0.1).check_nonnegative(5.0)
    CubicHazard(1.0, -1.0, 0.0, 0.0).check_nonnegative(1.0)
    with pytest.raises(ValueError, match="negative"):
        CubicHazard(1.0, 0.0, 0.0, -1.0).check_nonnegative(2.0)


def test_nonnegative_on_half_line() -> None:
    """Test exact half-line non-negativity."""
    assert CubicHazard(1.0, 0.0, 0.0, 0.0).nonnegative_on_half_line
    assert not CubicHazard(1.0, -1.0, 0.0, 0.0).nonnegative_on_half_line
    assert not CubicHazard(1.0, 1.0, -0.1, 0.0).nonnegative_on_half_line
    assert not CubicHazard(1.0, 1.0, 1.0, -0.01).nonnegative_on_half_line
    assert CubicHazard(1.0, 1.0, 1.0, 0.01).nonnegative_on_half_line
    assert CubicHazard(1.0, -1.0, 1.0, 0.5).nonnegative_on_half_line
    assert not CubicHazard(0.1, -1.0, 1.0, 0.5).nonnegative_on_half_line
    assert CubicHazard(0.0, 0.0, 0.0, 1.0).nonnegative_on_half_line
    assert not CubicHazard(0.9, 0.0, -3.0, 2.0).nonnegative_on_half_line
    assert CubicHazard(2.0, 0.0, -3.0, 2.0).nonnegative_on_half_line
    with pytest.raises(ValueError, match="not non-negative"):
        CubicHazard(1.0, 0.0, 0.0, -1.0).forward_offsets(0.0, 1.0)
    with pytest.raises(ValueError, match="negative"):
        CubicHazard(1.0, -4.0, 2.5, 0.2).backward_offsets(5.0, 0.1)


def test_from_polynomials_matches_piecewise_linear() -> None:
    """Test from_polynomials reproduces a PiecewiseLinear rate."""
    ends = [1.0, 2.5, 4.0]
    v0, v1 = [0.5, 1.0, 0.8], [1.0, 0.6, 1.5]
    lengths = np.diff([0.0, *ends])
    rows = [[a, (b - a) / ln] for a, b, ln in zip(v0, v1, lengths, strict=True)]
    ph = PiecewiseHazard.from_polynomials(rows, ends)
    assert all(isinstance(p, QuadraticHazard) for p in ph.pieces)
    lin = PiecewiseLinear(v0, v1, ends)
    ts = np.linspace(0.0, 4.0, 41)
    starts = np.array([0.0, *ends[:-1]])
    idx = np.minimum(np.searchsorted(ends, ts, side="left"), 2)
    u = ts - starts[idx]
    want_h = np.array(v0)[idx] + (np.array(v1) - np.array(v0))[idx] * u / lengths[idx]
    assert ph.hazard(ts) == pytest.approx(want_h, abs=1e-12)
    cum = np.concatenate(
        ([0.0], np.cumsum((np.array(v0) + np.array(v1)) / 2 * lengths))
    )
    want_c = (
        cum[idx]
        + np.array(v0)[idx] * u
        + ((np.array(v1) - np.array(v0))[idx] * u**2 / (2 * lengths[idx]))
    )
    assert ph.cumulative_hazard(ts) == pytest.approx(want_c, abs=1e-12)
    assert log_survival_probability(ph, ph, 3.0, 4.0) == pytest.approx(
        log_survival_probability(lin, lin, 3.0), abs=1e-10
    )


def test_from_polynomials_piece_classes_and_errors() -> None:
    """Test the chosen piece classes and the shape validation."""
    ph = PiecewiseHazard.from_polynomials(
        [[1.0], [1.0, 0.5, 0.1], [1.0, 0, 0, 0.1]], [1, 2, 3]
    )
    assert [type(p) for p in ph.pieces] == [
        QuadraticHazard,
        QuadraticHazard,
        CubicHazard,
    ]
    assert ph.pieces[2] == CubicHazard(1.0, 0.0, 0.0, 0.1)
    with pytest.raises(ValueError, match="one row per epoch"):
        PiecewiseHazard.from_polynomials([[1.0]], [1.0, 2.0])
    with pytest.raises(ValueError, match="between 1 and 4"):
        PiecewiseHazard.from_polynomials([[1.0, 0, 0, 0, 1.0]], [1.0])
    with pytest.raises(ValueError, match="between 1 and 4"):
        PiecewiseHazard.from_polynomials([[]], [1.0])


def test_simulate_with_piecewise_cubic_birth_rate() -> None:
    """Test the forward simulator accepts a piecewise-cubic birth rate."""
    birth = PiecewiseHazard.from_polynomials(
        [[0.8, 0.2, -0.1, 0.02], [0.9, 0.0, 0.1, -0.02]], [1.5, 3.0]
    )
    a = bdsurv.simulate_birth_death(birth, 0.5, random_seed=3)
    b = bdsurv.simulate_birth_death(birth, 0.5, random_seed=3)
    assert a.records.equals(b.records)
    assert len(a.records) >= 1
    assert (a.records["creation_time"] <= 3.0).all()
