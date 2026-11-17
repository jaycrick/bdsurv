"""Test QuadraticHazard, PiecewiseHazard and sample_residual_lifetimes."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import integrate, stats

from bdsurv._hazard import backward_hazard_time, forward_hazard_time
from bdsurv._seeding import count_draws
from bdsurv.hazards import (
    GompertzHazard,
    Hazard,
    LomaxHazard,
    PiecewiseHazard,
    PowerHazard,
    QuadraticHazard,
    sample_residual_lifetimes,
)
from bdsurv.rates import PiecewiseConstant, PiecewiseLinear, as_rate

N = 20_000
ENDS = [2.0, 5.0, 9.0]
E_GRID = [0.0, 1e-9, 0.01, 0.3, 1.0, 3.0, 8.0, 30.0]


# ---------------------------------------------------------------- Quadratic


def test_quadratic_known_values() -> None:
    """Test h and H at fixed parameters (T1)."""
    q = QuadraticHazard(1.0, 2.0, 3.0)
    assert float(q.hazard(2.0)) == pytest.approx(1 + 4 + 12, rel=1e-12)
    assert float(q.cumulative_hazard(2.0)) == pytest.approx(2 + 4 + 8, rel=1e-12)
    assert float(q.cumulative_between(1.0, 2.0)) == pytest.approx(
        14.0 - (1 + 1 + 1), rel=1e-12
    )
    assert not q.has_direct_sampler


@pytest.mark.parametrize("params", [(1.0, 0.5, 0.2), (0.3, -0.4, 0.5), (0.0, 0.0, 0.1)])
def test_quadratic_h_integrates(params: tuple[float, float, float]) -> None:
    """Test H against quadrature (T2)."""
    q = QuadraticHazard(*params)
    for t in (0.5, 2.0, 7.0):
        val, _ = integrate.quad(lambda s, q=q: float(q.hazard(s)), 0.0, t)
        assert float(q.cumulative_hazard(t)) == pytest.approx(val, rel=1e-9)


@pytest.mark.parametrize("t0", [0.0, 0.1, 1.0, 5.0])
def test_quadratic_forward_backward_identity(t0: float) -> None:
    """Test the forward identity (T5) and backward inversion."""
    q = QuadraticHazard(0.4, 0.3, 0.05)
    r = q.forward_offsets(np.full(len(E_GRID), t0), E_GRID)
    got = q.cumulative_between(t0, t0 + r)
    np.testing.assert_allclose(got, E_GRID, rtol=1e-10, atol=1e-12)
    x = t0 + 3.0
    d = q.backward_offsets(np.full(len(E_GRID), x), E_GRID)
    assert np.all((d >= 0.0) & (d <= x))
    inside = d < x
    np.testing.assert_allclose(
        q.cumulative_between(x - d, x)[inside],
        np.asarray(E_GRID)[inside],
        rtol=1e-10,
        atol=1e-12,
    )
    assert q.backward_time(x, 1e9) == 0.0


def test_quadratic_validation() -> None:
    """Test constructor and nonnegativity checks."""
    with pytest.raises(ValueError, match="a must be non-negative"):
        QuadraticHazard(-0.1, 0.0, 0.0)
    for bad in (math.nan, math.inf):
        with pytest.raises(ValueError, match="finite"):
            QuadraticHazard(1.0, bad, 0.0)
    QuadraticHazard(1.0, -1.0, 0.0).check_nonnegative(1.0)
    with pytest.raises(ValueError, match="negative on"):
        QuadraticHazard(1.0, -1.0, 0.0).check_nonnegative(2.0)
    # vertex dips below zero inside (0, T) though both ends are positive
    dip = QuadraticHazard(1.0, -4.0, 2.5)
    assert float(dip.hazard(0.0)) > 0
    assert float(dip.hazard(4.0)) > 0
    with pytest.raises(ValueError, match="negative on"):
        dip.check_nonnegative(4.0)
    # vertex outside the window is fine
    dip.check_nonnegative(0.1)
    assert QuadraticHazard(1.0, 0.0, 0.0).nonnegative_on_half_line
    assert QuadraticHazard(1.0, -1.0, 1.0).nonnegative_on_half_line
    assert not QuadraticHazard(1.0, -3.0, 1.0).nonnegative_on_half_line
    assert not QuadraticHazard(1.0, 1.0, -0.1).nonnegative_on_half_line


def test_quadratic_negative_curvature_window() -> None:
    """Test c < 0: valid on [0, T], capped forward works, uncapped raises."""
    q = QuadraticHazard(1.0, 0.5, -0.05)  # h(T=5) = 2.25 - ... > 0
    q.check_nonnegative(5.0)
    with pytest.raises(ValueError, match="negative on"):
        q.check_nonnegative(30.0)
    t = q.forward_time(1.0, 0.5, cap=5.0)
    assert 1.0 < t < 5.0
    assert float(q.cumulative_between(1.0, t)) == pytest.approx(0.5, rel=1e-10)
    assert q.forward_time(1.0, 1e3, cap=5.0) == 5.0
    with pytest.raises(ValueError, match="not non-negative"):
        q.forward_offsets(1.0, 0.5)
    with pytest.raises(ValueError, match="negative on"):
        q.backward_offsets(30.0, 0.5)


# ---------------------------------------------------------------- Piecewise


def _linear_pair() -> tuple[PiecewiseHazard, object]:
    starts = np.array([0.5, 1.0, 0.2])
    ends_v = np.array([1.0, 0.7, 2.0])
    ends = np.array(ENDS)
    lens = np.diff(ends, prepend=0.0)
    slopes = (ends_v - starts) / lens
    pieces = [QuadraticHazard(a, b, 0.0) for a, b in zip(starts, slopes, strict=True)]
    rate = as_rate(PiecewiseLinear(starts, ends_v, ends))
    return PiecewiseHazard(pieces, ends), rate


@pytest.mark.parametrize("t0", [0.0, 0.7, 2.0, 3.5, 5.0, 8.0])
def test_t9_matches_piecewise_linear_forward(t0: float) -> None:
    """Test forward walk equals ``forward_hazard_time`` (events in and across epochs)."""
    ph, rate = _linear_pair()
    cap = 9.0
    for e in E_GRID:
        want = forward_hazard_time(rate, t0, e, cap)  # pyright: ignore[reportArgumentType]
        assert ph.forward_time(t0, e, cap) == pytest.approx(want, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("x", [0.7, 2.0, 3.5, 5.0, 8.0, 9.0])
def test_t9_matches_piecewise_linear_backward(x: float) -> None:
    """Test backward walk equals ``backward_hazard_time``."""
    ph, rate = _linear_pair()
    for e in E_GRID:
        want = backward_hazard_time(rate, x, e)  # pyright: ignore[reportArgumentType]
        assert ph.backward_time(x, e) == pytest.approx(want, rel=1e-12, abs=1e-12)


def test_t9_matches_piecewise_constant() -> None:
    """Test Gompertz(v, 0) pieces against PiecewiseConstant."""
    vals = np.array([0.5, 1.5, 0.8])
    ph = PiecewiseHazard([GompertzHazard(v, 0.0) for v in vals], ENDS)
    assert ph.has_direct_sampler
    rate = as_rate(PiecewiseConstant(vals, np.array(ENDS)))
    for t0 in (0.0, 1.0, 2.0, 4.0, 7.5):
        for e in E_GRID:
            assert ph.forward_time(t0, e, 9.0) == pytest.approx(
                forward_hazard_time(rate, t0, e, 9.0), rel=1e-12, abs=1e-12
            )
            assert ph.backward_time(t0, e) == pytest.approx(
                backward_hazard_time(rate, t0, e), rel=1e-12, abs=1e-12
            )


def test_piecewise_offsets_arrays_and_no_event() -> None:
    """Test array offsets, ``inf`` past the end, and the cap."""
    ph, _ = _linear_pair()
    total = ph.cumulative_limit
    t0 = np.array([0.0, 1.0, 4.0, 8.0])
    r = ph.forward_offsets(t0, np.full(4, 0.2))
    np.testing.assert_allclose(ph.cumulative_between(t0, t0 + r), 0.2, rtol=1e-10)
    assert np.all(np.isinf(ph.forward_offsets(t0, np.full(4, total + 1.0))))
    assert ph.forward_time(1.0, total + 1.0) == math.inf
    assert ph.forward_time(1.0, total + 1.0, 7.0) == 7.0
    assert ph.forward_time(1.0, 100.0, 1.5) == 1.5
    assert ph.forward_time(10.0, 0.1, 12.0) == 12.0
    d = ph.backward_offsets(np.array([3.0, 9.0]), np.array([0.1, total + 5]))
    assert d[1] == 9.0
    assert d.shape == (2,)
    assert ph.backward_time(4.0, 0.0) == 4.0


def test_piecewise_weibull_singular_each_epoch() -> None:
    """Test identity H(t0+R)-H(t0)=e for pieces singular at each epoch start."""
    pieces = [PowerHazard.weibull(0.5, s) for s in (1.0, 2.0, 0.5)]
    ph = PiecewiseHazard(pieces, [1.0, 3.0, 6.0])
    assert ph.singular_points == (0.0, 1.0, 3.0)
    for t0 in (0.0, 0.3, 1.0, 2.5, 3.0, 4.0):
        e = np.array([1e-3, 0.1, 1.0, 2.0])
        r = ph.forward_offsets(np.full(4, t0), e)
        ok = np.isfinite(r)
        got = ph.cumulative_hazard(t0 + r[ok]) - ph.cumulative_hazard(t0)
        np.testing.assert_allclose(got, e[ok], rtol=1e-9)
    assert float(ph.hazard(0.0)) == math.inf or float(ph.hazard(0.0)) > 1e3


def test_piecewise_continuity_and_contents() -> None:
    """Test H is continuous at boundaries and the metadata."""
    ph, _ = _linear_pair()
    for tau in ENDS[:-1]:
        left = float(ph.cumulative_hazard(tau))
        right = float(ph.cumulative_hazard(tau + 1e-9))
        assert right - left == pytest.approx(float(ph.hazard(tau)) * 1e-9, abs=1e-8)
    assert float(ph.cumulative_hazard(0.0)) == 0.0
    assert float(ph.cumulative_hazard(100.0)) == ph.cumulative_limit
    assert ph.end_time == 9.0
    assert ph.breakpoints == (2.0, 5.0)
    assert ph.singular_points == ()
    kinked = PiecewiseHazard(
        [PiecewiseHazard([LomaxHazard(1.0, 1.0)], [1.0]), PowerHazard(1.0, -0.5)],
        [1.0, 4.0],
    )
    assert kinked.singular_points == (1.0,)
    assert kinked.breakpoints == (1.0,)
    assert kinked.hazard(np.array([0.5, 2.0])).shape == (2,)


def test_piecewise_validation() -> None:
    """Test constructor validation."""
    with pytest.raises(ValueError, match="same length"):
        PiecewiseHazard([LomaxHazard(1.0, 1.0)], [1.0, 2.0])
    with pytest.raises(ValueError, match="increasing"):
        PiecewiseHazard([LomaxHazard(1.0, 1.0)] * 2, [2.0, 1.0])
    with pytest.raises(ValueError, match="positive"):
        PiecewiseHazard([LomaxHazard(1.0, 1.0)], [0.0])
    with pytest.raises(TypeError, match="Hazard"):
        PiecewiseHazard([1.0], [1.0])  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="negative on"):
        PiecewiseHazard([QuadraticHazard(1.0, -1.0, 0.0)], [3.0])
    ph = PiecewiseHazard([QuadraticHazard(1.0, 0.0, 0.1)], [3.0])
    assert not ph.has_direct_sampler
    assert "PiecewiseHazard" in repr(ph)


# ---------------------------------------------------------------- Batch API

T0_ARRAY = np.tile(np.array([0.0, 0.5, 1.0, 2.5, 4.0]), N // 5)

FAMILIES: dict[str, Hazard] = {
    "weibull": PowerHazard.weibull(1.7, 3.0),
    "quadratic": QuadraticHazard(0.3, 0.1, 0.02),
    "piecewise": PiecewiseHazard(
        [QuadraticHazard(0.3, 0.2, 0.0), GompertzHazard(0.8, 0.1)], [3.0, 50.0]
    ),
}


def test_sample_reproducible() -> None:
    """Test same seed same output, different seed different output."""
    h = FAMILIES["weibull"]
    a = sample_residual_lifetimes(h, T0_ARRAY[:50], random_seed=3)
    b = sample_residual_lifetimes(h, T0_ARRAY[:50], random_seed=3)
    c = sample_residual_lifetimes(h, T0_ARRAY[:50], random_seed=4)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    assert a.shape == (50,)
    assert sample_residual_lifetimes(h, 1.0, random_seed=1).shape == ()


@pytest.mark.parametrize("name", ["weibull", "quadratic", "piecewise"])
def test_t5_t6_batch_distribution(name: str) -> None:
    """Test KS of H(t0+R)-H(t0) vs Exp(1) and conditional survival (T5, T6)."""
    h = FAMILIES[name]
    r = sample_residual_lifetimes(h, T0_ARRAY, random_seed=11)
    assert np.all(np.isfinite(r))
    z = h.cumulative_hazard(T0_ARRAY + r) - h.cumulative_hazard(T0_ARRAY)
    assert stats.kstest(z, "expon").pvalue > 1e-4
    for t0 in (0.0, 1.0, 4.0):
        rr = r[t0 == T0_ARRAY]
        n = rr.size
        for q in (0.2, 1.0, 2.0, 4.0, 8.0):
            s = math.exp(-float(h.cumulative_between(t0, t0 + q)))
            phat = float(np.mean(rr > q))
            assert abs(phat - s) <= 5 * math.sqrt(s * (1 - s) / n) + 1e-12


@pytest.mark.parametrize("name", ["weibull", "quadratic"])
def test_batch_end_time_truncation(name: str) -> None:
    """Test the fraction of ``inf`` against theory when end_time truncates."""
    h = FAMILIES[name]
    end = 5.0
    r = sample_residual_lifetimes(h, T0_ARRAY, random_seed=5, end_time=end)
    for t0 in (0.0, 2.5, 4.0):
        sel = r[t0 == T0_ARRAY]
        p = math.exp(-float(h.cumulative_between(t0, end)))
        frac = float(np.mean(np.isinf(sel)))
        assert abs(frac - p) <= 5 * math.sqrt(p * (1 - p) / sel.size)
        assert np.all(t0 + sel[np.isfinite(sel)] <= end)


def test_batch_quadratic_window_and_piecewise_end() -> None:
    """Test c < 0 works with end_time but not without; piecewise end gives inf."""
    q = QuadraticHazard(1.0, 0.5, -0.05)
    r = sample_residual_lifetimes(q, np.ones(100), random_seed=0, end_time=5.0)
    assert np.all(np.isfinite(r) | np.isinf(r))
    assert np.all(1.0 + r[np.isfinite(r)] < 5.0)
    with pytest.raises(ValueError, match="not non-negative"):
        sample_residual_lifetimes(q, np.ones(3), random_seed=0)
    with pytest.raises(ValueError, match="negative on"):
        sample_residual_lifetimes(q, np.ones(3), random_seed=0, end_time=30.0)
    ph = PiecewiseHazard([LomaxHazard(1.0, 1.0)], [0.5])
    rr = sample_residual_lifetimes(ph, np.full(2000, 0.1), random_seed=2)
    assert np.isinf(rr).mean() > 0.3
    assert np.all(0.1 + rr[np.isfinite(rr)] <= 0.5)


def test_batch_defective_gompertz() -> None:
    """Test the ``inf`` fraction of a defective Gompertz hazard."""
    h = GompertzHazard(1.0, -0.5)
    t0 = np.full(N, 1.0)
    r = sample_residual_lifetimes(h, t0, random_seed=9)
    p = math.exp(-(h.cumulative_limit - float(h.cumulative_hazard(1.0))))
    frac = float(np.mean(np.isinf(r)))
    assert abs(frac - p) <= 5 * math.sqrt(p * (1 - p) / N)


def test_batch_validation() -> None:
    """Test input validation."""
    h = LomaxHazard(1.0, 1.0)
    with pytest.raises(TypeError, match="Hazard"):
        sample_residual_lifetimes(1.0, [1.0])  # pyright: ignore[reportArgumentType]
    for bad in ([-1.0], [math.nan], [math.inf]):
        with pytest.raises(ValueError, match="start_times"):
            sample_residual_lifetimes(h, bad)
    with pytest.raises(ValueError, match="exceed end_time"):
        sample_residual_lifetimes(h, [6.0], end_time=5.0)
    with pytest.raises(ValueError, match="end_time"):
        sample_residual_lifetimes(h, [1.0], end_time=math.nan)
    ph = PiecewiseHazard([h], [2.0])
    with pytest.raises(ValueError, match="end_time of the hazard"):
        sample_residual_lifetimes(ph, [3.0])


def test_batch_counts_draws() -> None:
    """Test n exponentials are counted under role dynamics."""
    with count_draws() as counts:
        sample_residual_lifetimes(LomaxHazard(1.0, 1.0), np.ones(37), random_seed=1)
    assert counts.by_role("dynamics") == 37
    assert counts.total_draws == 37
