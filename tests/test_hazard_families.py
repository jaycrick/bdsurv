"""Test the closed-form hazard families against known values, quadrature and Brent."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import integrate, stats

from bdsurv.hazards import (
    GompertzHazard,
    Hazard,
    LogLogisticHazard,
    LomaxHazard,
    PowerHazard,
)
from bdsurv.rates import (
    PiecewiseConstant,
    PiecewiseLinear,
    Rate,
    as_rate,
)

EPS = float(np.finfo(float).eps)
T_END = 10.0
N = 20_000

HAZARDS: dict[str, Hazard] = {
    "weibull0.5": PowerHazard.weibull(0.5, 2.0),
    "weibull1": PowerHazard.weibull(1.0, 1.5),
    "weibull3": PowerHazard.weibull(3.0, 4.0),
    "rayleigh": PowerHazard.rayleigh(0.3),
    "power-0.5": PowerHazard(1.2, -0.5),
    "gompertz0.7": GompertzHazard(0.2, 0.7),
    "gompertz-0.4": GompertzHazard(0.8, -0.4),
    "gompertz0": GompertzHazard(0.8, 0.0),
    "loglogistic0.6": LogLogisticHazard(0.6, 3.0),
    "loglogistic2": LogLogisticHazard(2.0, 3.0),
    "lomax": LomaxHazard(2.5, 4.0),
}
RATES: dict[str, Rate] = {
    "rate-constant": as_rate(
        PiecewiseConstant(
            values=np.array([0.5, 1.5, 0.8]),
            epoch_end_times=np.array([2.0, 5.0, T_END]),
        )
    ),
    "rate-linear": as_rate(
        PiecewiseLinear(
            start_values=np.array([0.5, 1.0]),
            end_values=np.array([1.0, 2.0]),
            epoch_end_times=np.array([5.0, T_END]),
        )
    ),
}
ALL = {**HAZARDS, **RATES}
T0S = [0.0, 0.1, 1.0, 5.0]
E_GRID = [1e-12, 1e-6, 0.01, 0.3, 1.0, 3.0, 10.0, 30.0]


def _cap(h: Hazard | Rate) -> float:
    return h.end_time if isinstance(h, Rate) else math.inf


def _hz(h: Hazard | Rate, t: float) -> float:
    return float(h.hazard(t))


def _cum(h: Hazard | Rate, t: float) -> float:
    return float(h.cumulative_hazard(t))


def _slack(h: Hazard | Rate, t1: float, e: float) -> float:
    """Return the rounding allowance for ``H(t1) - H(t0) = e`` at absolute times."""
    return 1e-10 * e + 32 * EPS * (abs(_cum(h, t1)) + t1 * _hz(h, t1))


@pytest.mark.parametrize("name", ALL)
def test_t2_cumulative_matches_integrated_hazard(name: str) -> None:
    """Test t2 cumulative matches integrated hazard."""
    h = ALL[name]
    singular = getattr(h, "singular_points", ())
    lo = 1e-3 if singular else 0.0
    pts = [b for b in getattr(h, "breakpoints", ())] or None
    if isinstance(h, Rate):
        pts = [float(x) for x in h.ends[:-1]]
    for t in (0.5, 2.0, 7.0, 9.5):
        val, _ = integrate.quad(
            lambda s, h=h: _hz(h, s),
            lo,
            t,
            points=pts,
            epsabs=0.0,
            epsrel=1e-12,
            limit=200,
        )
        assert _cum(h, t) - _cum(h, lo) == pytest.approx(val, rel=1e-9)
    assert _cum(h, 0.0) == 0.0


@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("t0", T0S)
def test_t5_forward_identity(name: str, t0: float) -> None:
    """Test t5 forward identity."""
    h = ALL[name]
    cap = _cap(h)
    for e in E_GRID:
        t1 = h.forward_time(t0, e, cap)
        assert t1 >= t0
        if t1 >= cap:
            assert t1 == cap
            continue
        assert _cum(h, t1) - _cum(h, t0) == pytest.approx(e, abs=_slack(h, t1, e))


@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("x", [0.1, 1.0, 5.0])
def test_backward_identity_and_exhaustion(name: str, x: float) -> None:
    """Test backward identity and exhaustion."""
    h = ALL[name]
    for e in E_GRID:
        s = h.backward_time(x, e)
        assert 0.0 <= s <= x
        if s > 0.0:
            assert _cum(h, x) - _cum(h, s) == pytest.approx(
                e, abs=1e-10 * e + 32 * EPS * (_cum(h, x) + x * _hz(h, x))
            )
    assert h.backward_time(x, _cum(h, x) * 1.5 + 1.0) == 0.0


@pytest.mark.parametrize("name", HAZARDS)
def test_backward_offsets_array(name: str) -> None:
    """Test backward offsets array."""
    h = HAZARDS[name]
    x = np.array([0.0, 0.1, 1.0, 5.0])[:, None]
    e = np.array(E_GRID)[None, :]
    d = h.backward_offsets(x, e)
    assert d.shape == (4, len(E_GRID))
    assert np.all((d >= 0.0) & (d <= x))
    hx = h.cumulative_hazard(x)
    exhausted = e >= hx
    assert np.all(
        d[np.broadcast_to(exhausted, d.shape)] == np.broadcast_to(x, d.shape)[exhausted]
    )
    live = ~exhausted & (d > 0)
    lhs = (h.cumulative_between(x - d, x))[live]
    rhs = np.broadcast_to(e, d.shape)[live]
    assert np.allclose(lhs, rhs, rtol=1e-8, atol=1e-12)


def test_t1_known_values() -> None:
    """Test t1 known values."""
    cases: list[tuple[Hazard, float, float, float]] = [
        (PowerHazard.weibull(2.0, 2.0), 3.0, 1.5, 2.25),
        (PowerHazard.rayleigh(0.5), 3.0, 1.5, 2.25),
        (PowerHazard(2.0, -0.5), 4.0, 1.0, 8.0),
        (PowerHazard(2.0, 0.0), 4.0, 2.0, 8.0),
        (GompertzHazard(2.0, 0.5), 2.0, 2.0 * math.e, 4.0 * (math.e - 1.0)),
        (GompertzHazard(2.0, 0.0), 2.0, 2.0, 4.0),
        (GompertzHazard(2.0, -0.5), 2.0, 2.0 / math.e, 4.0 * (1.0 - 1.0 / math.e)),
        (LogLogisticHazard(2.0, 1.0), 3.0, 0.6, math.log(10.0)),
        (LomaxHazard(3.0, 2.0), 2.0, 0.75, 3.0 * math.log(2.0)),
    ]
    for h, t, hv, hc in cases:
        assert float(h.hazard(t)) == pytest.approx(hv, rel=1e-12)
        assert float(h.cumulative_hazard(t)) == pytest.approx(hc, rel=1e-12)
    assert float(PowerHazard.weibull(2.0, 2.0).hazard(0.0)) == 0.0
    assert float(PowerHazard(2.0, 0.0).hazard(0.0)) == 2.0
    assert float(PowerHazard(2.0, -0.5).hazard(0.0)) == math.inf
    assert float(LogLogisticHazard(0.5, 1.0).hazard(0.0)) == math.inf
    assert float(LogLogisticHazard(1.0, 2.0).hazard(0.0)) == pytest.approx(0.5)
    assert float(LogLogisticHazard(2.0, 2.0).hazard(0.0)) == 0.0


def test_t3_closed_form_at_zero() -> None:
    """Test t3 closed form at zero."""
    e = np.array([0.05, 1.0, 4.0])
    assert PowerHazard.weibull(2.0, 3.0).forward_offsets(0.0, e) == pytest.approx(
        3.0 * np.sqrt(e), rel=1e-12
    )
    assert PowerHazard.weibull(0.5, 3.0).forward_offsets(0.0, e) == pytest.approx(
        3.0 * e**2, rel=1e-12
    )
    assert GompertzHazard(2.0, 0.5).forward_offsets(0.0, e) == pytest.approx(
        np.log1p(0.25 * e) / 0.5, rel=1e-12
    )
    assert LogLogisticHazard(2.0, 3.0).forward_offsets(0.0, e) == pytest.approx(
        3.0 * np.sqrt(np.expm1(e)), rel=1e-12
    )
    assert LomaxHazard(2.0, 3.0).forward_offsets(0.0, e) == pytest.approx(
        3.0 * np.expm1(e / 2.0), rel=1e-12
    )


@pytest.mark.parametrize("name", HAZARDS)
@pytest.mark.parametrize("t0", T0S)
def test_t5_t6_distribution(name: str, t0: float) -> None:
    """Test t5 t6 distribution."""
    h = HAZARDS[name]
    seed = sorted(HAZARDS).index(name) * 7 + int(t0 * 10)
    rng = np.random.default_rng(seed)
    r = h.forward_offsets(np.full(N, t0), rng.exponential(size=N))
    assert r.dtype == np.float64
    finite = np.isfinite(r)
    rem = h.cumulative_limit - float(h.cumulative_hazard(t0))
    p_finite = -math.expm1(-rem) if math.isfinite(rem) else 1.0
    # T5: transformed draws are Exp(1) (truncated if defective).
    u = h.cumulative_between(t0, t0 + r[finite])
    ks = stats.kstest(u, lambda x: stats.expon.cdf(x) / p_finite)
    assert ks.pvalue > 1e-4
    # T6: empirical conditional survival at five offsets.
    for rr in (0.1, 0.5, 1.0, 2.0, 4.0):
        surv = math.exp(-float(h.cumulative_between(t0, t0 + rr)))
        phat = float(np.mean(r > rr))
        assert abs(phat - surv) <= 5 * math.sqrt(surv * (1 - surv) / N) + 1e-12


@pytest.mark.parametrize("name", HAZARDS)
def test_t7_direct_matches_brent(name: str) -> None:
    """Test t7 direct matches brent."""
    h = HAZARDS[name]
    e = np.array([1e-8, 1e-3, 0.1, 1.0, 5.0, 20.0])
    for t0 in T0S:
        fast = h.forward_offsets(t0, e)
        slow = h.forward_offsets_numerical(t0, e)
        inf = np.isinf(slow)
        assert np.array_equal(np.isinf(fast), inf)
        assert np.all(np.abs(fast[~inf] - slow[~inf]) <= 1e-9 * (1.0 + slow[~inf]))
    for x in (0.1, 1.0, 5.0, 30.0):
        fast = h.backward_offsets(x, e)
        slow = h.backward_offsets_numerical(x, e)
        assert np.all(np.abs(fast - slow) <= 1e-9 * (1.0 + slow))


@pytest.mark.parametrize("b", [0.0, 1e-14, -1e-14, 50.0, -50.0])
def test_t8_gompertz_boundaries(b: float) -> None:
    """Test t8 gompertz boundaries."""
    h = GompertzHazard(1.3, b)
    e = np.array([1e-9, 1e-3, 0.01, 0.5, 5.0])
    for t0 in (0.0, 1.0, 3.0):
        fast = h.forward_offsets(t0, e)
        slow = h.forward_offsets_numerical(t0, e)
        assert np.array_equal(np.isinf(fast), np.isinf(slow))
        fin = np.isfinite(slow)
        assert np.all(np.abs(fast[fin] - slow[fin]) <= 1e-9 * (1.0 + slow[fin]))
        assert not np.any(np.isnan(fast))
    for x in (0.0, 1.0, 3.0):
        fast = h.backward_offsets(x, e)
        slow = h.backward_offsets_numerical(x, e)
        assert not np.any(np.isnan(fast))
        assert np.all(np.abs(fast - slow) <= 1e-9 * (1.0 + slow))
    if abs(b) < 1e-10:
        ref = GompertzHazard(1.3, 0.0)
        assert h.forward_offsets(0.0, e) == pytest.approx(
            ref.forward_offsets(0.0, e), rel=1e-9
        )
        assert h.backward_offsets(3.0, e) == pytest.approx(
            ref.backward_offsets(3.0, e), rel=1e-9
        )
        assert float(h.cumulative_hazard(2.0)) == pytest.approx(2.6, rel=1e-12)


@pytest.mark.parametrize("k", [0.2, 1.0, 5.0])
def test_t8_weibull_large_t0(k: float) -> None:
    """Test t8 weibull large t0."""
    h = PowerHazard.weibull(k, 2e6)
    e = np.array([1e-9, 1e-4, 0.1, 1.0, 10.0])
    fast = h.forward_offsets(1e6, e)
    slow = h.forward_offsets_numerical(1e6, e)
    assert np.all(np.abs(fast - slow) <= 1e-9 * (1.0 + slow))
    assert h.cumulative_between(1e6, 1e6 + fast) == pytest.approx(e, rel=1e-9)


@pytest.mark.parametrize("t0", [1e3, 1e8, 1e12])
def test_t8_lomax_large_t0(t0: float) -> None:
    """Test t8 lomax large t0."""
    h = LomaxHazard(1.7, 2.0)
    e = np.array([1e-9, 1e-3, 1.0, 10.0])
    fast = h.forward_offsets(t0, e)
    slow = h.forward_offsets_numerical(t0, e)
    # Brent works with absolute times, so it resolves R only to the ulp of t0.
    assert np.all(np.abs(fast - slow) <= 1e-9 * (1.0 + slow) + 8 * EPS * t0)
    assert h.cumulative_between(t0, t0 + fast) == pytest.approx(e, rel=1e-9)


@pytest.mark.parametrize("t0", [0.0, 1e-10, 1e-3, 1.0])
def test_t8_loglogistic_shape_below_one(t0: float) -> None:
    """Test t8 loglogistic shape below one."""
    h = LogLogisticHazard(0.4, 2.0)
    e = np.array([1e-9, 1e-3, 0.5, 3.0])
    fast = h.forward_offsets(t0, e)
    slow = h.forward_offsets_numerical(t0, e)
    assert np.all(np.abs(fast - slow) <= 1e-9 * (1.0 + slow))
    assert h.singular_points == (0.0,)
    assert not np.any(np.isnan(h.hazard(np.array([0.0, 1e-300, 1.0]))))


@pytest.mark.parametrize(
    "make",
    [
        lambda: PowerHazard(0.0, 1.0),
        lambda: PowerHazard(math.inf, 1.0),
        lambda: PowerHazard(1.0, -1.0),
        lambda: PowerHazard(1.0, math.nan),
        lambda: PowerHazard(1.0, math.inf),
        lambda: PowerHazard.weibull(0.0, 1.0),
        lambda: PowerHazard.weibull(1.0, -1.0),
        lambda: PowerHazard.rayleigh(-1.0),
        lambda: GompertzHazard(0.0, 1.0),
        lambda: GompertzHazard(1.0, math.inf),
        lambda: GompertzHazard(1.0, math.nan),
        lambda: LogLogisticHazard(0.0, 1.0),
        lambda: LogLogisticHazard(1.0, math.inf),
        lambda: LomaxHazard(-1.0, 1.0),
        lambda: LomaxHazard(1.0, 0.0),
    ],
)
def test_t8_invalid_parameters(make) -> None:
    """Test t8 invalid parameters."""
    with pytest.raises(ValueError, match="must be"):
        make()


def test_parameters_read_only_and_repr() -> None:
    """Test parameters read only and repr."""
    h = PowerHazard.weibull(2.0, 3.0)
    with pytest.raises(AttributeError):
        h.c = 1.0  # type: ignore[misc]
    assert "PowerHazard" in repr(h)
    assert "b=-0.4" in repr(GompertzHazard(1.0, -0.4))
    assert "shape=2.0" in repr(LogLogisticHazard(2.0, 1.0))
    assert "scale=2.0" in repr(LomaxHazard(1.0, 2.0))
    w = PowerHazard.weibull(1.5, 2.5)
    assert (w.k, w.eta) == (pytest.approx(1.5), pytest.approx(2.5))
    assert PowerHazard(1.0, -0.5).singular_points == (0.0,)
    assert PowerHazard(1.0, 0.5).singular_points == ()


def test_scalar_and_array_shapes() -> None:
    """Test scalar and array shapes."""
    h = LomaxHazard(2.0, 1.0)
    assert h.forward_offsets(1.0, 0.5).shape == ()
    assert h.forward_offsets([0.0, 1.0], 0.5).shape == (2,)
    assert h.backward_offsets(np.ones((2, 3)), 0.5).shape == (2, 3)


def test_t10_defective_gompertz() -> None:
    """Test t10 defective gompertz."""
    h = GompertzHazard(1.0, -0.5)
    assert h.cumulative_limit == pytest.approx(2.0)
    t0 = 1.0
    rng = np.random.default_rng(2024)
    r = h.forward_offsets(np.full(N, t0), rng.exponential(size=N))
    rem = h.cumulative_limit - float(h.cumulative_hazard(t0))
    p = math.exp(-rem)
    frac = float(np.mean(np.isinf(r)))
    assert abs(frac - p) <= 5 * math.sqrt(p * (1 - p) / N)
    assert h.forward_time(t0, 5.0, cap=50.0) == 50.0
    assert h.forward_time(t0, 0.1, cap=50.0) < 50.0
    assert h.forward_time(t0, 5.0) == math.inf
    assert h.backward_time(2.0, 100.0) == 0.0
