"""Test LogisticHazard (with floors) and its stable helpers."""

from __future__ import annotations

import math
import warnings
from decimal import Decimal, localcontext

import numpy as np
import pytest
from _helpers import assert_valid_records
from scipy import integrate
from test_hazard_families import HAZARDS

from bdsurv import simulate_birth_death, simulate_contour_given_survival
from bdsurv.hazards import (
    Hazard,
    LogisticHazard,
    _log_expm1,  # pyright: ignore[reportPrivateUsage]
    _softplus,  # pyright: ignore[reportPrivateUsage]
    _softplus_diff,  # pyright: ignore[reportPrivateUsage]
    sample_residual_lifetimes,
)

EPS = float(np.finfo(float).eps)
SIGMOID_NAMES = [
    "logistic",
    "logistic-steep",
    "logistic-fall",
    "logistic-fall-neg-c",
]
SIGMOID = {name: HAZARDS[name] for name in SIGMOID_NAMES}
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


# Reference values with ``decimal`` (60 digits), free of any cancellation.
def _sp(x: Decimal) -> Decimal:
    return (1 + x.exp()).ln()


def _ref_softplus(x: float) -> Decimal:
    with localcontext() as ctx:
        ctx.prec = 60
        return _sp(Decimal(x))


def _ref_softplus_diff(x0: float, dx: float) -> Decimal:
    with localcontext() as ctx:
        ctx.prec = 60
        a = Decimal(x0)
        return _sp(a + Decimal(dx)) - _sp(a)  # the sum is exact in Decimal


def _close(got: float, ref: Decimal, rel: float = 1e-9) -> bool:
    return math.isclose(got, float(ref), rel_tol=rel, abs_tol=1e-300)


def _quad(h: Hazard, lo: float, hi: float) -> float:
    val, _ = integrate.quad(
        lambda s: float(h.hazard(s)), lo, hi, epsabs=0.0, epsrel=1e-12, limit=400
    )
    return val


# 1. Known values -----------------------------------------------------------


def test_known_values() -> None:
    """Test hand-computed values of hazard and cumulative hazard."""
    cases: list[tuple[Hazard, float, float, float]] = [
        # L=2, k=1, c=0: h(ln 3) = 2 * 3/4, H(ln 3) = 2 (log 4 - log 2).
        (LogisticHazard(2.0, 1.0, 0.0), math.log(3.0), 1.5, 2.0 * math.log(2.0)),
        # h(0) = L/2 and H(0) = 0 when t = 0 whatever c is.
        (LogisticHazard(3.0, 2.0, 0.0), 0.0, 1.5, 0.0),
        # Floor 0.5 + logistic 1 sigma(2t) is 1 + .5 tanh t, H(t) = t + .5 log cosh t.
        (
            LogisticHazard.with_floor(0.5, 1.0, 2.0, 0.0),
            1.0,
            1.0 + 0.5 * math.tanh(1.0),
            1.0 + 0.5 * math.log(math.cosh(1.0)),
        ),
        # LogisticHazard(2, 2, 0) is 1 + tanh t, H(t) = t + log cosh t.
        (
            LogisticHazard(2.0, 2.0, 0.0),
            0.5,
            1.0 + math.tanh(0.5),
            0.5 + math.log(math.cosh(0.5)),
        ),
    ]
    for h, t, hv, hc in cases:
        assert float(h.hazard(t)) == pytest.approx(hv, rel=1e-12)
        assert float(h.cumulative_hazard(t)) == pytest.approx(hc, rel=1e-12)


# 2. Junior-formula regression ---------------------------------------------


@pytest.mark.parametrize("t0", [0.0, 2.0, 5.0, 9.0])
@pytest.mark.parametrize("e", [0.05, 0.7, 4.0])
def test_logistic_forward_is_not_the_junior_formula(t0: float, e: float) -> None:
    """Test the forward offset includes the ``- x`` term the junior formula lacks."""
    h = LogisticHazard(1.0, 1.0, 5.0)
    x = h.k * (t0 - h.c)
    u = h.k * e / h.L
    wrong = math.log(math.exp(x) + (1.0 + math.exp(x)) * math.expm1(u)) / h.k
    right = float(h.forward_offsets(t0, e))
    brent = float(h.forward_offsets_numerical(t0, e))
    assert right == pytest.approx(brent, rel=1e-9, abs=1e-12)
    assert float(h.cumulative_between(t0, t0 + right)) == pytest.approx(e, rel=1e-9)
    # x != 0 here (c = 5, and t0 = 5 is the only exception): the formulas differ
    # by exactly x / k.
    assert wrong - right == pytest.approx(x / h.k, rel=1e-9, abs=1e-12)
    if x != 0.0:
        assert abs(wrong - right) > 0.5 * abs(x) / h.k
        assert abs(float(h.cumulative_between(t0, t0 + wrong)) - e) > 1e-3 * e
    else:
        assert wrong == pytest.approx(right, rel=1e-12)


# 3. The tanh recipe ----------------------------------------------------------


@pytest.mark.parametrize("b", [0.4, -0.4])
@pytest.mark.parametrize("a", [0.4, 0.6, 1.5])
@pytest.mark.parametrize("k", [0.3, 1.0])
def test_tanh_recipe_matches_floor_logistic(a: float, b: float, k: float) -> None:
    """Test ``a + b tanh(k(t-c))`` is ``with_floor(a-|b|, 2|b|, 2 sign(b) k, c)``."""
    c = 3.0
    sign = math.copysign(1.0, b)
    h = LogisticHazard.with_floor(a - abs(b), 2.0 * abs(b), 2.0 * sign * k, c)

    def ref(s: float) -> float:
        return a + b * math.tanh(k * (s - c))

    t = np.array([0.0, 0.5, 2.0, 3.0, 6.0, 20.0])
    assert h.hazard(t) == pytest.approx([ref(x) for x in t], rel=1e-12, abs=1e-14)
    for lo, hi in [(0.0, 2.0), (1.0, 6.0), (3.0, 3.5), (5.0, 40.0)]:
        val, _ = integrate.quad(ref, lo, hi, epsabs=0.0, epsrel=1e-12, limit=400)
        assert float(h.cumulative_between(lo, hi)) == pytest.approx(val, rel=1e-9)
    if b > 0 and a == b:
        lg = LogisticHazard(2.0 * b, 2.0 * k, c)
        assert lg.hazard(t) == pytest.approx([ref(x) for x in t], rel=1e-12)
        assert float(lg.cumulative_between(1.0, 6.0)) == pytest.approx(
            float(h.cumulative_between(1.0, 6.0)), rel=1e-10
        )


# 4. Stability of the helpers ------------------------------------------------


def test_softplus_stable() -> None:
    """Test ``_softplus`` at extreme arguments."""
    xs = [-1e3, -50.0, -1e-300, 0.0, 1e-300, 50.0, 1e3]
    got = _softplus(np.array(xs))
    assert np.all(np.isfinite(got))
    for x, g in zip(xs, got, strict=True):
        assert _close(float(g), _ref_softplus(x))
    assert got[3] == pytest.approx(math.log(2.0), rel=1e-15)


def test_log_expm1_stable() -> None:
    """Test ``_log_expm1`` from tiny to huge ``u``."""
    us = [1e-300, 1e-10, 1e-3, 1.0, 29.9, 30.1, 1e3, 1e300]
    got = _log_expm1(np.array(us))
    assert np.all(np.isfinite(got))
    for u, g in zip(us, got, strict=True):
        with localcontext() as ctx:
            ctx.prec = 700  # expm1(1e-300) needs more than 300 digits
            ref = (Decimal(u).exp() - 1).ln() if u < 700 else Decimal(u)
        assert _close(float(g), ref, rel=1e-12)
    assert got[0] == pytest.approx(math.log(1e-300), rel=1e-14)


@pytest.mark.parametrize("x0", [-1e3, -1.0, 0.0, 1.0, 1e3])
def test_softplus_diff_stable_extremes(x0: float) -> None:
    """Test ``_softplus_diff`` at ``+-1e3``, ``+-1e-300`` and ``0``."""
    dx = np.array([-1e3, -1e-300, 0.0, 1e-300, 1e3])
    got = _softplus_diff(x0, dx)
    assert np.all(np.isfinite(got))
    for d, g in zip(dx, got, strict=True):
        assert _close(float(g), _ref_softplus_diff(x0, float(d)), rel=1e-9)
    assert got[2] == 0.0


# 5. Large t0 and extreme parameters ----------------------------------------

EXTREME: list[Hazard] = [
    LogisticHazard(1e-6, 1e3, 2.0),
    LogisticHazard(1.0, 1.0, 5.0),
    LogisticHazard(1.0, 1e3, 2.0),
    LogisticHazard.with_floor(5e-7, 1e-6, 2e3, 2.0),
    LogisticHazard.with_floor(0.5, 1.0, 2e3, 2.0),
]


@pytest.mark.parametrize("t0", [1e3, 1e8])
@pytest.mark.parametrize("e", [1e-12, 50.0])
@pytest.mark.parametrize("h", EXTREME, ids=repr)
def test_large_t0_extreme_parameters(h: Hazard, t0: float, e: float) -> None:
    """Test the identity at large ``t0`` with no floating-point warnings."""
    # Underflow to 0 (e.g. exp(-1e8)) is exact and harmless; everything else raises.
    with warnings.catch_warnings(), np.errstate(all="raise", under="ignore"):
        warnings.simplefilter("error")
        r = float(h.forward_offsets(t0, e))
        got = float(h.cumulative_between(t0, t0 + r))
        hz = float(h.hazard(t0 + r))
    assert math.isfinite(r)
    assert r >= 0.0
    if h.has_direct_sampler:
        assert got == pytest.approx(e, rel=1e-9)
    else:
        # Brent works with absolute times, so R is resolved to the ulp of t0 + R.
        assert got == pytest.approx(e, abs=1e-9 * e + 8 * EPS * (t0 + r) * hz)


# 6. Validation ---------------------------------------------------------------


@pytest.mark.parametrize(
    "make",
    [
        lambda: LogisticHazard(0.0, 1.0, 0.0),
        lambda: LogisticHazard(1.0, 0.0, 0.0),
        lambda: LogisticHazard(math.inf, 1.0, 0.0),
        lambda: LogisticHazard(1.0, 1.0, math.nan),
    ],
)
def test_invalid_parameters(make) -> None:
    """Test non-finite and non-positive parameters are rejected."""
    with pytest.raises(ValueError, match="must be"):
        make()


def test_repr_and_flags() -> None:
    """Test ``repr`` style and which families have a direct sampler."""
    assert repr(LogisticHazard(1.0, 2.0, 3.0)) == "LogisticHazard(L=1.0, k=2.0, c=3.0)"
    assert LogisticHazard.has_direct_sampler
    assert not LogisticHazard.with_floor(0.2, 1.0, 2.0, 3.0).has_direct_sampler


# 7. Defective falling logistic -----------------------------------------------------------


def test_defective_logistic_limit_and_draws() -> None:
    """Test the cumulative limit and the share of infinite draws."""
    h = LogisticHazard(1.0, -2.0, 2.0)
    limit = 0.5 * float(_softplus(4.0))
    assert h.cumulative_limit == pytest.approx(limit, rel=1e-14)
    total, _ = integrate.quad(
        lambda s: float(h.hazard(s)), 0.0, math.inf, epsabs=0.0, epsrel=1e-12
    )
    assert h.cumulative_limit == pytest.approx(total, rel=1e-9)
    assert float(h.cumulative_hazard(1e4)) == pytest.approx(limit, rel=1e-12)
    t0, n = 1.0, 20_000
    r = h.forward_offsets(np.full(n, t0), np.random.default_rng(7).exponential(size=n))
    p = math.exp(-(limit - float(h.cumulative_hazard(t0))))
    frac = float(np.mean(np.isinf(r)))
    assert abs(frac - p) <= 5 * math.sqrt(p * (1 - p) / n)
    assert h.forward_time(t0, 10.0, cap=50.0) == 50.0
    assert h.forward_time(t0, 10.0) == math.inf


def test_other_families_are_not_defective() -> None:
    """Test the limit is infinite when the tail is not integrable to a limit."""
    assert LogisticHazard(1.0, 1.0, 5.0).cumulative_limit == math.inf
    assert LogisticHazard.with_floor(0.2, 1.0, 1.0, 5.0).cumulative_limit == math.inf
    assert LogisticHazard.with_floor(0.2, 1.0, -1.0, 2.0).cumulative_limit == math.inf


# 8. Non-negativity -----------------------------------------------------------


@pytest.mark.parametrize("name", SIGMOID_NAMES)
def test_hazard_nonnegative_on_grid(name: str) -> None:
    """Test ``h >= 0`` on ``[0, 1e4]`` for every shared instance."""
    t = np.concatenate([np.linspace(0.0, 1e4, 100_001), np.geomspace(1e-9, 1e4, 500)])
    hz = SIGMOID[name].hazard(t)
    assert np.all(np.isfinite(hz))
    assert np.all(hz >= 0.0)


# 9. Reproducibility ----------------------------------------------------------


@pytest.mark.parametrize("name", SIGMOID_NAMES)
def test_sample_residual_lifetimes_reproducible(name: str) -> None:
    """Test a fixed seed gives the same array for varying start times."""
    h = SIGMOID[name]
    t0 = np.random.default_rng(3).uniform(0.0, 10.0, size=300)
    a = sample_residual_lifetimes(h, t0, random_seed=11)
    b = sample_residual_lifetimes(h, t0, random_seed=11)
    c = sample_residual_lifetimes(h, t0, random_seed=12)
    assert a.shape == t0.shape
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


# 10. Simulator smoke runs ---------------------------------------------------


def test_simulate_birth_death_with_floor_logistic() -> None:
    """Test the forward simulator accepts floor-logistic and logistic rates."""
    birth = LogisticHazard.with_floor(0.5, 1.0, 2.0, 1.0)
    death = LogisticHazard(0.4, 1.0, 1.0)
    a = simulate_birth_death(birth, death, terminal_time=3.0, random_seed=5)
    b = simulate_birth_death(birth, death, terminal_time=3.0, random_seed=5)
    assert len(a.records) >= 1
    assert a.records.equals(b.records)
    assert_valid_records(a.records, 3.0, expected_dtypes=DTYPES)


def test_simulate_contour_with_logistic_birth() -> None:
    """Test the contour simulator exercises the backward inversion."""
    birth = LogisticHazard(1.5, 2.0, 1.0)
    tree = simulate_contour_given_survival(birth, 0.5, terminal_time=3.0, random_seed=2)
    assert len(tree.records) >= 1
    assert bool(tree.records["survived"].any())
    assert_valid_records(tree.records, 3.0, expected_dtypes=DTYPES)


# 11. Regressions for the Step 1 amendments ----------------------------------


@pytest.mark.parametrize(
    ("h", "lo", "hi"),
    [
        (LogisticHazard(1.0, -2.0, 2.0), 0.0, 5.0),
        (LogisticHazard(1.0, -2.0, 2.0), 1.0, 40.0),
        (LogisticHazard.with_floor(0.7, 0.6, -4.0, 3.0), 0.0, 6.0),
        (LogisticHazard.with_floor(0.7, 0.6, -4.0, 3.0), 2.0, 30.0),
        (LogisticHazard.with_floor(0.7, 0.6, -2.0, 100.0), 0.0, 50.0),
    ],
    ids=repr,
)
def test_falling_cumulative_between_matches_quadrature(
    h: Hazard, lo: float, hi: float
) -> None:
    """Test ``cumulative_between`` of falling logistics (``k < 0``) and floors."""
    assert float(h.cumulative_between(lo, hi)) == pytest.approx(
        _quad(h, lo, hi), rel=1e-9
    )


def test_falling_far_from_midpoint() -> None:
    """Test ``H(50)`` is about 65 when the midpoint is far away."""
    h = LogisticHazard.with_floor(0.7, 0.6, -2.0, 100.0)
    val = float(h.cumulative_between(0.0, 50.0))
    assert val == pytest.approx(65.0, rel=1e-2)
    assert val == pytest.approx(_quad(h, 0.0, 50.0), rel=1e-12)


def test_logistic_backward_large_u_and_extreme_scales() -> None:
    """Test the log-space backward form where the rho form rounded to 1."""
    h = LogisticHazard(1.0, 1.0, 5.0)
    assert float(h.backward_offsets(1000.0, 50.0)) == pytest.approx(50.0, rel=1e-12)
    assert float(h.backward_offsets_numerical(1000.0, 50.0)) == pytest.approx(
        50.0, rel=1e-9
    )
    g = LogisticHazard(1e-6, 1e3, 2.0)
    assert float(g.backward_offsets(1e6, 0.5)) == pytest.approx(5e5, rel=1e-9)
    assert float(g.backward_offsets_numerical(1e6, 0.5)) == pytest.approx(5e5, rel=1e-9)


X0_GRID = [-800.0, -50.0, 0.0, 50.0, 800.0]
DX_GRID = [-800.0, -40.0, -0.5, -1e-10, 1e-10, 0.5, 40.0, 800.0]


@pytest.mark.parametrize("x0", X0_GRID)
def test_softplus_diff_grid(x0: float) -> None:
    """Test ``_softplus_diff`` against a 60-digit reference, finite everywhere."""
    got = _softplus_diff(x0, np.array(DX_GRID))
    assert np.all(np.isfinite(got))
    for dx, g in zip(DX_GRID, got, strict=True):
        ref = _ref_softplus_diff(x0, dx)
        assert _close(float(g), ref, rel=1e-9), (x0, dx, float(g), float(ref))
    # The naive double-precision reference agrees wherever it does not cancel.
    sp = np.logaddexp(0.0, x0 + np.array(DX_GRID)), np.logaddexp(0.0, x0)
    naive = sp[0] - sp[1]
    ok = np.abs(naive) > 1e-3 * np.maximum(sp[0], sp[1])
    assert np.allclose(got[ok], naive[ok], rtol=1e-9, atol=0.0)
