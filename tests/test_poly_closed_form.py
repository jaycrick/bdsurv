"""Test the closed-form + Newton inversion of Quadratic/CubicHazard (``_polyroots``).

The reference is an exact-rational-coefficient bisection in 80-digit
:mod:`decimal` arithmetic (``mpmath`` is not a dependency).
Brent's method is *not* the reference: with ``xtol = 1e-14`` it is inaccurate for
tiny offsets (large ``n``) and for large ``t0``.
"""

from __future__ import annotations

import math
from decimal import Decimal, localcontext

import numpy as np
import pytest
from scipy import stats

from bdsurv import _polyroots
from bdsurv._hazard import next_event_offset
from bdsurv._polyroots import (
    ETA_NEWTON,
    Method,
    RegimeCounts,
    cubic_roots_array,
    cubic_roots_scalar,
    quartic_roots_array,
    quartic_roots_scalar,
)
from bdsurv.hazards import CubicHazard, HazardInversionError, QuadraticHazard

EPS = float(np.finfo(float).eps)
Poly = QuadraticHazard | CubicHazard

# The four benchmark parameter sets (all non-negative on [0, inf)).
BENCH: dict[str, Poly] = {
    "quadratic": QuadraticHazard(0.1, 0.05, 0.01),
    "quadratic-c0": QuadraticHazard(0.1, 0.1, 0.0),
    "cubic": CubicHazard(0.1, 0.02, 0.005, 0.0005),
    "cubic-d0": CubicHazard(0.1, 0.05, 0.01, 0.0),
}
METHODS: tuple[Method, ...] = ("gated", "closed", "newton")


def _coef(h: Poly) -> tuple[float, float, float, float]:
    return h._coefficients()  # pyright: ignore[reportPrivateUsage]


def _fo(
    h: Poly,
    t0: object,
    e: object,
    cap: float = math.inf,
    *,
    _method: Method = "gated",
    _counts: RegimeCounts | None = None,
) -> np.ndarray:
    """Default ``forward_offsets`` (or the capped hook); ``_polyroots`` if forced."""
    if _method == "gated" and _counts is None:
        if math.isfinite(cap):
            out = h._forward_offsets_capped(  # pyright: ignore[reportPrivateUsage]
                np.asarray(t0, dtype=float), np.asarray(e, dtype=float), cap
            )
            assert out is not None
            return out
        return h.forward_offsets(t0, e)  # pyright: ignore[reportArgumentType]
    if not math.isfinite(cap) and not h.nonnegative_on_half_line:
        h.forward_offsets(t0, e)  # pyright: ignore[reportArgumentType]
    return _polyroots.forward_offsets(
        h,
        _coef(h),
        t0,  # pyright: ignore[reportArgumentType]
        e,  # pyright: ignore[reportArgumentType]
        cap,
        method=_method,
        counts=_counts,
    )


def _bo(
    h: Poly,
    x: object,
    e: object,
    *,
    _method: Method = "gated",
    _counts: RegimeCounts | None = None,
) -> np.ndarray:
    """Default ``backward_offsets``; ``_polyroots`` if forced."""
    if _method == "gated" and _counts is None:
        return h.backward_offsets(x, e)  # pyright: ignore[reportArgumentType]
    xx = np.asarray(x, dtype=float)
    if xx.size:
        h.check_nonnegative(float(np.max(xx)))
    return _polyroots.backward_offsets(
        h,
        _coef(h),
        xx,
        e,  # pyright: ignore[reportArgumentType]
        method=_method,
        counts=_counts,
    )


def _ft(
    h: Poly,
    start: float,
    e: float,
    cap: float = math.inf,
    *,
    _method: Method = "gated",
    _counts: RegimeCounts | None = None,
) -> float:
    """Default ``forward_time``; ``_polyroots`` if forced."""
    if _method == "gated" and _counts is None:
        return h.forward_time(start, e, cap)
    return _polyroots.forward_time(
        h, _coef(h), start, e, cap, method=_method, counts=_counts
    )


def reference(h: Poly, t0: float, e: float, *, backward: bool = False) -> float:
    """High-precision root of ``P(r) = e`` as a Decimal-rounded float's source."""
    return float(reference_decimal(h, t0, e, backward=backward))


def reference_decimal(
    h: Poly, t0: float, e: float, *, backward: bool = False
) -> Decimal:
    """Bisection on the exact polynomial in local coordinates, 80 digits."""
    with localcontext() as ctx:
        ctx.prec = 80
        a, b, c, d = (Decimal(v) for v in _coef(h))
        t, ee = Decimal(t0), Decimal(e)
        al = a + t * (b + t * (c + t * d))
        be = b + t * (2 * c + 3 * d * t)
        ga = c + 3 * d * t
        de = d
        if backward:
            be, de = -be, -de

        def p(r: Decimal) -> Decimal:
            return r * (al + r * (be / 2 + r * (ga / 3 + r * de / 4))) - ee

        lo = Decimal(0)
        hi = ee / al if al > 0 else Decimal(1)
        while p(hi) < 0:
            hi *= 2
        for _ in range(260):
            mid = (lo + hi) / 2
            if p(mid) < 0:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2


def rel_err(value: float, ref: Decimal) -> float:
    """Relative error of a float against a Decimal reference."""
    return float(abs((Decimal(value) - ref) / ref))


# 1. Root solvers ---------------------------------------------------------------


def test_cubic_and_quartic_roots_match_numpy() -> None:
    """Cardano/trig and Ferrari find every real root, scalar and array alike."""
    rng = np.random.default_rng(0)
    coefs3 = rng.normal(size=(300, 4))
    coefs4 = rng.normal(size=(300, 5))
    roots3 = cubic_roots_array(*(coefs3[:, i].copy() for i in range(4)))
    roots4 = quartic_roots_array(*(coefs4[:, i].copy() for i in range(5)))
    for i in range(300):
        for coefs, arr, scalar in (
            (coefs3[i], roots3[i], cubic_roots_scalar(*coefs3[i])),
            (coefs4[i], roots4[i], quartic_roots_scalar(*coefs4[i])),
        ):
            expected = np.sort(
                [z.real for z in np.roots(coefs) if abs(z.imag) < 1e-9 * (1 + abs(z))]
            )
            for found in (np.sort(arr[~np.isnan(arr)]), np.sort(scalar)):
                assert len(found) == len(expected)
                np.testing.assert_allclose(found, expected, rtol=1e-6, atol=1e-6)


# 2. Closed form against the reference ----------------------------------------


@pytest.mark.parametrize("name", list(BENCH))
@pytest.mark.parametrize("method", METHODS)
def test_forward_matches_reference_grid(name: str, method: Method) -> None:
    """Forward offsets, uncapped and capped, at t0 in {0, 1, 100, 1e6}."""
    h = BENCH[name]
    e = np.array([1e-12, 1e-8, 1e-4, 1e-2, 0.5, 1.0, 10.0, 50.0])
    for t0 in (0.0, 1.0, 100.0, 1e6):
        counts = RegimeCounts()
        got = _fo(h, t0, e, _method=method, _counts=counts)
        # A forced regime may flag an element for Brent, which is itself
        # inaccurate for tiny offsets; only unflagged calls are held to 1e-12.
        forced_fallback = method != "gated" and counts.brent > 0
        for r, d in zip(got, e, strict=True):
            if not forced_fallback:
                assert rel_err(float(r), reference_decimal(h, t0, float(d))) < 1e-12
        if forced_fallback:
            continue
        # A window that contains some roots and not others.
        cap = t0 + 0.5
        capped = _fo(h, t0, e, cap, _method=method)
        for r, d in zip(capped, e, strict=True):
            exact = reference_decimal(h, t0, float(d))
            if exact < Decimal(cap - t0):
                assert rel_err(float(r), exact) < 1e-12
            else:
                assert r == pytest.approx(cap - t0, rel=1e-12)
        # The scalar path agrees with the vector path in offset terms.
        for r, d in zip(got, e, strict=True):
            t = _ft(h, t0, float(d), math.inf, _method=method)
            assert abs(t - (t0 + float(r))) <= 8 * EPS * t + 1e-13 * float(r)


@pytest.mark.parametrize("name", list(BENCH))
@pytest.mark.parametrize("method", METHODS)
def test_backward_matches_reference_grid(name: str, method: Method) -> None:
    """Backward offsets, including the exhausted case ``D = x``."""
    h = BENCH[name]
    e = np.array([1e-12, 1e-6, 1e-2, 0.5, 2.0, 1e3])
    for x in (0.0, 1.0, 100.0, 1e4):
        counts = RegimeCounts()
        try:
            got = _bo(h, x, e, _method=method, _counts=counts)
        except HazardInversionError:
            # A forced regime flagged an element for Brent, which cannot
            # resolve offsets of 1e-16 x (the default Brent path fails too).
            assert method != "gated"
            with pytest.raises(HazardInversionError):
                h.backward_offsets_numerical(x, e)
            continue
        if method != "gated" and counts.brent > 0:
            continue
        total = float(h.cumulative_between(0.0, x))
        for r, d in zip(got, e, strict=True):
            if x == 0.0 or total <= d:
                assert r == x
            else:
                exact = reference_decimal(h, x, float(d), backward=True)
                assert rel_err(float(r), exact) < 1e-11 or r == pytest.approx(x)


def test_agrees_with_brent_where_brent_is_accurate() -> None:
    """At rel 1e-10 against Brent for moderate t0 and e (the spec's grid)."""
    for h in BENCH.values():
        e = np.array([1e-4, 1e-2, 0.5, 1.0, 10.0, 50.0])
        for t0 in (0.0, 1.0):
            np.testing.assert_allclose(
                _fo(h, t0, e),
                h.forward_offsets_numerical(t0, e),
                rtol=1e-10,
            )
        x = np.array([3.0, 7.0])
        np.testing.assert_allclose(
            _bo(h, x, np.array([0.05, 0.4])),
            h.backward_offsets_numerical(x, np.array([0.05, 0.4])),
            rtol=1e-10,
        )


def test_default_is_closed_form_and_brent_stays_reachable() -> None:
    """The defaults use ``_polyroots``; the ``_numerical`` methods stay on Brent."""
    h = BENCH["cubic"]
    e = np.array([0.3, 1.2])
    np.testing.assert_array_equal(
        h.forward_offsets(2.0, e),
        _polyroots.forward_offsets(h, _coef(h), 2.0, e),
    )
    np.testing.assert_allclose(
        h.forward_offsets(2.0, e), h.forward_offsets_numerical(2.0, e), rtol=1e-10
    )
    assert h.forward_time(2.0, 0.3, 50.0) == pytest.approx(
        2.0 + float(h.forward_offsets_numerical(2.0, 0.3)), rel=1e-10
    )
    assert h.backward_time(9.0, 0.3) == pytest.approx(
        9.0 - float(h.backward_offsets_numerical(9.0, 0.3)), rel=1e-10
    )


@pytest.mark.parametrize("name", list(BENCH))
def test_backward_time_matches_vector(name: str) -> None:
    """Scalar ``backward_time`` agrees with ``backward_offsets``, incl. exhaustion."""
    h = BENCH[name]
    for x in (0.0, 1.0, 100.0, 1e4):
        for d in (0.0, 1e-9, 1e-2, 0.5, 2.0, 1e6):
            got = h.backward_time(x, d)
            ref = max(x - float(h.backward_offsets(x, d)), 0.0)
            assert abs(got - ref) <= 8 * EPS * max(x, 1.0) + 1e-13
            if float(h.cumulative_between(0.0, x)) <= d:
                assert got == 0.0


def test_capped_hook_matches_loop() -> None:
    """The single-draw capped path equals the per-element loop for the same seed."""
    from bdsurv._seeding import root_rng
    from bdsurv.hazards import sample_residual_lifetimes

    t0 = np.random.default_rng(3).uniform(0.0, 40.0, size=300)
    for h in (*BENCH.values(), QuadraticHazard(2.0, 0.0, -0.0005)):
        end = 50.0
        got = sample_residual_lifetimes(h, t0, random_seed=7, end_time=end)
        e = root_rng(7, "dynamics").exponential(size=t0.shape)
        times = np.array(
            [
                h.forward_time(float(s), float(d), end)
                for s, d in zip(t0, e, strict=True)
            ]
        )
        ref = np.where(times >= end, np.inf, times - t0)
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=0.0)
        assert np.array_equal(np.isinf(got), np.isinf(ref))


# 3. Degenerate dispatch -------------------------------------------------------


def test_linear_dispatch_is_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    """``c = d = 0`` returns ``next_event_offset`` exactly and uses no root finder."""
    h = QuadraticHazard(0.1, 0.1, 0.0)

    def boom(*_a: object, **_k: object) -> None:
        raise AssertionError

    for name in (
        "cubic_roots_scalar",
        "quartic_roots_scalar",
        "cubic_roots_array",
        "quartic_roots_array",
        "_newton_scalar",
        "_newton_array",
    ):
        monkeypatch.setattr(_polyroots, name, boom)
    for t0 in (0.0, 3.0, 1e4):
        for d in (1e-9, 0.7, 25.0):
            expected = next_event_offset(
                float(h.hazard(t0)), h.b, d
            )  # h'(t0) = b exactly
            assert float(_fo(h, t0, d)) == expected
            assert _ft(h, t0, d, math.inf) == t0 + expected


def test_cubic_with_d0_uses_the_cubic_solver(monkeypatch: pytest.MonkeyPatch) -> None:
    """``d = 0`` never calls Ferrari; the cubic solver is used in regime 2."""
    h = BENCH["cubic-d0"]

    def boom(*_a: object, **_k: object) -> None:
        raise AssertionError

    monkeypatch.setattr(_polyroots, "quartic_roots_scalar", boom)
    monkeypatch.setattr(_polyroots, "quartic_roots_array", boom)
    seen: list[int] = []
    real = _polyroots.cubic_roots_array
    monkeypatch.setattr(
        _polyroots, "cubic_roots_array", lambda *a: (seen.append(1), real(*a))[1]
    )
    got = _fo(h, 0.0, np.array([0.3, 5.0]), _method="closed")
    assert seen
    for r, d in zip(got, (0.3, 5.0), strict=True):
        assert rel_err(float(r), reference_decimal(h, 0.0, d)) < 1e-12
    assert _ft(h, 2.0, 3.0, math.inf, _method="closed") > 2.0


def test_quartic_is_used_for_d_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    """``d != 0`` calls Ferrari in regime 2."""
    h = BENCH["cubic"]
    seen: list[int] = []
    real = _polyroots.quartic_roots_array
    monkeypatch.setattr(
        _polyroots, "quartic_roots_array", lambda *a: (seen.append(1), real(*a))[1]
    )
    _fo(h, 0.0, np.array([0.3, 5.0]), _method="closed")
    assert seen


def test_near_degenerate_coefficients() -> None:
    """``c = 1e-12`` and ``d = 1e-15`` are handled (regime 1 or 2, no failure)."""
    for h in (
        QuadraticHazard(0.1, 0.1, 1e-12),
        CubicHazard(0.1, 0.1, 1e-12, 1e-15),
        CubicHazard(0.1, 0.1, 0.0, 1e-15),
    ):
        e = np.array([1e-9, 1e-3, 1.0, 40.0, 400.0])
        for t0 in (0.0, 10.0, 1e4):
            for method in METHODS:
                counts = RegimeCounts()
                got = _fo(h, t0, e, _method=method, _counts=counts)
                if method != "gated" and counts.brent > 0:
                    continue
                for r, d in zip(got, e, strict=True):
                    ex = reference_decimal(h, t0, float(d))
                    assert rel_err(float(r), ex) < 1e-10


# 4. Windows where the hazard is only valid on [0, T] -------------------------


def test_negative_leading_coefficient_picks_the_right_root() -> None:
    """With three real roots of ``P - e``, the root inside ``[0, T]`` is chosen."""
    h = CubicHazard(0.1, 1.0, -0.2, 0.01)  # h >= 0 on [0, 8]; P - e has 3+ real roots
    cap = 8.0
    h.check_nonnegative(cap)
    e = np.array([0.01, 0.2, 1.0, 3.0, 6.0, 12.0, 20.0, 60.0])
    for t0 in (0.0, 2.0, 5.0):
        for method in METHODS:
            got = _fo(h, t0, e, cap, _method=method)
            for r, d in zip(got, e, strict=True):
                exact = reference_decimal(h, t0, float(d))
                if exact < Decimal(cap - t0):
                    assert rel_err(float(r), exact) < 1e-12
                else:
                    assert r == pytest.approx(cap - t0)


def test_quadratic_negative_c_capped() -> None:
    """``c < 0`` is valid on ``[0, T]``; capped forward_time never exceeds the cap."""
    h = QuadraticHazard(0.5, 0.8, -0.1)
    h.check_nonnegative(8.0)
    for d in (0.01, 0.5, 3.0, 8.0, 1e3):
        t = _ft(h, 1.0, d, 8.0)
        assert 1.0 < t <= 8.0
        assert t == pytest.approx(h.forward_time(1.0, d, 8.0), rel=1e-9)
    with pytest.raises(ValueError, match="not non-negative"):
        _fo(h, 1.0, 0.3)


def test_hazard_touching_zero_falls_back_to_brent() -> None:
    """A near-double zero inside the window is ill-conditioned: Brent, flagged."""
    h = QuadraticHazard(1.0, -2.0, 1.0)  # (t - 1)^2
    counts = RegimeCounts()
    e = np.array([0.001, 0.2, 0.3])
    got = _fo(h, 0.0, e, 3.0, _counts=counts)
    assert counts.brent == 3
    np.testing.assert_allclose(got, h.forward_offsets_numerical(0.0, e), rtol=1e-9)
    scalar = RegimeCounts()
    _ft(h, 0.0, 0.2, 3.0, _counts=scalar)
    assert scalar.brent == 1
    # The whole offset is the right one even though every closed form is unreliable.
    exact = reference_decimal(h, 0.0, 0.2)
    assert rel_err(float(got[1]), exact) < 1e-9


def test_large_t0_does_not_cancel() -> None:
    """At t0 = 1e6 the local coordinates keep full accuracy where Brent loses it."""
    h = QuadraticHazard(1e-3, 1e-3, 1e-6)
    e = np.array([1e-6, 1.0, 100.0])
    got = _fo(h, 1e6, e)
    for r, d in zip(got, e, strict=True):
        assert rel_err(float(r), reference_decimal(h, 1e6, float(d))) < 1e-12


def test_exhaustion_and_zero_conventions() -> None:
    """``e = 0`` gives 0; a capped no-event gives ``cap - t0``; backward gives ``x``."""
    h = BENCH["cubic"]
    assert _fo(h, 3.0, 0.0) == 0.0
    assert _ft(h, 3.0, 0.0, 10.0) == 3.0
    assert _fo(h, 3.0, 1e6, 5.0) == 2.0
    assert _ft(h, 3.0, 1e6, 5.0) == 5.0
    assert _bo(h, 2.0, 1e6) == 2.0
    assert _bo(h, 0.0, 0.5) == 0.0
    with pytest.raises(ValueError, match="non-negative and finite"):
        _fo(h, 1.0, -1.0)
    with pytest.raises(ValueError, match="non-negative and finite"):
        _ft(h, 1.0, math.inf, 3.0)
    with pytest.raises(ValueError, match="non-negative and finite"):
        _bo(h, 1.0, math.nan)


# 5. Dependence on n (e = E / n) ----------------------------------------------

NS = (1.0, 10.0, 1e3, 1e5, 1e7)
T0S = (0.0, 10.0, 1e4)
E_BASE = (0.3, 1.0, 3.0)


def error_table(
    h: Poly,
) -> list[tuple[float, float, dict[str, float], dict[str, RegimeCounts]]]:
    """Per ``(t0, n)`` the worst relative error of each method, and regime counts."""
    rows = []
    for t0 in T0S:
        for n in NS:
            e = np.array(E_BASE) / n
            refs = [reference_decimal(h, t0, float(d)) for d in e]
            counts = {"closed": RegimeCounts(), "gated": RegimeCounts()}
            results = {
                "closed": _fo(h, t0, e, _method="closed", _counts=counts["closed"]),
                "gated": _fo(h, t0, e, _counts=counts["gated"]),
                "brent": h.forward_offsets_numerical(t0, e),
            }
            errs = {
                k: max(rel_err(float(r), x) for r, x in zip(v, refs, strict=True))
                for k, v in results.items()
            }
            rows.append((t0, n, errs, counts))
    return rows


@pytest.mark.parametrize("name", list(BENCH))
def test_gated_accuracy_vs_n(name: str) -> None:
    """Gated stays within max(1e-12, 8 eps (t0+r)/r) and never falls back.

    Run with ``-s`` to print the error table (forced closed / gated / Brent).
    """
    h = BENCH[name]
    rows = error_table(h)
    print(f"\n[{name}] t0, n: relative error closed / gated / brent; fallbacks")
    for t0, n, errs, counts in rows:
        print(
            f"  t0={t0:<7g} n={n:<8g} closed={errs['closed']:.1e} "
            f"gated={errs['gated']:.1e} brent={errs['brent']:.1e}  "
            f"gated regimes N/A/B="
            f"{counts['gated'].newton}/{counts['gated'].algebraic}/"
            f"{counts['gated'].brent}"
        )
    for t0, n, errs, counts in rows:
        r = float(reference(h, t0, E_BASE[0] / n))
        bound = max(1e-12, 8.0 * EPS * (t0 + r) / r)
        assert errs["gated"] <= bound, (t0, n)
        assert counts["gated"].brent == 0, (t0, n)
        assert errs["closed"] <= bound or counts["closed"].brent > 0


def test_forced_algebraic_degrades_at_extreme_n_and_gate_does_not() -> None:
    """At t0 = 1e4 and n = 1e11 Cardano/Ferrari loses digits; the gate does not."""
    h = BENCH["cubic"]
    e = np.array([0.3, 1.0, 3.0]) / 1e11
    refs = [reference_decimal(h, 1e4, float(d)) for d in e]
    gated = _fo(h, 1e4, e)
    closed = _fo(h, 1e4, e, _method="closed")
    err_g = max(rel_err(float(r), x) for r, x in zip(gated, refs, strict=True))
    err_c = max(rel_err(float(r), x) for r, x in zip(closed, refs, strict=True))
    assert err_g < 1e-14
    assert err_c > 10 * err_g


def test_gate_regimes_follow_eta() -> None:
    """Newton dominates at large n; the algebraic regime at small n and t0 = 0."""
    h = BENCH["cubic"]
    big = RegimeCounts()
    _fo(h, 0.0, np.full(10, 1.0 / 1e5), _counts=big)
    assert big.newton == 10
    small = RegimeCounts()
    _fo(h, 0.0, np.full(10, 40.0), _counts=small)
    assert small.algebraic == 10
    forced = RegimeCounts()
    _fo(h, 0.0, np.full(10, 1.0), _method="newton", _counts=forced)
    assert forced.newton + forced.brent == 10
    assert pytest.approx(1e-2) == ETA_NEWTON


@pytest.mark.parametrize("name", list(BENCH))
def test_no_fallback_on_benchmark_sets(name: str) -> None:
    """The gated path never uses Brent on the benchmark parameter sets."""
    h = BENCH[name]
    rng = np.random.default_rng(3)
    for n in (1.0, 1e2, 1e4):
        t0 = rng.uniform(0.0, 30.0, size=2000)
        e = rng.exponential(size=2000) / n
        counts = RegimeCounts()
        _fo(h, t0, e, _counts=counts)
        assert counts.brent == 0
        assert counts.total == 2000
        back = RegimeCounts()
        _bo(h, t0 + 1.0, e, _counts=back)
        assert back.brent == 0


# 6. Distribution (5 SE) -------------------------------------------------------


@pytest.mark.parametrize("name", ["quadratic", "cubic"])
@pytest.mark.parametrize("method", METHODS)
def test_closed_form_distribution(name: str, method: Method) -> None:
    """``R`` has the survival ``exp(-(H(t0+r) - H(t0)))`` to within 5 SE."""
    h = BENCH[name]
    n = 20_000
    rng = np.random.default_rng(17)
    for t0 in (0.0, 4.0):
        r = _fo(h, t0, rng.exponential(size=n), _method=method)
        for q in (0.1, 0.5, 1.0, 2.0, 4.0):
            p = math.exp(-float(h.cumulative_between(t0, t0 + q)))
            se = math.sqrt(p * (1.0 - p) / n)
            assert abs(float(np.mean(r > q)) - p) <= 5.0 * se
        u = 1.0 - np.exp(-h.cumulative_between(t0, t0 + r))
        assert stats.kstest(u, "uniform").pvalue > 1e-3


def test_backward_distribution() -> None:
    """The backward offset has ``P(D > d) = exp(-(H(x) - H(x - d)))``."""
    h = BENCH["cubic"]
    n = 20_000
    x = 6.0
    rng = np.random.default_rng(23)
    d = _bo(h, x, rng.exponential(size=n))
    for q in (0.5, 1.5, 3.0, 5.0):
        p = math.exp(-float(h.cumulative_between(x - q, x)))
        se = math.sqrt(p * (1.0 - p) / n)
        assert abs(float(np.mean(d > q)) - p) <= 5.0 * se
