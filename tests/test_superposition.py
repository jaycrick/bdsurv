"""Test k < 0 logistic hazards, SumHazard superposition and the split samplers."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest
from _helpers import assert_valid_records
from scipy import integrate, stats

from bdsurv import (
    count_draws,
    simulate_birth_death,
    simulate_contour_given_survival,
)
from bdsurv._seeding import root_rng
from bdsurv.hazards import (
    GompertzHazard,
    Hazard,
    LogisticHazard,
    PiecewiseHazard,
    PowerHazard,
    QuadraticHazard,
    SumHazard,
    sample_residual_lifetimes,
)

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
N = 20_000
N_LOOP = 3_000
RS = (0.1, 0.5, 1.0, 2.0, 4.0)
T0S = [0.0, 3.0, 5.0, 20.0]

FALLING = [(1.5, -0.8, 2.0), (0.8, -1.5, -1.0), (2.0, -3.0, 4.0)]
FLOOR_RISING = LogisticHazard.with_floor(0.2, 0.8, 1.0, 5.0)
FLOOR_FALLING = LogisticHazard.with_floor(0.2, 0.8, -1.0, 5.0)
SUMS: dict[str, Hazard] = {
    "floor-rising": FLOOR_RISING,
    "floor-falling": FLOOR_FALLING,
    "gompertz-makeham": SumHazard.with_floor(0.1, GompertzHazard(0.05, 0.3)),
    "hump": SumHazard((LogisticHazard(1.0, 2.0, 3.0), LogisticHazard(1.0, -2.0, 6.0))),
    "defective": SumHazard(
        (LogisticHazard(0.8, -1.0, 5.0), LogisticHazard(0.5, -2, 3))
    ),
    "three-parts": SumHazard(
        (
            GompertzHazard(0.1, 0.0),
            PowerHazard.weibull(1.5, 4.0),
            LogisticHazard(1, 1, 2),
        )
    ),
}


def _quad(h: Hazard, lo: float, hi: float) -> float:
    val, _ = integrate.quad(
        lambda s: float(h.hazard(s)), lo, hi, epsabs=0.0, epsrel=1e-12, limit=200
    )
    return val


def _p_finite(h: Hazard, t0: float) -> float:
    rem = h.cumulative_limit - float(h.cumulative_hazard(t0))
    return -math.expm1(-rem) if math.isfinite(rem) else 1.0


def _check_forward_law(h: Hazard, t0: np.ndarray, r: np.ndarray) -> None:
    """KS of ``H(t0+R) - H(t0)`` and 5-SE survival, for a common start ``t0``."""
    start = float(t0[0])
    n = len(r)
    finite = np.isfinite(r)
    p = _p_finite(h, start)
    if p * n < 50:  # a nearly-dead start: only the survival checks have power
        assert float(np.mean(finite)) <= p + 5 * math.sqrt(p / n) + 1e-12
        return
    u = h.cumulative_between(start, start + r[finite])
    ks = stats.kstest(u, lambda x: stats.expon.cdf(x) / p)
    assert ks.pvalue > 1e-4
    for rr in RS:
        surv = math.exp(-float(h.cumulative_between(start, start + rr)))
        phat = float(np.mean(r > rr))
        assert abs(phat - surv) <= 5 * math.sqrt(surv * (1 - surv) / n) + 1e-12


# 1. LogisticHazard with k < 0 ------------------------------------------------


@pytest.mark.parametrize(("L", "k", "c"), FALLING)
def test_falling_logistic_values_and_limit(L: float, k: float, c: float) -> None:
    """Hazard, cumulative hazard and the defect agree with quadrature."""
    h = LogisticHazard(L, k, c)
    assert float(h.hazard(c)) == pytest.approx(L / 2, rel=1e-12)
    for t in (0.0, 0.5, 2.0, 7.0):
        assert float(h.cumulative_hazard(t)) == pytest.approx(
            _quad(h, 0.0, t), rel=1e-9
        )
        assert float(h.cumulative_between(t, t + 1.5)) == pytest.approx(
            _quad(h, t, t + 1.5), rel=1e-9
        )
    limit, _ = integrate.quad(
        lambda s: float(h.hazard(s)), 0.0, np.inf, epsabs=0.0, epsrel=1e-12, limit=400
    )
    assert h.cumulative_limit == pytest.approx(limit, rel=1e-9)
    assert float(h.cumulative_hazard(1e6)) == pytest.approx(limit, rel=1e-9)
    assert LogisticHazard(L, -k, c).cumulative_limit == math.inf


@pytest.mark.parametrize(("L", "k", "c"), FALLING)
def test_falling_logistic_forward_matches_brent(L: float, k: float, c: float) -> None:
    """Closed-form forward offsets equal Brent, near the defect and beyond it."""
    h = LogisticHazard(L, k, c)
    for t0 in (0.0, 0.3, 2.0, 6.0):
        rem = h.cumulative_limit - float(h.cumulative_hazard(t0))
        e = rem * np.array([1e-8, 1e-3, 0.1, 0.5, 0.9, 0.99, 1 - 1e-4])
        fast = h.forward_offsets(t0, e)
        slow = h.forward_offsets_numerical(t0, e)
        assert np.all(np.isfinite(fast))
        assert fast == pytest.approx(slow, rel=1e-6, abs=1e-9)
        # The identity holds where the root is not astronomically large.
        ok = fast < 1e3
        back = h.cumulative_between(t0, t0 + fast[ok])
        assert back == pytest.approx(e[ok], rel=1e-8)
        beyond = np.array([rem * (1 + 1e-9), rem * 1.001, 2 * rem, rem + 5.0, 1e3])
        assert np.all(np.isinf(h.forward_offsets(t0, beyond)))
        assert np.all(np.isinf(h.forward_offsets_numerical(t0, beyond)))


@pytest.mark.parametrize(("L", "k", "c"), FALLING)
def test_falling_logistic_backward_matches_brent(L: float, k: float, c: float) -> None:
    """Closed-form backward offsets equal Brent, including exhaustion at 0."""
    h = LogisticHazard(L, k, c)
    for x in (0.2, 1.0, 4.0, 9.0):
        hx = float(h.cumulative_hazard(x))
        e = np.array([1e-8, 1e-3, 0.1 * hx, 0.5 * hx, 0.99 * hx, hx * (1 - 1e-6)])
        fast = h.backward_offsets(x, e)
        slow = h.backward_offsets_numerical(x, e)
        assert fast == pytest.approx(slow, rel=1e-6, abs=1e-8)
        assert np.all((fast >= 0) & (fast <= x))
        over = np.array([hx * (1 + 1e-9), hx * 1.01, 2 * hx + 1.0, 1e3])
        assert np.all(h.backward_offsets(x, over) == x)
        assert np.all(h.backward_offsets_numerical(x, over) == x)


@pytest.mark.parametrize(("L", "k", "c"), FALLING)
@pytest.mark.parametrize("t0", [0.0, 1.0, 5.0])
def test_falling_logistic_forward_law(L: float, k: float, c: float, t0: float) -> None:
    """The truncated law of the defective direct sampler (KS and 5 SE)."""
    h = LogisticHazard(L, k, c)
    rng = np.random.default_rng(int(10 * L + 3 * t0))
    r = h.forward_offsets(np.full(N, t0), rng.exponential(size=N))
    assert 0.0 < np.mean(np.isfinite(r)) <= 1.0
    p = _p_finite(h, t0)
    assert abs(np.mean(np.isfinite(r)) - p) <= 5 * math.sqrt(p * (1 - p) / N) + 1e-12
    _check_forward_law(h, np.full(N, t0), r)


@pytest.mark.parametrize(("L", "k", "c"), FALLING)
def test_falling_logistic_backward_law(L: float, k: float, c: float) -> None:
    """P(D > d) = exp(-(H(x) - H(x-d))) and P(D = x) = exp(-H(x))."""
    h = LogisticHazard(L, k, c)
    x = 3.0
    rng = np.random.default_rng(7)
    d = h.backward_offsets(np.full(N, x), rng.exponential(size=N))
    for dd in (0.2, 0.7, 1.5, 2.5):
        surv = math.exp(-float(h.cumulative_between(x - dd, x)))
        phat = float(np.mean(d > dd))
        assert abs(phat - surv) <= 5 * math.sqrt(surv * (1 - surv) / N) + 1e-12
    p0 = math.exp(-float(h.cumulative_hazard(x)))
    assert abs(np.mean(d == x) - p0) <= 5 * math.sqrt(p0 * (1 - p0) / N) + 1e-12


# 2. SumHazard: construction and validation ------------------------------------


def test_sum_values() -> None:
    """The sum's functions are the sums of the parts' functions."""
    a, b = GompertzHazard(0.3, 0.2), LogisticHazard(1.0, -1.0, 2.0)
    s = SumHazard((a, b))
    t = np.array([0.0, 1.0, 4.0])
    assert s.hazard(t) == pytest.approx(a.hazard(t) + b.hazard(t))
    assert s.cumulative_hazard(t) == pytest.approx(
        a.cumulative_hazard(t) + b.cumulative_hazard(t)
    )
    assert s.cumulative_between(t, t + 1.0) == pytest.approx(
        a.cumulative_between(t, t + 1.0) + b.cumulative_between(t, t + 1.0)
    )
    assert s.cumulative_limit == math.inf
    assert SUMS["defective"].cumulative_limit == pytest.approx(
        sum(p.cumulative_limit for p in SUMS["defective"].parts)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownArgumentType]
    )
    assert float(s.cumulative_between(1.0, 3.0)) == pytest.approx(
        _quad(s, 1.0, 3.0), rel=1e-9
    )
    assert s.n_draws == 2
    assert not s.has_direct_sampler


def test_sum_validation() -> None:
    """Too few parts, piecewise parts, parts with an end time and bad floors fail."""
    g = GompertzHazard(0.3, 0.2)
    with pytest.raises(ValueError, match="at least two"):
        SumHazard((g,))
    with pytest.raises(ValueError, match="at least two"):
        SumHazard(())
    pw = PiecewiseHazard.from_polynomials([[0.5], [1.0]], [1.0, 2.0])
    with pytest.raises(ValueError, match=r"piecewise|end_time"):
        SumHazard((g, pw))
    with pytest.raises(TypeError, match="Hazards"):
        SumHazard((g, 0.3))  # pyright: ignore[reportArgumentType]
    for bad in (-0.1, math.inf, math.nan):
        with pytest.raises(ValueError, match="floor"):
            SumHazard.with_floor(bad, g)
        with pytest.raises(ValueError, match="floor"):
            LogisticHazard.with_floor(bad, 1.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="k must"):
        LogisticHazard.with_floor(0.1, 1.0, 0.0, 0.0)


def test_sum_flattening_and_zero_floor() -> None:
    """Nested sums flatten; a zero floor returns the hazard unchanged."""
    a, b, c = (
        GompertzHazard(0.1, 0.0),
        GompertzHazard(0.2, 0.1),
        LogisticHazard(1, 1, 1),
    )
    s = SumHazard((SumHazard((a, b)), c))
    assert s.parts == (a, b, c)
    assert s.n_draws == 3
    assert SumHazard((a, SumHazard((b, c)))).parts == (a, b, c)
    assert SumHazard.with_floor(0.0, c) is c
    assert LogisticHazard.with_floor(0.0, 1.0, 1.0, 1.0) == LogisticHazard(1, 1, 1)
    f = SumHazard.with_floor(0.3, c)
    assert f == SumHazard((GompertzHazard(0.3, 0.0), c))
    assert f.n_draws == 2  # pyright: ignore[reportAttributeAccessIssue]


def test_sum_hazard_value_of_floor_constructors() -> None:
    """Rising and falling floors take the values floor + L sigma(k (t - c))."""
    for k in (1.0, -1.0):
        h = LogisticHazard.with_floor(0.2, 0.8, k, 5.0)
        t = np.array([0.0, 4.0, 5.0, 9.0])
        assert h.hazard(t) == pytest.approx(0.2 + 0.8 / (1 + np.exp(-k * (t - 5.0))))
    assert float(FLOOR_FALLING.hazard(0.0)) > float(FLOOR_FALLING.hazard(20.0))
    assert float(FLOOR_FALLING.hazard(60.0)) == pytest.approx(0.2, rel=1e-12)
    assert float(FLOOR_RISING.hazard(60.0)) == pytest.approx(1.0, rel=1e-12)


# 3. SumHazard: forward split law ----------------------------------------------


@pytest.mark.parametrize("name", SUMS)
@pytest.mark.parametrize("t0", T0S)
def test_forward_split_law(name: str, t0: float) -> None:
    """Vectorised split forward draws have the law of the summed hazard."""
    h = SUMS[name]
    rng = np.random.default_rng(sorted(SUMS).index(name) * 11 + int(t0))
    r = h.forward_offsets_split(np.full(N, t0), rng.exponential(size=(h.n_draws, N)))
    assert r.shape == (N,)
    _check_forward_law(h, np.full(N, t0), r)


@pytest.mark.parametrize("name", ["floor-rising", "floor-falling", "defective"])
def test_forward_split_varying_t0(name: str) -> None:
    """A mixed array of start times gives the right law within each group."""
    h = SUMS[name]
    rng = np.random.default_rng(99)
    t0 = rng.choice(np.array(T0S), size=4 * N)
    r = h.forward_offsets_split(t0, rng.exponential(size=(h.n_draws, 4 * N)))
    assert r.shape == t0.shape
    for s in T0S:
        sel = t0 == s
        _check_forward_law(h, t0[sel], r[sel])


@pytest.mark.parametrize("name", ["floor-rising", "floor-falling", "hump"])
@pytest.mark.parametrize("t0", [0.0, 5.0])
def test_forward_time_split_loop_law(name: str, t0: float) -> None:
    """The scalar ``forward_time_split`` loop has the same law."""
    h = SUMS[name]
    rng = np.random.default_rng(5 + int(t0))
    es = rng.exponential(size=(h.n_draws, N_LOOP))
    t = np.array(
        [h.forward_time_split(t0, es[:, i].tolist(), math.inf) for i in range(N_LOOP)]
    )
    _check_forward_law(h, np.full(N_LOOP, t0), t - t0)
    cap = t0 + 1.0
    capped = np.array(
        [h.forward_time_split(t0, es[:, i].tolist(), cap) for i in range(N_LOOP)]
    )
    assert np.all(capped <= cap)
    assert np.array_equal(capped, np.minimum(t, cap))


def test_split_is_min_of_parts() -> None:
    """The split offsets are the elementwise minimum of the parts' offsets."""
    h = SUMS["gompertz-makeham"]
    rng = np.random.default_rng(1)
    es = rng.exponential(size=(2, 50))
    t0 = rng.uniform(0, 10, 50)
    r = h.forward_offsets_split(t0, es)
    expect = np.minimum(
        h.parts[0].forward_offsets(t0, es[0]),  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownArgumentType]
        h.parts[1].forward_offsets(t0, es[1]),  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownArgumentType]
    )
    assert np.array_equal(r, expect)


def test_default_split_is_single_draw() -> None:
    """A one-draw hazard's split methods reduce to its single-E methods."""
    g = GompertzHazard(0.3, 0.2)
    assert g.n_draws == 1
    e = np.array([0.1, 1.0, 3.0])
    assert np.array_equal(
        g.forward_offsets_split(1.0, e[None, :]), g.forward_offsets(1.0, e)
    )
    assert g.forward_time_split(1.0, [0.7], 50.0) == g.forward_time(1.0, 0.7, 50.0)
    assert g.backward_time_split(3.0, [0.7]) == g.backward_time(3.0, 0.7)


# 4. SumHazard: backward split law ---------------------------------------------


@pytest.mark.parametrize("name", SUMS)
@pytest.mark.parametrize("x", [1.0, 4.0, 12.0])
def test_backward_split_law(name: str, x: float) -> None:
    """P(D > d) = exp(-(H(x) - H(x-d))) for d < x and P(time == 0) = exp(-H(x))."""
    h = SUMS[name]
    rng = np.random.default_rng(sorted(SUMS).index(name) * 13 + int(x))
    es = rng.exponential(size=(h.n_draws, N_LOOP))
    s = np.array([h.backward_time_split(x, es[:, i].tolist()) for i in range(N_LOOP)])
    assert np.all((s >= 0.0) & (s <= x))
    d = x - s
    for frac in (0.1, 0.3, 0.6, 0.9):
        dd = frac * x
        surv = math.exp(-float(h.cumulative_between(x - dd, x)))
        phat = float(np.mean(d > dd))
        assert abs(phat - surv) <= 5 * math.sqrt(surv * (1 - surv) / N_LOOP) + 1e-12
    p0 = math.exp(-float(h.cumulative_hazard(x)))
    phat0 = float(np.mean(s == 0.0))
    assert abs(phat0 - p0) <= 5 * math.sqrt(p0 * (1 - p0) / N_LOOP) + 1e-12


def test_backward_split_is_max_of_parts() -> None:
    """The split backward time is the largest of the parts' times."""
    h = SUMS["hump"]
    es = [0.4, 0.9]
    assert h.backward_time_split(5.0, es) == max(
        h.parts[0].backward_time(5.0, 0.4),  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownArgumentType]
        h.parts[1].backward_time(5.0, 0.9),  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownArgumentType]
    )


# 5. sample_residual_lifetimes -------------------------------------------------


@pytest.mark.parametrize("name", ["floor-rising", "floor-falling", "defective"])
def test_sample_residual_lifetimes_law(name: str) -> None:
    """The public sampler uses the split path and has the right law."""
    h = SUMS[name]
    r = sample_residual_lifetimes(h, np.full(N, 3.0), random_seed=4)
    _check_forward_law(h, np.full(N, 3.0), r)
    again = sample_residual_lifetimes(h, np.full(N, 3.0), random_seed=4)
    assert np.array_equal(r, again)
    capped = sample_residual_lifetimes(
        h, np.full(500, 3.0), random_seed=4, end_time=4.0
    )
    assert np.all(np.isinf(capped) | (capped <= 1.0 + 1e-12))
    rng_split = np.random.default_rng(np.random.SeedSequence(4))
    assert capped.shape == (500,)
    del rng_split


@pytest.mark.parametrize("name", ["floor-rising", "floor-falling", "defective"])
def test_capped_split_vectorised_matches_loop(name: str) -> None:
    """The vectorised capped split path equals the per-element loop, same seed."""
    h = SUMS[name]
    assert h.has_direct_split
    t0 = np.linspace(0.0, 6.0, 400)
    end_time = 7.0
    got = sample_residual_lifetimes(h, t0, random_seed=11, end_time=end_time)
    es = root_rng(11, "dynamics").exponential(size=(h.n_draws, t0.size))
    times = np.array(
        [
            h.forward_time_split(float(s), es[:, i].tolist(), end_time)
            for i, s in enumerate(t0)
        ]
    )
    loop = np.where(times >= end_time, np.inf, times - t0)
    assert np.array_equal(np.isinf(got), np.isinf(loop))
    fin = np.isfinite(got)
    assert fin.any()
    assert np.allclose(got[fin], loop[fin], rtol=1e-12, atol=1e-14)


def test_capped_split_floor_logistic_vectorised_matches_loop() -> None:
    """The same holds for a rising floor-plus-logistic."""
    h = LogisticHazard.with_floor(0.2, 0.8, 1.0, 5.0)
    assert h.has_direct_split
    t0 = np.linspace(0.0, 9.0, 300)
    got = sample_residual_lifetimes(h, t0, random_seed=5, end_time=10.0)
    es = root_rng(5, "dynamics").exponential(size=(2, t0.size))
    times = np.array(
        [
            h.forward_time_split(float(s), es[:, i].tolist(), 10.0)
            for i, s in enumerate(t0)
        ]
    )
    loop = np.where(times >= 10.0, np.inf, times - t0)
    assert np.array_equal(np.isinf(got), np.isinf(loop))
    fin = np.isfinite(got)
    assert np.allclose(got[fin], loop[fin], rtol=1e-12, atol=1e-14)


def test_non_direct_part_keeps_the_loop() -> None:
    """A part without a direct sampler is not vectorised (and still works)."""
    h = SumHazard((GompertzHazard(0.2, 0.0), QuadraticHazard(0.1, 0.05, 0.01)))
    assert not h.has_direct_split
    r = sample_residual_lifetimes(h, np.full(50, 1.0), random_seed=2, end_time=3.0)
    assert np.all(np.isinf(r) | (r <= 2.0 + 1e-12))


# 6. Floor logistic vs Brent ----------------------------------------------------------------------


@pytest.mark.parametrize("b", [0.4, -0.4])
@pytest.mark.parametrize("t0", [0.0, 5.0])
def test_floor_logistic_split_matches_brent_in_law(b: float, t0: float) -> None:
    """The split path and the Brent path of a floor+logistic agree in law."""
    h = LogisticHazard.with_floor(0.2, 0.8, 2.5 * b, 5.0)
    assert h.n_draws == 2
    rng = np.random.default_rng(int(10 * abs(b)) + int(t0) + (0 if b > 0 else 50))
    split = h.forward_offsets_split(np.full(N, t0), rng.exponential(size=(2, N)))
    brent = h.forward_offsets(np.full(N_LOOP, t0), rng.exponential(size=N_LOOP))
    assert stats.ks_2samp(split, brent).pvalue > 1e-4
    _check_forward_law(h, np.full(N, t0), split)
    # the scalar split loop, too
    es = rng.exponential(size=(2, 1000))
    loop = np.array(
        [h.forward_time_split(t0, es[:, i].tolist(), math.inf) for i in range(1000)]
    )
    assert stats.ks_2samp(loop - t0, brent).pvalue > 1e-4


@pytest.mark.parametrize("b", [0.5, -0.5])
def test_single_logistic_closed_form(b: float) -> None:
    """A rising or falling logistic is one draw, closed form, matching Brent."""
    h = LogisticHazard(1.0, 2.8 * b, 2.0)
    assert h.n_draws == 1
    assert h.has_direct_sampler
    e = np.array([1e-6, 0.01, 0.3, 1.0, 3.0])
    for t0 in (0.0, 1.0, 4.0):
        e_use = (
            e
            if b > 0
            else e[e < 0.9 * (h.cumulative_limit - float(h.cumulative_hazard(t0)))]
        )
        fast = h.forward_offsets(t0, e_use)
        slow = h.forward_offsets_numerical(t0, e_use)
        assert fast == pytest.approx(slow, rel=1e-9, abs=1e-12)


# 7. Simulators ----------------------------------------------------------------


@pytest.mark.parametrize("birth", [FLOOR_RISING, FLOOR_FALLING], ids=["rise", "fall"])
def test_simulate_birth_death_floor_logistic(birth: Hazard) -> None:
    """The forward simulator is reproducible and valid with a split birth."""
    a = simulate_birth_death(birth, 0.3, terminal_time=4.0, random_seed=12)
    b = simulate_birth_death(birth, 0.3, terminal_time=4.0, random_seed=12)
    c = simulate_birth_death(birth, 0.3, terminal_time=4.0, random_seed=13)
    assert a.records.equals(b.records)
    assert not a.records.equals(c.records)
    assert_valid_records(a.records, 4.0, expected_dtypes=DTYPES)
    d = simulate_birth_death(0.3, birth, terminal_time=4.0, random_seed=12)
    assert_valid_records(d.records, 4.0, expected_dtypes=DTYPES)


@pytest.mark.parametrize("birth", [FLOOR_RISING, FLOOR_FALLING], ids=["rise", "fall"])
def test_simulate_contour_floor_logistic(birth: Hazard) -> None:
    """The contour simulator is reproducible and valid with a split birth or death."""
    a = simulate_contour_given_survival(birth, 0.5, terminal_time=3.0, random_seed=2)
    b = simulate_contour_given_survival(birth, 0.5, terminal_time=3.0, random_seed=2)
    assert a.records.equals(b.records)
    assert bool(a.records["survived"].any())
    assert_valid_records(a.records, 3.0, expected_dtypes=DTYPES)
    e = simulate_contour_given_survival(1.0, birth, terminal_time=3.0, random_seed=2)
    f = simulate_contour_given_survival(1.0, birth, terminal_time=3.0, random_seed=2)
    assert e.records.equals(f.records)
    assert_valid_records(e.records, 3.0, expected_dtypes=DTYPES)


class _Unsplit(SumHazard):
    """A sum that hides the split: one draw and the Brent path."""

    @property
    def n_draws(self) -> int:
        return 1

    def forward_time_split(
        self, start: float, es: Sequence[float], cap: float = math.inf
    ) -> float:
        return Hazard.forward_time(self, start, es[0], cap)

    def backward_time_split(self, start: float, es: Sequence[float]) -> float:
        return Hazard.backward_time(self, start, es[0])


def _exp_draws_and_iterations(birth: Hazard, seed: int) -> tuple[int, int]:
    with count_draws() as dc:
        tree = simulate_birth_death(birth, 0.3, terminal_time=4.0, random_seed=seed)
    iterations = dc.extras["events"] + int(bool(tree.records["survived"].any()))
    return dc.draws[("dynamics", "exponential")], iterations


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_count_draws_counts_extra_exponentials(seed: int) -> None:
    """Each event of the general loop draws n_draws(birth) + n_draws(death) values."""
    split = SumHazard.with_floor(0.2, LogisticHazard(0.8, 1.0, 2.0))
    unsplit = _Unsplit(split.parts)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownArgumentType]
    assert split.n_draws == 2
    assert unsplit.n_draws == 1
    draws, iters = _exp_draws_and_iterations(split, seed)
    assert iters > 0
    assert draws == 3 * iters  # two for the birth, one for the constant death
    draws1, iters1 = _exp_draws_and_iterations(unsplit, seed)
    assert iters1 > 0
    assert draws1 == 2 * iters1


def test_unsplit_wrapper_has_same_law_as_split() -> None:
    """The wrapper that hides the split is the Brent path of the same hazard."""
    split = FLOOR_RISING
    unsplit = _Unsplit(split.parts)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownArgumentType]
    rng = np.random.default_rng(3)
    a = split.forward_offsets_split(np.full(N, 2.0), rng.exponential(size=(2, N)))
    b = unsplit.forward_offsets(np.full(N_LOOP, 2.0), rng.exponential(size=N_LOOP))
    assert stats.ks_2samp(a, b).pvalue > 1e-4


# 8. Seeded-stream invariance for a one-draw Hazard ----------------------------

# Golden values taken from the code at commit 8047c98 (before the split interface).
_B = GompertzHazard(0.6, 0.2)
_D = GompertzHazard(0.2, 0.1)


def _total(records: pd.DataFrame, col: str) -> float:
    return float(records[col].replace(np.inf, 0.0).sum())


def test_one_draw_hazard_stream_unchanged() -> None:
    """A one-draw Hazard consumes the stream exactly as before the split interface."""
    r = simulate_birth_death(_B, _D, terminal_time=5.0, random_seed=3).records
    assert len(r) == 83
    assert _total(r, "creation_time") == pytest.approx(323.17884078621904, rel=1e-13)
    assert _total(r, "death_time") == pytest.approx(81.73266860287558, rel=1e-13)
    t = simulate_contour_given_survival(
        _B, _D, terminal_time=4.0, random_seed=4
    ).records
    assert len(t) == 16
    assert _total(t, "creation_time") == pytest.approx(47.259031525463584, rel=1e-13)
    assert _total(t, "death_time") == pytest.approx(7.220614493792411, rel=1e-13)
    x = sample_residual_lifetimes(_B, [0.0, 1.0, 2.0], random_seed=5)
    assert x == pytest.approx(
        [2.540780294243754, 0.9312978466036195, 1.2762122675042602], rel=1e-13
    )


def test_one_draw_hazard_draws_one_exponential_each() -> None:
    """No extra exponentials are drawn for a one-draw Hazard."""
    with count_draws() as dc:
        tree = simulate_birth_death(_B, _D, terminal_time=5.0, random_seed=3)
    iters = dc.extras["events"] + int(bool(tree.records["survived"].any()))
    assert dc.draws[("dynamics", "exponential")] == 2 * iters
