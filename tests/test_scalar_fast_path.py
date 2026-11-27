"""The scalar ``math`` fast path agrees with the vectorised offsets."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from bdsurv.hazards import (
    CubicHazard,
    GompertzHazard,
    Hazard,
    LogisticHazard,
    QuadraticHazard,
)

ES = [0.0, 1e-300, 1e-12, 1e-3, 1.0, 50.0, 700.0]
LOGISTIC = [
    (1.0, 1.0, 5.0),
    (1.0, -1.0, 5.0),
    (1e-6, 1e3, 5.0),
    (1e-6, -1e3, 5.0),
    (1.0, 1e-3, 5.0),
    (1.0, -1e-3, 5.0),
    (1e6, 1e3, 0.0),
    (1e6, -1e3, 0.0),
    (2.0, 3.0, -4.0),
    (2.0, -3.0, -4.0),
]
GOMPERTZ = [(1.0, 0.0), (0.1, 0.2), (2.0, 3.0), (0.1, -0.2), (5.0, -1e3), (1e-6, 1e-3)]


def _starts(c: float = 5.0) -> list[float]:
    return [0.0, max(c, 1.0), 1e3, 1e8]


def _hazards() -> list[Hazard]:
    return [LogisticHazard(*p) for p in LOGISTIC] + [
        GompertzHazard(*p) for p in GOMPERTZ
    ]


def _vec_forward(h: Hazard, s: float, e: float, cap: float) -> float:
    return min(s + float(h.forward_offsets(s, e)), cap)


def _vec_backward(h: Hazard, s: float, e: float) -> float:
    return max(s - float(h.backward_offsets(s, e)), 0.0)


@pytest.mark.parametrize("cap", [math.inf, 1e4])
def test_scalar_matches_vectorised(cap: float) -> None:
    """Scalar matches vectorised."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for h in _hazards():
            c = getattr(h, "c", 5.0)
            for s in _starts(c):
                for e in ES:
                    f = h.forward_time(s, e, cap)
                    assert f == pytest.approx(
                        _vec_forward(h, s, e, cap), rel=1e-13, abs=1e-300
                    ), (h, s, e)
                    b = h.backward_time(s, e)
                    assert b == pytest.approx(
                        _vec_backward(h, s, e), rel=1e-13, abs=1e-300
                    ), (h, s, e)


def test_extreme_inputs_do_not_raise() -> None:
    """Extreme inputs do not raise."""
    big = [1e-300, 1e-12, 1.0, 1e3, 1e300]
    for h in _hazards():
        for s in [0.0, 1e-300, 1.0, 1e3, 1e8, 1e300]:
            for e in [*ES, *big]:
                r = h.forward_time(s, e, 1e308)
                assert r >= s or r == 1e308
                assert 0.0 <= h.backward_time(s, e) <= s


def test_e_zero_and_start_zero() -> None:
    """E zero and start zero."""
    for h in _hazards():
        assert h.forward_time(3.0, 0.0) == 3.0
        assert h.backward_time(3.0, 0.0) == 3.0
        assert h.backward_time(0.0, 1.0) == 0.0
        assert h.forward_time(0.0, 0.0) == 0.0


def test_defective_gives_cap() -> None:
    """Defective gives cap."""
    logistic = LogisticHazard(1.0, -1.0, 5.0)
    gompertz = GompertzHazard(0.1, -0.2)
    for h, e in [(logistic, 1e3), (gompertz, 1.0)]:
        assert math.isinf(float(h.forward_offsets(0.0, e)))
        assert h.forward_time(0.0, e, 123.0) == 123.0
        assert h.forward_time(0.0, e) == math.inf


def test_cap_clips() -> None:
    """Cap clips."""
    for h in _hazards():
        free = h.forward_time(1.0, 1.0)
        if math.isfinite(free):
            assert h.forward_time(1.0, 1.0, free / 2 + 0.5) == free / 2 + 0.5
            assert h.forward_time(1.0, 1.0, free * 2 + 1.0) == free


def test_backward_exhaustion_gives_zero() -> None:
    """Backward exhaustion gives zero."""
    assert LogisticHazard(1.0, 1.0, 5.0).backward_time(1.0, 1e3) == 0.0
    assert GompertzHazard(0.1, 0.2).backward_time(1.0, 1e3) == 0.0
    assert GompertzHazard(1.0, 0.0).backward_time(1.0, 5.0) == 0.0
    assert LogisticHazard(1.0, -1.0, 5.0).backward_time(1.0, 1e3) == 0.0


def test_floor_split_matches_vectorised() -> None:
    """Floor split matches vectorised."""
    rng = np.random.default_rng(3)
    for floor, k in [(0.2, 1.0), (0.3, -1.0)]:
        h = LogisticHazard.with_floor(floor, 0.8, k, 5.0)
        n = 300
        t0 = rng.uniform(0.0, 12.0, n)
        es = rng.exponential(size=(h.n_draws, n))
        vec = h.forward_offsets_split(t0, es)
        for i in range(n):
            got = h.forward_time_split(float(t0[i]), [float(v) for v in es[:, i]])
            assert got == pytest.approx(float(t0[i] + vec[i]), rel=1e-12, abs=1e-300)
        for i in range(n):
            got = h.backward_time_split(float(t0[i]), [float(v) for v in es[:, i]])
            assert 0.0 <= got <= t0[i]


def _old_check(h: QuadraticHazard | CubicHazard, end_time: float) -> bool:
    """Apply the numpy-based rule: ``h < -1e-12 * scale`` at the candidate times."""
    times = [0.0, end_time]
    if isinstance(h, QuadraticHazard):
        if h.c != 0.0:
            v = -h.b / (2.0 * h.c)
            if 0.0 < v < end_time:
                times.append(v)
        scale = abs(h.a) + abs(h.b) * end_time + abs(h.c) * end_time**2
    else:
        times += [t for t in h._critical_points() if 0.0 < t < end_time]
        scale = (
            abs(h.a)
            + abs(h.b) * end_time
            + abs(h.c) * end_time**2
            + abs(h.d) * end_time**3
        )
    return all(float(h.hazard(t)) >= -1e-12 * scale for t in times)


def test_check_nonnegative_unchanged() -> None:
    """Check nonnegative unchanged."""
    cases: list[tuple[QuadraticHazard | CubicHazard, float]] = [
        (QuadraticHazard(0.1, 0.05, 0.01), 10.0),
        (QuadraticHazard(1.0, -2.0, 1.0), 5.0),  # touches 0 at t = 1
        (QuadraticHazard(1.0, -2.0, 0.9999), 5.0),  # slightly negative
        (QuadraticHazard(1.0, -2.0, 1.0001), 5.0),
        (QuadraticHazard(0.0, 1.0, 0.0), 3.0),
        (QuadraticHazard(0.0, -1.0, 0.0), 3.0),
        (QuadraticHazard(1.0, -3.0, 0.0), 2.0),
        (QuadraticHazard(1.0, -3.0, 0.0), 0.3),
        (QuadraticHazard(0.5, 0.0, -0.1), 2.0),
        (QuadraticHazard(0.5, 0.0, -0.1), 3.0),
        (CubicHazard(1.0, -2.0, 1.0, 0.0), 5.0),
        (CubicHazard(1.0, -2.0, 1.0, 1e-3), 5.0),
        (CubicHazard(1.0, -2.0, 0.9, 0.0), 5.0),
        (CubicHazard(0.1, 0.0, -0.1, 0.02), 4.0),
        (CubicHazard(0.1, 0.0, -0.1, 0.02), 6.0),
        (CubicHazard(0.1, 0.0, -0.1, 0.02), 9.0),
        (CubicHazard(0.0, 0.0, 0.0, 1.0), 2.0),
        (CubicHazard(0.0, 0.0, 0.0, -1.0), 2.0),
    ]
    for h, end in cases:
        ok = _old_check(h, end)
        if ok:
            h.check_nonnegative(end)
        else:
            with pytest.raises(ValueError, match="is negative on"):
                h.check_nonnegative(end)
