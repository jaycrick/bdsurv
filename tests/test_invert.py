"""Test the Brent inversion of cumulative hazards and the ``Hazard`` defaults."""

from __future__ import annotations

import math

import numpy as np
import pytest

from bdsurv import PiecewiseLinear
from bdsurv._hazard import (
    backward_hazard_time,
    cumulative,
    forward_hazard_time,
    value,
)
from bdsurv._invert import backward_offset, forward_offset
from bdsurv.hazards import Hazard, HazardInversionError
from bdsurv.rates import as_rate


class Quadratic(Hazard):
    """Toy hazard ``h = 2 c t`` with ``H = c t^2``."""

    def __init__(self, c: float = 1.0) -> None:
        self.c = c

    def hazard(self, t):
        return 2.0 * self.c * np.asarray(t, dtype=np.float64)

    def cumulative_hazard(self, t):
        return self.c * np.asarray(t, dtype=np.float64) ** 2


class Constant(Hazard):
    """Toy hazard ``h = c``."""

    def __init__(self, c: float) -> None:
        self.c = c

    def hazard(self, t):
        return self.c + 0.0 * np.asarray(t, dtype=np.float64)

    def cumulative_hazard(self, t):
        return self.c * np.asarray(t, dtype=np.float64)


class Defective(Hazard):
    """Toy hazard ``h = exp(-t)`` with ``H(inf) = 1``."""

    def hazard(self, t):
        return np.exp(-np.asarray(t, dtype=np.float64))

    def cumulative_hazard(self, t):
        return -np.expm1(-np.asarray(t, dtype=np.float64))

    @property
    def cumulative_limit(self) -> float:
        return 1.0


class LogLog(Hazard):
    """Toy hazard with ``H = log(1 + log(1 + t))``, too slow to ever reach large ``e``."""

    def hazard(self, t):
        t = np.asarray(t, dtype=np.float64)
        return 1.0 / ((1.0 + t) * (1.0 + np.log1p(t)))

    def cumulative_hazard(self, t):
        return np.log1p(np.log1p(np.asarray(t, dtype=np.float64)))


class Blowup(Quadratic):
    """Toy hazard whose cumulative hazard is ``inf`` beyond ``t = 3``."""

    def cumulative_hazard(self, t):
        t = np.asarray(t, dtype=np.float64)
        return np.where(t > 3.0, np.inf, t**2)


class Nan(Quadratic):
    """Toy hazard that is ``nan`` beyond ``t = 3``."""

    def cumulative_hazard(self, t):
        t = np.asarray(t, dtype=np.float64)
        return np.where(t > 3.0, np.nan, t**2)


E_GRID = [1e-6, 0.1, 1.0, 5.0, 40.0]
T0_GRID = [0.0, 0.1, 1.0, 5.0]


@pytest.mark.parametrize("t0", T0_GRID)
@pytest.mark.parametrize("e", E_GRID)
def test_forward_matches_closed_form(t0: float, e: float) -> None:
    """Test forward matches closed form."""
    got = forward_offset(Quadratic(2.0), t0, e)
    assert got == pytest.approx(math.sqrt(t0**2 + e / 2.0) - t0, rel=1e-10, abs=1e-12)


@pytest.mark.parametrize("c", [0.3, 4.0])
def test_forward_constant_hazard(c: float) -> None:
    """Test forward constant hazard."""
    assert forward_offset(Constant(c), 2.0, 1.5) == pytest.approx(1.5 / c, rel=1e-11)


@pytest.mark.parametrize("x", [0.5, 3.0, 10.0])
@pytest.mark.parametrize("e", [1e-6, 0.1, 0.4])
def test_backward_matches_closed_form(x: float, e: float) -> None:
    """Test backward matches closed form."""
    got = backward_offset(Quadratic(), x, e)
    assert got == pytest.approx(x - math.sqrt(max(x * x - e, 0.0)), rel=1e-9, abs=1e-12)


def test_zero_draw() -> None:
    """Test zero draw."""
    assert forward_offset(Quadratic(), 1.0, 0.0) == 0.0
    assert backward_offset(Quadratic(), 1.0, 0.0) == 0.0


@pytest.mark.parametrize("bad", [-1.0, math.inf, math.nan])
def test_invalid_draw(bad: float) -> None:
    """Test invalid draw."""
    with pytest.raises(ValueError, match="e must be non-negative and finite"):
        forward_offset(Quadratic(), 1.0, bad)
    with pytest.raises(ValueError, match="e must be non-negative and finite"):
        backward_offset(Quadratic(), 1.0, bad)


def test_forward_cap() -> None:
    """Test forward cap."""
    hazard = Quadratic()
    # No event before the cap: the offset to the cap.
    assert forward_offset(hazard, 1.0, 100.0, cap=3.0) == pytest.approx(2.0)
    # An event before the cap: the exact root.
    assert forward_offset(hazard, 1.0, 3.0, cap=3.0) == pytest.approx(1.0, rel=1e-10)
    # A cap at or before the start.
    assert forward_offset(hazard, 2.0, 1.0, cap=2.0) == 0.0


def test_forward_never_evaluates_beyond_cap() -> None:
    """Test forward never evaluates beyond cap."""
    seen: list[float] = []

    class Spy(Quadratic):
        def cumulative_between(self, s, e):
            seen.append(float(e))
            return super().cumulative_between(s, e)

    forward_offset(Spy(), 1.0, 2.0, cap=2.5)
    assert max(seen) <= 2.5 + 1e-12


def test_defective_returns_inf() -> None:
    """Test defective returns inf."""
    hazard = Defective()
    assert forward_offset(hazard, 0.0, 1.0) == math.inf
    assert forward_offset(hazard, 0.0, 5.0) == math.inf
    remaining = 1.0 - float(hazard.cumulative_hazard(1.0))
    assert forward_offset(hazard, 1.0, remaining * 1.01) == math.inf
    r = forward_offset(hazard, 1.0, remaining * 0.5)
    assert float(hazard.cumulative_between(1.0, 1.0 + r)) == pytest.approx(
        remaining * 0.5, rel=1e-9
    )
    assert hazard.forward_time(0.0, 5.0, cap=7.0) == 7.0


def test_non_finite_evaluation_raises() -> None:
    """Test non finite evaluation raises."""
    for hazard in (Blowup(), Nan()):
        with pytest.raises(HazardInversionError):
            forward_offset(hazard, 0.0, 100.0)


def test_bracket_expansion_limit_raises() -> None:
    """Test bracket expansion limit raises."""
    with pytest.raises(HazardInversionError, match="no bracket"):
        forward_offset(LogLog(), 0.0, 30.0)


def test_non_convergence_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test non convergence raises."""
    import bdsurv._invert as inv

    def fail(*_args: object, **_kwargs: object) -> tuple[float, object]:
        class Info:
            converged = False
            flag = "stub"

        return 0.0, Info()

    monkeypatch.setattr(inv, "brentq", fail)
    with pytest.raises(HazardInversionError, match="did not converge"):
        forward_offset(Quadratic(), 0.0, 0.7)


def test_backward_exhaustion_returns_start() -> None:
    """Test backward exhaustion returns start."""
    assert backward_offset(Quadratic(), 2.0, 4.0) == 2.0
    assert backward_offset(Quadratic(), 2.0, 10.0) == 2.0
    assert Quadratic().backward_time(2.0, 10.0) == 0.0


def test_hazard_methods_vectorise() -> None:
    """Test hazard methods vectorise."""
    hazard = Quadratic()
    t0 = np.array([[0.0], [1.0]])
    e = np.array([0.5, 2.0, 3.0])
    got = hazard.forward_offsets(t0, e)
    assert got.shape == (2, 3)
    assert np.allclose(got, np.sqrt(t0**2 + e) - t0, rtol=1e-10)
    back = hazard.backward_offsets(np.array([3.0, 3.0]), np.array([1.0, 20.0]))
    assert np.allclose(back, [3.0 - math.sqrt(8.0), 3.0], rtol=1e-10)
    assert hazard.forward_offsets_numerical(1.0, 3.0).shape == ()


def test_time_methods() -> None:
    """Test time methods."""
    hazard = Quadratic()
    assert hazard.forward_time(1.0, 3.0) == pytest.approx(2.0, rel=1e-10)
    assert hazard.forward_time(1.0, 3.0, cap=1.5) == 1.5
    assert hazard.backward_time(3.0, 1.0) == pytest.approx(math.sqrt(8.0), rel=1e-10)


def test_defaults() -> None:
    """Test defaults."""
    hazard = Quadratic()
    assert hazard.cumulative_limit == math.inf
    assert hazard.breakpoints == ()
    assert hazard.singular_points == ()
    assert hazard.end_time is None
    assert not hazard.has_direct_sampler
    hazard.check_nonnegative(5.0)
    assert float(hazard.cumulative_between(1.0, 3.0)) == 8.0
    with pytest.raises(TypeError):
        Hazard()  # type: ignore[abstract]


def test_rate_methods_match_module_functions() -> None:
    """Test rate methods match module functions."""
    rate = as_rate(PiecewiseLinear([0.5, 2.0, 1.0], [1.5, 0.2, 1.0], [1.0, 2.5, 4.0]))
    t = np.array([0.0, 0.3, 1.0, 2.0, 3.9, 4.0])
    assert np.array_equal(rate.hazard(t), value(rate, t))
    assert np.array_equal(rate.cumulative_hazard(t), cumulative(rate, t))
    for start in (0.0, 0.7, 2.2):
        for e in (0.01, 0.5, 3.0, 50.0):
            assert rate.forward_time(start, e, 4.0) == forward_hazard_time(
                rate, start, e, 4.0
            )
    for start in (0.5, 2.2, 4.0):
        for e in (0.01, 0.5, 3.0, 50.0):
            assert rate.backward_time(start, e) == backward_hazard_time(rate, start, e)
