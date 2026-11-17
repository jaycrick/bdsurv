r"""Hazard functions with exact left-truncated sampling.

A :class:`Hazard` is a non-negative function $h$ of absolute time $t \ge 0$ with
cumulative hazard $H(t) = \int_0^t h$.
Given a start time $t_0$ and $E \sim \mathrm{Exp}(1)$, the residual time $R$
with $H(t_0 + R) - H(t_0) = E$ is the exact time to the next event.
:meth:`Hazard.forward_offsets` and :meth:`Hazard.backward_offsets` solve this
(forward in time, or backward from a start time).
Subclasses with a closed-form inverse override them and set
:attr:`Hazard.has_direct_sampler`; otherwise a bracketed Brent solver
(:mod:`bdsurv._invert`) is used.

Families with a direct (closed-form) sampler:
:class:`PowerHazard` (with ``weibull`` and ``rayleigh``),
:class:`GompertzHazard`, :class:`LogLogisticHazard` and :class:`LomaxHazard`.
Families that invert by Brent's method: :class:`QuadraticHazard`.
:class:`PiecewiseHazard` walks its epochs and uses each piece's own method, so it
is direct only if every piece is.

See ``docs/hazards.md`` for the formulas, conventions and how to add a family.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from bdsurv._invert import (
    HazardInversionError,
    backward_offset,
    forward_offset,
)
from bdsurv._seeding import root_rng
from bdsurv.rates import check_epoch_ends

__all__ = [
    "GompertzHazard",
    "Hazard",
    "HazardInversionError",
    "LogLogisticHazard",
    "LomaxHazard",
    "PiecewiseHazard",
    "PowerHazard",
    "QuadraticHazard",
    "sample_residual_lifetimes",
]

_Array = NDArray[np.float64]


class Hazard(ABC):
    """A hazard function of absolute time, with exact inversion of ``H``.

    Subclasses implement :meth:`hazard` and :meth:`cumulative_hazard`.
    Everything else has a default; override the offsets for a closed form.
    """

    has_direct_sampler: ClassVar[bool] = False

    @abstractmethod
    def hazard(self, t: ArrayLike) -> _Array:
        """Hazard ``h(t)``; may be ``inf`` at a singular point."""

    @abstractmethod
    def cumulative_hazard(self, t: ArrayLike) -> _Array:
        """Cumulative hazard ``H(t)``, with ``H(0) = 0``."""

    def cumulative_between(self, s: ArrayLike, e: ArrayLike) -> _Array:
        """Cumulative hazard ``H(e) - H(s)``; override for a stable form."""
        return self.cumulative_hazard(e) - self.cumulative_hazard(s)

    @property
    def cumulative_limit(self) -> float:
        """The limit ``H(inf)``; finite means the hazard is defective."""
        return math.inf

    @property
    def breakpoints(self) -> tuple[float, ...]:
        """Times where ``h`` is not smooth."""
        return ()

    @property
    def singular_points(self) -> tuple[float, ...]:
        """Times where ``h`` is infinite."""
        return ()

    @property
    def end_time(self) -> float | None:
        """The end of the domain, or ``None`` if the hazard is defined for all ``t``."""
        return None

    def check_nonnegative(self, end_time: float) -> None:  # noqa: B027
        """Raise ``ValueError`` if ``h < 0`` somewhere on ``[0, end_time]``.

        The default does nothing.
        """

    def forward_offsets(self, t0: ArrayLike, e: ArrayLike) -> _Array:
        """Offsets ``R >= 0`` with ``H(t0 + R) - H(t0) = e``; ``inf`` if defective."""
        return self.forward_offsets_numerical(t0, e)

    def backward_offsets(self, x: ArrayLike, e: ArrayLike) -> _Array:
        """Offsets ``D`` in ``[0, x]`` with ``H(x) - H(x - D) = e``.

        ``D = x`` (the time reaches 0) if ``H(x) <= e``.
        """
        return self.backward_offsets_numerical(x, e)

    def forward_time(self, start: float, e: float, cap: float = math.inf) -> float:
        """Return ``min(start + R, cap)`` for the offset ``R`` of ``e`` from ``start``.

        ``cap`` is also returned when ``R`` is infinite.
        A hazard without a direct sampler is never evaluated beyond ``cap``.
        """
        if self.has_direct_sampler:
            offset = float(self.forward_offsets(start, e))
        else:
            offset = forward_offset(self, start, e, cap)
        return min(start + offset, cap)

    def backward_time(self, start: float, e: float) -> float:
        """Return ``max(start - D, 0)`` for the offset ``D`` of ``e`` back."""
        return max(start - float(self.backward_offsets(start, e)), 0.0)

    def forward_offsets_numerical(self, t0: ArrayLike, e: ArrayLike) -> _Array:
        """:meth:`forward_offsets` by Brent's method, whatever the subclass."""
        a, b = np.broadcast_arrays(
            np.asarray(t0, dtype=np.float64), np.asarray(e, dtype=np.float64)
        )
        out = np.array(
            [
                forward_offset(self, float(s), float(d))
                for s, d in zip(a.flat, b.flat, strict=True)
            ],
            dtype=np.float64,
        )
        return out.reshape(a.shape)

    def backward_offsets_numerical(self, x: ArrayLike, e: ArrayLike) -> _Array:
        """:meth:`backward_offsets` by Brent's method, whatever the subclass."""
        a, b = np.broadcast_arrays(
            np.asarray(x, dtype=np.float64), np.asarray(e, dtype=np.float64)
        )
        out = np.array(
            [
                backward_offset(self, float(s), float(d))
                for s, d in zip(a.flat, b.flat, strict=True)
            ],
            dtype=np.float64,
        )
        return out.reshape(a.shape)


def _arr(x: ArrayLike) -> _Array:
    return np.asarray(x, dtype=np.float64)


def _check_positive(name: str, value: float) -> None:
    if not (math.isfinite(value) and value > 0.0):
        msg = f"{name} must be positive and finite"
        raise ValueError(msg)


@dataclass(frozen=True)
class PowerHazard(Hazard):
    r"""Power-law hazard $h(t) = c\,t^p$, with $c > 0$ and $p > -1$.

    With $k = p + 1$ and $\eta = ((p+1)/c)^{1/k}$ the cumulative hazard is
    $H(t) = (t/\eta)^k$, the Weibull form.
    For $t_0 > 0$ the inverse is
    $R = t_0\,\mathrm{expm1}(\mathrm{log1p}(e/H(t_0))/k)$,
    and for $t_0 = 0$ it is $R = \eta\,e^{1/k}$.
    Backwards from $x$, $D = -x\,\mathrm{expm1}(\mathrm{log1p}(-e/H(x))/k)$,
    or $D = x$ if $e \ge H(x)$.
    The sampler is direct (closed form).

    The hazard is infinite at $t = 0$ if $p < 0$ (a singular point).

    Parameters
    ----------
    c
        Scale of the hazard; positive and finite.
    p
        Exponent; finite and greater than -1.
    """

    c: float
    p: float

    has_direct_sampler: ClassVar[bool] = True

    def __post_init__(self) -> None:
        """Validate the parameters."""
        _check_positive("c", self.c)
        if not (math.isfinite(self.p) and self.p > -1.0):
            msg = "p must be finite and greater than -1"
            raise ValueError(msg)
        _check_positive("shape", self.k)
        _check_positive("scale", self.eta)

    @classmethod
    def weibull(cls, shape: float, scale: float) -> PowerHazard:
        """Weibull hazard with ``H(t) = (t / scale) ** shape``."""
        _check_positive("shape", shape)
        _check_positive("scale", scale)
        return cls(c=shape / scale**shape, p=shape - 1.0)

    @classmethod
    def rayleigh(cls, c: float) -> PowerHazard:
        """Rayleigh hazard ``h(t) = c * t``."""
        return cls(c=c, p=1.0)

    @property
    def k(self) -> float:
        """The Weibull shape ``p + 1``."""
        return self.p + 1.0

    @property
    def eta(self) -> float:
        """The Weibull scale ``((p + 1) / c) ** (1 / (p + 1))``."""
        return ((self.p + 1.0) / self.c) ** (1.0 / self.k)

    @property
    def singular_points(self) -> tuple[float, ...]:
        """``(0.0,)`` if ``p < 0``."""
        return (0.0,) if self.p < 0.0 else ()

    def hazard(self, t: ArrayLike) -> _Array:
        """Hazard ``c t^p``; infinite at 0 if ``p < 0``."""
        tt = _arr(t)
        with np.errstate(divide="ignore", over="ignore"):
            return self.c * tt**self.p

    def cumulative_hazard(self, t: ArrayLike) -> _Array:
        """Cumulative hazard ``(t / eta) ** k``."""
        with np.errstate(over="ignore"):
            return (_arr(t) / self.eta) ** self.k

    def cumulative_between(self, s: ArrayLike, e: ArrayLike) -> _Array:
        """``H(e) - H(s)`` as ``H(s) expm1(k log1p((e - s) / s))``."""
        ss, ee = np.broadcast_arrays(_arr(s), _arr(e))
        pos = ss > 0.0
        safe = np.where(pos, ss, 1.0)
        with np.errstate(over="ignore", divide="ignore"):
            h0 = (safe / self.eta) ** self.k
            stable = h0 * np.expm1(self.k * np.log1p((ee - safe) / safe))
        return np.where(pos, stable, self.cumulative_hazard(ee))

    def forward_offsets(self, t0: ArrayLike, e: ArrayLike) -> _Array:
        """Closed-form offsets ``R`` with ``H(t0 + R) - H(t0) = e``."""
        tt, ee = np.broadcast_arrays(_arr(t0), _arr(e))
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            h0 = (tt / self.eta) ** self.k
            pos = h0 > 0.0
            safe_t = np.where(pos, tt, 1.0)
            safe_h = np.where(pos, h0, 1.0)
            late = safe_t * np.expm1(np.log1p(ee / safe_h) / self.k)
            early = self.eta * ee ** (1.0 / self.k)
        return np.where(pos, late, early)

    def backward_offsets(self, x: ArrayLike, e: ArrayLike) -> _Array:
        """Closed-form offsets ``D`` with ``H(x) - H(x - D) = e`` (``x`` if spent)."""
        xx, ee = np.broadcast_arrays(_arr(x), _arr(e))
        with np.errstate(over="ignore"):
            hx = (xx / self.eta) ** self.k
        live = ee < hx
        safe_h = np.where(live, hx, 1.0)
        ratio = np.where(live, ee / safe_h, 0.0)
        d = -xx * np.expm1(np.log1p(-ratio) / self.k)
        return np.where(live, np.minimum(d, xx), xx)

    def __repr__(self) -> str:
        """Return ``Name(param=value, ...)``."""
        return f"PowerHazard(c={self.c!r}, p={self.p!r})"


@dataclass(frozen=True)
class GompertzHazard(Hazard):
    r"""Gompertz hazard $h(t) = a\,e^{bt}$, with $a > 0$ and $b$ real.

    The cumulative hazard is $H(t) = a\,t\,\varphi(bt)$ with
    $\varphi(x) = \mathrm{expm1}(x)/x$ and $\varphi(0) = 1$.
    It is defective ($H(\infty) = -a/b$) when $b < 0$.
    With $z = b e\,e^{-b t_0}/a$ the forward offset is
    $R = \mathrm{log1p}(z)/b$, and $R = \infty$ if $z \le -1$.
    With $w = -b e\,e^{-b x}/a$ the backward offset is
    $D = -\mathrm{log1p}(w)/b$, or $D = x$ if $w \le -1$.
    For $b = 0$ the hazard is constant and $R = e/a$.
    The sampler is direct (closed form).

    Parameters
    ----------
    a
        Hazard at time 0; positive and finite.
    b
        Exponential growth rate; finite, of either sign.
    """

    a: float
    b: float

    has_direct_sampler: ClassVar[bool] = True

    def __post_init__(self) -> None:
        """Validate the parameters."""
        _check_positive("a", self.a)
        if not math.isfinite(self.b):
            msg = "b must be finite"
            raise ValueError(msg)

    @property
    def cumulative_limit(self) -> float:
        """``-a / b`` if ``b < 0``, else ``inf``."""
        return -self.a / self.b if self.b < 0.0 else math.inf

    def hazard(self, t: ArrayLike) -> _Array:
        """Hazard ``a exp(b t)``."""
        with np.errstate(over="ignore"):
            return self.a * np.exp(self.b * _arr(t))

    def cumulative_hazard(self, t: ArrayLike) -> _Array:
        """Cumulative hazard ``a t phi(b t)``."""
        tt = _arr(t)
        x = self.b * tt
        nz = x != 0.0
        safe = np.where(nz, x, 1.0)
        with np.errstate(over="ignore"):
            phi = np.where(nz, np.expm1(safe) / safe, 1.0)
        return self.a * tt * phi

    def cumulative_between(self, s: ArrayLike, e: ArrayLike) -> _Array:
        """``H(e) - H(s)`` as ``(a / b) exp(b s) expm1(b (e - s))``."""
        ss, ee = np.broadcast_arrays(_arr(s), _arr(e))
        if self.b == 0.0:
            return self.a * (ee - ss)
        with np.errstate(over="ignore"):
            return (
                (self.a / self.b) * np.exp(self.b * ss) * np.expm1(self.b * (ee - ss))
            )

    def forward_offsets(self, t0: ArrayLike, e: ArrayLike) -> _Array:
        """Closed-form offsets ``R``; ``inf`` when the hazard is defective."""
        tt, ee = np.broadcast_arrays(_arr(t0), _arr(e))
        if self.b == 0.0:
            return ee / self.a
        with np.errstate(over="ignore", invalid="ignore"):
            z = self.b * ee * np.exp(-self.b * tt) / self.a
            dead = z <= -1.0
            r = np.log1p(np.where(dead, 0.0, z)) / self.b
        r = np.where(dead, np.inf, r)
        return np.where(ee == 0.0, 0.0, r)

    def backward_offsets(self, x: ArrayLike, e: ArrayLike) -> _Array:
        """Closed-form offsets ``D`` in ``[0, x]``; ``x`` if exhausted."""
        xx, ee = np.broadcast_arrays(_arr(x), _arr(e))
        if self.b == 0.0:
            return np.minimum(ee / self.a, xx)
        with np.errstate(over="ignore", invalid="ignore"):
            w = -self.b * ee * np.exp(-self.b * xx) / self.a
            dead = w <= -1.0
            d = -np.log1p(np.where(dead, 0.0, w)) / self.b
        d = np.where(dead, xx, np.minimum(d, xx))
        return np.where(ee == 0.0, 0.0, d)

    def __repr__(self) -> str:
        """Return ``Name(param=value, ...)``."""
        return f"GompertzHazard(a={self.a!r}, b={self.b!r})"


@dataclass(frozen=True)
class LogLogisticHazard(Hazard):
    r"""Log-logistic hazard with $H(t) = \mathrm{log1p}((t/\eta)^k)$.

    The hazard is $h(t) = (k/\eta)(t/\eta)^{k-1} / (1 + (t/\eta)^k)$,
    infinite at $t = 0$ if $k < 1$.
    With $x_0 = (t_0/\eta)^k$ the forward offset is
    $R = t_0\,\mathrm{expm1}(\mathrm{log1p}((1+x_0)\,\mathrm{expm1}(e)/x_0)/k)$
    ($R = \eta\,\mathrm{expm1}(e)^{1/k}$ when $t_0 = 0$).
    With $d = (1+x_x)(1 - e^{-e})$ and $x_x = (x/\eta)^k$ the backward offset is
    $D = -x\,\mathrm{expm1}(\mathrm{log1p}(-d/x_x)/k)$, or $D = x$ if $d \ge x_x$.
    The sampler is direct (closed form).

    Parameters
    ----------
    shape
        The shape $k$; positive and finite.
    scale
        The scale $\eta$; positive and finite.
    """

    shape: float
    scale: float

    has_direct_sampler: ClassVar[bool] = True

    def __post_init__(self) -> None:
        """Validate the parameters."""
        _check_positive("shape", self.shape)
        _check_positive("scale", self.scale)

    @property
    def singular_points(self) -> tuple[float, ...]:
        """``(0.0,)`` if ``shape < 1``."""
        return (0.0,) if self.shape < 1.0 else ()

    def _x(self, t: _Array) -> _Array:
        with np.errstate(over="ignore"):
            return (t / self.scale) ** self.shape

    def hazard(self, t: ArrayLike) -> _Array:
        """Hazard ``(k / eta) (t / eta)^(k - 1) / (1 + (t / eta)^k)``."""
        tt = _arr(t)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            num = (self.shape / self.scale) * (tt / self.scale) ** (self.shape - 1.0)
            out = num / (1.0 + self._x(tt))
        return np.where(np.isnan(out), 0.0, out)

    def cumulative_hazard(self, t: ArrayLike) -> _Array:
        """Cumulative hazard ``log1p((t / eta)^k)``."""
        return np.log1p(self._x(_arr(t)))

    def cumulative_between(self, s: ArrayLike, e: ArrayLike) -> _Array:
        """``H(e) - H(s)`` as ``log1p((x1 - x0) / (1 + x0))``."""
        x0 = self._x(_arr(s))
        x1 = self._x(_arr(e))
        with np.errstate(invalid="ignore"):
            return np.log1p((x1 - x0) / (1.0 + x0))

    def forward_offsets(self, t0: ArrayLike, e: ArrayLike) -> _Array:
        """Closed-form offsets ``R`` with ``H(t0 + R) - H(t0) = e``."""
        tt, ee = np.broadcast_arrays(_arr(t0), _arr(e))
        x0 = self._x(tt)
        pos = x0 > 0.0
        safe_t = np.where(pos, tt, 1.0)
        safe_x = np.where(pos, x0, 1.0)
        with np.errstate(over="ignore"):
            em = np.expm1(ee)
            late = safe_t * np.expm1(
                np.log1p((1.0 + safe_x) * em / safe_x) / self.shape
            )
            early = self.scale * em ** (1.0 / self.shape)
        return np.where(pos, late, early)

    def backward_offsets(self, x: ArrayLike, e: ArrayLike) -> _Array:
        """Closed-form offsets ``D`` with ``H(x) - H(x - D) = e`` (``x`` if spent)."""
        xv, ee = np.broadcast_arrays(_arr(x), _arr(e))
        xx = self._x(xv)
        d = (1.0 + xx) * -np.expm1(-ee)
        live = d < xx
        safe_x = np.where(live, xx, 1.0)
        ratio = np.where(live, d / safe_x, 0.0)
        out = -xv * np.expm1(np.log1p(-ratio) / self.shape)
        return np.where(live, np.minimum(out, xv), xv)

    def __repr__(self) -> str:
        """Return ``Name(param=value, ...)``."""
        return f"LogLogisticHazard(shape={self.shape!r}, scale={self.scale!r})"


@dataclass(frozen=True)
class LomaxHazard(Hazard):
    r"""Lomax (Pareto II) hazard $h(t) = k/(\eta + t)$.

    The cumulative hazard is $H(t) = k\,\mathrm{log1p}(t/\eta)$.
    The forward offset is $R = (\eta + t_0)\,\mathrm{expm1}(e/k)$.
    The backward offset is $D = (\eta + x)(-\mathrm{expm1}(-e/k))$, capped at $x$.
    The sampler is direct (closed form).

    Parameters
    ----------
    shape
        The shape $k$; positive and finite.
    scale
        The scale $\eta$; positive and finite.
    """

    shape: float
    scale: float

    has_direct_sampler: ClassVar[bool] = True

    def __post_init__(self) -> None:
        """Validate the parameters."""
        _check_positive("shape", self.shape)
        _check_positive("scale", self.scale)

    def hazard(self, t: ArrayLike) -> _Array:
        """Hazard ``k / (eta + t)``."""
        return self.shape / (self.scale + _arr(t))

    def cumulative_hazard(self, t: ArrayLike) -> _Array:
        """Cumulative hazard ``k log1p(t / eta)``."""
        return self.shape * np.log1p(_arr(t) / self.scale)

    def cumulative_between(self, s: ArrayLike, e: ArrayLike) -> _Array:
        """``H(e) - H(s)`` as ``k log1p((e - s) / (eta + s))``."""
        ss, ee = np.broadcast_arrays(_arr(s), _arr(e))
        return self.shape * np.log1p((ee - ss) / (self.scale + ss))

    def forward_offsets(self, t0: ArrayLike, e: ArrayLike) -> _Array:
        """Closed-form offsets ``R`` with ``H(t0 + R) - H(t0) = e``."""
        tt, ee = np.broadcast_arrays(_arr(t0), _arr(e))
        with np.errstate(over="ignore"):
            return (self.scale + tt) * np.expm1(ee / self.shape)

    def backward_offsets(self, x: ArrayLike, e: ArrayLike) -> _Array:
        """Closed-form offsets ``D`` with ``H(x) - H(x - D) = e`` (``x`` if spent)."""
        xx, ee = np.broadcast_arrays(_arr(x), _arr(e))
        return np.minimum((self.scale + xx) * -np.expm1(-ee / self.shape), xx)

    def __repr__(self) -> str:
        """Return ``Name(param=value, ...)``."""
        return f"LomaxHazard(shape={self.shape!r}, scale={self.scale!r})"


@dataclass(frozen=True)
class QuadraticHazard(Hazard):
    r"""Quadratic hazard $h(t) = a + b\,t + c\,t^2$, with $a \ge 0$.

    The cumulative hazard is $H(t) = a\,t + b\,t^2/2 + c\,t^3/3$.
    There is no direct sampler: the inversion is by Brent's method.

    The hazard need only be non-negative on the window that is used,
    so $c < 0$ is valid on $[0, T]$ if :meth:`check_nonnegative` passes.
    A capped :meth:`~Hazard.forward_time` never evaluates beyond its cap.
    The uncapped :meth:`forward_offsets` needs non-negativity on
    $[0, \infty)$ (:attr:`nonnegative_on_half_line`).

    Parameters
    ----------
    a
        Hazard at time 0; finite and non-negative.
    b
        Linear coefficient; finite.
    c
        Quadratic coefficient; finite.
    """

    a: float
    b: float
    c: float

    has_direct_sampler: ClassVar[bool] = False

    def __post_init__(self) -> None:
        """Validate the parameters."""
        if not all(math.isfinite(v) for v in (self.a, self.b, self.c)):
            msg = "a, b and c must be finite"
            raise ValueError(msg)
        if self.a < 0.0:
            msg = "a must be non-negative"
            raise ValueError(msg)

    @property
    def nonnegative_on_half_line(self) -> bool:
        """Whether ``h >= 0`` on ``[0, inf)``."""
        if self.c == 0.0:
            return self.b >= 0.0
        if self.c < 0.0:
            return False
        return self.b >= 0.0 or self.a - self.b**2 / (4.0 * self.c) >= 0.0

    def check_nonnegative(self, end_time: float) -> None:
        """Raise ``ValueError`` if ``h < 0`` somewhere on ``[0, end_time]``."""
        times = [0.0, end_time]
        if self.c != 0.0:
            vertex = -self.b / (2.0 * self.c)
            if 0.0 < vertex < end_time:
                times.append(vertex)
        scale = abs(self.a) + abs(self.b) * end_time + abs(self.c) * end_time**2
        for t in times:
            value = float(self.hazard(t))
            if value < -1e-12 * scale:
                msg = f"{self!r} is negative on [0, end_time]: h({t!r}) = {value!r}"
                raise ValueError(msg)

    def hazard(self, t: ArrayLike) -> _Array:
        """Hazard ``a + b t + c t^2``."""
        tt = _arr(t)
        return self.a + tt * (self.b + self.c * tt)

    def cumulative_hazard(self, t: ArrayLike) -> _Array:
        """Cumulative hazard ``a t + b t^2 / 2 + c t^3 / 3``."""
        tt = _arr(t)
        return tt * (self.a + tt * (self.b / 2.0 + self.c * tt / 3.0))

    def cumulative_between(self, s: ArrayLike, e: ArrayLike) -> _Array:
        """``H(e) - H(s)`` as ``(e - s)`` times a mean of ``h``."""
        ss, ee = np.broadcast_arrays(_arr(s), _arr(e))
        mean = (
            self.a
            + self.b * (ee + ss) / 2.0
            + self.c * (ee * ee + ee * ss + ss * ss) / 3.0
        )
        return (ee - ss) * mean

    def forward_offsets(self, t0: ArrayLike, e: ArrayLike) -> _Array:
        """Offsets ``R`` by Brent; needs ``h >= 0`` on ``[0, inf)``."""
        if not self.nonnegative_on_half_line:
            msg = (
                f"{self!r} is not non-negative on [0, inf); "
                "use a capped forward_time or an end_time"
            )
            raise ValueError(msg)
        return self.forward_offsets_numerical(t0, e)

    def backward_offsets(self, x: ArrayLike, e: ArrayLike) -> _Array:
        """Offsets ``D`` by Brent; needs ``h >= 0`` on ``[0, max(x)]``."""
        xx = _arr(x)
        if xx.size:
            self.check_nonnegative(float(np.max(xx)))
        return self.backward_offsets_numerical(x, e)

    def __repr__(self) -> str:
        """Return ``Name(param=value, ...)``."""
        return f"QuadraticHazard(a={self.a!r}, b={self.b!r}, c={self.c!r})"


class PiecewiseHazard(Hazard):
    r"""A hazard made of one :class:`Hazard` per epoch, in local time.

    Piece $k$ applies on $(\tau_{k-1}, \tau_k]$ with $\tau_0 = 0$, and is
    evaluated at the local time $u = t - \tau_{k-1}$.
    The hazard is defined on $[0, \tau_{\mathrm{last}}]$ only
    (:attr:`end_time`); beyond it ``t`` is clipped for :meth:`hazard` and
    :meth:`cumulative_hazard`, and the forward walk finds no event.

    The inversions walk the epochs, as ``forward_hazard_time`` and
    ``backward_hazard_time`` do for a piecewise-linear rate:
    an epoch whose remaining cumulative hazard is below the target is used up
    and the rest carries into the next epoch.
    Each piece inverts inside its epoch with its own
    :meth:`~Hazard.forward_time` (closed form or capped Brent).
    The scalar walk is repeated per element in the array methods.

    Parameters
    ----------
    pieces
        One hazard per epoch.
    epoch_end_times
        Epoch end times, finite, positive and strictly increasing.
    """

    def __init__(self, pieces: Sequence[Hazard], epoch_end_times: ArrayLike) -> None:
        """Validate the pieces and precompute the epoch table."""
        ends = np.atleast_1d(np.asarray(epoch_end_times, dtype=np.float64))
        check_epoch_ends("epoch_end_times", ends)
        pieces = tuple(pieces)
        if len(pieces) != ends.size:
            msg = "pieces and epoch_end_times must have the same length"
            raise ValueError(msg)
        for piece in pieces:
            if not isinstance(piece, Hazard):  # pyright: ignore[reportUnnecessaryIsInstance]
                msg = "each piece must be a Hazard"
                raise TypeError(msg)
        starts = np.concatenate(([0.0], ends[:-1]))
        lengths = ends - starts
        cum = np.array(
            [
                float(np.asarray(p.cumulative_between(0.0, ln)))
                for p, ln in zip(pieces, lengths, strict=True)
            ]
        )
        for p, ln in zip(pieces, lengths, strict=True):
            p.check_nonnegative(float(ln))
        self._pieces = pieces
        self._ends = ends
        self._starts = starts
        self._lengths = lengths
        self._cum_before = np.concatenate(([0.0], np.cumsum(cum)))
        self._py_ends = [float(v) for v in ends]
        self._py_starts = [float(v) for v in starts]
        self._py_lengths = [float(v) for v in lengths]

    @property
    def pieces(self) -> tuple[Hazard, ...]:
        """The hazard of each epoch."""
        return self._pieces

    @property
    def epoch_end_times(self) -> _Array:
        """The epoch end times."""
        return self._ends.copy()

    @property
    def has_direct_sampler(self) -> bool:  # pyright: ignore[reportIncompatibleVariableOverride]
        """Whether every piece has a direct sampler."""
        return all(p.has_direct_sampler for p in self._pieces)

    @property
    def end_time(self) -> float:
        """The end of the last epoch."""
        return self._py_ends[-1]

    @property
    def cumulative_limit(self) -> float:
        """The total cumulative hazard over ``[0, end_time]``."""
        return float(self._cum_before[-1])

    @property
    def breakpoints(self) -> tuple[float, ...]:
        """The interior epoch ends and the pieces' shifted breakpoints."""
        points = list(self._py_ends[:-1])
        for piece, start, length in zip(
            self._pieces, self._py_starts, self._py_lengths, strict=True
        ):
            points.extend(start + p for p in piece.breakpoints if 0.0 < p < length)
        return tuple(sorted(set(points)))

    @property
    def singular_points(self) -> tuple[float, ...]:
        """The pieces' singular points, shifted to absolute time."""
        points: list[float] = []
        for piece, start, length in zip(
            self._pieces, self._py_starts, self._py_lengths, strict=True
        ):
            points.extend(start + p for p in piece.singular_points if 0.0 <= p < length)
        return tuple(sorted(set(points)))

    def _locate(self, t: ArrayLike) -> tuple[_Array, NDArray[np.intp], _Array]:
        tt = np.clip(_arr(t), 0.0, self._py_ends[-1])
        idx = np.minimum(
            np.searchsorted(self._ends, tt, side="left"), self._ends.size - 1
        )
        return tt, idx, tt - self._starts[idx]

    def hazard(self, t: ArrayLike) -> _Array:
        """Hazard of the piece of the epoch of ``t`` (``t`` clipped to the window)."""
        tt, idx, u = self._locate(t)
        out = np.zeros(tt.shape)
        for k, piece in enumerate(self._pieces):
            mask = idx == k
            if np.any(mask):
                out[mask] = piece.hazard(u[mask])
        return out

    def cumulative_hazard(self, t: ArrayLike) -> _Array:
        """Cumulative hazard; constant beyond ``end_time``."""
        tt, idx, u = self._locate(t)
        out = np.zeros(tt.shape)
        for k, piece in enumerate(self._pieces):
            mask = idx == k
            if np.any(mask):
                out[mask] = self._cum_before[k] + piece.cumulative_hazard(u[mask])
        return out

    def _walk_forward(self, start: float, e: float, cap: float) -> float:
        """Return the event time, or ``cap`` if there is none before ``cap``."""
        ends = self._py_ends
        n = len(ends)
        k = bisect_right(ends, start)
        t = start
        remaining = e
        while k < n:
            if t >= cap:
                return cap
            tau = self._py_starts[k]
            length = self._py_lengths[k]
            piece = self._pieces[k]
            u0 = t - tau
            avail = float(np.asarray(piece.cumulative_between(u0, length)))
            if avail >= remaining:
                local_cap = min(length, cap - tau)
                return min(tau + piece.forward_time(u0, remaining, local_cap), cap)
            remaining -= avail
            t = ends[k]
            k += 1
        return cap

    def _walk_backward(self, start: float, e: float) -> float:
        if e == 0.0:
            return start
        k = min(bisect_left(self._py_ends, start), len(self._py_ends) - 1)
        u0 = min(start - self._py_starts[k], self._py_lengths[k])
        remaining = e
        while k >= 0:
            piece = self._pieces[k]
            avail = float(np.asarray(piece.cumulative_between(0.0, u0)))
            if avail > remaining:
                return self._py_starts[k] + piece.backward_time(u0, remaining)
            remaining -= avail
            k -= 1
            if k >= 0:
                u0 = self._py_lengths[k]
        return 0.0

    def forward_time(self, start: float, e: float, cap: float = math.inf) -> float:
        """Return ``min(start + R, cap)`` by the epoch walk (``cap`` if no event)."""
        return self._walk_forward(float(start), float(e), cap)

    def backward_time(self, start: float, e: float) -> float:
        """Return ``max(start - D, 0)`` by the epoch walk."""
        return self._walk_backward(float(start), float(e))

    def forward_offsets(self, t0: ArrayLike, e: ArrayLike) -> _Array:
        """Offsets ``R`` by the epoch walk; ``inf`` past ``end_time``."""
        a, b = np.broadcast_arrays(_arr(t0), _arr(e))
        out = np.array(
            [
                self._walk_forward(float(s), float(d), math.inf) - float(s)
                for s, d in zip(a.flat, b.flat, strict=True)
            ],
            dtype=np.float64,
        )
        return out.reshape(a.shape)

    def backward_offsets(self, x: ArrayLike, e: ArrayLike) -> _Array:
        """Offsets ``D`` in ``[0, x]`` by the epoch walk (``x`` if exhausted)."""
        a, b = np.broadcast_arrays(_arr(x), _arr(e))
        out = np.array(
            [
                float(s) - self._walk_backward(float(s), float(d))
                for s, d in zip(a.flat, b.flat, strict=True)
            ],
            dtype=np.float64,
        )
        return out.reshape(a.shape)

    def __repr__(self) -> str:
        """Return ``PiecewiseHazard(pieces=..., epoch_end_times=...)``."""
        return (
            f"PiecewiseHazard(pieces={list(self._pieces)!r}, "
            f"epoch_end_times={self._py_ends!r})"
        )


def sample_residual_lifetimes(
    hazard: Hazard,
    start_times: ArrayLike,
    *,
    random_seed: int | None = None,
    end_time: float = math.inf,
) -> _Array:
    r"""Sample the time to the next event from each start time.

    For each start time $t_0$ the returned value is the residual $R$ with
    $H(t_0 + R) - H(t_0) = E$ and $E \sim \mathrm{Exp}(1)$, so $R$ has the
    hazard $h(t_0 + \cdot)$ (exact left truncation).
    The $n$ exponentials come from one call of the generator of role
    ``"dynamics"``; the hazard object is shared, never copied.

    Parameters
    ----------
    hazard
        The hazard.
    start_times
        Start times; finite and non-negative (and at most ``end_time``).
    random_seed
        Seed of the generator; ``None`` takes entropy from the operating system.
    end_time
        End of the window; a residual that ends after it is ``inf``.
        If finite, the hazard is checked to be non-negative on ``[0, end_time]``.

    Returns
    -------
    numpy.ndarray
        The residual lifetimes, of the shape of ``start_times``;
        ``inf`` where there is no event (a defective hazard, past ``end_time``
        or past the ``end_time`` of a :class:`PiecewiseHazard`).

    Raises
    ------
    TypeError
        If ``hazard`` is not a :class:`Hazard`.
    ValueError
        If a start time is invalid, ``end_time`` is invalid, or the hazard is
        negative on ``[0, end_time]``.
    """
    if not isinstance(hazard, Hazard):  # pyright: ignore[reportUnnecessaryIsInstance]
        msg = "hazard must be a Hazard"
        raise TypeError(msg)
    t0 = _arr(start_times)
    if not np.all(np.isfinite(t0) & (t0 >= 0.0)):
        msg = "start_times must be finite and non-negative"
        raise ValueError(msg)
    if math.isnan(end_time) or end_time < 0.0:
        msg = "end_time must be non-negative"
        raise ValueError(msg)
    if np.any(t0 > end_time):
        msg = "start_times must not exceed end_time"
        raise ValueError(msg)
    if hazard.end_time is not None and np.any(t0 > hazard.end_time):
        msg = "start_times must not exceed the end_time of the hazard"
        raise ValueError(msg)
    finite = math.isfinite(end_time)
    if finite:
        hazard.check_nonnegative(end_time)
    e = root_rng(random_seed, "dynamics").exponential(size=t0.shape)
    if finite and not hazard.has_direct_sampler:
        times = np.array(
            [
                hazard.forward_time(float(s), float(d), end_time)
                for s, d in zip(t0.flat, e.flat, strict=True)
            ],
            dtype=np.float64,
        ).reshape(t0.shape)
        return np.where(times >= end_time, np.inf, times - t0)
    r = hazard.forward_offsets(t0, e)
    return np.where(t0 + r > end_time, np.inf, r)
