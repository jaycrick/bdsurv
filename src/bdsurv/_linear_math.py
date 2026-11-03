r"""Numpy maths for birth-death processes whose rates are piecewise linear in time.

Ported from ``bdpinf`` (``rate_functions``, ``math_general`` and
``math_extinction``), with notation

$$
\Lambda(t) = \int_0^t \lambda, \quad M(t) = \int_0^t \mu, \quad
\Gamma(t) = \Lambda(t) - M(t).
$$

- Survival: $\Pr(Z_t > 0) = 1 / (1 + I(t))$ with
  $I(t) = \int_0^t \mu(s) e^{-\Gamma(s)} ds$.
- Mean cumulative number of individuals ever born by $t$:
  $1 + \int_0^t \lambda(s) e^{\Gamma(s)} ds$.
- Hazard inversion: the time $s$ at which a rate's integral from a start time
  reaches an Exp(1) draw, forward or backward, segment by segment.

Integrals are computed in log space by composite Gauss-Legendre quadrature.
Each panel is short enough that the exponent changes by at most a few units,
so integrands of very large dynamic range stay accurate.
"""

from __future__ import annotations

import math
from typing import Literal, NamedTuple

import numpy as np
from numpy.typing import NDArray

from bdsurv.linear_rates import (
    ConstantRateLike,
    LinearRateLike,
    _as_linear,  # pyright: ignore[reportPrivateUsage]
    _next_event_offset,  # pyright: ignore[reportPrivateUsage]
)

_N_NODES = 20
_PANEL_EXPONENT = 4.0  # max change of the exponent within one panel
_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(_N_NODES)

_Array = NDArray[np.float64]


class Rate(NamedTuple):
    """A piecewise-linear rate in array form: segment ``k`` is ``(starts, ends]``."""

    starts: _Array
    ends: _Array
    v0: _Array  # value at the start of each segment
    slope: _Array
    cum_before: _Array  # integral from 0 to the start of each segment

    @property
    def end_time(self) -> float:
        """The end time ``T`` of the last segment."""
        return float(self.ends[-1])


def as_rate(rate: ConstantRateLike | LinearRateLike) -> Rate:
    """Convert a rate-like object (constant or linear epochs) to a :class:`Rate`."""
    lin = _as_linear(rate)
    ends = np.asarray(lin.epoch_end_times, dtype=np.float64)
    starts = np.concatenate([[0.0], ends[:-1]])
    v0 = np.asarray(lin.start_values, dtype=np.float64)
    v1 = np.asarray(lin.end_values, dtype=np.float64)
    if not (ends.shape == v0.shape == v1.shape):
        msg = "rate values and epoch_end_times must have the same shape"
        raise ValueError(msg)
    if ends[0] <= 0.0 or np.any(np.diff(ends) <= 0.0):
        msg = "epoch_end_times must be positive and strictly increasing"
        raise ValueError(msg)
    if not (np.all(np.isfinite(v0)) and np.all(np.isfinite(v1))):
        msg = "rate values must be finite"
        raise ValueError(msg)
    if np.any(v0 < 0.0) or np.any(v1 < 0.0):
        msg = "rate values must be non-negative"
        raise ValueError(msg)
    slope = (v1 - v0) / (ends - starts)
    cum_before = np.concatenate([[0.0], np.cumsum(0.5 * (v0 + v1) * (ends - starts))])
    return Rate(starts, ends, v0, slope, cum_before[:-1])


def _index(
    rate: Rate, t: float | _Array, *, side: Literal["left", "right"] = "left"
) -> NDArray[np.intp]:
    idx = np.searchsorted(rate.ends, t, side=side)
    return np.clip(idx, 0, rate.ends.size - 1)


def value(rate: Rate, t: float | _Array) -> _Array:
    """Evaluate the rate at ``t``."""
    t = np.asarray(t, dtype=np.float64)
    k = _index(rate, t)
    return rate.v0[k] + rate.slope[k] * (t - rate.starts[k])


def cumulative(rate: Rate, t: float | _Array) -> _Array:
    """Exact integral of the rate over ``[0, t]`` (``t`` clipped to ``[0, T]``)."""
    t = np.clip(np.asarray(t, dtype=np.float64), 0.0, rate.end_time)
    k = _index(rate, t)
    d = t - rate.starts[k]
    return rate.cum_before[k] + rate.v0[k] * d + 0.5 * rate.slope[k] * d * d


def forward_hazard_time(rate: Rate, start: float, e: float, cap: float) -> float:
    """Smallest ``s >= start`` with ``int_start^s rate = e``, or ``cap`` if none <= cap.

    Within a segment the equation is a quadratic, solved by the stable form in
    :func:`bdsurv.linear_rates._next_event_offset`; if the segment's hazard is
    used up before ``e`` is reached, the remainder carries into the next one.
    """
    k = int(_index(rate, start, side="right"))
    t = start
    remaining = e
    while k < rate.ends.size:
        end = float(rate.ends[k])
        if t >= cap:
            return cap
        a = float(rate.v0[k] + rate.slope[k] * (t - rate.starts[k]))
        b = float(rate.slope[k])
        offset = _next_event_offset(a, b, remaining)
        if t + offset <= end:
            return min(t + offset, cap)
        length = end - t
        remaining -= a * length + 0.5 * b * length * length
        t = end
        k += 1
    return cap


def backward_hazard_time(rate: Rate, start: float, e: float) -> float:
    """Largest ``s <= start`` with ``int_s^start rate = e``, or ``0`` if none."""
    k = int(_index(rate, start, side="left"))
    hi = min(start, float(rate.ends[k]))
    remaining = e
    while k >= 0:
        lo = float(rate.starts[k])
        length = hi - lo
        a = float(rate.v0[k] + rate.slope[k] * length)  # value at hi
        b = -float(rate.slope[k])  # slope when walking backwards
        offset = _next_event_offset(a, b, remaining)
        if offset <= length:
            return hi - offset
        remaining -= a * length + 0.5 * b * length * length
        hi = lo
        k -= 1
    return 0.0


def _logsumexp(x: _Array) -> float:
    peak = float(np.max(x))
    if not math.isfinite(peak):
        return -math.inf
    return peak + math.log(float(np.sum(np.exp(x - peak))))


def _log_integral(  # noqa: PLR0913
    weight: Rate,
    birth: Rate,
    death: Rate,
    t: float,
    *,
    sign: float,
) -> float:
    r"""Return $\ln \int_0^t w e^{\mathrm{sign}\,\Gamma}$ by panelled quadrature."""
    if t <= 0.0:
        return -math.inf
    edges = np.unique(
        np.concatenate(
            [
                [0.0, t],
                *(r.ends[r.ends < t] for r in (weight, birth, death)),
            ]
        )
    )
    lows, highs = edges[:-1], edges[1:]

    def rate_at(x: _Array) -> _Array:
        return sign * (value(birth, x) - value(death, x))

    d_lo, d_hi = np.abs(rate_at(lows + 1e-12)), np.abs(rate_at(highs - 1e-12))
    n_panels = np.ceil(np.maximum(d_lo, d_hi) * (highs - lows) / _PANEL_EXPONENT)
    panel_lo: list[_Array] = []
    panel_hi: list[_Array] = []
    for a, b, n in zip(lows, highs, n_panels, strict=True):
        grid: _Array = np.linspace(float(a), float(b), int(max(n, 1.0)) + 1)
        panel_lo.append(grid[:-1])
        panel_hi.append(grid[1:])
    lo, hi = np.concatenate(panel_lo), np.concatenate(panel_hi)
    half = 0.5 * (hi - lo)
    x = (0.5 * (lo + hi))[:, None] + half[:, None] * _NODES
    exponent = sign * (cumulative(birth, x) - cumulative(death, x))
    with np.errstate(divide="ignore"):
        log_w = np.log(value(weight, x))
    log_f = log_w + exponent + np.log(half[:, None] * _WEIGHTS)
    return _logsumexp(log_f.ravel())


def log_survival_probability(birth: Rate, death: Rate, t: float) -> float:
    r"""Return $\ln \Pr(Z_t > 0) = -\ln(1 + I(t))$ from one individual at time 0."""
    if t <= 0.0:
        return 0.0
    log_i = _log_integral(death, birth, death, t, sign=-1.0)
    return -float(np.logaddexp(0.0, log_i))


def log_mean_cumulative_individuals(birth: Rate, death: Rate, t: float) -> float:
    r"""Return $\ln\left(1 + \int_0^t \lambda(s) e^{\Gamma(s)} ds\right)$."""
    if t <= 0.0:
        return 0.0
    return float(np.logaddexp(0.0, _log_integral(birth, birth, death, t, sign=1.0)))


def mean_cumulative_individuals(birth: Rate, death: Rate, t: float) -> float:
    """Return the expected number of individuals ever born by ``t``, root included."""
    return math.exp(log_mean_cumulative_individuals(birth, death, t))
