r"""Numpy maths for birth-death processes whose rates are piecewise linear in time.

Ported from ``bdpinf`` (``rate_functions``, ``math_general`` and
``math_extinction``), with notation

$$
\Lambda(t) = \int_0^t \lambda, \quad M(t) = \int_0^t \mu, \quad
\Gamma(t) = \Lambda(t) - M(t).
$$

- Survival: $\Pr(Z_t > 0) = 1 / (1 + I(t))$ with
  $I(t) = \int_0^t \mu(s) e^{-\Gamma(s)} ds$ (Kendall (1948); see
  ``docs/references.md``).
  The formula is stated in ``docs/splitting.md``, section "Linear (time-varying)
  rates".
- Mean cumulative number of individuals ever born by $t$:
  $1 + \int_0^t \lambda(s) e^{\Gamma(s)} ds$
  (``docs/splitting.md``, section "Closed form and the mean number born").
- Hazard inversion: the time $s$ at which a rate's integral from a start time
  reaches an Exp(1) draw, forward or backward, segment by segment
  (``docs/contour_process.md``, section "Linear (time-varying) rates").

Integrals are computed in log space by composite Gauss-Legendre quadrature.
Each panel is short enough that the exponent changes by at most a few units,
so integrands of very large dynamic range stay accurate.

The quadrature is written here rather than delegated to a library integrator
because the integrand $\mu(s) e^{-\Gamma(s)}$ can span a very large dynamic
range.
The integral is therefore accumulated in log space, over panels short enough
that the exponent changes by a bounded amount (``_PANEL_EXPONENT``).
The tests cross-check this code against ``scipy.integrate.quad``.
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from bdsurv.rates import Rate, ResolvedRate

_N_NODES = 20
_PANEL_EXPONENT = 4.0  # max change of the exponent within one panel
_N_GRADED = 40  # geometric edges to the right of a singular point
# Rates are evaluated this far inside a panel's edges, where a rate may jump.
PANEL_EDGE_OFFSET: float = 1e-12
# ``|lambda - mu| * t`` below this is the critical case ``lambda == mu``.
CRITICAL_TOLERANCE: float = 1e-10
# Tolerances of ``close``.
CLOSE_REL_TOL: float = 1e-5
CLOSE_ABS_TOL: float = 1e-8
_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(_N_NODES)

_Array = NDArray[np.float64]


def close(a: float, b: float) -> bool:
    """Return whether ``a`` and ``b`` are close (relative 1e-5, absolute 1e-8)."""
    return math.isclose(a, b, rel_tol=CLOSE_REL_TOL, abs_tol=CLOSE_ABS_TOL)


def next_event_offset(a: float, b: float, e: float) -> float:
    """Solve ``a s + b s^2 / 2 = e`` for the smallest ``s >= 0``, or ``inf``.

    Here ``a`` and ``b`` are the value and slope of a rate that is linear in time,
    and ``e`` is an Exp(1) draw (``docs/contour_process.md``, section "Linear
    (time-varying) rates").
    The stable root is ``s = 2 e / (a + sqrt(a^2 + 2 b e))``.
    """
    disc = a * a + 2.0 * b * e
    if disc < 0.0:
        return math.inf
    denom = a + math.sqrt(disc)
    if denom <= 0.0:
        return math.inf
    return 2.0 * e / denom


def _index(
    rate: Rate, t: float | _Array, *, side: Literal["left", "right"] = "left"
) -> NDArray[np.intp]:
    idx = np.searchsorted(rate.ends, t, side=side)
    return np.clip(idx, 0, rate.ends.size - 1)


def value(rate: ResolvedRate, t: float | _Array) -> _Array:
    """Evaluate the rate at ``t``."""
    if not isinstance(rate, Rate):
        return rate.hazard(t)
    t = np.asarray(t, dtype=np.float64)
    k = _index(rate, t)
    return rate.v0[k] + rate.slope[k] * (t - rate.starts[k])


def cumulative(rate: ResolvedRate, t: float | _Array) -> _Array:
    """Exact integral of the rate over ``[0, t]`` (``t`` clipped to ``[0, T]``).

    A :class:`~bdsurv.hazards.Hazard` is evaluated as given, without clipping.
    """
    if not isinstance(rate, Rate):
        return rate.cumulative_hazard(t)
    # Lambda(t) or M(t) of docs/contour_conditioned.md, section "Overview": the
    # integral up to the segment start plus ``v0 d + slope d^2 / 2`` within it.
    t = np.clip(np.asarray(t, dtype=np.float64), 0.0, rate.end_time)
    k = _index(rate, t)
    d = t - rate.starts[k]
    return rate.cum_before[k] + rate.v0[k] * d + 0.5 * rate.slope[k] * d * d


def forward_hazard_time(
    rate: ResolvedRate, start: float, e: float, cap: float
) -> float:
    """Smallest ``s >= start`` with ``int_start^s rate = e``, or ``cap`` if none <= cap.

    Within a segment the equation is a quadratic, solved by the stable form in
    :func:`next_event_offset`; if the segment's hazard is
    used up before ``e`` is reached, the remainder carries into the next one
    (``docs/contour_process.md``, section "Linear (time-varying) rates").
    Plain Python floats and ``bisect`` are used: this runs once per contour step.
    A :class:`~bdsurv.hazards.Hazard` inverts its own cumulative hazard.
    """
    if not isinstance(rate, Rate):
        return rate.forward_time(start, e, cap)
    ends, starts, v0s, slopes = rate.py_ends, rate.py_starts, rate.py_v0, rate.py_slope
    n = len(ends)
    if n == 1 and slopes[0] == 0.0:  # constant rate: the jump is exponential
        v = v0s[0]
        return min(start + e / v, cap) if v > 0.0 else cap
    k = min(bisect_right(ends, start), n - 1)
    t = start
    remaining = e
    while k < n:
        end = ends[k]
        if t >= cap:
            return cap
        a = v0s[k] + slopes[k] * (t - starts[k])
        b = slopes[k]
        offset = next_event_offset(a, b, remaining)
        if t + offset <= end:
            return min(t + offset, cap)
        length = end - t
        remaining -= a * length + 0.5 * b * length * length
        t = end
        k += 1
    return cap


def backward_hazard_time(rate: ResolvedRate, start: float, e: float) -> float:
    """Largest ``s <= start`` with ``int_s^start rate = e``, or ``0`` if none.

    The mirror of :func:`forward_hazard_time`, walking the segments backwards
    (``docs/contour_process.md``, section "Linear (time-varying) rates").
    A :class:`~bdsurv.hazards.Hazard` inverts its own cumulative hazard.
    """
    if not isinstance(rate, Rate):
        return rate.backward_time(start, e)
    ends, starts, v0s, slopes = rate.py_ends, rate.py_starts, rate.py_v0, rate.py_slope
    if len(ends) == 1 and slopes[0] == 0.0:  # constant rate: the jump is exponential
        v = v0s[0]
        return max(start - e / v, 0.0) if v > 0.0 else 0.0
    k = min(bisect_left(ends, start), len(ends) - 1)
    hi = min(start, ends[k])
    remaining = e
    while k >= 0:
        lo = starts[k]
        length = hi - lo
        a = v0s[k] + slopes[k] * length  # value at hi
        b = -slopes[k]  # slope when walking backwards
        offset = next_event_offset(a, b, remaining)
        if offset <= length:
            return hi - offset
        remaining -= a * length + 0.5 * b * length * length
        hi = lo
        k -= 1
    return 0.0


def _edges_of(rate: ResolvedRate, t: float) -> list[float]:
    """Breakpoints of ``rate`` strictly inside ``(0, t)``."""
    points = rate.py_ends[:-1] if isinstance(rate, Rate) else rate.breakpoints
    return [p for p in points if 0.0 < p < t]


def _singular_of(rate: ResolvedRate) -> tuple[float, ...]:
    """Points where ``rate`` is infinite (none for a :class:`Rate`)."""
    return () if isinstance(rate, Rate) else rate.singular_points


def _between(rate: ResolvedRate, lo: _Array, hi: _Array) -> _Array:
    """Integral of ``rate`` over ``[lo, hi]``, stable for a hazard."""
    if isinstance(rate, Rate):
        return cumulative(rate, hi) - cumulative(rate, lo)
    return np.asarray(rate.cumulative_between(lo, hi), dtype=np.float64)


def _graded_edges(edges: list[float], singular: set[float]) -> list[float]:
    """Add geometric edges ``p + w 2^-j`` to the right of each singular point."""
    out = set(edges)
    for p in singular:
        if p < edges[-1]:
            right = min(e for e in edges if e > p)
            out.update(p + (right - p) * 2.0**-j for j in range(1, _N_GRADED + 1))
    return sorted(out)


def _split_panels(
    lows: _Array, highs: _Array, n_panels: _Array
) -> tuple[_Array, _Array]:
    """Split each edge interval into its panels; return the panel ends ``(lo, hi)``."""
    panel_lo: list[_Array] = []
    panel_hi: list[_Array] = []
    for a, b, n in zip(lows, highs, n_panels, strict=True):
        sub: _Array = np.linspace(float(a), float(b), int(max(n, 1.0)) + 1)
        panel_lo.append(sub[:-1])
        panel_hi.append(sub[1:])
    return np.concatenate(panel_lo), np.concatenate(panel_hi)


def _hazard_panels(
    birth: ResolvedRate,
    death: ResolvedRate,
    edges: set[float],
    singular: set[float],
) -> tuple[_Array, _Array]:
    """Panels of the generic (hazard) path for the edge set ``edges``.

    The edges are graded geometrically to the right of each singular point; each
    edge interval is split into ``ceil((dLambda + dM) / _PANEL_EXPONENT)`` panels,
    which bounds the change of the exponent within a panel even when
    ``lambda - mu`` changes sign.
    """
    grid = np.asarray(_graded_edges(sorted(edges), singular), dtype=np.float64)
    lows, highs = grid[:-1], grid[1:]
    n_panels = np.ceil(
        (_between(birth, lows, highs) + _between(death, lows, highs)) / _PANEL_EXPONENT
    )
    return _split_panels(lows, highs, n_panels)


def _rate_panels(
    birth: Rate, death: Rate, edges: _Array, *, sign: float
) -> tuple[_Array, _Array]:
    """Panels of the :class:`Rate` path for the sorted unique ``edges``.

    Each edge interval gets ``ceil(max |sign (lambda - mu)| at its ends * width /
    _PANEL_EXPONENT)`` panels, the rates being read ``PANEL_EDGE_OFFSET`` inside.
    """
    lows, highs = edges[:-1], edges[1:]

    def rate_at(x: _Array) -> _Array:
        return sign * (value(birth, x) - value(death, x))

    d_lo, d_hi = (
        np.abs(rate_at(lows + PANEL_EDGE_OFFSET)),
        np.abs(rate_at(highs - PANEL_EDGE_OFFSET)),
    )
    n_panels = np.ceil(np.maximum(d_lo, d_hi) * (highs - lows) / _PANEL_EXPONENT)
    return _split_panels(lows, highs, n_panels)


def _panel_log_f(
    weight: ResolvedRate,
    birth: ResolvedRate,
    death: ResolvedRate,
    lo: _Array,
    hi: _Array,
    *,
    sign: float,
) -> _Array:
    """Log of the weighted quadrature terms, shape ``(n_panels, n_nodes)``."""
    half = 0.5 * (hi - lo)
    x = (0.5 * (lo + hi))[:, None] + half[:, None] * _NODES
    exponent = sign * (cumulative(birth, x) - cumulative(death, x))
    with np.errstate(divide="ignore"):
        log_w = np.log(value(weight, x))
    return log_w + exponent + np.log(half[:, None] * _WEIGHTS)


def _panels(
    weight: ResolvedRate,
    birth: ResolvedRate,
    death: ResolvedRate,
    horizon: float,
    points: list[float],
    *,
    sign: float,
) -> tuple[_Array, _Array]:
    """Panel ends covering ``[0, horizon]`` with ``points`` (and ``0``) as edges."""
    rates = (weight, birth, death)
    if all(isinstance(r, Rate) for r in rates):
        edges = np.unique(
            np.concatenate(
                [
                    [0.0, *points],
                    *(r.ends[r.ends < horizon] for r in rates if isinstance(r, Rate)),
                ]
            )
        )
        assert isinstance(birth, Rate)
        assert isinstance(death, Rate)
        return _rate_panels(birth, death, edges, sign=sign)
    edge_set = {0.0, *points}.union(*(_edges_of(r, horizon) for r in rates))
    singular = {p for r in rates for p in _singular_of(r) if 0.0 <= p < horizon}
    return _hazard_panels(birth, death, edge_set, singular)


def log_integral(
    weight: ResolvedRate,
    birth: ResolvedRate,
    death: ResolvedRate,
    t: float,
    *,
    sign: float,
) -> float:
    r"""Return $\ln \int_0^t w e^{\mathrm{sign}\,\Gamma}$ by panelled quadrature.

    Here $w$ is the rate ``weight``, $\Gamma = \Lambda - M$ is the integral of
    ``birth`` minus that of ``death``, and ``sign`` is $\pm 1$.
    With ``weight=death`` and ``sign=-1`` this is $\ln I(t)$ for
    $I(t) = \int_0^t \mu e^{-\Gamma}$; with ``weight=birth`` and ``sign=+1`` it is
    the integral in the mean cumulative number born
    (``docs/splitting.md``, sections "Linear (time-varying) rates" and "Closed
    form and the mean number born").
    The panels are described in the module docstring.
    If any of the three is a :class:`~bdsurv.hazards.Hazard`, the edges are graded
    towards singular points (:func:`_hazard_panels`) instead.
    """
    if t <= 0.0:
        return -math.inf
    lo, hi = _panels(weight, birth, death, t, [t], sign=sign)
    log_f = _panel_log_f(weight, birth, death, lo, hi, sign=sign)
    return float(np.logaddexp.reduce(log_f.ravel()))


def log_integral_curve(
    weight: ResolvedRate,
    birth: ResolvedRate,
    death: ResolvedRate,
    times: _Array,
    *,
    sign: float,
) -> _Array:
    r"""Return $\ln \int_0^t w e^{\mathrm{sign}\,\Gamma}$ at every ``t`` in ``times``.

    ``times`` is a sorted, unique, one-dimensional array.
    The panels cover ``[0, max(times)]`` once, with every positive time an edge;
    the running log-sum over panels is read at the panel ending at each time, so
    the cost is that of a single :func:`log_integral` at ``max(times)`` and each
    value agrees with it to rounding.
    The value is ``-inf`` for ``t <= 0``.
    """
    out = np.full(times.shape, -np.inf)
    positive = times > 0.0
    if not positive.any():
        return out
    t_pos = times[positive]
    lo, hi = _panels(
        weight, birth, death, float(t_pos[-1]), [float(x) for x in t_pos], sign=sign
    )
    log_f = _panel_log_f(weight, birth, death, lo, hi, sign=sign)
    running = np.logaddexp.accumulate(np.logaddexp.reduce(log_f, axis=1))
    out[positive] = running[np.searchsorted(hi, t_pos)]
    return out


def log_survival_probability_constant(
    *,
    birth_rate: float,
    death_rate: float,
    time: float,
) -> float:
    """Return ``log Pr(Z_t > 0)`` from one individual, stable for large ``t``.

    With constant rates ``lam = birth_rate`` and ``mu = death_rate`` and
    ``g = lam - mu``, the integral ``I(t)`` of ``mu * exp(-Gamma(s))`` over
    ``[0, t]`` is ``mu (1 - exp(-g t)) / g``, or ``mu t`` when ``g t`` is
    negligible (the critical case ``lam == mu``).
    The survival probability is ``1 / (1 + I(t))`` (Kendall (1948); see
    ``docs/references.md`` and ``docs/splitting.md``, section "Closed form and
    the mean number born").
    """
    if time <= 0.0:
        return 0.0
    growth_rate = birth_rate - death_rate
    x = growth_rate * time
    # log_num is log I(t) with I(t) = mu (1 - exp(-g t)) / g, g = lam - mu
    # (docs/splitting.md, section "Closed form and the mean number born"), so
    # that survival is 1 / (1 + I(t)).
    if abs(x) < CRITICAL_TOLERANCE:
        log_num = math.log(death_rate) + math.log(time)
    elif x < 0.0:
        # kernel = expm1(|x|) / |g|, with log(expm1(y)) = y + log(1 - exp(-y))
        log_num = (
            math.log(death_rate)
            + (-x)
            + math.log(-math.expm1(x))
            - math.log(-growth_rate)
        )
    else:
        log_num = (
            math.log(death_rate) + math.log(-math.expm1(-x)) - math.log(growth_rate)
        )
    return -float(np.logaddexp(0.0, log_num))


def log_survival_probability_quadrature(
    birth: ResolvedRate, death: ResolvedRate, t: float
) -> float:
    r"""Return $\ln \Pr(Z_t > 0) = -\ln(1 + I(t))$ by panelled quadrature.

    The survival probability is $1 / (1 + I(t))$ with
    $I(t) = \int_0^t \mu(s) e^{-\Gamma(s)} ds$ and
    $\Gamma(s) = \int_0^s (\lambda - \mu)$; the survival probability is that of
    Kendall (1948), see ``docs/references.md`` and ``docs/splitting.md``,
    section "Linear (time-varying) rates".
    """
    if t <= 0.0:
        return 0.0
    log_i = log_integral(death, birth, death, t, sign=-1.0)
    return -float(np.logaddexp(0.0, log_i))


def _is_constant(rate: ResolvedRate, *, positive: bool) -> bool:
    """Whether ``rate`` is one epoch with zero slope (and a positive value if asked)."""
    if not isinstance(rate, Rate):
        return False
    return (
        rate.ends.size == 1
        and rate.slope[0] == 0.0
        and (not positive or rate.v0[0] > 0.0)
    )


def log_survival_probability(
    birth: ResolvedRate, death: ResolvedRate, t: float
) -> float:
    r"""Return $\ln \Pr(Z_t > 0) = -\ln(1 + I(t))$ from one individual at time 0.

    When both rates are constant (one epoch, zero slope) and the death rate is
    positive this is the closed form
    :func:`log_survival_probability_constant`; otherwise it is the quadrature
    :func:`log_survival_probability_quadrature`.
    Both evaluate $\Pr(Z_t > 0) = 1 / (1 + I(t))$ (Kendall (1948); see
    ``docs/references.md``).
    """
    if (
        isinstance(birth, Rate)
        and isinstance(death, Rate)
        and _is_constant(birth, positive=False)
        and _is_constant(death, positive=True)
    ):
        return log_survival_probability_constant(
            birth_rate=float(birth.v0[0]), death_rate=float(death.v0[0]), time=t
        )
    return log_survival_probability_quadrature(birth, death, t)


def log_mean_cumulative_individuals(
    birth: ResolvedRate, death: ResolvedRate, t: float
) -> float:
    r"""Return $\ln\left(1 + \int_0^t \lambda(s) e^{\Gamma(s)} ds\right)$.

    Here $\lambda$ is the birth rate and $\Gamma(s) = \int_0^s (\lambda - \mu)$
    (``docs/splitting.md``, section "Closed form and the mean number born").
    """
    if t <= 0.0:
        return 0.0
    return float(np.logaddexp(0.0, log_integral(birth, birth, death, t, sign=1.0)))


def log_mean_cumulative_individuals_curve(
    birth: ResolvedRate, death: ResolvedRate, times: _Array
) -> _Array:
    r"""Return $\ln\left(1 + \int_0^t \lambda e^{\Gamma}\right)$ at each ``t``.

    Equal to :func:`log_mean_cumulative_individuals` at each time (and exactly 0
    at ``t = 0``), from one panelled pass (:func:`log_integral_curve`).
    """
    return np.logaddexp(0.0, log_integral_curve(birth, birth, death, times, sign=1.0))


def mean_cumulative_individuals(
    birth: ResolvedRate, death: ResolvedRate, t: float
) -> float:
    """Return the expected number of individuals ever born by ``t``, root included."""
    return math.exp(log_mean_cumulative_individuals(birth, death, t))
