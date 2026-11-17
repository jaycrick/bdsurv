r"""Bracketed Brent inversion of a cumulative hazard.

Both solvers are built on :func:`scipy.optimize.brentq` and take any object with
the scalar interface of :class:`_HazardLike` (every :class:`~bdsurv.hazards.Hazard`).

- :func:`forward_offset` solves $H(t_0 + r) - H(t_0) = e$ for $r \ge 0$.
- :func:`backward_offset` solves $H(x) - H(x - d) = e$ for $d \in [0, x]$.

The hazard is never evaluated beyond a finite ``cap`` (forward) or outside
``[0, x]`` (backward), so a hazard that is only valid on ``[0, T]`` stays safe.
Every failure to invert raises :class:`HazardInversionError`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Protocol, cast

import numpy as np
from scipy.optimize import brentq  # pyright: ignore[reportUnknownVariableType]

BRENT_XTOL: float = 1e-14
BRENT_RTOL: float = 4.0 * float(np.finfo(float).eps)
MAX_BRACKET_EXPANSIONS: int = 200


type _Func = Callable[[float], float]


class HazardInversionError(RuntimeError):
    """The numerical inversion of a cumulative hazard failed."""


class _RootInfo(Protocol):
    """The part of scipy's root-finding report that is used."""

    converged: bool
    flag: str


class _HazardLike(Protocol):
    """The scalar interface that the solvers need."""

    @property
    def cumulative_limit(self) -> float:
        """The limit of the cumulative hazard at infinity."""
        ...

    def hazard(self, t: float) -> object:
        """Hazard at ``t``."""
        ...

    def cumulative_hazard(self, t: float) -> object:
        """Cumulative hazard at ``t``."""
        ...

    def cumulative_between(self, s: float, e: float) -> object:
        """Cumulative hazard over ``[s, e]``."""
        ...


def _check_draw(e: float) -> None:
    if not (math.isfinite(e) and e >= 0.0):
        msg = "e must be non-negative and finite"
        raise ValueError(msg)


def _finite(value: float, what: str) -> float:
    if not math.isfinite(value):
        msg = f"{what} is not finite"
        raise HazardInversionError(msg)
    return value


def _brent(g: _Func, lo: float, hi: float, *, xtol: float, rtol: float) -> float:
    """Return the root of ``g`` in the sign-changing bracket ``[lo, hi]``."""
    if g(hi) == 0.0:
        return hi
    try:
        root, info = cast(
            "tuple[float, _RootInfo]",
            brentq(
                g,
                lo,
                hi,
                xtol=xtol,
                rtol=rtol,  # pyright: ignore[reportArgumentType]
                full_output=True,
                disp=False,
            ),
        )
    except ValueError as err:
        msg = f"root finding failed: {err}"
        raise HazardInversionError(msg) from err
    if not info.converged:
        msg = f"root finding did not converge: {info.flag}"
        raise HazardInversionError(msg)
    return max(float(root), 0.0)


def _initial_upper(hazard: _HazardLike, t0: float, e: float) -> float:
    h0 = float(np.asarray(hazard.hazard(t0)))
    if math.isfinite(h0) and h0 > 0.0:
        guess = e / h0
        if math.isfinite(guess) and guess > 0.0:
            return guess
    return max(t0, 1.0)


def _expand(g: _Func, upper: float) -> tuple[float, float]:
    """Double ``upper`` until ``g`` is non-negative, returning the bracket."""
    lower = 0.0
    for _ in range(MAX_BRACKET_EXPANSIONS):
        if g(upper) >= 0.0:
            return lower, upper
        lower = upper
        upper *= 2.0
    msg = f"no bracket found after {MAX_BRACKET_EXPANSIONS} expansions"
    raise HazardInversionError(msg)


def forward_offset(
    hazard: _HazardLike,
    t0: float,
    e: float,
    cap: float = math.inf,
    *,
    xtol: float = BRENT_XTOL,
    rtol: float = BRENT_RTOL,
) -> float:
    """Solve ``H(t0 + r) - H(t0) = e`` for ``r >= 0``.

    Parameters
    ----------
    hazard
        The hazard to invert.
    t0
        The start time, at least 0.
    e
        The target cumulative hazard, non-negative and finite.
    cap
        A finite cap beyond which the hazard is never evaluated.
    xtol, rtol
        Tolerances of the root finder.

    Returns
    -------
    float
        The offset ``r``; ``0`` if ``e == 0``; ``inf`` if the hazard is
        defective and its remaining cumulative hazard is at most ``e``;
        ``cap - t0`` if there is no event before the finite ``cap`` (callers
        read this as "no event").

    Raises
    ------
    ValueError
        If ``e`` is negative or not finite.
    HazardInversionError
        If the hazard is not finite, no bracket is found or the root finder
        fails.

    """
    _check_draw(e)
    if e == 0.0:
        return 0.0
    remaining = hazard.cumulative_limit - float(
        np.asarray(hazard.cumulative_hazard(t0))
    )
    if e >= remaining:
        return math.inf

    def g(r: float) -> float:
        value = float(np.asarray(hazard.cumulative_between(t0, t0 + r)))
        return _finite(value, "cumulative hazard") - e

    if math.isfinite(cap):
        width = max(cap - t0, 0.0)
        if width == 0.0 or g(width) < 0.0:
            return width
        return _brent(g, 0.0, width, xtol=xtol, rtol=rtol)
    lower, upper = _expand(g, _initial_upper(hazard, t0, e))
    return _brent(g, lower, upper, xtol=xtol, rtol=rtol)


def backward_offset(
    hazard: _HazardLike,
    x: float,
    e: float,
    *,
    xtol: float = BRENT_XTOL,
    rtol: float = BRENT_RTOL,
) -> float:
    """Solve ``H(x) - H(x - d) = e`` for ``d`` in ``[0, x]``.

    Parameters
    ----------
    hazard
        The hazard to invert.
    x
        The start time, at least 0.
    e
        The target cumulative hazard, non-negative and finite.
    xtol, rtol
        Tolerances of the root finder.

    Returns
    -------
    float
        The offset ``d``; ``0`` if ``e == 0``; ``x`` (so the time reaches 0)
        if the hazard over ``[0, x]`` is at most ``e``.

    Raises
    ------
    ValueError
        If ``e`` is negative or not finite.
    HazardInversionError
        If the hazard is not finite or the root finder fails.

    """
    _check_draw(e)
    if e == 0.0 or x <= 0.0:
        return 0.0
    total = _finite(float(np.asarray(hazard.cumulative_between(0.0, x))), "hazard")
    if total <= e:
        return x

    def g(d: float) -> float:
        value = float(np.asarray(hazard.cumulative_between(x - d, x)))
        return _finite(value, "cumulative hazard") - e

    return _brent(g, 0.0, x, xtol=xtol, rtol=rtol)
