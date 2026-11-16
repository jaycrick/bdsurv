"""Survival probability of a birth-death process started from one individual."""

from __future__ import annotations

import math

from bdsurv._hazard import log_survival_probability as _log_survival_probability
from bdsurv.rates import RateInput, resolve_rates

__all__ = ["log_survival_probability"]


def log_survival_probability(
    birth_rate: RateInput,
    death_rate: RateInput,
    t: float,
    terminal_time: float | None = None,
) -> float:
    r"""Return the log of the probability that the process is not extinct at ``t``.

    The process starts from one individual at time 0, with per-individual birth
    rate $\lambda$ and death rate $\mu$.
    Its probability of survival to time $t$ is

    $$
    \Pr(Z_t > 0) = \frac{1}{1 + I(t)}, \quad
    I(t) = \int_0^t \mu(s) e^{-\Gamma(s)} ds, \quad
    \Gamma(s) = \int_0^s (\lambda - \mu).
    $$

    For constant rates the integral is closed-form and the result is stable for
    large ``t``; otherwise it is evaluated by panelled Gauss-Legendre quadrature
    in log space.
    The probability of extinction by ``t`` is ``-math.expm1(result)``.

    Parameters
    ----------
    birth_rate, death_rate
        Per-individual rates, each a float (constant on ``[0, T]``) or a
        rate-like (:class:`~bdsurv.rates.PiecewiseConstant`,
        :class:`~bdsurv.rates.PiecewiseLinear` or any object with the same
        attributes), finite and non-negative (zero is allowed).
        Rate-likes must have a common end time ``T``.
    t
        Time at which survival is evaluated, with ``0 <= t <= T``.
    terminal_time
        The horizon ``T``.
        When both rates are floats it defaults to ``t``; otherwise it must be
        ``None`` or equal to the end time of the rate-likes.

    Returns
    -------
    float
        ``log Pr(Z_t > 0)``, at most 0.

    Raises
    ------
    ValueError
        If a rate is invalid, the rate-likes differ in end time, ``terminal_time``
        is invalid or does not match the rates, or ``t`` is negative, not finite
        or beyond ``T``.

    """
    if not (math.isfinite(t) and t >= 0.0):
        msg = f"t must be finite and non-negative, got {t}"
        raise ValueError(msg)
    if terminal_time is None and all(
        isinstance(rate, int | float) for rate in (birth_rate, death_rate)
    ):
        # With no horizon given, t = 0 only needs the rates to be valid.
        terminal_time = t if t > 0.0 else 1.0
    birth, death, horizon = resolve_rates(birth_rate, death_rate, terminal_time)
    if t > horizon:
        msg = f"t must lie in [0, T] with T = {horizon}, got {t}"
        raise ValueError(msg)
    return _log_survival_probability(birth, death, t)
