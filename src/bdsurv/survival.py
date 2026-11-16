"""Survival probability and mean number born of a birth-death process.

Both start from one individual.
The survival probability is $1 / (1 + I(t))$ (Kendall (1948); see
``docs/references.md``).
"""

from __future__ import annotations

import math

from bdsurv._hazard import (
    log_mean_cumulative_individuals as _log_mean_cumulative_individuals,
)
from bdsurv._hazard import log_survival_probability as _log_survival_probability
from bdsurv.rates import Rate, RateInput, resolve_rates

__all__ = ["log_mean_cumulative_individuals", "log_survival_probability"]


def _resolve(
    birth_rate: RateInput,
    death_rate: RateInput,
    t: float,
    terminal_time: float | None,
) -> tuple[Rate, Rate]:
    """Validate ``t`` and the rates and return the resolved rate objects."""
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
    return birth, death


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

    This is the survival probability of Kendall (1948); see
    ``docs/references.md``.
    Here $Z_t$ is the number of individuals alive at time $t$ and the formula is
    stated in ``docs/splitting.md``, section "Linear (time-varying) rates".

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
    birth, death = _resolve(birth_rate, death_rate, t, terminal_time)
    return _log_survival_probability(birth, death, t)


def log_mean_cumulative_individuals(
    birth_rate: RateInput,
    death_rate: RateInput,
    t: float,
    terminal_time: float | None = None,
) -> float:
    r"""Return the log of the expected number of individuals ever born by ``t``.

    The process starts from one individual at time 0, with per-individual birth
    rate $\lambda$ and death rate $\mu$, and is not conditioned on survival.
    The count includes the root individual, and the expected value is

    $$
    1 + \int_0^t \lambda(s) e^{\Gamma(s)} ds, \quad
    \Gamma(s) = \int_0^s (\lambda - \mu).
    $$

    It is derived in ``docs/splitting.md``, section "Closed form and the mean
    number born".
    The integral is evaluated by panelled Gauss-Legendre quadrature in log space,
    so the result is stable for large ``t``.
    The expected number itself is ``math.exp(result)``.

    Parameters
    ----------
    birth_rate, death_rate
        Per-individual rates, each a float (constant on ``[0, T]``) or a
        rate-like (:class:`~bdsurv.rates.PiecewiseConstant`,
        :class:`~bdsurv.rates.PiecewiseLinear` or any object with the same
        attributes), finite and non-negative (zero is allowed).
        Rate-likes must have a common end time ``T``.
    t
        Time up to which births are counted, with ``0 <= t <= T``.
    terminal_time
        The horizon ``T``.
        When both rates are floats it defaults to ``t``; otherwise it must be
        ``None`` or equal to the end time of the rate-likes.

    Returns
    -------
    float
        The log of the expected number of individuals ever born by ``t``, at
        least 0 (it is exactly 0 at ``t = 0`` or when the birth rate is zero).

    Raises
    ------
    ValueError
        If a rate is invalid, the rate-likes differ in end time, ``terminal_time``
        is invalid or does not match the rates, or ``t`` is negative, not finite
        or beyond ``T``.

    """
    birth, death = _resolve(birth_rate, death_rate, t, terminal_time)
    return _log_mean_cumulative_individuals(birth, death, t)
