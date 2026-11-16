r"""Exact event simulation for birth-death processes with linear rates.

Within a segment where both rates are linear in time,
the total hazard with $n$ individuals alive at $t_0$ is $a + b s$,
where $s = t - t_0$, $a = n(\lambda(t_0) + \mu(t_0))$ and $b = n(\lambda' + \mu')$.
The cumulative hazard is $H(s) = a s + b s^2 / 2$,
so the next event time inverts $H(s) = E$ with $E = -\ln(1 - u)$:

$$
s = \frac{-a + \sqrt{a^2 + 2 b E}}{b}
  = \frac{2E}{a + \sqrt{a^2 + 2 b E}}.
$$

The second form is used in code;
it is stable as $b \to 0$ and equals $E / a$ at $b = 0$.
An event that falls past the end of the segment is discarded
and redrawn from the boundary,
which is exact because the event process has independent increments.
The event is a birth with probability $\lambda(t) / (\lambda(t) + \mu(t))$.
"""

from __future__ import annotations

import numpy as np

from bdsurv._forward import simulate_forward
from bdsurv.rates import (
    ConstantRateLike,
    LinearRateLike,
    PiecewiseConstant,
    PiecewiseLinear,
    resolve_rates,
)
from bdsurv.tree import SimulatedProcess

# The rate types now live in ``bdsurv.rates``; they are re-exported here until
# the module is removed.
__all__ = [
    "ConstantRateLike",
    "LinearRateLike",
    "PiecewiseConstant",
    "PiecewiseLinear",
    "simulate_birth_death_linear",
]


def simulate_birth_death_linear(
    birth_rate: ConstantRateLike | LinearRateLike,
    death_rate: ConstantRateLike | LinearRateLike,
    random_seed: int | None = None,
    *,
    ensure_non_extinct: bool = False,
    max_retries: int = 100,
) -> SimulatedProcess:
    r"""Simulate a birth-death process whose rates are linear in time.

    Each event time is drawn by inverting the cumulative hazard of the
    superposed birth and death processes,
    $s = 2E / (a + \sqrt{a^2 + 2bE})$ with $E \sim \mathrm{Exp}(1)$,
    where $a$ and $b$ are the total hazard and its slope, both scaled by the
    number alive.
    The event is a birth with probability $\lambda(t) / (\lambda(t) + \mu(t))$.
    Genealogy follows :func:`bdsurv.gillespie.simulate_birth_death_process`.

    Parameters
    ----------
    birth_rate, death_rate
        Per-individual rates over ``[0, T]`` with a common end time $T$.
        Their epoch boundaries need not coincide.
    random_seed
        Seed for reproducibility.
    ensure_non_extinct
        Retry until at least one individual survives to $T$.
    max_retries
        Maximum retries when ``ensure_non_extinct`` is true.

    Returns
    -------
    SimulatedProcess
        Simulated process with records, genealogy and spine.
        ``birth_rates`` and ``death_rates`` are ``None``.
        Its stored epochs are the merged segment boundaries.

    Raises
    ------
    ValueError
        If a rate has negative or non-finite values or invalid epoch end times,
        the end times differ, or ``max_retries`` is negative.
    RuntimeError
        If every retry went extinct.

    """
    birth, death, _ = resolve_rates(birth_rate, death_rate, None)
    records, segments = simulate_forward(
        birth,
        death,
        random_seed,
        ensure_non_extinct=ensure_non_extinct,
        max_retries=max_retries,
        name="simulate_birth_death_linear",
    )
    return SimulatedProcess(
        records=records,
        epoch_end_times=np.array([seg[1] for seg in segments]),
    )
