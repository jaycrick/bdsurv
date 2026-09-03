"""Single-epoch extinction probability for a linear birth-death process.

This is the closed-form special case (one epoch, scalar rates) of the
general multi-epoch extinction probability used elsewhere in the birth-death
process literature. Both simulators in this package only ever need the
single-epoch value, so this module reimplements just that case rather than
carrying a dependency on a JAX-based general solver.
"""

from __future__ import annotations

import math

_ATOL_CRITICAL: float = 1e-10


def pr_extinction(birth_rate: float, death_rate: float, terminal_time: float) -> float:
    """Return ``P(Z_T = 0)`` for a single-epoch linear birth-death process.

    Parameters
    ----------
    birth_rate : float
        Per-individual birth rate ``lambda``.
    death_rate : float
        Per-individual death rate ``mu``.
    terminal_time : float
        Time horizon ``T``.

    Returns
    -------
    float
        The extinction probability, in ``[0, 1)``.

    Notes
    -----
    For a population started from a single individual at time 0::

        x = (lambda - mu) * T
        kernel = T                          if |x| < 1e-10   (critical: lambda == mu)
        kernel = (1 - exp(-x)) / (lambda - mu)   otherwise
        num = mu * kernel
        P(Z_T = 0) = num / (1 + num)

    which is the standard closed form for the extinction probability of a
    linear birth-death process (see e.g. Kendall 1948).

    """
    growth_rate = birth_rate - death_rate
    x = growth_rate * terminal_time
    if abs(x) < _ATOL_CRITICAL:
        kernel = terminal_time
    else:
        kernel = (1.0 - math.exp(-x)) / growth_rate
    numerator = death_rate * kernel
    return numerator / (1.0 + numerator)
