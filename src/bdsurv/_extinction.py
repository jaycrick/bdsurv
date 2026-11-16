"""Single-epoch extinction probability, kept for compatibility.

``pr_extinction`` is kept because ``bdpinf`` imports it from here; it is
removed in 0.5.0.
Use :func:`bdsurv._hazard.log_survival_probability_constant` instead.
"""

from __future__ import annotations

import math

from bdsurv._hazard import log_survival_probability_constant


def pr_extinction(birth_rate: float, death_rate: float, terminal_time: float) -> float:
    """Return ``P(Z_T = 0)`` for a single-epoch linear birth-death process.

    Kept for compatibility; removed in 0.5.0.
    """
    return -math.expm1(
        log_survival_probability_constant(
            birth_rate=birth_rate, death_rate=death_rate, time=terminal_time
        )
    )
