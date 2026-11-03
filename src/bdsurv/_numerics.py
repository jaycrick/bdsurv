"""Small scalar helpers that avoid numpy's per-call dispatch overhead."""

from __future__ import annotations

_RTOL = 1e-5
_ATOL = 1e-8


def close(a: float, b: float) -> bool:
    """Return whether ``a`` and ``b`` are close, as ``numpy.isclose`` for finite floats.

    ``numpy.isclose(a, b)`` is ``abs(a - b) <= atol + rtol * abs(b)`` with the
    default ``rtol = 1e-5`` and ``atol = 1e-8``; on Python floats this is
    several times faster, which matters in the per-step loops.
    """
    return abs(a - b) <= _ATOL + _RTOL * abs(b)
