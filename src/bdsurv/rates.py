r"""Rates of birth-death processes: their representation and their validation.

A rate is a function of time on ``[0, T]``, piecewise constant or piecewise
linear over epochs.
Callers pass either a float or a *rate-like* object (:class:`ConstantRateLike`
or :class:`LinearRateLike`), or, for constant-rate simulators, one value per
epoch as arrays.
Every one of these enters through this module, which is the only place rates
(and their epoch end times) are validated:

- :func:`as_rate` converts a rate-like to the array form :class:`Rate`.
- :func:`resolve_rates` converts the two rates of a simulator and fixes ``T``.
- :func:`epoch_rates` validates the per-epoch arrays of the constant-rate
  simulators, through the same path as a rate-like.
- :func:`check_positive_rates` and :func:`check_terminal_time` are the checks of
  the constant-rate entry points, which require strictly positive rates.
  Rate-likes may be zero.
- :func:`merged_segments` splits ``[0, T]`` at the epoch ends of both rates.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, Protocol, cast

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import jax

# Edges of merged segments closer than this fraction of ``T`` are one edge.
MERGE_REL_TOL: float = 1e-12

type _Array = NDArray[np.float64]
type _Floats = Sequence[float] | NDArray[np.float64] | jax.Array


class ConstantRateLike(Protocol):
    """A piecewise-constant rate: ``values[k]`` on the k-th epoch."""

    @property
    def values(self) -> _Floats:
        """Rate in each epoch."""
        ...

    @property
    def epoch_end_times(self) -> _Floats:
        """Strictly increasing epoch end times."""
        ...


class LinearRateLike(Protocol):
    """A piecewise-linear rate, from ``start_values[k]`` to ``end_values[k]``."""

    @property
    def start_values(self) -> _Floats:
        """Rate at the start of each epoch."""
        ...

    @property
    def end_values(self) -> _Floats:
        """Rate at the end of each epoch."""
        ...

    @property
    def epoch_end_times(self) -> _Floats:
        """Strictly increasing epoch end times."""
        ...


@dataclass(frozen=True)
class PiecewiseConstant:
    """Numpy implementation of :class:`ConstantRateLike`."""

    values: _Floats
    epoch_end_times: _Floats


@dataclass(frozen=True)
class PiecewiseLinear:
    """Numpy implementation of :class:`LinearRateLike`."""

    start_values: _Floats
    end_values: _Floats
    epoch_end_times: _Floats


type RateInput = float | ConstantRateLike | LinearRateLike


class Rate(NamedTuple):
    """A piecewise-linear rate in array form: segment ``k`` is ``(starts, ends]``."""

    starts: _Array
    ends: _Array
    v0: _Array  # value at the start of each segment
    slope: _Array
    cum_before: _Array  # integral from 0 to the start of each segment
    # Float tuples mirroring the arrays, for the scalar hazard-inversion loops.
    py_starts: tuple[float, ...] = ()
    py_ends: tuple[float, ...] = ()
    py_v0: tuple[float, ...] = ()
    py_slope: tuple[float, ...] = ()

    @property
    def end_time(self) -> float:
        """The end time ``T`` of the last segment."""
        return float(self.ends[-1])


def check_strictly_increasing(name: str, values: _Array) -> None:
    """Raise ``ValueError`` unless ``values`` is strictly increasing."""
    if np.any(np.diff(values) <= 0.0):
        msg = f"{name} must be strictly increasing"
        raise ValueError(msg)


def check_epoch_ends(name: str, ends: _Array) -> None:
    """Raise ``ValueError`` unless epoch end times are finite, positive, increasing."""
    if ends.size == 0:
        msg = f"{name} must not be empty"
        raise ValueError(msg)
    if not np.all(np.isfinite(ends)):
        msg = f"{name} must be finite and strictly increasing"
        raise ValueError(msg)
    if ends[0] <= 0.0:
        msg = f"{name} must be positive"
        raise ValueError(msg)
    check_strictly_increasing(name, ends)


def check_positive_rates(
    birth_rate: float | _Array,
    death_rate: float | _Array,
    *,
    label: str = "birth_rate and death_rate",
) -> None:
    """Raise ``ValueError`` unless every birth and death rate is finite and positive.

    This is the check of the constant-rate simulators, which need strictly
    positive rates; rate-likes (see :func:`as_rate`) may be zero.
    """
    for rate in (birth_rate, death_rate):
        values = np.asarray(rate, dtype=np.float64)
        if not np.all(np.isfinite(values) & (values > 0.0)):
            msg = f"{label} must be positive and finite"
            raise ValueError(msg)


def check_terminal_time(terminal_time: float, name: str = "terminal_time") -> None:
    """Raise ``ValueError`` unless a terminal time is finite and positive."""
    if not (math.isfinite(terminal_time) and terminal_time > 0.0):
        msg = f"{name} must be positive and finite"
        raise ValueError(msg)


def as_rate(
    rate: ConstantRateLike | LinearRateLike, *, ends_name: str = "epoch_end_times"
) -> Rate:
    """Validate a rate-like object and convert it to a :class:`Rate`.

    The values and epoch end times must have the same shape, the end times must
    be finite, positive and strictly increasing, and the values finite and
    non-negative (zero is allowed).
    ``ends_name`` is how error messages name the epoch end times.
    """
    ends = np.atleast_1d(np.asarray(rate.epoch_end_times, dtype=np.float64))
    if hasattr(rate, "values"):
        v0 = v1 = np.atleast_1d(
            np.asarray(cast("ConstantRateLike", rate).values, dtype=np.float64)
        )
    else:
        linear = cast("LinearRateLike", rate)
        v0 = np.atleast_1d(np.asarray(linear.start_values, dtype=np.float64))
        v1 = np.atleast_1d(np.asarray(linear.end_values, dtype=np.float64))
    if not (ends.shape == v0.shape == v1.shape):
        msg = "rate values and epoch_end_times must have the same shape"
        raise ValueError(msg)
    check_epoch_ends(ends_name, ends)
    if not (np.all(np.isfinite(v0)) and np.all(np.isfinite(v1))):
        msg = "rate values must be finite"
        raise ValueError(msg)
    if np.any(v0 < 0.0) or np.any(v1 < 0.0):
        msg = "rate values must be non-negative"
        raise ValueError(msg)
    starts = np.concatenate([[0.0], ends[:-1]])
    slope = (v1 - v0) / (ends - starts)
    cum_before = np.concatenate([[0.0], np.cumsum(0.5 * (v0 + v1) * (ends - starts))])
    return Rate(
        starts,
        ends,
        v0,
        slope,
        cum_before[:-1],
        tuple(starts.tolist()),
        tuple(ends.tolist()),
        tuple(v0.tolist()),
        tuple(slope.tolist()),
    )


def epoch_rates(
    birth_rates: float | _Floats,
    death_rates: float | _Floats,
    times: float | _Floats,
    *,
    times_name: str = "Times",
) -> tuple[_Array, _Array, _Array]:
    """Validate per-epoch rates and epoch end times of a constant-rate simulator.

    Scalars are treated as one epoch.
    The three must have the same length, the rates must be finite and strictly
    positive, and the times finite, positive and strictly increasing; the times
    are checked as for a :class:`PiecewiseConstant` rate through
    :func:`as_rate`.
    ``times_name`` is how error messages name the times.

    Returns
    -------
    tuple of ndarray
        The birth rates, death rates and epoch end times as 1-D float64 arrays.

    """
    birth = np.atleast_1d(np.asarray(birth_rates, dtype=np.float64))
    death = np.atleast_1d(np.asarray(death_rates, dtype=np.float64))
    ends = np.atleast_1d(np.asarray(times, dtype=np.float64))
    if not (len(birth) == len(death) == len(ends)):
        msg = (
            f"birth_rates, death_rates, and {times_name.lower()} "
            "must have the same length"
        )
        raise ValueError(msg)
    check_positive_rates(birth, death, label="Birth rates and death rates")
    as_rate(PiecewiseConstant(birth, ends), ends_name=times_name)
    as_rate(PiecewiseConstant(death, ends), ends_name=times_name)
    return birth, death, ends


def _is_scalar(rate: RateInput) -> bool:
    return isinstance(rate, int | float)


def resolve_rates(
    birth_rate: RateInput, death_rate: RateInput, terminal_time: float | None
) -> tuple[Rate, Rate, float]:
    """Convert the rate inputs to :class:`Rate` arrays and fix the end time ``T``.

    Each rate is a float (constant on ``[0, T]``) or a rate-like.
    ``terminal_time`` is required when both are floats; otherwise it may be
    ``None`` or must equal the common end time of the rate-likes.

    Returns
    -------
    tuple
        ``(birth, death, T)``.

    Raises
    ------
    ValueError
        If ``terminal_time`` is not finite and positive, a rate is invalid, the
        rate-likes differ in end time, or ``terminal_time`` is missing or
        differs from their end time.

    """
    if terminal_time is not None:
        check_terminal_time(terminal_time)
    parsed = [
        None if _is_scalar(r) else as_rate(r)  # pyright: ignore[reportArgumentType]
        for r in (birth_rate, death_rate)
    ]
    ends = [r.end_time for r in parsed if r is not None]
    if not ends:
        if terminal_time is None:
            msg = "terminal_time is required when both rates are floats"
            raise ValueError(msg)
        horizon = float(terminal_time)
    else:
        horizon = ends[0]
        if not all(np.isclose(horizon, e) for e in ends):
            msg = "birth and death rates must have the same end time"
            raise ValueError(msg)
        if terminal_time is not None and not np.isclose(horizon, terminal_time):
            msg = "terminal_time must equal the end time of the rates"
            raise ValueError(msg)

    def finish(rate: RateInput, resolved: Rate | None) -> Rate:
        if resolved is not None:
            return resolved
        return as_rate(PiecewiseConstant([float(rate)], [horizon]))  # pyright: ignore[reportArgumentType]

    return (
        finish(birth_rate, parsed[0]),
        finish(death_rate, parsed[1]),
        horizon,
    )


def merged_segments(
    birth: Rate, death: Rate
) -> list[tuple[float, float, float, float, float, float]]:
    """Segments ``(start, end, birth_0, birth_slope, death_0, death_slope)``.

    Segment boundaries are the union of both rates' epoch end times (and 0),
    with edges closer than ``MERGE_REL_TOL * T`` merged.
    The rates must have the same end time (see :func:`resolve_rates`).
    """
    edges = np.union1d(
        np.concatenate([[0.0], birth.ends]), np.concatenate([death.ends])
    )
    edges = edges[np.concatenate([[True], np.diff(edges) > MERGE_REL_TOL * edges[-1]])]

    def line(rate: Rate, start: float, end: float) -> tuple[float, float]:
        k = min(
            int(np.searchsorted(rate.ends, 0.5 * (start + end))), len(rate.ends) - 1
        )
        k_start = float(rate.starts[k])
        slope = float(rate.slope[k])
        return float(rate.v0[k]) + slope * (start - k_start), slope

    out: list[tuple[float, float, float, float, float, float]] = []
    for s, e in itertools.pairwise(edges):
        b0, b1 = line(birth, float(s), float(e))
        d0, d1 = line(death, float(s), float(e))
        out.append((float(s), float(e), b0, b1, d0, d1))
    return out
