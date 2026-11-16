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

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple, Protocol

import jax
import numpy as np
from numpy.typing import NDArray

from bdsurv.tree import SimulatedProcess, build_records

_RelTol = 1e-12

_Floats = Sequence[float] | NDArray[np.float64] | jax.Array


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


class _LinearArrays(NamedTuple):
    start_values: np.ndarray
    end_values: np.ndarray
    epoch_end_times: np.ndarray


def _as_linear(rate: ConstantRateLike | LinearRateLike) -> _LinearArrays:
    ends = np.atleast_1d(np.asarray(rate.epoch_end_times, dtype=np.float64))
    if hasattr(rate, "values"):
        v = np.atleast_1d(np.asarray(rate.values, dtype=np.float64))  # type: ignore[union-attr]
        return _LinearArrays(v, v, ends)
    return _LinearArrays(
        np.atleast_1d(np.asarray(rate.start_values, dtype=np.float64)),  # type: ignore[union-attr]
        np.atleast_1d(np.asarray(rate.end_values, dtype=np.float64)),  # type: ignore[union-attr]
        ends,
    )


def _segments(
    birth: _LinearArrays, death: _LinearArrays
) -> list[tuple[float, float, float, float, float, float]]:
    """Segments ``(start, end, birth_0, birth_slope, death_0, death_slope)``.

    Segment boundaries are the union of both rates' epoch end times.
    """
    b_ends = np.asarray(birth.epoch_end_times)
    d_ends = np.asarray(death.epoch_end_times)
    if not np.isclose(b_ends[-1], d_ends[-1]):
        msg = "birth and death rates must have the same end time"
        raise ValueError(msg)
    edges = np.union1d(np.concatenate([[0.0], b_ends]), np.concatenate([d_ends]))
    edges = edges[np.concatenate([[True], np.diff(edges) > _RelTol * edges[-1]])]

    def line(rate: _LinearArrays, start: float, end: float) -> tuple[float, float]:
        ends = np.asarray(rate.epoch_end_times)
        k = min(int(np.searchsorted(ends, 0.5 * (start + end))), len(ends) - 1)
        k_start = 0.0 if k == 0 else float(ends[k - 1])
        v0 = float(np.asarray(rate.start_values)[k])
        v1 = float(np.asarray(rate.end_values)[k])
        slope = (v1 - v0) / (float(ends[k]) - k_start)
        return v0 + slope * (start - k_start), slope

    out: list[tuple[float, float, float, float, float, float]] = []
    for s, e in itertools.pairwise(edges):
        b0, b1 = line(birth, float(s), float(e))
        d0, d1 = line(death, float(s), float(e))
        out.append((float(s), float(e), b0, b1, d0, d1))
    return out


def _next_event_offset(a: float, b: float, e: float) -> float:
    """Solve ``a s + b s^2 / 2 = e`` for the smallest ``s >= 0``, or ``inf``."""
    disc = a * a + 2.0 * b * e
    if disc < 0.0:
        return math.inf
    denom = a + math.sqrt(disc)
    if denom <= 0.0:
        return math.inf
    return 2.0 * e / denom


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
        If the end times differ or ``max_retries`` is negative.
    RuntimeError
        If every retry went extinct.

    """
    birth = _as_linear(birth_rate)
    death = _as_linear(death_rate)
    segments = _segments(birth, death)
    max_time = segments[-1][1]
    if ensure_non_extinct and max_retries < 0:
        msg = "max_retries must be a non-negative integer"
        raise ValueError(msg)

    seeds = np.random.default_rng(random_seed)
    attempts = max_retries if ensure_non_extinct else 0
    for _ in range(attempts + 1):
        rng = np.random.default_rng(
            None if random_seed is None else int(seeds.integers(0, 2**31 - 1))
        )
        alive = [0]
        creation = [0.0]
        parent: list[int | None] = [None]
        death_time: list[float | None] = [None]

        for start, end, b0, b1, d0, d1 in segments:
            t = start
            while alive:
                n = len(alive)
                lam = b0 + b1 * (t - start)
                mu = d0 + d1 * (t - start)
                s = _next_event_offset(n * (lam + mu), n * (b1 + d1), rng.exponential())
                t += s
                if t >= end:
                    break
                lam = b0 + b1 * (t - start)
                mu = d0 + d1 * (t - start)
                if rng.random() * (lam + mu) < lam:
                    new_id = len(creation)
                    pos = 1 if n == 1 else int(rng.integers(1, n + 1))
                    parent.append(alive[pos - 1])
                    creation.append(t)
                    death_time.append(None)
                    alive.insert(pos, new_id)
                else:
                    death_time[alive.pop(int(rng.integers(0, n)))] = t
            if not alive:
                break

        if ensure_non_extinct and not alive:
            continue

        df = build_records(
            creation, parent, death_time, alive[0] if alive else None, max_time
        )
        return SimulatedProcess(
            records=df,
            epoch_end_times=np.array([seg[1] for seg in segments]),
        )

    msg = (
        f"simulate_birth_death_linear: exhausted {attempts} retries "
        "without producing a non-extinct population"
    )
    raise RuntimeError(msg)
