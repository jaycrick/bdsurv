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

from bdsurv._hazard import next_event_offset
from bdsurv._seeding import root_rng, spawn
from bdsurv.rates import (
    ConstantRateLike,
    LinearRateLike,
    PiecewiseConstant,
    PiecewiseLinear,
    merged_segments,
    resolve_rates,
)
from bdsurv.tree import SimulatedProcess, build_records

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
    segments = merged_segments(birth, death)
    max_time = segments[-1][1]
    if ensure_non_extinct and max_retries < 0:
        msg = "max_retries must be a non-negative integer"
        raise ValueError(msg)

    seeds = root_rng(random_seed, "seeding")
    attempts = max_retries if ensure_non_extinct else 0
    for _ in range(attempts + 1):
        rng = spawn(seeds, "dynamics")
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
                s = next_event_offset(n * (lam + mu), n * (b1 + d1), rng.exponential())
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
