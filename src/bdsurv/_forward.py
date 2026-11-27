r"""The one forward event kernel of birth-death processes.

This is the event-by-event exact simulation of Gillespie (1977); see
``docs/references.md``.
Its extension to rates that vary linearly within a segment is made here, by
inverting the cumulative hazard of the segment.

A process is advanced through the segments of :func:`bdsurv.rates.merged_segments`,
on each of which both per-individual rates are linear in time.
With $n$ individuals alive at time $t_0$ inside a segment, the total hazard is
$a + b s$ with $s = t - t_0$, $a = n(\lambda(t_0) + \mu(t_0))$ and
$b = n(\lambda' + \mu')$.
The next event offset inverts $a s + b s^2 / 2 = E$ with $E \sim \mathrm{Exp}(1)$;
on a segment where both slopes are zero this is the direct $E / a$.
An event past the end of the segment (or past the target time) is discarded and
redrawn from the boundary, which is exact because the event process has
independent increments.
The event is a birth with probability $\lambda(t) / (\lambda(t) + \mu(t))$.
The derivation is in ``docs/lifespans.md``, section "Forward simulation and the
other simulators".

A pair of rates in which either is a :class:`~bdsurv.hazards.Hazard` is run by
competing risks instead (:class:`RatePair`, ``_run_general``).
The $n$ alive individuals share one hazard, so the total birth hazard is
$n \lambda(t)$ and the total death hazard $n \mu(t)$, and these two event
processes have independent increments.
Draw $E_b, E_d \sim \mathrm{Exp}(1)$, invert $n \int \lambda = E_b$ and
$n \int \mu = E_d$ for the times $t_b$ and $t_d$, and take the earlier one as
the event; none before the target ends the run.
This is exact (Gillespie with competing risks) and needs no segments.
It uses two exponential draws per event, against one exponential and one
uniform draw on the linear path, so :func:`bdsurv._seeding.count_draws` totals
differ between the two paths.
The linear path itself is unchanged and its seeded output is bit-identical.

Genealogy convention: a newborn is inserted into ``alive_ids`` at a uniform
position at least 1, its parent is its left neighbour, and the spine is
``alive_ids[0]``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from bdsurv._hazard import (
    draw_count,
    forward_hazard_time_split,
    next_event_offset,
)
from bdsurv._seeding import bump, root_rng, spawn
from bdsurv.rates import Rate, ResolvedRate, merged_segments
from bdsurv.tree import build_records

type Segment = tuple[float, float, float, float, float, float]
"""``(start, end, lam_0, lam_slope, mu_0, mu_slope)``.

``lam_0`` and ``mu_0`` are the birth and death rates at ``start``;
``lam_slope`` and ``mu_slope`` are their slopes (``lambda'`` and ``mu'``).
"""


@dataclass(frozen=True)
class RatePair:
    """Birth and death rates run by competing risks (see the module docstring).

    ``end_time`` is the common horizon of the two rates.
    """

    birth: ResolvedRate
    death: ResolvedRate
    end_time: float


type Dynamics = list[Segment] | RatePair
"""Either merged linear segments or a general pair of rates."""


def build_dynamics(
    birth: ResolvedRate, death: ResolvedRate, end_time: float
) -> Dynamics:
    """Return merged segments if both rates are ``Rate``, else a :class:`RatePair`."""
    if isinstance(birth, Rate) and isinstance(death, Rate):
        return merged_segments(birth, death)
    return RatePair(birth, death, end_time)


@dataclass
class ForwardState:
    """A forward-simulated population: who is alive and every individual's record.

    Parameters
    ----------
    current_time
        Time up to which the process has been simulated.
    alive_ids
        Ids of the individuals alive, in genealogy order (the spine first).
    creation_time, parent_id, death_time
        Per-individual columns, indexed by individual id.
        ``parent_id`` is ``None`` for the root and ``death_time`` is ``None``
        while the individual is alive.

    """

    current_time: float = 0.0
    alive_ids: list[int] = field(default_factory=lambda: [0])
    creation_time: list[float] = field(default_factory=lambda: [0.0])
    parent_id: list[int | None] = field(default_factory=lambda: [None])
    death_time: list[float | None] = field(default_factory=lambda: [None])

    @classmethod
    def root(cls) -> ForwardState:
        """Return the state with one individual (id 0) created at time 0."""
        return cls()

    def copy(self) -> ForwardState:
        """Return an independent copy; no list is shared with this state."""
        return ForwardState(
            current_time=self.current_time,
            alive_ids=self.alive_ids.copy(),
            creation_time=self.creation_time.copy(),
            parent_id=self.parent_id.copy(),
            death_time=self.death_time.copy(),
        )

    @property
    def n_individuals(self) -> int:
        """Number of individuals ever created, alive or dead."""
        return len(self.creation_time)

    def to_records(self, terminal_time: float) -> pd.DataFrame:
        """Return the records table; the spine is ``alive_ids[0]`` if any."""
        return build_records(
            self.creation_time,
            self.parent_id,
            self.death_time,
            self.alive_ids[0] if self.alive_ids else None,
            terminal_time,
        )


def _apply_event(
    state: ForwardState,
    event_time: float,
    *,
    is_birth: bool,
    rng: np.random.Generator,
) -> None:
    """Apply a birth or death at ``event_time`` to a uniformly chosen individual."""
    bump("events")
    n = len(state.alive_ids)
    if is_birth:
        child_id = state.n_individuals
        position = 1 if n == 1 else int(rng.integers(1, n + 1))
        state.parent_id.append(state.alive_ids[position - 1])
        state.creation_time.append(event_time)
        state.death_time.append(None)
        state.alive_ids.insert(position, child_id)
    else:
        dead_id = state.alive_ids.pop(int(rng.integers(0, n)))
        state.death_time[dead_id] = event_time


def _next_event_time(
    state: ForwardState, segment: Segment, rng: np.random.Generator
) -> float:
    """Return the next event time within ``segment`` (``inf`` if none)."""
    start, _, lam_0, lam_slope, mu_0, mu_slope = segment
    n = len(state.alive_ids)
    local = state.current_time - start
    # a = n (lambda + mu) at the current time and b = n (lambda' + mu'), the
    # total hazard being a + b s; the offset s solves a s + b s^2 / 2 = E with
    # E ~ Exp(1) (docs/lifespans.md, section "Forward simulation and the other
    # simulators").
    a = n * ((lam_0 + lam_slope * local) + (mu_0 + mu_slope * local))
    exponential = float(rng.exponential())
    if lam_slope == 0.0 and mu_slope == 0.0:
        offset = exponential / a if a > 0.0 else float("inf")
    else:
        offset = next_event_offset(a, n * (lam_slope + mu_slope), exponential)
    return state.current_time + offset


def _run_segment(
    state: ForwardState,
    segment: Segment,
    stop: float,
    rng: np.random.Generator,
    on_event: Callable[[ForwardState, float], None] | None,
) -> None:
    """Advance ``state`` inside ``segment`` to ``stop`` or until extinction."""
    start, _, lam_0, lam_slope, mu_0, mu_slope = segment
    while state.current_time < stop and state.alive_ids:
        event_time = _next_event_time(state, segment, rng)
        if event_time >= stop:
            state.current_time = stop
            return
        state.current_time = event_time
        local = event_time - start
        lam = lam_0 + lam_slope * local
        mu = mu_0 + mu_slope * local
        # The event is a birth with probability lam / (lam + mu).
        _apply_event(
            state,
            event_time,
            is_birth=bool(rng.random() * (lam + mu) < lam),
            rng=rng,
        )
        if on_event is not None:
            on_event(state, event_time)


def _draws(rng: np.random.Generator, m: int, n: int) -> list[float]:
    """Draw ``m`` exponentials over ``n``; ``m == 1`` is the original scalar draw."""
    if m == 1:
        return [float(rng.exponential()) / n]
    return (rng.exponential(size=m) / n).tolist()


def _run_general(
    state: ForwardState,
    pair: RatePair,
    stop: float,
    rng: np.random.Generator,
    on_event: Callable[[ForwardState, float], None] | None,
) -> None:
    """Advance ``state`` to ``stop`` or extinction by competing risks."""
    while state.current_time < stop and state.alive_ids:
        n = len(state.alive_ids)
        t = state.current_time
        e_birth = _draws(rng, draw_count(pair.birth), n)
        e_death = _draws(rng, draw_count(pair.death), n)
        t_birth = forward_hazard_time_split(pair.birth, t, e_birth, stop)
        t_death = forward_hazard_time_split(pair.death, t, e_death, stop)
        event_time = min(t_birth, t_death)
        if event_time >= stop:  # the cap is returned when there is no event
            state.current_time = stop
            return
        state.current_time = event_time
        _apply_event(state, event_time, is_birth=t_birth < t_death, rng=rng)
        if on_event is not None:
            on_event(state, event_time)


def advance(
    state: ForwardState,
    dynamics: Dynamics,
    target_time: float,
    rng: np.random.Generator,
    on_event: Callable[[ForwardState, float], None] | None = None,
) -> None:
    """Advance ``state`` in place from its current time to ``target_time``.

    Parameters
    ----------
    state
        The population to advance; modified in place.
    dynamics
        Segments as built by :func:`bdsurv.rates.merged_segments` (those
        ending at or before ``state.current_time`` are skipped), or a
        :class:`RatePair` run by competing risks.
    target_time
        Time to advance to; on return ``state.current_time == target_time``,
        also if the population has died out.
    rng
        Source of the draws: ``exponential`` for the event offset, ``random``
        for birth versus death (segments only; a :class:`RatePair` draws a
        second exponential instead), ``integers`` for the position.
    on_event
        If given, called as ``on_event(state, event_time)`` after each event.

    Raises
    ------
    ValueError
        If ``target_time`` is before ``state.current_time``.

    """
    if target_time < state.current_time:
        msg = "target_time cannot be less than current_time"
        raise ValueError(msg)

    if isinstance(dynamics, RatePair):
        _run_general(state, dynamics, target_time, rng, on_event)
        state.current_time = target_time
        return
    for segment in dynamics:
        if segment[1] <= state.current_time:
            continue
        stop = min(segment[1], target_time)
        _run_segment(state, segment, stop, rng, on_event)
        if not state.alive_ids or stop >= target_time:
            break
    state.current_time = target_time


def retry_until_survival[T](
    attempt: Callable[[], T | None],
    *,
    ensure_non_extinct: bool,
    max_retries: int,
    name: str,
) -> T:
    """Run ``attempt`` until it returns a result, with at most ``max_retries`` retries.

    Parameters
    ----------
    attempt
        Zero-argument callable returning a result, or ``None`` if the attempt
        went extinct and should be redone.
    ensure_non_extinct
        If false there is a single attempt and its result is returned as is
        (``attempt`` is then expected to return it even if extinct).
    max_retries
        Number of redraws allowed when ``ensure_non_extinct`` is true.
    name
        Name of the calling simulator, for the error message.

    Raises
    ------
    ValueError
        If ``ensure_non_extinct`` and ``max_retries`` is negative.
    RuntimeError
        If no attempt returned a result.

    """
    if ensure_non_extinct and max_retries < 0:
        msg = "max_retries must be a non-negative integer"
        raise ValueError(msg)
    retries = max_retries if ensure_non_extinct else 0
    for _ in range(retries + 1):
        result = attempt()
        if result is not None:
            return result
    msg = (
        f"{name}: exhausted {retries} retries "
        "without producing a non-extinct population"
    )
    raise RuntimeError(msg)


def simulate_forward(  # noqa: PLR0913  # one internal call site
    birth: ResolvedRate,
    death: ResolvedRate,
    end_time: float,
    random_seed: int | None,
    *,
    ensure_non_extinct: bool,
    max_retries: int,
    name: str,
) -> tuple[pd.DataFrame, Dynamics]:
    """Simulate one process from a single root to the common end time of the rates.

    Parameters
    ----------
    birth, death
        Per-individual rates with a common end time ``T``.
    end_time
        The common end time ``T``.
    random_seed
        Seed for reproducibility.
    ensure_non_extinct, max_retries
        As in :func:`retry_until_survival`.
    name
        Name of the calling simulator, for the error message.

    Returns
    -------
    tuple
        The records table at ``T`` and the dynamics used.

    """
    dynamics = build_dynamics(birth, death, end_time)
    terminal_time = end_time
    seeds = root_rng(random_seed, "seeding")

    def attempt() -> ForwardState | None:
        state = ForwardState.root()
        advance(state, dynamics, terminal_time, spawn(seeds, "dynamics"))
        return None if ensure_non_extinct and not state.alive_ids else state

    state = retry_until_survival(
        attempt,
        ensure_non_extinct=ensure_non_extinct,
        max_retries=max_retries,
        name=name,
    )
    return state.to_records(terminal_time), dynamics
