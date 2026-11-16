r"""Forward simulation of birth-death processes with time-varying rates.

:func:`simulate_birth_death` runs the one forward event kernel
(:mod:`bdsurv._forward`) over rates that are piecewise constant or piecewise
linear in time.
The method is the exact event-by-event simulation of Gillespie (1977); see
``docs/references.md``.
The extension to rates that vary linearly within a segment is made here, by
inverting the cumulative hazard of the segment, as below
(``docs/lifespans.md``, section "Forward simulation and the other simulators").
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

from bdsurv._forward import simulate_forward
from bdsurv.rates import RateInput, as_rate_like, resolve_rates
from bdsurv.tree import SimulatedTree

__all__ = ["simulate_birth_death"]


def simulate_birth_death(
    birth_rate: RateInput,
    death_rate: RateInput,
    terminal_time: float | None = None,
    random_seed: int | None = None,
    *,
    ensure_non_extinct: bool = False,
    max_retries: int = 100,
) -> SimulatedTree:
    r"""Simulate a birth-death process from one individual at time 0 up to ``T``.

    Individuals are born at rate $\lambda$ and die at rate $\mu$, each rate a
    function of time that is constant or linear within an epoch.
    Each event time is drawn by inverting the cumulative hazard of the
    superposed birth and death processes,
    $s = 2E / (a + \sqrt{a^2 + 2bE})$ with $E \sim \mathrm{Exp}(1)$,
    where $a$ and $b$ are the total hazard and its slope, both scaled by the
    number alive.
    This is the event-by-event method of Gillespie (1977); see
    ``docs/references.md``.
    The extension to rates linear within a segment is by inversion of the
    cumulative hazard, not part of that reference.
    The event is a birth with probability $\lambda(t) / (\lambda(t) + \mu(t))$.
    The process is not conditioned on survival, so it may be extinct at ``T``.

    Genealogy convention: a newborn is inserted among the individuals alive at a
    uniformly chosen position after the first one, its parent is its left
    neighbour, and the spine is the first individual alive.
    The spine is marked in ``is_spine`` if anybody is alive at ``T``.

    Parameters
    ----------
    birth_rate, death_rate
        Per-individual rates, each a float (constant on ``[0, T]``) or a
        rate-like (:class:`~bdsurv.rates.PiecewiseConstant`,
        :class:`~bdsurv.rates.PiecewiseLinear` or any object with the same
        attributes), finite and non-negative (zero is allowed).
        Rate-likes must have a common end time ``T``; their epoch boundaries need
        not coincide.
    terminal_time
        The time ``T``.
        Required when both rates are floats; otherwise ``None`` or equal to the
        end time of the rate-likes.
    random_seed
        Seed for reproducibility.
    ensure_non_extinct
        If true, redraw until at least one individual survives to ``T``, up to
        ``max_retries`` redraws.
    max_retries
        Maximum number of redraws when ``ensure_non_extinct`` is true.

    Returns
    -------
    SimulatedTree
        The records of every individual born, with the rates stored as rate
        objects (a float rate as a one-epoch
        :class:`~bdsurv.rates.PiecewiseConstant`) and ``terminal_time`` set to ``T``.

    Raises
    ------
    ValueError
        If a rate has negative or non-finite values or invalid epoch end times,
        the end times differ, ``terminal_time`` is missing or invalid or does not
        match the rates, or ``max_retries`` is negative.
    RuntimeError
        If ``ensure_non_extinct`` is true and every retry went extinct.

    """
    birth, death, horizon = resolve_rates(birth_rate, death_rate, terminal_time)
    records, _ = simulate_forward(
        birth,
        death,
        random_seed,
        ensure_non_extinct=ensure_non_extinct,
        max_retries=max_retries,
        name="simulate_birth_death",
    )
    return SimulatedTree(
        records=records,
        birth_rate=as_rate_like(birth_rate, horizon),
        death_rate=as_rate_like(death_rate, horizon),
        terminal_time=horizon,
    )
