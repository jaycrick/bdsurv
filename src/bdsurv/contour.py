"""Simulate a survival-conditioned birth-death tree via contour-process reconstruction.

This is the "exact" method: the contour of a tree conditioned to survive to a
terminal time ``T`` is generated directly, then decoded back into an
individual-level genealogy via a two-stack reconstruction.
See ``docs/contour_process.md`` for the construction and the reconstruction.
Lambert (2010) is the standard reference for contour processes of splitting
trees; see ``docs/references.md``.

The contour alternates downward jumps (driven by the birth rate) and upward
jumps (driven by the death rate) between ``0`` and ``T``.
:class:`Walker` is the one stepper: it draws the jumps by hazard inversion at
the absolute time of each jump and runs excursions between the levels ``0`` and
``T``.
This module builds the survival-conditioned contour from excursions
(:func:`_survival_contour`), joins pieces (:func:`join_pieces`) and decodes a
contour list into a tree (:func:`contour_process_list_to_tree`, which
validates it); ``contour_conditioned`` builds the population- and
age-conditioned contours from the same walker.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from bdsurv._hazard import backward_hazard_time, close, cumulative, forward_hazard_time
from bdsurv._seeding import bump, root_rng
from bdsurv.hazards import Hazard
from bdsurv.rates import (
    ConstantRateLike,
    LinearRateLike,
    RateInput,
    ResolvedRate,
    as_rate_like,
    resolve_rates,
)
from bdsurv.tree import SimulatedTree, build_records

MIN_CONTOUR_PROCESS_LENGTH = 3


@dataclass(frozen=True)
class Walker:
    """The step and run primitives of the contour construction.

    Downward jumps use the birth rate, upward jumps the death rate.
    A step never returns its start level, and endpoints are snapped exactly
    to ``0`` and ``T`` (the decoder compares with ``close``).

    Parameters
    ----------
    birth, death
        The birth and death rates over ``[0, T]``.
    terminal_time
        The time ``T``.
    rng
        Generator providing the ``Exp(1)`` draws of the jumps.
    """

    birth: ResolvedRate
    death: ResolvedRate
    terminal_time: float
    rng: np.random.Generator

    def down(self, x: float) -> float:
        """Return the next level below ``x``, snapped to ``0`` when close."""
        # The level s < x with int_s^x lambda = E, E ~ Exp(1) (``docs/
        # contour_conditioned.md``, section "Steps").
        while True:
            s = backward_hazard_time(self.birth, x, float(self.rng.exponential()))
            if close(s, 0.0):
                s = 0.0
            if s != x:
                return s

    def up(self, y: float) -> float:
        """Return the next level above ``y``, snapped to ``T`` when close."""
        # The level s > y with int_y^s mu = E, E ~ Exp(1) (``docs/
        # contour_conditioned.md``, section "Steps").
        while True:
            s = forward_hazard_time(
                self.death, y, float(self.rng.exponential()), self.terminal_time
            )
            if close(s, self.terminal_time):
                s = self.terminal_time
            if s != y:
                return s

    def run(self, path: list[float], next_move: Literal["up", "down"]) -> list[float]:
        """Extend ``path`` by alternating moves until it reaches ``0`` or ``T``.

        A downward move ends the run at ``0`` and an upward move at ``T``.
        The number of steps appended is added to the ``contour_steps`` counter.
        """
        start = len(path)
        while True:
            if next_move == "down":
                path.append(self.down(path[-1]))
                if path[-1] == 0.0:
                    break
                next_move = "up"
            else:
                path.append(self.up(path[-1]))
                if path[-1] == self.terminal_time:
                    break
                next_move = "down"
        bump("contour_steps", len(path) - start)
        return path

    def top_excursion(self) -> list[float]:
        """Run from ``T``; ends at ``T`` (``TT``) or at ``0`` (``T0``)."""
        return self.run([self.terminal_time], "down")

    def bottom_excursion(self) -> list[float]:
        """Run from ``0``; ends at ``T`` (``0T``) or at ``0`` (rejected by callers)."""
        return self.run([0.0], "up")


def join_pieces(
    first: list[float], tt: list[list[float]], last: list[float]
) -> list[float]:
    """Join ``first`` (``0..T``), the ``T..T`` excursions and ``last`` (``T..0``).

    Parameters
    ----------
    first
        A piece from ``0`` to ``T``.
    tt
        Excursions from ``T`` to ``T``, in order.
    last
        A piece from ``T`` to ``0``.

    Returns
    -------
    list[float]
        The contour, without repeating the shared endpoints.
    """
    contour = list(first)
    for excursion in tt:
        contour.extend(excursion[1:])
    contour.extend(last[1:])
    return contour


def sample_until[T](make: Callable[[], T], accept: Callable[[T], bool]) -> T:
    """Call ``make`` until ``accept`` holds for its result, and return that result.

    Parameters
    ----------
    make
        Draws one candidate; called with no arguments.
    accept
        Whether to keep a candidate.

    Returns
    -------
    T
        The first accepted candidate.
    """
    while True:
        candidate = make()
        if accept(candidate):
            return candidate


def _survival_contour(walker: Walker) -> list[float]:
    """Generate the contour of a tree conditioned to survive to ``T``.

    The spine piece is the reversed first top excursion that ends at ``0``;
    an excursion from ``T`` that returns to ``T`` first is rejected (the spine
    restarts from ``T``; ``docs/contour_process.md``, section "Concepts") and the
    next one starts afresh from ``T``.
    Further top excursions are then drawn; each one ending at ``T`` is another
    individual alive at ``T`` and is kept, and the first one ending at ``0`` is
    the last piece.
    The number of rejected spine excursions is added to ``spine_resets``.

    Parameters
    ----------
    walker
        The stepper, holding the rates, ``T`` and the generator.

    Returns
    -------
    list[float]
        The contour, from ``0`` to ``0``, visiting ``T`` once per survivor.
    """
    draws = 0

    def make_spine() -> list[float]:
        nonlocal draws
        draws += 1
        return walker.top_excursion()

    spine = sample_until(make_spine, lambda piece: piece[-1] == 0.0)
    bump("spine_resets", draws - 1)
    tt: list[list[float]] = []
    while True:
        excursion = walker.top_excursion()
        if excursion[-1] == 0.0:
            return join_pieces(spine[::-1], tt, excursion)
        tt.append(excursion)


def validate_contour_process_list(contour_process_list: list[float]) -> None:
    """Validate the alternating peak/valley contour structure.

    Validation rules enforced:
    - List length is odd and at least 3.
    - List begins and ends with 0.
    - Interior values strictly alternate between local maxima (odd indices)
      and local minima (even indices).

    Raises
    ------
    ValueError
        If any validation rule is violated.

    """
    if len(contour_process_list) < MIN_CONTOUR_PROCESS_LENGTH:
        msg = "contour_process_list must contain at least three values"
        raise ValueError(msg)
    if len(contour_process_list) % 2 == 0:
        msg = "contour_process_list must have odd length"
        raise ValueError(msg)
    if not close(contour_process_list[0], 0.0):
        msg = "contour_process_list must begin with 0"
        raise ValueError(msg)
    if not close(contour_process_list[-1], 0.0):
        msg = "contour_process_list must end with 0"
        raise ValueError(msg)

    values = [float(value) for value in contour_process_list]
    for index in range(1, len(values) - 1):
        previous_value = values[index - 1]
        current_value = values[index]
        next_value = values[index + 1]
        if index % 2 == 1:
            if not (current_value > previous_value and current_value > next_value):
                msg = (
                    "contour_process_list must alternate with strict local maxima "
                    "at odd interior positions"
                )
                raise ValueError(msg)
        elif not (current_value < previous_value and current_value < next_value):
            msg = (
                "contour_process_list must alternate with strict local minima "
                "at even interior positions"
            )
            raise ValueError(msg)


def contour_process_list_to_tree(
    contour_process_list: list[float],
    *,
    birth_rate: ConstantRateLike | LinearRateLike | Hazard,
    death_rate: ConstantRateLike | LinearRateLike | Hazard,
) -> SimulatedTree:
    """Reconstruct a ``SimulatedTree`` from a validated contour list.

    The contour is interpreted as alternating death and birth times. This
    function returns a canonical `SimulatedTree` where records that reach
    the terminal time are treated as surviving, and exactly one surviving
    individual is marked as the spine: the individual that was last-born
    among those alive when the process first reaches the terminal time
    (equivalently, the individual at the top of the genealogy stack when T
    is first hit).
    The tree's ``birth_rate`` and ``death_rate`` are the rate objects given, and
    its ``terminal_time`` is the largest value of the contour.
    """
    validate_contour_process_list(contour_process_list)

    values = [float(value) for value in contour_process_list]
    terminal_time = float(max(values))

    creation = [values[0]]
    parent: list[int | None] = [None]
    death_time = [float("nan")]

    birth_times_stack = [values[0]]
    individual_number_stack = [0]
    next_individual_id = 1
    # Spine = the individual at the top of the genealogy stack the first time
    # the process reaches T; equivalently, the last-born individual alive at T.
    spine_individual_id: int | None = None

    for index in range(1, len(values) - 1):
        current_value = values[index]
        # Odd indices are death times for the current top-of-stack individual
        if index % 2 == 1:
            if close(current_value, terminal_time):
                # Individual that 'dies' at terminal time is considered alive
                death_time[individual_number_stack[-1]] = float("nan")
                if spine_individual_id is None:
                    spine_individual_id = individual_number_stack[-1]
            else:
                death_time[individual_number_stack[-1]] = current_value
            continue

        # Even indices are birth times: pop ancestors with birth times >=
        # current birth time
        while birth_times_stack and birth_times_stack[-1] >= current_value:
            birth_times_stack.pop()
            individual_number_stack.pop()

        if not individual_number_stack:
            raise ValueError("contour_process_list is not valid for reconstruction")

        creation.append(current_value)
        parent.append(individual_number_stack[-1])
        death_time.append(float("nan"))

        birth_times_stack.append(current_value)
        individual_number_stack.append(next_individual_id)
        next_individual_id += 1

    # Records with death_time == terminal_time are considered alive.
    records_df = build_records(
        creation, parent, death_time, spine_individual_id, terminal_time
    )

    return SimulatedTree(
        records=records_df,
        birth_rate=birth_rate,
        death_rate=death_rate,
        terminal_time=terminal_time,
    )


def simulate_contour_given_survival(
    birth_rate: RateInput,
    death_rate: RateInput,
    terminal_time: float | None = None,
    random_seed: int | None = None,
) -> SimulatedTree:
    r"""Simulate a birth-death tree conditioned to survive to ``T``, exactly.

    The contour of the tree is generated directly by excursions of the
    :class:`Walker` between the levels ``0`` and ``T`` and decoded into a
    genealogy (see ``docs/contour_process.md``).
    Lambert (2010) is the standard reference for contour processes of splitting
    trees; see ``docs/references.md``.
    The exponential jumps of the constant-rate case are replaced by hazard
    inversion at the absolute time of the jump
    (``docs/contour_process.md``, section "Linear (time-varying) rates").
    A downward jump from level $x$ goes to the time $s < x$ at which
    $\int_s^x \lambda = E$, and an upward jump from $y$ to the time $s > y$ at
    which $\int_y^s \mu = E$, where $E \sim \mathrm{Exp}(1)$; each is solved
    exactly as a quadratic within every linear segment, and as an exponential
    draw for a constant rate.
    Upward jumps are capped at $T$, and a spine excursion that reaches $T$ is
    discarded and redrawn from $T$.

    Parameters
    ----------
    birth_rate, death_rate
        Per-individual rates, each a float (constant on ``[0, T]``) or a
        rate-like (:class:`~bdsurv.rates.PiecewiseConstant`,
        :class:`~bdsurv.rates.PiecewiseLinear` or any object with the same
        attributes), finite and non-negative.
        Rate-likes must have a common end time ``T``.
        The death rate must not be identically zero.
    terminal_time
        The time ``T`` the tree is conditioned to survive to.
        Required when both rates are floats; otherwise ``None`` or equal to the
        end time of the rate-likes.
    random_seed
        Seed for reproducibility.

    Returns
    -------
    SimulatedTree
        The tree with exactly one surviving individual marked ``is_spine``.
        Its ``birth_rate`` and ``death_rate`` are the rate objects (a float
        rate as a one-epoch :class:`~bdsurv.rates.PiecewiseConstant`).

    Raises
    ------
    ValueError
        If a rate or ``terminal_time`` is invalid, the rate-likes differ in end
        time, ``terminal_time`` is missing or does not match the rates, or the
        death rate is identically zero.

    """
    birth, death, horizon = resolve_rates(birth_rate, death_rate, terminal_time)
    if cumulative(death, horizon) <= 0.0:
        msg = "death_rate must not be identically zero"
        raise ValueError(msg)
    walker = Walker(birth, death, horizon, root_rng(random_seed, "contour"))
    return contour_process_list_to_tree(
        _survival_contour(walker),
        birth_rate=as_rate_like(birth_rate, horizon),
        death_rate=as_rate_like(death_rate, horizon),
    )
