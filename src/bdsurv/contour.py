"""Simulate a survival-conditioned birth-death tree via contour-process reconstruction.

This is the "exact" method: the contour (Lukasiewicz-style exploration path)
of a tree conditioned to survive to a terminal time ``T`` is generated
directly, then decoded back into an individual-level genealogy via a
two-stack reconstruction. See ``docs/contour_process.md`` for the full
derivation.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from bdsurv._hazard import backward_hazard_time, close, cumulative, forward_hazard_time
from bdsurv._rng_counter import bump, make_rng
from bdsurv.rates import (
    ConstantRateLike,
    LinearRateLike,
    check_positive_rates,
    check_terminal_time,
    resolve_rates,
)
from bdsurv.tree import SimulatedTree, build_records

_Step = Callable[[float, np.random.Generator], float]

MIN_CONTOUR_PROCESS_LENGTH = 3


def _generate_component(
    *,
    down_step: _Step,
    up_step: _Step,
    terminal_time: float,
    rng: np.random.Generator,
    reset_at_terminal: bool,
) -> list[float]:
    """Generate one contour excursion from ``T`` down to 0.

    ``down_step(x, rng)`` returns the next (lower) level after a downward jump
    from ``x``, floored at 0; ``up_step(y, rng)`` returns the next (higher)
    level after an upward jump from ``y``, capped at ``terminal_time``.
    With ``reset_at_terminal`` (the spine) an upward jump that reaches
    ``terminal_time`` discards the excursion and restarts it from ``[T]``.
    """
    component = [float(terminal_time)]
    steps = resets = 0
    while True:
        downward = down_step(component[-1], rng)
        component.append(downward)
        steps += 1
        if close(downward, 0.0):
            bump("contour_steps", steps)
            bump("spine_resets", resets)
            return component

        upward = up_step(component[-1], rng)
        if reset_at_terminal and close(upward, terminal_time):
            component = [float(terminal_time)]
            resets += 1
            continue
        component.append(upward)


def _constant_steps(
    birth_rate: float, death_rate: float, terminal_time: float
) -> tuple[_Step, _Step]:
    """Downward jumps are Exp(birth_rate); upward jumps Exp(death_rate)."""

    def down(x: float, rng: np.random.Generator) -> float:
        return max(x - float(rng.exponential(scale=1.0 / birth_rate)), 0.0)

    def up(y: float, rng: np.random.Generator) -> float:
        return min(y + float(rng.exponential(scale=1.0 / death_rate)), terminal_time)

    return down, up


def _generate_non_spine_component(
    *,
    birth_rate: float,
    death_rate: float,
    terminal_time: float,
    rng: np.random.Generator,
) -> list[float]:
    """Generate the non-spine contour component.

    Meaning
    -------
    The non-spine component is an excursion started at the terminal time
    ``T`` that alternates between downward jumps (exponential with rate
    ``birth_rate``) and upward jumps (exponential with rate ``death_rate``),
    terminated when a downward jump reaches 0. The returned list contains the
    successive visited levels, including the starting ``T`` and the final 0.

    Assumptions
    -----------
    - ``birth_rate`` and ``death_rate`` are positive scalars.
    - ``terminal_time`` is a positive scalar.
    - ``rng`` is a NumPy ``Generator`` used for reproducibility.

    Returns
    -------
    list[float]
        The non-spine contour list, e.g. ``[T, x1, y1, x2, ..., 0]``.

    """
    down, up = _constant_steps(birth_rate, death_rate, float(terminal_time))
    return _generate_component(
        down_step=down,
        up_step=up,
        terminal_time=float(terminal_time),
        rng=rng,
        reset_at_terminal=False,
    )


def _generate_spine_component(
    *,
    birth_rate: float,
    death_rate: float,
    terminal_time: float,
    rng: np.random.Generator,
) -> list[float]:
    """Generate the spine contour component.

    Meaning
    -------
    The spine component follows the same alternating construction as the
    non-spine component, but with one important difference: if an upward
    jump would reach the terminal time ``T``, the spine excursion is reset to
    the single-element list ``[T]`` and generation restarts. This implements
    the reset-on-terminal-time rule for the distinguished spine lineage.

    Returns
    -------
    list[float]
        The spine contour list, e.g. ``[T, x1, y1, x2, ..., 0]``.

    """
    down, up = _constant_steps(birth_rate, death_rate, float(terminal_time))
    return _generate_component(
        down_step=down,
        up_step=up,
        terminal_time=float(terminal_time),
        rng=rng,
        reset_at_terminal=True,
    )


def _build_contour_process_list(
    spine_component: list[float],
    non_spine_component: list[float],
) -> list[float]:
    """Construct the contour process from the two contour components.

    The contour process list is built by reversing the spine component and
    appending the non-spine component without its first element (the shared
    terminal-time ``T``) to avoid duplicating the endpoint. Example:

        spine_component = [T, a, b, 0]
        non_spine_component = [T, c, d, 0]

    Yields
    ------
        contour = [0, b, a, T, c, d, 0]

    """
    if not spine_component:
        msg = "spine_component must not be empty"
        raise ValueError(msg)
    if not non_spine_component:
        msg = "non_spine_component must not be empty"
        raise ValueError(msg)

    return list(reversed(spine_component)) + non_spine_component[1:]


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
    birth_rate: float,
    death_rate: float,
    birth_rate_function: ConstantRateLike | LinearRateLike | None = None,
    death_rate_function: ConstantRateLike | LinearRateLike | None = None,
) -> SimulatedTree:
    """Reconstruct a ``SimulatedTree`` from a validated contour list.

    The contour is interpreted as alternating death and birth times. This
    function returns a canonical `SimulatedTree` where records that reach
    the terminal time are treated as surviving, and exactly one surviving
    individual is marked as the spine: the individual that was last-born
    among those alive when the process first reaches the terminal time
    (equivalently, the individual at the top of the genealogy stack when T
    is first hit).
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
        birth_rate=float(birth_rate),
        death_rate=float(death_rate),
        terminal_time=terminal_time,
        birth_rate_function=birth_rate_function,
        death_rate_function=death_rate_function,
    )


def seeded_rng(random_seed: int | None) -> np.random.Generator:
    """Return the contour generator, seeded through a child seed when given."""
    if random_seed is None:
        return make_rng(None, "contour")
    base_seed_rng = make_rng(random_seed, "seeding")
    return make_rng(int(base_seed_rng.integers(0, 2**31 - 1)), "contour")


def simulate_birth_death_process_from_contour_process(
    birth_rate: float,
    death_rate: float,
    terminal_time: float,
    random_seed: int | None = None,
) -> SimulatedTree:
    """Simulate a birth-death tree using contour-process reconstruction.

    Raises
    ------
    ValueError
        If a rate or ``terminal_time`` is not finite and positive.

    """
    check_positive_rates(birth_rate, death_rate)
    check_terminal_time(terminal_time)

    rng = seeded_rng(random_seed)

    spine_component = _generate_spine_component(
        birth_rate=float(birth_rate),
        death_rate=float(death_rate),
        terminal_time=float(terminal_time),
        rng=rng,
    )
    non_spine_component = _generate_non_spine_component(
        birth_rate=float(birth_rate),
        death_rate=float(death_rate),
        terminal_time=float(terminal_time),
        rng=rng,
    )
    contour_process_list = _build_contour_process_list(
        spine_component=spine_component,
        non_spine_component=non_spine_component,
    )
    validate_contour_process_list(contour_process_list)
    return contour_process_list_to_tree(
        contour_process_list,
        birth_rate=float(birth_rate),
        death_rate=float(death_rate),
    )


def simulate_linear_rate_contour_process(
    birth_rate: ConstantRateLike | LinearRateLike,
    death_rate: ConstantRateLike | LinearRateLike,
    random_seed: int | None = None,
) -> SimulatedTree:
    r"""Simulate a survival-conditioned tree with time-varying linear rates.

    The construction is that of
    :func:`simulate_birth_death_process_from_contour_process`, with the
    exponential jumps replaced by hazard inversion at the absolute time of the
    jump.
    A downward jump from level $x$ goes to the time $s < x$ at which
    $\int_s^x \lambda = E$, and an upward jump from $y$ to the time $s > y$ at
    which $\int_y^s \mu = E$, where $E \sim \mathrm{Exp}(1)$; each is solved
    exactly as a quadratic within every linear segment.
    Upward jumps are capped at $T$, and the spine excursion resets when one
    reaches $T$, as in the constant-rate case.

    Parameters
    ----------
    birth_rate, death_rate
        Per-individual rates over ``[0, T]``, piecewise constant or piecewise
        linear, non-negative, with a common end time $T$ (the terminal time).
        The death rate must not be identically zero.
    random_seed
        Seed for reproducibility.

    Returns
    -------
    SimulatedTree
        ``birth_rate`` and ``death_rate`` are NaN; the rates are stored in
        ``birth_rate_function`` and ``death_rate_function``.

    Raises
    ------
    ValueError
        If the rates are invalid, have different end times, or the death rate
        is identically zero.

    """
    birth, death, terminal_time = resolve_rates(birth_rate, death_rate, None)
    if cumulative(death, terminal_time) <= 0.0:
        msg = "death_rate must not be identically zero"
        raise ValueError(msg)

    def down(x: float, rng: np.random.Generator) -> float:
        return backward_hazard_time(birth, x, float(rng.exponential()))

    def up(y: float, rng: np.random.Generator) -> float:
        return forward_hazard_time(death, y, float(rng.exponential()), terminal_time)

    rng = seeded_rng(random_seed)
    spine_component = _generate_component(
        down_step=down,
        up_step=up,
        terminal_time=terminal_time,
        rng=rng,
        reset_at_terminal=True,
    )
    non_spine_component = _generate_component(
        down_step=down,
        up_step=up,
        terminal_time=terminal_time,
        rng=rng,
        reset_at_terminal=False,
    )
    contour_process_list = _build_contour_process_list(
        spine_component=spine_component,
        non_spine_component=non_spine_component,
    )
    validate_contour_process_list(contour_process_list)
    return contour_process_list_to_tree(
        contour_process_list,
        birth_rate=float("nan"),
        death_rate=float("nan"),
        birth_rate_function=birth_rate,
        death_rate_function=death_rate,
    )
