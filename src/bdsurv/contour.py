"""Simulate a survival-conditioned birth-death tree via contour-process reconstruction.

This is the "exact" method: the contour (Lukasiewicz-style exploration path)
of a tree conditioned to survive to a terminal time ``T`` is generated
directly, then decoded back into an individual-level genealogy via a
two-stack reconstruction. See ``docs/contour_process.md`` for the full
derivation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from bdsurv.tree import SimulatedTree

MIN_CONTOUR_PROCESS_LENGTH = 3


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
    if birth_rate <= 0.0 or death_rate <= 0.0:
        msg = "birth_rate and death_rate must be positive"
        raise ValueError(msg)
    if terminal_time <= 0.0:
        msg = "terminal_time must be positive"
        raise ValueError(msg)

    component = [float(terminal_time)]
    while True:
        x = float(component[-1])
        downward = max(x - float(rng.exponential(scale=1.0 / birth_rate)), 0.0)
        component.append(downward)
        if np.isclose(downward, 0.0):
            return component

        y = float(component[-1])
        upward = min(
            y + float(rng.exponential(scale=1.0 / death_rate)),
            float(terminal_time),
        )
        component.append(upward)


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
    if birth_rate <= 0.0 or death_rate <= 0.0:
        msg = "birth_rate and death_rate must be positive"
        raise ValueError(msg)
    if terminal_time <= 0.0:
        msg = "terminal_time must be positive"
        raise ValueError(msg)

    component = [float(terminal_time)]
    while True:
        x = float(component[-1])
        downward = max(x - float(rng.exponential(scale=1.0 / birth_rate)), 0.0)
        component.append(downward)
        if np.isclose(downward, 0.0):
            return component

        y = float(component[-1])
        upward = min(
            y + float(rng.exponential(scale=1.0 / death_rate)),
            float(terminal_time),
        )
        if np.isclose(upward, terminal_time):
            component = [float(terminal_time)]
            continue
        component.append(upward)


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


def _validate_contour_process_list(contour_process_list: list[float]) -> None:
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
    if not np.isclose(contour_process_list[0], 0.0):
        msg = "contour_process_list must begin with 0"
        raise ValueError(msg)
    if not np.isclose(contour_process_list[-1], 0.0):
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


def _contour_process_list_to_tree(
    contour_process_list: list[float],
    *,
    birth_rate: float,
    death_rate: float,
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
    _validate_contour_process_list(contour_process_list)

    values = [float(value) for value in contour_process_list]
    terminal_time = float(max(values))

    records: list[dict[str, Any]] = [
        {
            "individual_id": 0,
            "parent_id": None,
            "creation_time": values[0],
            "death_time": float("nan"),
            "is_spine": False,
        },
    ]

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
            if np.isclose(current_value, terminal_time):
                # Individual that 'dies' at terminal time is considered alive
                records[individual_number_stack[-1]]["death_time"] = float(
                    "nan",
                )
                if spine_individual_id is None:
                    spine_individual_id = individual_number_stack[-1]
            else:
                records[individual_number_stack[-1]]["death_time"] = current_value
            continue

        # Even indices are birth times: pop ancestors with birth times >=
        # current birth time
        while birth_times_stack and birth_times_stack[-1] >= current_value:
            birth_times_stack.pop()
            individual_number_stack.pop()

        if not individual_number_stack:
            raise ValueError("contour_process_list is not valid for reconstruction")

        records.append(
            {
                "individual_id": next_individual_id,
                "parent_id": individual_number_stack[-1],
                "creation_time": current_value,
                "death_time": float("nan"),
                "is_spine": False,
            },
        )

        birth_times_stack.append(current_value)
        individual_number_stack.append(next_individual_id)
        next_individual_id += 1

    records_df = pd.DataFrame(records)

    # Records with death_time == terminal_time are considered alive.
    records_df["survived"] = records_df["death_time"].isna()
    records_df["age"] = np.where(
        records_df["survived"],
        terminal_time - records_df["creation_time"],
        np.nan,
    )
    records_df["lifespan"] = np.where(
        ~records_df["survived"],
        records_df["death_time"] - records_df["creation_time"],
        np.nan,
    )

    # Mark the spine: the individual last-born when T was first reached.
    records_df["is_spine"] = False
    if spine_individual_id is not None:
        records_df.loc[spine_individual_id, "is_spine"] = True

    return SimulatedTree(
        records=records_df,
        birth_rate=float(birth_rate),
        death_rate=float(death_rate),
        terminal_time=terminal_time,
    )


def simulate_birth_death_process_from_contour_process(
    birth_rate: float,
    death_rate: float,
    terminal_time: float,
    random_seed: int | None = None,
) -> SimulatedTree:
    """Simulate a birth-death tree using contour-process reconstruction."""
    if birth_rate <= 0.0 or death_rate <= 0.0:
        msg = "birth_rate and death_rate must be positive"
        raise ValueError(msg)
    if terminal_time <= 0.0:
        msg = "terminal_time must be positive"
        raise ValueError(msg)

    base_seed_rng = np.random.default_rng(random_seed)
    rng = (
        np.random.default_rng(int(base_seed_rng.integers(0, 2**31 - 1)))
        if random_seed is not None
        else np.random.default_rng()
    )

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
    _validate_contour_process_list(contour_process_list)
    return _contour_process_list_to_tree(
        contour_process_list,
        birth_rate=float(birth_rate),
        death_rate=float(death_rate),
    )
