"""Input validation of every simulator and level-grid entry point, as one table.

Each row is ``(entry point, bad arguments, message fragment, label)``: the entry
point is called with its valid base arguments updated by the bad ones, and must
raise ``ValueError`` whose message contains the fragment.
Fragments are short and stable (``positive``, ``increasing``, ``finite``): the
message names the argument and the requirement.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from bdsurv import (
    PiecewiseConstant,
    PiecewiseLinear,
    SplittingConfig,
    build_target_ranges,
    compute_level_replicates,
    default_max_restarts,
    sample_conditioned_survival_per_posterior,
    sample_conditioned_survival_process,
    select_level_times_for_band,
    simulate_birth_death,
    simulate_birth_death_lifespans,
    simulate_birth_death_lifespans_arrays,
    simulate_contour_given_ages,
    simulate_contour_given_population,
    simulate_contour_given_survival,
    simulate_survival_splitting,
)

NAN = float("nan")
INF = float("inf")


def _constant(values: list[float], ends: list[float]) -> PiecewiseConstant:
    return PiecewiseConstant(np.array(values), np.array(ends))


def _linear(starts: list[float], ends_values: list[float], ends: list[float]):
    return PiecewiseLinear(np.array(starts), np.array(ends_values), np.array(ends))


def _arr(*values: float) -> np.ndarray:
    return np.array(values, dtype=np.float64)


def _splitting(**kwargs: Any):
    return simulate_survival_splitting(SplittingConfig(**kwargs))


FORWARD_BASE: dict[str, Any] = {
    "birth_rates": _arr(1.0, 0.8),
    "death_rates": _arr(0.5, 0.6),
    "times": _arr(1.0, 2.0),
    "random_seed": 0,
}
FLOAT_BASE: dict[str, Any] = {
    "birth_rate": 0.9,
    "death_rate": 1.0,
    "terminal_time": 5.0,
    "random_seed": 0,
}
RATE_LIKE_BASE: dict[str, Any] = {
    "birth_rate": _constant([1.0, 0.8], [2.0, 4.0]),
    "death_rate": _linear([0.3, 0.5], [0.5, 0.9], [2.0, 4.0]),
    "random_seed": 0,
}
CONDITIONED_BASE: dict[str, Any] = {
    "birth_rate": 0.9,
    "death_rate": 1.0,
    "terminal_time": 5.0,
    "random_seed": 0,
}
SPLITTING_BASE: dict[str, Any] = {
    "birth_rate": 0.9,
    "death_rate": 1.0,
    "terminal_time": 5.0,
    "n_particles": 20,
    "random_seed": 0,
    "max_restarts": 5,
}
RATE_LIKE_SPLITTING_BASE: dict[str, Any] = {
    "birth_rate": _constant([1.0, 0.8], [2.0, 4.0]),
    "death_rate": _constant([0.5, 0.6], [2.0, 4.0]),
    "n_particles": 20,
    "random_seed": 0,
    "max_restarts": 5,
}
BAND_BASE: dict[str, Any] = {
    "birth_rate": 0.9,
    "death_rate": 1.0,
    "terminal_time": 5.0,
}
REPLICATES_BASE: dict[str, Any] = {
    "birth_rate": 0.9,
    "death_rate": 1.0,
    "level_times": [0.0, 2.0, 5.0],
}
RESTARTS_BASE: dict[str, Any] = {**REPLICATES_BASE, "replicates": [2, 2]}
# The helpers with rate-like rates; their end time is 4.
RATE_LIKE_RATES: dict[str, Any] = {
    "birth_rate": _constant([1.0, 0.8], [2.0, 4.0]),
    "death_rate": _linear([0.3, 0.5], [0.5, 0.9], [2.0, 4.0]),
}
RATE_LIKE_BAND_BASE: dict[str, Any] = dict(RATE_LIKE_RATES)
RATE_LIKE_REPLICATES_BASE: dict[str, Any] = {
    **RATE_LIKE_RATES,
    "level_times": [0.0, 2.0, 4.0],
}
RATE_LIKE_RESTARTS_BASE: dict[str, Any] = {
    **RATE_LIKE_REPLICATES_BASE,
    "replicates": [2, 2],
}
ERA_BASE: dict[str, Any] = {
    "birth_rates": [0.6, 0.5],
    "death_rates": [0.2, 0.3],
    "epoch_end_times": [3.0, 6.0],
    "target_ranges": np.array([[1, 6], [0, 2**31 - 1]]),
    "split_factor": 2,
    "n_initial": 4,
    "random_seed": 0,
    "max_population": 500,
}
POSTERIOR_BASE: dict[str, Any] = {
    "posterior_birth_rates": np.array([[0.6, 0.5], [0.6, 0.5]]),
    "posterior_death_rates": np.array([[0.2, 0.3], [0.2, 0.3]]),
    "epoch_end_times": [3.0, 6.0],
    "target_ranges": np.array([[1, 6], [0, 2**31 - 1]]),
    "split_factor": 2,
    "n_initial": 4,
    "random_seed": 0,
    "max_population": 500,
}

# entry point name -> (callable, valid base arguments); "(float)" and "(rate-like)"
# name the same function called with float rates and with rate objects.
ENTRY_POINTS: dict[str, tuple[Callable[..., Any], dict[str, Any]]] = {
    "simulate_birth_death(float)": (simulate_birth_death, FLOAT_BASE),
    "simulate_birth_death(rate-like)": (simulate_birth_death, RATE_LIKE_BASE),
    "simulate_birth_death_lifespans": (simulate_birth_death_lifespans, FORWARD_BASE),
    "simulate_birth_death_lifespans_arrays": (
        simulate_birth_death_lifespans_arrays,
        FORWARD_BASE,
    ),
    "simulate_contour_given_survival(float)": (
        simulate_contour_given_survival,
        FLOAT_BASE,
    ),
    "simulate_contour_given_survival(rate-like)": (
        simulate_contour_given_survival,
        RATE_LIKE_BASE,
    ),
    "simulate_contour_given_population": (
        simulate_contour_given_population,
        {**CONDITIONED_BASE, "n_alive": 2},
    ),
    "simulate_contour_given_ages": (
        simulate_contour_given_ages,
        {**CONDITIONED_BASE, "ages": [1.0, 2.0]},
    ),
    "simulate_survival_splitting(float)": (_splitting, SPLITTING_BASE),
    "simulate_survival_splitting(rate-like)": (
        _splitting,
        RATE_LIKE_SPLITTING_BASE,
    ),
    "select_level_times_for_band(float)": (select_level_times_for_band, BAND_BASE),
    "select_level_times_for_band(rate-like)": (
        select_level_times_for_band,
        RATE_LIKE_BAND_BASE,
    ),
    "compute_level_replicates(float)": (compute_level_replicates, REPLICATES_BASE),
    "compute_level_replicates(rate-like)": (
        compute_level_replicates,
        RATE_LIKE_REPLICATES_BASE,
    ),
    "default_max_restarts(float)": (default_max_restarts, RESTARTS_BASE),
    "default_max_restarts(rate-like)": (
        default_max_restarts,
        RATE_LIKE_RESTARTS_BASE,
    ),
    "build_target_ranges": (build_target_ranges, {"observed_counts": [5, 3, 0]}),
    "sample_conditioned_survival_process": (
        sample_conditioned_survival_process,
        ERA_BASE,
    ),
    "sample_conditioned_survival_per_posterior": (
        sample_conditioned_survival_per_posterior,
        POSTERIOR_BASE,
    ),
}

Row = tuple[str, dict[str, Any], str, str]


def _rows(entry: str, fragment: str, bad: dict[str, list[Any]]) -> list[Row]:
    """Rows ``(entry, {arg: value}, fragment, label)`` for each bad value."""
    return [
        (entry, {arg: value}, fragment, f"{arg}={_show(value)}")
        for arg, values in bad.items()
        for value in values
    ]


def _show(value: Any) -> str:
    """Render a bad value for a test id."""
    if isinstance(value, np.ndarray):
        text = str(value.tolist())
    elif isinstance(value, PiecewiseConstant):
        values = np.asarray(value.values).tolist()
        ends = np.asarray(value.epoch_end_times).tolist()
        text = f"const{values}@{ends}"
    elif isinstance(value, PiecewiseLinear):
        starts = np.asarray(value.start_values).tolist()
        stops = np.asarray(value.end_values).tolist()
        ends = np.asarray(value.epoch_end_times).tolist()
        text = f"line{starts}->{stops}@{ends}"
    else:
        text = str(value)
    return text.replace(" ", "")


def _forward_rows(entry: str) -> list[Row]:
    """Rows of the two epoch-array simulators, which need strictly positive rates."""
    rows = _rows(
        entry,
        "positive",
        {
            "birth_rates": [
                _arr(0.0, 0.8),
                _arr(-1.0, 0.8),
                _arr(NAN, 0.8),
                _arr(INF, 0.8),
            ],
            "death_rates": [
                _arr(0.0, 0.6),
                _arr(-1.0, 0.6),
                _arr(NAN, 0.6),
                _arr(INF, 0.6),
            ],
            "times": [_arr(-1.0, 1.0), _arr(0.0, 1.0)],
        },
    )
    rows += _rows(
        entry,
        "increasing",
        {"times": [_arr(2.0, 1.0), _arr(1.0, 1.0), _arr(1.0, NAN), _arr(1.0, INF)]},
    )
    rows += _rows(
        entry,
        "same length",
        {
            "birth_rates": [_arr(1.0)],
            "death_rates": [_arr(0.5, 0.6, 0.7)],
            "times": [_arr(1.0, 2.0, 3.0)],
        },
    )
    return rows


ROWS: list[Row] = []
for _entry in (
    "simulate_birth_death_lifespans",
    "simulate_birth_death_lifespans_arrays",
):
    ROWS += _forward_rows(_entry)
ROWS += [
    (
        "simulate_birth_death_lifespans",
        {"ensure_non_extinct": True, "max_retries": -1},
        "non-negative",
        "max_retries=-1",
    ),
    *_rows(
        "simulate_birth_death_lifespans",
        "at least 1",
        {"max_population": [0, -5]},
    ),
    *_rows(
        "simulate_birth_death_lifespans_arrays",
        "at least 1",
        {"max_population": [0, -5]},
    ),
]


def _float_rate_rows(entry: str, *, terminal_time_required: bool = True) -> list[Row]:
    """Rows for the entry points called with float rates and ``terminal_time``.

    A float rate of zero is valid, so none of these rows has one.
    """
    rows = _rows(entry, "non-negative", {"birth_rate": [-1.0], "death_rate": [-1.0]})
    rows += _rows(entry, "finite", {"birth_rate": [NAN, INF], "death_rate": [NAN, INF]})
    rows += _rows(entry, "positive", {"terminal_time": [0.0, -1.0, NAN, INF]})
    if terminal_time_required:
        rows += _rows(entry, "terminal_time is required", {"terminal_time": [None]})
    return rows


def _rate_like_rows(entry: str) -> list[Row]:
    """Rows for the entry points called with rate-likes for both rates."""
    rows = _rows(
        entry,
        "non-negative",
        {
            "birth_rate": [_constant([-1.0, 0.8], [2.0, 4.0])],
            "death_rate": [_linear([-0.5, 0.5], [0.5, 0.9], [2.0, 4.0])],
        },
    )
    rows += _rows(
        entry,
        "finite",
        {
            "birth_rate": [
                _constant([NAN, 0.8], [2.0, 4.0]),
                _constant([INF, 0.8], [2.0, 4.0]),
            ],
            "death_rate": [
                _linear([NAN, 0.5], [0.5, 0.9], [2.0, 4.0]),
                _linear([0.3, 0.5], [INF, 0.9], [2.0, 4.0]),
            ],
        },
    )
    rows += _rows(
        entry,
        "increasing",
        {
            "birth_rate": [
                _constant([1.0, 0.8], [6.0, 4.0]),
                _constant([1.0, 0.8], [2.0, NAN]),
                _constant([1.0, 0.8], [2.0, INF]),
            ],
            "death_rate": [_linear([0.3, 0.5], [0.5, 0.9], [6.0, 4.0])],
        },
    )
    rows += _rows(
        entry,
        "positive",
        {
            "birth_rate": [
                _constant([1.0, 0.8], [0.0, 4.0]),
                _constant([1.0, 0.8], [-1.0, 4.0]),
            ]
        },
    )
    rows += _rows(
        entry,
        "same end time",
        {"death_rate": [_linear([0.3, 0.5], [0.5, 0.9], [2.0, 5.0])]},
    )
    rows += _rows(
        entry,
        "same shape",
        {"birth_rate": [_constant([1.0], [2.0, 4.0])]},
    )
    return rows


ROWS += _float_rate_rows("simulate_birth_death(float)")
ROWS += _rate_like_rows("simulate_birth_death(rate-like)")
for _entry in ("simulate_birth_death(float)", "simulate_birth_death(rate-like)"):
    ROWS.append(
        (
            _entry,
            {"ensure_non_extinct": True, "max_retries": -1},
            "non-negative",
            "max_retries=-1",
        )
    )
ROWS.append(
    (
        "simulate_birth_death(rate-like)",
        {"terminal_time": 5.0},
        "terminal_time must equal",
        "terminal_time-differs-from-rates-end-time",
    )
)

# Survival-conditioned contour: a death rate that is identically zero is rejected.
ROWS += _float_rate_rows("simulate_contour_given_survival(float)")
ROWS += _rate_like_rows("simulate_contour_given_survival(rate-like)")
ROWS += [
    (
        "simulate_contour_given_survival(float)",
        {"death_rate": 0.0},
        "identically zero",
        "death_rate=0.0",
    ),
    (
        "simulate_contour_given_survival(rate-like)",
        {"death_rate": _constant([0.0, 0.0], [2.0, 4.0])},
        "identically zero",
        "death_rate=identically-zero",
    ),
    (
        "simulate_contour_given_survival(rate-like)",
        {"terminal_time": 5.0},
        "terminal_time must equal",
        "terminal_time-differs-from-rates-end-time",
    ),
]

# Conditioned contours.
for _entry in ("simulate_contour_given_population", "simulate_contour_given_ages"):
    ROWS += _rows(
        _entry,
        "non-negative",
        {"birth_rate": [-1.0], "death_rate": [-1.0]},
    )
    ROWS += _rows(
        _entry,
        "finite",
        {"birth_rate": [NAN, INF], "death_rate": [NAN, INF]},
    )
    ROWS += _rows(_entry, "positive", {"terminal_time": [0.0, -1.0, NAN, INF]})
    ROWS += _rows(_entry, "terminal_time is required", {"terminal_time": [None]})
    ROWS.append(
        (
            _entry,
            {
                "birth_rate": _constant([1.0], [5.0]),
                "death_rate": _constant([1.0], [6.0]),
                "terminal_time": None,
            },
            "end time",
            "rates-with-different-end-times",
        )
    )
    ROWS.append(
        (
            _entry,
            {"birth_rate": _constant([1.0], [4.0]), "terminal_time": 5.0},
            "end time",
            "terminal_time-differs-from-rates-end-time",
        )
    )
    ROWS.append(
        (
            _entry,
            {
                "birth_rate": _constant([1.0, 0.8], [6.0, 4.0]),
                "terminal_time": None,
            },
            "increasing",
            "birth_rate-epoch-ends=[6.0,4.0]",
        )
    )
ROWS += _rows("simulate_contour_given_population", "at least 1", {"n_alive": [0, -1]})
ROWS += _rows(
    "simulate_contour_given_ages",
    "empty",
    {"ages": [[]]},
)
ROWS += _rows(
    "simulate_contour_given_ages",
    "(0, T]",
    {"ages": [[0.0, 1.0], [-1.0, 1.0], [6.0], [NAN]]},
)

# Splitting.
ROWS += _float_rate_rows("simulate_survival_splitting(float)")
ROWS += _rate_like_rows("simulate_survival_splitting(rate-like)")
ROWS.append(
    (
        "simulate_survival_splitting(rate-like)",
        {"terminal_time": 5.0},
        "terminal_time must equal",
        "terminal_time-differs-from-rates-end-time",
    )
)
for _entry in (
    "simulate_survival_splitting(float)",
    "simulate_survival_splitting(rate-like)",
):
    ROWS += _rows(_entry, "n_particles must be positive", {"n_particles": [0, -1]})
    ROWS += _rows(_entry, "max_restarts", {"max_restarts": [0, -1]})
ROWS += _rows(
    "simulate_survival_splitting(float)",
    "n_levels must be positive",
    {"n_levels": [0, -1]},
)
ROWS += _rows(
    "simulate_survival_splitting(float)",
    "increasing",
    {"level_times": [[0.0, 3.0, 2.0, 5.0]]},
)
ROWS.append(
    (
        "simulate_survival_splitting(float)",
        {"level_times": [0.0, 2.0, 5.0], "replicates": [2]},
        "one entry per level",
        "replicates-length-differs-from-intervals",
    )
)
ROWS += _rows(
    "simulate_survival_splitting(float)",
    "requires level_times",
    {"replicates": [[2]]},
)

# Level grid, replicates and restarts.
ROWS += _float_rate_rows("select_level_times_for_band(float)")
ROWS += _rate_like_rows("select_level_times_for_band(rate-like)")
for _entry in ("compute_level_replicates", "default_max_restarts"):
    ROWS += _float_rate_rows(f"{_entry}(float)", terminal_time_required=False)
    ROWS += _rate_like_rows(f"{_entry}(rate-like)")
ROWS += _rows(
    "select_level_times_for_band(float)",
    "increasing",
    {"initial_level_times": [[0.0, 3.0, 2.0, 5.0]]},
)
ROWS += _rows(
    "compute_level_replicates(float)",
    "increasing",
    {"level_times": [[0.0, 3.0, 2.0], [0.0, 2.0, 2.0]]},
)
for _entry in ("compute_level_replicates", "default_max_restarts"):
    ROWS += _rows(
        f"{_entry}(rate-like)",
        "must not exceed T",
        {"level_times": [[0.0, 2.0, 9.0], [0.0, 4.5]]},
    )
    ROWS += _rows(
        f"{_entry}(float)",
        "must not exceed T",
        {"terminal_time": [3.0]},
    )
ROWS += _rows(
    "default_max_restarts(float)",
    "increasing",
    {"level_times": [[0.0, 3.0, 2.0]]},
)
ROWS += _rows(
    "default_max_restarts(float)",
    "one entry per level",
    {"replicates": [[2], [2, 2, 2]]},
)
ROWS += _rows(
    "default_max_restarts(float)",
    "replicates",
    {"replicates": [[0, 2], [-1, 2]]},
)
ROWS += _rows(
    "default_max_restarts(float)",
    "target_success",
    {"target_success": [0.0, 1.0, -0.5, NAN]},
)
ROWS += _rows(
    "build_target_ranges",
    "non-negative",
    {"observed_counts": [[-1, 2, 0]], "tolerance": [-1]},
)
ROWS += _rows("build_target_ranges", "at least", {"observed_counts": [[]]})

# Era-conditioned sampler.
ROWS += _rows(
    "sample_conditioned_survival_process",
    "positive",
    {
        "birth_rates": [[0.0, 0.5], [-1.0, 0.5], [NAN, 0.5], [INF, 0.5]],
        "death_rates": [[0.0, 0.3], [-1.0, 0.3], [NAN, 0.3], [INF, 0.3]],
    },
)
ROWS += _rows(
    "sample_conditioned_survival_process",
    "increasing",
    {"epoch_end_times": [[6.0, 3.0], [3.0, 3.0]]},
)
ROWS += _rows(
    "sample_conditioned_survival_process",
    "positive",
    {"epoch_end_times": [[0.0, 3.0], [-1.0, 3.0]]},
)
ROWS += _rows(
    "sample_conditioned_survival_process",
    "same length",
    {"birth_rates": [[0.6]], "epoch_end_times": [[3.0, 6.0, 9.0]]},
)
ROWS += _rows(
    "sample_conditioned_survival_process",
    "positive",
    {"split_factor": [0, -1], "n_initial": [0, -1], "n_samples": [0]},
)
ROWS += _rows(
    "sample_conditioned_survival_process",
    "target_ranges",
    {
        "target_ranges": [
            np.array([[1, 6]]),
            np.array([[1, 6], [0, 9], [0, 9]]),
        ]
    },
)
ROWS += _rows(
    "sample_conditioned_survival_per_posterior",
    "2-D",
    {
        "posterior_birth_rates": [np.array([0.6, 0.5])],
        "posterior_death_rates": [np.array([0.2, 0.3])],
    },
)
ROWS += _rows(
    "sample_conditioned_survival_per_posterior",
    "shape",
    {"posterior_death_rates": [np.array([[0.2, 0.3]])]},
)
ROWS += _rows(
    "sample_conditioned_survival_per_posterior",
    "positive",
    {
        "posterior_birth_rates": [np.array([[-1.0, 0.5], [0.6, 0.5]])],
        "split_factor": [0],
        "n_initial": [0],
    },
)
ROWS += _rows(
    "sample_conditioned_survival_per_posterior",
    "target_ranges",
    {"target_ranges": [np.array([[1, 6]])]},
)


def _case_id(row: Row) -> str:
    return f"{row[0]}-{row[3]}"


def _params() -> list[Any]:
    return [pytest.param(row[0], row[1], row[2], id=_case_id(row)) for row in ROWS]


def call_entry_point(entry: str, bad: dict[str, Any]) -> Any:
    """Call ``entry`` with its valid base arguments updated by ``bad``."""
    function, base = ENTRY_POINTS[entry]
    return function(**{**base, **bad})


@pytest.mark.parametrize(("entry", "bad", "fragment"), _params())
def test_bad_input_is_rejected(entry: str, bad: dict[str, Any], fragment: str) -> None:
    """The entry point raises ``ValueError`` naming the problem."""
    with pytest.raises(ValueError, match=re.escape(fragment)):
        call_entry_point(entry, bad)


def test_ids_are_unique() -> None:
    """Parameter ids must be unique, or a failure could not name its row."""
    ids = [_case_id(row) for row in ROWS]
    assert len(ids) == len(set(ids)), "duplicate case ids"
