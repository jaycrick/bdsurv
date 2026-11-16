"""The result type shared by both survival-conditioned simulators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike

    from bdsurv.linear_rates import ConstantRateLike, LinearRateLike

RECORD_COLUMNS = (
    "individual_id",
    "parent_id",
    "creation_time",
    "death_time",
    "is_spine",
    "survived",
    "age",
    "lifespan",
)


def build_records(
    creation_time: ArrayLike,
    parent_id: Sequence[int | float | None] | ArrayLike,
    death_time: Sequence[float | None] | ArrayLike,
    spine_id: int | None,
    terminal_time: float,
    **extra_columns: ArrayLike,
) -> pd.DataFrame:
    """Build the per-individual records table of a simulated tree or process.

    This is the only place the derived columns are computed.

    Parameters
    ----------
    creation_time : array-like of float
        Birth time of each individual. Individual ``i`` is row ``i``, so
        ``individual_id`` is ``0..n-1`` in input order.
    parent_id : array-like of int, float or None
        Parent of each individual; ``None`` or NaN for an individual with no
        parent.
    death_time : array-like of float or None
        Death time of each individual; ``None`` or NaN for one that survives.
    spine_id : int or None
        The ``individual_id`` marked ``is_spine``; ``None`` marks nobody.
    terminal_time : float
        End of the observation window, used for the ``age`` of survivors.
    **extra_columns : array-like
        Further columns appended after the standard ones, in the order given.

    Returns
    -------
    pd.DataFrame
        Columns ``individual_id`` (int64), ``parent_id`` (float64, NaN for no
        parent), ``creation_time`` (float64), ``death_time`` (float64, NaN for
        survivors), ``is_spine`` (bool), ``survived`` (bool, true when
        ``death_time`` is NaN), ``age`` (float64, ``terminal_time -
        creation_time`` for survivors, NaN otherwise), ``lifespan`` (float64,
        ``death_time - creation_time`` for the dead, NaN otherwise), then
        ``extra_columns``.

    """
    creation = np.asarray(creation_time, dtype=np.float64)
    parent = np.asarray(parent_id, dtype=np.float64)
    death = np.asarray(death_time, dtype=np.float64)
    ids = np.arange(len(creation), dtype=np.int64)
    survived = np.isnan(death)
    columns = {
        "individual_id": ids,
        "parent_id": parent,
        "creation_time": creation,
        "death_time": death,
        "is_spine": ids == (-1 if spine_id is None else spine_id),
        "survived": survived,
        "age": np.where(survived, terminal_time - creation, np.nan),
        "lifespan": np.where(survived, np.nan, death - creation),
    }
    return pd.DataFrame({**columns, **extra_columns})


@dataclass(frozen=True)
class SimulatedTree:
    """A single-epoch birth-death tree conditioned on survival to ``terminal_time``.

    Parameters
    ----------
    records : pd.DataFrame
        One row per individual, with columns ``individual_id``, ``parent_id``,
        ``creation_time``, ``death_time``, ``is_spine``, ``survived``, ``age``,
        and ``lifespan``. Exactly one surviving individual has ``is_spine``
        set to ``True``.
    birth_rate : float
        Per-individual birth rate used to generate the tree.
    death_rate : float
        Per-individual death rate used to generate the tree.
    terminal_time : float
        The time horizon ``T`` the tree is conditioned to survive to.
    birth_rate_function, death_rate_function : rate-like, optional
        For trees simulated with time-varying (piecewise-linear) rates, the
        rates used; ``birth_rate`` and ``death_rate`` are then NaN.

    """

    records: pd.DataFrame
    birth_rate: float
    death_rate: float
    terminal_time: float
    birth_rate_function: ConstantRateLike | LinearRateLike | None = None
    death_rate_function: ConstantRateLike | LinearRateLike | None = None


@dataclass(frozen=True)
class SimulatedProcess:
    """A birth-death process simulated forward over one or more rate epochs.

    Unlike :class:`SimulatedTree`, the process is not conditioned on survival,
    and the rates may change at ``epoch_end_times``.

    Parameters
    ----------
    records : pd.DataFrame
        One row per individual, with columns ``individual_id``, ``parent_id``,
        ``creation_time``, ``death_time``, ``is_spine``, ``survived``, ``age``,
        and ``lifespan``. Some simulators add further columns, such as
        ``birth_epoch``.
    birth_rates : np.ndarray or None
        Per-individual birth rate in each epoch, shape ``(n_epochs,)``.
        ``None`` when the rates are not piecewise constant, as for
        :func:`bdsurv.linear_rates.simulate_birth_death_linear`.
    death_rates : np.ndarray or None
        Per-individual death rate in each epoch; see ``birth_rates``.
    epoch_end_times : np.ndarray
        Strictly increasing end time of each epoch; the last entry is the
        terminal time.
    start_time : float
        Time at which the process starts.
    initial_population : int
        Number of individuals alive at ``start_time``.

    """

    records: pd.DataFrame
    epoch_end_times: np.ndarray
    birth_rates: np.ndarray | None = None
    death_rates: np.ndarray | None = None
    start_time: float = 0.0
    initial_population: int = 1
