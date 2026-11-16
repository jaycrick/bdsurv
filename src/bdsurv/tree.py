"""The records table builder and the one result type of every simulator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike

    from bdsurv.rates import ConstantRateLike, LinearRateLike

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
    """A simulated birth-death tree, with the rates and horizon it was built with.

    Every simulator returns this type: the forward simulators, the contour
    simulators, multilevel splitting and the lifespan simulator.

    Parameters
    ----------
    records : pd.DataFrame
        One row per individual, with columns ``individual_id``, ``parent_id``,
        ``creation_time``, ``death_time``, ``is_spine``, ``survived``, ``age``,
        and ``lifespan``. Some simulators add further columns, such as
        ``birth_epoch``. The survival-conditioned simulators mark exactly one
        surviving individual as ``is_spine``.
    birth_rate : ConstantRateLike or LinearRateLike
        Per-individual birth rate used to generate the tree. A rate-like given
        by the caller is stored as given, a float as a one-epoch
        :class:`~bdsurv.rates.PiecewiseConstant`, and the per-epoch arrays of the
        lifespan simulator as a :class:`~bdsurv.rates.PiecewiseConstant`.
    death_rate : ConstantRateLike or LinearRateLike
        Per-individual death rate used to generate the tree; see ``birth_rate``.
    terminal_time : float
        The time horizon ``T`` at which the tree is observed.

    """

    records: pd.DataFrame
    birth_rate: ConstantRateLike | LinearRateLike
    death_rate: ConstantRateLike | LinearRateLike
    terminal_time: float
