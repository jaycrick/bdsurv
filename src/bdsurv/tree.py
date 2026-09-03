"""The result type shared by both survival-conditioned simulators."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


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

    """

    records: pd.DataFrame
    birth_rate: float
    death_rate: float
    terminal_time: float
