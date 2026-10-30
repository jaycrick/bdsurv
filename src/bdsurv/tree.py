"""The result type shared by both survival-conditioned simulators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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
