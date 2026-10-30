"""Accessors over ``SimulatedProcess.records`` shared by the simulator tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bdsurv.tree import SimulatedProcess


def surviving_records(process: SimulatedProcess) -> pd.DataFrame:
    """Rows of individuals alive at the terminal time."""
    return process.records[process.records["survived"]].copy()


def extinct_records(process: SimulatedProcess) -> pd.DataFrame:
    """Rows of individuals that died before the terminal time."""
    return process.records[~process.records["survived"]].copy()


def creation_times(process: SimulatedProcess) -> list[float]:
    """Creation times of the surviving individuals."""
    return surviving_records(process)["creation_time"].astype(float).tolist()


def ages(process: SimulatedProcess) -> np.ndarray:
    """Ages of the surviving individuals at the terminal time."""
    return surviving_records(process)["age"].to_numpy(dtype=np.float64)


def n_individuals(process: SimulatedProcess) -> int:
    """Return the number of surviving individuals."""
    return len(surviving_records(process))
