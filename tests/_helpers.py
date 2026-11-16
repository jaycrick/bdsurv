"""Accessors over ``SimulatedTree.records`` shared by the simulator tests."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from bdsurv.tree import SimulatedTree


def surviving_records(process: SimulatedTree) -> pd.DataFrame:
    """Rows of individuals alive at the terminal time."""
    return process.records[process.records["survived"]].copy()


def extinct_records(process: SimulatedTree) -> pd.DataFrame:
    """Rows of individuals that died before the terminal time."""
    return process.records[~process.records["survived"]].copy()


def creation_times(process: SimulatedTree) -> list[float]:
    """Creation times of the surviving individuals."""
    return surviving_records(process)["creation_time"].astype(float).tolist()


def ages(process: SimulatedTree) -> np.ndarray:
    """Ages of the surviving individuals at the terminal time."""
    return surviving_records(process)["age"].to_numpy(dtype=np.float64)


def n_individuals(process: SimulatedTree) -> int:
    """Return the number of surviving individuals."""
    return len(surviving_records(process))


def _rows(records: pd.DataFrame, mask: pd.Series | np.ndarray, limit: int = 5) -> str:
    """Render the offending rows for an assertion message."""
    offending = records[np.asarray(mask, dtype=bool)]
    shown = offending.head(limit).to_string()
    extra = len(offending) - limit
    return f"{len(offending)} offending row(s):\n{shown}" + (
        f"\n... and {extra} more" if extra > 0 else ""
    )


def assert_valid_records(
    records: pd.DataFrame,
    terminal_time: float,
    *,
    expected_dtypes: Mapping[str, str],
) -> None:
    """Assert every invariant of a records table returned by a simulator.

    ``expected_dtypes`` maps each expected column name, in order, to its dtype
    string.  The schema is data because the simulators do not all produce the
    same one; the other invariants are common to all of them.
    """
    n = len(records)
    assert list(records.columns) == list(expected_dtypes), (
        f"column list: got {list(records.columns)}, expected {list(expected_dtypes)}"
    )
    got_dtypes = {c: str(records[c].dtype) for c in records.columns}
    assert got_dtypes == dict(expected_dtypes), (
        f"column dtypes: got {got_dtypes}, expected {dict(expected_dtypes)}"
    )

    ids = records["individual_id"].to_numpy()
    assert n > 0, "records are empty: every process has at least a root"
    assert np.array_equal(ids, np.arange(n)), (
        f"ids are 0..n-1: got {ids[:10].tolist()} for {n} rows"
    )

    parent = records["parent_id"].astype(float).to_numpy()
    has_parent = ~np.isnan(parent)
    assert not has_parent[0], (
        f"root has no parent: row 0 has parent_id {records['parent_id'].iloc[0]!r}"
    )
    assert has_parent[1:].all(), (
        f"every non-root has a parent: {_rows(records, ~has_parent & (ids > 0))}"
    )

    child = records[has_parent]
    parent_rows = records.iloc[parent[has_parent].astype(int)]
    creation = records["creation_time"].to_numpy(dtype=np.float64)
    # ``death_time`` is object dtype (all None) when nobody died, in some simulators.
    death = records["death_time"].to_numpy(dtype=np.float64)
    child_creation = child["creation_time"].to_numpy(dtype=np.float64)
    parent_creation = parent_rows["creation_time"].to_numpy(dtype=np.float64)
    parent_death = parent_rows["death_time"].to_numpy(dtype=np.float64)
    born_before = parent_creation <= child_creation
    assert born_before.all(), (
        f"parent was born before its child: {_rows(child, ~born_before)}"
    )
    alive_at_birth = np.isnan(parent_death) | (parent_death >= child_creation)
    assert alive_at_birth.all(), (
        f"parent was alive at the child's creation: {_rows(child, ~alive_at_birth)}"
    )

    dead = ~np.isnan(death)
    lifespan_positive = np.ones(n, dtype=bool)
    lifespan_positive[dead] = death[dead] > creation[dead]
    assert lifespan_positive.all(), (
        f"death_time > creation_time: {_rows(records, ~lifespan_positive)}"
    )

    survived = records["survived"].to_numpy(dtype=bool)
    assert np.array_equal(survived, ~dead), (
        f"survived iff death_time is NaN: {_rows(records, survived != ~dead)}"
    )

    age = records["age"].to_numpy()
    lifespan = records["lifespan"].to_numpy()
    age_expected = np.where(survived, terminal_time - creation, np.nan)
    age_ok = np.isclose(age, age_expected, rtol=1e-12, atol=1e-12, equal_nan=True)
    assert age_ok.all(), (
        f"age is T - creation_time for survivors and NaN otherwise: "
        f"{_rows(records, ~age_ok)}"
    )
    lifespan_expected = np.where(survived, np.nan, death - creation)
    lifespan_ok = np.isclose(
        lifespan, lifespan_expected, rtol=1e-12, atol=1e-12, equal_nan=True
    )
    assert lifespan_ok.all(), (
        f"lifespan is death_time - creation_time for the dead and NaN otherwise: "
        f"{_rows(records, ~lifespan_ok)}"
    )

    spine = records["is_spine"].to_numpy(dtype=bool)
    expected_spines = 1 if survived.any() else 0
    assert int(spine.sum()) == expected_spines, (
        f"exactly one spine iff there is a survivor: got {int(spine.sum())} "
        f"spine(s) with {int(survived.sum())} survivor(s): {_rows(records, spine)}"
    )
    assert survived[spine].all(), (
        f"the spine survived: {_rows(records, spine & ~survived)}"
    )
