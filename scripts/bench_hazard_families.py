r"""Benchmark the simulation cost of matched constant, linear and sigmoid rates.

Every family rises from about 0 to 1 around ``c = 5`` with midpoint slope
``kappa / 4`` (``kappa = 1``), on the window ``[0, T]`` with ``T = 1000``:

=============  ==================================================  ===============
family         object                                              path
=============  ==================================================  ===============
constant       ``GompertzHazard(1, 0)``                            hazard-direct
constant       ``as_rate(PiecewiseConstant([1], [T]))``            rate
linear         ``QuadraticHazard(0, 1 / T_lin, 0)``, ``T_lin=2c``  hazard-brent
linear         ``as_rate(PiecewiseLinear([0, 1], [1, 1],           rate
               [T_lin, T]))``
pw-linear      ``as_rate(PiecewiseLinear(ramp))``                  rate
pw-linear      ``PiecewiseHazard.from_polynomials(ramp)``          hazard-piecewise
logistic       ``LogisticHazard(1, 1, 5)``                         hazard-direct
floor-logistic ``LogisticHazard.with_floor(.2, .8, 1, 5)``         split
floor-logistic ``LogisticHazard.with_floor(.2, .8, -1, 5)``        split
-fall
quadratic      ``QuadraticHazard(.1, .05, .01)``                   hazard-brent,
                                                                   closed-forced,
                                                                   closed-gated
quadratic-c0   ``QuadraticHazard(.1, .1, 0)``                      (same three)
cubic          ``CubicHazard(.1, .02, .005, .0005)``               (same three)
cubic-d0       ``CubicHazard(.1, .05, .01, 0)``                    (same three)
=============  ==================================================  ===============

The polynomial families compare three inversions of the same function:
``hazard-brent`` is the Brent path (the polynomial wrapped in ``_BrentOnly``),
``closed-forced`` is ``bdsurv._polyroots.forward_offsets`` with ``method="closed"``
(Cardano/Ferrari plus two Newton steps in every element, no gate) and
``closed-gated`` is the default gated closed form (Newton from the quadratic
truncation when the nonlinearity is small, else Cardano/Ferrari; see
``bdsurv._polyroots``).
All parameters are non-negative on ``[0, inf)``.

``n_alive`` axis
    The simulators invert with ``e = E / n`` for ``n`` individuals alive.
    The polynomial families (all three paths) are run with ``n_alive`` in
    {1, 100, 10000}: ``invert``, ``individual`` and ``full_batch`` use ``E / n_alive``.
    ``full_batch`` of a closed path (and of any family with ``n_alive > 1``) draws
    ``E`` with ``root_rng(SEED, "dynamics")``, divides by ``n_alive`` and inverts
    with the path's method, capped at ``T`` (``inf`` where there is no event);
    ``individual`` loops ``forward_time`` (gated) or ``_polyroots.forward_time``
    (forced).
    Other families have ``n_alive = 1``.  Only ``invert``, ``individual`` and
    ``full_batch`` are run for ``n_alive > 1`` and for the closed paths (the other
    measures do not depend on either).
    The closed paths add the measures ``regime_newton``, ``regime_algebraic``,
    ``regime_brent`` (counts of elements per regime, from one untimed call of the
    ``invert`` operation on the cell) and ``fallback_fraction`` (Brent share).
    End-to-end timings exist only for ``n_alive = 1`` and not for the closed
    paths (the simulators use the default Brent inversion).

The ``floor-logistic`` families draw two exponentials per event and invert each
part in closed form (the superposition sampler): floor 0.2 plus ``0.8 sigma(t - 5)``.
``floor-logistic-fall`` is the falling variant (starts at 0.2 + 0.8 sigma(5)
and decays to the floor 0.2).

The Hazard linear family is the unbounded line ``t / T_lin`` (it is 100 at
``T``); the Rate linear family is the same line capped at 1 after ``T_lin``.
The pw-linear ramp is 0 on ``[0, c-2]``, rises linearly to 1 on ``[c-2, c+2]``
and stays at 1.

Start times ``t0`` (arrays drawn once, with fixed seeds, outside every timer):
S1 all 0; S2 ``U(0, 2c)``; S3 ``U(c, c+2)``; S4 ``c + 20``.
Batch sizes 1e3, 1e5, 1e6.

Measures (``seconds`` holds the median-able time of one run, except where noted):

``hazard``, ``cumulative``
    ``h(t0)`` and ``H(t0)`` vectorised over the batch (Rates: ``_hazard.value`` and
    ``_hazard.cumulative``).
``between_scalar``
    1e4 scalar calls of ``cumulative_between(t0_i, t0_i + 1)`` (Rates:
    ``H(t1) - H(t0)``); ``seconds`` is the per-call time (total / 1e4) and ``n`` is 1e4.
``invert``
    Hazards: ``forward_offsets(t0, E)`` on pre-drawn ``E`` (uncapped).
    ``split`` path: ``forward_offsets_split(t0, E)`` with pre-drawn ``E`` of shape
    ``(n_draws, n)``.
    Rates: a loop of ``forward_hazard_time(rate, t0_i, e_i, T)``.
``full_batch``
    Hazards: ``sample_residual_lifetimes(h, t0, random_seed, end_time=T)``; for
    non-direct hazards this already loops ``forward_time`` with a cap, so it differs
    from the uncapped ``invert``.
    Rates: ``root_rng(SEED, "dynamics").exponential(size=n)`` plus the loop.
``individual``
    A Python loop of ``h.forward_time(t0_i, e_i, T)`` (Rates: the same loop as
    ``invert``), which is how the simulators call it.
    ``split`` path: a loop of ``h.forward_time_split(t0_i, es_i, T)``.
``end_to_end``
    ``simulate_birth_death(birth_rate=<family>, death_rate=0.9, terminal_time=10)``,
    with the family rebuilt on ``[0, 10]`` (same function; the simulator requires the
    rates' end time to equal ``terminal_time``), one row per seed (``repeat`` = seed,
    0..19); ``n`` is the number of seeds.

Non-timing measures put their value in the ``seconds`` column; the ``unit`` column
says what it is:

``evals_per_root``
    Mean number of ``cumulative_between`` calls per root (``unit`` ``count``), over
    the first 1000 draws of the cell with a counting wrapper (capped at ``T``).
    0 for closed-form and ``split`` hazards (note ``split closed form``); NaN for
    Rates and for the piecewise hazard, whose walk calls its pieces.
``no_event_fraction``
    Fraction of the ``full_batch`` draws with no event before ``T`` (``unit``
    ``fraction``).
``mean_residual``
    Mean residual lifetime of the ``full_batch`` draws that had an event (``unit``
    ``time``); in scenario S1 this is the family's mean event time.

CSV columns: ``family, path, scenario, n, measure, repeat, seconds`` plus ``unit``,
``note`` (non-empty when a cell used a reduced number of repeats or was cut short)
and ``n_alive`` (1 except on the ``n_alive`` axis).
One warm-up call on the first 1000 elements is excluded from every timed cell.
Runtime trim: loop and Brent cells (Rates, ``hazard-brent``, ``hazard-piecewise``)
use 3 repeats at n = 1e5; at n = 1e6 they run only scenario S2 with ``full_batch``
and one repeat (note ``trimmed``), since their time is linear in n.
Direct and ``split`` cells keep the full grid.
Note that ``sample_residual_lifetimes`` with a finite ``end_time`` loops
``forward_time_split`` per element for a ``split`` hazard, so its ``full_batch``
is a Python loop even though the hazard is closed form.
``--max-cell-seconds`` caps a cell: if the first repeat exceeds it the rest are
skipped; at n = 1e6 a cell whose projection from the previous size exceeds it runs
once.

Usage
-----
    uv run python scripts/bench_hazard_families.py --quick --out /tmp/bench
    uv run python scripts/bench_hazard_families.py --summary
"""

import argparse
import json
import math
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

# Loaded by path from a test, ``scripts/`` is not on the import path yet.
_SCRIPTS = str(Path(__file__).resolve().parent)
if _SCRIPTS not in sys.path:
    sys.path.append(_SCRIPTS)

import _bench_common as common
import numpy as np
import pandas as pd

from bdsurv import (
    CubicHazard,
    GompertzHazard,
    LogisticHazard,
    PiecewiseConstant,
    PiecewiseHazard,
    PiecewiseLinear,
    QuadraticHazard,
    _polyroots,
    simulate_birth_death,
)
from bdsurv._hazard import cumulative, forward_hazard_time, value
from bdsurv._polyroots import RegimeCounts
from bdsurv.hazards import Hazard, sample_residual_lifetimes
from bdsurv.rates import as_rate

from bdsurv._seeding import root_rng  # isort: skip

KAPPA = 1.0
C = 5.0
T = 1000.0
T_LIN = 2.0 * C
SEED = 20261126
SIZES = (1_000, 100_000, 1_000_000)
REPEATS = 7
REPEATS_BIG_LOOP = 3
N_SCALAR = 10_000
N_COUNT = 1_000
N_WARM = 1_000
N_E2E_SEEDS = 20
E2E_TERMINAL = 10.0
DEFAULT_MAX_CELL_SECONDS = 600.0
DEFAULT_OUT = common.ROOT / "benchmarks" / "data"
CSV_NAME = "hazard_families.csv"
ENV_NAME = "hazard_families_environment.json"
COLUMNS = ["family", "path", "scenario", "n", "measure", "repeat", "seconds"]
ALL_COLUMNS = [*COLUMNS, "unit", "note", "n_alive"]
N_ALIVE_AXIS = (1, 100, 10_000)
POLY_NAMES = ("quadratic", "quadratic-c0", "cubic", "cubic-d0")
CLOSED_METHODS = {"closed-forced": "closed", "closed-gated": "gated"}
ALIVE_ONLY_MEASURES = ("invert", "individual", "full_batch")
MEASURES = (
    "hazard",
    "cumulative",
    "between_scalar",
    "invert",
    "full_batch",
    "individual",
)
QUICK_MEASURES = ("hazard", "cumulative", "between_scalar", "invert", "full_batch")
SCENARIOS = ("S1", "S2", "S3", "S4")
QUICK_SCENARIOS = ("S1", "S2")
QUICK_SIZES = (1_000,)
QUICK_REPEATS = 2
QUICK_E2E_SEEDS = 3


@dataclass(frozen=True)
class Family:
    """One benchmarked rate: its object, interface path and end-to-end input."""

    name: str
    path: str
    obj: Any  # a Hazard, or a resolved ``Rate``
    e2e: Any  # what ``simulate_birth_death`` receives as the birth rate
    n_alive: int = 1  # the ``E / n_alive`` scaling of the inversion target

    @property
    def is_rate(self) -> bool:
        """Whether the object is a resolved piecewise-linear ``Rate``."""
        return not isinstance(self.obj, Hazard)

    @property
    def split(self) -> bool:
        """Whether the path is the superposition (split) sampler."""
        return self.path == "split"

    @property
    def closed(self) -> bool:
        """Whether the path is a closed-form + Newton polynomial inversion."""
        return self.path in CLOSED_METHODS

    @property
    def method(self) -> str:
        """The ``_method`` of a closed path."""
        return CLOSED_METHODS[self.path]

    @property
    def n_draws(self) -> int:
        """Exponentials per event (1 unless the hazard splits)."""
        return 1 if self.is_rate else self.obj.n_draws

    @property
    def loopish(self) -> bool:
        """Whether inversion is a per-element Python loop (Rates, Brent)."""
        if self.split:
            return False
        return self.is_rate or not self.obj.has_direct_sampler


POLYS: dict[str, Hazard] = {
    "quadratic": QuadraticHazard(0.1, 0.05, 0.01),
    "quadratic-c0": QuadraticHazard(0.1, 0.1, 0.0),
    "cubic": CubicHazard(0.1, 0.02, 0.005, 0.0005),
    "cubic-d0": CubicHazard(0.1, 0.05, 0.01, 0.0),
}


def _ramp(end: float) -> tuple[list[float], list[float], list[float]]:
    """Return start values, end values and end times of the pw-linear ramp."""
    lo, hi = C - 2.0 / KAPPA, C + 2.0 / KAPPA
    return [0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [lo, hi, end]


def build_families(end: float = T) -> list[Family]:
    """Return the matched families of the module docstring on ``[0, end]``.

    ``end`` is ``T`` for the timing grid; the end-to-end runs rebuild the same
    functions with ``end = terminal_time``, because ``simulate_birth_death``
    requires the rates' end time to equal ``terminal_time``.
    """
    const = PiecewiseConstant([1.0], [end])
    if end > T_LIN:
        lin = PiecewiseLinear([0.0, 1.0], [1.0, 1.0], [T_LIN, end])
    else:
        lin = PiecewiseLinear([0.0], [end / T_LIN], [end])
    v0, v1, ends = _ramp(end)
    ramp = PiecewiseLinear(v0, v1, ends)
    lo, hi = ends[0], ends[1]
    ramp_hazard = PiecewiseHazard.from_polynomials(
        [[0.0], [0.0, 1.0 / (hi - lo)], [1.0]], ends
    )
    out = [
        Family("constant", "hazard-direct", GompertzHazard(1.0, 0.0), None),
        Family("constant", "rate", as_rate(const), const),
        Family(
            "linear",
            "hazard-brent",
            _BrentOnly(QuadraticHazard(0.0, 1.0 / T_LIN, 0.0)),
            None,
        ),
        Family("linear", "rate", as_rate(lin), lin),
        Family("pw-linear", "rate", as_rate(ramp), ramp),
        Family("pw-linear", "hazard-piecewise", ramp_hazard, None),
        Family("logistic", "hazard-direct", LogisticHazard(1.0, KAPPA, C), None),
        Family(
            "floor-logistic",
            "split",
            LogisticHazard.with_floor(0.2, 0.8, KAPPA, C),
            None,
        ),
        Family(
            "floor-logistic-fall",
            "split",
            LogisticHazard.with_floor(0.2, 0.8, -KAPPA, C),
            None,
        ),
    ]
    for name, poly in POLYS.items():
        out.append(Family(name, "hazard-brent", _BrentOnly(poly), None))
        out.extend(Family(name, path, poly, None) for path in CLOSED_METHODS)
    return [
        Family(f.name, f.path, f.obj, f.obj if f.e2e is None else f.e2e) for f in out
    ]


def with_n_alive(families: list[Family]) -> list[Family]:
    """Expand the ``n_alive`` axis (polynomial families)."""
    out: list[Family] = []
    for f in families:
        on_axis = f.name in POLY_NAMES
        out.extend(replace(f, n_alive=n) for n in (N_ALIVE_AXIS if on_axis else (1,)))
    return out


def check_matched() -> None:
    """Assert that the logistic has value 1/2 and slope kappa/4 at ``t = c``."""
    logistic = LogisticHazard(1.0, KAPPA, C)
    step = 1e-5
    assert abs(float(logistic.hazard(C)) - 0.5) < 1e-12
    slope = (float(logistic.hazard(C + step)) - float(logistic.hazard(C - step))) / (
        2 * step
    )
    assert abs(slope - KAPPA / 4) < 1e-8


def scenario_starts(scenario: str, n: int) -> np.ndarray:
    """Return the ``t0`` array of a scenario (fixed seed, drawn outside timers)."""
    index = SCENARIOS.index(scenario)
    rng = np.random.default_rng([SEED, index, n])
    if scenario == "S1":
        return np.zeros(n)
    if scenario == "S2":
        return rng.uniform(0.0, 2.0 * C, size=n)
    if scenario == "S3":
        return rng.uniform(C, C + 2.0 / KAPPA, size=n)
    return np.full(n, C + 20.0 / KAPPA)


def draw_exponentials(n: int) -> np.ndarray:
    """Return the pre-drawn ``E`` (fixed seed) for a batch size."""
    return np.random.default_rng([SEED, 99, n]).exponential(size=n)


class _BrentOnly(Hazard):
    """A hazard forced onto the single-``E`` Brent path (no closed form, no split)."""

    def __init__(self, inner: Hazard) -> None:
        self.inner = inner

    def __repr__(self) -> str:
        return f"BrentOnly({self.inner!r})"

    def hazard(self, t: Any) -> Any:
        return self.inner.hazard(t)

    def cumulative_hazard(self, t: Any) -> Any:
        return self.inner.cumulative_hazard(t)

    def cumulative_between(self, s: Any, e: Any) -> Any:
        return self.inner.cumulative_between(s, e)

    @property
    def cumulative_limit(self) -> float:
        return self.inner.cumulative_limit


class _Counting(Hazard):
    """Delegates to a hazard and counts its ``cumulative_between`` calls."""

    def __init__(self, inner: Hazard) -> None:
        self.inner = inner
        self.calls = 0
        self.has_direct_sampler = inner.has_direct_sampler  # pyright: ignore[reportIncompatibleVariableOverride]

    def hazard(self, t: Any) -> Any:
        return self.inner.hazard(t)

    def cumulative_hazard(self, t: Any) -> Any:
        return self.inner.cumulative_hazard(t)

    def cumulative_between(self, s: Any, e: Any) -> Any:
        self.calls += 1
        return self.inner.cumulative_between(s, e)

    @property
    def cumulative_limit(self) -> float:
        return self.inner.cumulative_limit


def evals_per_root(fam: Family, t0: np.ndarray, e: np.ndarray) -> tuple[float, str]:
    """Mean ``cumulative_between`` calls per root (untimed) and a note."""
    if fam.is_rate:
        return math.nan, "rate: no hazard evaluations"
    if isinstance(fam.obj, PiecewiseHazard):
        return math.nan, "piecewise walk calls its pieces; not counted"
    if fam.split:
        return 0.0, "split closed form"
    if fam.closed:
        return math.nan, "closed form + Newton; see regime counts"
    if fam.obj.has_direct_sampler:
        return 0.0, "closed form"
    counter = _Counting(fam.obj)
    scaled = (e / fam.n_alive)[:N_COUNT].tolist()
    for s, d in zip(t0[:N_COUNT].tolist(), scaled, strict=True):
        counter.forward_time(s, d, T)
    return counter.calls / min(N_COUNT, len(t0)), ""


class Batch:
    """The inputs of one cell group: arrays, lists and the Rate loop helper."""

    def __init__(self, scenario: str, n: int) -> None:
        self.n = n
        self.t0 = scenario_starts(scenario, n)
        self.e = draw_exponentials(n)
        self.t0_list = self.t0.tolist()
        self.e_list = self.e.tolist()
        self._multi: dict[int, np.ndarray] = {}
        self._rows: dict[int, list[list[float]]] = {}
        self._scaled: dict[int, tuple[np.ndarray, list[float]]] = {}

    def e_scaled(self, n_alive: int) -> tuple[np.ndarray, list[float]]:
        """Return ``E / n_alive`` as an array and a list (cached)."""
        if n_alive not in self._scaled:
            scaled = self.e / n_alive
            self._scaled[n_alive] = (scaled, scaled.tolist())
        return self._scaled[n_alive]

    def e_split(self, m: int) -> np.ndarray:
        """Return pre-drawn exponentials of shape ``(m, n)`` (cached, fixed seed)."""
        if m not in self._multi:
            rng = np.random.default_rng([SEED, 98, self.n, m])
            self._multi[m] = rng.exponential(size=(m, self.n))
        return self._multi[m]

    def e_rows(self, m: int) -> list[list[float]]:
        """Return :meth:`e_split` as one list of ``m`` values per element."""
        if m not in self._rows:
            self._rows[m] = self.e_split(m).T.tolist()
        return self._rows[m]

    def head(self, m: int) -> "Batch":
        """Return a view of the first ``m`` elements (for warm-ups)."""
        out = object.__new__(Batch)
        out.n = min(m, self.n)
        out.t0, out.e = self.t0[:m], self.e[:m]
        out.t0_list, out.e_list = self.t0_list[:m], self.e_list[:m]
        out._multi, out._rows, out._scaled = {}, {}, {}
        return out


def _rate_loop(rate: Any, b: Batch, e_list: list[float]) -> np.ndarray:
    return np.array(
        [
            forward_hazard_time(rate, s, d, T)
            for s, d in zip(b.t0_list, e_list, strict=True)
        ]
    )


def operations(fam: Family) -> dict[str, Callable[[Batch], Any]]:
    """Return the timed operation of each batch measure for a family."""
    obj = fam.obj
    if fam.is_rate:

        def full(b: Batch) -> Any:
            e = root_rng(SEED, "dynamics").exponential(size=b.n).tolist()
            return _rate_loop(obj, b, e)

        def rate_invert(b: Batch) -> Any:
            return _rate_loop(obj, b, b.e_list)

        return {
            "hazard": lambda b: value(obj, b.t0),
            "cumulative": lambda b: cumulative(obj, b.t0),
            "invert": rate_invert,
            "full_batch": full,
            "individual": rate_invert,
        }
    if fam.split:
        m = fam.n_draws
        return {
            "hazard": lambda b: obj.hazard(b.t0),
            "cumulative": lambda b: obj.cumulative_hazard(b.t0),
            "invert": lambda b: obj.forward_offsets_split(b.t0, b.e_split(m)),
            "full_batch": lambda b: sample_residual_lifetimes(
                obj, b.t0, random_seed=SEED, end_time=T
            ),
            "individual": lambda b: np.array(
                [
                    obj.forward_time_split(s, es, T)
                    for s, es in zip(b.t0_list, b.e_rows(m), strict=True)
                ]
            ),
        }
    n_alive = fam.n_alive
    if fam.closed:
        method = fam.method
        return {
            "hazard": lambda b: obj.hazard(b.t0),
            "cumulative": lambda b: obj.cumulative_hazard(b.t0),
            "invert": lambda b: _closed_invert(
                obj, b.t0, b.e_scaled(n_alive)[0], method
            ),
            "full_batch": lambda b: _closed_full(obj, b, n_alive, method),
            "individual": lambda b: np.array(
                [
                    _closed_time(obj, s, d, method)
                    for s, d in zip(b.t0_list, b.e_scaled(n_alive)[1], strict=True)
                ]
            ),
        }

    def full(b: Batch) -> Any:
        if n_alive == 1:
            return sample_residual_lifetimes(obj, b.t0, random_seed=SEED, end_time=T)
        e = (root_rng(SEED, "dynamics").exponential(size=b.n) / n_alive).tolist()
        times = np.array(
            [obj.forward_time(s, d, T) for s, d in zip(b.t0_list, e, strict=True)]
        )
        return np.where(times >= T, np.inf, times - b.t0)

    return {
        "hazard": lambda b: obj.hazard(b.t0),
        "cumulative": lambda b: obj.cumulative_hazard(b.t0),
        "invert": lambda b: obj.forward_offsets(b.t0, b.e_scaled(n_alive)[0]),
        "full_batch": full,
        "individual": lambda b: np.array(
            [
                obj.forward_time(s, d, T)
                for s, d in zip(b.t0_list, b.e_scaled(n_alive)[1], strict=True)
            ]
        ),
    }


def _closed_invert(
    obj: Any,
    t0: np.ndarray,
    e: np.ndarray,
    method: str,
    cap: float = math.inf,
    counts: RegimeCounts | None = None,
) -> np.ndarray:
    """Invert by the default (gated) path, or ``_polyroots`` with ``method`` forced."""
    if method == "gated" and counts is None:
        if math.isfinite(cap):
            return obj._forward_offsets_capped(t0, e, cap)  # noqa: SLF001
        return obj.forward_offsets(t0, e)
    return _polyroots.forward_offsets(
        obj,
        obj._coefficients(),  # noqa: SLF001
        t0,
        e,
        cap,
        method=method,
        counts=counts,
    )


def _closed_time(obj: Any, s: float, d: float, method: str) -> float:
    if method == "gated":
        return obj.forward_time(s, d, T)
    coef = obj._coefficients()  # noqa: SLF001
    return _polyroots.forward_time(obj, coef, s, d, T, method=method)


def _closed_full(obj: Any, b: Batch, n_alive: int, method: str) -> np.ndarray:
    """Draw ``E`` as the Rates do, divide by ``n_alive`` and invert, capped at ``T``."""
    e = root_rng(SEED, "dynamics").exponential(size=b.n) / n_alive
    r = _closed_invert(obj, b.t0, e, method, T)
    return np.where(b.t0 + r >= T, np.inf, r)


def regime_counts(fam: Family, b: Batch) -> RegimeCounts:
    """Count the regimes of one untimed ``invert`` call of a closed path."""
    counts = RegimeCounts()
    _closed_invert(fam.obj, b.t0, b.e_scaled(fam.n_alive)[0], fam.method, counts=counts)
    return counts


def scalar_between(fam: Family, t0: np.ndarray) -> Callable[[], None]:
    """Return a function making ``N_SCALAR`` scalar ``cumulative_between`` calls."""
    starts = np.resize(t0, N_SCALAR).tolist()
    obj = fam.obj
    if fam.is_rate:

        def run_rate() -> None:
            for s in starts:
                float(cumulative(obj, s + 1.0)) - float(cumulative(obj, s))

        return run_rate

    def run() -> None:
        for s in starts:
            obj.cumulative_between(s, s + 1.0)

    return run


def timed_cell(
    run: Callable[[], Any],
    warm: Callable[[], Any],
    repeats: int,
    budget: float,
    projected: float | None,
) -> tuple[list[float], Any, str]:
    """Time ``run`` ``repeats`` times after an untimed warm-up.

    Returns the times, the last result and a note.  Stops after the first repeat
    if it exceeds ``budget``; runs once if ``projected`` exceeds ``budget``.
    """
    warm()
    note = ""
    if projected is not None and projected > budget:
        repeats = 1
        note = f"projected {projected:.0f}s > budget; 1 repeat"
    times: list[float] = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = run()
        times.append(time.perf_counter() - start)
        if times[0] > budget:
            if repeats > 1:
                note = f"first repeat {times[0]:.0f}s > budget; skipped the rest"
            break
    return times, result, note


def cell_plan(
    fam: Family, scenario: str, n: int, measure: str, repeats: int
) -> tuple[int, bool] | None:
    """Return ``(repeats, trimmed)`` for a batch cell, or ``None`` to skip it.

    The runtime trim: a loop or Brent cell uses 3 repeats at n = 1e5, and at
    n = 1e6 runs only S2 ``full_batch``, once.
    """
    if not fam.loopish or n < 100_000:
        return repeats, False
    if n < 1_000_000:
        return REPEATS_BIG_LOOP, False
    if scenario == "S2" and measure == "full_batch":
        return 1, True
    return None


def _log(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


def run_benchmark(  # noqa: C901, PLR0912, PLR0915
    *,
    quick: bool,
    max_cell_seconds: float,
    families: list[Family] | None = None,
) -> pd.DataFrame:
    """Run the grid and return the rows (columns ``ALL_COLUMNS``)."""
    check_matched()
    families = families if families is not None else with_n_alive(build_families())
    sizes = QUICK_SIZES if quick else SIZES
    scenarios = QUICK_SCENARIOS if quick else SCENARIOS
    measures = QUICK_MEASURES if quick else MEASURES
    repeats_small = QUICK_REPEATS if quick else REPEATS
    rows: list[dict[str, Any]] = []

    def add(
        fam: Family, sc: str, n: int, measure: str, rep: int, x: float, **kw: str
    ) -> None:
        rows.append(
            {
                "family": fam.name,
                "path": fam.path,
                "scenario": sc,
                "n": n,
                "measure": measure,
                "repeat": rep,
                "seconds": x,
                "unit": kw.get("unit", "s"),
                "note": kw.get("note", ""),
                "n_alive": fam.n_alive,
            }
        )

    batches = {(sc, n): Batch(sc, n) for sc in scenarios for n in sizes}
    for fam in families:
        ops = operations(fam)
        fam_measures = (
            tuple(m for m in measures if m in ALIVE_ONLY_MEASURES)
            if fam.closed or fam.n_alive > 1
            else measures
        )
        medians: dict[tuple[str, str, int], float] = {}
        for sc in scenarios:
            _log(f"{fam.name}/{fam.path} n_alive={fam.n_alive} {sc}")
            for n in sizes:
                batch = batches[sc, n]
                warm = batch.head(N_WARM)
                for measure in fam_measures:
                    if measure == "between_scalar":
                        if n != sizes[0]:
                            continue
                        fn = scalar_between(fam, batch.t0)
                        times, _, note = timed_cell(
                            fn, fn, repeats_small, max_cell_seconds, None
                        )
                        for i, x in enumerate(times):
                            add(
                                fam,
                                sc,
                                N_SCALAR,
                                measure,
                                i,
                                x / N_SCALAR,
                                unit="s/call",
                                note=note,
                            )
                        continue
                    op = ops[measure]
                    plan = cell_plan(fam, sc, n, measure, repeats_small)
                    if plan is None:
                        continue
                    reps, trimmed = plan
                    if fam.split:  # pre-draw outside the timer
                        for b in (batch, warm):
                            b.e_rows(fam.n_draws)
                    prev = [m for m in sizes if m < n]
                    projected = None
                    if n >= 1_000_000 and prev and (measure, sc, prev[-1]) in medians:
                        projected = medians[measure, sc, prev[-1]] * n / prev[-1]
                    times, result, note = timed_cell(
                        lambda op=op, batch=batch: op(batch),
                        lambda op=op, warm=warm: op(warm),
                        reps,
                        max_cell_seconds,
                        projected,
                    )
                    medians[measure, sc, n] = statistics.median(times)
                    if trimmed:
                        note = f"trimmed; {note}" if note else "trimmed"
                    for i, x in enumerate(times):
                        add(fam, sc, n, measure, i, x, note=note)
                    if measure == "full_batch":
                        res = np.asarray(result, dtype=float)
                        if fam.is_rate:
                            happened = res < T
                            residual = res - batch.t0
                        else:
                            happened = np.isfinite(res)
                            residual = res
                        add(
                            fam,
                            sc,
                            n,
                            "no_event_fraction",
                            0,
                            float(1 - happened.mean()),
                            unit="fraction",
                        )
                        mean = (
                            float(residual[happened].mean())
                            if happened.any()
                            else math.nan
                        )
                        add(fam, sc, n, "mean_residual", 0, mean, unit="time")
                        count, cnote = evals_per_root(fam, batch.t0, batch.e)
                        add(
                            fam,
                            sc,
                            n,
                            "evals_per_root",
                            0,
                            count,
                            unit="count",
                            note=cnote,
                        )
                        if fam.closed:
                            reg = regime_counts(fam, batch)
                            for name, x in (
                                ("regime_newton", reg.newton),
                                ("regime_algebraic", reg.algebraic),
                                ("regime_brent", reg.brent),
                            ):
                                add(fam, sc, n, name, 0, float(x), unit="count")
                            add(
                                fam,
                                sc,
                                n,
                                "fallback_fraction",
                                0,
                                reg.fallback_rate,
                                unit="fraction",
                            )
                    med = medians[measure, sc, n]
                    _log(f"  {measure} n={n}: median {med:.4g}s {note}")
    rows += end_to_end_rows(families, quick=quick)
    return pd.DataFrame(rows, columns=ALL_COLUMNS)


def end_to_end_rows(families: list[Family], *, quick: bool) -> list[dict[str, Any]]:
    """Time ``simulate_birth_death`` over seeds, one row per seed."""
    n_seeds = QUICK_E2E_SEEDS if quick else N_E2E_SEEDS
    rows: list[dict[str, Any]] = []
    short = {(f.name, f.path): f.e2e for f in build_families(E2E_TERMINAL)}
    for fam in families:
        if fam.closed or fam.n_alive > 1:
            continue
        e2e = short[fam.name, fam.path]
        _log(f"end_to_end {fam.name}/{fam.path}")
        simulate_birth_death(e2e, 0.9, terminal_time=E2E_TERMINAL, random_seed=10_000)
        for seed in range(n_seeds):
            start = time.perf_counter()
            simulate_birth_death(e2e, 0.9, terminal_time=E2E_TERMINAL, random_seed=seed)
            rows.append(
                {
                    "family": fam.name,
                    "path": fam.path,
                    "scenario": "e2e",
                    "n": n_seeds,
                    "measure": "end_to_end",
                    "repeat": seed,
                    "seconds": time.perf_counter() - start,
                    "unit": "s",
                    "note": "",
                    "n_alive": 1,
                }
            )
    return rows


def summarise(rows: pd.DataFrame) -> pd.DataFrame:
    """Return the median ``seconds`` of every cell."""
    keys = ["family", "path", "n_alive", "scenario", "n", "measure"]
    return rows.groupby(keys, sort=False)["seconds"].median().reset_index()


def main(argv: list[str] | None = None) -> None:
    """Run the benchmark and write the CSV and environment files."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--quick", action="store_true", help="tiny grid for a draft")
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT, help="output directory"
    )
    parser.add_argument(
        "--max-cell-seconds", type=float, default=DEFAULT_MAX_CELL_SECONDS
    )
    parser.add_argument("--summary", action="store_true", help="print median per cell")
    args = parser.parse_args(argv)

    rows = run_benchmark(quick=args.quick, max_cell_seconds=args.max_cell_seconds)
    args.out.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.out / CSV_NAME, index=False)
    record = common.environment_record()
    for key in ("host", "slurm_job"):
        record.pop(key, None)
    env = {
        **record,
        "quick": args.quick,
        "seed": SEED,
        "T": T,
        "c": C,
        "kappa": KAPPA,
        "max_cell_seconds": args.max_cell_seconds,
    }
    (args.out / ENV_NAME).write_text(json.dumps(env, indent=2) + "\n")
    print(f"Wrote {len(rows)} rows to {args.out / CSV_NAME}")
    if args.summary:
        with pd.option_context("display.max_rows", None, "display.width", 200):
            print(summarise(rows).to_string(index=False, float_format="{:.4g}".format))


if __name__ == "__main__":
    main()
