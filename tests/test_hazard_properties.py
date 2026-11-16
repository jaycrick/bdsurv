"""Property tests (Hypothesis) of the hazard inversion and the level-grid maths.

The strategies are bounded so that the maths is well conditioned: rates in
``[0.05, 5]`` and times in ``[0.1, 20]``.  No strategy has been narrowed to hide
a counterexample.
"""

from __future__ import annotations

import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from bdsurv import PiecewiseConstant, PiecewiseLinear
from bdsurv._hazard import (
    backward_hazard_time,
    cumulative,
    forward_hazard_time,
    log_survival_probability_constant,
    log_survival_probability_quadrature,
    next_event_offset,
)
from bdsurv.rates import Rate, as_rate
from bdsurv.splitting import (
    compute_level_replicates,
    select_level_times_for_band,
)

PROPERTY_SETTINGS = settings(max_examples=100, deadline=None)

rates_st = st.floats(min_value=0.05, max_value=5.0)
times_st = st.floats(min_value=0.1, max_value=20.0)

# Absolute tolerance on a hazard (an integral of a rate over at most 15 time
# units, so of size at most 75) and on the scale-free checks.
HAZARD_TOL = 1e-9
# The level-grid band is ``Pr(t_i) / Pr(t_{i-1}) >= exp(-2)``, up to this slack.
BAND_SLACK = 1e-12
LOWER_BAND = math.exp(-2.0)


@st.composite
def piecewise_linear_rates(draw: st.DrawFn) -> Rate:
    """Draw a piecewise-linear rate on one to three epochs of length 0.1 to 5."""
    n_epochs = draw(st.integers(min_value=1, max_value=3))
    lengths = draw(
        st.lists(
            st.floats(min_value=0.1, max_value=5.0),
            min_size=n_epochs,
            max_size=n_epochs,
        )
    )
    starts = draw(st.lists(rates_st, min_size=n_epochs, max_size=n_epochs))
    ends = draw(st.lists(rates_st, min_size=n_epochs, max_size=n_epochs))
    return as_rate(
        PiecewiseLinear(np.array(starts), np.array(ends), np.cumsum(np.array(lengths)))
    )


@PROPERTY_SETTINGS
@given(
    a=st.floats(min_value=0.05, max_value=5.0),
    b=st.floats(min_value=-1.0, max_value=1.0),
    e=st.floats(min_value=0.01, max_value=5.0),
)
def test_next_event_offset_solves_the_quadratic(a: float, b: float, e: float) -> None:
    """``s = next_event_offset(a, b, e)`` solves ``a s + b s^2 / 2 = e``.

    The residual is asserted below ``HAZARD_TOL`` (scaled by ``1 + e``).  With a
    falling hazard (``b < 0``) the equation has no root when the cumulative
    hazard peaks below ``e``, and the function returns ``inf``.
    """
    s = next_event_offset(a, b, e)
    if math.isinf(s):
        assert b < 0.0, f"returned inf for a={a}, b={b}, e={e} with a rising hazard"
        peak = a * a / (-2.0 * b)
        assert peak < e * (1.0 + HAZARD_TOL), (
            f"returned inf although the cumulative hazard peaks at {peak}, above "
            f"e={e} (a={a}, b={b})"
        )
        return
    assert s >= 0.0, f"offset {s} is negative for a={a}, b={b}, e={e}"
    residual = a * s + 0.5 * b * s * s - e
    assert abs(residual) <= HAZARD_TOL * (1.0 + e), (
        f"residual {residual:.3g} of a*s + b*s^2/2 - e exceeds "
        f"{HAZARD_TOL * (1.0 + e):.3g} for a={a}, b={b}, e={e}, s={s}"
    )
    if b < 0.0:
        assert s <= a / -b * (1.0 + HAZARD_TOL), (
            f"offset {s} is past the peak {a / -b} of the cumulative hazard, so it "
            f"is not the smallest root (a={a}, b={b}, e={e})"
        )


@PROPERTY_SETTINGS
@given(
    rate=piecewise_linear_rates(),
    start_fraction=st.floats(min_value=0.0, max_value=1.0),
    e=st.floats(min_value=0.01, max_value=10.0),
)
def test_forward_hazard_time_inverts_cumulative(
    rate: Rate, start_fraction: float, e: float
) -> None:
    """``forward_hazard_time`` returns ``s`` with ``H(s) - H(start) = e``.

    The cap is the end time ``T``; when no ``s <= T`` reaches ``e`` the result is
    ``T`` and the hazard used up to ``T`` is at most ``e``.  The tolerance is
    ``HAZARD_TOL`` on the hazard.
    """
    cap = rate.end_time
    start = start_fraction * cap
    s = forward_hazard_time(rate, start, e, cap)
    used = float(cumulative(rate, s) - cumulative(rate, start))
    assert start <= s <= cap, f"time {s} is outside [{start}, {cap}]"
    if s < cap:
        assert abs(used - e) <= HAZARD_TOL, (
            f"hazard between start={start} and s={s} is {used}, not e={e} "
            f"(error {used - e:.3g}, tolerance {HAZARD_TOL})"
        )
    else:
        assert used <= e + HAZARD_TOL, (
            f"capped at T={cap} but the hazard from start={start} is {used}, "
            f"above e={e}"
        )


@PROPERTY_SETTINGS
@given(
    rate=piecewise_linear_rates(),
    start_fraction=st.floats(min_value=0.0, max_value=1.0),
    e=st.floats(min_value=0.01, max_value=10.0),
)
def test_backward_hazard_time_inverts_cumulative(
    rate: Rate, start_fraction: float, e: float
) -> None:
    """``backward_hazard_time`` returns ``s`` with ``H(start) - H(s) = e``.

    When even ``s = 0`` does not reach ``e`` the result is ``0`` and the hazard
    from 0 to ``start`` is at most ``e``.  The tolerance is ``HAZARD_TOL``.
    """
    start = start_fraction * rate.end_time
    s = backward_hazard_time(rate, start, e)
    used = float(cumulative(rate, start) - cumulative(rate, s))
    assert 0.0 <= s <= start, f"time {s} is outside [0, {start}]"
    if s > 0.0:
        assert abs(used - e) <= HAZARD_TOL, (
            f"hazard between s={s} and start={start} is {used}, not e={e} "
            f"(error {used - e:.3g}, tolerance {HAZARD_TOL})"
        )
    else:
        assert used <= e + HAZARD_TOL, (
            f"returned 0 but the hazard from 0 to start={start} is {used}, above e={e}"
        )


@PROPERTY_SETTINGS
@given(birth=rates_st, death=rates_st, t=times_st)
def test_quadrature_survival_matches_closed_form(
    birth: float, death: float, t: float
) -> None:
    """Quadrature ``log Pr(Z_t > 0)`` equals the closed form for constant rates.

    Asserted with relative tolerance 1e-9 and absolute tolerance 1e-12.
    """
    rate_b = as_rate(PiecewiseConstant(np.array([birth]), np.array([t])))
    rate_d = as_rate(PiecewiseConstant(np.array([death]), np.array([t])))
    got = log_survival_probability_quadrature(rate_b, rate_d, t)
    expected = log_survival_probability_constant(
        birth_rate=birth, death_rate=death, time=t
    )
    assert math.isclose(got, expected, rel_tol=1e-9, abs_tol=1e-12), (
        f"quadrature {got!r} vs closed form {expected!r} for birth={birth}, "
        f"death={death}, t={t}: difference {got - expected:.3g}"
    )


@PROPERTY_SETTINGS
@given(
    birth=rates_st,
    death=rates_st,
    terminal_time=times_st,
    n_levels=st.one_of(st.none(), st.integers(min_value=1, max_value=60)),
)
def test_band_grid_is_increasing_and_in_band(
    birth: float, death: float, terminal_time: float, n_levels: int | None
) -> None:
    """The level grid rises strictly from 0 to ``T`` with every ratio in the band.

    Every ``Pr(t_i) / Pr(t_{i-1})`` is at least ``exp(-2)``, up to a slack of
    ``BAND_SLACK``.
    """
    grid = select_level_times_for_band(
        birth_rate=birth,
        death_rate=death,
        terminal_time=terminal_time,
        n_levels=n_levels,
    )
    assert grid[0] == 0.0, f"grid starts at {grid[0]}, not 0"
    assert math.isclose(grid[-1], terminal_time, rel_tol=1e-12), (
        f"grid ends at {grid[-1]}, not T={terminal_time}"
    )
    assert np.all(np.diff(grid) > 0.0), f"grid is not strictly increasing: {grid}"
    log_pr = np.array(
        [
            log_survival_probability_constant(
                birth_rate=birth, death_rate=death, time=float(t)
            )
            for t in grid
        ]
    )
    ratios = np.exp(np.diff(log_pr))
    assert np.all(ratios >= LOWER_BAND - BAND_SLACK), (
        f"ratios {ratios[ratios < LOWER_BAND - BAND_SLACK]} are below the lower "
        f"band {LOWER_BAND} (slack {BAND_SLACK}) on grid {grid}"
    )


@PROPERTY_SETTINGS
@given(
    birth=rates_st,
    death=rates_st,
    terminal_time=times_st,
    n_levels=st.integers(min_value=1, max_value=60),
)
def test_replicates_keep_expected_lines_at_least_one(
    birth: float, death: float, terminal_time: float, n_levels: int
) -> None:
    """``R_1 ... R_i Pr(t_i) >= 1`` for every level, checked in log space.

    The grid is a band-adjusted one (a grid far outside the band can need an
    astronomically large replicate count, which is refused).  The check allows
    the ``1e-9`` replicate tolerance of the implementation.
    """
    grid = select_level_times_for_band(
        birth_rate=birth,
        death_rate=death,
        terminal_time=terminal_time,
        n_levels=n_levels,
    )
    replicates = compute_level_replicates(
        birth_rate=birth, death_rate=death, level_times=grid
    )
    assert len(replicates) == len(grid) - 1, (
        f"{len(replicates)} replicate counts for {len(grid) - 1} intervals"
    )
    assert np.all(replicates >= 2), f"replicate counts below 2: {replicates}"
    log_cumulative = np.cumsum(np.log(replicates.astype(np.float64)))
    for i, log_r in enumerate(log_cumulative, start=1):
        log_pr = log_survival_probability_constant(
            birth_rate=birth, death_rate=death, time=float(grid[i])
        )
        assert log_r + log_pr >= -1e-8, (
            f"R_1...R_{i} Pr(t_{i}) = exp({log_r + log_pr:.3g}) is below 1: "
            f"replicates {replicates}, grid {grid}"
        )


@PROPERTY_SETTINGS
@given(
    value=st.floats(min_value=0.0, max_value=5.0),
    end=times_st,
    start_fraction=st.floats(min_value=0.0, max_value=1.0),
    e=st.floats(min_value=0.01, max_value=10.0),
)
def test_constant_rate_fast_path_matches_general_branch(
    value: float, end: float, start_fraction: float, e: float
) -> None:
    """The one-segment constant-rate branch agrees with the general solver.

    The general branch is reached by splitting the same rate into two equal
    constant segments, so it integrates the same hazard.
    """
    one = as_rate(PiecewiseConstant(np.array([value]), np.array([end])))
    two = as_rate(
        PiecewiseConstant(np.array([value, value]), np.array([end / 2.0, end]))
    )
    start = start_fraction * end
    assert math.isclose(
        forward_hazard_time(one, start, e, end),
        forward_hazard_time(two, start, e, end),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ), f"forward mismatch (value={value}, end={end}, start={start}, e={e})"
    assert math.isclose(
        backward_hazard_time(one, start, e),
        backward_hazard_time(two, start, e),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ), f"backward mismatch (value={value}, end={end}, start={start}, e={e})"
