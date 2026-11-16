r"""Altair charts for the Arley benchmark pages.

The benchmark rows come from ``bench_arley_comparison.py --grid``: every grid
death-rate slope ``m`` (``mu(t) = m t``, birth rate 1) over the terminal-time grid.
This module turns them into the cost-per-tree curves (mean with a 95% confidence
interval, and median with a 10th-90th percentile band, for each method), the
splitting diagnostics, a graph of the birth and death rates over time, and the
random-draws-per-tree curves and charts built from ``count_draws.py``'s rows.
It mirrors the constant-rate charts of the method-comparison page, keyed by
``(mu, slope, T)`` where ``mu`` is the constant rate the slope is paired with.
"""

import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import altair as alt
import pandas as pd

# Loaded by path from a notebook, ``scripts/`` is not on the import path yet.
_SCRIPTS = str(Path(__file__).resolve().parent)
if _SCRIPTS not in sys.path:
    sys.path.append(_SCRIPTS)

import _charts_common as common

LABEL_CONTOUR = common.LABEL_CONTOUR
LABEL_RUN = common.LABEL_RUN
LABEL_KEPT = common.LABEL_KEPT
PALETTE = common.PALETTE
BIRTH_COLOR = "#228833"
DEATH_COLOR = "#AA3377"
GROUP = ["mu", "slope", "T"]
color_encoding = common.color_encoding
x_by_t = common.x_by_t


def curves_from_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Mean, median and bands of seconds per tree, one block per method label."""
    return common.curves_from_rows(rows, GROUP)


def curves_from_draws(rows: pd.DataFrame) -> pd.DataFrame:
    """Mean draws per tree with a 95% interval, one block per method label.

    ``rows`` are the Arley rows of ``draws.csv``.
    Splitting is charged per set of trees (the whole run) and per kept tree.
    """
    return common.curves_from_draws(rows, GROUP)


def _key_tooltips(mu_title: str) -> tuple[alt.Tooltip, ...]:
    """Tooltips naming the cell: the slope, the paired constant rate and ``T``."""
    return (
        alt.Tooltip("slope:Q", format=".3~g", title="slope m"),
        alt.Tooltip("mu:Q", format="g", title=mu_title),
        alt.Tooltip("T:Q", format="g"),
    )


def draws_chart(
    data: pd.DataFrame,
    x: alt.X,
    title: str | None = None,
    *,
    legend_font_size: int = 14,
    width: int = 900,
    height: int = 380,
) -> alt.LayerChart:
    """Mean random numbers drawn per tree against ``x``, with a 95% interval band.

    Clicking a legend entry shows that method and fades the rest; shift-click
    adds more and double-click resets.
    """
    options = common.CostChartOptions(
        statistic="mean",
        measure="draws",
        legend_font_size=legend_font_size,
        width=width,
        height=height,
        key_tooltips=_key_tooltips("paired constant μ"),
        value_format=",.0f",
        title=title,
    )
    return common.cost_chart(data, x, options)


def diagnostics_from_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Median trees kept, attempts and levels per splitting run, by cell."""
    return (
        rows[rows["method"] == "splitting"]
        .groupby(GROUP)
        .agg(
            trees=("n_trees", "median"),
            attempts=("n_attempts", "median"),
            levels=("levels", "median"),
        )
        .reset_index()
    )


def x_by_slope(slopes: list[float]) -> alt.X:
    """Log death-rate-slope axis, tight to the data."""
    return common.log_axis(
        "slope",
        "death-rate slope m (log scale)",
        slopes,
        tight=True,
        label_format=".3~g",
    )


def cost_chart(
    data: pd.DataFrame,
    x: alt.X,
    *,
    statistic: str = "mean",
    legend_font_size: int = 14,
    width: int = 900,
    height: int = 380,
) -> alt.LayerChart:
    """Seconds per tree against ``x`` for each method, with a band.

    ``statistic`` is ``"mean"`` (with a 95% confidence interval) or ``"median"``
    (with the 10th to 90th percentile).

    Clicking a legend entry shows that method and fades the rest; shift-click
    adds more and double-click resets.
    """
    options = common.CostChartOptions(
        statistic=statistic,
        legend_font_size=legend_font_size,
        width=width,
        height=height,
        key_tooltips=_key_tooltips("matched constant μ"),
    )
    return common.cost_chart(data, x, options)


def diagnostic_chart(data: pd.DataFrame) -> alt.HConcatChart:
    """Three small panels: trees kept, attempts and levels of splitting against T."""
    panels = []
    for column, title in [
        ("trees", "trees kept per run"),
        ("attempts", "attempts per run"),
        ("levels", "levels"),
    ]:
        panels.append(
            alt.Chart(data)
            .mark_line(strokeWidth=3, point=alt.OverlayMarkDef(size=60, filled=True))
            .encode(
                x=alt.X(
                    "T:Q",
                    scale=alt.Scale(type="log"),
                    title="terminal time T (log scale)",
                    axis=alt.Axis(format="g", values=[1, 10, 100]),
                ),
                y=alt.Y(f"{column}:Q", scale=alt.Scale(zero=True), title=title),
                tooltip=[alt.Tooltip("T:Q", format="g"), f"{column}:Q"],
                color=alt.value(PALETTE[LABEL_KEPT]),
            )
            .properties(width=270, height=220, title=title)
        )
    return alt.hconcat(*panels)


def _slope_term(slope: float) -> str:
    """``t/k`` when the slope is ``1/k`` for an integer ``k``, else ``slope t``."""
    k = round(1.0 / slope)
    if k >= 2 and abs(slope * k - 1.0) < 1e-9:
        return f"t/{k}"
    return f"{slope:.3g} t"


@dataclass(frozen=True)
class RateChartOptions:
    """Display options of ``rate_chart``.

    Parameters
    ----------
    t_max
        Last time shown.
    y_max
        Top of the rate axis; the death rate is clipped there.
    legend_font_size, annotation_font_size
        Font sizes of the legend labels and of the crossing annotation.
    t_ticks, rate_ticks
        Tick values of the two axes; automatic if None.
    width, height
        Chart size in pixels.
    """

    t_max: float = 100.0
    y_max: float = 3.0
    legend_font_size: int = 14
    annotation_font_size: int = 13
    t_ticks: Sequence[float] | None = None
    rate_ticks: Sequence[float] | None = None
    width: int = 900
    height: int = 260


def rate_chart(
    slope: float, options: RateChartOptions | None = None, **overrides: object
) -> alt.LayerChart:
    r"""Birth rate ``1`` and Arley death rate ``slope * t`` against time.

    The death rate is clipped at ``y_max`` so the crossing at ``t = 1 / slope``
    stays visible even for steep slopes; it keeps rising beyond the plot.
    Before the crossing the birth rate exceeds the death rate (the process is
    supercritical), after it the process is subcritical.

    ``overrides`` are fields of ``RateChartOptions`` (``t_max``, ``y_max``,
    ``legend_font_size``, ``annotation_font_size``, ``t_ticks``, ``rate_ticks``,
    ``width``, ``height``) replacing those of ``options``; any other keyword is a
    ``TypeError``.
    """
    opts = replace(options or RateChartOptions(), **overrides)
    t_max, y_max = opts.t_max, opts.y_max
    t_ticks, rate_ticks = opts.t_ticks, opts.rate_ticks
    legend_font_size = opts.legend_font_size
    annotation_font_size = opts.annotation_font_size
    lines = pd.DataFrame(
        {
            "t": [0.0, t_max, 0.0, t_max],
            "rate": [1.0, 1.0, 0.0, slope * t_max],
            "series": ["birth rate λ = 1"] * 2
            + [f"death rate μ(t) = {_slope_term(slope)}"] * 2,
        }
    )
    domain = [
        "birth rate λ = 1",
        f"death rate μ(t) = {_slope_term(slope)}",
    ]
    base = alt.Chart(lines).encode(
        x=alt.X(
            "t:Q",
            scale=alt.Scale(domain=[0, t_max], nice=False),
            title="time t",
            axis=alt.Axis(format="g")
            if t_ticks is None
            else alt.Axis(values=t_ticks, format="d"),
        ),
        y=alt.Y(
            "rate:Q",
            scale=alt.Scale(domain=[0, y_max], nice=False),
            title="rate per individual",
            axis=alt.Axis()
            if rate_ticks is None
            else alt.Axis(values=rate_ticks, format="d"),
        ),
    )
    series = base.mark_line(strokeWidth=3, clip=True).encode(
        color=alt.Color(
            "series:N",
            scale=alt.Scale(domain=domain, range=[BIRTH_COLOR, DEATH_COLOR]),
            legend=alt.Legend(
                title=None,
                orient="bottom",
                labelFontSize=legend_font_size,
                labelLimit=0,
            ),
        ),
        tooltip=[
            "series",
            alt.Tooltip("t:Q", format="g"),
            alt.Tooltip("rate:Q", format=".3g"),
        ],
    )
    layers: list[alt.Chart | alt.LayerChart] = [series]
    crossing = 1.0 / slope
    if crossing <= t_max:
        marker = pd.DataFrame(
            {"t": [crossing], "label": [f"μ(t) = λ at t = {crossing:.3g}"]}
        )
        layers.append(
            alt.Chart(marker)
            .mark_rule(strokeDash=[6, 4], color="#888888")
            .encode(x="t:Q", tooltip=["label"])
        )
        layers.append(
            alt.Chart(marker)
            .mark_text(
                align="left",
                dx=6,
                dy=-8,
                color="#555555",
                fontSize=annotation_font_size,
            )
            .encode(x="t:Q", y=alt.value(14), text="label")
        )
    return alt.layer(*layers).properties(width=opts.width, height=opts.height)
