"""Altair building blocks shared by the benchmark chart modules.

``method_comparison_charts.py`` (constant rates, keyed by ``(mu, T)``) and
``arley_charts.py`` (Arley rates, keyed by ``(mu, slope, T)``) draw the same
cost-per-tree curves from the same kind of benchmark rows.
This module holds what they share: the labels and palette, the summaries that
turn rows into curves, the axis helpers and the layered cost chart.
Each chart module keeps its own axes, tooltips and defaults.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace

import altair as alt
import numpy as np
import pandas as pd

LABEL_CONTOUR = "Contour process (exact)"
LABEL_RUN = "Splitting, per set of trees (one run)"
LABEL_KEPT = "Splitting, per kept tree"
PALETTE = {LABEL_CONTOUR: "#004488", LABEL_RUN: "#DDAA33", LABEL_KEPT: "#BB5566"}
THEME = "bdsurv"


def color_encoding(
    legend_font_size: int = 14, methods: list[str] | None = None
) -> alt.Color:
    """Return the colour encoding that maps each method label to its palette colour.

    Parameters
    ----------
    legend_font_size
        Font size of the legend labels.
    methods
        Method labels to show, in legend order; all labels in ``PALETTE`` if None.

    Returns
    -------
    altair.Color
        Encoding of the ``method`` column with a clickable legend at the bottom.
    """
    shown = methods or list(PALETTE)
    return alt.Color(
        "method:N",
        scale=alt.Scale(domain=shown, range=[PALETTE[m] for m in shown]),
        legend=alt.Legend(
            title=None, orient="bottom", labelFontSize=legend_font_size, labelLimit=0
        ),
    )


def _bdsurv_theme() -> alt.theme.ThemeConfig:
    return alt.theme.ThemeConfig(
        {
            "config": {
                "view": {"stroke": "transparent"},
                "axis": {
                    "labelFontSize": 14,
                    "titleFontSize": 16,
                    "gridColor": "#e6e6e6",
                },
                "title": {"fontSize": 18, "anchor": "start"},
            }
        }
    )


def register_theme() -> None:
    """Register the ``bdsurv`` Altair theme and enable it, once per process.

    Later calls do nothing, so the theme is registered exactly once however many
    chart modules call this.
    """
    if THEME not in alt.theme.names():
        alt.theme.register(THEME, enable=True)(_bdsurv_theme)


def band(
    frame: pd.DataFrame, column: str, label: str, group: Sequence[str]
) -> pd.DataFrame:
    """Summarise ``column`` by cell: median, mean, 95% CI, and 10th-90th band.

    Parameters
    ----------
    frame
        Benchmark rows with the ``group`` columns.
    column
        Name of the column to summarise.
    label
        Method label written to the ``method`` column of the result.
    group
        The columns that identify a cell, for example ``["mu", "T"]`` or
        ``["mu", "slope", "T"]``.

    Returns
    -------
    pandas.DataFrame
        One row per cell, without the cells whose median is missing.
    """
    grouped = frame.groupby(list(group))[column]
    sem = grouped.std() / np.sqrt(grouped.count())
    out = pd.DataFrame(
        {
            "median": grouped.median(),
            "mean": grouped.mean(),
            # Normal-approximation 95% confidence interval for the mean.
            "ci_lo": (grouped.mean() - 1.96 * sem).clip(lower=0),
            "ci_hi": grouped.mean() + 1.96 * sem,
            "lo": grouped.quantile(0.1),
            "hi": grouped.quantile(0.9),
            "replicates": grouped.count(),
        }
    ).reset_index()
    out["method"] = label
    return out.dropna(subset=["median"])


def _curves(rows: pd.DataFrame, column: str, group: Sequence[str]) -> pd.DataFrame:
    rows = rows.assign(
        per_kept=np.where(rows["n_trees"] > 0, rows[column] / rows["n_trees"], np.nan)
    )
    contour_rows = rows[rows["method"] == "contour"]
    split_rows = rows[rows["method"] == "splitting"]
    return pd.concat(
        [
            band(contour_rows, column, LABEL_CONTOUR, group),
            band(split_rows, column, LABEL_RUN, group),
            band(split_rows, "per_kept", LABEL_KEPT, group),
        ],
        ignore_index=True,
    )


def curves_from_rows(rows: pd.DataFrame, group: Sequence[str]) -> pd.DataFrame:
    """Summarise seconds per tree by cell, one block per method label.

    Splitting is charged per set of trees (the whole run) and per kept tree.

    Parameters
    ----------
    rows
        Benchmark rows (``seconds``, ``n_trees``, ``method`` and the ``group`` columns).
    group
        The columns that identify a cell.

    Returns
    -------
    pandas.DataFrame
        The output of ``band`` for each of the three method labels.
    """
    return _curves(rows, "seconds", group)


def curves_from_draws(rows: pd.DataFrame, group: Sequence[str]) -> pd.DataFrame:
    """Summarise random draws per tree by cell, one block per method label.

    Splitting is charged per set of trees (the whole run) and per kept tree.

    Parameters
    ----------
    rows
        Rows of ``draws.csv`` (``draws_total``, ``n_trees``, ``method`` and the
        ``group`` columns).
    group
        The columns that identify a cell.

    Returns
    -------
    pandas.DataFrame
        The output of ``band`` for each of the three method labels.
    """
    return _curves(rows, "draws_total", group)


def log_axis(
    column: str,
    title: str,
    ticks: Sequence[float],
    *,
    tight: bool = False,
    label_format: str = "g",
) -> alt.X:
    """Return a log-scale x encoding with a tick at every value in ``ticks``.

    Parameters
    ----------
    column
        Quantitative column to plot.
    title
        Axis title.
    ticks
        Values at which to put ticks.
    tight
        Fit the scale to the data, with no empty stretch before the first value.
    label_format
        Format of the tick labels.
    """
    scale = (
        alt.Scale(type="log", nice=False, padding=12)
        if tight
        else alt.Scale(type="log")
    )
    return alt.X(
        f"{column}:Q",
        scale=scale,
        title=title,
        axis=alt.Axis(
            values=[float(t) for t in ticks], format=label_format, labelOverlap=True
        ),
    )


def x_by_t(ts: Sequence[float]) -> alt.X:
    """Return the log-scale x encoding for terminal time, with ticks at ``ts``."""
    return log_axis("T", "terminal time T (log scale)", ts)


@dataclass(frozen=True)
class CostChartOptions:
    """Display options of ``cost_chart``, with the constant-rate defaults.

    Parameters
    ----------
    lam
        Birth rate, used to place the critical-point rule when ``mark_critical``.
    log_y
        Use a log-scale y axis if True, a zero-based linear axis otherwise.
    methods
        Method labels to show; all of them if None.
    show_band
        Draw the percentile band (median) or 95% interval (mean) around each curve.
    statistic
        Column to plot, ``"median"`` or ``"mean"``.
    measure
        ``"seconds"`` or ``"draws"``; sets the axis title and the tooltip title.
    mark_critical
        Draw a vertical rule at ``mu = lam``, separating super- from subcritical.
    legend_font_size
        Font size of the legend labels.
    width, height
        Chart size in pixels.
    key_tooltips
        Tooltips that identify the cell, shown after the method; the x field if None.
    value_format
        Number format of the statistic and of the band ends in the tooltip.
    title
        Chart title, if any.
    """

    lam: float | None = None
    log_y: bool = True
    methods: list[str] | None = None
    show_band: bool = True
    statistic: str = "median"
    measure: str = "seconds"
    mark_critical: bool = False
    legend_font_size: int = 14
    width: int = 900
    height: int = 380
    key_tooltips: tuple[alt.Tooltip, ...] | None = None
    value_format: str = ".3g"
    title: str | None = None


def _y_title(options: CostChartOptions) -> str:
    if options.measure == "draws":
        unit = f"{options.statistic} random draws per tree"
    elif options.statistic == "mean":
        unit = "mean sec per tree"
    else:
        unit = "seconds per tree"
    return f"{unit} (log scale)" if options.log_y else unit


def _interval(statistic: str) -> tuple[str, str, str | None]:
    # The median gets the 10th-90th percentile band, the mean a 95% interval.
    if statistic == "mean":
        return "ci_lo", "ci_hi", "95% CI"
    return "lo", "hi", None


def _tooltip(x: alt.X, options: CostChartOptions) -> list[str | alt.Tooltip]:
    statistic = options.statistic
    short = "draws" if options.measure == "draws" else "s"
    lo, hi, band_name = _interval(statistic)
    fmt = options.value_format
    key = options.key_tooltips or (alt.Tooltip(x.shorthand, format="g"),)
    return [
        "method",
        *key,
        alt.Tooltip(f"{statistic}:Q", format=fmt, title=f"{statistic} {short}"),
        alt.Tooltip(f"{lo}:Q", format=fmt, title=f"{band_name or '10th pct'} low"),
        alt.Tooltip(f"{hi}:Q", format=fmt, title=f"{band_name or '90th pct'} high"),
        "replicates",
    ]


def cost_chart(
    data: pd.DataFrame,
    x: alt.X,
    options: CostChartOptions | None = None,
    **overrides: object,
) -> alt.LayerChart:
    """Build the layered cost-per-tree chart, one curve (and optional band) per method.

    Clicking a legend entry shows that method and fades the rest; shift-click adds
    more and double-click resets.

    Parameters
    ----------
    data
        Curves from ``curves_from_rows`` or ``curves_from_draws``.
    x
        The x encoding, such as ``x_by_t``.
    options
        Display options; ``CostChartOptions()`` if None.
    **overrides
        Fields of ``CostChartOptions`` to replace in ``options``.

    Returns
    -------
    altair.LayerChart
        Band, line and points per method, plus the critical rule if requested.

    Raises
    ------
    TypeError
        If an override is not a field of ``CostChartOptions``.
    ValueError
        If ``mark_critical`` is set without ``lam``.
    """
    opts = replace(options or CostChartOptions(), **overrides)
    statistic = opts.statistic
    y_scale = alt.Scale(type="log") if opts.log_y else alt.Scale(zero=True)
    y_title = _y_title(opts)
    lo, hi, _ = _interval(statistic)
    y = alt.Y(f"{statistic}:Q", scale=y_scale, title=y_title)
    selection = alt.selection_point(fields=["method"], bind="legend")

    def fade(shown: float, hidden: float) -> alt.Opacity:
        return alt.condition(selection, alt.value(shown), alt.value(hidden))

    base = alt.Chart(data).encode(
        x=x, color=color_encoding(opts.legend_font_size, opts.methods)
    )
    area = base.mark_area().encode(
        y=alt.Y(f"{lo}:Q", scale=y_scale, title=y_title),
        y2=f"{hi}:Q",
        opacity=fade(0.15, 0.02),
    )
    line = base.mark_line(strokeWidth=3).encode(y=y, opacity=fade(1.0, 0.08))
    points = base.mark_point(size=70, filled=True).encode(
        y=y, tooltip=_tooltip(x, opts), opacity=fade(1.0, 0.08)
    )
    layers = [area, line, points] if opts.show_band else [line, points]
    if opts.mark_critical:
        if opts.lam is None:
            msg = "mark_critical needs the birth rate lam"
            raise ValueError(msg)
        # Splits the supercritical (mu < lambda) and subcritical regimes.
        critical = (
            alt.Chart(pd.DataFrame({"mu": [opts.lam], "label": ["critical"]}))
            .mark_rule(strokeDash=[6, 4], color="#888888")
            .encode(x="mu:Q", tooltip=["label"])
        )
        layers.append(critical)
    chart = (
        alt.layer(*layers)
        .add_params(selection)
        .properties(width=opts.width, height=opts.height)
    )
    return chart.properties(title=opts.title) if opts.title else chart
