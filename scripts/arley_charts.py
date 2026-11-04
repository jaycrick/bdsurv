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

import altair as alt
import numpy as np
import pandas as pd

LABEL_CONTOUR = "Contour process (exact)"
LABEL_RUN = "Splitting, per set of trees (one run)"
LABEL_KEPT = "Splitting, per kept tree"
PALETTE = {LABEL_CONTOUR: "#0072B2", LABEL_RUN: "#E69F00", LABEL_KEPT: "#D55E00"}
BIRTH_COLOR = "#009E73"
DEATH_COLOR = "#CC79A7"
GROUP = ["mu", "slope", "T"]


def color_encoding(legend_font_size: int = 14) -> alt.Color:
    """Colour the three cost series, with a clickable legend at the bottom."""
    return alt.Color(
        "method:N",
        scale=alt.Scale(domain=list(PALETTE), range=list(PALETTE.values())),
        legend=alt.Legend(
            title=None, orient="bottom", labelFontSize=legend_font_size, labelLimit=0
        ),
    )


def _band(frame: pd.DataFrame, column: str, label: str) -> pd.DataFrame:
    grouped = frame.groupby(GROUP)[column]
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


def curves_from_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Mean, median and bands of seconds per tree, one block per method label."""
    rows = rows.assign(
        per_kept=np.where(
            rows["n_trees"] > 0, rows["seconds"] / rows["n_trees"], np.nan
        )
    )
    contour_rows = rows[rows["method"] == "contour"]
    split_rows = rows[rows["method"] == "splitting"]
    return pd.concat(
        [
            _band(contour_rows, "seconds", LABEL_CONTOUR),
            _band(split_rows, "seconds", LABEL_RUN),
            _band(split_rows, "per_kept", LABEL_KEPT),
        ],
        ignore_index=True,
    )


def curves_from_draws(rows: pd.DataFrame) -> pd.DataFrame:
    """Mean draws per tree with a 95% interval, one block per method label.

    ``rows`` are the Arley rows of ``draws.csv``.
    Splitting is charged per set of trees (the whole run) and per kept tree.
    """
    rows = rows.assign(
        per_kept=np.where(
            rows["n_trees"] > 0, rows["draws_total"] / rows["n_trees"], np.nan
        )
    )
    contour_rows = rows[rows["method"] == "contour"]
    split_rows = rows[rows["method"] == "splitting"]
    return pd.concat(
        [
            _band(contour_rows, "draws_total", LABEL_CONTOUR),
            _band(split_rows, "draws_total", LABEL_RUN),
            _band(split_rows, "per_kept", LABEL_KEPT),
        ],
        ignore_index=True,
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
    y_title = "mean random draws per tree (log scale)"
    y_scale = alt.Scale(type="log")
    y = alt.Y("mean:Q", scale=y_scale, title=y_title)
    selection = alt.selection_point(fields=["method"], bind="legend")

    def fade(shown: float, hidden: float) -> alt.Opacity:
        return alt.condition(selection, alt.value(shown), alt.value(hidden))

    tooltip = [
        "method",
        alt.Tooltip("slope:Q", format=".3~g", title="slope m"),
        alt.Tooltip("mu:Q", format="g", title="paired constant μ"),
        alt.Tooltip("T:Q", format="g"),
        alt.Tooltip("mean:Q", format=",.0f", title="mean draws"),
        alt.Tooltip("ci_lo:Q", format=",.0f", title="95% CI low"),
        alt.Tooltip("ci_hi:Q", format=",.0f", title="95% CI high"),
        "replicates",
    ]
    base = alt.Chart(data).encode(x=x, color=color_encoding(legend_font_size))
    area = base.mark_area().encode(
        y=alt.Y("ci_lo:Q", scale=y_scale, title=y_title),
        y2="ci_hi:Q",
        opacity=fade(0.15, 0.02),
    )
    line = base.mark_line(strokeWidth=3).encode(y=y, opacity=fade(1.0, 0.08))
    points = base.mark_point(size=70, filled=True).encode(
        y=y, tooltip=tooltip, opacity=fade(1.0, 0.08)
    )
    chart = (
        alt.layer(area, line, points)
        .add_params(selection)
        .properties(width=width, height=height)
    )
    return chart.properties(title=title) if title else chart


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


def x_by_t(ts: list[float]) -> alt.X:
    """Log terminal-time axis with a tick at every grid value."""
    return alt.X(
        "T:Q",
        scale=alt.Scale(type="log"),
        title="terminal time T (log scale)",
        axis=alt.Axis(values=[float(t) for t in ts], format="g", labelOverlap=True),
    )


def x_by_slope(slopes: list[float]) -> alt.X:
    """Log death-rate-slope axis, tight to the data."""
    return alt.X(
        "slope:Q",
        scale=alt.Scale(type="log", nice=False, padding=12),
        title="death-rate slope m (log scale)",
        axis=alt.Axis(
            values=[float(m) for m in slopes], format=".3~g", labelOverlap=True
        ),
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
    y_scale = alt.Scale(type="log")
    unit = "mean sec per tree" if statistic == "mean" else "seconds per tree"
    y_title = f"{unit} (log scale)"
    lo, hi, band_name = (
        ("ci_lo", "ci_hi", "95% CI") if statistic == "mean" else ("lo", "hi", None)
    )
    y = alt.Y(f"{statistic}:Q", scale=y_scale, title=y_title)
    selection = alt.selection_point(fields=["method"], bind="legend")

    def fade(shown: float, hidden: float) -> alt.Opacity:
        return alt.condition(selection, alt.value(shown), alt.value(hidden))

    tooltip = [
        "method",
        alt.Tooltip("slope:Q", format=".3~g", title="slope m"),
        alt.Tooltip("mu:Q", format="g", title="matched constant μ"),
        alt.Tooltip("T:Q", format="g"),
        alt.Tooltip(f"{statistic}:Q", format=".3g", title=f"{statistic} s"),
        alt.Tooltip(f"{lo}:Q", format=".3g", title=f"{band_name or '10th pct'} low"),
        alt.Tooltip(f"{hi}:Q", format=".3g", title=f"{band_name or '90th pct'} high"),
        "replicates",
    ]
    base = alt.Chart(data).encode(x=x, color=color_encoding(legend_font_size))
    area = base.mark_area().encode(
        y=alt.Y(f"{lo}:Q", scale=y_scale, title=y_title),
        y2=f"{hi}:Q",
        opacity=fade(0.15, 0.02),
    )
    line = base.mark_line(strokeWidth=3).encode(y=y, opacity=fade(1.0, 0.08))
    points = base.mark_point(size=70, filled=True).encode(
        y=y, tooltip=tooltip, opacity=fade(1.0, 0.08)
    )
    return (
        alt.layer(area, line, points)
        .add_params(selection)
        .properties(width=width, height=height)
    )


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


def rate_chart(
    slope: float,
    *,
    t_max: float = 100.0,
    y_max: float = 3.0,
    legend_font_size: int = 14,
    annotation_font_size: int = 13,
    t_ticks: list[float] | None = None,
    rate_ticks: list[float] | None = None,
    width: int = 900,
    height: int = 260,
) -> alt.LayerChart:
    r"""Birth rate ``1`` and Arley death rate ``slope * t`` against time.

    The death rate is clipped at ``y_max`` so the crossing at ``t = 1 / slope``
    stays visible even for steep slopes; it keeps rising beyond the plot.
    Before the crossing the birth rate exceeds the death rate (the process is
    supercritical), after it the process is subcritical.
    """
    lines = pd.DataFrame(
        {
            "t": [0.0, t_max, 0.0, t_max],
            "rate": [1.0, 1.0, 0.0, slope * t_max],
            "series": ["birth rate λ = 1"] * 2
            + [f"death rate μ(t) = {slope:.3g} t"] * 2,
        }
    )
    domain = [
        "birth rate λ = 1",
        f"death rate μ(t) = {slope:.3g} t",
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
    return alt.layer(*layers).properties(width=width, height=height)
