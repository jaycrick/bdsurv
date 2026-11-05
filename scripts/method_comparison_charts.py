"""Altair charts shared by the method-comparison page and the workshop talk.

The benchmark rows come from ``bench_method_comparison.py``; this module turns
them into the cost-per-tree curves (median and 10th-90th percentile band for
each method) and draws them.
"""

import altair as alt
import numpy as np
import pandas as pd

LABEL_CONTOUR = "Contour process (exact)"
LABEL_RUN = "Splitting, per set of trees (one run)"
LABEL_KEPT = "Splitting, per kept tree"
PALETTE = {LABEL_CONTOUR: "#004488", LABEL_RUN: "#DDAA33", LABEL_KEPT: "#BB5566"}


def color_encoding(
    legend_font_size: int = 14, methods: list[str] | None = None
) -> alt.Color:
    shown = methods or list(PALETTE)
    return alt.Color(
        "method:N",
        scale=alt.Scale(domain=shown, range=[PALETTE[m] for m in shown]),
        legend=alt.Legend(
            title=None, orient="bottom", labelFontSize=legend_font_size, labelLimit=0
        ),
    )


COLOR = color_encoding()


@alt.theme.register("bdsurv", enable=True)
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


def band(frame: pd.DataFrame, column: str, label: str) -> pd.DataFrame:
    grouped = frame.groupby(["mu", "T"])[column]
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
    """Median and percentile band of seconds per tree, one block per method label."""
    rows = rows.assign(
        per_kept=np.where(rows["n_trees"] > 0, rows["seconds"] / rows["n_trees"], np.nan)
    )
    contour_rows = rows[rows["method"] == "contour"]
    split_rows = rows[rows["method"] == "splitting"]
    return pd.concat(
        [
            band(contour_rows, "seconds", LABEL_CONTOUR),
            band(split_rows, "seconds", LABEL_RUN),
            band(split_rows, "per_kept", LABEL_KEPT),
        ],
        ignore_index=True,
    )


def curves_from_draws(rows: pd.DataFrame) -> pd.DataFrame:
    """Mean draws per tree with a 95% interval, one block per method label.

    ``rows`` are the constant-rate rows of ``draws.csv``.
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
            band(contour_rows, "draws_total", LABEL_CONTOUR),
            band(split_rows, "draws_total", LABEL_RUN),
            band(split_rows, "per_kept", LABEL_KEPT),
        ],
        ignore_index=True,
    )


def x_by_t(ts: list[float]) -> alt.X:
    return alt.X(
        "T:Q",
        scale=alt.Scale(type="log"),
        title="terminal time T (log scale)",
        axis=alt.Axis(values=[float(t) for t in ts], format="g", labelOverlap=True),
    )


def x_by_mu(mus: list[float]) -> alt.X:
    return alt.X(
        "mu:Q",
        # Tight to the data, so there is no empty stretch before the first mu.
        scale=alt.Scale(type="log", nice=False, padding=12),
        title="death rate μ (log scale)",
        axis=alt.Axis(values=[float(m) for m in mus], format="g", labelOverlap=True),
    )


def cost_chart(
    data: pd.DataFrame,
    x: alt.X,
    *,
    lam: float,
    log_y: bool = True,
    methods: list[str] | None = None,
    show_band: bool = True,
    statistic: str = "median",
    measure: str = "seconds",
    mark_critical: bool = False,
    legend_font_size: int = 14,
    width: int = 900,
    height: int = 380,
) -> alt.Chart:
    y_scale = alt.Scale(type="log") if log_y else alt.Scale(zero=True)
    if measure == "draws":
        unit = f"{statistic} random draws per tree"
        short = "draws"
    else:
        unit = "mean sec per tree" if statistic == "mean" else "seconds per tree"
        short = "s"
    y_title = f"{unit} (log scale)" if log_y else unit
    # The median gets the 10th-90th percentile band, the mean a 95% interval.
    lo, hi, band_name = (
        ("ci_lo", "ci_hi", "95% CI") if statistic == "mean" else ("lo", "hi", None)
    )
    y = alt.Y(f"{statistic}:Q", scale=y_scale, title=y_title)
    # Clicking a legend entry shows that method and fades the rest;
    # shift-click adds more, double-click resets.
    selection = alt.selection_point(fields=["method"], bind="legend")

    def fade(shown: float, hidden: float) -> alt.Opacity:
        return alt.condition(selection, alt.value(shown), alt.value(hidden))

    tooltip = [
        "method",
        alt.Tooltip(x.shorthand, format="g"),
        alt.Tooltip(f"{statistic}:Q", format=".3g", title=f"{statistic} {short}"),
        alt.Tooltip(f"{lo}:Q", format=".3g", title=f"{band_name or '10th pct'} low"),
        alt.Tooltip(f"{hi}:Q", format=".3g", title=f"{band_name or '90th pct'} high"),
        "replicates",
    ]
    base = alt.Chart(data).encode(x=x, color=color_encoding(legend_font_size, methods))
    area = base.mark_area().encode(
        y=alt.Y(f"{lo}:Q", scale=y_scale, title=y_title),
        y2=f"{hi}:Q",
        opacity=fade(0.15, 0.02),
    )
    line = base.mark_line(strokeWidth=3).encode(y=y, opacity=fade(1.0, 0.08))
    points = base.mark_point(size=70, filled=True).encode(
        y=y, tooltip=tooltip, opacity=fade(1.0, 0.08)
    )
    layers = [area, line, points] if show_band else [line, points]
    if mark_critical:
        # Splits the supercritical (mu < lambda) and subcritical regimes.
        critical = (
            alt.Chart(pd.DataFrame({"mu": [lam], "label": ["critical"]}))
            .mark_rule(strokeDash=[6, 4], color="#888888")
            .encode(x="mu:Q", tooltip=["label"])
        )
        layers.append(critical)
    return (
        alt.layer(*layers)
        .add_params(selection)
        .properties(width=width, height=height)
    )
