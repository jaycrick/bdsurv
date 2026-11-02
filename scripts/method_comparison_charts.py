"""Altair charts shared by the method-comparison page and the workshop talk.

The benchmark rows come from ``bench_method_comparison.py``; this module turns
them into the cost-per-tree curves (median and 10th-90th percentile band for
each method) and draws them.
"""

import altair as alt
import numpy as np
import pandas as pd

LABEL_CONTOUR = "Contour process (exact)"
LABEL_RUN = "Splitting, per independent tree (one run)"
LABEL_KEPT = "Splitting, per kept tree"
PALETTE = {LABEL_CONTOUR: "#0072B2", LABEL_RUN: "#E69F00", LABEL_KEPT: "#D55E00"}
COLOR = alt.Color(
    "method:N",
    scale=alt.Scale(domain=list(PALETTE), range=list(PALETTE.values())),
    legend=alt.Legend(title=None, orient="bottom", labelFontSize=14, labelLimit=0),
)


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
    out = pd.DataFrame(
        {
            "median": grouped.median(),
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
    mark_critical: bool = False,
    width: int = 900,
    height: int = 380,
) -> alt.Chart:
    y_scale = alt.Scale(type="log") if log_y else alt.Scale(zero=True)
    y_title = "seconds per tree (log scale)" if log_y else "seconds per tree"
    y = alt.Y("median:Q", scale=y_scale, title=y_title)
    # Clicking a legend entry shows that method and fades the rest;
    # shift-click adds more, double-click resets.
    selection = alt.selection_point(fields=["method"], bind="legend")

    def fade(shown: float, hidden: float) -> alt.Opacity:
        return alt.condition(selection, alt.value(shown), alt.value(hidden))

    tooltip = [
        "method",
        alt.Tooltip(x.shorthand, format="g"),
        alt.Tooltip("median:Q", format=".3g", title="median s"),
        alt.Tooltip("lo:Q", format=".3g", title="10th pct"),
        alt.Tooltip("hi:Q", format=".3g", title="90th pct"),
        "replicates",
    ]
    base = alt.Chart(data).encode(x=x, color=COLOR)
    area = base.mark_area().encode(
        y=alt.Y("lo:Q", scale=y_scale, title=y_title),
        y2="hi:Q",
        opacity=fade(0.15, 0.02),
    )
    line = base.mark_line(strokeWidth=3).encode(y=y, opacity=fade(1.0, 0.08))
    points = base.mark_point(size=70, filled=True).encode(
        y=y, tooltip=tooltip, opacity=fade(1.0, 0.08)
    )
    layers = [area, line, points]
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
