"""Altair charts shared by the method-comparison page and the workshop talk.

The benchmark rows come from ``bench_method_comparison.py``; this module turns
them into the cost-per-tree curves (median and 10th-90th percentile band for
each method) and draws them.
The labels, palette, summaries and the chart itself are in ``_charts_common``.
"""

import sys
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
GROUP = ["mu", "T"]
color_encoding = common.color_encoding
x_by_t = common.x_by_t
COLOR = color_encoding()

common.register_theme()


def curves_from_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Median and percentile band of seconds per tree, one block per method label."""
    return common.curves_from_rows(rows, GROUP)


def curves_from_draws(rows: pd.DataFrame) -> pd.DataFrame:
    """Mean draws per tree with a 95% interval, one block per method label.

    ``rows`` are the constant-rate rows of ``draws.csv``.
    Splitting is charged per set of trees (the whole run) and per kept tree.
    """
    return common.curves_from_draws(rows, GROUP)


def x_by_mu(mus: list[float]) -> alt.X:
    """Return the log-scale x encoding for the death rate, with ticks at ``mus``."""
    return common.log_axis("mu", "death rate μ (log scale)", mus, tight=True)


def cost_chart(
    data: pd.DataFrame, x: alt.X, *, lam: float, **overrides: object
) -> alt.LayerChart:
    """Build the layered cost-per-tree chart, one curve (and optional band) per method.

    Parameters
    ----------
    data
        Curves from ``curves_from_rows`` or ``curves_from_draws``.
    x
        The x encoding, from ``x_by_t`` or ``x_by_mu``.
    lam
        Birth rate, used to place the critical-point rule when ``mark_critical``.
    **overrides
        Any field of ``_charts_common.CostChartOptions`` (``log_y``, ``methods``,
        ``show_band``, ``statistic``, ``measure``, ``mark_critical``,
        ``legend_font_size``, ``width``, ``height``); anything else is a
        ``TypeError``.
    """
    return common.cost_chart(data, x, common.CostChartOptions(lam=lam), **overrides)
