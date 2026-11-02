#!/usr/bin/env python3
"""
Shared "ASL / Spiral-group" plot style.

Reproduces the grey-panel performance-plot look used in the course example
report and in the reference student reports: a light-grey plotting panel with
white horizontal gridlines, no surrounding frame, a bold left-aligned title and
a smaller descriptive subtitle that carries the axis quantities (so the y-axis
label is dropped), muted line colours, and filled markers that stay
distinguishable in greyscale.

Usage:
    import asl_style
    asl_style.apply()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ...
    asl_style.title(ax, "Runtime Comparison of Different Implementations",
                        "Runtime [seconds] vs. #shapes")
    asl_style.finish(ax)
"""

from __future__ import annotations

import matplotlib.pyplot as plt

# Panel / ink colours taken from the reference reports.
PANEL = "#E6E6E6"      # light-grey plotting area
GRID = "#FFFFFF"       # white horizontal gridlines
INK = "#000000"        # title / strong text
MUTED = "#595959"      # axis ticks, subtitle, inline annotations

# Muted, print-friendly series colours (stay readable on grey, and when the
# PDF is printed in greyscale). Ordered so the first few are well separated.
SERIES_COLORS = [
    "#1F3B73",  # dark blue
    "#A4262C",  # dark red
    "#2E2E2E",  # near-black
    "#3C7D3C",  # dark green
    "#7A5195",  # muted purple
    "#B8860B",  # dark goldenrod
]

# Marker cycle so series are distinguishable without relying on colour.
SERIES_MARKERS = ["o", "s", "D", "^", "v", "P"]


def apply() -> None:
    """Set rcParams for the ASL grey-panel style."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Calibri", "Helvetica", "Arial", "DejaVu Sans"],
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": PANEL,
            "axes.edgecolor": PANEL,
            "axes.linewidth": 0.0,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 1.1,
            "grid.alpha": 1.0,
            "text.color": INK,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
        }
    )


def title(ax, main: str, subtitle: str | None = None) -> None:
    """Bold left-aligned title with an optional smaller grey subtitle line.

    The subtitle is where the axis quantities live, e.g.
    "Runtime [seconds] vs. #shapes" -- matching the reference reports.
    """
    if subtitle:
        ax.set_title(subtitle, loc="left", fontsize=10, color=MUTED, pad=6)
        ax.text(
            0.0,
            1.06,
            main,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            color=INK,
            ha="left",
            va="bottom",
        )
    else:
        ax.set_title(main, loc="left", fontsize=12, fontweight="bold", color=INK, pad=8)


def finish(ax) -> None:
    """Strip the frame and vertical gridlines, keep white horizontal gridlines."""
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=1.1)
    ax.grid(axis="x", visible=False)
    ax.tick_params(length=0)
