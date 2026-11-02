#!/usr/bin/env python3
"""
Plot end-to-end timing for three implementations on one chart:
  Python (baseline), Optimized C++, and SIMD C++.

Reads the per-implementation timing CSVs (columns: implementation, batch,
tokens, channels, layers, avg_ms, min_ms, max_ms) and overlays avg time with
a min/max band per test case. No pandas dependency -- uses stdlib csv.

Usage:
  python plot_timing_three.py --python P.csv --optimized O.csv --simd S.csv --out out.png [--logy]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import asl_style


JOIN_KEYS = ("batch", "tokens", "channels")


def load_results(path: Path) -> dict:
    """Return {(batch, tokens, channels): {avg_s, min_s, max_s}}."""
    rows = {}
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            key = tuple(int(float(r[k])) for k in JOIN_KEYS)
            rows[key] = {
                "avg_s": float(r["avg_ms"]) / 1000.0,
                "min_s": float(r["min_ms"]) / 1000.0,
                "max_s": float(r["max_ms"]) / 1000.0,
            }
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Python vs Optimized vs SIMD timing.")
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--optimized", type=Path, required=True)
    parser.add_argument("--simd", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--logy", action="store_true", help="Use a log y-axis.")
    args = parser.parse_args()

    series = {
        "Python": load_results(args.python),
        "Optimized C++": load_results(args.optimized),
        "SIMD C++": load_results(args.simd),
    }

    common = set(series["Python"])
    for s in series.values():
        common &= set(s)
    cases = sorted(common, key=lambda k: (k[1], k[2], k[0]))
    if not cases:
        raise ValueError("No test cases common to all three CSV files.")

    case_labels = [f"B={b}, T={t}, C={c}" for (b, t, c) in cases]

    asl_style.apply()
    colors = {
        "Python": asl_style.SERIES_COLORS[1],         # dark red (baseline)
        "Optimized C++": asl_style.SERIES_COLORS[0],  # dark blue
        "SIMD C++": asl_style.SERIES_COLORS[2],       # near-black
    }
    markers = {
        "Python": "o",
        "Optimized C++": "s",
        "SIMD C++": "D",
    }

    _, ax = plt.subplots(figsize=(7.5, 4.5))
    x_positions = list(range(len(cases)))

    for label, data in series.items():
        color = colors[label]
        avg = [data[k]["avg_s"] for k in cases]
        lo = [data[k]["min_s"] for k in cases]
        hi = [data[k]["max_s"] for k in cases]
        ax.fill_between(x_positions, lo, hi, color=color, alpha=0.12, linewidth=0)
        ax.plot(
            x_positions,
            avg,
            color=color,
            linewidth=1.8,
            marker=markers[label],
            markersize=5.5,
            markerfacecolor=color,
            markeredgecolor=color,
            label=label,
            zorder=3,
        )
        # Inline series label at the right end of each curve.
        ax.annotate(
            label,
            xy=(x_positions[-1], avg[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9,
            color=color,
            fontweight="bold",
        )

    py = series["Python"]
    simd = series["SIMD C++"]
    for idx, k in enumerate(cases):
        ratio = py[k]["avg_s"] / simd[k]["avg_s"]
        ax.annotate(
            f"{ratio:.1f}x",
            xy=(idx, simd[k]["avg_s"]),
            xytext=(0, -14),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=asl_style.MUTED,
        )

    if args.logy:
        ax.set_yscale("log")
        subtitle = "Runtime [seconds, log scale] vs. test case"
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3g"))
    else:
        subtitle = "Runtime [seconds] vs. test case"
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    asl_style.title(ax, "End-to-End Timing: Python vs Optimized vs SIMD C++", subtitle)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(case_labels, rotation=20, ha="right", fontsize=8)
    # Leave room on the right for the inline series labels.
    ax.set_xlim(-0.3, len(cases) - 1 + 0.3)
    ax.margins(x=0.12)
    asl_style.finish(ax)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
