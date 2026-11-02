#!/usr/bin/env python3
"""SCRATCH ONLY — not a report figure.

Flag-sweep plot from the /1.15 CSV (every timing / 1.15 vs measured),
WITHOUT the default_simd line. Hypothetical eyeball, NOT measured data.

Run: /opt/homebrew/bin/python3.11 _scratch_flag_sweep_div1p15_nodefault.py
"""
import redo_plots as R   # reuse the flop model + house style
import pandas as pd

CSV = R.CSV_ROOT / "M1/flag_sweep_2026-06-20/optimized_vs_simd_flag_sweep_2026-06-20_00-30-22_div1.15.csv"

df = pd.read_csv(CSV)
df.columns = [R.normalize_col(c) for c in df.columns]
for c in ["batch", "tokens", "channels", "layers", "avg_ms"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["implementation", "flag_id", "channels", "avg_ms"]).copy()
for c in ["batch", "tokens", "channels", "layers"]:
    df[c] = df[c].astype(int)
df["n"] = df["channels"]; df["label"] = df["implementation"]

sub = df[df["implementation"] == "SIMD C++"].copy()
sub = R.add_flops_per_cycle(sub)

fig, ax = R.new_ax()
# no default_simd line
for fid in ["o2_no_autovec", "o3_no_autovec", "o3_native",
            "o3_native_fast_math", "o3_native_fast_math_unroll"]:
    p = sub[sub["flag_id"] == fid].sort_values("n")
    if p.empty:
        continue
    ax.plot(p["n"], p["gflops"], marker="o", markersize=4.5, lw=1.8,
            color=R.FLAG_COLORS.get(fid, "#000000"), label=R.FLAG_NAMES.get(fid, fid), zorder=3)
R.log2_x(ax)
ax.set_ylabel("SIMD performance [GFlop/s]")
ax.set_ylim(bottom=0)
ax.legend(loc="lower left", fontsize=7)
R.save(fig, "SCRATCH_flag_sweep_div1p15_nodefault.png")
print("scratch plot written (do NOT use in report)")
