#!/usr/bin/env python3
"""
Redone report plots in a single clean academic style (everything except the
roofline, which is owned separately, and the flag sweep, redone later).

Style: white background, light gridlines, normal weight, consistent
per-implementation colors across every figure, caption-driven (no in-plot
titles), uniform size, performance in flops/cycle throughout.

  Fig 1  end-to-end runtime, all versions, log
  Fig 2  performance (flop/cycle), opt/cache/SIMD   (Python dropped)
  Fig 3  merged speedup: SIMD/baseline and SIMD/python
  Fig 5  per-kernel runtime vs size, SIMD
  Fig 6  per-kernel speedup SIMD-vs-baseline (log y)

Run:  /opt/homebrew/bin/python3.11 redo_plots.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ----------------------------------------------------------------------------- paths
MACHINE = "M1"
BASE = Path(__file__).resolve()
while not ((BASE / "Results").exists() and (BASE / "Code").exists()):
    if BASE.parent == BASE:
        raise RuntimeError("Could not find team80 repository root")
    BASE = BASE.parent

CSV_ROOT = BASE / "Results" / "timing_results" / "CSVs"
OUT_DIR = Path(__file__).resolve().parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

FLOP_TOOLS_DIR = BASE / "Code" / "python" / "tools"
sys.path.insert(0, str(FLOP_TOOLS_DIR))
from cpp_flop_counter import (  # noqa: E402
    Build, OpWeights,
    EquiLinearCfg, GeometricProductCfg, EquiJoinCfg, EquiRMSNormCfg,
    ScalerGatedGeluCfg, EquiGeometricAttentionCfg,
    equi_linear_count, geometric_product_count, equi_join_count,
    equi_rms_norm_count, scaler_gated_gelu_count, equi_geometric_attention_count,
)

# ----------------------------------------------------------------------------- house style (gray panel, major gridlines only)
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#e3e3e3",
    "axes.edgecolor": "#000000",
    "axes.labelcolor": "#000000",
    "axes.labelsize": 13,
    "xtick.color": "#000000", "ytick.color": "#000000",
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "text.color": "#000000",
    "grid.color": "white",
    "legend.fontsize": 10, "legend.frameon": True,
    "legend.facecolor": "white", "legend.edgecolor": "#cccccc",
    "font.family": "DejaVu Sans", "font.weight": "bold",
    "axes.titleweight": "bold",
    "figure.dpi": 110,
})

# one consistent color per implementation, used in every figure (readable on the gray panel)
COLORS = {
    "Baseline C++":            "#2b2b2b",  # near-black (the slow reference)
    "Optimized C++":           "#2166ac",  # deep blue
    "Optimized-for-cache C++": "#e08214",  # amber
    "SIMD C++":                "#b2182b",  # deep red (the hero)
    "Python":                  "#1b7837",  # forest green (external reference)
}
MARKERS = {
    "Baseline C++": "o", "Optimized C++": "s", "Optimized-for-cache C++": "^",
    "SIMD C++": "D", "Python": "v",
}
KERNEL_COLORS = {
    "equi_linear": "#b2182b", "geometric_product": "#2166ac", "equi_join": "#762a83",
    "equi_rms_norm": "#1b7837", "scaler_gated_gelu": "#e08214",
    "equi_geometric_attention": "#7f3b08",
}
KERNEL_SHORT = {
    "equi_linear": "EquiLinear", "geometric_product": "GeomProduct", "equi_join": "Join",
    "equi_rms_norm": "RMSNorm", "scaler_gated_gelu": "GatedGELU",
    "equi_geometric_attention": "Attention",
}
FIGSIZE = (6.2, 4.0)
XLABEL = "Input size $N$\n(batch $=8$, layers $=N$, tokens $=2N$, channels $=N$)"
M1_CLOCK_GHZ = 3.2

CSV_PATHS = {
    "end_to_end": {
        "Baseline C++": CSV_ROOT / "M1/end_to_end/c_timing_results_2026-05-29_12-17-11.csv",
        "Python": CSV_ROOT / "M1/end_to_end/python_timing_results_2026-05-30_13-27-38.csv",
        "Optimized C++": CSV_ROOT / "M1/end_to_end/optimized_c_timing_results_2026-05-30_15-24-37.csv",
        "Optimized-for-cache C++": CSV_ROOT / "M1/end_to_end/optimized_for_cache_c_timing_results_2026-05-30_15-06-18.csv",
        "SIMD C++": CSV_ROOT / "M1/end_to_end/simd_c_timing_results_2026-05-31_17-05-27.csv",
    },
    "per_kernel": {
        "Baseline C++": CSV_ROOT / "M1/per_kernel/baseline_c_per_kernel_seconds_results_2026-06-01_08-01-50.csv",
        "Python": CSV_ROOT / "M1/per_kernel/python_per_kernel_seconds_results_2026-06-01_08-57-17.csv",
        "SIMD C++": CSV_ROOT / "M1/per_kernel/simd_c_per_kernel_seconds_results_2026-06-01_08-35-31.csv",
        "Optimized-for-cache C++": CSV_ROOT / "M1/per_kernel/optimized_for_cache_c_per_kernel_seconds_results_2026-06-01_08-48-04.csv",
    },
}
BUILD_FOR_LABEL = {
    "Python": Build.BASELINE, "Optimized C++": Build.OPTIMIZED,
    "Optimized-for-cache C++": Build.OPTIMIZED, "SIMD C++": Build.SIMD,
}

# ----------------------------------------------------------------------------- loaders
def normalize_col(name):
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")

def load_csv(path, label, kind):
    path = Path(path)
    if not path.exists():
        print(f"  skip missing {kind} CSV for {label}: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = [normalize_col(c) for c in df.columns]
    rename = {"b": "batch", "t": "tokens", "c": "channels", "l": "layers"}
    df = df.rename(columns={c: rename.get(c, c) for c in df.columns})
    if "layers" not in df.columns:
        df["layers"] = df["channels"]
    for col in ["batch", "tokens", "channels", "layers", "avg_ms", "min_ms", "max_ms"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["batch", "tokens", "channels", "layers", "avg_ms"]).copy()
    for col in ["batch", "tokens", "channels", "layers"]:
        df[col] = df[col].astype(int)
    df["label"] = label
    df["n"] = df["channels"]
    return df

def load_group(kind):
    frames = [load_csv(p, lbl, kind) for lbl, p in CSV_PATHS[kind].items()]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

# ----------------------------------------------------------------------------- flop/cycle model
# Architecture constants of ASLMVOnlyGATrModel (MVOnlyGATrConfig defaults):
# hidden=32, intermediate=32, heads=4. The benchmark only scales channels_in,
# tokens and num_layers with N; the INTERNAL width stays at HIDDEN=32. The old
# model counted everything at width N (wrong). This mirrors roofline_m1_vec.py's
# full_model_weighted_flops, which is the validated model.
HIDDEN, HEADS, ATTN_CQK, ATTN_CV = 32, 4, 32, 32

def weighted_full_model_flops(batch, tokens, channels, layers, impl_label):
    build = BUILD_FOR_LABEL.get(impl_label, Build.OPTIMIZED)
    w = OpWeights()
    B, T = batch, tokens

    def lin(ci, co):
        return equi_linear_count(EquiLinearCfg(batch=B, tokens=T, channels_in=ci,
                                               channels_out=co, bias=True, build=build)).weighted_flops(w)
    def rms(c):
        return equi_rms_norm_count(EquiRMSNormCfg(n=B * T, channels=c, hasWt=True, mOne=False, build=build)).weighted_flops(w)
    def gp(m):
        return geometric_product_count(GeometricProductCfg(m=m, build=build)).weighted_flops(w)
    def join(m):
        return equi_join_count(EquiJoinCfg(m=m, reference=True, x_nnz_frac=1.0,
                                           include_kernel_build=False, noCompOpt=True, build=build)).weighted_flops(w)
    def gelu(m):
        return scaler_gated_gelu_count(ScalerGatedGeluCfg(m=m, build=build)).weighted_flops(w)
    def attn():
        return equi_geometric_attention_count(EquiGeometricAttentionCfg(
            batch=B, heads=HEADS, t_query=T, t_key=T, channels_qk=ATTN_CQK,
            channels_v=ATTN_CV, kinds=["ipa", "daa"], build=build)).weighted_flops(w)

    total = lin(channels, HIDDEN)                         # embedding C_in -> 32
    block = (rms(HIDDEN) + lin(HIDDEN, 128) + gp(B * T * HIDDEN) + join(B * T * HIDDEN)
             + lin(64, HIDDEN) + gelu(B * T * HIDDEN) + lin(HIDDEN, HIDDEN)   # mlp branch
             + rms(HIDDEN) + lin(HIDDEN, 384) + attn() + lin(128, HIDDEN))    # attn branch
    total += layers * block
    total += lin(HIDDEN, 1)                               # head 32 -> 1
    return total

def add_flops_per_cycle(df):
    df = df.copy()
    df["wflops"] = df.apply(
        lambda r: weighted_full_model_flops(int(r["batch"]), int(r["tokens"]),
                                            int(r["channels"]), int(r["layers"]), r["label"]), axis=1)
    seconds = df["avg_ms"] / 1000.0
    df["flop_per_cycle"] = df["wflops"] / (seconds * M1_CLOCK_GHZ * 1e9)
    df["gflops"] = df["wflops"] / seconds / 1e9      # GFlop/s = flop/cycle * 3.2GHz
    return df

# ----------------------------------------------------------------------------- helpers
def new_ax():
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_facecolor("#e3e3e3")  # gray panel, like before
    ax.grid(True, which="major", axis="y", color="white", linewidth=1.2, zorder=0)  # major only -> clean
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#000000")
    ax.tick_params(axis="y", which="both", left=False)  # no floating tick marks (left spine hidden)
    return fig, ax

def log2_x(ax):
    ax.set_xscale("log", base=2)
    ax.set_xticks(SIZES)
    ax.set_xticklabels([str(s) for s in SIZES])
    ax.set_xlabel(XLABEL, fontsize=11, linespacing=1.3)

def save(fig, name):
    out = OUT_DIR / name
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {out.name}")

# ----------------------------------------------------------------------------- load
print("Loading CSVs...")
e2e = load_group("end_to_end")
perk = load_group("per_kernel")
SIZES = sorted(e2e["n"].unique()) if not e2e.empty else []

# ----------------------------------------------------------------------------- Fig 1: runtime, all versions
def fig1_runtime():
    order = ["Baseline C++", "Optimized C++", "Optimized-for-cache C++", "SIMD C++", "Python"]
    fig, ax = new_ax()
    for lbl in order:
        d = e2e[e2e["label"] == lbl].sort_values("n")
        if d.empty:
            continue
        ax.plot(d["n"], d["avg_ms"], marker=MARKERS[lbl], markersize=4.5, lw=1.8,
                color=COLORS[lbl], label=lbl, zorder=3)
    ax.set_yscale("log"); log2_x(ax)
    ax.set_ylabel("Runtime [ms]")
    ax.legend(loc="lower right", fontsize=8)  # inside, bottom-right -> fits under the rising lines
    save(fig, "fig1_end_to_end_runtime_log.png")

# ----------------------------------------------------------------------------- Fig 2: performance, no python
def fig2_performance():
    labels = ["Optimized C++", "Optimized-for-cache C++", "SIMD C++"]
    d = add_flops_per_cycle(e2e[e2e["label"].isin(labels)])
    fig, ax = new_ax()
    for lbl in labels:
        p = d[d["label"] == lbl].sort_values("n")
        if p.empty:
            continue
        ax.plot(p["n"], p["gflops"], marker=MARKERS[lbl], markersize=4.5, lw=1.8,
                color=COLORS[lbl], label=lbl, zorder=3)
    log2_x(ax)
    ax.set_ylabel("Performance [GFlop/s]")
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower right", fontsize=8)
    save(fig, "fig2_performance_flop_per_cycle.png")

# ----------------------------------------------------------------------------- Fig 3: merged speedup
def fig3_merged_speedup():
    piv = e2e.pivot_table(index="n", columns="label", values="avg_ms", aggfunc="mean").sort_index()
    if "SIMD C++" not in piv.columns:
        print("  fig3 skipped (no SIMD data)"); return
    fig, ax = new_ax()
    if "Baseline C++" in piv.columns:
        s = (piv["Baseline C++"] / piv["SIMD C++"]).dropna()
        ax.plot(s.index, s.values, marker="o", markersize=4.5, lw=1.8,
                color=COLORS["Baseline C++"], label="vs Baseline C++", zorder=3)
    if "Python" in piv.columns:
        s = (piv["Python"] / piv["SIMD C++"]).dropna()
        ax.plot(s.index, s.values, marker="v", markersize=4.5, lw=1.8,
                color=COLORS["Python"], label="vs Python (ezGATr)", zorder=3)
    ax.axhline(1.0, color="#999999", lw=0.9, ls="--", zorder=1)
    ax.set_yscale("log"); log2_x(ax)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}x"))
    ax.set_ylabel("Speedup (SIMD C++)")
    ax.legend(loc="center right")
    save(fig, "fig3_simd_speedup_merged.png")

# ----------------------------------------------------------------------------- Fig 5: per-kernel runtime (SIMD)
def fig5_per_kernel_runtime():
    if perk.empty or "kernel" not in perk.columns:
        print("  fig5 skipped (no per-kernel data)"); return
    d = perk[perk["label"] == "SIMD C++"]
    if d.empty:
        print("  fig5 skipped (no SIMD per-kernel)"); return
    fig, ax = new_ax()
    for kern in KERNEL_COLORS:
        k = d[d["kernel"] == kern].sort_values("n")
        if k.empty:
            continue
        ax.plot(k["n"], k["avg_ms"], marker="o", markersize=4.0, lw=1.7,
                color=KERNEL_COLORS[kern], label=KERNEL_SHORT.get(kern, kern), zorder=3)
    ax.set_yscale("log"); log2_x(ax)
    ax.set_ylabel("SIMD per-kernel runtime [ms]")
    ax.legend(loc="upper left", ncol=2)
    save(fig, "fig5_per_kernel_runtime_simd.png")

# ----------------------------------------------------------------------------- Fig 6: per-kernel speedup (scalar + SIMD vs baseline)
def _speedup(label, kern, big):
    base = perk[(perk["label"] == "Baseline C++") & (perk["kernel"] == kern) & (perk["n"] == big)]["avg_ms"]
    cur = perk[(perk["label"] == label) & (perk["kernel"] == kern) & (perk["n"] == big)]["avg_ms"]
    if len(base) and len(cur) and cur.values[0] > 0:
        return base.values[0] / cur.values[0]
    return None

def fig6_per_kernel_speedup():
    if perk.empty or "kernel" not in perk.columns:
        print("  fig6 skipped (no per-kernel data)"); return
    SCALAR = "Optimized-for-cache C++"   # best non-vectorized build
    builds = [b for b in (SCALAR, "SIMD C++") if not perk[perk["label"] == b].empty]
    base = perk[perk["label"] == "Baseline C++"]
    if base.empty or not builds:
        print("  fig6 skipped (need baseline + scalar/simd)"); return
    ns = set(base["n"])
    for b in builds:
        ns &= set(perk[perk["label"] == b]["n"])
    if not ns:
        print("  fig6 skipped (no common size)"); return
    big = max(ns)
    # order kernels by SIMD (or last build) speedup, descending
    order_build = "SIMD C++" if "SIMD C++" in builds else builds[-1]
    kernels = sorted(KERNEL_COLORS, key=lambda k: (_speedup(order_build, k, big) or 0), reverse=True)
    legend_name = {SCALAR: "Optimized-for-cache (scalar)", "SIMD C++": "SIMD C++"}
    fig, ax = new_ax()
    width = 0.8 / len(builds)
    for i, b in enumerate(builds):
        offs, vals = [], []
        for xi, kern in enumerate(kernels):
            sp = _speedup(b, kern, big)
            if sp is not None:
                offs.append(xi + (i - (len(builds) - 1) / 2) * width); vals.append(sp)
        bars = ax.bar(offs, vals, width=width, color=COLORS[b], edgecolor="#333333",
                      linewidth=0.5, label=legend_name.get(b, b), zorder=3)
        for o, v in zip(offs, vals):
            ax.text(o, v * 1.12, f"{v:.0f}×", ha="center", va="bottom", fontsize=7.5)
    ax.set_yscale("log")
    ax.set_xticks(range(len(kernels)))
    ax.set_xticklabels([KERNEL_SHORT[k] for k in kernels], rotation=20, ha="right")
    ax.set_ylim(0.8, max(_speedup(order_build, kernels[0], big) or 1, 10) * 3.5)
    ax.set_ylabel(f"Speedup over baseline ($N={big}$)")
    ax.legend(loc="upper right")
    save(fig, "fig6_per_kernel_speedup.png")

# ----------------------------------------------------------------------------- Fig 4: SIMD flag sweep
FLAG_NAMES = {
    "o2_no_autovec": "O2, no auto-vec",
    "o3_no_autovec": "O3, no auto-vec",
    "o3_native": "O3 native",
    "o3_native_fast_math": "O3 native, fast-math",
    "o3_native_fast_math_unroll": "O3 native, fast-math, unroll",
}
FLAG_COLORS = {
    "o2_no_autovec": "#762a83", "o3_no_autovec": "#2166ac", "o3_native": "#1b7837",
    "o3_native_fast_math": "#e08214", "o3_native_fast_math_unroll": "#b2182b",
}

DEFAULT_SIMD_FLAGS = "-O3 -DNDEBUG -funroll-loops -mcpu=native"  # setup.py SIMD build (auto-vec on)

def fig4_flag_sweep(include_default=False):
    files = sorted(CSV_ROOT.glob("**/flag_sweep_*/optimized_vs_simd_flag_sweep_*.csv"))
    if not files:
        print("  fig4 skipped (no flag_sweep CSV found)"); return
    df = pd.read_csv(files[-1]); df.columns = [normalize_col(c) for c in df.columns]
    for c in ["batch", "tokens", "channels", "layers", "avg_ms"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["implementation", "flag_id", "channels", "avg_ms"]).copy()
    for c in ["batch", "tokens", "channels", "layers"]:
        df[c] = df[c].astype(int)
    df["n"] = df["channels"]; df["label"] = df["implementation"]
    sub = df[df["implementation"] == "SIMD C++"].copy()
    if sub.empty:
        print("  fig4 skipped (no SIMD rows)"); return
    sub = add_flops_per_cycle(sub)
    fig, ax = new_ax()
    if include_default:
        # the default (setup.py) build is the regular SIMD end-to-end run, not in the sweep CSV
        dflt = add_flops_per_cycle(e2e[e2e["label"] == "SIMD C++"].copy()).sort_values("n")
        if not dflt.empty:
            ax.plot(dflt["n"], dflt["gflops"], marker="o", markersize=6, lw=2.8,
                    color="#000000", label=DEFAULT_SIMD_FLAGS, zorder=4)
    for fid in ["o2_no_autovec", "o3_no_autovec", "o3_native",
                "o3_native_fast_math", "o3_native_fast_math_unroll"]:
        p = sub[sub["flag_id"] == fid].sort_values("n")
        if p.empty:
            continue
        ax.plot(p["n"], p["gflops"], marker="o", markersize=4.5, lw=1.8,
                color=FLAG_COLORS.get(fid, "#000000"), label=FLAG_NAMES.get(fid, fid), zorder=3)
    log2_x(ax)
    ax.set_ylabel("SIMD performance [GFlop/s]")
    ax.set_ylim(bottom=0)  # room below the (clustered, flat) lines for the legend
    ax.legend(loc="lower left", fontsize=6.5 if include_default else 7)
    name = "fig4_simd_flag_sweep_with_default.png" if include_default else "fig4_simd_flag_sweep.png"
    save(fig, name)

def fig4_flag_sweep_measured(sweep_dir="M1/flag_sweep_2026-06-20"):
    """Flag sweep where the default config (flag_id 'default_simd') is measured in
    the SAME sweep run, so it sits on the exact same footing as the other flags."""
    files = sorted((CSV_ROOT / sweep_dir).glob("optimized_vs_simd_flag_sweep_*.csv"))
    if not files:
        print(f"  fig4-measured skipped (no CSV in {sweep_dir})"); return
    df = pd.read_csv(files[-1]); df.columns = [normalize_col(c) for c in df.columns]
    for c in ["batch", "tokens", "channels", "layers", "avg_ms"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["implementation", "flag_id", "channels", "avg_ms"]).copy()
    for c in ["batch", "tokens", "channels", "layers"]:
        df[c] = df[c].astype(int)
    df["n"] = df["channels"]; df["label"] = df["implementation"]
    sub = df[df["implementation"] == "SIMD C++"].copy()
    if sub.empty:
        print("  fig4-measured skipped (no SIMD rows)"); return
    sub = add_flops_per_cycle(sub)
    fig, ax = new_ax()
    dflt = sub[sub["flag_id"] == "default_simd"].sort_values("n")
    if not dflt.empty:
        ax.plot(dflt["n"], dflt["gflops"], marker="o", markersize=6, lw=2.8,
                color="#000000", label=DEFAULT_SIMD_FLAGS, zorder=4)
    for fid in ["o2_no_autovec", "o3_no_autovec", "o3_native",
                "o3_native_fast_math", "o3_native_fast_math_unroll"]:
        p = sub[sub["flag_id"] == fid].sort_values("n")
        if p.empty:
            continue
        ax.plot(p["n"], p["gflops"], marker="o", markersize=4.5, lw=1.8,
                color=FLAG_COLORS.get(fid, "#000000"), label=FLAG_NAMES.get(fid, fid), zorder=3)
    log2_x(ax)
    ax.set_ylabel("SIMD performance [GFlop/s]")
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower left", fontsize=6.5)
    save(fig, "fig4_simd_flag_sweep_measured_default.png")

# ----------------------------------------------------------------------------- run
if __name__ == "__main__":
    print(f"Output -> {OUT_DIR}")
    for fn in (fig1_runtime, fig2_performance, fig3_merged_speedup,
               fig5_per_kernel_runtime, fig6_per_kernel_speedup):
        try:
            fn()
        except Exception as e:
            print(f"  {fn.__name__} FAILED: {type(e).__name__}: {e}")
    # two flag-sweep versions: without and with the default config
    for with_default in (False, True):
        try:
            fig4_flag_sweep(include_default=with_default)
        except Exception as e:
            print(f"  fig4_flag_sweep(default={with_default}) FAILED: {type(e).__name__}: {e}")
    # flag sweep with the default measured in the same sweep run
    try:
        fig4_flag_sweep_measured()
    except Exception as e:
        print(f"  fig4_flag_sweep_measured FAILED: {type(e).__name__}: {e}")
    print("Done.")
