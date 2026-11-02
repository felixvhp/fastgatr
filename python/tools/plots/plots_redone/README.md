# Redone report plots

Regenerated with `redo_plots.py` (run with `/opt/homebrew/bin/python3.11 redo_plots.py`).
Applies the review changes to the experiments section. **Roofline is owned separately and is NOT here.**

| file | replaces / role | change vs old report |
|---|---|---|
| `fig1_end_to_end_runtime_log.png` | end-to-end runtime, all versions (log) | kept (runtime → op-count mixing is OK) |
| `fig2_performance_flop_per_cycle.png` | performance, opt/cache/SIMD | **Python dropped** (multithreaded + different op count); also shows the memory→compute transition with channels |
| `fig3_simd_speedup_merged.png` | SIMD speedup | **two speedup plots merged into one** (vs baseline + vs Python) |
| `fig4` (SIMD flag sweep) | one flag sweep instead of two | **NOT regenerated** — no `flag_sweep_*` CSV is committed; rerun the sweep or copy the CSV in |
| `fig5_per_kernel_runtime_simd.png` | per-kernel runtime vs size | **ADDED** — where time goes; attention + EquiLinear dominate at scale |
| `fig6_per_kernel_speedup.png` | per-kernel speedup SIMD/baseline (log y) | **ADDED** — which kernels gained most |

## Numbers to quote in the text (fill the XX / YY placeholders)
- **Best performance:** SIMD ≈ **7.5 flop/cycle** at n=64 ≈ **24 GFlop/s** ≈ **23 %** of the 32 flop/cycle (≈102 GFlop/s) NEON peak. → `XX ≈ 24 GFlop/s`, `YY ≈ 23`.
- **SIMD over baseline:** ~1000–1500× (flattens at large n). Baseline only runs to n=32 (too slow at n=64) — say so in the caption.
- **SIMD over Python (ezGATr):** **1.7–4.5×**, declining with size.
- **Per-kernel speedups (n=64):** EquiLinear **2378×**, GeomProduct 276×, Join 258×, Attention 8×, RMSNorm 2×, GatedGELU 1×.

## The analytical story these unlock
- **Bottleneck shift (fig5 + fig6):** EquiLinear was optimized 2378× (the original hotspot), so it is no longer dominant; attention got only 8×, so **attention is now the top kernel at scale**. This is the "did optimizing the hotspot shift the bottleneck?" answer.
- **Memory→compute transition (fig2):** flop/cycle rises with channel count (≈0 at small n → 7.5 at n=64), i.e. memory-bound at small channels (intensity ≈ 0.4·C), compute-bound at large — ties to the roofline.
- **Why GELU/RMSNorm barely moved:** GELU 1× / RMSNorm 2× — consistent with them being lightly optimized (GELU's `tanh` left scalar).

x-axis is `channels`, which scales together with batch/tokens/layers across the 7 sizes (1→64).
