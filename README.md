# FASTGATr: a fast single-core C++ implementation of the Geometric Algebra Transformer

FASTGATr reimplements the forward pass of the Geometric Algebra Transformer (GATr, Brehmer et al., NeurIPS 2023) in C++ and optimises it for a single CPU core. Starting from the PyTorch reference implementation [EzGATr](https://github.com/Guest400123064/ezgatr), we replaced six layers with hand-written and generated C++ kernels exposed to Python through pybind11, validated every build against the reference, and measured the result on an Apple M1.

The final SIMD build runs **1.7 to 4.5x faster than the PyTorch reference** and up to 1500x faster than a direct C++ translation, with end-to-end mean absolute percentage error below 0.1%.

Built by a team of four for the Advanced Systems Lab ("How to Write Fast Numerical Code") at ETH Zurich, spring 2026, supervised by Mohamed-Hicham Leghettas and Prof. Markus Püschel. The full write-up is in [report.pdf](report.pdf).

## Results

Measured on one Apple M1 core, 10 warm-up and 30 timed runs per size, across seven problem sizes from (batch, tokens, channels, layers) = (8, 2, 1, 1) to (8, 128, 64, 64).

| Measurement | Value |
|---|---|
| End-to-end speedup, SIMD C++ vs PyTorch (EzGATr) | 1.7x to 4.5x across all sizes |
| End-to-end speedup, SIMD C++ vs baseline C++ | up to 1500x |
| Sustained performance, SIMD build | 27 to 32 GFlop/s (36% of the measured 89.4 GFlop/s NEON peak) |
| Sustained performance, best scalar build | about 14 GFlop/s |
| Operation-count reduction from sparsity | 96x EquiLinear, 21x geometric product, 51x equivariant join |
| Correctness | end-to-end MAPE below 0.1% against PyTorch |

Per-kernel speedup over the baseline C++ at the largest size: EquiLinear 2378x, geometric product 276x, equivariant join 258x, geometric attention about 8x, RMSNorm and gated GELU 1 to 2x.

## What is in scope

Six operations from GATr, each a drop-in replacement for the EzGATr PyTorch operator:

- **EquiLinear**: equivariant linear layer built from a fixed sparse 9x16x16 basis. The dominant cost at scale.
- **Geometric product** and **equivariant join**: bilinear maps through fixed structure tensors with 192 and 81 non-zero entries out of 4096.
- **Equivariant geometric attention**: pairwise scores from inner-product and directional features over multivector channels, then softmax and value accumulation.
- **EquiRMSNorm** and **scalar-gated GELU**: pointwise normalisation and activation.

## How the speedup was obtained

The work went through four kernel sets, all kept in this repository so each step can be measured in isolation.

| Stage | Directory | What changed |
|---|---|---|
| Baseline | `cpp/baseline` | Direct translation of EzGATr's maths into C++ pybind extensions: dense evaluation, coefficient tensors loaded at runtime. |
| Optimised (scalar) | `cpp/optimized` | Sparsity exploitation (2304 to 24 terms in EquiLinear, 4096 to 192 and 81 in the bilinear maps), generated kernels, signs as add/sub instead of multiply, independent per-blade accumulators, scalar replacement, a closed-form attention score, operator fusion. |
| Cache-optimised | `cpp/optimized_for_cache` | Repacked EquiLinear weights from [Cout, Cin, 9] to [Cin, Cout, 9] with normalisation folded in. Instruction-delivery stalls fell from 41.5% to 6.7% in Xcode Instruments. |
| SIMD | `cpp/simd` | Hand-vectorised hot kernels with 128-bit ARM NEON intrinsics, choosing a vectorisation axis per kernel: within a multivector for EquiLinear and RMSNorm, across the batch (AoS to SoA transpose) for the geometric product and join, across channels for attention. |

Two findings shaped the design. Most of the gain came from removing work rather than approaching the arithmetic peak: sparsity cut flops by about 100x but left bytes moved unchanged, so the kernels end up memory-bound, and a standard DRAM roofline misclassifies them. A cache-aware roofline with measured L1, L2 and DRAM bandwidths explains the results. After optimisation the bottleneck moved from EquiLinear to attention, whose softmax and value accumulation are streaming passes with low arithmetic intensity.

## Repository layout

```
cpp/
  baseline/             direct C++ translation of the PyTorch operators
  optimized/            scalar kernels with sparsity exploitation and fusion
  optimized_for_cache/  repacked data layout
  simd/                 ARM NEON kernels
  generated_kernels/    sparse kernels emitted by the code generator
basis/                  structure tensors for the bilinear maps (PyTorch files)
python/
  ezgatr_extensions/    monkey-patching layer that swaps C++ kernels into the EzGATr model
  setup.py              builds the pybind11 extensions
  example/              end-to-end forward pass
  tests/validation/     per-kernel and end-to-end correctness tests against PyTorch
  tests/timing/         timing harnesses in seconds and in cycles
  tools/                code generators, flop and data-movement counters, profiling, plots
report.pdf              the project report
```

The sparse kernels are never written by hand. `python/tools/generate_sparse_kernel.py` and `python/tools/generate_simd_sparse_kernels.py` walk the structure tensors in `basis/`, keep only the non-zero terms, and emit the scalar and NEON C++ in `cpp/generated_kernels/`.

## Building and running

The NEON kernels target ARM64 (Apple Silicon). The baseline and scalar builds compile on x86-64 as well.

```bash
pip install -r requirements.txt   # or install torch, numpy, pybind11, ezgatr, pytest
cd python
pip install -e .                  # builds the C++ extensions into python/ops/
python example/main.py            # end-to-end forward pass
pytest tests/validation           # correctness against the PyTorch reference
```

`python/tools/build_fast.py build_ext --inplace --force` rebuilds the extensions with the optimised flag set. See [BUILD_NOTES.md](BUILD_NOTES.md) for the original build and tooling notes. Timing scripts write CSV files to a `Results/` directory at the repository root, which is not tracked.

## Team

Equal contributions by Hong Phuc Vu, Hafsa Sheikh Mohamud Ahmed, Elitsa Popova and Omar Abdesslem. The report's contribution section details who did what.

## References and attribution

- Johann Brehmer, Pim de Haan, Sönke Behrends and Taco Cohen. 2023. Geometric Algebra Transformer. *Advances in Neural Information Processing Systems* 36.
- The PyTorch reference and the Python model this project patches into is [EzGATr](https://github.com/Guest400123064/ezgatr) (MIT License, Copyright (c) 2024 Guest400123064).

This repository is released under the MIT License (see [LICENSE](LICENSE)).
