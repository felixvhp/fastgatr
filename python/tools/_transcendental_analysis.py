"""Definitive check on the REAL reported model (weighted_full_model_flops)."""
from __future__ import annotations
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cpp_flop_counter import (  # noqa: E402
    Build, OpWeights, FlopCount,
    EquiGeometricAttentionCfg, EquiJoinCfg, EquiLinearCfg, EquiRMSNormCfg,
    GeometricProductCfg, ScalerGatedGeluCfg,
    equi_geometric_attention_count, equi_join_count, equi_linear_count,
    equi_rms_norm_count, geometric_product_count, scaler_gated_gelu_count,
)
W = OpWeights()
HIDDEN, HEADS, ATTN_CQK, ATTN_CV = 32, 4, 32, 32
B = 8
BUILD = Build.SIMD

def full_model(channels, tokens, layers, build):
    """Return (total FlopCount, attention-only FlopCount) for the real network."""
    T = tokens
    def lin(ci, co): return equi_linear_count(EquiLinearCfg(B, T, ci, co, True, build))
    def rms(c): return equi_rms_norm_count(EquiRMSNormCfg(B*T, c, True, False, build))
    def gp(m): return geometric_product_count(GeometricProductCfg(m, build))
    def join(m): return equi_join_count(EquiJoinCfg(m, True, 1.0, False, True, build))
    def gelu(m): return scaler_gated_gelu_count(ScalerGatedGeluCfg(m, build))
    def attn(): return equi_geometric_attention_count(EquiGeometricAttentionCfg(
        B, HEADS, T, T, ATTN_CQK, ATTN_CV, ["ipa","daa"], build=build))
    total = FlopCount()
    total += lin(channels, HIDDEN)
    a = attn()
    block = (rms(HIDDEN) + lin(HIDDEN,128) + gp(B*T*HIDDEN) + join(B*T*HIDDEN)
             + lin(64,HIDDEN) + gelu(B*T*HIDDEN) + lin(HIDDEN,HIDDEN)
             + rms(HIDDEN) + lin(HIDDEN,384) + a + lin(128,HIDDEN))
    for _ in range(layers):
        total += block
    total += lin(HIDDEN,1)
    attn_total = FlopCount()
    for _ in range(layers):
        attn_total += a
    return total, attn_total

print("="*150)
print(f"REAL reported model (HEADS=4, Cqk=Cv=32, HIDDEN=32), build={BUILD.value}, sweep N=1..64 (T=2N, channels=N, layers=N)")
print("="*150)
print(f"{'N':>4} {'flat_ops':>16} {'weighted':>16} {'wt/flat':>8} {'exp_cnt':>14} {'exp%ops':>8} "
      f"{'div%ops':>8} {'heavy%ops':>9} {'attn%flat':>9} {'exp%(of attn)':>12}")
for N in [1,2,4,8,16,32,64]:
    tot, attn = full_model(N, 2*N, N, BUILD)
    flat = tot.flops_total(); wt = tot.weighted_flops(W)
    heavy = tot.div+tot.sqrt+tot.exp+tot.tanh
    af = attn.flops_total()
    print(f"{N:>4} {flat:>16,.0f} {wt:>16,.0f} {wt/flat:>8.4f} {tot.exp:>14,.0f} "
          f"{100*tot.exp/flat:>7.3f}% {100*tot.div/flat:>7.3f}% {100*heavy/flat:>8.3f}% "
          f"{100*af/flat:>8.1f}% {100*attn.exp/af:>11.3f}%")

print()
print("Per-key runtime cartoon for attention with the REAL config (Cv=32 -> dv=512, heads=4):")
dv = ATTN_CV*16
cscore = 32  # cqk
per = (7+7)+(9+6+6+3)+(2+2)
score_ops = cscore*per; val_ops = 2*dv
cheap_cyc = (score_ops+val_ops)/4.0
for E in [8,15,30]:
    print(f"  expf={E:2d}cyc: cheap-arith~{cheap_cyc:.0f} cyc/key vs exp~{E} -> exp ~{100*E/(E+cheap_cyc):.1f}% of per-key compute")
