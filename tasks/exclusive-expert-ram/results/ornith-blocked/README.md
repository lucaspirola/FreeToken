# Ornith: aborted 2026-09-22, NOT performance numbers

These two files are the **evidence that Phase 6 is blocked**, not a result.
Nothing here may be quoted as an Ornith performance figure, and no
`-record.json` was written: the arm was stopped before its second probe pass.

## What happened

`ornith-baseline-q8q8` (whole model in RAM, KV q8_0/q8_0, ratio 1.00, empty
GPU, port 1920) started correctly and served. Pass 1 measured:

| prompt | TTFT s | prefill tok/s | decode tok/s |
|---|---|---|---|
| 8 009   | 0.10   | 81 876 (prefix cache hit, not a prefill) | 151.0 |
| 32 022  | 169.32 | 189   | 140.1 |
| 80 023  | 347.60 | 230   | 129.9 |
| 200 015 | 144.00 | 1 389 | 106.7 |

Decode is plausible. **Prefill is not**: 189-230 tok/s where Nemotron does
~9 100. The owner stopped the sweep on sight of this - "pointless to
benchmark something that's visibly broken."

## Cause (not a mirror-pool defect)

FreeToken ships no tuned NVFP4 prefill table for Ornith's GEMM shapes, so
`fused_nvfp4` falls back to MiniMax-M2's default tiles and warns once
(`_warn_untuned_prefill`). The handicap is in the dense GEMM, identical in
the whole-model and pool arms, so every Ornith comparison would have been
made through a ~40x prefill penalty that hides whatever the pool costs.

Ornith geometry (verified against the checkpoint): H=2048, I=512, E=256,
top-8, silu **gated**, 40 MoE layers. Gated, so gate_up N = 2*I = 1024.

The two tables that must exist:

    nvfp4,E=256,N=1024,K=2048,device_name=NVIDIA_GeForce_RTX_5080.json   gate_up
    nvfp4,E=256,N=2048,K=512,device_name=NVIDIA_GeForce_RTX_5080.json    down

`benchmarks/tune_nvfp4_moe.py` can now produce both (commit 3edb190):

    benchmarks/tune_nvfp4_moe.py --model ornith --prefill --write

That run costs real GPU time and has **not** been authorized. Until it is,
Phase 6 stays blocked and no Ornith record exists.

Nothing reusable was found upstream: github.com/ornith-ai is 13 repos, docs
plus CUDA-L1/CUDA-L2 (dense HGEMM on A100/3090/H100; Blackwell sm_120 is on
their to-do; no FP4, no grouped GEMM, no MoE).

The `geometry.txt` line is the KV allocation the journal printed at startup
(262 144 tokens, K+V = 2.66 GiB) - kept because it is the only surviving
record that this arm reached readiness.
