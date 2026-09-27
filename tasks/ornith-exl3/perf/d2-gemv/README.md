# D2: decode GEMV occupancy (Ornith EXL3 5.0bpw, box ft-dev, RTX 5080 gen4)

These are box numbers.

## Cause

`gemv_regs.py` (ptxas -v on sm_120, CPU only, `results/gemv_regs-box.txt`) shows the problem in the old kernel:
- The single-warp GEMV program used **168 registers and spilled 1.4 KB per thread**.
- The culprit was the output-rotation epilogue. It multiplied the accumulated 128 columns by the 128 x 128 Hadamard held in registers: `[64,128]` H rows for the per-split path, or a `[128,128]` product in the in-kernel split-K reduce. That is 256-512 fp32 values per thread in a 32-thread program.
- At 168 registers an SM holds 12 of these warps, so the loop ran latency-bound. The split policy's "8 programs per SM" optimum was a symptom of this.

Nsight Compute cannot confirm it on this box: the counters are blocked (ERR_NVGPUCTRPERM). The register report and the timings below are the evidence.

## Change (`kernel/triton/exl3.py`, `layers/quantization/linear/exl3.py`)

- **Epilogue:** `_had128_8x16` computes H128 = H8 (x) H16 on the column vector viewed [8, 16]. That is 8x8x16 + 8x16x16 products, with the entries generated rather than loaded, which is the same factorisation the prologue rotation already used.
- **Split-K reduce:** each split now stores its unrotated partial. The last split sums the planes in fixed order, so the result is still deterministic, and rotates once.
- **Register cap:** `GEMV_MAXNREG = 64`. Uncapped, ptxas takes 80; 64 is spill-free and fills the 32 resident blocks an SM allows.
- **Bands:** `BANDS` (16-row bands per loop step) is kept as a knob at 1. At most it moves things by 0.5 us.
- **Split policy (`pick_split_k`):** 20 single-warp programs per SM, at most 32 splits. Past 32 splits the serial plane sum in the last split costs more than it gains: o_proj takes 13.3 us at s32 and 14.4 us at s64.

## Per-kernel budget (`bench_gemv_d2.py`, graph-replayed device time, weights rotated over > L2)

Efficiency is weight bytes / time / 818 GB/s. "Old" is the pre-D2 kernel with its launcher's split; "D2" is the shipped kernel with the shipped split.

| shape | MB | old us (eff) | D2 us (eff) | D2 split |
|---|---:|---:|---:|---:|
| GDN in_proj 2048x12288 | 15.73 | 34.3 (0.56) | **26.4 (0.73)** | 16 |
| attn q+gate/k/v 2048x9216 | 11.80 | 23.5 (0.61) | **21.5 (0.67)** | 16 |
| o_proj/out_proj 4096x2048 | 5.24 | 14.5 (0.44) | **13.3 (0.48)** | 32 |
| shared gate/up 2048x1024 (5-bit model; 7 bits in the checkpoint) | 1.31 | 9.7 | **8.0** | 32 |
| shared down 512x2048 | 0.66 | 6.7 | **5.8** | 16 |
| MoE gate/up x8 routes | 10.49 | 33.4 (0.38) | **26.3 (0.49)** | 16 |
| MoE down x8 routes | 5.24 | 18.4 (0.35) | **14.1 (0.46)** | 8 |

- The target of >= 0.8 of the DRAM bound on the big shapes is **not met**. The best shape is in_proj at 0.73-0.76.
- The small MoE and shared launches are bound by launch and ramp time. A 5 MB kernel at 0.8 would have to finish in 8 us.
- What remains for them is work per launch: fusing the shared expert into, or alongside, the routed launch (D3), not per-kernel tuning.

Per decode token (30 GDN + 10 attention + 40 MoE layers), the GEMVs go from ~4.6 to ~3.7 ms.

## End to end (whole + saver, ratio 1.00, flashinfer extend, PREROT=1, pass 2; bracketed base / D2 / base)

The base arm is the P2b tree (flag off), i.e. the old GEMV.

| prompt | base 1 | **D2** | base 2 | D2 vs base |
|---:|---:|---:|---:|---:|
| 8K decode tok/s | 126.1 | **134.7** | 126.1 | +6.8% |
| 32K | 121.8 | **128.9** | 121.7 | +5.9% |
| 80K | 106.5 | **111.0** | 106.5 | +4.2% |

- Prefill is unchanged: 7424 / 7364 / 7373 tok/s at 8K.
- All arms had captures=1 and no tracebacks.
- An unbracketed D2 run with 128K (`results/exl3e2e-d2d`) gave 136.3 / 130.4 / 112.8 / 111.9.
- The decode target of 140 tok/s at 8K is not reached yet: 134.7.

## Numerics (`results/exl3tests-d2d`, `results/exl3logits-d2d`)

- Tests: 318 passed (test_exl3).
- Logits against exllamav3 (ratio 0.90) are within the earlier kernels' spread:
  - short pre1: KL mean 2.02e-2, top-1 0.968.
  - long pre1: KL mean 1.37e-2, top-1 1.000.
  - long pre0: KL mean 3.22e-2.
  - saver vs whole: bit-equal.
- The epilogue changes the fp32 summation order of the output rotation. Nothing else changes.
