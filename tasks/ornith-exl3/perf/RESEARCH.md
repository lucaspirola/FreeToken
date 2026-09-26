# EXL3 kernels: prior art, what applies on sm_120 (RTX 5080), and the roofline

Scope: a bounded research pass on 2026-09-24 before optimising the EXL3 kernels. Each source is
listed with what it says and whether it applies on GeForce Blackwell. The measured roofline
(box numbers) and the ranked list of what to try come after.

## Sources

| source | what it says | applies on sm_120? |
|---|---|---|
| QTIP, Tseng et al., NeurIPS 2024 ([arXiv 2406.11235](https://arxiv.org/abs/2406.11235)) | Bitshift trellis plus compute-based Gaussian codes (1MAD ~4 instr/weight, 3INST 3, HYB ~2 amortised). No trellis or codebook is stored, so decode runs in registers inside the matvec. The 4090 matvec reaches **840 GB/s effective** (2-bit 70B, 25.8 tok/s), about 83% of 1008 GB/s. | Yes. It sets the decode GEMV target at ~80-85% of DRAM. EXL3 `mul1` is QTIP's 1MAD code. |
| exllamav3 v1.5.1 source (box `/root/exllamav3-src`, `exllamav3_ext/quant/`) | **Decode (bsz 1):** `exl3_gemv_int8_sq_kernel`, a single cooperative launch. It does the input Hadamard (phase 1a), then a GEMV that quantises the activations to int8 and gets `codebook(x)*a` from one `dp4a`, because mul1 is affine in the byte sum. The epilogue does the output Hadamard and svh. The default mode (2) is plain int8 at "~0.9% output RMS deviation"; mode 1 adds a residual pass for ~15-16-bit activation precision. **Up to 144 rows:** `exl3_gemm_kernel`, a cooperative kernel that runs the input Hadamard, `grid.sync`, then a Marlin-style mma.sync GEMM with the trellis decoded in the mainloop, 16 rows at a time. **Over 144 rows (prefill):** reconstruct plus a GEMM. From 1024 rows on, `reconstruct_had_kernel` folds **both Hadamards and suh/svh into the reconstructed weight**, so the GEMM runs on raw x. The GEMM is exllamav3's own **`f16acc::gemm_kernel` (fp16 accumulation)**, not cuBLAS. **MoE:** fused `exl3_moe` kernel for small per-expert row counts, and a batched-reconstruct tier above that. | Yes, all of it is mma.sync or dp4a, so it runs on sm_120 as is. Measured on the box below. |
| GeForce Blackwell ISA: [SageAttention #291](https://github.com/thu-ml/SageAttention/issues/291), [Blackwell GPU wiki](https://0xsero.github.io/blackwell-gpu-wiki/blackwell/tcgen05-and-tmem/), [backend.ai on sm_12x](https://www.backend.ai/blog/2026-02-is-dgx-spark-actually-a-blackwell), [microbenchmark paper arXiv 2512.02189](https://arxiv.org/html/2512.02189v3) | sm_120 has **no TMEM/tcgen05**. `wgmma` fails in ptxas on sm_120 (#291); the wiki claims otherwise but gives no evidence. No clusters > 1, and ~99 KB of shared memory per block. Tensor cores are reached through **extended mma.sync** (fp16/bf16, fp8/fp6/fp4 block-scaled). | Hopper and datacenter-Blackwell designs are out: Machete, CUTLASS SM90/SM100 mixed-input and tcgen05 kernels. The template to follow is Ampere/Marlin-style mma.sync, which is also what Triton `tl.dot` lowers to here. |
| Marlin, Frantar et al. ([arXiv 2408.11743](https://arxiv.org/abs/2408.11743)) | 4-bit mixed GEMM near the ideal 3.87x up to batch 16-32. It uses async global loads, a circular shared-memory queue, dequant overlapped with MMA, and striped partitioning with a lock-based reduction. | Yes, it is the design for a fused trellis-decode-in-mainloop GEMM (the "only if" prototype in P3). EXL3 decode costs ~2-4 ALU ops per weight, not Marlin's ~1, so the mainloop is heavier. |
| Machete ([Red Hat](https://developers.redhat.com/articles/2024/10/14/introducing-machete-mixed-input-gemm-kernel), [vLLM #7174](https://github.com/vllm-project/vllm/pull/7174)) | A Hopper mixed-input GEMM built on wgmma and TMA. It overlaps upconversion with MMA and uses an offline weight re-layout. | Only the idea of an offline re-layout for conversion-friendly registers applies; EXL3 tiles are already bitstream-ordered. |
| FLUTE ([arXiv 2407.10960](https://arxiv.org/abs/2407.10960)) | A LUT-quantised GEMM. It restructures weights offline to cut bit manipulation, duplicates the LUT in shared memory, and is 2-4x at batch < 32. | Partially. EXL3 mul1 is compute-based and has no LUT; the offline restructuring idea is already done in EXL3. |
| Triton W4A16 split-K ([arXiv 2402.00025](https://arxiv.org/abs/2402.00025), [PyTorch blog](https://pytorch.org/blog/accelerating-triton/)) | Fuses dequant, tl.dot and atomic split-K for skinny M: +65% on A100, +124% on H100. | Yes for small-M Triton GEMMs. Our GEMV already reduces split-K deterministically in-kernel; atomics would break saver==whole bit-equality. |
| MoE: [vLLM fused MoE](https://docs.vllm.ai/en/latest/design/moe_kernel_features/), [MonoMoE arXiv 2609.04244](https://arxiv.org/abs/2609.04244), [PyTorch MoE locality](https://pytorch.org/blog/accelerating-moe-model/) | Tokens are sorted into BLOCK_M blocks per expert: one grouped launch per projection, with tuned tile configs. MonoMoE is a persistent weight-major megakernel for decode that runs routing, both projections, activation and reduction in one launch: 1.54x over vLLM's Triton grouped GEMM on H200 (FP8). Grouped launch order helps L2 reuse. | Yes. Prefill already uses the sorted-block scheme (`moe_align_block_size`); what is missing is tile tuning and enough rows per expert. Decode already covers all 8 routes in one GEMV launch per projection; a megakernel would only save launches (4 per layer: gu, silu_had, dn, combine). |
| vllm-exl3 ports ([lna-lab/vllm-exl3](https://github.com/lna-lab/vllm-exl3)), [exllamav3 PR #330](https://github.com/turboderp-org/exllamav3/pull/330) | Community ports keep fp16 reconstruction for prefill. #330 adds bf16 I/O to the decode GEMM/MGEMM and removes the dtype conversions (~18-27 us per projection on a 5090, +5.6-11.6% vLLM throughput); it was closed and kept downstream. | Yes for the glue: our bf16->fp16 copies around EXL3 calls are the same waste. |

## Measured roofline (box numbers, ft-dev RTX 5080, 2026-09-24)

`bench_roofline.py` (ours, the production launchers) and `bench_roofline_exllamav3.py` (exllamav3 on the
same shapes) give the raw per-kernel data in `roofline/`. The bounds are measured on the box, not taken
from the spec sheet:
- DRAM copy: **818 GB/s** (read+write, 1 GiB).
- fp16 GEMM with fp32 accumulate: **121.5 TF/s** (bf16 121.3), 8192^3 through cuBLAS.
- exllamav3's fp16-accumulate GEMM runs at up to **170 TF/s** on the same card.

"eff" is the bound time (compulsory bytes at DRAM speed, or FLOPs at the fp32-accumulate peak) divided
by the measured time.

Decode, per call (bsz 1):

| kernel (shape) | ours us | GB/s | eff | exllamav3 us (int8 GEMV, fused had) | per token |
|---|---|---|---|---|---|
| GEMV GDN in_proj 2048x12288 | 30.1 (+had_rows 2.7) | 524 | 0.64 | 22.8 (688 GB/s) | x30 layers |
| GEMV attn q+gate/k/v 2048x9216 | 19.9 (+2.6) | 597 | 0.73 | 17.0 (695) | x10 |
| GEMV o_proj/out_proj 4096x2048 (prologue rotation) | 14.3 | 368 | 0.45 | 11.0 (477) | x40 |
| GEMV shared gate/up 2048x1024 | 9.8 | 135 | 0.17 | 6.3 (209) | x40 |
| GEMV shared down 512x2048 | 6.8 | 98 | 0.12 | 5.6 (116) | x40 |
| GEMV lm_head 2048x248320, 6 bit | 442.8 (+3.1) | 864 | 1.06 | 429.4 (888) | x1 |
| MoE 8 routes: 2 GEMVs | 34.5 | 463 | 0.57 | (not run) | x40 |
| MoE silu+had / combine | 3.2 / 0.8 | | | | x40 |

Prefill, M = 8192 rows, per layer:

| op | ours: kernels (us) | total us | exllamav3 total us |
|---|---|---|---|
| GDN in_proj 2048x12288 | had_rows 482, reconstruct 60, cuBLAS 3446 (119.7 TF/s, eff 0.98), had_cols 1488 | 5476 | **2456** (reconstruct_had 75 + f16acc GEMM 2382 at 173 TF/s) |
| attn q+gate/k/v 2048x9216 | 367 + 47 + 2616 + 1099 | 4129 | **1827** |
| o_proj 4096x2048 | 259 + 20 + 1175 + 214 | 1668 | **806** |
| shared gate/up 2048x1024 | 251 + 8 + 316 + 103 | 678 | **213** |
| shared down 512x2048 | 35 + 4 + 157 + 237 | 433 | **129** |
| MoE, 8 routes over 256 experts | had_rows 2258, reconstruct_experts 10869 (64 calls, 8.46 GB at 778 GB/s, eff 0.95), exl3_gemm 16281 (25.3 TF/s, 64 calls), torch glue 1840, align 53 | **31.3 ms** | (not run) |

What the table says:
1. **The MoE prefill costs 31 ms/layer, ~1.25 s per 8K chunk over 40 layers, and dominates.**
   - Its grouped GEMM runs at 25 TF/s, 21% of the fp32-accumulate peak.
   - `reconstruct_experts` is not ALU-bound: it moves 8.46 GB at 95% of the copy bandwidth. It rewrites
     every expert **four times per 8K chunk**, because the MoE sub-chunks tokens by
     `PREFILL_CHUNK_TOKENS = 2048`.
   - The same sub-chunking caps the GEMM at ~64 rows per expert per launch. A 128-column decoded tile
     then serves 64 rows: 64 FLOP per byte, which at 818 GB/s is at most ~52 TF/s. That makes the GEMM
     bandwidth-starved whatever its tiles are.
2. **The dense prefill spends 36-60% of its time in rotations.** `had_rows` and `had_cols` together cost
   more than half the cuBLAS time on every shape. exllamav3 folds both rotations into the
   reconstruct (75 us) and uses an fp16-accumulate GEMM at ~170 TF/s. It does the same projection in
   45% of our time.
3. **Decode GEMVs run at 0.45-0.73 of DRAM on the big shapes and 0.12-0.17 on the small ones.**
   exllamav3's int8 GEMV is 1.2-1.6x faster. The lm_head is already at the bound.

## Ranked plan

The orchestrator fixed the order (P1-P4, D1-D3). The evidence above sets what each step contains:
- **P1, MoE grouped GEMM.** A tuned Triton grouped GEMM over the decoded slab, with a config sweep for
  sm_120. It also processes enough rows per expert per launch: a larger MoE token sub-chunk, bounded by
  scratch memory, so that each decoded weight tile serves ~256 rows instead of ~64. That same change
  cuts `reconstruct_experts` 4x (8K chunk: 4 -> 1 reconstruct per expert).
- **P2, rotations and glue.** Dense: a folded reconstruct (both Hadamards, suh and svh in the weight;
  exllamav3's `reconstruct_had`), so the GEMM reads raw x and writes the final output. MoE: fuse the
  gather+rotation and the combine, and remove the torch glue.
- **P3, reconstruct.** It is already at DRAM speed, so the gain is in writing it fewer times (P1) and
  in writing fp16 only once per chunk. The Marlin-style fused prototype comes only if 8K prefill is
  still under 6K tok/s.
- **Candidate outside the list: fp16-accumulate GEMMs,** as exllamav3 uses (~1.4x on every dense
  prefill GEMM). This changes numerics, so it needs a logits check against exllamav3. Noted, not
  scheduled.
- **D2, GEMV parity.** The reference is exllamav3's int8 GEMV (mode 2, ~0.9% RMS deviation). Parity
  with its fp16 GEMV is the fair target for an exact kernel; that reference still has to be measured
  (`EXL3_INT8_GEMV=0`).
