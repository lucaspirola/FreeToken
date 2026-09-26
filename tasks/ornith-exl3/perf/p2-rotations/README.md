# P2: rotations and glue (Ornith EXL3 5.0bpw, RTX 5080 box, "box numbers" ft-dev gen4)

## Changes

* **Dense prefill, >= 1024 rows** (`FREETOKEN_EXL3_FOLD`, default on):
  * `reconstruct_folded` decodes `W_full = diag(suh) H W_hat H diag(svh) / 128` in 2048-column slabs. This is exllamav3's reconstruct_had.
  * The raw activations then go through one cuBLAS fp16 GEMM, so had_rows + GEMM + had_cols become reconstruct + GEMM.
  * The first version built the whole W_full and a full fp16 product at once. That raised the measured prefill transient to 1206 MiB, and the server refused to start: the mirror pool was sized against the 1 GiB estimate. Slabbing keeps the transient at 1.12 GiB.
* **MoE prefill input fold** (`FREETOKEN_EXL3_MOE_FOLD`, default off): built and measured, and it is a loss.
  * Folding `diag(suh) H` into 256 experts' decoded weights costs ~206 GFLOP per 8K chunk.
  * Rotating the routed activations costs ~43 GFLOP.
  * Measured: reconstruct_folded takes 6.15 ms, against reconstruct at 2.65 ms plus had_rows at 1.68 ms.
* **had_rows**: each program handles up to 16 128-column blocks and loads H once. Outputs are bit-identical.
  * The kernel is mma-bound, not DRAM-bound. The exact hi/lo split runs two dots per element, 137 GFLOP for the MoE gate_up input, which is 1.13 ms at 121.5 TF/s.
  * MoE layer: 2.28 -> 1.68 ms.
  * The multi-block path is used only from 4096 rows on. Decode keeps one program per block.

## Per-kernel budget, dense projections at 8192 rows

`bench_roofline.py`, per call, box. Before = roofline r0 (../roofline); after = results/exl3roof-p2.

| projection | before: had_rows + reconstruct + GEMM + had_cols | after: cast + reconstruct_folded + GEMM + slab copy | exllamav3 |
|---|---:|---:|---:|
| GDN in_proj 2048x12288 | 482 + 60 + 3446 + 1488 = **5476 us** | 86 + 384 + 3419 + 487 = **4376 us** | 2456 |
| attn q+gate/k/v 2048x9216 | 367 + 47 + 2616 + 1099 = **4129** | 86 + 291 + 2581 + 357 = **3315** | 1827 |
| o_proj 4096x2048 | 259 + 20 + 1175 + 214 = **1668** | 169 + 113 + 1165 + 79 = **1526** | 806 |
| shared gate/up 2048x1024 | 251 + 8 + 316 + 103 = **678** | 86 + 33 + 294 + 27 = **440** | 213 |
| shared down 512x2048 | 35 + 4 + 157 + 237 = **433** | 23 + 17 + 157 + 54 = **251** | 129 |

The GEMM runs at 0.90-0.99 of the fp32-accumulate mma bound (121.5 TF/s). Two things remain:

* The host glue: the bf16->fp16 cast of x, and the fp16->bf16 copy of each slab product.
* The accumulate precision: exllamav3 accumulates in fp16 at ~170 TF/s.

Both are P2b. The bf16-reading, bf16-writing fp16-accumulate GEMM is already on the branch behind `FREETOKEN_EXL3_F16ACC`, and the dense GEMM is measured at 196-217 TF/s.

## Rotations + glue per 8K chunk (40 layers: 30 GDN + 10 attention)

| | before (r0) | after (P2) |
|---|---:|---:|
| dense rotations/glue per layer (GDN / attn) | 3069 / 2565 us | 1011 / 881 us |
| MoE had_rows + combine/act glue per layer | 4106 us | 2328 us |
| **per 8K chunk** | **282 ms** | **132 ms** |

The target was <= 100 ms, so it is missed by 32 ms. The rest is the dense cast/copy glue (~38 ms) and the fp32 MoE down output read by the combine (0.65 ms per layer; fp16 halves it). Both belong to P2b's numerics A/B. The folded reconstruct costs +~18 ms per chunk over the plain decode; the 8K end-to-end gain below already accounts for it.

## End to end (whole model, RAM saver, ratio 1.00, pass 2 of record, `fuse/job-e2e.sh`, PREROT=1)

| prompt | P1 prefill tok/s | + dense fold (p2) | + had_rows re-tile (p2d) | decode P1 / p2d |
|---:|---:|---:|---:|---:|
| 8K | 6706 | 7052 | 7203 | 128.2 / 127.8 |
| 32K | 5653 | 5924 | 6065 | 123.6 / 124.9 |
| 80K | 4006 | 4119 | 4180 | 106.5 / 110.0 |
| 128K | 2992 | 3068 | 3098 | 103.6 / 101.3 |

captures=1, no tracebacks, prefill transient 1.12 GiB (unchanged from P1), arena 5304 of 6076 slots.
Results: results/exl3e2e-{p2,p2d}.

## Logits (`fuse/job-logits.sh`, ratio 0.90, teacher-forced against exllamav3), results/exl3logits-p2

| set | metric | P1 | P2 (dense fold) |
|---|---|---:|---:|
| short | FT pre1 vs exllamav3 KL mean / top-1 | 2.47e-2 / 0.968 | 2.47e-2 / 0.968 |
| short | FT pre0 vs exllamav3 KL mean / top-1 | 2.42e-2 / 0.935 | 2.42e-2 / 0.935 |
| long | FT pre1 vs exllamav3 KL mean / top-1 | 3.57e-2 / 0.935 | 1.04e-2 / 0.968 |
| long | FT pre0 vs exllamav3 KL mean / top-1 | 1.03e-2 / 1.000 | 1.64e-2 / 0.935 |
| both | saver vs whole | bit-equal | bit-equal |

Within the spread of the earlier kernels. The short prompt is under 1024 rows, so it does not take the fold and its numbers are identical to P1. The had_rows re-tile is bit-identical: `bench_had_rows.py` asserts `torch.equal` across all tiles.
