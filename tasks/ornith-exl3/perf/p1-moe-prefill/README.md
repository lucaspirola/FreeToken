# P1: MoE prefill GEMM (Ornith EXL3 5.0bpw, RTX 5080 box, "box numbers" ft-dev gen4)

Change (python/freetoken/moe/fused_exl3.py, kernel/triton/exl3.py):

* `PREFILL_CHUNK_TOKENS` 2048 -> 8192: one MoE sub-chunk per 8K prefill chunk, ~256 routed rows per
  expert instead of ~64, so each decoded W_hat tile feeds 4x the rows and the reconstruct runs once
  per chunk instead of 4x.
* decoded-slab GEMM tiles from a box sweep (72 configs at 8192 tokens, `bench_moe_prefill.py`):
  block_m 32 (16 below 16 rows/expert), block_k 64, 3 stages, 8 warps (was 64/32/2/4).
* top-k combine through `splitk_combine` (one kernel, fixed k order) instead of the fp32
  cast/mul/sum torch glue.

Outputs are bit-identical to the old tiles (rel err 0.00 vs the chunk-2048 baseline in the sweep).

## Per-kernel budget, one MoE layer at 8192 tokens (256 experts, top-8, uniform routing)

`bench_moe_prefill.py 8192 --quick`, torch.profiler device time, box.
Bounds: fp16 mma.sync GEMM 121.5 TF/s, DRAM 818 GB/s (tasks/ornith-exl3/perf/roofline).

| kernel | before (chunk 2048) | after (P1) | work | bound | after / bound |
|---|---:|---:|---|---:|---:|
| exl3_gemm (gate_up + down, decoded slab) | 16.28 ms | 6.42 ms | 412 GFLOP | 3.39 ms | 1.89 |
| reconstruct_experts (W_hat -> fp16) | 10.88 ms | 2.65 ms | 2.11 GB | 2.58 ms | 1.03 |
| had_rows (gate_up + down inputs) | 2.26 ms | 2.28 ms | 0.70 GB | 0.86 ms | 2.65 |
| top-k combine | 0.63 ms (splitk_combine) | 0.65 ms | | | |
| act_and_mul | 0.15 ms | 0.19 ms | | | |
| align / sort | 0.05 ms | 0.05 ms | | | |
| **layer** | **30.08 ms** (31.21 with the old torch combine) | **12.20 ms** | | | |

Transient memory of the layer: 800 -> 1152 MiB; the server's measured prefill headroom went 1.00 ->
1.12 GiB, which the expert arena pays with 64 slots (5368 -> 5304 of 6076).

Target was 527 -> <= 200 ms of MoE GEMM per 8K chunk (40 layers): 16.28 -> 6.42 ms per layer is
651 -> 257 ms. Missed by 57 ms; what is left is GEMM efficiency (64 TF/s, 0.53 of the mma.sync
bound). The fp16-accumulate variant is P2b.

## End to end (whole model, RAM saver, ratio 1.00, pass 2 of record, `fuse/job-e2e.sh`, PREROT=1)

| prompt | prefill tok/s before (pre1) | after (P1) | decode tok/s before | after |
|---:|---:|---:|---:|---:|
| 8K | 4237 | 6706 | 128.5 | 128.2 |
| 32K | 3762 | 5653 | 124.8 | 123.6 |
| 80K | 2958 | 4006 | 107.5 | 106.5 |
| 128K | 2385 | 2992 | 105.1 | 103.6 |

captures=1, no tracebacks. Decode moves within the run-to-run spread of these probes.
Results: results/exl3e2e-{pre1,p1} (from /root/K/results on ft-dev, md5-verified).

## Logits

Teacher-forced 32-token windows against exllamav3's reference logits (`fuse/job-logits.sh`, ratio
0.90), results/exl3logits-p1; the pre-P1 spread of the same check is results/exl3logits-v1long
(long) and the v1 short run.

| set | metric | pre-P1 | P1 |
|---|---|---:|---:|
| short | FT (pre1) vs exllamav3 KL mean / top-1 | 1.94e-2 / 0.935 | 2.47e-2 / 0.968 |
| short | FT (pre0) vs exllamav3 KL mean / top-1 | 2.18e-2 / 0.903 | 2.42e-2 / 0.935 |
| long (40x filler) | FT (pre1) vs exllamav3 KL mean / top-1 | 4.34e-2 / 0.935 | 3.57e-2 / 0.935 |
| long | FT (pre0) vs exllamav3 KL mean / top-1 | 1.00e-2 / 0.968 | 1.03e-2 / 1.000 |
| both | saver vs whole | bit-equal | bit-equal |

Within the spread the pre-P1 kernels already show against exllamav3 (and against each other: pre1 vs
pre0 KL mean 3.2e-2 long). The prefill path itself is bit-identical to the chunk-2048 tiles in the
layer bench; the E2E logits move only through the chunking of the combine.
