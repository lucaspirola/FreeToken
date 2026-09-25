# Ornith EXL3 5.0bpw decode/prefill stack: final ft-dev table (BOX NUMBERS, Vast RTX 5080, PCIe gen4), 2026-09-25

This is the baseline for the owner's-machine final run. Branch exp/ornith-exl3 at the D3c commit (52020bc).

Setup for every row: ratio 1.00, flashinfer extend, `FREETOKEN_EXL3_GEMV_PREROT=1`, `FREETOKEN_EXL3_F16ACC` off, q8_0 KV, nothing else on the GPU. The numbers are pass 2 of `fuse/job-e2e.sh`, 127 generated tokens. Each step's README holds its bracket, tests and output gate.

## (a) Saver (`--expert-residency mirror --moe-mirror-host-rows -1`), decode tok/s

| step | 8K | 32K | 80K | README |
|---|---:|---:|---:|---|
| P4/P2b (old GEMV) | 126.1 | 121.8 | 106.5 | `p2b-f16acc`, `d2-gemv` base |
| D2 GEMV occupancy | 134.7 | 128.9 | 111.0 | `d2-gemv` |
| D3a shared expert on a side stream | 146.3 | 139.0 | 117.8 | `d3-sidestream` |
| **D3c shared gate on the side stream** | **152.0** | **144.4** | **122.0** | `d3c-glue` |

Prefill (saver, D3c): 8K 7377, 32K 6715, 80K 5371 tok/s. With the P2b fp16-accumulate flag on (default off; see `p2b-f16acc` for why): 8K 8523, 32K 7963, 80K 6232.

## (b) Whole model (`--expert-residency whole`), decode tok/s

Ornith EXL3's 20.2 GB of routed experts exceed the 16 GB GPU, so this mode also crosses PCIe on misses.

| step | 8K | 32K | 80K |
|---|---:|---:|---:|
| D2 | 145.5 | 136.5 | 124.0 |
| D3a | 159.8 | 148.7 | 134.1 |
| **D3c** | **165.2** | **154.1** | **138.3** |

Prefill (whole, D3c): 7861 / 7069 / 5653 tok/s.

## (c) Profiled 8K decode (saver, D3c; `profile_step.py` wall-clock split, `d3c-glue/results/exl3prof-d3c2`)

The profiler inflates every kernel and gap, so its window (8.17 ms) is longer than E2E (6.58 ms). The shares are what matter.

| ms/token | |
|---|---:|
| EXL3 GEMV | 3.60 |
| expert miss loads (`fast_index_copy_kinds`, PCIe) | 1.55 |
| saver writebacks (DtoH) | 0.40 |
| rest (router, cuBLAS, norms, GDN/attention kernels, mirror bookkeeping) | 1.72 |
| idle | 0.91 |

**Derived no-miss decode at 8K: 199 tok/s** (E2E minus the miss loads). It is 216 without the writebacks, with a lower bound of 161 from the profiled window. The ft-dev kernel target of >= 150 is met.

## Not in this branch: the scheduler's per-step sync (exp/overlap-sync, round 3)

nsys (`d3c-glue/results/exl3nsys-d3c2`) shows the GPU idle ~0.56 ms between decode graph replays.

- Cause: `attention/triton.py` `prepare_metadata` builds `prefix_lens`/`indptr` with a blocking pageable host->device copy. The overlap loop's stream is ordered behind the forward in flight, so the host waits for step N before it prepares N+1.
- It is generic: Nemotron whole and saver show it too.
- The fix is on exp/overlap-sync (from exp/reorg-round2), measured separately, and rides round 3.

## What to run on the owner's machine

Run the same three arms:
- `fuse/job-e2e.sh`, with `RESIDENCY=whole` for (b);
- `fuse/job-prof.sh` for (c).

Keep the server down and nothing else on the GPU.

What should change there:
- The gen5 link roughly halves the miss/writeback copy time.
- The saver target there is 140 tok/s at 8K.

---

# Where EXL3 prefill time goes — BOX NUMBERS (ft-dev Vast RTX 5080), 2026-09-24

`profile_prefill.py` (torch.profiler, one 8000-token chunk after two warmups, ratio 1.00, saver on,
scratch/exl3-headroom code): wall 9.64 s (830 tok/s), GPU time 10.18 s —
`profile_prefill-8k-box-2026-09-24.txt`, trace `.trace.json.gz`.

| share | what |
|---|---|
| 89.6% (9.12 s) | `_exl3_gemm_kernel`, 800 launches |
|   of which ~5.6 s | MoE experts (8 sub-chunks of 1024 tok x 40 layers, gate_up + down, BM=16) |
|   of which ~3.5 s | dense projections (GDN in_proj 1.84 s, attn qkv+gate 0.46 s, o/out_proj 0.99 s, shared expert 0.21 s) |
| 5.8% (0.59 s) | `fast_index_copy_multi` (116x, mirror prefill assembly) |
| 1.1% | Hadamard rows |
| 0.9% / 0.8% | attention / GDN FLA kernels |

`bench_reconstruct_vs_gemm.py` (`...-box-2026-09-24.txt`), same shapes, random trellis:
exl3_gemm runs at 5-6.6 TF/s on every dense shape, while `reconstruct` (decode W_hat to fp16 once:
0.10 ms for 2048x12288) + cuBLAS fp16 is 13-18x faster (GDN in_proj M=8000: 61.1 ms -> 3.5 ms).
Decoding is cheap; the kernel re-decodes each [K,128] weight column once per M-block (125x for
M=8000 at BM=64, ~16x per expert per 8K for the MoE) and its MMA pipeline (BK=32, 2 stages, decode
on the critical path) is far from tensor-core peak. All 256 experts of a layer reconstruct in 4.5 ms.

Projection (not measured): dense 3.5 s -> ~0.2 s via reconstruct+cuBLAS above a row threshold;
MoE 5.6 s -> ~0.5 s via per-layer grouped reconstruct into an fp16 scratch + the existing grouped
fp16/bf16 MoE GEMM; the 0.59 s mirror assembly copy then becomes the next item. ~1.7 s per 8K chunk
~= 4-5K tok/s, the NVFP4 range.

## After dba11a2 + c90a28e: what grows with context, and decode (box numbers, 2026-09-24)

`profile_step.py prefill 128000` (`profile_prefill-128k-box-2026-09-24.txt`): 53.2 s wall (2408 tok/s),
16 chunks. Every chunk costs ~2.0 s of non-attention work (constant); `_extend_attention_split_kernel`
(triton, q8_0 KV, 10 full-attention layers) grows linearly from 0.09 s (chunk 0) to ~3 s (chunk 15),
25.4 s in total, ~56 TF/s effective. The fall from 4.2K tok/s at 8K to 2.4K at 128K is the attention
kernel, shared code (not EXL3), and it is not far from what a triton fp16 kernel reaches here.

`profile_step.py decode` before the GEMV change (`profile_decode-8k-before-...`, `-128-before-...`;
profiler on, so absolute rates run below the probe): 30.3 ms/token at ctx 8000, 29.7 at ctx 128
(context does not matter), GPU busy 96%. `_exl3_gemv_kernel` 18.3 ms/token (241 launches), the mirror
swap copies (`fast_index_copy_multi`, 120 launches) 7.7 ms/token, everything else ~3 ms.

### GEMV rewrite (bitstream order)
The old kernel decoded each weight with its own two 32-bit gathers, 64-bit shifts and index math, and
reduced across the K axis every step: 83-107 GB/s of weights. The new `_exl3_gemv_kernel` walks each
16x16 tile in bitstream order: code t = 32 (c & 7) + [r2 r1 (c >> 3) r3 r0], so the 32 codes of one
(c & 7) are one contiguous bit run; a lane owns one run per 16-row band, and every word index, shift
and row of its 32 codes is a compile-time constant (the few words load once, identical addresses CSE),
decoded with a funnel shift + dp4a in 32-bit ops, accumulating columns c and c + 8. The Hadamard
epilogue reads H's rows in that column order. `bench_gemv.py` (`bench_gemv-box-2026-09-24.txt`),
kernel time at the launcher's split:

| shape | old | new | exllamav3 (`bench_exllamav3_gemv...`, incl. input rotation) |
|---|---|---|---|
| GDN in_proj 2048x12288 | 189 us | 46 us | 31 us |
| attn q+gate/k/v 2048x9216 | 110 us | 28 us | 24 us |
| o_proj/out_proj 4096x2048 | 64 us | 20 us | 19 us |
| MoE gate/up x8 routes | 110 us | 32 us | — |
| MoE down x8 routes | 62 us | 20 us | — |

`pick_split_k` now targets <= 8 single-warp programs per SM (the sweep's best on every shape).
In decode (`profile_decode-8k-gemv-new-...`): GEMV 18.3 -> 3.9 ms/token, decode window 30.3 -> 16.0
ms/token (33 -> 62 tok/s with the profiler on).

### Next decode bottleneck: mirror writebacks are SM stores to pinned host memory at 2.3 GB/s
The swap copies are now 52% of decode (7.7 ms/token for 17.8 swaps + 5.4 writebacks per token).
`bench_mirror_copy.py` (`bench_mirror_copy-box-2026-09-24.txt`): H2D (SM loads from pinned memory)
~25.9 GB/s, matching DMA (23.5 GB/s); D2H (SM stores into pinned memory) 2.3 GB/s, 10x below DMA
D2H (24.4 GB/s): ~0.84 ms per 1.98 MB expert written back. On this box (docker, EPYC 7402P, PCIe gen4)
the writebacks alone cost ~4.5 ms/token. Shared residency code (`residency.copy_missing_mirror`,
`fast_index_copy_multi`), not EXL3; whether the owner's WSL2 5080 has the same D2H penalty is not
measured (run `bench_mirror_copy.py` there with the server down).
`bench_sm_d2h.py` (`bench_sm_d2h-box-2026-09-24.txt`): no store pattern gets around the cap. SM stores
into pinned memory run at 2.1-2.2 GB/s for 16 KiB to 256 KiB per program, with default, `.wt` or
`.cs` stores and 4 or 8 warps, and at 3.4-3.8 GB/s with just 2 programs. DMA of the same 2 MB runs at
22.9 GB/s. On this host only a copy engine gets full-speed writebacks. That means a DMA path
(cudaMemcpyAsync / batched memcpy) with the device-side writeback list, which is a design change in the
shared mirror, not a kernel tweak. After the GEMV rewrite it is the largest decode cost left on this box
(pass 1 decode 40-59 tok/s vs 116-119 warm).
