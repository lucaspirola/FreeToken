# Ornith RTX 2000 Ada counter-guided cache tuning

Date: 2026-08-31  
Host: NVIDIA RTX 2000 Ada Generation (sm_89), 16 GiB, 70 W, WSL 2  
Stack: Windows driver 595.95, CUDA toolkit 13.1, PyTorch 2.11.0+cu130,
Triton 3.6.0

## Accepted: borrow mixed-GGUF prefill buffers for decode

Mixed-size GGUF caches previously allocated Q4's two 256-expert prefill buffers
separately from the compact decode classes. The uniform Q6 cache already borrowed
the same two buffers as ordinary LFU rows between prefills. Q4 now does the same:
the largest size class owns the 512 legacy-width rows, decode can populate them,
and prefill invalidates their exact ownership records before overwriting them.
The requested cache bytes and VRAM budget are unchanged.

Paired Q4_K_M / INT4, 32K virtual context, greedy AIME decode256, LFU, 0.97
memory ratio, second (warm) request:

| Metric | Separate buffers | Borrowed buffers | Change |
|---|---:|---:|---:|
| Usable decode slots | 5,998 | 6,510 | +8.5% |
| Aggregate warm miss rate | 6.4518% | 5.9699% | -7.5% relative |
| GPU step | 15.987 ms | 15.726 ms | -1.6% |
| Wall span | 16.626 ms/token | 16.369 ms/token | -1.5% |
| Decode throughput | 60.12 tok/s | 61.07 tok/s | +1.6% |
| Greedy output SHA1 | `71b637b727f6` | `71b637b727f6` | identical |

The production-shaped concurrency gate used two serialized 2,048-token prompts,
128/64 generated tokens, a 262K virtual INT4 KV ceiling, and 96K physical growth.
Both agents recovered only their own passcode (`passed: true`), with no foreign
passcode or prompt leakage. The main agent continued at 54.28 tok/s after helper
teardown.

## Added: in-engine GPU-step telemetry

`--moe-collect-stats` now records timing-enabled CUDA events around each real
forward, without synchronizing the hot path. Events are consumed at the scheduler's
existing completion barrier and summarized at idle alongside the per-layer LFU miss
statistics. `bench_decode_moe.py` can pass the flag through.

Warm single-session profiles show that Python/scheduler work is not the primary
decode limit:

| Model / KV | GPU step | Wall span | Stream coverage | CPU enqueue | Decode |
|---|---:|---:|---:|---:|---:|
| Q4_K_M / INT4 (before borrowed buffers) | 15.987 ms | 16.626 ms | 96.2% | 0.901 ms | 60.12 tok/s |
| Q4_K_M / INT4 (borrowed buffers) | 15.726 ms | 16.369 ms | 96.1% | 0.843 ms | 61.07 tok/s |
| Q6_K / Q8_0 | 20.414 ms | 21.195 ms | 96.3% | 1.012 ms | 47.19 tok/s |

For these single-session runs, eliminating every gap outside the measured GPU step
would cap scheduler-only gains at roughly 4%. Q6's 10.45% warm expert miss rate is
the larger remaining cost.

## Accepted: two-warp Q6 gate/up MMVQ on Ada

Nsight Compute attached to the spawned scheduler after GPU performance-counter
access was enabled. Ornith's Q6 routed+shared gate/up MMVQ measured 88.22 us per
layer, 80.86% DRAM throughput, 58.83% compute throughput, and 47.03% achieved
occupancy with the old one-warp block. Four gate/up warps had already lost on
Ada; the previously untested two-warp midpoint was retained.

Paired 32K Q6_K / Q8_0 runs (two runs per launch) gave:

| Gate/up warps | Mean decode | Mean GPU step | Greedy SHA1 |
|---:|---:|---:|---|
| 1 | 47.65 tok/s | 20.342 ms | `2c795ba5a29e` |
| **2 (retained on sm_89)** | **47.87 tok/s** | **20.246 ms** | `2c795ba5a29e` |

That is a repeatable but deliberately modest +0.47% decode / -0.47% GPU-step
change. Q4_K_M / INT4 was neutral-to-positive (61.30 to 61.33 tok/s; 15.724 to
15.696 ms GPU step) and retained `71b637b727f6`. Ada therefore uses two gate/up
warps and the previously selected four down warps for both formats. The process
start overrides remain available for future hardware A/B checks.

The profiler's analogous suggestion for dense Q6 MMVQ was rejected at model
level: grouping 1/2/4 output rows produced 47.55/47.53/47.31 tok/s. The temporary
dense dispatch was removed.

## Rejected: format-specific shared-MMVQ warp dispatch

Isolated kernels favored Q4 gate/up at four warps and down at one warp, but two
model-level candidates averaged 59.36 tok/s versus the fresh 59.17 tok/s control
(+0.31%, within host variance). Both retained `71b637b727f6`; the dispatch change
was removed.

## Rejected: Ada extend-attention launch changes

All cases used Ornith's exact 16Q/2KV/D256 geometry, a 4K new chunk, INT4 and Q8_0
KV, and the dequantized-output oracle. The retained 8-warps/2-stage launch won every
32K and 131K cached-prefix case.

| Warps / stages | INT4 32K | INT4 131K | Q8_0 32K | Q8_0 131K |
|---|---:|---:|---:|---:|
| **8 / 2 (retained)** | **231.406 ms** | **954.438 ms** | **216.960 ms** | **841.299 ms** |
| 4 / 2 | 242.569 ms | 1038.412 ms | 263.115 ms | 1025.182 ms |
| 8 / 1 | 367.016 ms | 1376.316 ms | 352.285 ms | 1317.452 ms |
| 4 / 1 | 253.101 ms | 1024.165 ms | 280.974 ms | 1061.059 ms |

## Rejected: changing the LFU decay interval

The decode benchmark now supports `--warm-problem` so a candidate cache policy
can be measured after a different routing trace, rather than only after replaying
the exact measured prompt. It also waits for one scheduler idle boundary before
teardown so the GPU-event and MoE miss summaries correspond to the reported
second request.

Q6_K / Q8_0, greedy decode 256, 32K pool, output hash `2c795ba5a29e` in every run:

| LFU decay | repeated prompt tok/s | repeated GPU ms | switched prompt tok/s | switched GPU ms |
|---:|---:|---:|---:|---:|
| 128 | 46.47 | 20.691 | 42.29 | 22.896 |
| **256 (retained)** | **47.35** | **20.387** | **41.49** | **23.452** |
| 512 | 47.80 | 20.117 | 39.45 | 24.592 |

The 512-step candidate helps an identical repeated route by about 1%, but loses
4.9% after a prompt switch. The 128-step candidate gains 1.9% on the switch but
loses 1.9% on the repeated route. Q4_K_M / INT4 was effectively neutral between
256 and 512 (61.30 versus 61.26 tok/s, same `71b637b727f6` hash). The existing
256-step aging therefore remains the balanced production policy; no serving-path
change from this sweep was retained.

## Rejected: narrowing the LFU remap launch

Nsight Compute measured the cold-miss `_ensure_experts_sized_kernel` at 24.83 us
for Q6's 4,720-slot cache. The single-CTA reduction used 256 threads, 171
registers/thread, 16.64% achieved occupancy, and no local-memory spills. A paired
full-model screen checked whether halving the launch width would reduce the
single-wave overhead:

| Remap warps | Decode | GPU step | Warm miss rate | Greedy SHA1 |
|---:|---:|---:|---:|---|
| 4 | 49.50 tok/s | 19.563 ms | 9.2533% | `f133a27bc01c` |
| **8 (retained)** | **49.74 tok/s** | **19.489 ms** | 9.2533% | `f133a27bc01c` |

The narrower block lost 0.5% and was removed. Any meaningful remap improvement
must therefore reduce the global victim-selection work rather than merely retune
the existing Triton launch width.

## Confirmed: the Ada expert-copy launch saturates PCIe

Nsight Compute measured the exact two-bank Q6 cold-copy node with eight missing
experts. The retained `1024 threads x 16 blocks/bank` launch completed in 1.61 ms,
read mapped host memory at 14.94 GB/s over PCIe, reached 65.42% occupancy, and
spilled no local memory. This is above the earlier application-level 12.8 GB/s
estimate and leaves no credible bandwidth win from widening the gather grid.

At the measured second-request miss rate (0.7403 expert/layer), the implied Q6
expert traffic remains a material decode cost, but it can only be reduced by a
better hit rate/cache representation or hidden under independent computation.
The transfer kernel itself is retained unchanged.

## Rejected: overlap expert copies with the shared expert

A CUDA-graph-safe fork/join candidate moved routed-expert copies to the prefill
side stream while the compute stream evaluated Ornith's independent shared expert.
This required splitting the existing two-launch routed+shared MMVQ fusion. The
candidate remained coherent, but the extra activation quantization, kernels, and
rounding-dependent routing cost more than the overlap hid:

| Path | Decode | GPU step | Second-request miss rate | Output |
|---|---:|---:|---:|---|
| **Fused routed+shared (retained)** | **49.74 tok/s** | **19.489 ms** | 9.2533% | coherent |
| Split shared / overlapped copy | 47.29 tok/s | 20.456 ms | 10.4036% | coherent |

The experiment was removed in full. The fused path remains both faster and more
numerically stable for the routing trajectory.

## Validation

- 59 relevant mixed-GGUF/offload/cache tests passed.
- The broader scheduler/engine/benchmark selection passed 253 tests; two unrelated
  tests select FlashInfer explicitly and fail because this environment does not
  install FlashInfer.
- Q4 and Q6 greedy output hashes remained stable throughout the accepted phase.
