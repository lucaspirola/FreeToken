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

## Validation

- 59 relevant mixed-GGUF/offload/cache tests passed.
- The broader scheduler/engine/benchmark selection passed 253 tests; two unrelated
  tests select FlashInfer explicitly and fail because this environment does not
  install FlashInfer.
- Q4 and Q6 greedy output hashes remained stable throughout the accepted phase.
