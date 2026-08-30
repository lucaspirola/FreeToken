# Ornith RTX 5080 short scheduler tuning

Date: 2026-08-30  
Host: NVIDIA GeForce RTX 5080 (sm_120), 16 GiB, WSL  
Workload: independent 4,096-token prompts, 256 forced decode tokens, LFU expert cache,
growable KV, four concurrent requests. Q6 used `FREETOKEN_PIN_BUDGET_GB=20` plus
pageable GPU experts. The configured host-RAM reserve remained 3 GiB.

The inner loop deliberately stopped using 1M prompts: these experiments target batch
formation and expose the utilization difference immediately. Each result checked the
answer only through its first `<|im_end|>` because the throughput request sets
`ignore_eos=true` to obtain an exact decode count. Every retained run recovered its own
numeric needle and no other request's needle in that answer prefix.

## Accepted: one GGUF MoE prefill lane, batched decode

Grouped decode scales, but grouped independent prefills are unusually slow on this
GGUF MoE path. The scheduler now auto-selects one prefill sequence per forward for
growable quantized GGUF MoE models with concurrency, rotates unfinished long prompts between
chunks, and continues to batch every runnable decode request. Other model families keep
their prior grouped-prefill behavior. `--max-prefill-sequences 0` restores it explicitly.

| Metric | Q4/INT4 grouped | Q4/INT4 one lane | Q6/Q8 grouped | Q6/Q8 one lane |
|---|---:|---:|---:|---:|
| Wall time | 13.469 s | **7.265 s (-46.1%)** | 34.702 s | **27.385 s (-21.1%)** |
| Aggregate prompt rate | 1,216.40 tok/s | **2,255.31 tok/s (+85.4%)** | 472.13 tok/s | **598.28 tok/s (+26.7%)** |
| Final prefill instant | 5,066.90 tok/s | **5,355.39 tok/s** | 646.16 tok/s | **891.24 tok/s** |
| Final prefill average | 1,277.08 tok/s | **5,355.39 tok/s** | 590.28 tok/s | **891.24 tok/s** |
| Simultaneous decode | **318.90 tok/s** | 271.79 tok/s | **108.61 tok/s** | 106.90 tok/s |
| Answer-prefix isolation | pass | pass | pass | pass |

The status reporter's average is keyed by batch geometry, so the one-lane series reports
the last matching single-lane shape while aggregate prompt rate is the end-to-end metric.
Q4 trades some short-run simultaneous decode throughput for a much larger prefill/latency
win; Q6 decode is effectively unchanged.

## Decode concurrency ceiling screen

Q4 with eight live decode streams captured graphs for `[1, 2, 4, 8]` successfully and
left 0.38 GiB free after capture. Its eight-stream simultaneous decode rate was
403.86 tok/s versus 318.90 tok/s at four streams (+26.6%). All eight answer prefixes
were isolated and coherent. This is an optional throughput geometry, not the production
default: reserving eight GDN/graph lanes reduced the initial MoE cache from 5,635 to
4,823 slots, which imposes a permanent small-batch decode toll even when the extra agents
are absent. Dynamic graph/GDN-lane growth is the follow-up required before making eight
on-request lanes free at idle.

## Rejected screens

- A 20 ms idle admission window successfully placed four requests in the first prefill,
  but Q4 average prefill collapsed from 1,277.08 to 337.77 tok/s and wall time grew from
  13.469 to 51.045 seconds. Admission coalescing was removed.
- Re-enabling scheduler/forward overlap with growable KV reduced main-agent progress by
  57.8% during helper prefill and did not improve helper latency. It remains disabled.
- Two output-row warps improved an isolated Q6 fused-expert microkernel, but the paired
  live run delivered only 79.54 simultaneous tok/s versus 108.61 with the retained
  four-warp kernel. Both Q4 and Q6 therefore keep four warps.
- Eight output-row warps lost 3-6% for Q4 and tied Q6 in the isolated screen.
- Replacing Q6's static pageable-layer ranking with the nine lowest miss-count layers
  from a 535-step four-agent profile reduced predicted staged rows by about 10%, but
  live simultaneous decode fell from 97.2 tok/s (old placement, profiling enabled) to
  90.2 tok/s (candidate, profiling disabled). Per-layer page/stall locality dominates
  the raw miss count; the existing ranking remains in place.
