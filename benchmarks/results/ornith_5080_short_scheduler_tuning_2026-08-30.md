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

## Accepted: task-aware pageable gather workers

The persistent CPU gather pool now waits only for workers that can contribute to the
current copy. A one-row decode miss has two weight-bank copies, so the caller and one
worker execute them while the other workers remain outside the completion barrier;
larger multi-agent batches still activate the full pool.

On a paired Q6/Q8 gate with the same eight pageable layers, two 1,024-token runs
averaged 53.24 tok/s versus 52.27 tok/s (+1.9%), while measured pageable gather time
fell from 8.70 to 8.25 seconds (-5.2%). Four-session simultaneous decode remained
batch-scaled and edged from 105.47 to 106.17 tok/s (+0.7%). All single- and
four-session arithmetic answers were coherent.

## Accepted: exact three-agent elastic graph

The 2-to-4 on-request tier now captures decode batch size 3 as well as 1, 2, and
4. A main agent with two helpers no longer pads every decode step with a fourth
dummy request. Two paired three-session Q6/Q8 runs raised simultaneous decode
from 79.72 to 91.14 tok/s (+14.3%) and end-to-end aggregate decode from 53.90 to
58.78 tok/s (+9.1%). All six answers were coherent. The extra graph is destroyed
when demand returns to two; both gates restored the 4,036-slot idle MoE cache from
the 3,632-slot burst geometry.

## Accepted: demand-sized elastic capacity

The scheduler now grows Hybrid-GDN state and graphs to the smallest tier that can
admit live demand instead of jumping directly from the initial tier to the maximum.
For three Q6/Q8 agents this used 19 rather than 25 physical GDN slots and retained
3,782 rather than 3,632 MoE slots. Against the exact-size-3 graph baseline,
simultaneous decode rose from 91.14 to 94.96 tok/s (+4.2%) and aggregate decode
from 58.78 to 60.40 tok/s (+2.8%). A four-agent regression gate still reached the
full 25-slot tier, produced all four correct answers, and delivered 110.43 tok/s
simultaneous decode. Teardown restored the original 13 GDN / 4,036 MoE slots.

## Accepted: coalesced intermediate shrink

Intermediate elastic shrink tiers now require two seconds of stable demand. Returning
to the compact initial tier remains immediate. In a staggered four-agent teardown this
removed the short-lived 4-to-3 recapture and went directly 4-to-2, restoring 13 GDN /
4,036 MoE slots without an extra rebuild. The four-agent regression remained coherent
at 105.65 tok/s simultaneous decode.

## Accepted: adaptive small-prompt grouping

The automatic one-lane GGUF policy now groups only fresh prompts of at most 1,280
templated tokens when the complete group fits one prefill budget. Continuations,
larger prompts, and explicit `--max-prefill-sequences` values retain their prior
semantics. Four ~1,032-token requests completed in 12.86 seconds versus 21.97 seconds
with unconditional serialization (-41.5%); the first arrival retained a 3.83-second
TTFT and the other three shared one prefill. All answers were coherent. Four
~2,056-token controls stayed on the serialized path, preserving main-agent latency.

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
