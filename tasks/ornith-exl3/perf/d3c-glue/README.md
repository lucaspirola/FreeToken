# D3c: per-token glue (Ornith EXL3 5.0bpw, box ft-dev, RTX 5080 gen4)

These are box numbers, and every arm ran with nothing else on the GPU.

## The change: the shared-expert gate joins the shared expert on the side stream

`shared_expert_gate` is a 2048 -> 1 bf16 projection. cuBLAS runs it as a split-K gemv plus a `splitKreduce`, which cost ~5.5 us per MoE layer. It still ran on the main stream before the D3a fork, although only the final `fused_shared_expert_add_` reads it, after the join. It now runs on the side stream after the shared expert (`models/qwen3_5_moe/moe.py`, `_overlap_shared`).

- The kernels, their order and their inputs are the same, so the output is bit-identical. Greedy output at 85K is the same 818 tokens, sha1 `ed5c265a64`, as D2/D3a (`results/exl3greedy-f16acc-d3c2`).
- Tests: 357 passed, test_exl3 plus test_fused_moe including a new signed-zero tie case for the router (`results/exl3tests-d3c2`).

## (a) Saver E2E (mirror, ratio 1.00, flashinfer extend, PREROT=1, pass 2; bracketed D3a / D3c / D3a)

| prompt | D3a 1 | **D3c** | D3a 2 | change |
|---:|---:|---:|---:|---:|
| 8K decode tok/s | 146.2 | **152.0** | 146.0 | +4.0% |
| 32K | 139.0 | **144.4** | 138.9 | +3.9% |
| 80K | 117.7 | **122.0** | 117.7 | +3.7% |

- Prefill is unchanged: 7389 / 7377 / 7378 at 8K.
- Every arm had captures=1 and 0 tracebacks.
- The base arms reproduce D3a's 146.3 / 139.0 / 117.8.

## (b) Whole-model E2E (`--expert-residency whole`)

| prompt | D3a | **D3c** | change |
|---:|---:|---:|---:|
| 8K decode tok/s | 159.8 | **165.2** | +3.4% |
| 32K | 148.6 | **154.1** | +3.7% |
| 80K | 134.0 | **138.3** | +3.2% |

## (c) Profiled 8K decode, wall-clock split (`profile_step.py`, saver)

`profile_step.py` now also splits the decode window's wall clock among the kernels running at each instant (1/n each when n overlap) and reports the idle time. Summed kernel durations exceeded the window once the side stream existed (GPU busy 102-105%). PDL kernels also count their `gdc_wait` spin, which made the router look like 4.8 us/layer.

| ms/token | D3a (`results/exl3prof-d3c2-base`) | D3c (`results/exl3prof-d3c2`) |
|---|---:|---:|
| exl3 GEMV | 3.730 | 3.599 |
| copy: miss loads (`fast_index_copy_kinds`) | 1.577 | 1.548 |
| copy: saver writebacks (DtoH) | 0.383 | 0.395 |
| rest (router, cuBLAS, norms, GDN/attention kernels, mirror bookkeeping) | 1.729 | 1.721 |
| idle: no kernel running | 0.923 | 0.905 |
| window (under the profiler) | 8.34 | 8.17 |

**Derived no-miss decode at 8K** (D3c):

| basis | ms/token | tok/s |
|---|---:|---:|
| E2E (1/152.0 = 6.58 ms) minus the miss loads | 5.03 | **199** |
| the same, also minus the writebacks | 4.64 | 216 |
| lower bound: profiled window minus both copies | 6.22 | 161 |

Every basis clears the ft-dev kernel target of >= 150.

## Tried and dropped

`results/exl3bench-d3c`, `results/exl3tests-d3c`; the code is not kept.

**Packed-key router top-k:**
- The idea: one int64 max per pick (order-preserving logit image << 32 | inverted lane) instead of a float max, an index min and a gather sum.
- The ids are identical in every case.
- The renormalized weights differ in the last bit in 23 of 114 cases. The gathered values come out in another layout, so the K-axis sum runs in another order.
- Graph-replayed, it takes 1.34 us against 1.54 us per launch. The profile's 4.8 us was the PDL wait.
- Not worth a non-bit-exact router.

**One launch for the mirror's pre-ensure reset:**
- The idea: replace the `victim_ids`/`prior_ids` fills and the `prev_slot_of_id` snapshot copy with one kernel.
- It is correct, but it takes 1.90 us against 2.31 us per layer, 0.016 ms/token. Not worth new shared-mirror code.

## Found: the scheduler serializes host and GPU each decode step (not EXL3 code)

`job-nsys.sh` + `nsys_split.py` produce an Nsight Systems timeline with `--cuda-graph-trace=graph`, with no torch.profiler, so the overhead is low. Results in `results/exl3nsys-d3c2`.

What the timeline shows:
- Each decode step is one graph replay, median 5.84 ms.
- Between replays the GPU idles a steady **565 us** (p10 559, p90 577). That is **~0.56 ms per token, ~8% of a 6.6 ms token**.
- Eager GPU work between replays is only 0.04 ms/token, so the gap is host time.

The runtime API trace shows the cause:
- After launching step N, the host drains step N-1 and issues the saver's writeback DMAs (~1.8 ms).
- It then makes one synchronous device read: a `cudaMemcpyAsync` followed by `cudaStreamSynchronize`, which blocks ~3.9-4.8 ms until graph N finishes.
- Only then does it spend ~540 us preparing step N+1 (pageable H2D, cumsum, the static-buffer copies) and launch it.
- This happens in 120 of 127 steps.

The run used the overlap loop (`Scheduler loop: overlap`, `FREETOKEN_GROWABLE_OVERLAP=1`), but this read defeats it. The Python site is being located with `--python-backtrace=cuda`. Removing that sync would hide the 0.56 ms, about +8% (saver 152 -> ~166 tok/s on this box). The fix belongs in the scheduler/residency code that every model shares, not in the EXL3 kernels.
