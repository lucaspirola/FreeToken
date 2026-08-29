# Ornith RTX 5080 scheduler, memory-safety, and 1M gates

Date: 2026-08-30  
Host: NVIDIA GeForce RTX 5080 (sm_120), 16 GiB, WSL  
Models: Ornith-1.5-35B-A3B GGUF Q4_K_M and Q6_K

This pass hardened elastic multi-agent serving before further throughput tuning.
The production host-RAM reserve remains **3 GiB**. FreeToken estimates the GGUF
expert-bank allocation before loading it, rejects an unsafe configuration early,
and can file-back selected pageable expert layers when the pinned-memory budget is
smaller than the model.

## Scheduler and session lifecycle

- Adaptive scheduling uses observed forward latency to resize prefill lanes and
  decode bursts. In the retained two-agent Q4 A/B, helper TTFT fell from 40.03 to
  36.82 seconds, the established agent advanced 303 rather than 196 tokens, and
  its longest pause fell from 9.92 to 8.82 seconds. Both answers were coherent.
- Automatically derived Claude Code and Codex sessions have a 30-second soft idle
  grace. After it expires their cached prefix becomes pressure-evictable, while an
  explicit session remains a hard lease. A stopped launched client closes all
  sessions belonging to its launch immediately.
- Finished request pages become evictable and the growable arena compacts surviving
  private tails before decommitting complete VMM segments. Returned VRAM expands the
  expert cache, so remaining agents do not keep paying the departed agent's KV cost.

## Pageable Q6 placement

The exact Q6 host-bank estimate is 24.61 GiB (Q4 is 18.16 GiB). Under the measured
post-weight host budget, Q6 selected ten low-miss pageable layers: 7, 8, 14, 15, 17,
18, 19, 20, 30, and 39. Compared with the generic placement, sampled pageable rows
fell from 2,235 to 1,504, staged traffic from 5.371 to 3.615 GiB, and pageable gather
time from 1.687 to 0.746 seconds. The answer remained coherent.

## Growable-VMM failure and fix

An early Q4 attempt failed at the 327,680-token growth boundary with
`cuMemSetAccess: CUDA_ERROR_NOT_READY`; the WSL/DXG driver logged a failed residency
operation. This was GPU commit headroom, not host-RAM OOM. Rebuilding the expert
cache through PyTorch's caching allocator left hundreds of MiB reserved in partially
occupied segments.

Growable KV now allocates its large expert-cache banks directly through CUDA VMM,
releases those mappings before replacement, reserves a permanent 256 MiB commit
cushion, and checks live driver-visible free memory before every growth. If the
planned expert geometry cannot fund the next mapping plus the guard, growth is
refused before the CUDA context is damaged.

## Exact 1 Mi-token capacity gates

Each gate used a 1,048,576-token ceiling, a 1,048,448-token deterministic needle
prompt, and a 128-token output allowance. Thus the prompt plus requested generation,
not the prompt alone, occupies the complete context budget. Both recovered passcode
`5663623` and returned a coherent explicit answer.

| Exact gate | Q4_K_M + Q4_0/INT4 KV | Q6_K + Q8_0 KV |
|---|---:|---:|
| Growth increment | 65,536 tokens | 131,072 tokens |
| YaRN factor | 4 | 4 |
| Average prefill | **331.82 tok/s** | **207.73 tok/s** |
| Final partial-chunk instant prefill | 163.83 tok/s | 149.16 tok/s |
| Timed decode | **46.03 tok/s** | **15.21 tok/s** |
| Final physical KV | 5.70 GiB | 10.70 GiB |
| Final expert-cache slots | 2,952 | 258 |
| Coherent needle retrieval | PASS | PASS |

The Q6 gate uses `--linear-state-slots 5`, the minimum single-request GatedDeltaNet
state geometry. The default nine slots correctly fails preflight at this context
size (11.32 GiB required versus an 11.16 GiB guarded budget). It also uses pageable
GPU experts, `FREETOKEN_PIN_BUDGET_GB=20`, and `--memory-ratio 0.99`. This is a
single-session capacity configuration, not the recommended geometry for many live
agents.

The Q4 gate finished with 1.00 GiB live free versus 0.60 GiB required by its final
commit. Q6 finished with 1.77 GiB free versus 1.58 GiB required. Neither run OOMed.

## Next throughput work

NVTOP showed that CUDA-busy time did not translate into full effective compute:
roughly 50% for Q4 and 22% for Q6 in the observed phases. The next work is a shared
overlapped prefill/decode scheduler pipeline for both pairs, plus asynchronous
pageable-expert prefetch for Q6. Iteration should use short representative A/B runs
with instant and average prefill, decode latency, fairness, and a coherence gate.
Only retained winners need another long-context validation; a 1M run is no longer an
inner-loop benchmark.
