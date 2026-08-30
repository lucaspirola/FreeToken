# Ornith RTX 5080 elastic sessions and pageable CUDA graphs (2026-08-30)

Host: RTX 5080 16 GiB under WSL, approximately 29 GiB host RAM. The production
host-memory reserve remained 3 GiB throughout. These are short inner-loop gates;
the already completed 262K/1M capacity results remain the long-context reference.

## Accepted implementation

- Elastic hybrid-GDN capacity starts with four agents and admits up to eight. It
  compacts and copies every occupied recurrent-state slot, remaps live requests and
  radix snapshots, resizes CUDA graph coverage, and funds the temporary state from
  the MoE cache. When demand returns to four it releases the extra state and graphs.
- Q6 pageable expert staging no longer fences Python or copies a fixed 32-row device
  buffer. A `cudaLaunchHostFunc` node gathers only routed misses into 78.8 MiB of
  mapped pinned staging; the fused GPU scatter reads it directly. The side stream is
  overlapped with the independent shared-expert calculation and is graph replayable.
- Pageable row gathering now uses a persistent four-worker CPU pool (override with
  `FREETOKEN_PAGEABLE_GATHER_THREADS`). The pool parallelizes the two packed expert
  banks and routed rows without creating threads in the CUDA callback.
- Pageable staging is now charged to the same resident-host budget as the expert
  banks (157.5 MiB at an eight-agent Q6 ceiling), so the configured 3 GiB host-RAM
  reserve remains intact instead of being reduced implicitly by wider admission.
- Elastic MoE rebuilds refresh the pageable scatter's cache-sized identity index.
  This fixes Q6 graph recapture across 4→8→4, while retaining the existing bounded
  staging arena and resetting its telemetry to the same epoch as the GPU counters.
- Idle decode telemetry ranks layers by measured gather time per step (using miss
  count estimates for presently pinned layers), persists a profile tied to the exact
  GGUF path/size/mtime, and applies it on the next clean start. Live WSL
  `cudaHostRegister` swaps are not enabled by default because a rejected registration
  can poison that CUDA context.

## Coherence and performance gates

| Pair / gate | Prefill instant | Prefill average | Simultaneous decode | Result |
|---|---:|---:|---:|---|
| Q6_K + Q8_0, 4 agents, 4,096 prompt + 256 output, zero-copy graph path | 1,733.34 tok/s | 1,733.34 tok/s | 106.37 tok/s | 4/4 coherent |
| Q4_K_M + INT4, elastic 4→8→4, 4,096 prompt + 128 output | 5,122.26 tok/s | 5,122.26 tok/s | 171.10 tok/s | 8/8 coherent |
| Q6 placement training run, 4,096 + 160 | 757.06 tok/s | 757.06 tok/s | 84.12 tok/s | 4/4 coherent |
| Q6 persisted placement applied, same short gate | 819.99 tok/s | 819.99 tok/s | 94.38 tok/s | 4/4 coherent |
| Q6 parallel pageable gather, 4,096 + 256 | 852.70 tok/s | 852.70 tok/s | 105.00 tok/s | 4/4 coherent |
| Q6 serial gather control, 4,096 + 256 | 2,206.21 tok/s | 2,206.21 tok/s | 86.21 tok/s | 4/4 coherent |
| Q6/Q8 elastic 4→8→4, 4,096 + 128 | 2,113.14 tok/s | 2,113.14 tok/s | 126.62 tok/s | 8/8 coherent |
| Q4/INT4 elastic 4→8→4, 4,096 + 128 | 5,042.52 tok/s | 5,042.52 tok/s | 229.47 tok/s | 8/8 coherent |

The Q4 burst changed GDN slots `25 → 49 → 25` and MoE slots
`5,635 → 4,682 → 5,635`; exact restoration means there is no permanent decode
residency toll after helper agents stop. The Q6 profile application reduced pageable
rows from 8,446 to 7,562 (10.5%) and measured gather time from 4.12 s to 3.66 s while
improving simultaneous decode by 12.2% in that paired run. The higher 106.37 tok/s
Q6 result remains the accepted best because short-run host-copy timing is variable.
In consecutive production-shape gates, four workers improved simultaneous decode
from 86.21 to 105.00 tok/s (+21.8%) and reduced measured gather time from 4.79 to
1.61 seconds. Host-budget fluctuation made the serial run use ten pageable layers
versus nine for the parallel run, so the decode delta is indicative rather than a
strict paired estimate. Normalized measured gather bandwidth was 16.4 versus
6.1 GiB/s; the identical-geometry callback microbenchmark measured 36.2 versus
10.8 GiB/s (3.3x). The serial run's unusually high prefill is unrelated because
pageable gathering is decode-only.

The Q6 elastic run used ten pageable layers and restored GDN slots `49 → 25` and
MoE slots `3,033 → 3,736` after the burst. Its 126.62 tok/s simultaneous decode is
18.5% above the same-day four-agent 106.82 tok/s control, with no permanent
small-batch residency toll. Expansion still has a visible one-time pause because
several GiB of MoE cache must be rebuilt to fund the extra recurrent state; elastic
eight-agent mode is therefore a throughput option, not a latency-free admission.

## Rejected measured hot-row tier

A 512 MiB pinned tier selected the most frequent individual experts from the saved
LFU profile, trading one fully pinned layer for 208 hot rows spread over ten pageable
layers. It passed all four coherence/isolation checks and reduced staged traffic from
14.66 to 14.28 GiB, but simultaneous decode fell from 106.82 to 101.79 tok/s and
prefill fell from 2,374.64 to 1,400.67 tok/s. The implementation was removed rather
than retaining a losing default or dormant complexity.

## Command shape

Q4 elastic production shape:

```bash
ft serve --model /path/to/Ornith-1.5-35B-Q4_K_M.gguf \
  --moe-backend offload --moe-cache-auto \
  --max-running-requests 8 --elastic-initial-requests 4 \
  --num-tokens 262144 --kv-cache-dtype int4 \
  --kv-grow-step-tokens 65536 --host-ram-reserve-gb 3
```

Q6 pageable/profile shape (four resident lanes, optional burst to eight):

```bash
ft serve --model /path/to/Ornith-1.5-35B-Q6_K.gguf \
  --moe-backend offload --moe-cache-auto --moe-pageable-gpu \
  --moe-collect-stats --max-running-requests 8 \
  --elastic-initial-requests 4 \
  --num-tokens 262144 --kv-cache-dtype q8_0 \
  --kv-grow-step-tokens 131072 --host-ram-reserve-gb 3
```
