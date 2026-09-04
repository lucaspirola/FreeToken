# Nemotron-3.5-Lightning-30B-A3B-NVFP4 — Phase 2 task 2B4 (offload cache sizing + backend study)

Run date 2026-09-04, HEAD `1184c4d` plus the working-tree benchmark changes listed at the
end. Host: WSL2, RTX 5080 16 GB (sm_120), CUDA 13 / Torch 2.11 / Triton 3.6, 33 GiB RAM,
Core Ultra 7 265K (16 cores, **no AVX-512**). Every GPU run under `scripts/gpu_lock.sh`;
the GPU was shared (serially) with the sibling needle-investigation agent.

Model geometry: 52 layers (23 Mamba-2, 23 MoE, 6 full attention), hidden 2688,
128 routed experts at top-6, **ungated ReLU²** experts (I=1856), one shared expert.

Common server flags unless stated: `FREETOKEN_PIN_BUDGET_GB=17`, `--moe-backend offload`,
`--kv-cache-dtype q8_0 --attention-backend triton`, `--memory-ratio 0.85`,
`--max-prefill-length 8192`, `--host-ram-reserve-gb 3`, no `--moe-pageable-gpu`.

---

## 1. `ft bench bw` at the real (ungated) geometry

`benchbw.py` assumed SwiGLU everywhere (`2 * I` rows on the gate_up side). It now carries a
`Workload.gated` flag and a `nemotron3.5-lightning` preset; ungated is implemented for the
two formats that actually have an ungated runtime layout (`bf16`, `nvfp4`) and raises
`NotImplementedError` for the rest instead of inventing a size. Unit tests in
`tests/moe/test_benchbw.py` (11 cases).

```
ft bench bw --model nemotron3.5-lightning --isa auto
```

| | value |
|---|---|
| per-expert bytes (ungated NVFP4) | **5 621 632 B = 5.3612 MiB** (gated would be 8 431 616 B) |
| routed banks, 23 layers × 128 experts | **15.413 GiB** (matches the plan's 15.39 GiB) |
| CPU STREAM read ceiling | 109.0 GB/s |
| PCIe linear H2D / D2H ceiling | 56.8 / 39.6 GB/s |
| **PCIe expert gather** (`copy_missing`, fused) | **52.9 GB/s** |
| **CPU MoE GEMV** (`CpuMoeExecutor`, `avx2+vnni(nvfp4-w4a8)`) | **66.9 GB/s** |
| CPU/PCIe ratio | 1.26× (threshold 2.0×) → **offload** |
| overlapped pair | CPU 45.2 + PCIe 46.2 GB/s → hybrid fetch split 50.5 % |

### The plan's "cpu/hybrid unavailable" note is stale

`tasks/nemotron35-plan.md` says `_cpu_moe_act_ok` (engine) excludes `relu2`, so cpu/hybrid
are unavailable for this checkpoint. That is **no longer true at HEAD**:

- `engine.py:_CPU_MOE_ACTS` already lists `"relu2"`.
- `moe/cpu_executor.py` maps `relu2 → ActKind 4`, requires NVFP4 banks for it
  (`cpu_executor.py:174`), and handles the ungated bank shape explicitly
  (`expected_up_rows = I if activation == "relu2" else 2 * I`, `cpu_executor.py:394`).
- The compiled extension on this host reports `max_generic_act_id() == 4`.
- The activation is applied in the ISA-independent epilogue (`act_apply`, `cpu_moe_ext.cpp:84`),
  and the NVFP4 dot product has a real AVX2 path (`dot_nvfp4_avx2`), so **AVX-512 is not
  required**. A direct construction on this host succeeds:
  `CPU MoE executor ready: threads=4 isa=avx2+vnni(nvfp4-w4a8) fmt=nvfp4 H=2688 I=1856 act=relu2`.

So `cpu`/`hybrid` are *available*; `ft bench bw` recommends `offload` on bandwidth, not on
capability. Measured end-to-end below (§2d). No implementation was done for this — it
already exists.

---

## 2. Decode telemetry

All decode runs: `bench_decode_moe.py` through `/v1/chat/completions`, AIME-25 problems,
`--greedy` (so every configuration decodes the **same** tokens and the routing is identical
across the sweep — all six sweep rows share output sha1 `f9232f2f7a68`), 256 tokens, warm-up
generation first. `--moe-collect-stats` counters are **differenced** across the warm-up and
measured windows, so the miss rates below are steady-state, not cold-start.

Per-expert bytes 5.36 MiB, 23 MoE layers, measured PCIe gather 52.9 GB/s, so
`PCIe ms/step = 23 × missing_per_layer × 5.36 MiB / 52.9 GB/s`.

### 2a. Cache rate × policy at bs=1 (`--moe-cache-rate`, 2 944 total experts)

| rate | slots | policy | decode tok/s | ms/step | miss/layer | miss rate | PCIe MiB/step | PCIe ms | PCIe % of step |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0.4 | 1 178 | lru | 95.01 | 10.53 | 1.696 | 28.3 % | 209.2 | 4.15 | 39 % |
| 0.4 | 1 178 | **lfu** | **107.59** | 9.29 | 1.563 | 26.1 % | 192.7 | 3.82 | 41 % |
| 0.5 | 1 472 | lru | 114.77 | 8.71 | 1.235 | 20.6 % | 152.3 | 3.02 | 35 % |
| 0.5 | 1 472 | **lfu** | **123.81** | 8.08 | 1.093 | 18.2 % | 134.7 | 2.67 | 33 % |
| 0.6 | 1 767 | **lru** | **138.31** | 7.23 | 0.789 | 13.2 % | 97.3 | 1.93 | 27 % |
| 0.6 | 1 767 | lfu | 130.02 | 7.69 | 0.719 | 12.0 % | 88.6 | 1.76 | 23 % |
| auto | 1 832 | lru | **143.22** | 6.98 | 0.718 | 12.0 % | 88.6 | 1.76 | 25 % |

**LFU always has the lower miss rate** (26.1 vs 28.3, 18.2 vs 20.6, 12.0 vs 13.2 %), but it
only pays off while misses still dominate. At rate 0.6 the miss rate is low enough that LFU's
extra per-step frequency bookkeeping and aging costs more than the ~0.2 ms of PCIe it saves,
and LRU wins by 6 %. Crossover is around a **15 % miss rate**.

`--moe-cache-auto` at bs=1 resolves to 1 832 slots (rate 0.62) and is the best single-stream
configuration measured — there is no reason to pin a rate below it at bs=1.

### 2b. Concurrency and the bs=16 cliff

`--moe-cache-auto`, `--nvfp4-backend triton`. "Slots" is the auto-resolved
`moe_cache_size`; the recurrent-state pool takes `4·mr + max(4, 2·mr) + 1` slots of 47 MiB
each and is funded out of the same VRAM, which is why the expert cache shrinks as
`--max-running-requests` grows.

| bs | state slots | state GiB | expert slots | active experts/layer | miss/layer | miss rate | ms/step | per-stream tok/s | aggregate tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9 | 0.41 | 1 832 | 6.00 | 0.72 | 12.0 % | 6.98 | 143.2 | **143.2** |
| 2 | 13 | 0.60 | 1 797 | 11.61 | 1.93 | 16.6 % | 11.38 | 87.9 | **175.3** |
| 8 | 49 | 2.25 | 1 483 | 38.46 | 13.58 | 35.3 % | 47.08 | 21.2 | **169.7** |
| 16 (lru) | 97 | 4.45 | 1 063 | 61.48 | 61.21 | **99.6 %** | 183.0 | 5.5 | **87.4** |
| 16 (**lfu**) | 97 | 4.45 | 1 063 | 61.58 | 31.45 | **51.1 %** | 95.0 | 10.5 | **168.2** |

At bs=16 the decode working set is 61.5 distinct experts/layer × 23 layers ≈ **1 414 experts**,
which does not fit the 1 063 slots the auto sizer can afford after the 4.45 GiB state pool.
**LRU degenerates to a 99.6 % miss rate — complete thrash**, and the step lands on the PCIe
roofline (7.5 GiB/step / 52.9 GB/s = 150 ms of the measured 183 ms, 82 %).
**LFU is the fix: it pins the hot experts instead of cycling the whole set, halving the miss
rate to 51.1 % and nearly doubling aggregate throughput (93.7 → 168.2 tok/s, 1.80×)** — a
free win from one flag. (Repeat runs put plain LRU at bs=16 between 87 and 94 tok/s, so the
LFU margin is far outside the run-to-run spread.)

Aggregate throughput peaks at bs=2–8 and *falls* at bs=16 under LRU; with LFU, bs=16 is back
level with bs=2/8.

### 2c. `--nvfp4-backend triton` vs `flashinfer` (b12x), end to end

The `benchmarks/results/nvfp4_moe_kernels_5080.jsonl` microbenchmark (cold L2, isolated GEMM)
says b12x wins prefill 2.3–5.9× and batched decode 1.6× at M≥8, and loses M≤4 decode. On the
**real offload path** none of that survives:

| workload | triton | flashinfer (b12x) | winner |
|---|---:|---:|---|
| decode bs=1 (aggregate tok/s) | **143.22** | 137.52 | triton +4 % |
| decode bs=2 | **175.31** | 172.87 | triton +1 % |
| decode bs=8 | **169.69** | 144.40 | **triton +18 %** |
| decode bs=16 | 87.39 | 87.68 | tie |
| 32K prefill, end-to-end tok/s (run 1 / run 2) | **5 622.6 / 5 776.5** | 4 527.5 / 4 842.8 | **triton +19–24 %** |
| 32K prefill, engine average (run 1 / run 2) | **4 398.1 / 4 940.7** | 3 738.5 / 4 191.3 | triton +18 % |
| 32K prefill, engine instant, last chunk (run 1 / run 2) | 1 197.9 / 1 226.2 | **1 300.9 / 1 330.2** | b12x +8–9 % |
| decode after the 32K prefill | **114.9 / 116.2** | 112.2 / 110.6 | triton +2–5 % |

The 32K prefill A/B was run twice (the second round with every kernel cache already warm on
disk, so it is free of first-call autotune/JIT); both rounds agree on direction and margin.

Miss rates are identical between the two backends at every batch size (12.0/12.0 %,
16.6/17.0 %, 35.3/35.4 %, 99.6/99.6 %), so this is purely the GEMM path, not cache behaviour.

Why the flip: on the offload path the expert weights arrive **by DMA into the slot cache and
are then read L2-warm**, and both prefill and decode are dominated by moving those weights,
not by the GEMM. The roofline table above shows PCIe alone accounts for 25–88 % of every
decode step. b12x's tensor-core advantage applies to the shrinking remainder, while its
per-launch overhead and its separate bank layout apply to every call. The one place b12x
does show its microbenchmark form is the *instant* (steady-state, last-chunk) prefill rate,
+9 %; the end-to-end average is still triton's because b12x pays more in the early chunks.
Triton also got a large 2B2/2B3 tuning pass at `893f43f`/`5cfc5bf` (prefill M=8192 went
74.6 ms → 29.5 ms/layer in the same jsonl), which is what closed the gap.

### 2d. `--moe-backend hybrid` is not competitive (bs=1, rate 0.5, lru)

`ft bench bw` reports CPU-MoE 66.9 vs PCIe 52.9 GB/s (1.26×, below the 2× rule → `offload`),
but the *overlapped* pair sums to 91.4 GB/s, which looks like a reason to try hybrid anyway.
Measured end to end it is not close:

| backend | decode tok/s | ms/step | miss/layer | fetched over PCIe | computed on CPU |
|---|---:|---:|---:|---:|---:|
| offload | **118.13** | 8.47 | 1.235 | 0 % (all GPU) | — |
| hybrid | 32.90 | 30.40 | 1.317 | 31.9 % | 0.897 experts/layer |

Hybrid is **3.6× slower**. The CPU leg does not overlap the way the microbench pair does: it
runs per layer inside the step, contends with the tokenizer/scheduler processes for the same
cores and DRAM, and blocks the layer that needs it. `ft bench bw`'s `offload` recommendation
is right; the overlapped-sum reading is not a usable predictor.

### 2e. bf16 recurrent state: closed

The plan's "bf16 SSM state option" (to halve the 4.45 GiB state pool at bs=16 and buy ~415
expert slots) is **rejected at config time** on the kernel path:

```
ValueError: the Mamba-2 SSD kernels require an fp32 recurrent state pool;
FREETOKEN_MAMBA_SSM_DTYPE selected torch.bfloat16
```

So `--moe-cache-policy lfu` and `--elastic-initial-requests` are the only levers for the
bs=16 cliff today. (A control run with `FREETOKEN_MAMBA_SSM_DTYPE=float32` at bs=16 measured
88.1 tok/s, matching the other bs=16 LRU runs.)

---

## 3. Single-session 1M profile sizing

`--max-running-requests 1 --max-seq-len-override 1048576 --num-tokens 1048576
--kv-grow-step-tokens 131072 --kv-cache-dtype q8_0 --attention-backend triton
--moe-cache-auto --nvfp4-backend triton --linear-state-slots 5 --max-prefill-length 8192`,
synthetic-needle prompts of growing size.

### `--linear-state-slots` floor

Probed directly. `_linear_pool_num_slots` (`kvcache/linear_state_pool.py:295`) enforces
`4·max_running_requests + 1` for `hybrid_radix`:

| value | result |
|---:|---|
| 3 | `ValueError: --linear-state-slots 3 is below the 5-slot working-set floor for max_running_requests=1 and cache_type='hybrid_radix'` |
| 4 | same error |
| **5** | **accepted, serves** |

The default without the override is 9 slots (`4·1 + max(4, 2·1) + 1`), so pinning 5 returns
**4 slots × 47 MiB ≈ 188 MiB** to the expert cache (≈ 35 expert slots).

### Expert slots vs KV growth

Each committed 131 072-token KV step costs ~0.40 GiB and ~76 expert slots:

| committed KV | KV physical | MoE expert slots | cache rate |
|---:|---:|---:|---:|
| initial (`--moe-cache-auto`) | — | 1 786 | 0.607 |
| 262 144 | 0.84 GiB | 1 663 | 0.565 |
| 393 216 | 1.24 GiB | 1 586 | 0.539 |
| 524 288 | 1.64 GiB | 1 510 | 0.513 |
| 655 360 | 2.04 GiB | 1 434 | 0.487 |
| *1 048 576 (extrapolated, 8 steps)* | *3.3 GiB* | *≈ 1 180* | *≈ 0.40* |

So a full 1M-token session still leaves ~1 180 expert slots — a rate of 0.40, whose bs=1
decode cost is measured in §2a at ~95–108 tok/s before the attention/KV cost of the long
context is added. **VRAM is not the blocker for a 1M single session.**

### Measured at each committed size

| prompt tokens | prefill end-to-end | prefill engine avg | decode tok/s | needle |
|---:|---:|---:|---:|---|
| 131 088 | **3 007 tok/s** | 2 942 | **72.6** | PASS |
| 262 160 | 1 790 tok/s | 1 775 | **51.8** | FAIL |
| 524 304 | 997 tok/s | 995 | **32.0** | FAIL |

Decode scales as expected for a growing KV read (72.6 → 51.8 → 32.0 tok/s); prefill
throughput halves each time the context doubles, i.e. total prefill cost is quadratic — 526 s
for a cold 524K-token prompt.

**The needle is missed at 262K and 524K.** Caveats: this is the raw `/v1/completions` greedy
continuation that the sibling needle investigation has already shown to be an unanchored
marginal continuation at 131K (it passes here at 131K, and passed at 131K through the chat
endpoint in that investigation); and 262 144 is exactly the checkpoint's tokenizer
`model_max_length`, with 524K beyond it. Treat the **coherent ceiling as ~131K–256K** and
re-test the long end through `/v1/chat/completions` before advertising 1M. Sizing-wise the
profile is sound; quality above 256K is unproven and this run is evidence against it.
(These runs used a snapshot of `bench_long_context.py` taken before `ec54e21`, which moved the
needle probe onto the chat endpoint — so the 262K/524K retest is now a one-command follow-up
with the committed script.)

### Bug found: growable KV + `--nvfp4-backend flashinfer` cannot start

Every 1M run first crashed at init with

```
File "python/freetoken/kernel/vmm.py", line 76, in __init__
ValueError: unsupported VMM tensor dtype: torch.int32
```

`--kv-grow-step-tokens` sets `cache.direct_device_banks`, so the expert slot cache is
allocated as `VMMTensor`s; the b12x repacked banks include an int32 bank, and
`VMMTensor._DTYPE_NAMES` covers only uint8/int8/fp16/bf16/fp32/fp8. Growable KV therefore
works **only** with the triton bank layout today. Fix is either adding int32 to
`_DTYPE_NAMES` + `parse_dtype` in `csrc/vmm_tensor.cpp`, or rejecting the combination at
config time with a real message. Not implemented here (out of 2B4 scope, touches csrc).
Since triton is the recommended backend anyway (§2c), this is not blocking.

---

## 4. MTP go/no-go input (Phase 4 gate)

The plan's rule: *"from 2B4's per-step expert-miss telemetry, estimate the verify step cost
with up to 6(k+1) experts per MoE layer touched. If the projected decode gain at bs=1 is
< 1.25× on the offload path, Phase 4 is skipped."*

**Measured verify-step proxy.** A k=1 verify step is a 2-token forward. The bs=2 decode run
is exactly that shape and is measured directly:

| | tokens/step | active experts/layer | miss/layer | ms/step |
|---|---:|---:|---:|---:|
| normal decode (bs=1) | 1 | 6.00 | 0.718 | **6.98** |
| 2-token step (bs=2) | 2 | **11.61** of a possible 12 | 1.926 | **11.38** |

Two independent tokens touch 11.61 of 12 possible experts per layer — routing is **97 %
disjoint**, i.e. the "up to 12 experts/layer instead of 6" worst case is essentially what
happens. Expert traffic per layer rises 2.68× (0.718 → 1.926 misses) and the step costs
**T_verify / T_1 = 1.63×**.

**Projected gain.** With mean accepted length λ (vLLM saw ~63 % acceptance on Super → λ ≈ 1.63)
and a draft step costing one extra transformer block (1 of 23 MoE layers ≈ +5–7 % of a step):

```
speedup = λ / (T_draft/T1 + T_verify/T1) = 1.63 / (0.07 + 1.63) = 0.96×
```

For the ≥ 1.25× gate to be met, the verify step would have to cost
`T_verify/T1 ≤ λ/1.25 − 0.07 = 1.23×`, i.e. the two verified tokens would have to share
roughly 70 % of their experts. The telemetry says two tokens share 3 %.

Consecutive tokens *within one sequence* are more correlated than two different AIME problems,
so 1.63× is an upper bound on the verify cost and 0.96× a lower bound on the speedup. But the
gap is large: even a hypothetical perfectly-correlated verify (both tokens routing to the same
6 experts, so PCIe is unchanged and only compute doubles) projects
`1.63 / (0.07 + ~1.15) ≈ 1.34×` — and that is the *ceiling*, requiring routing behaviour the
measurements contradict.

> **Recommendation: NO-GO on Phase 4 (MTP).** The projected bs=1 decode gain on this host's
> offload path is 0.96–1.34×, centred well below the plan's 1.25× bar, and the pessimistic end
> is a *regression*. The mechanism is exactly the risk the plan flagged: on an offload MoE the
> step is PCIe-bound (25–88 % of every step is expert DMA), and a verify step that touches
> ~2× the experts pays ~2× the PCIe, cancelling the token amplification.
>
> Two cheaper wins are already banked by this study and should be taken first:
> `--moe-cache-policy lfu` at bs=16 (**1.80×** aggregate) and the cache-rate/backend defaults
> below (bs=1 went 95 → 143 tok/s across this sweep). Phase 4's flag should not be built.
> If MTP is ever revisited, the precondition is a resident (non-offloaded) expert set, not a
> better sampler.

---

## 5. Recommended defaults

### NVFP4 backend for Lightning: **triton**

Leave `EngineConfig.nvfp4_backend` at its `"triton"` default and **do not** pass
`--nvfp4-backend auto/flashinfer` for this checkpoint. `auto` currently resolves to `b12x`
for Lightning (sm_120, ungated relu2, `moe_intermediate_size` 1856 ≥ the 1024 threshold in
`moe/nvfp4_backends.py:_b12x_min_intermediate`), which this study measures as **slower or
equal at every batch size and 18–24 % slower on a 32K prefill**, and which additionally
cannot start with `--kv-grow-step-tokens`. The docs' claim that `auto` → b12x is the right
pick for Lightning is based on the isolated microbenchmark and is now contradicted end to end.

### Cache rate and policy

- **bs=1 / 1M profile**: `--moe-cache-auto` (resolves to ~1 830 slots, rate 0.62) with the
  default `--moe-cache-policy lru`. Do not pin a lower rate.
- **16-way profile**: `--moe-cache-auto --moe-cache-policy lfu`. LFU is worth **1.80×**
  aggregate throughput at bs=16 and is never worse than LRU whenever the miss rate is above
  ~15 %, which is every configuration with more than ~2 running requests.
- Rule of thumb from §2a: **LFU above a 15 % miss rate, LRU below it.** With
  `--moe-collect-stats`, the scheduler's idle `MoE decode miss stats` line reports the rate
  directly.
- `--moe-backend offload` (not `cpu`/`hybrid`), confirmed both by `ft bench bw` and by the
  3.6× end-to-end loss in §2d.

### Launch lines

P2 — 16 concurrent (the only change vs the previous line is `--moe-cache-policy lfu`):

```
FREETOKEN_PIN_BUDGET_GB=17 ft serve --model $LIGHTNING \
  --max-running-requests 16 --elastic-initial-requests 4 --kv-grow-step-tokens 65536 \
  --num-tokens 262144 --max-seq-len-override 131072 --kv-cache-dtype q8_0 \
  --attention-backend triton --moe-backend offload --moe-cache-auto \
  --moe-cache-policy lfu --memory-ratio 0.85 --max-prefill-length 8192 \
  --host-ram-reserve-gb 3 --enable-cache-report
```

1M single-session:

```
FREETOKEN_PIN_BUDGET_GB=17 ft serve --model $LIGHTNING \
  --max-running-requests 1 --max-seq-len-override 1048576 --num-tokens 1048576 \
  --kv-grow-step-tokens 131072 --kv-cache-dtype q8_0 --attention-backend triton \
  --moe-backend offload --moe-cache-auto --linear-state-slots 5 \
  --memory-ratio 0.85 --max-prefill-length 8192 --host-ram-reserve-gb 3 \
  --session-spill-ram-gb 12 --session-spill-dir <nvme>
```

`--nvfp4-backend` is deliberately absent from both (the `triton` default is the
recommendation). `--linear-state-slots 5` is the accepted floor and buys ~35 expert slots.

**Addendum 2026-09-04 (task 3F).** Five slots seat padding + live + 2 ping-pong + exactly one
idle session lease, so a second conversation's first turn finds the GDN pool full. That used to
kill the scheduler (`LinearStatePool exhausted: need 1, have 0` from the chunk commit's
unguarded `pool.alloc(1)`); it now spills the LRU idle lease on demand instead. The floor is
unchanged, but prefer `--linear-state-slots 6` when two sessions alternate — one extra slot
(47 MiB) avoids a checkpoint + restore per turn. See `docs/nemotron.md`.

---

## Reproduction

Working-tree changes this study needed (none committed):

- `python/freetoken/moe/benchbw.py` — `Workload.gated`, ungated bank specs, the
  `nemotron3.5-lightning` preset; `tests/moe/test_benchbw.py` (11 cases).
- `benchmarks/bench_decode_moe.py` — `--concurrency`, `--nvfp4-backend`,
  `--moe-collect-stats` (with cold/warm delta differencing), `--server-arg`;
  `tests/benchmarks/test_bench_decode_moe.py` (30 cases).
- `benchmarks/bench_long_context.py` — shares the same passthrough options, real `--cache-rate`.
- `python/freetoken/scheduler/scheduler.py` — `run_when_idle` now always logs
  `MoE decode miss stats per layer: <json>` when `--moe-collect-stats` is on;
  `tests/scheduler/test_moe_stats_logging.py` (4 cases).

`uv run pytest tests/benchmarks tests/scheduler tests/moe -q` → all pass; ruff clean on the
touched files.

Raw rows: `decode_sweep.jsonl`, `decode_backend.jsonl`, `decode_pol16.jsonl`,
`decode_family.jsonl`, `prefill32k.jsonl`, `onem.jsonl` and the per-run server logs under the
session scratchpad; `benchbw_lightning.json` for §1.
