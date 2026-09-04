# NVIDIA Nemotron on FreeToken

Serving notes for the `NemotronHForCausalLM` family. The checkpoint rows live in
[`docs/models.md`](models.md); the Switchyard router contract lives in
[`docs/switchyard.md`](switchyard.md).

## Status

| Phase | State |
|---|---|
| Phase 1 — bring-up | done |
| Phase 2 — kernels (Mamba-2 SSD + sm_120 MoE fast path) | in progress |
| Phase 3 — Switchyard compliance | done |
| Phase 3 — soak run | pending |

## Nemotron 3 Super

Nemotron 3 Super uses its native hybrid Mamba-2 / full-attention / latent-MoE
architecture. The NVFP4 release needs about 60 GiB of host RAM for expert banks and
10.3 GiB of resident GPU weights. Its Mamba-2 recurrent state is ~160 MiB per
sequence, so FreeToken serves one concurrent Super session (`single_stream_only`,
which forces `--max-running-requests 1` and a bs=1 decode graph). On WSL,
`--moe-pageable-gpu` keeps the pin-budget overflow banks pageable, stages only their
routed misses through a small pinned buffer, and still executes every ReLU² expert on
GPU. Decode gathers are CUDA-graph host nodes and overlap the shared expert
calculation; idle telemetry saves a model-scoped time-cost ranking that is applied on
the next clean start. A minimal all-GPU-compute launch is:

```
ft serve --model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
  --max-running-requests 1 --moe-backend offload --moe-cpu-layers 0 \
  --moe-pageable-gpu --moe-cache-auto
```

## Nemotron 3.5 Lightning

Nemotron 3.5 Lightning (30B-A3B) is the same `NemotronHForCausalLM` family with a
smaller, non-latent geometry: 52 layers (23 Mamba-2, 23 MoE, 6 full-attention),
hidden 2688, 128 routed experts at top-6 with **ungated ReLU²** experts (up+down
only, I=1856) plus one shared expert.

### Sizing on a 16 GB GPU

- **Host RAM**: the NVFP4 routed-expert banks are 15.4 GiB (≈16.5 GB). With
  `--host-ram-reserve-gb 6` and the ~4 GiB non-bank process footprint, plan on
  ≥ 23 GiB of MemAvailable. Run `python benchmarks/preflight_nemotron_host.py`
  first — it reports MemAvailable/SwapFree, the pin budget, the pinned/pageable
  layer split, VRAM holders, stale `~/.cache/torch_extensions` locks and stray
  workers, and exits non-zero when the host cannot take the load.
- **VRAM**: ~2.3 GiB of resident weights, 47 MiB of Mamba-2 state per sequence
  slot, ≤ 0.8 GiB of `q8_0` KV at 256K tokens, and the rest to the MoE slot cache.
- **Context**: `max_position_embeddings` is 1,048,576, but the tokenizer's
  `model_max_length` is 262,144 — treat 262K as the served ceiling and pin the
  working window with `--max-seq-len-override`.
- **Concurrency**: the 47 MiB state fits many slots, so Lightning is *not*
  `single_stream_only`: up to 16 concurrent requests, with
  `--elastic-initial-requests` starting the recurrent-state/graph working set
  small and growing it on demand.
- **WSL pin quota**: the CUDA host-registration budget is 0.4 × RAM. Below the
  15.4 GiB of banks, the overflow layers need `--moe-pageable-gpu` (which disables
  the decode CUDA graphs). Raising `FREETOKEN_PIN_BUDGET_GB` to ≥ 17 (backed by a
  `.wslconfig` `memory=` large enough to hold it) pins every layer and keeps the
  graphs; the preflight script prints which side of the line this host is on.
- **Expert GEMM**: keep the **`triton`** default — do not pass `--nvfp4-backend auto`
  or `flashinfer` for this checkpoint. In isolation flashinfer's sm_120 W4A16 fused MoE
  (`b12x`) looks like the winner (`benchmarks/bench_nvfp4_moe_kernels.py`, per MoE layer,
  cold L2: 2.3× Triton on an M=8192 prefill chunk, 1.6× on batched decode at M=8/16), and
  `auto` therefore resolves to it here (sm_120, ungated relu2, `moe_intermediate_size`
  1856 ≥ the 1024 threshold). **End to end on the offload path it loses**
  (task 2B4, `benchmarks/results/nemotron35_lightning_5080_cache_study_2026-09-04.md`):
  32K prefill 5 623–5 777 tok/s on Triton vs 4 528–4 843 on b12x (Triton +19–24 %, two
  rounds), decode +4 % at bs=1, +18 % at bs=8, tied at bs=2/16. On the offload path the
  experts arrive by DMA and are read L2-warm, and 25–88 % of every decode step is expert
  PCIe traffic, so the tensor-core advantage applies only to the shrinking remainder while
  b12x's launch overhead applies to every call. b12x also **cannot start with
  `--kv-grow-step-tokens`**: growable KV allocates the slot cache as VMM tensors and the
  repacked b12x banks include an int32 bank that `VMMTensor` does not support. Revisit if
  the expert set ever becomes GPU-resident. `--nvfp4-backend marlin` is rejected at config
  time (its fused kernel assumes a gated `[2I, H]` bank and a silu epilogue), and
  `--moe-backend cpu`/`hybrid` pins the layout back to `triton` because CPU decode reads
  the native ModelOpt rows.
- **MoE backend**: `offload`. `cpu`/`hybrid` *are* available for this checkpoint — the
  CPU MoE executor handles ungated ReLU² NVFP4 banks on plain AVX2+VNNI (no AVX-512
  needed), and `ft bench bw --model nemotron3.5-lightning` measures the kernel at
  66.9 GB/s against a 52.9 GB/s PCIe gather — but the 1.26× ratio is below the 2× hybrid
  threshold, and measured end to end `hybrid` decodes **3.6× slower** than `offload`
  (32.9 vs 118.1 tok/s at bs=1).
- **Expert-cache policy**: `--moe-cache-policy lfu` for the 16-way profile,
  the `lru` default for single-stream. At bs=16 the decode working set is ~61 distinct
  experts per layer (~1 414 across 23 layers) against the ~1 063 slots left after the
  4.45 GiB recurrent-state pool, and LRU degenerates to a **99.6 % miss rate**; LFU pins
  the hot experts, halves that to 51 %, and is worth **1.80×** aggregate throughput
  (93.7 → 168.2 tok/s). Rule of thumb: LFU above a ~15 % miss rate, LRU below it — read
  the rate off the scheduler's idle `MoE decode miss stats` line under
  `--moe-collect-stats`. `FREETOKEN_MAMBA_SSM_DTYPE=bfloat16` is **not** an option for
  shrinking the state pool: the Mamba-2 SSD kernels require an fp32 state pool and reject
  it at config time.

### Launch profiles

P1 — bring-up profile (single stream, no quantized KV):

```
ft serve --model ~/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --max-running-requests 1 --moe-backend offload --moe-pageable-gpu --moe-cache-auto \
  --num-tokens 65536 --memory-ratio 0.85 --max-prefill-length 8192 --host-ram-reserve-gb 6
```

P2 — serving profile (16 concurrent, elastic KV, prefix cache, quantized KV):

```
FREETOKEN_PIN_BUDGET_GB=17 \
ft serve --model ~/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --max-running-requests 16 --elastic-initial-requests 4 --kv-grow-step-tokens 65536 \
  --num-tokens 262144 --max-seq-len-override 131072 --kv-cache-dtype q8_0 \
  --attention-backend triton --moe-backend offload --moe-cache-auto \
  --moe-cache-policy lfu \
  --memory-ratio 0.85 --max-prefill-length 8192 --host-ram-reserve-gb 6 --enable-cache-report
```

`--moe-cache-policy lfu` is the 2B4 recommendation and is worth 1.80× aggregate decode at
16 concurrent requests. With `FREETOKEN_PIN_BUDGET_GB=17` every expert layer is pinned and
`--moe-pageable-gpu` is not needed (keeping the decode CUDA graphs).

Quantized KV requires `--attention-backend triton`; bf16 KV with the FlashInfer
backend is the fallback (KV is only +0.75 GiB at 262K). `--tool-call-parser auto`
resolves to `qwen3_coder` and `--reasoning-parser auto` to `qwen3` for this
checkpoint.

Add the Switchyard serving-compliance flags from [`docs/switchyard.md`](switchyard.md)
when FreeToken is fronted by the router.

### KV dtype

Phase 1 A/B (2026-09-04) chose `q8_0` — `fp8_e4m3` flipped first tokens on
cached-prefix reuse 3/6 runs; equal VRAM and reasoning score. See
`benchmarks/results/nemotron35_lightning_5080_2026-09-04.md`.

### Prefill chunk size

`--max-prefill-length 8192` with `--memory-ratio 0.85`. The SSD kernels are in; a 32 768-token
synthetic needle prefills at 5 623–5 777 tok/s end to end and decodes at ~115 tok/s.

### Measured throughput (2026-09-04, task 2B4)

Decode through `/v1/chat/completions`, `--moe-cache-auto`, Triton expert GEMM:

| running requests | expert slots | decode miss rate | per-stream tok/s | aggregate tok/s |
|---:|---:|---:|---:|---:|
| 1 | 1 832 | 12.0 % | 143.2 | 143.2 |
| 2 | 1 797 | 16.6 % | 87.9 | 175.3 |
| 8 | 1 483 | 35.3 % | 21.2 | 169.7 |
| 16, `lru` | 1 063 | 99.6 % | 5.5 | 87.4 |
| 16, **`lfu`** | 1 063 | 51.1 % | 10.5 | **168.2** |

### 1M single-session profile

Many long-lived agent sessions, each up to 1M tokens, few decoding at any instant.
There is no live host-KV tier for active sequences (every decode step reads the whole
KV; 3 GB/step over PCIe ≈ 100 ms/token — not viable). Instead:

- Growable KV (VMM segments, `--kv-grow-step-tokens`) funds KV from the expert cache
  as sessions grow.
- Session spill (`--session-spill-ram-gb`, `--session-spill-dir`) checkpoints idle
  sessions' KV + Mamba state to RAM then NVMe and restores exactly on the next turn.
- KV per 1M session ≈ 3 GB at fp8 + 47 MiB Mamba state → 2–3 concurrently decoding 1M
  sessions on the 5080; ~4 spilled sessions fit in the remaining host RAM
  (40 GB − 16.5 GB banks − process).

User decision (2026-09-04): the 1M profile runs **one** resident session
(`--max-running-requests 1`); all other sessions queue and are served in sequence via
spill/restore. The 16-way P2 profile remains for short-context Switchyard traffic.

Residency policy (task 3E, decided 2026-09-04):

- **No spill while the queue is empty.** The resident session's KV + Mamba state stays
  in VRAM until another session's request needs the slot; only then is it checkpointed
  (on demand, not on an idle timer). TTL-based release must not evict a resident
  session that nobody is waiting on. When an admission fails for lack of KV/state
  slots, the oldest idle reclaimable lease is reclaimed (re-run at the
  admission-failure point, not only at message receipt); the grace timer remains only
  as a very long safety (configurable, default off/∞).
- **Retention by capacity + age.** Checkpoint lifetime is decoupled from lease TTL:
  `--session-spill-limit-gb` (default 50) is the total across RAM+disk, and the
  oldest-by-last-use checkpoints are evicted first when the cap or the filesystem
  guard is hit instead of refusing. TTL closes the lease but keeps the checkpoint; a
  later request with the exact prefix restores it.
- **Survive restart.** Checkpoints are keyed on disk by session id + prompt-prefix
  hash + K/V layout fingerprint (manifest JSON next to chunks); startup scans the
  spill root, adopts valid records, and deletes stale/foreign ones; shutdown no longer
  rmtrees. Restore still requires an exact prefix + fingerprint match.

Measured sizing (task 2B4, 2026-09-04 — see
`benchmarks/results/nemotron35_lightning_5080_cache_study_2026-09-04.md`):

```
FREETOKEN_PIN_BUDGET_GB=17 \
ft serve --model ~/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --max-running-requests 1 --max-seq-len-override 1048576 --num-tokens 1048576 \
  --kv-grow-step-tokens 131072 --kv-cache-dtype q8_0 --attention-backend triton \
  --moe-backend offload --moe-cache-auto --linear-state-slots 5 \
  --memory-ratio 0.85 --max-prefill-length 8192 --host-ram-reserve-gb 6 \
  --session-spill-ram-gb 12 --session-spill-dir <nvme>
```

- **`--linear-state-slots 5` is the accepted floor** at `--max-running-requests 1`
  (`4·mr + 1` for `hybrid_radix`); 3 and 4 are rejected at startup. The default is 9, so
  pinning 5 returns ~188 MiB (≈ 35 expert slots) to the MoE cache.
- **Prefer 6 when two conversations alternate.** Five slots are padding + live + 2 ping-pong
  + *exactly one* idle session lease, so a second session's first turn finds the pool full
  and the scheduler spills the idle lease on demand to get its snapshot slot (correct — it is
  the 3E residency policy — but it costs a checkpoint + restore per alternating turn, and at
  1M that is GiB of KV). One extra slot (47 MiB) lets an idle lease and a live request
  coexist. Before 2026-09-04 this shortage was fatal rather than slow: the chunk commit's
  unguarded `pool.alloc(1)` raised `LinearStatePool exhausted` and killed the scheduler.
- **Expert slots vs KV growth**: each committed 131 072-token KV step costs ~0.40 GiB and
  ~76 expert slots. Auto starts at 1 786 slots and steps 1 663 (262K) → 1 586 (393K) →
  1 510 (524K) → 1 434 (655K); a full 1M session extrapolates to ~1 180 slots (rate 0.40).
  **VRAM is not the blocker for one 1M session.**
- **Throughput** on a growing synthetic-needle prompt: 131K prefill 3 007 tok/s / decode
  72.6 tok/s; 262K 1 790 / 51.8; 524K 997 / 32.0. Prefill cost is quadratic in context
  (526 s for a cold 524K prompt).
- **Coherence caveat**: the needle passes at 131K but is missed at 262K and 524K on the raw
  `/v1/completions` continuation. 262 144 is exactly the tokenizer's `model_max_length`.
  Treat **~131K–256K as the coherent ceiling** and re-verify the long end through
  `/v1/chat/completions` before advertising 1M (`bench_long_context.py` asks through the chat
  endpoint as of `ec54e21`; the 2B4 runs predate that).
- `--nvfp4-backend flashinfer` **cannot be combined with `--kv-grow-step-tokens`** (growable
  KV allocates the slot cache as VMM tensors; the b12x banks include an int32 bank
  `VMMTensor` does not support). The `triton` default is required here, and is the
  recommendation anyway.

Still outstanding for the gate: three sessions grown to ~1M each with disjoint needles, one
spilled and restored, all coherent, recording spill/restore times.
