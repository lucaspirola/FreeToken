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
  `--host-ram-reserve-gb 3` and the ~4 GiB non-bank process footprint, plan on
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
- **Expert GEMM**: ungated ReLU² experts are served by the Triton NVFP4 kernels.
  `--nvfp4-backend marlin` is rejected at config time (its fused kernel assumes a
  gated `[2I, H]` bank).

### Launch profiles

P1 — bring-up profile (single stream, no quantized KV):

```
ft serve --model ~/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --max-running-requests 1 --moe-backend offload --moe-pageable-gpu --moe-cache-auto \
  --num-tokens 65536 --memory-ratio 0.90 --max-prefill-length 4096 --host-ram-reserve-gb 3
```

P2 — serving profile (16 concurrent, elastic KV, prefix cache, quantized KV):

```
ft serve --model ~/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --max-running-requests 16 --elastic-initial-requests 4 --kv-grow-step-tokens 65536 \
  --num-tokens 262144 --max-seq-len-override 131072 --kv-cache-dtype q8_0 \
  --attention-backend triton --moe-backend offload --moe-pageable-gpu --moe-cache-auto \
  --memory-ratio 0.90 --max-prefill-length 4096 --host-ram-reserve-gb 3 --enable-cache-report
```

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

Default `--max-prefill-length 4096` until the SSD kernels are wired in, then
re-measure.

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

The gate (after the Phase 2 kernels, before Phase 4) runs the 1M profile
`--max-seq-len-override 1048576 --num-tokens 1048576 --kv-cache-dtype fp8_e4m3
--attention-backend triton --kv-grow-step-tokens 131072 --max-running-requests 1
--linear-state-slots <minimum accepted> --session-spill-ram-gb 12 --session-spill-dir
<nvme>`; three sessions grown to ~1M each with disjoint needles, one spilled and
restored, all coherent, recording prefill/decode tok/s and spill/restore times.
