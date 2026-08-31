# Supported models

FreeToken loads HF safetensors checkpoints directly (plus native GGUF for
Gemma-4, Qwen3.5-MoE/Ornith, and Laguna). The checkpoints below are known-good —
the prebuilt kernels are tuned for them; other checkpoints of the same architectures
work too.

| Model | HF checkpoints |
|---|---|
| DeepSeek-V4 | [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) |
| GLM-5.2 | [nvidia/GLM-5.2-NVFP4](https://huggingface.co/nvidia/GLM-5.2-NVFP4) |
| GLM-4.7 | [nvidia/GLM-4.7-NVFP4](https://huggingface.co/nvidia/GLM-4.7-NVFP4) |
| Qwen3.6 / Qwen3.5 MoE | [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) ([-FP8](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8)), [nvidia/Qwen3.6-35B-A3B-NVFP4](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4), [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) ([-FP8](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-FP8)) |
| Ornith 1.5 35B-A3B | [ornith-ai/Ornith-1.5-35B-A3B-GGUF](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF) (native Q4_K_M GGUF) |
| Qwen3.6 dense | [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) ([-FP8](https://huggingface.co/Qwen/Qwen3.6-27B-FP8)), [nvidia/Qwen3.6-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.6-27B-NVFP4) |
| Qwen3-MoE | [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) |
| gpt-oss | [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b), [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) |
| Gemma-4 | [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it), [nvidia/Gemma-4-26B-A4B-NVFP4](https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4), [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it), [nvidia/Gemma-4-31B-IT-NVFP4](https://huggingface.co/nvidia/Gemma-4-31B-IT-NVFP4) .. |
| Poolside Laguna-S 2.1 | compressed-tensors INT4 safetensors (including its BF16 expert tail), native GGUF |
| NVIDIA Nemotron 3 Super | [nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4) |
| MiniMax-M2.5 | [nvidia/MiniMax-M2.5-NVFP4](https://huggingface.co/nvidia/MiniMax-M2.5-NVFP4) |
| Muse-Glimmer | [meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B), [RedHatAI/Muse-Glimmer-30B-NVFP4](https://huggingface.co/RedHatAI/Muse-Glimmer-30B-NVFP4) |

## MoE backends

`ft serve --moe-backend {auto,fused,offload,cpu,hybrid}`:

- **fused** — experts resident on GPU (needs the VRAM); never auto-selected.
- **offload** — experts live in host RAM, an LRU cache of expert slots on GPU;
  misses stream over PCIe.
- **cpu** — misses are computed on the CPU instead of fetched.
- **hybrid** — per step, fetches some misses over PCIe and computes the rest on
  CPU, overlapped. Run `ft bench bw` once per machine to calibrate the split.
- **auto** — dense models always resolve to `fused`; MoE models resolve to
  `offload`, upgraded to `hybrid` when a cached `ft bench bw` profile
  recommends it.

## Notes

- `ft checkpoint` conversion is optional — it pre-converts a checkpoint into
  FreeToken's fast-load format, and `ft serve --model` auto-detects the result.
- DeepSeek-V4 checkpoints must keep the `inference/config.json` subdir — the
  authoritative model args are read from there.
- Multimodal checkpoints are served text-only.
- Laguna-S INT4 needs the `offload` backend. On WSL, FreeToken automatically
  keeps enough layers on CPU when the mixed INT4/BF16 banks exceed the CUDA
  pinned-memory budget.
- A single-session 200K Laguna configuration on a 16 GB GPU should reserve the
  minimum 256 expert slots, use INT4 KV, and keep the SWA pool near its working-set
  floor: `--max-running-requests 1 --max-seq-len-override 200000 --num-tokens 200000
  --kv-cache-dtype int4 --moe-cache-size 256 --disable-moe-prefill-overlap
  --swa-full-tokens-ratio 0.006 --memory-ratio 0.95`.
- For Ornith Q4_K_M at 200K on a 16 GB GPU, use one request, Q4_0 KV, 5,000
  expert slots, and the default 8K prefill chunks: `--max-running-requests 1
  --max-seq-len-override 200000 --num-tokens 200000 --kv-cache-dtype q4_0
  --moe-backend offload --moe-cache-size 5000 --max-prefill-length 8192
  --memory-ratio 0.95`. On the RTX 2000 Ada/WSL test host, cold 32K TTFT was
  51.1 s at 8K chunks versus 54.6 s at 16K; `--moe-prefill-hit-d2d` was slower
  on this stack and should remain disabled. Install the optional SGLang kernel
  (`freetoken[sgl]`) for faster expert-route alignment. FreeToken's Q4_0 path matches
  llama.cpp's block quantizer and is validated with normal answers, OpenAI tool calls,
  and a 55.6K-token Claude Code Bash-tool round trip. The sm_89 attention tuning reduces
  a synthetic 200K full-attention layer from 2.42 ms to 0.92 ms; the live 55.9K decode
  ran at 36--48 tok/s after warmup. A coherent 169.9K-token live generation completed
  in 425.8 s of prefill and decoded at 27--35 tok/s. The full 262,144-token pool also
  fits this 16 GB host with the command above after replacing both 200000 values with
  262144. Ada now uses llama.cpp's int8-MMA Q4_K/Q6_K kernels only for measured Ornith
  shapes and batch bands: dense 8192-output projections at 8--448/512 rows, the
  Q6_K 2048-output projection at 8--64 rows, and both top-8 routed projections at
  272--16384 tokens. Larger dense chunks return to transient dequant+cuBLAS, which is
  faster on this 70 W GPU. In a cold, identical 96,026-token server A/B, TTFT fell from
  176.91 s (`FREETOKEN_GGUF_DISABLE_MMA=1`) to 125.26 s (29.2% lower, 1.41x faster).
  `int4` remains an alias for `q4_0`. The independently quantized Q8-K/Q6-V and
  Q6-K/Q5-V tiers are also architecture-tuned on sm_89: the conservative eight-split
  fallback left long batch-one attention under-parallelized, while measured 32-split
  Q8/Q6 and 64-split Q6/Q5 launches reduce an isolated near-262K full-attention layer
  by 58.6% and 60.3%, respectively. Both passed live Q6_K 32K needle and growable-KV
  gates; see `benchmarks/results/ornith_ada_asymmetric_kv_2026-08-31.md`.
- On Blackwell (sm_120, e.g. RTX 5080 16 GB) the same command serves the **full
  262,144-token window**: `--attention-backend triton --max-seq-len-override 262144
  --num-tokens 262144 --kv-cache-dtype q4_0 --max-running-requests 1
  --moe-backend offload --moe-cache-auto --max-prefill-length 8192`. Pass the
  backend explicitly: sm_120 auto-resolves to FlashInfer, which cannot read the
  quantized KV pool. For one long Q4_K_M + Q4_0 KV session, add
  `--kv-grow-step-tokens 65536`; for Q6_K + Q8_0 KV use
  `--kv-grow-step-tokens 131072`. The Q8-specific 128K step avoids the expensive
  intermediate expert-cache rebuild seen with 64K steps on this host. Both reserve
  the full virtual KV address range while physically committing fixed-size segments.
  At each boundary FreeToken gives exactly enough expert-cache
  VRAM to the new KV segment, preserves all earlier KV at stable addresses, and
  recaptures decode graphs once after the final prefill chunk. Multiple requests
  use independent page-table rows over one shared physical arena; growth follows
  their aggregate live-page demand. Growable quantized GGUF MoE serving runs one prompt lane
  per prefill forward and rotates unfinished long prompts between 8K chunks; on the
  RTX 5080, grouping independent prompt lanes made expert prefill substantially
  slower. Pass `--max-prefill-sequences 0` to restore grouped prefill. Decode remains
  continuously batched across every runnable agent. A 32-step decode burst between helper-prefill
  chunks prevents an established request from being starved. When a request stops,
  unlocked finished prefixes are evicted, surviving request-owned tail pages are
  compacted into low holes, complete VMM segments are decommitted, and the released
  VRAM expands the MoE cache again. A protected high prefix is never moved, so it
  conservatively delays shrink rather than risking stale radix references. This mode
  requires the Triton MHA path, plain `offload`, and `--moe-cache-auto`; it is opt-in
  because changing expert geometry trades early-context speed against final-context
  decode residency. Runtime logs show both instant and cumulative-average prefill speed.
  The attention launch tables are
  architecture-aware: the sm_120 Q4_0 decode launch (64 splits, 64-token tiles)
  runs a synthetic 262K full-attention layer in 0.36 ms versus 0.82 ms with the
  sm_89 tuning, and the extend/prefill kernels drop to 4 warps (1.12x on long-Q4
  prefix extension, 2x on cold chunks). BLOCK_N=16 silently corrupts the packed
  Q4 loader on sm_120 exactly as on sm_89 and stays excluded. On sm_120 the
  Q4_K/Q6_K GGUF matmuls (dense prefill and large routed-expert batches) run on
  llama.cpp's int8-tensor-core MMQ (vendored under `kernel/csrc/gguf_mmq/`,
  JIT-built on first use): ~13x over the DP4A kernels and ~1.3x over transient
  dequant+cuBLAS at 8K-token chunks, with the same lossless packed weights. A
  second sm_120 pass tunes the fused routed+shared decode MMVQ to four output-row
  warps (about +2.2% live decode for both Q4 and Q6), moves the grouped-MMA prefill
  crossover from 320 to 272 tokens, and sends only measured small/output Q6_K
  projections back to transient dequant+cuBLAS (+18.0% on an 8K cold-prefill live
  A/B). Q4_K retains uncapped MMA: the analogous isolated-kernel change reduced
  end-to-end overlapped prefill and was rejected. At the exact 262,144-token gate,
  the Q6_K + Q8_0 128K-step configuration reached 527.36 tok/s average prefill,
  416.57 tok/s on the last full chunk, and 78.40 tok/s decode while recovering the
  deterministic needle coherently. That is +62.1% prefill and +32.0% decode over
  the same growable configuration with 64K steps. A repeated-growth gate then ran
  a 524,000-token prompt under a 524,288-token ceiling with YaRN factor 2. All
  three 128K transitions completed (expert slots 4,096 → 3,544 → 2,991 →
  2,438), and the coherent exact-needle result reached 402.10 tok/s average
  prefill, 268.71 tok/s on the last full chunk, and 62.28 tok/s decode. Two-agent
  teardown gates used disjoint 70K prompts and passcodes. Q6/Q8 released 1.33 GiB
  at helper exit (262K → 128K, 3,483 → 4,036 expert slots) and recovered to about
  130 tok/s steady decode; Q4/INT4 released 0.35 GiB (192K → 128K, 5,644 → 5,842
  slots) and reached 154.04 tok/s over the final 80-token tail. Both surviving
  agents remained coherent and neither emitted the other agent's passcode. Claude
  Code and Codex receive persistent KV leases automatically: FreeToken binds
  Claude's `X-Claude-Code-Session-Id` (plus its child-agent id) and Codex's stable
  `prompt_cache_key`/thread headers to separate internal session ids. No client
  patch or custom request field is required. Other clients can opt in by sending
  the same `session_id` (and optional `session_ttl_seconds`, default 300) on each
  full-conversation request to `/v1/chat/completions`, `/v1/completions`,
  `/v1/messages`, or `/v1/responses`. The resolved id is returned as
  `X-FreeToken-Session-Id` for observability and explicit teardown.
  A normal EOS/stop/max-token turn ends only the turn: its reusable prefix and final
  GDN snapshot stay protected for the next request. Sessions serialize their turns;
  concurrent use of one id returns `session ... is busy`. `DELETE
  /v1/sessions/{session_id}` is a scheduler barrier and returns `closed` only after
  the lease is released (or `not_found`); a stream disconnect/abort also closes it,
  and an inactive lease expires after its TTL. Because neither agent protocol sends
  a normal process-exit event, a cleanly idle auto-bound session is reclaimed at that
  timeout; cancellation and disconnect reclaim it immediately. Both clients still
  send the complete conversation on each turn—the lease retains computation state,
  not message history. OpenAI `previous_response_id` storage is not emulated.
  In single-rank growable Hybrid-GDN mode, an idle automatically bound session is
  checkpointed before its soft GPU lease is released. FreeToken keeps checkpoints in
  RAM up to `--session-spill-ram-gb` only while `MemAvailable` remains above
  `--host-ram-reserve-gb`; overflow streams to the bounded `--session-spill-dir` disk
  tier. A later request restores only when the client-resubmitted token prefix and the
  exact K/V layout fingerprint match. Mismatch, damage, or capacity pressure discards
  the checkpoint and performs ordinary prefill, never approximate reuse. Explicit
  client-named sessions remain hard GPU leases until close/TTL. Disk restore keeps
  exactly one bounded chunk in look-ahead, overlapping NVMe/deserialization of the
  next layer with installation of the current layer. Set
  `--session-spill-dir off` to disable the cold tier.
  Closing a helper makes its pages evictable immediately, allowing the growable KV
  arena to decommit unused suffix segments and restore MoE residency.
  Hybrid-GDN serving can also reserve only the normal four-agent recurrent-state and
  graph footprint while admitting an eight-agent burst with
  `--max-running-requests 8 --elastic-initial-requests 4`. Demand above four compacts
  and preserves live/session GDN states, trades MoE residency for 8-way state and
  graphs, then reverses the trade as soon as demand returns to four. An RTX 5080 Q4
  gate preserved all eight independent answers across 25 → 49 → 25 physical GDN
  slots and restored the exact original 5,635-slot MoE cache after the burst.
  For Q6 hosts constrained by WSL's CUDA pin quota, pageable routed misses now use
  CUDA-graph host callbacks and mapped pinned zero-copy scatter while shared-expert
  GPU work runs concurrently. Persistent placement is opt-in because an arbitrary
  production workload can train a model-wide regression: `--moe-pageable-profile read`
  applies an existing model-file-scoped time-cost ranking, while `train` also enables
  decode telemetry and updates it at idle boundaries. The default `off` keeps the
  validated deterministic placement and avoids unsafe live host re-registration.
  The production host-memory guard is 3 GiB. Growable mode now places its large
  resizeable expert-cache banks in direct CUDA VMM allocations rather than the
  PyTorch caching allocator, keeps a 256 MiB physical-commit cushion, and refuses a
  growth step before `cuMemSetAccess` if live free VRAM is insufficient. Exact
  1,048,576-token capacity gates (1,048,448 prompt + 128 output allowance, YaRN 4)
  passed coherently: Q4_K_M + Q4_0 KV averaged 331.82 tok/s prefill and 46.03 tok/s
  decode, ending with 5.70 GiB physical KV and 2,952 expert slots; Q6_K + Q8_0 KV
  averaged 207.73 tok/s prefill and 15.21 tok/s decode, ending with 10.70 GiB KV and
  258 expert slots. The latter is a single-request capacity profile and requires
  `--linear-state-slots 5`, `--memory-ratio 0.99`, `--moe-pageable-gpu`, and a
  sufficiently large pin budget; the default nine linear-state slots are correctly
  rejected by the guarded 1M preflight. See
  `benchmarks/results/ornith_5080_scheduler_safety_1m_2026-08-30.md` for the full
  safety, session, scheduler, and measurement notes.
  Q6 deployments may instead opt into independent `--kv-cache-dtype-k q8_0
  --kv-cache-dtype-v q6_0`. This keeps Q8 keys and packs only values to Q6, reducing
  the measured 64K physical ceiling from 0.74 to 0.66 GiB and exposing 33 more expert
  slots. An exact 65,408-prompt + 128-output gate crossed a 32K growth boundary,
  averaged 515.91 tok/s prefill (693.56 tok/s final chunk), decoded at 47.06 tok/s,
  and recovered the needle coherently. Isolated Q8/Q8 attention remains faster, so
  this is a long-context capacity/residency trade rather than the new universal
  default. See `benchmarks/results/ornith_5080_asymmetric_kv_2026-08-30.md`.
  A more compact opt-in lane is `--kv-cache-dtype-k q6_0
  --kv-cache-dtype-v q5_0`. It cuts logical KV another 20% versus Q8/Q6 and crosses
  over to faster isolated decode attention around 131K on RTX 5080, but exact-64K
  live decode and final-chunk prefill were about 7% slower. It therefore remains a
  long-context/capacity tier rather than the Q6 default. See
  `benchmarks/results/ornith_5080_q6_q5_kv_2026-08-30.md`.
- Nemotron 3 Super uses its native hybrid Mamba-2 / full-attention / latent-MoE
  architecture. The NVFP4 release needs about 60 GiB of host RAM for expert banks and
  10.3 GiB of resident GPU weights. FreeToken currently serves one concurrent Nemotron
  session. On WSL, `--moe-pageable-gpu` keeps the pin-budget overflow banks pageable,
  stages only their routed misses through a small pinned buffer, and still executes every
  ReLU² expert on GPU. Decode gathers are CUDA-graph host nodes and overlap the shared
  expert calculation; idle telemetry saves a model-scoped time-cost ranking that is
  applied on the next clean start. A minimal
  all-GPU-compute launch is:
  `ft serve --model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
  --max-running-requests 1 --moe-backend offload --moe-cpu-layers 0
  --moe-pageable-gpu --moe-cache-auto`.
