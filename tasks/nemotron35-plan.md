# Nemotron 3.5 Lightning 30B-A3B-NVFP4 on FreeToken (RTX 5080, Switchyard)

## Context

Goal: serve `~/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` from FreeToken on this
host (RTX 5080 16 GB sm_120, WSL2, CUDA 13 / Torch 2.11 / Triton 3.6, 29 GB host RAM, 16 cores)
as an upstream `openai_chat` target for NVIDIA Switchyard (local clone `~/ai/Switchyard`).

FreeToken already has a `nemotron_h` package (`python/freetoken/models/nemotron_h/`) built for
Nemotron-3-Super-120B (latent MoE). Lightning is the same `NemotronHForCausalLM` family with a
different geometry, so this is a delta job plus perf work plus serving compliance, not a port.

User decisions (2026-09-03):
- Concurrency target: up to 16 running requests, elastic KV capacity, prefix caching.
- Port Triton Mamba-2 SSD kernels (replace HF pure-PyTorch reference).
- MTP speculative decoding: included as a final, time-boxed phase.
- Orchestration: lead session steers; Opus subagents implement, parallel where files don't overlap.

## Verified facts

Model (from config/index/quant config):
- 52 layers: 23 mamba, 23 moe, 6 attention (idx 5,12,19,26,33,42). hidden 2688, vocab 131072, untied lm_head.
- MoE: 128 routed experts, top-6, ReLU² ungated (up+down only, I=1856), shared expert I=3712,
  sigmoid router + `e_score_correction_bias`, `norm_topk_prob`, `routed_scaling_factor` 2.5,
  `n_group=topk_group=1`, **`moe_latent_size=null`** (Super had latent MoE).
- Mamba-2: 64 heads × 64 head_dim (inner 4096), n_groups 8, d_state 128, conv 4, chunk 128;
  in_proj out = 10304 (z 4096 + xBC 6144 + dt 64). State ≈ 48 MiB/seq fp32.
- Attention: 32 heads / 2 KV heads / head_dim 128, NoPE (rope keys are vestigial). KV ≈ 3 KiB/token fp8.
- `max_position_embeddings` 1,048,576; tokenizer `model_max_length` 262,144; eos ids [2, 11].
- Quant (modelopt MIXED_PRECISION): experts + shared experts + lm_head W4A16 NVFP4 group 16;
  mamba in/out_proj FP8 per-tensor (+input_scale); attention qkvo, router (fp32), conv1d,
  embeddings, `mtp.*` BF16; `k_scale/v_scale` present (FP8 KV calibration, unused).
- Bytes: routed experts 15.39 GiB; non-expert 2.20 GiB (+2.49 GiB BF16 MTP head). Total 20.08 GiB.
- Chat template: ChatML; `enable_thinking` (default on) → generation prompt ends `<think>\n`,
  off → `<think></think>`; tools as XML, tool calls Qwen3-Coder nested XML; `<think>`/`<tool_call>`
  tokens are non-special. Sampling: temperature 1.0, top_p 0.95.

FreeToken gaps (verified):
- F1 `nemotron_h/config.py:103,135` `int(moe_latent_size)` crashes on None.
- F2 `nemotron_h/config.py:56-62` transformers 5.8 renames layer types to `linear_attention` /
  `full_attention`; `full_attention` is unmapped → `NemotronHBlock` raises (`model.py:310`).
- S1 `config.py:128` `attn_quant` derived from "any FP8 module" (only Mamba is FP8 here).
- S2 `model.py:269-280` always builds `fc1/fc2_latent_proj`; need identity path and
  `make_moe_layer(hidden_size=hidden_size)`.
- S3 shared experts + lm_head dequantized to BF16 (~1.6 GB VRAM); `Nvfp4DenseLinear` /
  `Nvfp4LMHead` exist in `kernel/triton/nvfp4_linear.py`.
- S4 `config.py:140` `single_stream_only=True` → `engine.py:2075-2084` forces bs=1.
- `engine.py:809` uses `hidden_size` where `expert_hidden_size` is meant.
- Marlin/b12x NVFP4 MoE backends assert silu (`moe/nvfp4_backends.py:481,737`); Nemotron is
  locked to the Triton W4A16 backend.
- Mamba-2 scan/decode are HF pure-PyTorch (`model.py:104-111,151-168`) with a per-request
  Python loop in `_prefill_scan`. Conv1d is kernelized. `kernel/fla/` is GDN, not SSD.
- No speculative decoding infra anywhere; `weight.py:54` drops `mtp.*`.
- No `kernel/aot_models.py` entry (JIT fallback). Tests: `tests/models/test_nemotron_h.py`
  uses a SimpleNamespace fixture with legacy layer names (cannot catch F2).

Switchyard contract (from `crates/switchyard-translation/src/codecs/openai_chat/*`, docs):
- FreeToken is an upstream `format = "openai_chat"` target; only `/v1/chat/completions` matters.
- Sends `max_completion_tokens` (never `max_tokens`), tools/tool_choice, temperature/top_p,
  stream, reasoning_effort, response_format verbatim, optional parallel_tool_calls,
  prompt_cache_key, stream_options, top_logprobs, stop. Never sends n/seed/logprobs.
- Reads `reasoning_content` or `reasoning`; SSE `[DONE]`; usage incl.
  `prompt_tokens_details.cached_tokens`, `completion_tokens_details.reasoning_tokens`.
- Context overflow must be HTTP 400 + `error.code == "context_length_exceeded"` (also as first
  SSE event) or route fallthrough breaks.
- Judge/classifier targets need schema-valid JSON in `content` (json_schema or json_object).
- Affinity headers: `x-switchyard-session-id`, `x-claude-code-session-id`, `x-codex-session-id`.
- Traffic: 16 serial workers, 8K–32K prompts, 512–8K shared prefixes, growing conversations,
  tool-call bursts re-sending history, 16–64 tool catalogs.

Host facts that shape the plan:
- RTX 5080 16 GB; a `llama-server` (Qwen3-Embedding-4B) currently holds ~10 GB VRAM. Must be stopped before any GPU run.
- 29 GB RAM, swap 8 GB fully used, MemAvailable ~20 GB. Expert banks 15.4 GiB + process ≈ 26.6 of 29 GB.
- WSL CUDA pin quota is 0.4×RAM ≈ 11.6 GiB (`engine.py:1822 _pin_budget_bytes`), below the 15.4 GiB
  banks → `--moe-pageable-gpu` needed (which disables decode CUDA graphs, `offload_cache.py:174`),
  or raise the quota via `FREETOKEN_PIN_BUDGET_GB` / `.wslconfig`. Decide by measurement in Phase 1.
- CPU: Core Ultra 7 265K, 16 cores, **no AVX-512**; `_cpu_moe_act_ok` (engine.py:2166) excludes relu2,
  so `cpu`/`hybrid` MoE backends are unavailable for this model today. `offload` only.
- Installed: flashinfer 0.6.17 (`flashinfer.mamba.selective_state_update` JIT builds for sm_120 with
  `compute_120f`; `SSDCombined` prefill rejects sm_120; b12x W4A16 MoE `SUPPORTED_MOE_ACTIVATIONS =
  {"silu","relu2"}` — the silu-only lock is FreeToken's in `moe/nvfp4_backends.py:214-226,744`).
  sgl_kernel has conv1d only. No mamba_ssm / vLLM → SSD prefill kernels must be vendored.

## Orchestration model

Lead session steers; Opus subagents implement (`Agent`, model `opus`, `isolation: worktree` when
touching overlapping files). Every task: implement → tests → run tests → report. Phases run in
order; waves inside a phase run in parallel only where file sets are disjoint. After each phase:
full focused test run, ruff, `git diff --check`, commit. Mirror this plan into `tasks/todo.md`
with checkboxes at implementation start, update `tasks/lessons.md` on corrections.

Host preparation before any GPU run (C0): user authorized clearing the GPU completely — kill the
llama-server (pid 3673410, Qwen3-Embedding-4B) and anything else holding VRAM; kill stray FreeToken workers by
venv path (`pgrep -f /home/lucas/ai/FreeToken/.venv/bin/python`), never `pkill -f "ft serve"`;
remove stale `~/.cache/torch_extensions/*/lock`; `free -g` must show ≥ 19 GiB available.

---

## Phase 1 — Bring-up: Lightning loads, multi-request, correct output

### Wave 1 (parallel, disjoint files)

**1A. Model package** — `python/freetoken/models/nemotron_h/{config.py,model.py,weight.py}`,
`tests/models/test_nemotron_h.py`
- `config.py`: map `{"mamba","linear_attention"}→mamba`, `{"attention","full_attention"}→attention`,
  `moe→moe`, raise on `mlp`. `moe_latent_size: int|None`; `expert_hidden_size = latent or hidden_size`.
  Split `_quantized_modules` into fp8 / nvfp4-dense / lm_head; `module_quant()` returns `"nvfp4"`
  for dense NVFP4 (env `FREETOKEN_NEMOTRON_DENSE_DEQUANT=1` restores the old dequant path).
  `attn_quant` = fp8 only if q/k/v/o modules are fp8. Set `dense_quant`/`lm_head_quant="nvfp4"`.
  Read `time_step_min`, `mlp_hidden_act` (assert relu2), `mamba_ssm_cache_dtype` (warn if not
  fp32), assert `n_group==topk_group==1`, `n_shared_experts==1`. `single_stream_only` becomes
  `state_bytes_per_slot > 96 MiB` (Lightning 47 MiB → False; Super ~160 MiB → True unchanged);
  env `FREETOKEN_NEMOTRON_MULTI_STREAM=1` overrides.
- `model.py`: `_linear()` gains `"nvfp4"` → `Nvfp4DenseLinear` (`kernel/triton/nvfp4_linear.py:847`);
  `NemotronHMoE` skips `fc1/fc2_latent_proj` when latent is None and passes
  `hidden_size=config.expert_hidden_size`, `activation=config.hidden_act`; lm_head → `Nvfp4LMHead`
  when `lm_head_quant=="nvfp4"` (pattern `models/qwen3_5_moe/model.py:94-104`); pass
  `dt_limit=(time_step_min, inf)` to both scan calls (HF prefill parity).
- `weight.py`: for nvfp4 dense modules yield `weight` (u8), `weight_scale` (fp8),
  `weight_global` (fp16, `weight_scale_2` expanded per row; mirror `qwen3_5_moe/weight.py:369
  _nvfp4_parts`) instead of dequantizing. Keep `mtp.*` and `k_scale/v_scale` skips.
- Tests: fixtures built through real `transformers.NemotronHConfig` (Super-like and Lightning-like
  7-layer slice) so the layer-type remap is exercised; assert attention group ids, `expert_hidden_size
  == 2688`, no latent-proj keys, quant axes, state-dict dtypes/shapes for shared expert and lm_head,
  `single_stream_only` False/True, ungated bank byte estimate, relu2 assert. `needs_weights` GPU test:
  real shared-expert + lm_head tensors through `Nvfp4DenseLinear`/`Nvfp4LMHead` vs dequant reference.

**1B. Engine / CLI / AOT / docs** — `engine/engine.py`, `server/args.py`, `kernel/aot_models.py`,
`docs/models.md`, `tests/engine/`, `tests/server/`, new `benchmarks/preflight_nemotron_host.py`
- `engine.py:809` → `expert_hidden_size or hidden_size`. Config-time `ValueError` for
  `--nvfp4-backend marlin` with ungated experts (b12x becomes allowed in Phase 2).
  Test: `single_stream_only=False` keeps `max_running_req=16` and elastic graph sizes.
- `server/args.py:160,203`: add `nemotron_h` / `nemotron-3.5` markers; test auto-selects
  `qwen3_coder` + reasoning parser for the Lightning path.
- `kernel/aot_models.py`: add Lightning entry (hidden 2688, kv (2,128), top_k 6, I 1856, nvfp4,
  new `expert_gated=False` field so `expert_bank_row_bytes` uses I not 2I).
- Preflight script (torch-free): MemAvailable/SwapFree/pin budget/pageable layer estimate/VRAM
  holders/stale locks; non-zero exit on violations.
- `docs/models.md`: Lightning row + rewritten Nemotron note (host RAM ~16.5 GB, 1M window, 16 req).

**1C. Gates (new files only)** — `benchmarks/parity_nemotron_h_layers.py`,
`benchmarks/results/nemotron35_lightning_5080_<date>.md`
- Per-layer parity vs HF modules with dequantized weights (layers 0 mamba, 1 moe, 5 attention;
  T=512): expert ids ≥ 99.5% identical, cosine > 0.999.
- After 1A lands: P1 smoke → parity → P2 → batch invariance (8 prompts alone vs 16 concurrent,
  greedy, identical tokens), prefix-cache equality + radix hit counters, elastic ramp 1→6→16→1
  without OOM, needle via `benchmarks/bench_long_context.py --synthetic-needle` at 64K/128K/256K
  with `q8_0`, `fp8_e4m3`, `auto` KV; tool-call round trip; `tests/e2e/test_aime.py` subset.

### Launch profiles
P1 (bring-up, single stream):
```
ft serve --model ~/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --max-running-requests 1 --moe-backend offload --moe-pageable-gpu --moe-cache-auto \
  --num-tokens 65536 --memory-ratio 0.90 --max-prefill-length 4096 --host-ram-reserve-gb 3
```
P2 (target: 16 concurrent, elastic, prefix cache, quantized KV):
```
ft serve --model ~/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --max-running-requests 16 --elastic-initial-requests 4 --kv-grow-step-tokens 65536 \
  --num-tokens 262144 --max-seq-len-override 131072 --kv-cache-dtype q8_0 \
  --attention-backend triton --moe-backend offload --moe-pageable-gpu --moe-cache-auto \
  --memory-ratio 0.90 --max-prefill-length 8192 --host-ram-reserve-gb 3 --enable-cache-report
```
User decision (2026-09-03): KV cache is FP8 (`fp8_e4m3`, FreeToken block scales; checkpoint k_scale/v_scale ignored). Quantized KV requires `--attention-backend triton`; bf16 KV + FlashInfer is the fallback (KV is
only +0.75 GiB at 262K). Try `FREETOKEN_PIN_BUDGET_GB` ≥ 17 to drop `--moe-pageable-gpu` and
regain decode graphs; measure both.

GPU budget at P2 (MiB): weights 2,260; Mamba slots 47 each (25 slots at 4 req = 1,175; 97 at
16 = 4,560; consider `FREETOKEN_MAMBA_SSM_DTYPE=bfloat16` in Phase 2); KV q8_0 ≤ 816; MoE slot
cache ~6.6–9.9 GB (1,230–1,850 slots of 5.36 MiB).

### Phase 1 risks
Super numerics shift slightly (dt clamp, native NVFP4 dense) and cannot be retested here;
bs>1 CUDA-graph capture of the pure-torch decode scan is unproven (fallback
`--cuda-graph-max-bs 1`); pure-torch chunk scan memory at 8K chunks (lower to 4096 if needed);
host OOM if anything else is resident.

---

## Phase 2 — Kernels: Mamba-2 SSD + sm_120 MoE fast path

Contracts (owned by 2A1, consumed by all):
- `LinearGatedDeltaGroupConfig.state_layout: "kv"|"mamba2"` (`models/config.py:151`) with
  `track_chunk_size` 64/128. Nemotron sets `mamba2`: recurrent pool becomes `[L, slots, 64, 64, 128]`
  = `[H, P, N]` (flashinfer / SSD native layout, no transposes); `_linear_local_dims` branch gives
  conv_dim 6144. `attention/linear.py::_build_track_metadata` and `hybrid_radix_cache.py:57` use
  `group.track_chunk_size` instead of the FLA constant, so radix snapshots land on ×128 boundaries.
- `FLAMetadata.mamba2: Mamba2Metadata|None` (chunk_size, cu_chunk_seqlens, last_chunk_indices,
  seq_idx, chunk_offsets, num_chunks), built in `build_fla_metadata`.
- Kernel API in `kernel/triton/mamba2/__init__.py`: `mamba2_prefill(x, dt, B, C, *, A, D, dt_bias,
  meta, cu_seqlens, state_source, indices, return_intermediate_states)` and `mamba2_decode(x, dt, B,
  C, *, A, D, dt_bias, state_source, indices, out)` (graph-safe, fixed shapes, no host sync).
  Kernels write fresh outputs; pool scatter in the wrapper (autotune-safe).
- Gated RMSNorm: reuse `kernel/fla/layernorm_gated.py::rms_norm_gated(group_size=512,
  norm_before_gate=False, activation="silu")`.
- Old pure-torch path moves to `models/nemotron_h/mamba2_reference.py` (`FREETOKEN_MAMBA2_REF=1` A/B).

**T0. Probe** (30 min): build `flashinfer.mamba.selective_state_update` on sm_120 (dim 64, dstate
128, H 64, G 8, bs 1..16, `state_batch_indices`, `dt_softplus`); check vs HF reference, time it,
record JIT build time. Failure → Triton fallback only.

### Wave 1 (parallel)
- **2A1** layout/metadata/model wiring + reference module; tests in `tests/kvcache/`,
  `tests/kvcache/radix/` (parametrize `track_chunk_size ∈ {64,128}`), `tests/models/`.
- **2A2** Vendor vLLM `ops/ssd_*.py` (sequence-aligned chunk variant, Apache-2, attribution header)
  into `kernel/triton/mamba2/`. sm_120 adaptations: prune autotune configs to ~100 KB smem, ≤8
  configs, `autotune_cache_kwargs`, never specialize on seqlen; fp32 cumsum/states, bf16 IO.
  Tests `tests/kernels/test_mamba2_ssd.py` vs HF fp32 reference: T ∈ {1,5,127,128,129,300,1024,4097},
  varlen batch with mixed initial states, intermediate-state indexing, chunk continuation,
  prefill+1 decode == prefill T+1. Bench `benchmarks/bench_mamba2_ssd.py`: 8K ≤ 1.5 ms/layer,
  32K ≤ 7 ms/layer, ≥ 20× torch reference, < 700 MB transient at 32K.
- **2A3** Decode: flashinfer SSU wrapper (stride-0 expanded A/D/dt_bias cached per layer,
  `pad_slot_id=-1`, preallocated out) + Triton port of vLLM `_selective_scan_update_kernel` as
  fallback; `FREETOKEN_MAMBA2_DECODE=auto|flashinfer|triton`; explicit `warm_mamba2_decode()` before
  graph capture. Gated norm swap. Tests: bs {1,7,16}, pad rows untouched, CUDA-graph replay equals
  eager. Target bs=1 ≤ 15 µs/layer, bs=16 ≤ 80 µs/layer.
- **2B1** Enable b12x for relu2: `moe/nvfp4_backends.py` accept `activation in {silu,relu2}`,
  ungated repack (skip gate/up half-swap, `gu_packed.size(1)==I`), pass `activation=` to
  `prepare_w4a16_packed_weights` / `_launch_sm120_w4a16_moe`; force `triton` when decode target is
  not GPU. Tests in `tests/moe/test_nvfp4_backends.py` (ungated decode/prefill vs dequant ref,
  selection matrix). Bench `benchmarks/bench_nvfp4_moe_kernels.py` (H 2688, I 1856, E 128, top-6;
  M 1..16 decode, 256..8192 prefill): b12x ≥ 70% HBM roofline at M=1, ≥ 2× Triton at M 8/16. Then
  make `auto` resolve to b12x for ungated sm_120.
- **2B3** sm_120 tuning of `kernel/triton/nvfp4_linear.py` for (3712×2688), (2688×3712),
  (131072×2688) at M 1..8192; optional ReLU² epilogue for shared expert. Bench
  `benchmarks/bench_nvfp4_dense.py`: lm_head M=1 ≤ 300 µs.

### Wave 2
- **2A4** Integrate, delete hot-loop reference path, run kernel/kvcache/scheduler/model tests, AIME
  e2e with `FREETOKEN_MAMBA2_REF=1` A/B, prefix-cache regression on ×128 snapshots.
- **2B2** Triton NVFP4 fallback tuning: ReLU² fused into gemm1 epilogue (`ACT` constexpr), decode
  config table keyed by (N,K,top_k,sm), prefill JSON config
  `moe/configs/triton_3_6_0/nvfp4,E=128,N=1856,K=2688,device_name=NVIDIA_GeForce_RTX_5080.json`
  via `benchmarks/tune_nvfp4_moe.py --write`.
- **2B4** Cache sizing study: `ft bench bw` with real geometry (ungated support in benchbw),
  `bench_decode_moe.py` at bs 1 and 16 with per-layer miss stats, sweep cache rate/policy,
  bf16 SSM state option; PCIe ceiling estimate ~85 ms/step at bs=16 with 50% hit rate. Record
  recommended launch line in `docs/models.md`.

### Phase 2 risks
Triton 3.6 smem overflows on GB203; in-place-state kernels must be single-config; ×128 snapshot
granularity change touches hybrid radix tests; bf16 IO drift vs fp32 reference (use relative
tolerances); flashinfer JIT minutes on first run must precede capture; b12x banks incompatible
with CPU paths; per-expert `down_global` must be row-constant (verify on checkpoint early).

---

## Phase 3 — Switchyard compliance + serving profile

Audit result (code read): supported today — `max_completion_tokens` alias
(`api_models.py:98-102`), tools/tool_choice, temperature/top_p, SSE + `[DONE]`,
`reasoning_content` deltas, `cached_tokens` (needs `--enable-cache-report`), non-stream 400
`context_length_exceeded`, generation_config sampling defaults, eos {2,11}, qwen3_coder parser.
Missing/broken — streaming overflow error is the **second** SSE event (role chunk first,
`openai_api.py:249-255`); `response_format` hard-rejected (`:159-163,653-655`); no
`reasoning_tokens`; no session binding on chat completions (`client_sessions.py` only covers
Anthropic/Responses); `force_nonempty_content` absent; reasoning parser can fold `<tool_call>`
into reasoning when the model skips `</think>`; `max_seq_len` defaults to 1M from config.

### Wave 1 (parallel; merge order A → C → B → D)
- **3A wire/errors** — `openai_api.py`, `generation.py`, `scheduler.py:857`:
  `prerender_error` → `preflight_error` tokenizes and returns 400 `context_length_exceeded` with a
  message containing both "maximum context length" and "prompt is too long", for stream and
  non-stream; stream generator pulls the first event before emitting the role chunk so an error is
  the first SSE event; `completion_tokens_details.reasoning_tokens` counted from acks while the
  reasoning parser is in reasoning state (`GenDone.reasoning_tokens`); `developer` role → system
  for non-Harmony templates; typed `prompt_cache_key`, `user`, `top_logprobs` (400 only if > 0),
  `Function.strict`. Tests in `tests/server/test_openai_api.py` (+ overflow-first-event, alias,
  10-optional-field acceptance, 64-tool catalog).
- **3B JSON mode** — `api_models.py` `ResponseFormat`, `GenSpec.json_mode/json_schema`, new
  `server/json_output.py`: default `enable_thinking=False` for JSON calls unless set; inject
  schema instruction into the system block; buffer content, strip think/fences, extract first
  balanced object, validate (small built-in validator: type/required/enum/properties); one retry
  at temperature 0 with the error appended; on final failure return raw content (Switchyard falls
  back to strong target). Stream: single content delta before finish. Tests
  `tests/server/test_openai_json_mode.py`. No grammar dependency (none in uv.lock; sampler has no
  mask hook under CUDA graphs).
- **3C sessions + parsers** — `client_sessions.py::chat_session_id` (precedence: explicit
  `session_id` → `x-switchyard-session-id` → `x-claude-code-session-id` → `x-codex-session-id` →
  `prompt_cache_key` → `session-id`; agent split via `x-switchyard-agent-id` /
  `x-claude-code-agent-id`), reclaimable auto-bound ids, `X-FreeToken-Session-Id` on stream and
  JSON responses, one lease-less resubmit on "session is busy". `NemotronV3ReasoningParser`
  (`ThinkReasoningParser` + `tool_start_token="<tool_call>"`), registered as `nemotron_v3`,
  auto-selected for nemotron markers; `force_nonempty_content` (pop from `chat_template_kwargs`,
  default true when thinking off or `--force-nonempty-content`): empty content + reasoning → swap,
  streaming emits one trailing content delta. Tests `tests/server/test_nemotron_v3_parsers.py`,
  `test_client_sessions.py`. Docs: header list in `docs/models.md`.
- **3D profile + e2e** — served max len = `min(max_position_embeddings, tokenizer
  model_max_length)` unless overridden (`engine/config.py:202` or nemotron config); `docs/switchyard.md`
  with `routes.toml` (targets `lightning` thinking-on + `lightning_fast` thinking-off with
  `force_nonempty_content`, `passthrough` route, `stage_router` route with classifier on
  `lightning_fast`, `response_format_type = "json_schema"`); `scripts/switchyard_e2e.{sh,py}`:
  start FreeToken (P2 + `--served-model-name nemotron-3.5-lightning --reasoning-parser nemotron_v3
  --tool-call-parser qwen3_coder --max-output-tokens 16384`), `cargo build --release -p
  switchyard-server -p switchyard-soak`, contract checks (alias, cached_tokens on repeat,
  json_schema verdict, 300K prompt → 400 code, streaming first event, session header echo,
  tool-call burst → `finish_reason == "tool_calls"`), then soak at concurrency 16:
  ```
  switchyard-soak --base-url http://127.0.0.1:4000 --model switchyard/passthrough \
    --duration 20m --concurrency 16 --max-output-tokens 256 --prompt-bytes 16384 \
    --context-window 131072 --scenario prefix-reuse --scenario growing-conversation \
    --scenario tool-call-burst --scenario large-tool-catalog --scenario long-context \
    --max-error-fraction 0
  ```
  then `switchyard/stage`, then `--scenario-set resilience`. Agent smoke: Claude Code and Codex
  through Switchyard. Pass: 0 errors, prefix-reuse TTFT shared < unique, cached_tokens monotonic in
  growing-conversation, no unhandled "session is busy".

### Phase 3 risks
Preflight tokenization cost at 32K (flag-gated); judge + main call on one session id (busy
fallback loses prefix reuse for that call); JSON mode is probabilistic without constrained decoding;
16-way decode on 16 GB shrinks the MoE cache — measure tok/s at bs=16.

---

## Phase 4 — MTP speculative decoding (time-boxed, behind a flag)

Flag `--speculative-mtp-tokens N` (`EngineConfig.spec_mtp_tokens`, default 0). Off = byte-identical
state dict, cache geometry, and graph path (asserted by test).

Design (k=1 first, chained k≤3 later):
- `models/nemotron_h/mtp.py::NemotronHMTPHead`: `enorm`, `hnorm`, `eh_proj` (5376→2688),
  attention block (own KV layer: extend `FullAttentionGroupConfig.layer_ids` with id 52), MoE
  block (bank id 23 through `OffloadMoeCache`; BF16 experts NVFP4-packed at load by a new
  `quantize_bf16_to_nvfp4` matching the modelopt layout, +0.6 GiB host), `final_layernorm`;
  shares `backbone.embeddings` and `lm_head` (SGLang bug precedent). Input = target's pre-`norm_f`
  hidden (expose from backbone forward). Reference: vLLM `nemotron_h_mtp.py`.
- Step: target decode → sample x1 → draft x2 = argmax MTP(x1, h0) → verify as an extend of k+1
  tokens → `engine/spec_sample.py::rejection_sample` (greedy exact-match; stochastic accept with
  prob p_target under top-p; resample from residual) → commit accepted prefix + bonus.
- Mamba state: verify path runs the decode SSU k+1 times on a scratch copy writing per-position
  states `[L, B, k+1, H, P, N]` (48 MiB × (k+1) × B; cap spec at bs ≤ 8), then `index_copy_` the
  accepted position into the live slot. Exact, no re-scan, no chunk-scan on tiny extends.
- KV: rejected positions stay allocated and are overwritten next step; `allocate_paged`
  pre-allocates k+1 per req; `_make_write_tuple` takes variable counts.
- Scheduler: `Req.complete_n(n)`, one `DetokenizeMsg` per token, EOS/stop truncation inside an
  accepted run, tool-call anchor check `<=` with recheck.
- Graphs: eager only in 4a; 4b captures draft + target decode, verify stays eager if the attention
  backend cannot capture uniform extends.

Ordered tasks: (1) mtp.py + loading + NVFP4 packing, reference-forward test; (2) rejection
sampler + statistical tests; (3) Mamba per-position verify + bit-identical commit test; (4)
scheduler multi-token commit; (5) engine orchestration, greedy equivalence spec on == spec off;
(6) bench + gate; (7) graphs; (8) k=2/3. Subagents: E (1,5), F (2,4), G (3), H (6).

Go/no-go on the 5080 (bs 1 and 8, 2K prompt, 512 decode): mean accepted length ≥ 1.55 at k=1
(vLLM saw ~63% on Super), decode ≥ 1.25× at bs=1, ≤ 5% regression at bs=8. Otherwise ship
disabled. Main risk: verify touches up to 6(k+1) experts per MoE layer on the offload path,
amplifying PCIe fetches.

---

## Verification summary
- Unit: `uv run pytest tests/models tests/engine tests/server tests/moe tests/kernels tests/kvcache
  tests/scheduler -m "not slow"`; ruff; `git diff --check`.
- Phase 1: parity script, batch invariance, prefix-cache equality, elastic ramp, needle 64K–256K,
  tool round trip, AIME subset. Results file under `benchmarks/results/`.
- Phase 2: kernel tests vs HF reference, microbenchmarks with targets above, `bench_decode_moe.py`
  before/after (bs=1 ≥ 2× Phase 1), 8K TTFT ≤ 2 s, AIME A/B with `FREETOKEN_MAMBA2_REF=1`.
- Phase 3: contract checks + Switchyard soak scenarios at concurrency 16 + Claude Code/Codex smoke.
- Phase 4: greedy equivalence, acceptance histogram, tok/s gate.

## Addendum 2026-09-03 — parallel 1M sessions (user goal)

Model: many long-lived agent sessions, each up to 1M tokens, few decoding at any instant.
No live host-KV tier for active sequences (every decode step reads the whole KV; 3 GB/step over
PCIe ≈ 100 ms/token — not viable). Instead rely on what exists and validate it for Nemotron:
- Growable KV (VMM segments, `--kv-grow-step-tokens`) funds KV from the expert cache as sessions grow.
- Session spill (`--session-spill-ram-gb`, `--session-spill-dir`) checkpoints idle sessions' KV +
  Mamba state to RAM then NVMe and restores exactly on the next turn (validated on Ornith only).
- KV per 1M session ≈ 3 GB at fp8 + 47 MiB Mamba state → 2–3 concurrently decoding 1M sessions
  on the 5080; ~4 spilled sessions fit in the remaining host RAM (40 GB − 16.5 GB banks − process).
User decision (2026-09-04): 1M profile runs ONE resident session; all other sessions queue and are
served in sequence via spill/restore. The 16-way profile remains for short-context Switchyard traffic.
Gate (after Phase 2 kernels, before Phase 4): 1M profile `--max-seq-len-override 1048576
--num-tokens 1048576 --kv-cache-dtype fp8_e4m3 --attention-backend triton --kv-grow-step-tokens
131072 --max-running-requests 1 --session-spill-ram-gb 12
--session-spill-dir <nvme>`; three sessions grown to ~1M each with disjoint needles, one spilled
and restored, all coherent; record prefill/decode tok/s and spill/restore times.
KV dtype: Phase 1 A/B (2026-09-04) chose q8_0 — fp8_e4m3 flipped first tokens on cached-prefix reuse 3/6 runs; equal VRAM and reasoning score. See benchmarks/results/nemotron35_lightning_5080_2026-09-04.md.

## Addendum 2026-09-04 — task 3E session residency policy (DECIDED 2026-09-04)

1M profile: `--max-running-requests 1`; other sessions queue and are served in sequence.
- **No spill while the queue is empty.** The resident session's KV + Mamba state stays in VRAM
  until another session's request needs the slot; only then is it checkpointed (on demand, not on
  an idle timer). TTL-based release must not evict a resident session that nobody is waiting on.
- **Bounded checkpoint store with age-based eviction.** A byte cap (e.g. `--session-spill-limit-gb
  50`, RAM + NVMe tiers combined or per tier) and eviction of the oldest checkpoints first when the
  cap is reached. Deleted sessions fall back to ordinary prefill on their next request.
Audit (HEAD 584f17c): auto sessions are checkpointed 30 s after turn end regardless of demand
(`scheduler.py:1192 _release_due_soft_sessions`, `auto_session_grace_seconds`); admission-time
reclaim (`scheduler.py:1206`) runs once and skips a mid-turn session, so a queued request waits for
the timer; TTL (300 s) closes the session AND destroys its checkpoint (`_close_session`,
`_discard_session_spill`); disk tier 64 GiB cap refuses instead of evicting, no age order, no
on-disk key, rmtree on shutdown, leaked `server-*` dirs after a crash; explicit `session_id`
leases are never spilled; all header-derived ids are reclaimable.
Decided scope (all three):
1. Spill on demand only: no timer release while nothing is waiting; when an admission fails for
   lack of KV/state slots, reclaim the oldest idle reclaimable lease (re-run at the
   admission-failure point, not only at message receipt). Keep the grace timer only as a very long
   safety (configurable, default off/∞).
2. Retention by capacity + age: checkpoint lifetime decoupled from lease TTL. `--session-spill-limit-gb`
   (default 50) total across RAM+disk; evict oldest-by-last-use first when the cap or the filesystem
   guard is hit instead of refusing. TTL closes the lease but keeps the checkpoint; a later request
   with the exact prefix restores it.
3. Survive restart: checkpoints keyed on disk by session id + prompt-prefix hash + K/V layout
   fingerprint (manifest JSON next to chunks); startup scans the spill root, adopts valid records,
   deletes stale/foreign ones; shutdown no longer rmtrees. Restore still requires exact prefix +
   fingerprint match.
