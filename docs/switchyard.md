# FreeToken as a Switchyard upstream

[NVIDIA Switchyard](https://github.com/NVIDIA/switchyard) routes agent traffic across
model tiers. FreeToken serves as a Switchyard **`openai_chat`** upstream: the router
sends `/v1/chat/completions` and nothing else, so every promise below is about that
one route.

This page is the operator's reference for the Nemotron 3.5 Lightning profile on a
16 GiB RTX 5080, but nothing here is Nemotron-specific except the launch line.
The model-side sizing, launch profiles and 1M single-session notes live in
[`docs/nemotron.md`](nemotron.md).

Automated checks live in `scripts/switchyard_e2e.py` (wrapper:
`scripts/switchyard_e2e.sh`).

---

## 1. Launch FreeToken

The serving profile (P2 — 16 concurrent requests, elastic KV, prefix cache, FP8 KV):

```bash
ft serve --model ~/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --max-running-requests 16 --elastic-initial-requests 4 --kv-grow-step-tokens 65536 \
  --num-tokens 262144 --max-seq-len-override 131072 --kv-cache-dtype q8_0 \
  --attention-backend triton --moe-backend offload --moe-pageable-gpu --moe-cache-auto \
  --memory-ratio 0.85 --max-prefill-length 8192 --host-ram-reserve-gb 6 \
  --enable-cache-report \
  --served-model-name nemotron-3.5-lightning \
  --reasoning-parser nemotron_v3 --tool-call-parser qwen3_coder \
  --force-nonempty-content --max-output-tokens 16384
```

The serving-compliance half of that line:

| Flag | Why Switchyard needs it |
|---|---|
| `--served-model-name nemotron-3.5-lightning` | The id `GET /v1/models` advertises; `[targets.*].id` in `routes.toml` must match. |
| `--reasoning-parser nemotron_v3` | Splits `<think>…</think>` into `reasoning_content` and escapes to a tool call when the model opens `<tool_call>` without closing the think block. (`auto` also selects it for Nemotron-3.x.) |
| `--tool-call-parser qwen3_coder` | Lightning emits Qwen3-Coder nested-XML tool calls. |
| `--enable-cache-report` | Populates `usage.prompt_tokens_details.cached_tokens` (always present with the flag on, absent without it). Without it the router sees no prefix reuse and the soak's `prefix-reuse` scenario cannot be graded. |
| `--force-nonempty-content` | A thinking turn that produces only reasoning answers with the reasoning text instead of an empty `content`. Switchyard treats empty content as a failed turn. |
| `--max-output-tokens 16384` | Ceiling for a request that sends no `max_completion_tokens`. |
| `--kv-cache-dtype q8_0` | FP8 KV (FreeToken block scales; the checkpoint's `k_scale`/`v_scale` are ignored). Requires `--attention-backend triton`. |

Optional knobs that change the contract: `--no-context-preflight` (see §5),
`--json-retry N` (see §4), `--hidden-states-dir DIR` (see §6). Default listen
address is `127.0.0.1:1919`.

### Served context window

With no `--max-seq-len-override`, FreeToken now serves
`min(max_position_embeddings, tokenizer_config.model_max_length)`. Lightning's
checkpoint states 1,048,576 positions against a 262,144-token tokenizer window, so
the default served window is 262,144 rather than 1M. A tokenizer config with no
`model_max_length`, or with the `int(1e30)` "unbounded" sentinel transformers
writes, leaves the geometry untouched — no other model's behavior changes.

Whatever the resulting window is, it is advertised as `max_model_len` /
`context_length` on `GET /v1/models`, and it must equal the `context_window` you put
in `routes.toml`: the router sizes its own overflow handling from that number.
The P2 line pins 131,072 explicitly, which is what fits the 5080's KV budget at 16
concurrent requests.

---

## 2. `routes.toml`

Validated against `switchyard-runner`'s serde structs (`config.rs`, `algorithm.rs`,
both `deny_unknown_fields`) — an invented key fails at startup. Regenerate it with
`scripts/switchyard_e2e.py soak` (it writes this file), and check any edit with:

```bash
switchyard-server --config routes.toml --dry-run
```

```toml
schema_version = 1

[llm_clients.freetoken]
format = "openai_chat"
base_url = "http://127.0.0.1:1919/v1"
max_retries = 2

# Capable tier: thinking on (the checkpoint's default).
[targets.lightning]
id = "nemotron-3.5-lightning"
llm_client = "freetoken"

[targets.lightning.extra_body.chat_template_kwargs]
enable_thinking = true

# Efficient tier and classifier: thinking off, and answer with the reasoning text
# rather than an empty message if the turn produces only reasoning.
[targets.lightning_fast]
id = "nemotron-3.5-lightning-fast"
llm_client = "freetoken"

[targets.lightning_fast.extra_body.chat_template_kwargs]
enable_thinking = false
force_nonempty_content = true

[routes.passthrough]
id = "switchyard/passthrough"
type = "passthrough"
target = "lightning"
context_window = 131072
tool_calling = true
reasoning = true

[routes.stage]
id = "switchyard/stage"
type = "stage_router"
picker = "efficient_first"
capable_target = "lightning"
efficient_target = "lightning_fast"
confidence_threshold = 0.6
recent_turn_window = 28
context_window = 131072
tool_calling = true
reasoning = true

[routes.stage.classifier]
target = "lightning_fast"
base_threshold = 0.6
classify_trigger = "user_turn"
response_format_type = "json_schema"
max_output_tokens = 512
```

Four things that are easy to get wrong:

- **Tables, not arrays.** `[targets.<name>]` and `[routes.<name>]`, never
  `[[targets]]`. The algorithm keys (`type`, `target`, `picker`, …) are flat inside
  the route table, selected by `type`.
- **The two targets must not share a model id.** Switchyard keeps one target per
  `(llm_client, model id)` pair and drops the other with a `WARN`. There is one GPU
  here, so both tiers are the same process; giving the efficient tier the id
  `nemotron-3.5-lightning-fast` keeps them distinct. FreeToken echoes the request's
  `model` back without validating it, so any distinct string reaches the same server.
- **`api_key_env` names an environment variable, not the secret**, and
  `switchyard-server` refuses to start when that variable is unset. FreeToken needs
  no bearer token locally, so the key is simply omitted above.
- **There is no timeout key** in `[llm_clients.*]`. Per-request timeouts are the
  client's (or the soak's `--request-timeout`).

Start it with:

```bash
switchyard-server --config routes.toml --host 127.0.0.1 --port 4000
# health: GET http://127.0.0.1:4000/health -> {"status":"ok"}
# routes: GET http://127.0.0.1:4000/v1/models -> switchyard/passthrough, switchyard/stage
```

Clients then send `model: "switchyard/passthrough"` (or `switchyard/stage`).

---

## 3. Session binding and headers

Switchyard forwards the caller's headers to the upstream minus a reserved set, and
always sends `x-switchyard-session-id` when it has a session. FreeToken binds that
id to a KV session lease, so a conversation keeps its prefix (and, for hybrid
models, its recurrent-state snapshot) across turns.

Precedence on `/v1/chat/completions`:

1. the request's own `session_id` field (the client owns the lease),
2. `X-Switchyard-Session-Id`,
3. `X-Claude-Code-Session-Id`,
4. `X-Codex-Session-Id`,
5. the OpenAI `prompt_cache_key` field,
6. `Session-Id` / `X-Session-Id`.

`X-Switchyard-Agent-Id` and `X-Claude-Code-Agent-Id` split a sub-agent onto its own
lease, so a parent and its child neither serialize on nor evict each other's prefix.

The resolved id comes back as **`X-FreeToken-Session-Id`** on both the JSON and the
streaming response. Switchyard does not read it — affinity is entirely router-side —
but it is how you confirm binding, and it is what `DELETE /v1/sessions/{id}` takes.

An id FreeToken *inferred* from a header is reclaimable, and a `session … is busy`
collision (a classifier call landing on the same conversation as the turn it grades)
is retried once without a lease rather than failed. That retry loses prefix reuse for
that one call; it never surfaces as an error.

---

## 4. JSON mode

A `stage_router` classifier asks the efficient target for an `EscalationVerdict` with
`response_format` type `json_schema` (`crates/libsy/src/prompts/escalation/schema.json`):

```json
{"type": "object",
 "properties": {"escalate": {"type": "boolean"}, "reason": {"type": "string"}},
 "required": ["escalate", "reason"], "additionalProperties": false}
```

FreeToken has no constrained decoding, so JSON mode is prompted and then enforced
after the fact:

- the schema is appended to the system block and `enable_thinking` defaults to
  **false** for the call (an explicit `chat_template_kwargs.enable_thinking` or a
  `reasoning_effort` still wins);
- the completion is buffered, stripped of think residue and code fences, and the
  first balanced JSON value is extracted and re-emitted as canonical JSON — on the
  streaming path as a single content delta before the finish chunk;
- a `json_schema` answer that fails validation is retried once at temperature 0 with
  the validator error fed back as a user turn (`--json-retry`, `FREETOKEN_JSON_RETRY`;
  0 disables);
- a final failure returns the raw text with **HTTP 200**, never a 4xx/5xx. That is
  deliberate: Switchyard scores an unparseable verdict as *ambiguous* (stay on the
  efficient tier) and a judge route falls through to a stronger target, whereas an
  error status would break the route.

Because it is prompted rather than constrained, treat JSON mode as probabilistic. If
a classifier route proves flaky, `response_format_type = "json_object"` is the more
forgiving mode (Switchyard then pretty-prints the schema into the prompt itself and
validates locally).

`/v1/completions` still rejects `response_format`; only chat has JSON mode.

---

## 5. Context overflow

Switchyard routes on `error.code`: an over-length prompt must come back as
**HTTP 400** with `error.code == "context_length_exceeded"`, or the route falls
through instead of retargeting to a larger-window tier.

- **Non-stream:** 400 with that code. The message contains both "maximum context
  length" and "prompt is too long", so clients matching on prose also recover.
- **Stream:** by default the frontend preflight tokenizes the rendered prompt
  *before* the stream opens, so an overflow is answered as the same plain 400 JSON —
  the stream never starts, and no queue slot is spent. With `--no-context-preflight`
  the scheduler catches it instead and the error rides as the **first** SSE event
  (`{"error": {…, "code": "context_length_exceeded"}}`), ahead of any role chunk,
  followed by `[DONE]`.

`scripts/switchyard_e2e.py contract` accepts either shape and reports which one it
saw. The one thing that would be a bug is an error arriving *after* a role chunk.

The preflight costs one extra tokenizer pass per request (~1.2 µs/token); at 32K
prompts that is measurable, which is what `--no-context-preflight` is for.

---

## 6. Hidden-state probe target

Switchyard's prefill complexity router (branch `prefill-complexity-router-v1-port`,
`crates/switchyard-components/src/prefill_probe/scorer.rs`) does not route on the answer
— it routes on the prompt's *residual stream*. It posts one throwaway completion to a
probe server, reads a `.safetensors` artifact off shared storage, mean-pools it per
layer, and feeds the result to a learned head. FreeToken serves that contract; the probe
weights are trained outside both repos and are not FreeToken's concern.

Start the server with the export enabled — the directory is the **only** path FreeToken
will ever write, canonicalized once at startup:

```bash
mkdir -p /tmp/ft-hidden-states
ft serve --model ~/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  ... --hidden-states-dir /tmp/ft-hidden-states
```

Without the flag the feature is off and a probe request is a 400. The router's client
must be able to read that same path (a shared mount, or the same host).

### The request

```bash
curl http://127.0.0.1:1919/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nemotron-3.5-lightning",
    "messages": [{"role": "user", "content": "Return one short sentence."}],
    "max_tokens": 1,
    "kv_transfer_params": {
      "hidden_states_path": "/tmp/ft-hidden-states",
      "include_output_tokens": false
    }
  }'
```

| `kv_transfer_params` field | Meaning |
|---|---|
| `hidden_states_path` | Directory to write into. Must be `--hidden-states-dir` or a subdirectory of it — resolved through symlinks and `..`, and refused otherwise. Omit it to use the root. |
| `layer_ids` | Which blocks to export. Default: every block, in forward order (52 on Lightning). Must be contiguous from 0 and ascending — Switchyard's loader indexes the middle axis positionally, so a gap would silently mislabel features. |
| `include_output_tokens` | Accepted and ignored. FreeToken exports prompt positions only, which is all the router pools. |

`kv_transfer_params` is typed **only** on `/v1/chat/completions`. On `/v1/completions`,
`/v1/messages` and `/v1/responses` it lands in the untyped extras and is ignored.

### The response

```json
{
  "id": "chatcmpl-7", "object": "chat.completion", "model": "nemotron-3.5-lightning",
  "choices": [{"index": 0, "message": {"role": "assistant", "content": "The"},
               "finish_reason": "length"}],
  "usage": {"prompt_tokens": 42, "completion_tokens": 1, "total_tokens": 43},
  "kv_transfer_params": {
    "hidden_states_path": "/tmp/ft-hidden-states/6f1c….safetensors"
  }
}
```

Read the path from the response; the file name is a uuid FreeToken chooses. On the
stream path the same object rides on the terminal chunk (the router never streams).

### The artifact

| Key | Shape | Dtype |
|---|---|---|
| `hidden_states` | `[prompt_tokens, layers, hidden]` | BF16 |
| `token_ids` | `[prompt_tokens]` | I64 |

`hidden_states[t, i]` is the **post-block residual stream**: the value block `i` leaves
behind after adding its mixer output to `x`, before the next block's input norm and
before `norm_f`. It is not the final-norm output and not logits. On Nemotron-H every
block is one "layer" here regardless of what it mixes — Lightning's 23 mamba, 23 MoE
and 6 attention blocks are one 52-deep stream, and the router wants the stream.

`token_ids` are the prompt tokens actually forwarded, in order, so a consumer never has
to re-tokenize to line the rows up. It is optional in vLLM's contract; FreeToken always
writes it, and Switchyard validates it when present.

The file is written under an exclusive `flock` before the response goes out. Switchyard
polls for the path (20 × 50 ms) and then takes `LOCK_EX` itself, so a reader that opens
it mid-write blocks rather than parsing a truncated header. It **deletes** the artifact
once it has scored it; FreeToken never cleans the directory up, so a client that stops
consuming will fill the disk.

### What a probe request does differently

- **It bypasses prefix reuse.** A cached prefix would leave those positions out of the
  forward and therefore out of the artifact. `Req.no_prefix_cache` makes the match run
  against the empty prefix; the completed prompt is still committed to the radix tree,
  so ordinary traffic behind the probe still hits.
- **It binds no session lease.** `x-switchyard-session-id` and `prompt_cache_key` are
  ignored for a probe: a lease protects a prefix for a next turn, and the probe refuses
  to reuse a prefix and has no next turn. This also stops concurrent probes on one
  conversation from serializing on a `session … is busy`.
- **It is capped at `--hidden-states-max-tokens` (default 4096) prompt tokens.** A
  longer prompt is a 400 with `error.code = context_length_exceeded`. The cap is a size
  guard, not a context guard: every layer of every prompt token is exported, so one
  4096-token probe over 52 layers at hidden 2688 is ~1.1 GiB. The check runs frontend
  side even with `--no-context-preflight`, and again in the scheduler.
- **It costs nothing when absent.** Without `kv_transfer_params` no sink is installed;
  the model forward reads one attribute and the captured decode graphs never see it.

### Verifying it

CPU: `tests/server/test_hidden_states_probe.py` (wire + validation + writer round trip),
`tests/scheduler/test_hidden_states_no_prefix_reuse.py`,
`tests/models/test_nemotron_h_hidden_states.py` (the hook captures the post-block
residual, not `norm_f`).

GPU, against a running server (P1 profile plus `--hidden-states-dir`). The served model
and the bf16 reference do not fit in host RAM at the same time, so the run is two-phase —
capture while the server is up, score once it is stopped:

```bash
# phase A, server up
uv run benchmarks/probe_hidden_states_parity.py \
  --model ~/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --base-url http://127.0.0.1:1919 --hidden-states-dir /tmp/ft-hidden-states \
  --prompt-tokens 300 --capture-only
# phase B, server stopped
scripts/gpu_lock.sh uv run benchmarks/probe_hidden_states_parity.py \
  --model ~/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --hidden-states-dir /tmp/ft-hidden-states --artifact /tmp/ft-hidden-states/<uuid>.safetensors
```

The reference is transformers' own `NemotronHBlock` stack with the modelopt checkpoint
streamed one block at a time (`from_pretrained` cannot load this release, and dense bf16
NemotronH is 58.8 GiB). Result 2026-09-04 on the RTX 5080: all 52 layers ≥ 0.998840,
median 0.999760 —
[`benchmarks/results/nemotron35_lightning_5080_hidden_states_parity_2026-09-04.md`](../benchmarks/results/nemotron35_lightning_5080_hidden_states_parity_2026-09-04.md).

It sends one 300-token probe, loads the artifact, and compares each layer's mean-pooled
vector against `transformers.AutoModelForCausalLM(output_hidden_states=True)` on CPU in
bf16 over the artifact's own `token_ids` (HF's `hidden_states[i + 1]` is block `i`'s
output). Per-layer cosine must exceed 0.99, which absorbs NVFP4/FP8 drift while still
catching an off-by-one layer index, a final-norm leak, or a dropped prefill chunk.

---

## 7. Known limitations

| Not supported | Behavior |
|---|---|
| `logprobs` / `top_logprobs > 0` | 400 `invalid_request_error`. Switchyard never sends them; `top_logprobs: 0` is accepted. |
| `n > 1` | 400 "Only n=1 is supported". Switchyard never sends `n`. |
| `logit_bias`, `function_call` (legacy) | 400. Use `tools`/`tool_choice`. |
| `seed` | Ignored; Switchyard never sends it. |
| Constrained decoding | None — see §4. |
| Multiple upstreams | One GPU, one process: both tiers are the same weights with different `chat_template_kwargs`. |
| `x-switchyard-session-id` in Switchyard's own session stats | Not recorded router-side (Switchyard `docs/known_issues.md`); FreeToken still binds it. |

Everything Switchyard *does* send is supported: `max_completion_tokens` (it never
sends `max_tokens`), `tools`/`tool_choice`/`parallel_tool_calls`, `temperature`,
`top_p`, `stream` + `stream_options`, `reasoning_effort`, `response_format`,
`prompt_cache_key`, `user`, `stop`, and the `developer` role (mapped to `system` for
non-Harmony templates). Responses carry `reasoning_content` (Switchyard also accepts
`reasoning`), `usage.prompt_tokens_details.cached_tokens`,
`usage.completion_tokens_details.reasoning_tokens`, and terminate SSE with `[DONE]`.

---

## 8. Running the checks

Build the Rust binaries once:

```bash
cd ~/ai/Switchyard && cargo build --release -p switchyard-server -p switchyard-soak
```

**Wire contract** (FreeToken only, no router needed):

```bash
scripts/switchyard_e2e.sh contract --base-url http://127.0.0.1:1919 \
  --model nemotron-3.5-lightning
```

Checks: `max_completion_tokens` alias; `reasoning_content` + `reasoning_tokens`;
`cached_tokens > 0` on a repeated prompt; a schema-valid `EscalationVerdict` in both
stream and non-stream JSON mode; `x-switchyard-session-id` → stable
`X-FreeToken-Session-Id` across two turns and on the stream; a tool-call burst
reaching `finish_reason == "tool_calls"` with parseable arguments, reassembled from
SSE deltas; and an oversize prompt producing the 400/first-SSE-event overflow above.
Exit code is 0 only when every check passes.

**Soak through the router** (starts `switchyard-server`, waits for `/health`, runs
`switchyard-soak`, parses its verdict):

```bash
scripts/switchyard_e2e.sh soak --base-url http://127.0.0.1:1919 --duration 20m
```

which runs, per route (`switchyard/passthrough` then `switchyard/stage`):

```bash
switchyard-soak --base-url http://127.0.0.1:4000 --model switchyard/passthrough \
  --duration 20m --concurrency 16 --max-output-tokens 256 --prompt-bytes 16384 \
  --context-window-tokens 131072 \
  --scenario prefix-reuse --scenario growing-conversation \
  --scenario tool-call-burst --scenario large-tool-catalog --scenario long-context \
  --max-error-rate 0 --request-timeout 600 --results-dir <workdir>/results-...
```

Note the flag spellings: `--context-window-tokens` and `--max-error-rate` (not
`--context-window` / `--max-error-fraction`). `--request-timeout` is the soak's own
client timeout; its 120 s default is shorter than a 118K-token `long-context` or
`context-overflow` prefill queued behind fifteen siblings on one 16 GiB card, so
`switchyard_e2e.py soak` raises it to 600 s (`--request-timeout`) — otherwise the
run reports client timeouts as upstream errors. Then the resilience group:

```bash
scripts/switchyard_e2e.sh soak --duration 10m --scenario-set resilience \
  --route switchyard/passthrough
```

The soak preflights `GET /health` and requires the exact `--model` id to appear in
the router's `GET /v1/models` before sending load; it also scrapes the router's
`/metrics` for `switchyard_total_requests` / `switchyard_total_errors`. `--results-dir`
must not already exist. The verdict is read from `summary.json` (`passed`,
`failure_reasons`, `requests`, `failures`, `error_rate`) when present, else from the
terminal `Soak PASS: …` / `Soak FAIL: …` line, else from the exit code.

**Pass criteria for Phase 3:** 0 request errors; prefix-reuse TTFT lower on shared
prefixes than on unique ones; `cached_tokens` monotonic within a
growing-conversation; no unhandled `session is busy`.

**Agent smoke tests** (manual, one terminal each):

```bash
scripts/switchyard_e2e.sh agents   # prints the exact env lines
```

Claude Code points `ANTHROPIC_BASE_URL` at the router (`switchyard-server` serves
`/v1/messages` and translates down to the `openai_chat` upstream); Codex points
`OPENAI_BASE_URL` at `<router>/v1`. Both use `switchyard/passthrough` as the model.

---

## 9. Troubleshooting

| Symptom | Cause |
|---|---|
| `prompt_tokens_details` missing from `usage` | `--enable-cache-report` missing. With the flag on the field is always present, so `cached_tokens: 0` means a real miss, not a disabled report. |
| Router `WARN … reuses model id … the other is dropped` | Two targets share `id` on one `llm_client`; give the efficient tier its own id. |
| `could not read api_key_env X: environment variable not found` | `api_key_env` names an unset variable — export it or drop the key. |
| Soak fails immediately with an unknown model | `--model` must be a **route** `id`, not a target id or FreeToken's model name. |
| Soak reports metrics-check failures | It is scraping the *router's* `/metrics`, not FreeToken's; the router must be the `--base-url`. |
| Empty assistant messages on the efficient tier | `force_nonempty_content` not set (server flag or `chat_template_kwargs`). |
| Overflow answered 500 or with a bare message | Check `error.code`; only `context_length_exceeded` makes the route retarget. |
| Soak intervals go `status=STALLED` | The FreeToken backend scheduler died and in-flight requests hang until the client timeout. Since 2026-09-04 `/health` answers **503** with the dead worker's name (it polls the worker handles, not just the supervisor's latched `fatal_error`), and the stop is bounded instead of hanging in "Waiting for background tasks to complete". Check the FreeToken log for `Backend supervisor: backend worker … exited`. A pre-fix server answers `health=ok` throughout the stall. |
| `unknown scenario "x"` | Valid ids: `short-interactive`, `long-context`, `decode-heavy`, `prefix-reuse`, `mixed-traffic`, `growing-conversation`, `large-tool-catalog`, `tool-call-burst`, `stage-transitions`, `classifier-mix`, `context-overflow`, `failure-pressure`, `client-cancellation`. Sets: `core`, `agentic`, `resilience`, `standard`, `all`. |
