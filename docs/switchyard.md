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
  --memory-ratio 0.85 --max-prefill-length 8192 --host-ram-reserve-gb 3 \
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
| `--enable-cache-report` | Populates `usage.prompt_tokens_details.cached_tokens`. Without it the router sees no prefix reuse and the soak's `prefix-reuse` scenario cannot be graded. |
| `--force-nonempty-content` | A thinking turn that produces only reasoning answers with the reasoning text instead of an empty `content`. Switchyard treats empty content as a failed turn. |
| `--max-output-tokens 16384` | Ceiling for a request that sends no `max_completion_tokens`. |
| `--kv-cache-dtype q8_0` | FP8 KV (FreeToken block scales; the checkpoint's `k_scale`/`v_scale` are ignored). Requires `--attention-backend triton`. |

Optional knobs that change the contract: `--no-context-preflight` (see §5),
`--json-retry N` (see §4). Default listen address is `127.0.0.1:1919`.

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

## 6. Known limitations

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

## 7. Running the checks

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

## 8. Troubleshooting

| Symptom | Cause |
|---|---|
| `cached_tokens` always 0 | `--enable-cache-report` missing. |
| Router `WARN … reuses model id … the other is dropped` | Two targets share `id` on one `llm_client`; give the efficient tier its own id. |
| `could not read api_key_env X: environment variable not found` | `api_key_env` names an unset variable — export it or drop the key. |
| Soak fails immediately with an unknown model | `--model` must be a **route** `id`, not a target id or FreeToken's model name. |
| Soak reports metrics-check failures | It is scraping the *router's* `/metrics`, not FreeToken's; the router must be the `--base-url`. |
| Empty assistant messages on the efficient tier | `force_nonempty_content` not set (server flag or `chat_template_kwargs`). |
| Overflow answered 500 or with a bare message | Check `error.code`; only `context_length_exceeded` makes the route retarget. |
| Soak intervals go `status=STALLED` | The FreeToken backend scheduler died and in-flight requests hang until the client timeout. Since 2026-09-04 `/health` answers **503** with the dead worker's name (it polls the worker handles, not just the supervisor's latched `fatal_error`), and the stop is bounded instead of hanging in "Waiting for background tasks to complete". Check the FreeToken log for `Backend supervisor: backend worker … exited`. A pre-fix server answers `health=ok` throughout the stall. |
| `unknown scenario "x"` | Valid ids: `short-interactive`, `long-context`, `decode-heavy`, `prefix-reuse`, `mixed-traffic`, `growing-conversation`, `large-tool-catalog`, `tool-call-burst`, `stage-transitions`, `classifier-mix`, `context-overflow`, `failure-pressure`, `client-cancellation`. Sets: `core`, `agentic`, `resilience`, `standard`, `all`. |
