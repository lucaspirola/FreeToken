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
| `--served-model-alias NAME` (repeatable, optional) | Extra ids the same model answers to. `GET /v1/models` lists the served name first, then each alias in flag order (one card each, same `root`); `/v1/chat/completions`, `/v1/completions`, `/v1/messages`, `/v1/responses` and `/v1/models/{id}` accept any of them. The response `model` field echoes the id the request named, so a router keying on the id it dispatched with sees it back. Empty or duplicate names fail at startup. Not in `serve.sh`; pass it via `SOAK_EXTRA_ARGS` or the launch line. |
| `--strict-model-name` (optional) | Refuse (`404`, `code: model_not_found`) a request whose `model` is neither the served name nor an alias. Off by default: any name is accepted and echoed, because Anthropic-protocol clients send `claude-*` names to whatever proxy they are pointed at. |
| `--reasoning-parser nemotron_v3` | Splits `<think>…</think>` into `reasoning_content` and escapes to a tool call when the model opens `<tool_call>` without closing the think block. (`auto` also selects it for Nemotron-3.x.) |
| `--tool-call-parser qwen3_coder` | Lightning emits Qwen3-Coder nested-XML tool calls. |
| `--enable-cache-report` | Populates `usage.prompt_tokens_details.cached_tokens` (always present with the flag on, absent without it). Without it the router sees no prefix reuse and the soak's `prefix-reuse` scenario cannot be graded. |
| `--force-nonempty-content` | A thinking turn that produces only reasoning answers with the reasoning text instead of an empty `content`. Switchyard treats empty content as a failed turn. |
| `--max-output-tokens 16384` | Ceiling for a request that sends no `max_completion_tokens`. |
| `--kv-cache-dtype q8_0` | FP8 KV (FreeToken block scales; the checkpoint's `k_scale`/`v_scale` are ignored). Requires `--attention-backend triton`. |
| `--pin-prefix-min-tokens N` (default 1024; 0 disables) | Prefix auto-pin (hybrid radix cache): a cached prefix that a *second* request reuses with `cached_tokens >= N` is locked against eviction until `DELETE /v1/cache/pins`. See §3a. |
| `--pin-prefix-max-tokens N` (default 65536; 0 = unlimited) | Total pinned-token budget. Past it, new prefixes are not pinned and `scheduler.prefix.pin_budget_refusals` counts each refusal. |

Optional knobs that change the contract: `--no-context-preflight` (see §5),
`--json-retry N` (see §4), `--hidden-states-dir DIR`, `--pooled-sink-dir DIR` (see §6).
Default listen
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

### 3a. Prefix pinning and the prefix counters

Session leases keep *one conversation's* prefix resident; they do nothing for a prefix
that many conversations share (a long system prompt, a tool manifest, a repository
briefing). Under LRU pressure that shared prefix is evicted exactly as often as any other
leaf, and every requester that lands after the eviction pays the prefill again.

With `--pin-prefix-min-tokens N` (default 1024) the hybrid radix cache pins such a prefix the
first time it is **reused**: when a prompt is admitted with `cached_tokens >= N`, the matched
node's root path takes a persistent lock — the tree's own `inc_lock` (full-KV ref on
node..root, recurrent-state ref on the node and on every snapshot-bearing ancestor), held by
the cache manager instead of a request — so `evict_full` cannot take the path and
`evict_mamba` cannot tombstone its snapshot. "Reused" needs no producer bookkeeping: a
freshly admitted prompt has produced nothing, so any `cached_tokens > 0` at admission is by
construction a second request on KV a first one donated. A prefix is pinned once
(re-matching it is a no-op); a longer prompt through an already-pinned path adds only the
tokens below it to the ledger.

Pins are released only by **`DELETE /v1/cache/pins`** (returns the released
`pinned_prefixes` / `pinned_tokens`), by a cache rebuild (the tree is discarded), or by a
restart. `--pin-prefix-max-tokens` (default 65536) caps the total; a pin that would exceed
it is refused whole and counted. **Pins never starve admission on purpose**: a pinned path is
protected KV like a session lease, so a reserve or allocate that fails only because too much
is protected fails exactly as it does today — the server logs one warning naming the pins
and does *not* unpin automatically. If `pinned_tokens` is a large fraction of the pool and
`prefill.refusals` / `fresh_admits_deferred` climb, lower the budget or `DELETE` the pins.
The plain (non-hybrid) radix cache and the SWA radix are not pinned in this version.

`/v1/stats` reports the reuse the admission gate actually saw under `scheduler.prefix`,
counted once per prompt on its first chunk (where `PromptAdmittedMsg` is built):

| Field | Meaning |
|---|---|
| `hits`, `misses` | prompts admitted with `cached_tokens > 0` / `== 0` (a multimodal or hidden-state-probe prompt that bypasses the tree is a miss). |
| `hit_tokens` | sum of `cached_tokens` over hits. |
| `pooled_hits`, `pooled_hit_tokens` | the subset of `hits` / `hit_tokens` taken by pooled hidden-state probes (`kv_transfer_params.pooling`), which only resume from snapshot nodes carrying pooled sums (see "Pooled requests and the prefix cache" in §6). |
| `miss_tokens` | sum of the **full prompt length** over misses: the tokens the prefill forwards for them, last token included (`match_req` never matches the last token, so a repeated prompt is a hit with `cached_tokens = prompt_tokens - 1`). The forwarded remainder of a hit is `prompt_tokens_total - hit_tokens - miss_tokens`. |
| `pinned_prefixes`, `pinned_tokens` | gauges: distinct pinned match nodes and the distinct tokens their root paths cover. |
| `pin_budget_refusals` | pins refused by `--pin-prefix-max-tokens`. |

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

Without the flag the file export is off and a file probe request is a 400. The router's
client must be able to read that same path (a shared mount, or the same host). The
inline **pooled** variant below needs neither the flag nor shared storage.

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
| `pooling` | `"mean"`, `"last"` or `"both"`: return the pooled prompt vectors inline (see "Inline pooled hidden states" below). Set without `hidden_states_path`, no file is written. |
| `pooled_sink` | One path segment (`^[A-Za-z0-9._-]{1,64}$`, not `.`/`..`): the subdirectory of `--pooled-sink-dir` whose `pooled.jsonl` also receives this request's pooled vectors (see "Pooled sink (JSONL)" below). Requires `pooling`; refused (400) without the server flag. |

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

### Inline pooled hidden states

`pooling` returns what the router would compute from the artifact -- one vector per
layer -- on the response itself, so a consumer needs no shared storage and no file
read. It works **without `--hidden-states-dir`** (the hook is model-side and free; the
flag guards only what FreeToken writes to disk), and because nothing indexes the layers
positionally, `layer_ids` may be **any ascending, unique subset** of `0..num_layers-1`.
The engine keeps only a running float32 sum and the last row per layer -- O(layers x
hidden) on the host, accumulated across prefill chunks -- so the
`--hidden-states-max-tokens` cap **does not apply**.

```bash
curl http://127.0.0.1:1919/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nemotron-3.5-lightning",
    "messages": [{"role": "user", "content": "Return one short sentence."}],
    "max_tokens": 1,
    "kv_transfer_params": {"pooling": "both", "layer_ids": [12, 24, 36, 51]}
  }'
```

```json
{
  "id": "chatcmpl-7", "object": "chat.completion", "model": "nemotron-3.5-lightning",
  "choices": [{"index": 0, "message": {"role": "assistant", "content": "The"},
               "finish_reason": "length"}],
  "usage": {"prompt_tokens": 42, "completion_tokens": 1, "total_tokens": 43},
  "kv_transfer_params": {
    "pooled": {
      "layer_ids": [12, 24, 36, 51],
      "hidden": 2688,
      "prompt_tokens": 42,
      "prefix_tokens": 0,
      "dtype": "float32",
      "mean": "<base64 of float32 [4, 2688], row-major, little-endian>",
      "mean_suffix": "<base64 ...>",
      "last": "<base64 ...>"
    }
  }
}
```

| `pooled` field | Meaning |
|---|---|
| `layer_ids` | The layers actually pooled, in the order the rows come back (the request's list, or every block by default). |
| `hidden` | Row width (2688 on Lightning). |
| `prompt_tokens` | Positions pooled: every prompt token, chat-template tokens included -- the full prompt length whether or not a prefix was reused. |
| `prefix_tokens` | Positions served from the prefix cache's stored sums instead of this request's forward (`P`; `0` on a miss, and always `0` when a file is also written). See "Pooled requests and the prefix cache". |
| `dtype` | Always `float32`. |
| `mean` | Only with `"mean"`/`"both"`: base64 of `[len(layer_ids), hidden]` float32, row-major, little-endian. The mean over **all** prompt positions `[0, prompt_tokens)` -- the same number as mean-pooling the artifact's `hidden_states[:, i]`, exact on a prefix hit too. |
| `mean_suffix` | Accompanies `mean`: same encoding; the mean over the positions this request actually forwarded, `[prefix_tokens, prompt_tokens)`. Equal to `mean` when `prefix_tokens == 0`. |
| `last` | Only with `"last"`/`"both"`: same encoding; the residual at the final prompt position (the artifact's `hidden_states[-1, i]`, i.e. `token_ids[-1]`). The final position is always forwarded (a match never covers it), so this is unaffected by a hit. |

Decode it with

```python
import base64, numpy as np
pooled = response["kv_transfer_params"]["pooled"]
mean = np.frombuffer(base64.b64decode(pooled["mean"]), dtype="<f4").reshape(len(pooled["layer_ids"]), pooled["hidden"])
```

Each vector is ~10 KiB per layer (~560 KiB for all 52 in one pooling), on the response
body rather than on disk. The mean is accumulated in float32 from the bf16 residual,
so it is within float32 summation-order noise of pooling the BF16 artifact client-side
(`tests/server/test_hidden_states_pooled.py` pins the parity, single- and multi-chunk).

Pooling composes with the file: send `pooling` **and** `hidden_states_path` and the
response carries both `hidden_states_path` and `pooled` from one capture -- but then the
file rules apply to the whole request (`--hidden-states-dir` set, contiguous-from-0
`layer_ids`, the token cap, and the full prefix-cache bypass). `max_tokens` may exceed
1; the vectors are pooled from the prefill and the completion is whatever it is. A
pooled request binds no session lease, like the file probe (below). On the stream path
`pooled` rides on the terminal chunk, like `hidden_states_path`.

#### Pooled requests and the prefix cache

A pooled-only request (no `hidden_states_path`) **does** take prefix-cache hits, and
its `mean` stays exact. The hybrid radix cache attaches a GDN state snapshot to the
tree at 128-token boundaries (Mamba-2's scan chunk; 64 on a GDN model); when the
request that produced a snapshot was itself a pooled request, the same node also
stores the float32 sum of the residual stream over every position before that boundary
-- for **all** 52 layers, whatever `layer_ids` that request asked for, so any later
subset can be served (~560 KiB per node, host memory, evicted with the node; a
tombstoned snapshot drops its sums, since a hit needs both). A pooled request matches
only nodes that carry these sums (`scheduler.prefix.pooled_hits`); it inherits the sum
for `[0, P)`, forwards `[P, prompt_tokens)` as usual, and reports:

- `prefix_tokens = P`, `prompt_tokens` = the full prompt length;
- `mean` = the exact mean over `[0, prompt_tokens)` (inherited sum + forwarded sum);
- `mean_suffix` = the mean over the forwarded positions `[P, prompt_tokens)` only;
- `last` = the final prompt position, as always.

Because `P` is a snapshot boundary, not the true common prefix, `mean_suffix` may
include up to 127 tokens that another request would count as prefix (the tokens
between the last boundary and where the prompts diverge). Nodes donated by plain
(non-pooled) requests carry no sums and are simply skipped by a pooled match; a pooled
request that recomputes such a prefix adds the sums to the existing node. A pooled
request that also writes a file keeps the full bypass (`prefix_tokens` is always `0`).
Pinned prefixes (§3a) keep their sums. `usage.prompt_tokens_details.cached_tokens`
(with `--enable-cache-report`) equals `prefix_tokens` for such a request.

### Pooled sink (JSONL)

Start the server with `--pooled-sink-dir DIR` (an existing directory, canonicalized at
startup) and every `pooling` request is **also** appended as one JSON line to
`DIR/<pooled_sink>/pooled.jsonl`, where `<pooled_sink>` is the request's
`kv_transfer_params.pooled_sink` or `default`. The subdirectory is created on demand;
the file is created `0o666 & ~umask` (like the artifact) and each line is appended
under an exclusive `flock`, so a collector may `flock` + truncate/rotate it between
lines, and several servers may share one file. The inline response is unchanged; the
write runs in a worker thread after the engine has answered and **never fails the
response** -- a failed write is a `pooled sink write failed` warning in the server log.

```json
{"model": "nemotron-3.5-lightning", "messages": [...], "max_tokens": 1,
 "kv_transfer_params": {"pooling": "both", "layer_ids": [12, 24, 36, 51], "pooled_sink": "run-2026-09-06"}}
```

Line schema (keys in this order):

| Key | Meaning |
|---|---|
| `ts` | Unix time of the write, float seconds. |
| `request_id` | The response's `id` (`chatcmpl-<uid>`), so a line joins its HTTP response. |
| `session_id` | The FreeToken session lease the turn was bound to (`X-FreeToken-Session-Id`), or `null`. A pooled request binds no lease (see "What a probe request does differently"), so this is `null` today; the key is kept for a future opt-in. |
| `x_switchyard_session_id` | The request's `x-switchyard-session-id` header, stripped, or `null`. This is the conversation identity the router already sends; use it to group lines. |
| `model` | The `model` the client named (echoed, not the served id). |
| `prompt_tokens`, `prefix_tokens`, `layer_ids`, `hidden`, `dtype` | Copied from the response's `pooled` object (`prefix_tokens`: positions served from the prefix cache's pooled sums, `0` on a miss). |
| `mean`, `mean_suffix`, `last` | Copied from `pooled`: base64 float32 `[len(layer_ids), hidden]`, row-major, little-endian; `mean`/`last` present only when requested, `mean_suffix` whenever `mean` is (the mean over the forwarded positions `[prefix_tokens, prompt_tokens)`; equals `mean` on a miss). |
| `prompt_sha256` | Hex SHA-256 of the **rendered chat-template prompt** (UTF-8) -- the string the frontend tokenizer's `render_prompt` produces for this request's messages, tools and `chat_template_kwargs`, i.e. the exact text the worker encodes. Two lines with equal hashes were pooled over the same token sequence. `null` if this server has no frontend tokenizer or the render fails. (Not a hash of token ids: those never reach the API layer.) |

Read it back with `jq`:

```sh
# one row per line: request id, switchyard session, prompt length, layer count
jq -r '[.request_id, .x_switchyard_session_id, .prompt_tokens, (.layer_ids|length)] | @tsv' \
  /var/lib/freetoken/pooled/run-2026-09-06/pooled.jsonl
# decode one line's mean vectors in Python
python - <<'PY'
import base64, json, numpy as np
line = json.loads(open("pooled.jsonl").readline())
mean = np.frombuffer(base64.b64decode(line["mean"]), dtype="<f4").reshape(len(line["layer_ids"]), line["hidden"])
PY
```

Validation, all `400` on `param: kv_transfer_params`: `pooled_sink` without `pooling`;
a name that is not one path segment; a name on a server started without
`--pooled-sink-dir` (`pooled sink is disabled; start the server with --pooled-sink-dir`).
A `pooling` request that names no sink on such a server is served normally and writes
nothing. `tests/server/test_pooled_sink.py` pins all of this.

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

- **A file probe bypasses prefix reuse.** A cached prefix would leave those positions
  out of the forward and therefore out of the artifact. `Req.no_prefix_cache` makes the
  match run against the empty prefix; the completed prompt is still committed to the
  radix tree, so ordinary traffic behind the probe still hits. A pooled-only request
  instead reuses prefixes that carry pooled sums (see "Pooled requests and the prefix
  cache") and donates sums with the snapshots it commits.
- **It binds no session lease.** `x-switchyard-session-id` and `prompt_cache_key` are
  ignored for a probe: a lease protects a prefix for a next turn, and the probe refuses
  to reuse a prefix and has no next turn. This also stops concurrent probes on one
  conversation from serializing on a `session … is busy`.
- **A file probe is capped at `--hidden-states-max-tokens` (default 4096) prompt
  tokens.** A longer prompt is a 400 with `error.code = context_length_exceeded`. The
  cap is a size guard, not a context guard: every layer of every prompt token is
  exported, so one 4096-token probe over 52 layers at hidden 2688 is ~1.1 GiB. The
  check runs frontend side even with `--no-context-preflight`, and again in the
  scheduler. A pooled-only probe is not capped (its state is O(layers x hidden)).
- **It costs nothing when absent.** Without `kv_transfer_params` no sink is installed;
  the model forward reads one attribute and the captured decode graphs never see it.

### Verifying it

CPU: `tests/server/test_hidden_states_probe.py` (wire + validation + writer round trip),
`tests/server/test_hidden_states_pooled.py` (pooled variant: rules, chunked
accumulation, response placement, parity with client-side pooling of the artifact,
prefix hits through the hybrid cache manager: inherited sums, boundary sums,
`prefix_tokens`/`mean_suffix`), `tests/kvcache/radix/test_hybrid_radix_pooled.py`
(sums on the tree: pooled match gating, dedup, tombstone/eviction/split/lock),
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

### First-step logprobs

Opt-in, OpenAI-shaped, first token only: `"logprobs": true` with `"top_logprobs": k`
(0..20) returns the **first sampled token's** logprob and the `k` most likely tokens of
that same step under `choices[0].logprobs`. No later step is populated — the object has
exactly one `content` entry — because the value comes from the final prefill chunk's
logits row, computed eagerly where the sampler runs (prefill is never CUDA-graphed) and
never on a decode step. With thinking on, that first token is the first *reasoning*
token. `logprobs: true` without `top_logprobs` (or `0`) gives the sampled token's
logprob with an empty `top_logprobs`; `top_logprobs > 20` and `top_logprobs > 0`
without `logprobs: true` are 400s. Not requested → `choices[0].logprobs` is `null`.

The distribution is `log_softmax` over the **full vocabulary of the raw logits** in
float32, before temperature / top-p / top-k, so it is the model's own next-token
distribution, comparable across requests with different sampling settings; the sampled
token's `logprob` is read from that same distribution (with sampling on, it need not be
the top entry). `token` is the tokenizer's decoding of the single id (a partial UTF-8
piece decodes to U+FFFD); `bytes` is that string's UTF-8. Prefix reuse is unaffected: a
fully cached prompt still forwards its last token, which is the row scored.

```bash
curl -s http://127.0.0.1:1919/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "nemotron-3.5-lightning",
  "messages": [{"role": "user", "content": "The capital of France is"}],
  "max_completion_tokens": 4, "logprobs": true, "top_logprobs": 2
}'
```

```json
{
  "id": "chatcmpl-7", "object": "chat.completion", "model": "nemotron-3.5-lightning",
  "choices": [{"index": 0, "message": {"role": "assistant", "content": "Paris."},
               "logprobs": {"content": [{
                 "token": "Paris", "logprob": -0.031, "bytes": [80, 97, 114, 105, 115],
                 "top_logprobs": [
                   {"token": "Paris", "logprob": -0.031, "bytes": [80, 97, 114, 105, 115]},
                   {"token": " Paris", "logprob": -3.9, "bytes": [32, 80, 97, 114, 105, 115]}
                 ]}]},
               "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 14, "completion_tokens": 2, "total_tokens": 16}
}
```

On the stream path the same object rides on the **terminal chunk** (the one carrying
`finish_reason`), next to `kv_transfer_params`: the first token's text delta may be held
back by the reasoning parser or a partial stop string, so no earlier chunk is reliably
the first token's. Every other chunk omits `logprobs`. `/v1/completions` keeps
rejecting `logprobs` (400) as before.

---

## 7. Known limitations

| Not supported | Behavior |
|---|---|
| `logprobs` beyond the first token | `logprobs: true` returns the first sampled token only (§6, "First-step logprobs"); `top_logprobs > 20`, or `> 0` without `logprobs: true`, is a 400. Switchyard sends `top_logprobs: 0`, a no-op. |
| `n > 1` | 400 "Only n=1 is supported". Switchyard never sends `n`. |
| `logit_bias`, `function_call` (legacy) | 400. Use `tools`/`tool_choice`. |
| `seed` | Ignored; Switchyard never sends it. |
| Constrained decoding | None — see §4. |
| Multiple upstreams | One GPU, one process: both tiers are the same weights with different `chat_template_kwargs`. |
| `x-switchyard-session-id` in Switchyard's own session stats | Not recorded router-side (Switchyard `docs/known_issues.md`); FreeToken still binds it. |

Everything Switchyard *does* send is supported: `max_completion_tokens` (it never
sends `max_tokens`), `tools`/`tool_choice`/`parallel_tool_calls`, `temperature`,
`top_p`, `stream` + `stream_options`, `reasoning_effort`, `response_format`,
`prompt_cache_key`, `user`, `stop`, `logprobs`/`top_logprobs` (first step, §6), and the
`developer` role (mapped to `system` for
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

## 9. Capturing and replaying traces

Everything in §8 is *synthetic*: a fixed scenario mix, fixed prompt lengths, a closed
16-client loop. That is the right shape for a regression gate — every profile in it is a
failure we have already had — and the wrong shape for "does this change help the traffic we
actually serve". A trace closes that gap: capture what really arrived, then replay it.

### Capturing

```bash
ft serve --model ... --trace-dir /var/tmp/ft-trace          # everything else as in §1
```

One JSON line per **completed** request, appended to
`/var/tmp/ft-trace/trace-<stamp>-<pid>.jsonl` (mode 0600, one file per process, created if
absent). Covers `/v1/chat/completions` and `/v1/completions`, success, error and client
disconnect alike. Off by default.

| Field | Meaning |
|---|---|
| `t` | arrival, epoch seconds — the frontend's, not the scheduler's |
| `route`, `model`, `rid`, `stream` | which endpoint, which model id, the response id |
| `session` | the **resolved** session id (§3), i.e. what the KV lease was bound to |
| `prompt_tokens`, `cached_tokens` | prompt length and prefix-cache hit. `cached_tokens` is the raw count, **not** gated on `--enable-cache-report`: the flag gates the wire, and a trace whose cached fraction depended on a reporting flag could not be replayed |
| `max_tokens`, `sampling` | the *resolved* params (model defaults folded in), plus `tools` as a catalog size, `reasoning_effort`, `response_format` and `chat_template_kwargs` |
| `output_tokens`, `reasoning_tokens`, `finish_reason` | what came back |
| `ttft_ms`, `duration_ms` | time to first token (streaming only — the non-streaming path has no first-token hook and records `null` rather than inventing one), and arrival to completion |
| `status`, `error_code` | `ok` / `error` / `abort`; `abort` is a client disconnect, `error` carries e.g. `context_length_exceeded` |
| `prompt_sha256`, `msg_chain`, `msg_chars`, `msg_roles` | the prompt's shape — see below |

**No prompt text is written.** What is written instead is the *prefix chain*:

```
chain[i] = sha256(chain[i-1] || canonical(message_i))        (16 hex chars kept)
```

Two requests share exactly the first `m` messages iff their chains agree for `m` entries,
which is precisely the boundary the radix prefix cache keys on. With `msg_chars` and
`msg_roles` beside it, a replay can rebuild prompts of the right length, the right role
sequence and the right sharing graph while holding none of the content. `--trace-include-text`
adds the messages themselves, for replaying one's own traffic; the file then carries
everything the clients sent.

Overhead is off the request path: the record is built from values the handler already has and
handed to a writer thread through a bounded queue, exactly as `request_logger` does, and a
full queue drops records rather than applying back-pressure to serving. With the flag absent
the call sites cost one boolean.

### Replaying

```bash
python benchmarks/trace_replay.py --trace /var/tmp/ft-trace \
  --base-url http://127.0.0.1:1919 --model nemotron-3.5-lightning --out replay.json
```

It preserves inter-arrival times (`--speed 2.0` halves them) and session affinity: each
traced session becomes one replay session bound with `x-switchyard-session-id`, and its turns
are issued strictly in order, one at a time. Overlapping the turns of one conversation would
destroy the prefix structure the whole exercise reconstructs.

It prints its own p50/p95/p99 TTFT and latency, tok/s, cached fraction and error rate **beside
the trace's own**, so the comparison is against measured behaviour rather than a synthetic
baseline.

With text stored it replays verbatim. With only hashes it regenerates deterministic filler,
seeded by `chain[i]`, whose length is **a pure function of that message's own `msg_chars`**:

```
words_i = max(1, round(msg_chars[i] * scale))
```

Purity is the load-bearing property. Turn *k+1* contains turn *k*'s messages, so a length that
depended on the request a message sits in would make the same message come out at two lengths
in two turns; the shared prefix would break at message 0 and the replay would run at ~0 %
reuse against a trace that measured 74 %. The price is that `scale` is one global constant —
fitted so the median replayed prompt matches the median traced one, against a
three-probe calibration of the live server's tokenizer (`tokens = a·words + b + c·messages`;
the per-message term is the chat template's per-turn cost, and omitting it mis-sizes long
conversations specifically). Read `prompt_tokens_err_p50/p95` in the output before trusting a
run: above ~10 % the trace mixes prompt kinds (code, CJK, base64) too different for one
constant, and you want `--trace-include-text` or a per-kind split.

`--dry-run` builds every prompt and reports reconstruction fidelity without a server.

### Converting to a CPU-gate profile

```bash
python benchmarks/trace_to_profile.py --trace /var/tmp/ft-trace --out trace.profile.json
python benchmarks/scheduler_replay.py --profile-file trace.profile.json --ticks 4000
```

`scheduler_replay.py`'s five profiles are hand-written geometries. `trace_to_profile.py`
derives a sixth from real traffic — prompt-length quantile buckets as scenarios, median
`cached_tokens/prompt_tokens` as the reuse fraction, median `output_tokens` as the decode
cost, turns-per-session and per-turn growth as the session-residency model, distinct first
messages as the prefix families, and the peak of the arrival/finish interval sweep as the
client population. Pool size, prefill budget and the lane cap are server flags a trace cannot
observe: they are passed through (`--pool-pages`, `--prefill-budget`) and default to the P2
profile's. The result runs on the real `PrefillManager` / `CacheManager` with no GPU and no
model, exactly as the other profiles do.

CPU coverage: `tests/server/test_request_trace.py` (writer, prefix-chain sharing, no text
leak) and `tests/benchmarks/test_trace_replay.py` (capture → replay → metrics against a
stdlib fake server, and trace → profile → a real `scheduler_replay` run).

---

## 10. Troubleshooting

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

---

## 11. Production observability

Everything above answers "did this change help", on traffic we chose. This section is the
other question — "what is the deployment actually doing, right now and last Tuesday" — and
it has exactly three sources. Each is independent, each is cheap, and none of them needs the
GPU box to be doing anything special at the time.

**1. The server log, with the invariant on.** Run the server with
`FREETOKEN_SCHEDULER_INVARIANT=warn` (what `switchyard_soak/serve.sh` exports) and redirect
its output to a file. That gives the per-batch `Prefill batch …` / `Decode batch …` lines and
the pressure markers, which `benchmarks/switchyard_soak/analyze.py <log>` turns into
throughput, occupancy, lanes per prefill batch and the §R7 starvation signature. `warn` makes
a finishability violation say so at the moment it happens; the *count* is published either
way (see `scheduler.prefill.invariant` below), so the env var buys the offending pass's
context, not the fact that it occurred.

**2. `/v1/stats` sampling.** The scheduler counters (`python/freetoken/scheduler/counters.py`)
are cumulative for the server process, so a rate is a difference between two snapshots. The
soak takes two, at its phase boundaries. A production server needs a time series:

```bash
python benchmarks/ops/stats_sampler.py sample --base-url http://127.0.0.1:1919 \
  --interval 60 --out ~/.cache/freetoken/stats/%Y-%m-%d.jsonl
```

One JSON line per sample, appended; the `%`-escapes are expanded per sample, which is what
rotates the file at midnight. The server being down is recorded (`"ok": false`) rather than
fatal — the outage is the interval you most wanted. As a service:

```bash
install -Dm644 benchmarks/ops/freetoken-stats-sampler.service \
  ~/.config/systemd/user/freetoken-stats-sampler.service
systemctl --user daemon-reload && systemctl --user enable --now freetoken-stats-sampler
```

Read it back with per-hour deltas (`--bucket 600` for ten-minute buckets, `--json` for the
raw numbers):

```bash
python benchmarks/ops/stats_sampler.py summarize ~/.cache/freetoken/stats/
```

One row per bucket: completed requests, errors and client disconnects, prefill refusals,
`fresh_admits_deferred`, `fresh_admits_blocked_by_cap`, session spills / restores /
`restores_deferred`, radix match-memo hits, finishability-invariant violations, the MoE decode
expert-cache and extend-cache hit rates computed **over the window** (never a lifetime
average — that is why `counters.py` publishes raw counts), and p50/p95 of the sampled
`requests.p95_ms` and `ttft_mean_ms` gauges. A restart resets the counters, so `uptime_s`
going backwards is detected and the new process is counted from zero instead of producing a
huge negative delta; gauges and high-water marks are excluded from the differencing. Add
`--requests` to also pull `/v1/requests?since=<cursor>` into each sample.

**3. `--trace-dir`.** §9's capture, which is the only one of the three that sees an
individual request. Once you have both a trace and a soak run, ask whether the gate's traffic
resembles the real thing:

```bash
python benchmarks/trace_load_report.py --trace /var/tmp/ft-trace \
  --soak-run benchmarks/switchyard_soak/runs/<tag>
```

It prints the real load's arrival rate and arrivals-vs-completions per minute, peak
concurrency, sessions active per minute, prompt / new (`prompt − cached`) / output token
distributions (p50/p90/p99/max), prefix reuse, the per-route split, session count, turns and
lifetime, TTFT and latency quantiles and the abort/disconnect rate — each beside the soak's
corresponding number with the ratio between them, and a verdict block naming which of the
soak's assumptions hold and which do not ("real p99 prompt is 4.5x the soak's mean prompt").
The soak's numbers come from the run's own `/v1/stats` phase snapshots wherever possible
(the same `_flat` delta `analyze.py` uses, so the two cannot drift) and from the soak
client's flags otherwise; every row says which. It ends with the exact
`trace_to_profile.py` command — `--agents` sized to the measured peak concurrency — that
turns the trace into a `scheduler_replay.py` profile, closing the loop back to §9.

Both tools are stdlib-only and need neither torch nor the venv. CPU coverage:
`tests/benchmarks/test_ops_observability.py` (sampler against a stdlib server serving a real
`build_scheduler_counters` document, restart and outage handling, window-vs-lifetime ratios;
load report over a trace written by the real writer, and a synthetic `runs/<tag>/`).
