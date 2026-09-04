# Nemotron 3.5 Lightning on a 5080 — Switchyard end-to-end soak (task 3D)

2026-09-04 · FreeToken at `ec54e21` (this task changed only `docs/switchyard.md` and
`scripts/switchyard_e2e.py`; every file on the crash path below is pristine at
`1184c4d`) · Switchyard `switchyard-server` / `switchyard-soak` built 2026-09-04
01:21 · RTX 5080 16 GiB, WSL2, 33 GiB host RAM, GPU held exclusively through
`scripts/gpu_lock.sh` for the whole server lifetime.

## Launch

The P2 profile plus the Switchyard serving-compliance flags, with
`FREETOKEN_PIN_BUDGET_GB=17`:

```
ft serve --model ~/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --host 127.0.0.1 --port 1919 \
  --max-running-requests 16 --elastic-initial-requests 4 --kv-grow-step-tokens 65536 \
  --num-tokens 262144 --max-seq-len-override 131072 --kv-cache-dtype q8_0 \
  --attention-backend triton --moe-backend offload --moe-pageable-gpu --moe-cache-auto \
  --memory-ratio 0.85 --max-prefill-length 8192 --host-ram-reserve-gb 3 \
  --enable-cache-report --served-model-name nemotron-3.5-lightning \
  --reasoning-parser nemotron_v3 --tool-call-parser qwen3_coder \
  --force-nonempty-content --max-output-tokens 16384
```

At `FREETOKEN_PIN_BUDGET_GB=17` the loader reports *"expert banks fit the CUDA pin
budget; keeping every layer on the direct pinned RAM -> GPU path"* — no pageable
overflow, decode graphs kept. Ready in **22 s** (weights 2.9 s, 18.3 GB of expert
banks 8 s, KV 262,144 tokens = 0.80 GiB, 2.19 GiB free VRAM after init, graphs
captured at bs 1–4). `GET /v1/models` advertises `max_model_len = 131072`.

## Verdict per step

| Step | Result |
|---|---|
| 1 — wire contract | **PASS** 12/12 |
| 2 — soak `switchyard/passthrough`, 10 m, c=16 | **FAIL** — backend scheduler crashed at t≈3 min |
| 2 — soak `switchyard/stage`, 10 m, c=16 | **FAIL** — same crash at t≈6 min |
| 3 — resilience set, 10 m, c=16 | **PASS** — 267 requests, 0 errors |
| 4 — agent smoke (Claude Code, Codex) | **PASS** on the assertion (exit 0, non-empty text); neither agent actually executed a tool |

## 1. Wire contract — 12/12

```
scripts/switchyard_e2e.sh contract --base-url http://127.0.0.1:1919 --model nemotron-3.5-lightning
```

All twelve checks pass: served id + 131,072 window; `max_completion_tokens` honored
(16/16); `reasoning_content` 1,276 ch + `reasoning_tokens` 476; `cached_tokens`
0 → 1,920 on a repeat; schema-valid `EscalationVerdict` in both stream and
non-stream JSON mode; `x-switchyard-session-id` → a stable
`X-FreeToken-Session-Id` across two turns and on the stream; `finish_reason =
tool_calls` with parseable arguments both non-stream and reassembled from SSE
deltas; overflow → HTTP 400 `context_length_exceeded`, rejected by the frontend
preflight before the stream opens.

The first run scored 10/12 on the two overflow checks. That was a **bug in the
check, not the server**: it built its filler as `"token " * ((served_max+4096)//2)`,
and one repetition of `"token "` is *one* token, so the "oversize" prompt was
67,601 tokens — comfortably inside the window, and the 200 the server returned was
correct. Verified directly:

| prompt | result |
|---|---|
| `"token " * 67584` | HTTP 200, `prompt_tokens = 67601` |
| `"token " * 135168` | HTTP 400, `code = context_length_exceeded`, 135,185 tokens |

`scripts/switchyard_e2e.py` now sizes the filler by token count.

## 2. Soak through the router — both routes crash the backend

`switchyard-soak`, concurrency 16, `--max-output-tokens 256 --prompt-bytes 16384
--context-window-tokens 131072 --max-error-rate 0 --request-timeout 600`, scenarios
`prefix-reuse, growing-conversation, tool-call-burst, large-tool-catalog,
long-context`.

| route | requests | ok | errors | error rate | p50 | p95 | p99 |
|---|---|---|---|---|---|---|---|
| `switchyard/passthrough` | 199 | 183 | 16 | 8.04 % | 13,005 ms | 600,002 ms | 616,431 ms |
| `switchyard/stage` | 113 | 97 | 16 | 14.16 % | 39,446 ms | 600,002 ms | 600,002 ms |

Every error is `timeout`, and in both runs all sixteen are the *same* sixteen: the
requests in flight when the server stopped answering. The interval log shows the
shape clearly (passthrough):

```
 60s reqs=57  errors=0  rps=0.95   p95=23,559 ms  health=ok  status=OK
120s reqs=137 errors=0  rps=1.333  p95=23,886 ms  health=ok  status=OK
180s reqs=183 errors=0  rps=0.767  p95=14,813 ms  health=ok  status=OK
240s reqs=183 errors=0  rps=0      p95=—         health=ok  status=STALLED
...  (six more STALLED intervals)
753s reqs=199 errors=16(8.04%)     p95=616,431 ms health=ok  status=DEGRADED
```

### Server-side bug: `LinearStatePool exhausted` kills the scheduler

Reproduced twice, once per route, with an identical stack. FreeToken log
(`freetoken-TP0-scheduler`):

```
Prefill batch, #new-seq: 4, #new-token: 4982, #cached-token: 7306, token usage: 0.31,
  #mamba-slot: 96/96, mamba usage: 1.00, #running-req: 12, #queue-req: 4
Process freetoken-TP0-scheduler:
Traceback (most recent call last):
  ...
  File "python/freetoken/scheduler/scheduler.py", line 591, in normal_loop
    self._process_last_data(ongoing_data)
  File "python/freetoken/scheduler/scheduler.py", line 715, in _process_last_data
    self.cache_manager.cache_req(req, finished=False)
  File "python/freetoken/scheduler/cache.py", line 561, in cache_req
    return self._cache_req_hybrid(req, finished=finished)
  File "python/freetoken/scheduler/cache.py", line 706, in _cache_req_hybrid
    pp[frozen_idx] = pool.alloc(1)[0]
  File "python/freetoken/kvcache/linear_state_pool.py", line 104, in alloc
    raise RuntimeError(
RuntimeError: LinearStatePool exhausted: need 1, have 0
ERROR  Backend supervisor: backend worker freetoken-TP0-scheduler exited
ERROR  Backend worker is gone and cannot be restarted; stopping the API server
```

Both crashes happen on the log line where `mamba usage` first reaches **1.00
(96/96 slots)**. The path is the prefill-chunk commit in `_cache_req_hybrid`: after
donating the frozen Mamba-2 snapshot to the radix tree it must hand the request a
replacement ping-pong slot. `ensure_mamba_slots(1)` is best-effort — it `break`s
when `evict_mamba` can free nothing because every snapshot is locked or live — and
the following `pool.alloc(1)` then raises unconditionally, taking the whole
scheduler process down. There is no request-level fallback (skip the donate, keep
the existing slot, defer the commit), so a *cache-management* step turns a
transient resource shortage into a fatal process exit.

Trigger, both times: `--max-running-requests 16` with prefix-cache donates in
flight. The pool cannot simply be made bigger on this card — the Mamba-2 state is
47 MiB per slot, so `--linear-state-slots 256` is refused at startup (*"cache budget
too small: minimum plan needs 1.69 GB > budget −1.80 GB"*), and
`--linear-state-slots` is rejected outright alongside `--elastic-initial-requests`.
96 slots is what a 16 GiB card affords at 16-way concurrency, and 16-way concurrency
reaches it.

Two follow-on defects make the crash worse than a crash:

1. **`/health` keeps answering `{"status":"ok"}` after the backend is gone.** All 10
   soak liveness checks passed during the nine-minute stall (`health_failures = 0`,
   `metrics_failures = 0`, `detected_server_restarts = 0`), so a router or
   supervisor polling `/health` sees a healthy upstream that answers nothing.
2. **The process never exits.** After *"stopping the API server"* the frontend logs
   *"Waiting for background tasks to complete"* and hangs there indefinitely — 38
   minutes observed, port closed, VRAM and the 18.3 GB of expert banks still held,
   `wchan = do_epoll_wait`. It had to be `SIGKILL`ed. A systemd/supervisor restart
   policy keyed on process exit will never fire.

In-flight requests are neither failed nor answered; they hang until the client's own
timeout (600 s here).

## 3. Resilience set — PASS

```
scripts/switchyard_e2e.sh soak --duration 10m --scenario-set resilience \
  --route switchyard/passthrough --max-error-rate 0.5
```

| requests | ok | errors | p50 | p95 | p99 |
|---|---|---|---|---|---|
| 267 | 267 | 0 | 34,801 ms | 121,780 ms | 130,322 ms |

Per scenario: `context-overflow` 95, `failure-pressure` 90, `client-cancellation`
82 — all successful; the invalid-request canary passed 1/1; no liveness, metrics or
process check failed. The `--max-error-rate 0.5` allowance turned out to be unused:
`failure-pressure` and `client-cancellation` inject their faults with
`[scenario:upstream_500]`-style *markers in the prompt text*, which only a mock
upstream interprets. Against a real FreeToken upstream they are ordinary prompts, so
the documented `ErrorExpectation::MIXED` / `::ALL` never materializes and the
default `--max-error-rate 0` would also have passed. `context-overflow` sends
0.9 × 131,072 ≈ 118 K-token prompts, which is what puts p95 above two minutes.

## 4. Agent smoke tests

Both agents were driven non-interactively against `switchyard-server` on
`localhost:4000`, from a scratch directory holding a three-line `README.md`,
`sorter.py` and `widgets.csv`.

**Claude Code** — `ANTHROPIC_BASE_URL=http://localhost:4000
ANTHROPIC_MODEL=switchyard/stage claude -p "list the files in this directory and
summarize README.md in two sentences"`: **exit 0**, `is_error = false`,
`num_turns = 1`, 3.3 s, non-empty result. But the result is *"Let me first explore
the current directory to see what files are available, and then read the README.md
file."* — the model narrated the tool call in prose and stopped instead of emitting
one, so no file was ever read. An earlier run of the same prompt produced a longer
answer that likewise contained a `bash` block as text and a summary of a README that
does not exist in that directory.

**Codex** — `codex exec --config model_provider=openai-custom --config
model=switchyard/passthrough` (`wire_api = "responses"`; the built-in `openai`
provider id cannot be overridden, and `wire_api = "chat"` is no longer supported by
codex 0.153.2; stdin must be `/dev/null` or `codex exec` blocks reading it):
**exit 0**, non-empty final text — but the text is a raw, unparsed Qwen3-Coder tool
call:

```
<function=exec_command>
<parameter=cmd>
ls -la
</parameter>
...
</function>
</tool_call>
```

Note the closing `</tool_call>` with no opening tag: the model emitted a malformed
envelope, which is why `--tool-call-parser qwen3_coder` did not lift it into
`tool_calls`. 20,177 tokens, one router request, HTTP 200.

The transport is **not** at fault. A direct Anthropic-format probe through the same
router returns a real tool call:

```
POST http://127.0.0.1:4000/v1/messages  {"model":"switchyard/passthrough", ...,
  "tools":[{"name":"list_files", "input_schema":{...}}]}
-> stop_reason = "tool_use", content block types = ['thinking','text','tool_use']
```

So Switchyard's `/v1/messages` → `openai_chat` translation, FreeToken's
`qwen3_coder` parser and the reasoning split all work; what fails is the 30B model's
own tool-emission reliability inside a real agent's much larger system prompt.

## 5. Recorded numbers

### TTFT / ITL (single stream, 24 K-token prompt, `enable_thinking: false`)

| arm | TTFT | `cached_tokens` |
|---|---|---|
| cold (first sight of the prefix) | 4,115 ms | 0 / 24,024 |
| warm repeat #1 | 887 ms | 23,936 |
| warm repeat #2 | 390 ms | 23,936 |
| warm repeat #3 | 379 ms | 23,936 |
| shared prefix + different suffix, ×4 | 880 / 399 / 402 / 393 ms | 23,936 |

Prefix reuse is worth **10.3×** on TTFT (4,115 → 399 ms) and recovers 99.6 % of the
prompt. Median inter-token latency on the cold, 64-chunk streams was
**7.2–7.8 ms** (≈ 130–139 tok/s single-stream decode); warm streams returned their
short answer as a single content delta, so ITL is not measurable there.

Under the resilience load (concurrency 16, 118 K-token prompts) the server's own
aggregates were `ttft_mean_ms = 13,245` and `p95_ms = 120,412`.

### `cached_tokens` in a growing conversation

Six turns on one `x-switchyard-session-id`, ~1,027 tokens added per turn:

| turn | prompt_tokens | cached_tokens |
|---|---|---|
| 0 | 1,033 | 0 |
| 1 | 2,060 | 1,036 |
| 2 | 3,087 | 2,063 |
| 3 | 4,114 | 3,090 |
| 4 | 5,141 | 4,117 |
| 5 | 6,168 | 5,144 |

Strictly monotonic, and `X-FreeToken-Session-Id` was the same
`auto:switchyard:81efddc8…` on all six turns.

**One unexplained observation.** In one sequence, four consecutive requests sharing
a 24,576-token prefix reported `cached_tokens = 0` every time (≈3.9 s each), while an
equivalent-size prompt reused 23,936 tokens minutes earlier, and the *same* prefix
reused normally in a later sequence. The prefill log for the failing sequence shows
`#cached-token: 0` on every chunk — the prefix was never committed. This happened
while the elastic Mamba pool was still small (`#mamba-slot: 4/24`); a hybrid prefix
node is only reusable when its Mamba snapshot survives, so snapshot eviction is the
likely cause. Not root-caused; it is a reuse *miss*, not a wrong answer.

### `session … is busy`

**Zero** occurrences across all three server generations (the two crashed soaks and
the resilience run) — including the stage route, whose classifier calls land on the
same conversation as the turn they grade.

### `/v1/stats`

| snapshot | uptime | completed | p95_ms | ttft_mean_ms | prompt tok | completion tok | KV pages | Mamba slots | VRAM |
|---|---|---|---|---|---|---|---|---|---|
| before (fresh) | 0 s | 0 | 0 | 0 | 0 | 0 | — | — | 0 |
| after contract + probes | 1,229 s | 41 | 5,978 | 2,909 | 844,435 | 1,021 | 69,133 / 131,072 | 4 / 24 | 4.25 GB |
| after resilience soak | 649 s | 267 | 120,412 | 13,245 | 11,214,535 | 61,166 | 126,446 / 262,144 | 50 / 96 | 8.03 GB |

Both elastic pools grow as documented: KV 131,072 → 262,144 pages and the Mamba
pool 24 → 96 slots under load. The counters reset per server generation, so the
crashed passthrough run's terminal stats could not be read — its process had already
stopped serving.

## Pass criteria for Phase 3

| criterion | result |
|---|---|
| 0 request errors | **NO** — 16 timeouts per soak, caused by the scheduler crash |
| prefix-reuse TTFT lower on shared prefixes | **YES** — 4,115 ms → 399 ms, 10.3× |
| `cached_tokens` monotonic within a growing conversation | **YES** — 0 → 5,144 over six turns |
| no unhandled `session is busy` | **YES** — zero occurrences |

Phase 3D is **blocked on the `LinearStatePool exhausted` crash**. Everything the
router itself depends on — the wire contract, session binding, JSON mode, tool
translation, overflow signalling, prefix reporting — is correct; the server simply
does not survive ten minutes of sixteen-way agentic traffic on this card.

## Changes made by this task

- `scripts/switchyard_e2e.py`: size the overflow filler by token count (it was half
  the needed length, so the check failed against a correct server); accept
  `--base-url` / `--model` / `--router-port` / `--switchyard-dir` *after* the
  subcommand, which is the spelling `docs/switchyard.md` documents and which
  previously errored; pass `--request-timeout` (default 600 s) through to
  `switchyard-soak`, whose 120 s default is shorter than a queued 118 K-token
  prefill on this GPU.
- `docs/switchyard.md`: document `--request-timeout`, and add the
  `status=STALLED` + `health=ok` symptom to the troubleshooting table.
