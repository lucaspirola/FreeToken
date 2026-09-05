# CLI reference

```
ft <command> [args]
```

| Command | Purpose |
|---|---|
| `ft serve` | Start the API server (OpenAI `/v1/*`, Anthropic `/v1/messages`, Responses) |
| `ft shell` | Chat with a server in the terminal |
| `ft ctl` | Query and manage a running server over HTTP |
| `ft launch` | Configure and launch a coding agent against a server |
| `ft checkpoint` | Convert an HF checkpoint to the FTW fast-load format |
| `ft bench bw` | Benchmark CPU vs PCIe bandwidth to calibrate the MoE backend |

`ft --version` prints the installed version (torch-free; nightly wheels carry a
`+g<sha>` build stamp, tagged releases a bare version). Every command supports
`--help`.

## ft serve

```bash
ft serve --model <path-or-hf-id> [options]
```

`--model` is the only required flag — dtype, attention backend, MoE backend,
MoE cache size, KV capacity, CUDA-graph sizes and the tool-call/reasoning
parsers all resolve automatically from the checkpoint and the GPU.

### Model

| Flag | Default | Meaning |
|---|---|---|
| `--model-path`, `--model` | required | Local dir, HF repo id, or an FTW dir (auto-detected) |
| `--served-model-name` | basename of `--model` | Model id reported by `/v1/models` |

### Server & runtime

| Flag | Default | Meaning |
|---|---|---|
| `--host` | 127.0.0.1 | Bind address |
| `--port` | 1919 | Bind port |
| `--max-running-requests` | 4 | Max concurrently running requests |
| `--elastic-initial-requests` | off | Hybrid-GDN startup capacity; grows to `--max-running-requests` on demand and shrinks after sessions release state |
| `--max-output-tokens` | 32768 | Default output budget for requests that omit one |
| `--max-seq-len-override` | from checkpoint | Max sequence length |
| `--max-prefill-length` | 8192 | Chunked-prefill chunk size in tokens |
| `--cuda-graph-max-bs`, `--graph` | = max running requests | Max batch size captured as CUDA graphs |
| `--decode-log-interval` | 40 | Scheduler status line every N decode steps |

### KV cache & memory

| Flag | Default | Meaning |
|---|---|---|
| `--memory-ratio` | 0.9 | Fraction of free VRAM the engine may use (weights + MoE cache + KV) |
| `--num-pages` / `--num-tokens` | auto | KV capacity override in pages / tokens (mutually exclusive; auto sizes from VRAM left after weights and MoE cache) |
| `--page-size` | 1 | KV page size; DSV4 forces 128, the TRTLLM backend needs 16/32/64, SWA models require 1 |
| `--cache-type` | radix | `radix` (prefix reuse; SWA/GDN-aware variants picked automatically) or `naive` |
| `--attention-backend`, `--attn` | auto | `trtllm`/`fi`/`fa`/`triton`/`dsv4_sparse`/`dsa`; `prefill,decode` pair allowed; auto picks per model + GPU |
| `--session-spill-dir` | auto | Cold storage for idle automatic-agent KV/GDN checkpoints; `off` disables it |
| `--session-spill-ram-gb` | 4 | RAM-first cold-session budget, additionally bounded by the host reserve; sized to hold one look-ahead checkpoint beside the one being written |
| `--session-spill-disk-gb` | 64 | Per-server disk/NVMe cold-session budget |
| `--session-spill-limit-gb` | 50 | Total retained checkpoint bytes (RAM + disk); a spill over the cap evicts least-recently-used checkpoints instead of failing |
| `--session-spill-state-stride` | 65536 | Spacing (tokens) of the extra recurrent-state boundaries a checkpoint carries; a restore resumes at the deepest boundary the client's tokens still match |
| `--session-spill-capture-states` / `--no-...` | auto | Copy the recurrent state to the host every stride of prefilled tokens so a cold restore has cut points; auto = on below 6 state slots per running request (e.g. `--linear-state-slots 5`), costing up to 8 states (~376 MiB on Lightning) of host RAM per resident turn |
| `--session-spill-persist` / `--no-session-spill-persist` | on | Keep checkpoints across restarts (startup adopts manifests matching this model + K/V layout, deletes the rest) or wipe them on exit |
| `--auto-session-grace-seconds` | 0 (off) | Safety-net timer after which an idle auto-bound session may be checkpointed, and only while a request is queued or the pools are full; 0 keeps it resident until an admission needs the space |
| `--host-ram-reserve-gb` | 3 | Minimum `MemAvailable` kept outside expert banks and RAM session checkpoints |

### MoE offload

See [models.md](models.md#moe-backends) for what each backend does.

| Flag | Default | Meaning |
|---|---|---|
| `--moe-backend` | auto | `fused`/`offload`/`cpu`/`hybrid`; auto → offload, or hybrid with a `ft bench bw` profile |
| `--moe-cache-size` / `--moe-cache-rate` / `--moe-cache-auto` | auto | GPU expert-cache size as slots / fraction of all experts / sized from free VRAM (mutually exclusive; auto is enabled by default for offload-family backends) |
| `--kv-reserve-tokens` | 8192 | KV token floor reserved before `--moe-cache-auto` fills experts |
| `--moe-cpu-threads` | physical cores | CPU worker threads for the cpu/hybrid executor |
| `--moe-cpu-layers` | all on GPU | With `offload`: which MoE layers decode on CPU (`3,7,11`, a count, or a fraction) |
| `--moe-pageable-gpu` | off | On WSL pin-quota overflow, asynchronously gather selected misses into mapped pinned staging; all expert math and CUDA graph replay remain on GPU |
| `--moe-pageable-profile` | off | Persistent pageable-layer policy: `off` uses the deterministic built-in placement, `read` applies an existing model-scoped profile, and `train` also updates it from telemetry |
| `--moe-hybrid-max-fetch` | auto | With `hybrid`: max experts fetched over PCIe per layer per step; rest computed on CPU |
| `--moe-prefill-hit-d2d` | off | Prefill: copy cache-hit experts device-side, stream only misses (CUDA >= 13) |
| `--disable-moe-prefill-overlap` | overlap on | Disable the two-buffer prefill copy overlap |

### API behaviour

| Flag | Default | Meaning |
|---|---|---|
| `--sampling-defaults` | model | Fill unspecified sampling params from the checkpoint's `generation_config.json` (`none` = framework defaults) |
| `--tool-call-parser` | auto | Tool-call format; auto-inferred from the model family |
| `--reasoning-parser` | auto | Splits chain-of-thought into `reasoning_content`; auto-inferred; `off` disables |
| `--enable-cache-report` | off | Report prefix-cache hits in each response's usage block |
| `--no-context-preflight` | off | Skip the frontend tokenize-and-check that returns HTTP 400 `context_length_exceeded` before a prompt is queued |
| `--force-nonempty-content` | off | When a turn ends with empty content and no tool call, move the reasoning text into `content` (coding agents, Switchyard) |
| `--json-retry` | 1 | Greedy repair resubmissions when a `response_format` json answer is not valid JSON; 0 disables |
| `--hidden-states-dir` | off | Enable the Switchyard prefill-probe export and make this directory its only permitted root; a request opts in per call with top-level `kv_transfer_params` and gets the artifact path back ([switchyard.md](switchyard.md#6-hidden-state-probe-target)) |
| `--hidden-states-max-tokens` | 4096 | Prompt-token cap for one hidden-state probe; a longer prompt is a 400 `context_length_exceeded` (the artifact is tokens x layers x hidden) |
| `--trace-dir` | off | Append one JSON line per completed request here (arrival, session, route, token counts, sampling, TTFT, abort) for `benchmarks/trace_replay.py`; the prompt is stored as a hash chain, never as text ([switchyard.md](switchyard.md#9-capturing-and-replaying-traces)) |
| `--trace-include-text` | off | Also write the prompt messages into the trace; only for replaying one's own traffic. Requires `--trace-dir` |

## ft shell

```bash
ft shell                                    # attach to a running server
ft shell --model ~/models/Qwen3.6-35B-A3B   # serve + chat in one process
```

- Attach mode talks to `--server URL` (default `http://127.0.0.1:1919`)
- `/help` inside the shell lists the commands (`/think`, `/cache`, `/reset`).

## ft ctl

```bash
ft ctl [--base-url http://127.0.0.1:1919] [--timeout 10] [--json] <subcommand>
```

| Subcommand | Endpoint | Purpose |
|---|---|---|
| `health` | `GET /health` | Server status, model, load progress |
| `stats` | `GET /v1/stats` | Throughput, latency, VRAM, pool occupancy |
| `generate [prompt] [--max-tokens N] [--ignore-eos]` | `POST /generate` | Raw completion smoke test (no chat template) |
| `cache` | `GET /v1/cache/status` | Cache pool table |
| `cache --moe N \| --kv N \| --mamba N \| --swa N [--wait 300]` | `POST /v1/cache/rebuild` | Live pool resizing without a restart (`k`/`m` suffixes; `--kv`/`--swa` in tokens) |
| `requests [--since N] [--limit N]` | `GET /v1/requests` | Recent request ring |

## ft launch

```bash
ft launch {claude,codex,dsh,hermes,openclaw,opencode} [options] [-- <agent args>]
```

Discovers the served model via `/v1/models`, writes the agent's provider
config, installs the agent CLI if missing, then launches it. Cloud API keys
(`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, …) are cleared from the child
environment so the agent cannot silently fall back to a paid endpoint.

| Flag | Meaning |
|---|---|
| `--server URL` | Server to point the agent at (default `http://127.0.0.1:1919`) |
| `--dry-run` | Print the planned config changes and command, touch nothing |
| `-y`, `--yes` | Approve install/config prompts |
| `--config` | Configure without launching |
| `--install-only` | Just install the agent CLI (needs no server) |
| `--force-reinstall` | Re-run the agent installer |
| `-- <args>` | Forwarded verbatim to the agent |

## ft checkpoint

```bash
ft checkpoint --model <hf_dir> --out <ftw_dir> [--dtype bfloat16] [--moe-backend offload] [--shard-gib 8] [--device cuda:0]
```

Converts an HF safetensors checkpoint to FTW, FreeToken's self-contained
fast-load format; point `ft serve --model` at the output dir. `--moe-backend
offload` (default) packs experts into offload banks; `--moe-backend triton`
keeps them dense for resident serving. See the FTW caveats in
[models.md](models.md#notes).

## ft bench bw

```bash
ft bench bw                       # once per machine
ft bench bw --dtype nvfp4,bf16    # only the formats you serve
```

Measures host-RAM vs PCIe bandwidth with the real cpu/offload MoE kernels and
writes a profile (`~/.cache/freetoken/benchbw.json`) that `ft serve
--moe-backend auto` and `--moe-hybrid-max-fetch -1` read. Profiles are keyed on
expert format + GPU name, so a profile from different hardware is ignored
rather than misapplied. Selection flags: `--dtype`, `--model`, `--formats`,
`--isa`; decision rule: `--threshold` (default 2.0 — recommend hybrid when CPU
bandwidth > 2× PCIe).
