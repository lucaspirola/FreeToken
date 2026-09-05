"""Single-stream (bs=1) decode benchmark for any MoE model on any offload backend.

Measures through the real serving path: for each backend the bench spawns ``ft serve``,
sends a warmed chat request over /v1/chat/completions with ``stream=true``, and
timestamps every SSE event as it arrives. Numbers therefore include the scheduler,
detokenizer, and HTTP/SSE hop -- what a client actually sees -- not bare engine forwards.

Method -- at bs=1 the server emits one delta event per decode step, and the final chunk
(``stream_options.include_usage``) reports exact token counts, so

    decode_tok_s = (completion_tokens - 1) / (t_last_event - t_first_event)

which stays correct even when the detokenizer coalesces a few tokens into one event
(multibyte characters): the window is still anchored on the first and last token's
arrival. ``ignore_eos`` keeps the step count at exactly ``D`` regardless of sampling.
TTFT is the measured run's warm first-token latency (template rendering + prefill
included). Engine-internal diagnostics (expert-cache miss rate, hybrid fetch split) are
not exposed over the API and are not reported; VRAM is the server's live /v1/stats figure.

Prompt: an AIME-25 problem sent as a chat message with thinking enabled -- a real
reasoning workload, so expert routing is representative. The server renders the chat
template (including checkpoint-shipped encoders like DSV4's ``encoding_dsv4.py``). The
problems come from the ``math-ai/aime25`` dataset on the Hub, downloaded into the usual
HF cache on first run; ``--aime`` points at a local jsonl instead.

Sampling: the checkpoint's recommended params (``generation_config.json``), falling back
to temperature 1.0 / top_p 0.95 / top_k 64 for fields the checkpoint does not specify --
resolved here and sent explicitly, because the server's own unspecified-field defaults
are greedy and would silently degrade the routing workload for checkpoints without a
full sampling recommendation. The generated text is per-server-process deterministic
(fresh server, fixed request sequence), so one text sha1 per backend is a real
cross-backend check; token ids are not visible over the API, so this is a weaker
invariant than the old in-process id hash. ``--greedy`` sends temperature 0 for the
stricter comparison.

MoE stats: ``--moe-collect-stats`` turns on the engine's per-layer decode hit/miss
counters and scrapes the scheduler's idle-boundary dump out of the server log into the
JSON row. Those counters are cumulative from server start, so the bench snapshots them
after the warm-up generation and again after the measured one and reports the difference
(``moe_stats_window: "measured"``) -- a single dump would average the cold cache in and
roughly double the apparent miss rate. Both raw snapshots are kept under
``moe_stats_cumulative``.

Concurrency: ``--concurrency N`` runs N streams at once, each a distinct AIME-25 problem
(N consecutive problems from ``--problem``, wrapping), so the offload cache sees N
independent routes rather than N copies of one. Per-stream decode tok/s is reported as
median and min; the aggregate is total decoded steps across streams divided by the wall
window from the first token event of any stream to the last token event of any stream.

Run (one backend):
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python python benchmarks/bench_decode_moe.py \
        --model /path/to/model

Run (all three backends, one server per backend):
    ... --model /path/to/model --backend offload,cpu,hybrid --json out.json

Run (offload-cache sizing study: 4 concurrent streams with per-layer miss stats):
    ... --model /path/to/model --concurrency 4 --cache-rate 0.35 --moe-collect-stats \
        --nvfp4-backend triton --server-arg "--host-ram-reserve-gb 3" --json out.json
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import statistics
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

# Applied for every field the checkpoint's generation_config.json does not specify.
FALLBACK_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 64}

# AIME-25 problems, pulled from the Hub into the usual HF cache on first run.
AIME_REPO = "math-ai/aime25"
AIME_FILE = "test.jsonl"
# Reasoning models need the answer format spelled out; the boxed answer is also what makes
# a run spot-checkable by eye.
BOXED_INSTRUCTION = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

# Verbatim from the server's --nvfp4-backend choices (freetoken/server/args.py).
NVFP4_BACKENDS = ("auto", "marlin", "flashinfer", "triton")

# Emitted by the scheduler's run_when_idle when --moe-collect-stats is on. The dict is a
# python repr (ast.literal_eval); the per-layer list is json.dumps (json.loads).
MOE_STATS_PREFIX = "MoE decode miss stats: "
MOE_STATS_PER_LAYER_PREFIX = "MoE decode miss stats per layer: "


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="checkpoint dir (or .ftw)")
    p.add_argument(
        "--backend",
        default="offload",
        help="comma list of offload|cpu|hybrid; one server per backend",
    )
    p.add_argument(
        "--aime",
        default=os.environ.get("FREETOKEN_AIME25_JSONL"),
        help=f"local jsonl instead of downloading {AIME_REPO}; default $FREETOKEN_AIME25_JSONL",
    )
    p.add_argument("--problem", type=int, default=0, help="0-based AIME problem index")
    p.add_argument(
        "--warm-problem",
        type=int,
        default=None,
        help="optional different 0-based problem used only to warm the expert cache",
    )
    p.add_argument("--decode", type=int, default=256, help="decode tokens to measure (D)")
    p.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="streams to run at once; each gets its own AIME problem (consecutive from "
        "--problem, wrapping). N>1 also sets the server's --max-running-requests and "
        "--cuda-graph-max-bs to N",
    )
    p.add_argument(
        "--cache",
        type=int,
        default=0,
        help="GPU expert cache slots; 0 = auto-size from free VRAM",
    )
    p.add_argument("--cache-rate", type=float, default=None, help="cache slots as a fraction of L*E")
    p.add_argument(
        "--cache-policy",
        choices=("lru", "lfu"),
        default="lru",
        help="expert-cache eviction policy",
    )
    p.add_argument(
        "--hybrid-fetch",
        type=int,
        default=-1,
        help="hybrid: max PCIe fetches/layer; -1 = auto (benched pcie/cpu bandwidth fraction)",
    )
    p.add_argument("--mem-ratio", type=float, default=0.9, help="target VRAM utilization")
    p.add_argument("--no-graph", action="store_true", help="eager decode instead of CUDA graph")
    p.add_argument(
        "--greedy",
        action="store_true",
        help="force temperature 0 (ignore the checkpoint's sampling) so ids are comparable",
    )
    p.add_argument(
        "--server-timeout",
        type=float,
        default=1800,
        help="seconds to wait for the spawned server to become ready",
    )
    p.add_argument(
        "--max-context",
        type=int,
        default=None,
        help="server --max-seq-len-override AND --num-tokens, for a full-context run "
        "(e.g. a long-context Ornith/Laguna session); default keeps the prior "
        "8192 + --decode sizing with the server's own --num-tokens default",
    )
    p.add_argument(
        "--kv-cache-dtype",
        default=None,
        help="server --kv-cache-dtype (auto|q8_0|fp8_e4m3|int4|q4_0); default leaves the "
        "server's own default (auto, unquantized) in place",
    )
    p.add_argument(
        "--prefill-chunk",
        type=int,
        default=None,
        help="server --max-prefill-length; default leaves the server's own default chunk size",
    )
    p.add_argument(
        "--prefill-hit-d2d",
        action="store_true",
        help="pass --moe-prefill-hit-d2d to the server (off by default, matching the server default)",
    )
    add_server_passthrough_args(p)
    p.add_argument("--json", dest="json_out", default=None, help="append the result rows here")
    return p.parse_args(argv)


def add_server_passthrough_args(p: argparse.ArgumentParser) -> None:
    """Server flags this bench forwards untouched; shared with bench_long_context.

    Every one is opt-in: absent, ``serve_cmd`` emits nothing and the server keeps its own
    default."""
    p.add_argument(
        "--nvfp4-backend",
        choices=NVFP4_BACKENDS,
        default=None,
        help="server --nvfp4-backend (routed-expert GEMM); default leaves the server's own",
    )
    p.add_argument(
        "--moe-collect-stats",
        action="store_true",
        help="pass --moe-collect-stats to the server (bench_decode_moe additionally scrapes "
        "the scheduler's per-layer decode miss stats out of the log into its JSON row)",
    )
    p.add_argument(
        "--server-arg",
        action="append",
        default=None,
        metavar="ARGS",
        help="extra flags appended verbatim to the ft serve command, split on whitespace "
        "(repeatable), e.g. --server-arg '--host-ram-reserve-gb 3'",
    )


def load_problem(path: str | None, index: int) -> tuple[str, str]:
    """One AIME-25 (problem, answer)."""
    return load_problems(path, index, 1)[0]


def load_problems(path: str | None, index: int, count: int = 1) -> list[tuple[str, str]]:
    """``count`` consecutive AIME-25 (problem, answer) pairs from ``index``, wrapping.

    Downloads the dataset unless ``path`` overrides it. Accepts both the Hub schema
    (``problem``) and the pre-formatted jsonl some local copies use (``prompt``, answer
    instruction already appended)."""
    if not path:
        from huggingface_hub import hf_hub_download

        try:
            path = hf_hub_download(AIME_REPO, AIME_FILE, repo_type="dataset")
        except Exception as e:  # offline, rate-limited, repo moved
            sys.exit(f"could not fetch {AIME_REPO}/{AIME_FILE} ({e}); pass --aime <local jsonl>")
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not 0 <= index < len(rows):
        sys.exit(f"--problem {index} out of range ({len(rows)} problems available)")
    out = []
    for i in range(max(1, count)):
        row = rows[(index + i) % len(rows)]
        text = row.get("problem") or row["prompt"]
        if "boxed" not in text:
            text = f"{text}\n{BOXED_INSTRUCTION}"
        out.append((text, str(row.get("answer", ""))))
    return out


def resolve_sampling(model_path: str, greedy: bool) -> tuple[dict, str]:
    """Checkpoint-recommended sampling with per-field fallback; returns (params, source).

    Resolved client-side and sent explicitly: the server fills unspecified fields with
    its framework defaults (temperature 0 / no filtering), not with these fallbacks."""
    if greedy:
        return {"temperature": 0.0, "top_p": 1.0, "top_k": -1}, "greedy (--greedy)"
    recommended: dict = {}
    cfg = Path(model_path) / "generation_config.json"
    if cfg.is_file():
        raw = json.loads(cfg.read_text())
        recommended = {k: raw[k] for k in FALLBACK_SAMPLING if raw.get(k) is not None}
        if raw.get("do_sample") is False or recommended.get("temperature") == 0.0:
            return {"temperature": 0.0, "top_p": 1.0, "top_k": -1}, "checkpoint (greedy)"
    params = {**FALLBACK_SAMPLING, **recommended}
    if params["top_k"] == 0:
        params["top_k"] = -1  # HF spells "no top-k filtering" as 0; the API as -1
    taken = sorted(recommended)
    source = f"checkpoint{taken} + fallback" if taken else "fallback (no generation_config)"
    return params, source


def get_json(url: str, timeout: float = 10) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.load(resp)


def free_port() -> int:
    # FreeToken uses the API port plus the adjacent port for torch.distributed.
    # Probe the pair together; checking only the API port makes otherwise valid
    # benchmark runs fail nondeterministically with EADDRINUSE during startup.
    while True:
        with socket.socket() as api, socket.socket() as distributed:
            api.bind(("127.0.0.1", 0))
            port = api.getsockname()[1]
            if port == 65535:
                continue
            try:
                distributed.bind(("127.0.0.1", port + 1))
            except OSError:
                continue
            return port


def concurrency_of(args: argparse.Namespace) -> int:
    """Stream count, tolerating namespaces built before --concurrency existed."""
    return max(1, int(getattr(args, "concurrency", 1) or 1))


def serve_cmd(args: argparse.Namespace, backend: str, port: int) -> list[str]:
    max_seq_len = args.max_context if args.max_context is not None else 8192 + args.decode
    # Read every optional attribute with getattr: bench_long_context imports this module
    # and calls serve_cmd with its own namespace, which need not carry the newer options.
    streams = concurrency_of(args)
    cmd = [
        sys.executable, "-m", "freetoken.cli", "serve",
        "--model", args.model,
        "--host", "127.0.0.1", "--port", str(port),
        "--moe-backend", backend,
        "--max-running-requests", str(streams),
        "--max-seq-len-override", str(max_seq_len),
        "--memory-ratio", str(args.mem_ratio),
        "--cuda-graph-max-bs", "0" if args.no_graph else str(streams),
        "--moe-hybrid-max-fetch", str(args.hybrid_fetch),
        "--moe-cache-policy", args.cache_policy,
    ]
    # Every flag below is opt-in and omitted unless passed, so a bare invocation keeps
    # the server's own defaults exactly as before this option set existed.
    if args.max_context is not None:
        cmd += ["--num-tokens", str(args.max_context)]
    if getattr(args, "kv_grow_step_tokens", None):
        cmd += ["--kv-grow-step-tokens", str(args.kv_grow_step_tokens)]
    if args.kv_cache_dtype is not None:
        cmd += ["--kv-cache-dtype", args.kv_cache_dtype]
        # Consumer Blackwell auto-selects FlashInfer, but FreeToken's packed
        # Q4/Q8 KV pools are implemented by the Triton backend.  Keep this
        # benchmark self-contained so its documented full-context commands
        # exercise the requested cache format instead of failing at startup.
        if args.kv_cache_dtype not in {"auto", "bf16"}:
            cmd += ["--attention-backend", "triton"]
    if args.prefill_chunk is not None:
        cmd += ["--max-prefill-length", str(args.prefill_chunk)]
    if args.prefill_hit_d2d:
        cmd.append("--moe-prefill-hit-d2d")
    if getattr(args, "nvfp4_backend", None):
        cmd += ["--nvfp4-backend", args.nvfp4_backend]
    if getattr(args, "moe_collect_stats", False):
        cmd.append("--moe-collect-stats")
    if args.cache > 0:
        cmd += ["--moe-cache-size", str(args.cache)]
    elif args.cache_rate is not None:
        cmd += ["--moe-cache-rate", str(args.cache_rate)]
    else:
        cmd.append("--moe-cache-auto")
    for extra in getattr(args, "server_arg", None) or []:
        cmd += extra.split()
    return cmd


def scrape_moe_dumps(log_path: str) -> list[tuple[dict | None, list]]:
    """Every MoE decode-miss dump the scheduler has written, oldest first.

    It prints the aggregate dict as a python repr and the per-layer list as json, in that
    order, at each idle boundary; a dump is closed by its per-layer line. Malformed lines
    are ignored rather than failing the whole bench."""
    dumps: list[tuple[dict | None, list]] = []
    pending: dict | None = None
    try:
        text = Path(log_path).read_text(errors="replace")
    except OSError:
        return dumps
    for line in text.splitlines():
        i = line.find(MOE_STATS_PER_LAYER_PREFIX)
        if i >= 0:
            try:
                per_layer = json.loads(line[i + len(MOE_STATS_PER_LAYER_PREFIX):].strip())
            except ValueError:
                continue
            dumps.append((pending, per_layer))
            pending = None
            continue
        i = line.find(MOE_STATS_PREFIX)
        if i >= 0:
            try:
                pending = ast.literal_eval(line[i + len(MOE_STATS_PREFIX):].strip())
            except (ValueError, SyntaxError):
                continue
    return dumps


def scrape_moe_stats(log_path: str) -> tuple[dict | None, list | None, int]:
    """The scheduler's latest dump as ``(stats, per_layer, dumps_seen)``."""
    dumps = scrape_moe_dumps(log_path)
    if not dumps:
        return None, None, 0
    stats, per_layer = dumps[-1]
    return stats, per_layer, len(dumps)


def wait_for_moe_stats(
    log_path: str, seen_before: int, timeout: float = 60.0
) -> tuple[dict | None, list | None, int]:
    """Block until the scheduler dumps a *new* stats pair (it only does so when idle).

    Returns dump number ``seen_before`` (0-based) as ``(stats, per_layer, cursor)``; feed
    the cursor back in to wait for the next dump. On timeout the cursor is unchanged."""
    deadline = time.monotonic() + timeout
    while True:
        dumps = scrape_moe_dumps(log_path)
        if len(dumps) > seen_before:
            stats, per_layer = dumps[seen_before]
            if stats is not None:
                return stats, per_layer, seen_before + 1
        if time.monotonic() >= deadline:
            print(
                f"[bench] WARNING: no new MoE stats in the server log after {timeout:.0f}s",
                flush=True,
            )
            return None, None, seen_before
        time.sleep(1.0)


def _raw_layer_counters(row: dict) -> dict:
    """Undo the per-step ratios back into the raw cumulative counters.

    The counters are integers, so rounding recovers them exactly from the logged
    round-trippable float ratios."""
    steps = int(row["steps"])
    return {
        "steps": steps,
        "active": round(row["active_per_step"] * steps),
        "missing": round(row["missing_per_step"] * steps),
        "fetched": round(row["fetched_per_step"] * steps),
    }


def moe_stats_delta(
    after_warmup: tuple[dict, list], after_measured: tuple[dict, list]
) -> tuple[dict, list]:
    """Stats for the measured window alone, as (aggregate, per_layer).

    The cache's counters are cumulative from server start (``reset_stats`` only runs on a
    cache rebuild), so a single dump averages the cold warm-up generation together with
    the warm measured one and overstates the steady-state miss rate. Differencing two
    idle-boundary snapshots isolates the measured window."""
    stats_a, layers_a = after_warmup
    stats_b, layers_b = after_measured
    base = {int(row["layer"]): _raw_layer_counters(row) for row in layers_a}

    per_layer: list[dict] = []
    tot = {"steps": 0, "active": 0, "missing": 0, "fetched": 0}
    for row in layers_b:
        layer = int(row["layer"])
        now = _raw_layer_counters(row)
        was = base.get(layer, {"steps": 0, "active": 0, "missing": 0, "fetched": 0})
        d = {k: max(0, now[k] - was[k]) for k in now}
        for k in tot:
            tot[k] += d[k]
        s, a = d["steps"], d["active"]
        out = {
            "layer": layer,
            "steps": s,
            "active_per_step": (d["active"] / s) if s else 0.0,
            "missing_per_step": (d["missing"] / s) if s else 0.0,
            "miss_rate": (d["missing"] / a) if a else 0.0,
            "fetched_per_step": (d["fetched"] / s) if s else 0.0,
        }
        # The remaining fields are cumulative host counters; difference them the same way.
        prev = next((r for r in layers_a if int(r["layer"]) == layer), {})
        for key in (
            "pageable_stage_calls",
            "pageable_rows",
            "pageable_plan_wait_seconds",
            "pageable_gather_seconds",
        ):
            if key in row:
                out[key] = max(0, row[key] - prev.get(key, 0))
        per_layer.append(out)

    calls, active, missing, fetched = (
        tot["steps"], tot["active"], tot["missing"], tot["fetched"]
    )
    stats = {
        "layer_calls": calls,
        "active_per_layer": (active / calls) if calls else 0.0,
        "missing_per_layer": (missing / calls) if calls else 0.0,
        "miss_rate": (missing / active) if active else 0.0,
        "fetched_per_layer": (fetched / calls) if calls else 0.0,
        "cpu_per_layer": ((missing - fetched) / calls) if calls else 0.0,
        "fetch_rate": (fetched / missing) if missing else 0.0,
    }
    # Scalars the aggregate line carries that are cumulative counts/seconds, not ratios.
    for key in (
        "prefill_hit_rows",
        "prefill_rows",
        "pageable_stage_calls",
        "pageable_rows",
        "pageable_gib",
        "pageable_plan_wait_seconds",
        "pageable_gather_seconds",
    ):
        if key in stats_b:
            stats[key] = max(0, stats_b[key] - stats_a.get(key, 0))
    return stats, per_layer


def die_with_log(msg: str, log_path: str) -> None:
    tail = "".join(Path(log_path).read_text().splitlines(keepends=True)[-30:])
    sys.exit(f"[bench] {msg}\n[bench] server log tail ({log_path}):\n{tail}")


def wait_ready(origin: str, proc: subprocess.Popen, log_path: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            die_with_log(f"server exited with code {proc.returncode} during startup", log_path)
        try:
            health = get_json(f"{origin}/health", timeout=5)
        except (OSError, ValueError):  # not bound yet / reset / partial response
            time.sleep(1.0)
            continue
        if health.get("status") == "error":
            die_with_log(f"server reported startup error: {health}", log_path)
        if health.get("maintenance") == "serving":
            return
        time.sleep(1.0)
    die_with_log(f"server not ready after {timeout:.0f}s", log_path)


def pump_output(src, log_f) -> None:
    """Mirror the server's output to our terminal while keeping the log file complete.

    Raw byte chunks (read1, not line-buffered) so \\r progress bars render live."""
    for chunk in iter(lambda: src.read1(65536), b""):
        log_f.write(chunk)
        log_f.flush()
        sys.stdout.buffer.write(chunk)
        sys.stdout.flush()


def stop_server(proc: subprocess.Popen) -> None:
    """SIGTERM the whole session (frontend + scheduler/tokenizer workers), escalate.

    Best-effort by design: it runs in ``finally`` and must not mask the real error.
    killpg runs even when the frontend already exited -- a crashed frontend leaves live
    non-daemon workers in the group, and they hold the GPU."""
    for sig, wait_s in ((signal.SIGTERM, 90), (signal.SIGKILL, 30)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:  # whole group already gone
            pass
        try:
            proc.wait(timeout=wait_s)
            break
        except subprocess.TimeoutExpired:
            continue
    time.sleep(3)  # let the driver reclaim VRAM before the next backend's server


def stream_generate(origin: str, model_id: str, problem: str, sampling: dict,
                    args: argparse.Namespace) -> dict:
    """One streamed chat completion; returns per-token arrival stamps, text, and usage."""
    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": problem}],
        "max_tokens": args.decode,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": True},
        **sampling,
    }
    req = urllib.request.Request(
        f"{origin}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    stamps: list[float] = []
    pieces: list[str] = []
    usage: dict | None = None
    t0 = time.perf_counter()
    try:
        resp = urllib.request.urlopen(req, timeout=1800)
    except urllib.error.HTTPError as e:
        sys.exit(f"[bench] request failed: HTTP {e.code}: {e.read()[:500]!r}")
    # Iterate the SSE stream line by line as bytes; json.loads decodes UTF-8 itself.
    # (A text-mode reader keyed off the content-type would decode latin-1: the server
    # sends ensure_ascii=False JSON with no charset on text/event-stream.)
    with resp:
        for raw in resp:
            line = raw.strip()
            if not line or not line.startswith(b"data:"):
                continue  # blank separators between events
            payload = line[len(b"data:"):].strip()
            if payload == b"[DONE]":
                break
            now = time.perf_counter()
            chunk = json.loads(payload)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta") or {}
                text = delta.get("reasoning_content") or delta.get("content")
                if text:
                    stamps.append(now)
                    pieces.append(text)
    if usage is None:
        sys.exit("[bench] stream ended without a usage chunk; is this a FreeToken server?")
    return {"t0": t0, "stamps": stamps, "text": "".join(pieces), "usage": usage}


def run_streams(origin: str, model_id: str, problems: list[str], sampling: dict,
                args: argparse.Namespace) -> list[dict]:
    """One streamed completion per problem, all in flight together (one thread each).

    time.perf_counter is process-wide, so the per-thread stamps are directly comparable
    and the aggregate window can span streams. A single stream runs inline, keeping the
    bs=1 path exactly as it was."""
    if len(problems) == 1:
        return [stream_generate(origin, model_id, problems[0], sampling, args)]
    results: list[dict | None] = [None] * len(problems)
    errors: list[BaseException | None] = [None] * len(problems)

    def worker(i: int) -> None:
        try:
            results[i] = stream_generate(origin, model_id, problems[i], sampling, args)
        except BaseException as e:  # SystemExit from stream_generate included
            errors[i] = e

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(problems))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for e in errors:
        if e is not None:
            raise e
    return [r for r in results if r is not None]


def decode_metrics(results: list[dict], decode: int) -> dict:
    """Decode timings from the streamed arrival stamps of one or more streams.

    Single stream: exactly the bs=1 definitions -- (completion_tokens - 1) over the window
    between the first and last token event. Multiple streams: the same per-stream figure
    (reported as median/min), plus the aggregate -- every decoded step across streams over
    the wall window from the first token event of *any* stream to the last of any."""
    per_stream = []
    gaps: list[float] = []
    for r in results:
        stamps, usage = r["stamps"], r["usage"]
        if len(stamps) < 2:
            sys.exit(f"[bench] need >=2 token events to measure decode, got {len(stamps)}")
        completion = usage["completion_tokens"]
        if completion != decode:
            print(f"[bench] WARNING: completion_tokens={completion} != --decode {decode}",
                  flush=True)
        window = stamps[-1] - stamps[0]
        per_stream.append({
            "steps": completion - 1,
            "window": window,
            "tok_s": (completion - 1) / window if window > 0 else 0.0,
            "ttft_ms": (stamps[0] - r["t0"]) * 1e3,
            "first": stamps[0],
            "last": stamps[-1],
            "events": len(stamps),
            "completion_tokens": completion,
            "prompt_tokens": usage["prompt_tokens"],
            "sha1": hashlib.sha1(r["text"].encode()).hexdigest()[:12],
        })
        gaps += [(b - a) * 1e3 for a, b in zip(stamps, stamps[1:])]
    gaps.sort()

    first = per_stream[0]
    total_steps = sum(s["steps"] for s in per_stream)
    agg_window = max(s["last"] for s in per_stream) - min(s["first"] for s in per_stream)
    agg_tok_s = total_steps / agg_window if agg_window > 0 else 0.0
    stream_tok_s = [s["tok_s"] for s in per_stream]
    ttfts = sorted(s["ttft_ms"] for s in per_stream)
    single = len(per_stream) == 1
    out = {
        "prompt_tokens": first["prompt_tokens"],
        "decode_steps": first["steps"],
        "decode_seconds": first["window"],
        "decode_tok_s": first["tok_s"] if single else agg_tok_s,
        "ms_per_token": (
            (first["window"] / first["steps"] * 1e3 if first["steps"] else 0.0)
            if single
            else (agg_window / total_steps * 1e3 if total_steps else 0.0)
        ),
        "event_ms_p50": gaps[len(gaps) // 2],
        "event_ms_p99": gaps[min(len(gaps) - 1, int(len(gaps) * 0.99))],
        "ttft_ms": first["ttft_ms"] if single else statistics.median(ttfts),
        "events": first["events"],
        "completion_tokens": first["completion_tokens"],
        "output_sha1": first["sha1"],
    }
    if not single:
        out.update({
            "decode_tok_s_streams": stream_tok_s,
            "decode_tok_s_stream_median": statistics.median(stream_tok_s),
            "decode_tok_s_stream_min": min(stream_tok_s),
            "decode_tok_s_aggregate": agg_tok_s,
            "decode_steps_total": total_steps,
            "decode_window_s": agg_window,
            "ttft_ms_p50": statistics.median(ttfts),
            "ttft_ms_max": max(ttfts),
            "output_sha1_streams": [s["sha1"] for s in per_stream],
        })
    return out


def run_one(args: argparse.Namespace, backend: str) -> dict:
    streams = concurrency_of(args)
    problems = load_problems(args.aime, args.problem, streams)
    problem, answer = problems[0]
    # fork/main's --warm-problem: warm on a *different* problem, so a stickier cache
    # policy cannot win merely by replaying the measured request's own routing trace.
    warm_problem = (
        load_problem(args.aime, args.warm_problem)[0]
        if getattr(args, "warm_problem", None) is not None
        else None
    )
    sampling, sampling_src = resolve_sampling(args.model, args.greedy)
    port = free_port()
    origin = f"http://127.0.0.1:{port}"
    fd, log_path = tempfile.mkstemp(prefix=f"bench-serve-{backend}-", suffix=".log")
    cmd = serve_cmd(args, backend, port)

    print(
        f"[bench] model={args.model}\n"
        f"[bench] backend={backend} cache={args.cache or args.cache_rate or 'auto'} "
        f"policy={args.cache_policy} "
        f"mem_ratio={args.mem_ratio} decode={args.decode} graph={not args.no_graph}\n"
        f"[bench] max_context={args.max_context or f'{8192 + args.decode} (default)'} "
        f"kv_cache_dtype={args.kv_cache_dtype or 'auto (default)'} "
        f"prefill_chunk={args.prefill_chunk or 'default'} prefill_hit_d2d={args.prefill_hit_d2d}\n"
        f"[bench] sampling={sampling} <- {sampling_src}\n"
        f"[bench] server log: {log_path}",
        flush=True,
    )
    if streams > 1 or args.nvfp4_backend or args.moe_collect_stats or args.server_arg:
        print(
            f"[bench] concurrency={streams} nvfp4_backend={args.nvfp4_backend or 'default'} "
            f"moe_collect_stats={args.moe_collect_stats} "
            f"server_arg={args.server_arg or []}",
            flush=True,
        )

    with os.fdopen(fd, "wb") as log_f:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True
        )
        pump = threading.Thread(target=pump_output, args=(proc.stdout, log_f), daemon=True)
        pump.start()
        try:
            wait_ready(origin, proc, log_path, args.server_timeout)
            model_id = get_json(f"{origin}/v1/models")["data"][0]["id"]
            print(f"[bench] model_id={model_id}", flush=True)
            if streams == 1:
                print(f"[bench] AIME25 #{args.problem} (answer {answer})", flush=True)
            else:
                shown = ", ".join(a for _, a in problems)
                print(
                    f"[bench] AIME25 #{args.problem}..+{streams - 1} (answers {shown})",
                    flush=True,
                )
            texts = [p for p, _ in problems]

            seen = scrape_moe_stats(log_path)[2] if args.moe_collect_stats else 0

            # Warm the expert cache to a steady-state decode working set.
            warm_texts = [warm_problem] * streams if warm_problem is not None else texts
            run_streams(origin, model_id, warm_texts, sampling, args)
            snap_a: tuple[dict, list] | None = None
            if args.moe_collect_stats:
                # The scheduler only dumps counters at an idle boundary, i.e. once a
                # generation drains -- wait for the line rather than racing it.
                stats_a, layers_a, seen = wait_for_moe_stats(log_path, seen)
                if stats_a is not None and layers_a is not None:
                    snap_a = (stats_a, layers_a)

            results = run_streams(origin, model_id, texts, sampling, args)
            snap_b: tuple[dict, list] | None = None
            if args.moe_collect_stats:
                stats_b, layers_b, seen = wait_for_moe_stats(log_path, seen)
                if stats_b is not None and layers_b is not None:
                    snap_b = (stats_b, layers_b)
            else:
                # Nothing to wait for on the stats line, but run_when_idle() also emits
                # fork/main's CUDA-event GPU batch profile; tearing the server down at the
                # last SSE chunk loses it. Let the scheduler cross its idle boundary.
                time.sleep(1.0)
            stats = get_json(f"{origin}/v1/stats")
        finally:
            stop_server(proc)
            pump.join(timeout=10)

    # The measured window alone when both idle-boundary snapshots landed; otherwise the
    # cumulative counters, flagged as such so a cache-sizing study never mistakes the
    # cold-start average for steady state.
    moe_stats: dict | None = None
    moe_stats_per_layer: list | None = None
    moe_stats_window: str | None = None
    if snap_a and snap_b:
        moe_stats, moe_stats_per_layer = moe_stats_delta(snap_a, snap_b)
        moe_stats_window = "measured"
    elif snap_b or snap_a:
        moe_stats, moe_stats_per_layer = snap_b or snap_a
        moe_stats_window = "cumulative"

    metrics = decode_metrics(results, args.decode)
    row = {
        "model": args.model,
        "backend": backend,
        "problem": args.problem,
        "concurrency": streams,
        "nvfp4_backend": args.nvfp4_backend,
        "cache_rate": args.cache_rate,
        "cache_policy": args.cache_policy,
        **metrics,
        "vram_gib": stats.get("vram_bytes", 0) / 2**30,
        "sampling": sampling,
        "moe_stats": moe_stats,
        "moe_stats_per_layer": moe_stats_per_layer,
        "moe_stats_window": moe_stats_window,
        "moe_stats_cumulative": {
            "after_warmup": {"stats": snap_a[0], "per_layer": snap_a[1]} if snap_a else None,
            "after_measured": {"stats": snap_b[0], "per_layer": snap_b[1]} if snap_b else None,
        } if (snap_a or snap_b) else None,
        "server_log": log_path,
    }

    if streams > 1:
        print(f"\n==== decode concurrency={streams} [{backend}] via /v1/chat/completions ====",
              flush=True)
        print(f"  aggregate decode  : {row['decode_tok_s_aggregate']:8.2f} tok/s  "
              f"({row['decode_steps_total']} steps in {row['decode_window_s']:.3f} s)")
        print(f"  per-stream decode : median {row['decode_tok_s_stream_median']:8.2f} / "
              f"min {row['decode_tok_s_stream_min']:8.2f} tok/s")
        print(f"  TTFT (warm)       : p50 {row['ttft_ms_p50']:8.1f} / "
              f"max {row['ttft_ms_max']:8.1f} ms  (prompt {row['prompt_tokens']} tok)")
        print(f"  event gaps (all)  : p50 {row['event_ms_p50']:.3f} / "
              f"p99 {row['event_ms_p99']:.3f} ms")
    else:
        print(f"\n==== decode bs=1 [{backend}] via /v1/chat/completions ====", flush=True)
        print(f"  decode throughput : {row['decode_tok_s']:8.2f} tok/s  ({row['ms_per_token']:.3f} ms/token)")
        print(f"  TTFT (warm)       : {row['ttft_ms']:8.1f} ms  (prompt {row['prompt_tokens']} tok)")
        print(f"  decode measured   : {row['decode_steps']} steps in "
              f"{row['decode_seconds']:.3f} s  "
              f"(event p50 {row['event_ms_p50']:.3f} / p99 {row['event_ms_p99']:.3f} ms, "
              f"{row['events']} events)")
    print(f"  vram (server)     : {row['vram_gib']:8.2f} GiB")
    sha_note = "greedy" if args.greedy else "sampled, per-server deterministic"
    print(f"  output sha1       : {row['output_sha1']}  ({sha_note}; compare across backends)")
    if moe_stats is not None:
        print(f"  moe stats window  : {moe_stats_window} (warm-up excluded)"
              if moe_stats_window == "measured"
              else f"  moe stats window  : {moe_stats_window} (includes the cold warm-up!)")
        print(f"  moe decode stats  : {moe_stats}")
        print(f"  moe per-layer     : {len(moe_stats_per_layer or [])} layers in the JSON row")
    print(f"  output sample     : {results[0]['text'][:240]!r}")
    return row


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    backends = [b.strip() for b in args.backend.split(",") if b.strip()]
    unknown = [b for b in backends if b not in ("offload", "cpu", "hybrid")]
    if unknown:
        sys.exit(f"unknown backend(s): {unknown}")
    if args.concurrency < 1:
        sys.exit(f"--concurrency must be >= 1, got {args.concurrency}")

    failed = []
    for backend in backends:
        try:
            row = run_one(args, backend)
        # SystemExit inherits BaseException, not Exception, so name both: a mid-decode
        # connection drop (server crash) must not abort the remaining backends either.
        except (SystemExit, Exception) as e:
            if len(backends) == 1:
                raise
            print(f"\n[bench] backend {backend} failed: {e!r}", flush=True)
            failed.append(backend)
            continue
        if args.json_out:
            with open(args.json_out, "a") as f:
                f.write(json.dumps(row) + "\n")
    if failed:
        print(f"\n[bench] backends that failed: {failed}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
