"""Mixed-phase agent scheduler benchmark.

Start a short-context main agent, wait for its first decoded token, then admit a
long-context helper. This exercises the prefill/decode time slicer directly while
checking that each request recovers only its own deterministic needle.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import threading
import time
import urllib.request

import bench_decode_moe as common
import bench_long_context as long_context
import bench_multi_agent_context as multi_agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--kv-cache-dtype", required=True)
    parser.add_argument("--kv-grow-step-tokens", required=True, type=int)
    parser.add_argument("--num-tokens", required=True, type=int)
    parser.add_argument("--max-context", type=int, default=65_536)
    parser.add_argument("--main-prompt-tokens", type=int, default=4_096)
    parser.add_argument("--helper-prompt-tokens", type=int, default=32_768)
    parser.add_argument("--main-decode", type=int, default=1_024)
    parser.add_argument("--helper-decode", type=int, default=64)
    parser.add_argument("--prefill-chunk", type=int, default=8_192)
    parser.add_argument("--mem-ratio", type=float, default=0.97)
    parser.add_argument("--fixed-scheduler", action="store_true")
    parser.add_argument("--moe-pageable-gpu", action="store_true")
    parser.add_argument("--moe-collect-stats", action="store_true")
    parser.add_argument("--server-timeout", type=float, default=1_800)
    parser.add_argument("--json", dest="json_out")
    return parser.parse_args()


def _serve_cmd(args: argparse.Namespace, port: int) -> list[str]:
    command = [
        os.sys.executable,
        "-m",
        "freetoken.cli",
        "serve",
        "--model",
        args.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--moe-backend",
        "offload",
        "--moe-cache-auto",
        "--moe-cache-policy",
        "lfu",
        "--max-running-requests",
        "2",
        "--max-seq-len-override",
        str(args.max_context),
        "--num-tokens",
        str(args.num_tokens),
        "--memory-ratio",
        str(args.mem_ratio),
        "--cuda-graph-max-bs",
        "2",
        "--kv-cache-dtype",
        args.kv_cache_dtype,
        "--attention-backend",
        "triton",
        "--max-prefill-length",
        str(args.prefill_chunk),
        "--kv-grow-step-tokens",
        str(args.kv_grow_step_tokens),
    ]
    if args.fixed_scheduler:
        command.append("--disable-adaptive-scheduler")
    if args.moe_pageable_gpu:
        command.append("--moe-pageable-gpu")
    if args.moe_collect_stats:
        command.append("--moe-collect-stats")
    return command


def _stream(
    origin: str,
    model: str,
    prompt: str,
    decode: int,
    first_token: threading.Event | None = None,
) -> dict:
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": decode,
        "ignore_eos": True,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        f"{origin}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    stamps: list[float] = []
    pieces: list[str] = []
    usage = None
    with urllib.request.urlopen(request, timeout=7_200) as response:
        for raw_line in response:
            line = raw_line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            now = time.perf_counter()
            chunk = json.loads(payload)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                piece = choice.get("text", "")
                if piece:
                    stamps.append(now)
                    pieces.append(piece)
                    if first_token is not None:
                        first_token.set()
    if usage is None or not stamps:
        raise RuntimeError("stream completed without token stamps and final usage")
    return {
        "started": started,
        "ended": time.perf_counter(),
        "stamps": stamps,
        "text": "".join(pieces),
        "usage": usage,
    }


def main() -> int:
    args = parse_args()
    from freetoken.models.gguf.tokenizer import load_gguf_tokenizer

    tokenizer = load_gguf_tokenizer(args.model)
    main_raw, main_expected = long_context.synthetic_needle_sample()
    main_prompt, _, main_tokens = long_context.trim_filler(
        tokenizer, main_raw, main_expected, args.main_prompt_tokens
    )
    helper_raw, helper_expected = multi_agent.synthetic_agent_sample(1)
    helper_prompt, _, helper_tokens = long_context.trim_filler(
        tokenizer, helper_raw, helper_expected, args.helper_prompt_tokens
    )

    port = common.free_port()
    origin = f"http://127.0.0.1:{port}"
    fd, log_path = tempfile.mkstemp(prefix="bench-mixed-scheduler-", suffix=".log")
    mode = "fixed" if args.fixed_scheduler else "adaptive"
    print(
        f"[mixed] mode={mode} main={main_tokens} helper={helper_tokens} log={log_path}",
        flush=True,
    )
    results: dict[str, dict] = {}
    errors: dict[str, BaseException] = {}
    first_token = threading.Event()

    def run(label: str, model: str, prompt: str, decode: int) -> None:
        try:
            results[label] = _stream(
                origin,
                model,
                prompt,
                decode,
                first_token if label == "main" else None,
            )
        except BaseException as error:
            errors[label] = error
            first_token.set()

    with os.fdopen(fd, "wb") as log_file:
        process = subprocess.Popen(
            _serve_cmd(args, port),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            common.wait_ready(origin, process, log_path, args.server_timeout)
            model = common.get_json(f"{origin}/v1/models")["data"][0]["id"]
            print("[mixed] server ready; starting main then helper", flush=True)
            main_thread = threading.Thread(
                target=run,
                args=("main", model, main_prompt, args.main_decode),
                daemon=True,
            )
            main_thread.start()
            if not first_token.wait(900):
                raise RuntimeError("main agent produced no first token")
            helper_thread = threading.Thread(
                target=run,
                args=("helper", model, helper_prompt, args.helper_decode),
                daemon=True,
            )
            helper_thread.start()
            main_thread.join()
            helper_thread.join()
            # Let the scheduler's first idle pass flush opt-in MoE profile summaries before
            # SIGTERM. Without this grace the aggregate line may win the race while the
            # per-layer device counters are still being copied to the host.
            if args.moe_collect_stats:
                time.sleep(2.0)
        finally:
            common.stop_server(process)

    if errors:
        raise SystemExit({label: repr(error) for label, error in errors.items()})
    main_result, helper_result = results["main"], results["helper"]
    during = [
        stamp
        for stamp in main_result["stamps"]
        if helper_result["started"] <= stamp <= helper_result["ended"]
    ]
    gaps = [right - left for left, right in zip(during, during[1:])]
    summary = {
        "mode": mode,
        "main_prompt_tokens": main_tokens,
        "helper_prompt_tokens": helper_tokens,
        "main_own": main_expected in main_result["text"],
        "main_foreign": helper_expected in main_result["text"],
        "helper_own": helper_expected in helper_result["text"],
        "helper_foreign": main_expected in helper_result["text"],
        "helper_ttft_s": helper_result["stamps"][0] - helper_result["started"],
        "helper_wall_s": helper_result["ended"] - helper_result["started"],
        "main_tokens_during_helper": len(during),
        "main_max_gap_s": max(gaps) if gaps else None,
        "server_log": log_path,
    }
    coherent = (
        summary["main_own"]
        and summary["helper_own"]
        and not summary["main_foreign"]
        and not summary["helper_foreign"]
    )
    summary["passed"] = coherent
    print("MIXED_SCHEDULER_RESULT " + json.dumps(summary), flush=True)
    if args.json_out:
        with open(args.json_out, "a") as output:
            output.write(json.dumps(summary) + "\n")
    return 0 if coherent else 2


if __name__ == "__main__":
    raise SystemExit(main())
