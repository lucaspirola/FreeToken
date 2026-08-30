"""Concurrent long-context isolation and growable-KV benchmark.

Each request carries a different deterministic needle and a disjoint filler prefix. Requests
are launched together, so the gate detects page-table/KV cross-talk while their aggregate live
pages force the shared VMM arena across one or more physical growth boundaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time

import bench_decode_moe as common
import bench_long_context as long_context


PASSCODES = (
    "7319041",
    "8462759",
    "2951384",
    "6748203",
    "9184076",
    "3526918",
    "4801735",
    "7695240",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--agents", type=int, default=2, choices=range(2, len(PASSCODES) + 1))
    parser.add_argument("--prompt-tokens", type=int, default=70_000)
    parser.add_argument("--decode", type=int, default=64)
    parser.add_argument(
        "--helper-decode",
        type=int,
        help="shorter output budget for agents 1..N (exercises teardown while agent 0 lives)",
    )
    parser.add_argument("--max-context", type=int, default=131_072, help="per-agent ceiling")
    parser.add_argument("--num-tokens", type=int, default=262_144, help="shared KV-page ceiling")
    parser.add_argument("--kv-cache-dtype", default="q8_0")
    parser.add_argument("--kv-grow-step-tokens", type=int, default=131_072)
    parser.add_argument("--prefill-chunk", type=int, default=8192)
    parser.add_argument("--mem-ratio", type=float, default=0.97)
    parser.add_argument("--cache-policy", choices=("lru", "lfu"), default="lfu")
    parser.add_argument("--moe-pageable-gpu", action="store_true")
    parser.add_argument("--moe-collect-stats", action="store_true")
    parser.add_argument("--max-prefill-sequences", type=int)
    parser.add_argument("--server-timeout", type=float, default=1800)
    parser.add_argument(
        "--agent-stagger",
        type=float,
        default=0.0,
        help="seconds between request starts; makes agent 0 the established main agent",
    )
    parser.add_argument("--json", dest="json_out")
    return parser.parse_args()


def synthetic_agent_sample(agent: int) -> tuple[str, str]:
    expected = PASSCODES[agent]
    # Different text from token zero prevents the radix cache from making this an accidental
    # shared-prefix benchmark. Keep ample filler so tokenizer-space trimming reaches 128K/agent.
    before = f"Agent {agent} orchard record marks amber inactive.\n" * 20_000
    after = f"Agent {agent} harbor record marks violet inactive.\n" * 20_000
    prompt = (
        f"You are isolated agent {agent}. Ignore every inactive record and remember only your "
        f"own secret. Never use another agent's answer.\n\n{before}\n"
        f"Agent {agent}'s secret passcode is {expected}.\n{after}\n"
        f"What is agent {agent}'s secret passcode? State its digits clearly."
    )
    return prompt, expected


def serve_cmd(args: argparse.Namespace, port: int) -> list[str]:
    command = [
        sys.executable,
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
        args.cache_policy,
        "--max-running-requests",
        str(args.agents),
        "--max-seq-len-override",
        str(args.max_context),
        "--num-tokens",
        str(args.num_tokens),
        "--memory-ratio",
        str(args.mem_ratio),
        "--cuda-graph-max-bs",
        str(args.agents),
        "--kv-cache-dtype",
        args.kv_cache_dtype,
        "--attention-backend",
        "triton",
        "--max-prefill-length",
        str(args.prefill_chunk),
        "--kv-grow-step-tokens",
        str(args.kv_grow_step_tokens),
    ]
    if args.moe_pageable_gpu:
        command.append("--moe-pageable-gpu")
    if args.moe_collect_stats:
        command.append("--moe-collect-stats")
    if args.max_prefill_sequences is not None:
        command.extend(("--max-prefill-sequences", str(args.max_prefill_sequences)))
    return command


def _prefill_rates(log_path: str) -> dict[str, float | int | None]:
    pattern = re.compile(
        r"input throughput \(token/s\): ([0-9.]+) instant, ([0-9.]+) average"
    )
    samples: list[tuple[float, float]] = []
    with open(log_path, errors="replace") as server_log:
        for line in server_log:
            match = pattern.search(line)
            if match:
                samples.append((float(match.group(1)), float(match.group(2))))
    if not samples:
        return {
            "prefill_samples": 0,
            "prefill_instant_tok_s": None,
            "prefill_average_tok_s": None,
        }
    instant, average = samples[-1]
    return {
        "prefill_samples": len(samples),
        "prefill_instant_tok_s": instant,
        "prefill_average_tok_s": average,
    }


def main() -> int:
    args = parse_args()
    decode_by_agent = [args.decode] + [args.helper_decode or args.decode] * (args.agents - 1)
    if args.prompt_tokens + max(decode_by_agent) > args.max_context:
        raise SystemExit("prompt plus decode exceeds the per-agent context ceiling")
    if args.agents * args.prompt_tokens + sum(decode_by_agent) > args.num_tokens:
        raise SystemExit("aggregate requested tokens exceed the shared KV ceiling")

    from freetoken.models.gguf.tokenizer import load_gguf_tokenizer

    tokenizer = load_gguf_tokenizer(args.model)
    prompts: list[str] = []
    expected: list[str] = []
    for agent in range(args.agents):
        raw, needle = synthetic_agent_sample(agent)
        prompt, _original, actual = long_context.trim_filler(
            tokenizer, raw, needle, args.prompt_tokens
        )
        if actual != args.prompt_tokens:
            raise SystemExit(f"agent {agent} trimmed to {actual}, expected {args.prompt_tokens}")
        prompts.append(prompt)
        expected.append(needle)

    port = common.free_port()
    origin = f"http://127.0.0.1:{port}"
    fd, log_path = tempfile.mkstemp(prefix="bench-multi-agent-", suffix=".log")
    print(
        f"[multi] agents={args.agents} prompt={args.prompt_tokens} decode={args.decode} "
        f"per_agent_context={args.max_context} shared_kv={args.num_tokens}\n"
        f"[multi] expected={expected} log={log_path}",
        flush=True,
    )

    cmd = serve_cmd(args, port)
    results: list[dict | None] = [None] * args.agents
    errors: list[BaseException | None] = [None] * args.agents
    barrier = threading.Barrier(args.agents)

    def run_agent(agent: int, model_id: str) -> None:
        try:
            barrier.wait()
            if args.agent_stagger:
                time.sleep(agent * args.agent_stagger)
            results[agent] = long_context.stream_completion(
                origin, model_id, prompts[agent], decode_by_agent[agent]
            )
        except BaseException as error:  # retain every worker failure for the main thread
            errors[agent] = error

    with os.fdopen(fd, "wb") as log_f:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True
        )
        pump = threading.Thread(
            target=common.pump_output, args=(proc.stdout, log_f), daemon=True
        )
        pump.start()
        try:
            common.wait_ready(origin, proc, log_path, args.server_timeout)
            model_id = common.get_json(f"{origin}/v1/models")["data"][0]["id"]
            started = time.perf_counter()
            workers = [
                threading.Thread(target=run_agent, args=(agent, model_id), daemon=True)
                for agent in range(args.agents)
            ]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join()
            wall_seconds = time.perf_counter() - started
            stats = common.get_json(f"{origin}/v1/stats")
        finally:
            common.stop_server(proc)
            pump.join(timeout=10)

    if any(errors):
        raise SystemExit(f"agent failures: {errors}")

    rows = []
    passed = True
    for agent, result in enumerate(results):
        assert result is not None
        stamps = result["stamps"]
        usage = result["usage"]
        completion = int(usage["completion_tokens"])
        decode_steps = completion - 1
        decode_seconds = stamps[-1] - stamps[0]
        ttft_seconds = stamps[0] - result["t0"]
        # The throughput leg deliberately uses ignore_eos=True to obtain an exact decode count.
        # Judge answer isolation only through the first Qwen end-of-turn marker; text after it is
        # forced continuation and often invents a new copy of the visible benchmark template.
        answer_prefix = result["text"].split("<|im_end|>", 1)[0]
        own_found = expected[agent] in answer_prefix
        foreign_found = any(
            needle in answer_prefix for other, needle in enumerate(expected) if other != agent
        )
        post_eos_foreign_prompt = any(
            f"Agent {other} orchard record marks amber inactive." in result["text"]
            for other in range(args.agents)
            if other != agent
        )
        coherent = own_found and not foreign_found
        passed &= coherent
        row = {
            "agent": agent,
            "expected": expected[agent],
            "expected_found": own_found,
            "foreign_found": foreign_found,
            "post_eos_foreign_prompt": post_eos_foreign_prompt,
            "prompt_tokens": int(usage["prompt_tokens"]),
            "decode_tokens": completion,
            "ttft_seconds": ttft_seconds,
            "decode_tok_s": decode_steps / decode_seconds,
            "output_sha1": hashlib.sha1(result["text"].encode()).hexdigest()[:12],
            "output": result["text"],
        }
        rows.append(row)
        print(
            f"  agent {agent}: ttft={ttft_seconds:.3f}s decode={row['decode_tok_s']:.2f} "
            f"tok/s own={own_found} foreign={foreign_found} "
            f"post_eos_foreign_prompt={post_eos_foreign_prompt} output={result['text']!r}",
            flush=True,
        )

    summary = {
        "model": args.model,
        "agents": args.agents,
        "prompt_tokens_per_agent": args.prompt_tokens,
        "shared_kv_tokens": args.num_tokens,
        "kv_grow_step_tokens": args.kv_grow_step_tokens,
        "wall_seconds": wall_seconds,
        "aggregate_prompt_tok_s": args.agents * args.prompt_tokens / wall_seconds,
        "vram_gib": stats.get("vram_bytes", 0) / 2**30,
        "passed": passed,
        "agents_result": rows,
        "server_log": log_path,
    }
    first_decode = min(result["stamps"][0] for result in results)
    last_decode = max(result["stamps"][-1] for result in results)
    total_decode_steps = sum(row["decode_tokens"] - 1 for row in rows)
    summary["aggregate_decode_tok_s"] = total_decode_steps / (
        last_decode - first_decode
    )
    overlap_start = max(result["stamps"][0] for result in results)
    overlap_end = min(result["stamps"][-1] for result in results)
    overlap_tokens = sum(
        max(
            0,
            len([stamp for stamp in result["stamps"] if overlap_start <= stamp <= overlap_end])
            - 1,
        )
        for result in results
    )
    summary["simultaneous_decode_tok_s"] = (
        overlap_tokens / (overlap_end - overlap_start)
        if overlap_tokens and overlap_end > overlap_start
        else None
    )
    summary.update(_prefill_rates(log_path))
    if args.helper_decode is not None:
        main_stamps = results[0]["stamps"]
        helper_done = max(result["stamps"][-1] for result in results[1:])
        during = [stamp for stamp in main_stamps if stamp <= helper_done]
        after = [stamp for stamp in main_stamps if stamp > helper_done]
        tail = after[-min(80, len(after)) :]
        summary["teardown"] = {
            "main_tokens_while_helpers_live": len(during),
            "main_tokens_after_helpers_stop": len(after),
            "first_post_helper_token_seconds": after[0] - helper_done if after else None,
            "largest_post_helper_token_gap_seconds": (
                max(b - a for a, b in zip(after, after[1:], strict=False))
                if len(after) > 1 else None
            ),
            "main_post_teardown_tok_s": (
                (len(after) - 1) / (after[-1] - after[0]) if len(after) > 1 else None
            ),
            "main_post_teardown_tail_tok_s": (
                (len(tail) - 1) / (tail[-1] - tail[0]) if len(tail) > 1 else None
            ),
        }
        print(f"[multi] teardown={summary['teardown']}", flush=True)
    print(
        f"[multi] wall={wall_seconds:.3f}s aggregate_prompt="
        f"{summary['aggregate_prompt_tok_s']:.2f} tok/s pass={passed}",
        flush=True,
    )
    print(
        f"[multi] prefill={summary['prefill_instant_tok_s']} instant / "
        f"{summary['prefill_average_tok_s']} average tok/s; aggregate decode="
        f"{summary['aggregate_decode_tok_s']:.2f} tok/s; simultaneous decode="
        f"{summary['simultaneous_decode_tok_s']} tok/s",
        flush=True,
    )
    if args.json_out:
        with open(args.json_out, "a") as output:
            output.write(json.dumps(summary) + "\n")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
