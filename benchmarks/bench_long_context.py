"""Cold single-request long-context benchmark through FreeToken's serving API.

The default workload is one prepared RULER needle-in-a-haystack sample. Its filler is
trimmed in token space when necessary while preserving the leading instruction, the
needle, and the trailing question. Unlike bench_decode_moe.py, this harness does not
issue a full-prompt warmup: exactly one expensive request is measured.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request

import bench_decode_moe as common


DEFAULT_RULER = (
    "/home/lucas/ai/bench/ruler_data/ruler/nemotron_256k/"
    "niah_single_1/test.jsonl"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--ruler", default=DEFAULT_RULER)
    p.add_argument("--sample", type=int, default=0)
    p.add_argument(
        "--synthetic-needle",
        action="store_true",
        help="use the built-in deterministic needle prompt instead of --ruler",
    )
    p.add_argument("--target-prompt-tokens", type=int, default=261_800)
    p.add_argument("--decode", type=int, default=128)
    p.add_argument("--max-context", type=int, default=262_144)
    p.add_argument("--rope-yarn-factor", type=float)
    p.add_argument("--rope-yarn-original-context", type=int)
    p.add_argument("--kv-cache-dtype", required=True)
    p.add_argument("--kv-cache-dtype-k")
    p.add_argument("--kv-cache-dtype-v")
    p.add_argument("--prefill-chunk", type=int, default=8192)
    p.add_argument("--mem-ratio", type=float, default=0.97)
    p.add_argument("--kv-grow-step-tokens", type=int)
    p.add_argument("--linear-state-slots", type=int)
    p.add_argument("--cache-policy", choices=("lru", "lfu"), default="lfu")
    p.add_argument("--server-timeout", type=float, default=1800)
    p.add_argument("--moe-pageable-gpu", action="store_true")
    p.add_argument("--host-ram-reserve-gb", type=float, default=3.0)
    p.add_argument("--json", dest="json_out")
    p.add_argument(
        "--baseline-json",
        help="compare against the last JSON/JSONL row and fail on a speed regression",
    )
    p.add_argument("--max-prefill-regression-pct", type=float, default=3.0)
    p.add_argument("--max-instant-prefill-regression-pct", type=float, default=5.0)
    p.add_argument("--max-decode-regression-pct", type=float, default=3.0)
    p.add_argument("--min-prefill-tok-s", type=float)
    p.add_argument("--min-instant-prefill-tok-s", type=float)
    p.add_argument("--min-decode-tok-s", type=float)
    # Attributes consumed by common.serve_cmd.
    p.set_defaults(
        backend="offload",
        cache=0,
        cache_rate=None,
        hybrid_fetch=-1,
        no_graph=False,
        prefill_hit_d2d=False,
    )
    return p.parse_args()


def load_row(path: str, index: int) -> tuple[str, str]:
    with open(path) as f:
        for row_index, line in enumerate(f):
            if row_index == index:
                row = json.loads(line)
                expected = row.get("expected_answer", "")
                if isinstance(expected, list):
                    expected = expected[0]
                return row["question"], str(expected)
    raise SystemExit(f"sample {index} not found in {path}")


def synthetic_needle_sample() -> tuple[str, str]:
    """Portable retrieval/coherence gate when the external RULER data is absent."""
    expected = "5663623"
    # Deliberately exceed the default 261.8K-token target for normal GGUF
    # tokenizers so the default invocation really exercises the requested
    # long-context length; trim_filler then preserves the centered needle.
    before = "The orchard ledger says the copper marker is inactive.\n" * 50_000
    after = "The harbor ledger says the silver marker is inactive.\n" * 50_000
    question = (
        "Read the records below. Remember the one secret passcode and ignore all "
        "inactive marker descriptions.\n\n"
        f"{before}\nThe secret passcode is {expected}.\n{after}\n"
        "What is the secret passcode? State the digits clearly."
    )
    return question, expected


def _needle_token_index(ids: list[int], needle_ids: list[int]) -> int:
    width = len(needle_ids)
    for index in range(len(ids) - width + 1):
        if ids[index : index + width] == needle_ids:
            return index
    raise ValueError("expected answer token sequence is absent from the RULER prompt")


def trim_filler(
    tokenizer, text: str, expected: str, target: int
) -> tuple[str, int, int]:
    """Remove only unprotected token spans, preserving needle and question."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    original = len(ids)
    if original <= target:
        return text, original, original
    if target < 4096:
        raise ValueError("target prompt must leave at least 4096 tokens for protected regions")

    needle_ids = tokenizer.encode(expected, add_special_tokens=False)
    needle = _needle_token_index(ids, needle_ids)
    # Protect the instruction, generous context around the needle, and the final
    # question. Remove from the largest filler gaps first, keeping source order.
    protected = [
        (0, min(512, original)),
        (max(0, needle - 512), min(original, needle + len(needle_ids) + 512)),
        (max(0, original - 2048), original),
    ]
    protected.sort()
    merged: list[tuple[int, int]] = []
    for begin, end in protected:
        if merged and begin <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((begin, end))
    gaps: list[tuple[int, int]] = []
    cursor = 0
    for begin, end in merged:
        if cursor < begin:
            gaps.append((cursor, begin))
        cursor = end
    if cursor < original:
        gaps.append((cursor, original))

    remove = original - target
    cuts: list[tuple[int, int]] = []
    for begin, end in sorted(gaps, key=lambda pair: pair[1] - pair[0], reverse=True):
        count = min(remove, end - begin)
        if count:
            cuts.append((begin, begin + count))
            remove -= count
        if remove == 0:
            break
    if remove:
        raise ValueError("target is smaller than the protected RULER regions")
    for begin, end in sorted(cuts, reverse=True):
        del ids[begin:end]

    trimmed = tokenizer.decode(ids, skip_special_tokens=False)
    actual = len(tokenizer.encode(trimmed, add_special_tokens=False))
    if expected not in trimmed or text[-512:] not in trimmed:
        raise AssertionError("filler trimming removed the needle or final question")
    return trimmed, original, actual


def stream_completion(
    origin: str, model_id: str, prompt: str, decode: int
) -> dict:
    body = {
        "model": model_id,
        "prompt": prompt,
        "max_tokens": decode,
        "ignore_eos": True,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        f"{origin}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    stamps: list[float] = []
    pieces: list[str] = []
    usage = None
    t0 = time.perf_counter()
    try:
        response = urllib.request.urlopen(req, timeout=7200)
    except urllib.error.HTTPError as error:
        raise SystemExit(
            f"request failed: HTTP {error.code}: {error.read()[:1000]!r}"
        ) from error
    with response:
        for raw in response:
            line = raw.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[len(b"data:") :].strip()
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
    if usage is None:
        raise SystemExit("stream ended without a usage chunk")
    return {"t0": t0, "stamps": stamps, "text": "".join(pieces), "usage": usage}


def prefill_rates(log_path: str) -> dict[str, float | int | None]:
    """Return the final scheduler-reported chunk and cumulative prefill rates."""
    pattern = re.compile(
        r"input throughput \(token/s\): ([0-9.]+) instant, ([0-9.]+) average"
    )
    samples: list[tuple[float, float]] = []
    with open(log_path, errors="replace") as server_log:
        for line in server_log:
            if match := pattern.search(line):
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


def load_last_json_row(path: str) -> dict:
    with open(path) as baseline_file:
        rows = [line for line in baseline_file if line.strip()]
    if not rows:
        raise ValueError(f"baseline file is empty: {path}")
    return json.loads(rows[-1])


def acceptance_failures(row: dict, args: argparse.Namespace) -> list[str]:
    """Apply absolute and paired-baseline performance gates after coherence passes."""
    failures: list[str] = []

    def metric(data: dict, key: str) -> float | None:
        value = data.get(key)
        if value is None and key == "prefill_average_tok_s":
            value = data.get("prefill_tok_s")
        return value

    absolute = (
        ("prefill_average_tok_s", args.min_prefill_tok_s, "average prefill"),
        (
            "prefill_instant_tok_s",
            args.min_instant_prefill_tok_s,
            "instant prefill",
        ),
        ("decode_tok_s", args.min_decode_tok_s, "decode"),
    )
    for key, minimum, label in absolute:
        value = metric(row, key)
        if minimum is not None and (value is None or value < minimum):
            failures.append(f"{label} {value!r} is below {minimum:.2f} tok/s")

    if args.baseline_json:
        baseline = load_last_json_row(args.baseline_json)
        incompatible = []
        for key in (
            "model",
            "expected",
            "prompt_tokens",
            "decode_tokens",
            "max_context",
            "kv_cache_dtype",
            "kv_cache_dtype_k",
            "kv_cache_dtype_v",
            "prefill_chunk",
        ):
            if key in baseline and key in row and baseline[key] != row[key]:
                incompatible.append(f"{key}={row[key]!r} vs {baseline[key]!r}")
        if incompatible:
            failures.append("incompatible performance baseline: " + ", ".join(incompatible))
            return failures
        relative = (
            (
                "prefill_average_tok_s",
                args.max_prefill_regression_pct,
                "average prefill",
            ),
            (
                "prefill_instant_tok_s",
                args.max_instant_prefill_regression_pct,
                "instant prefill",
            ),
            ("decode_tok_s", args.max_decode_regression_pct, "decode"),
        )
        for key, allowed_pct, label in relative:
            value, control = metric(row, key), metric(baseline, key)
            if value is None or control is None or control <= 0:
                failures.append(f"{label} cannot be compared ({value!r} vs {control!r})")
                continue
            floor = control * (1.0 - allowed_pct / 100.0)
            if value < floor:
                regression = (1.0 - value / control) * 100.0
                failures.append(
                    f"{label} regressed {regression:.2f}% "
                    f"({value:.2f} vs {control:.2f} tok/s; limit {allowed_pct:.2f}%)"
                )
    return failures


def main() -> int:
    args = parse_args()
    if args.synthetic_needle:
        question, expected = synthetic_needle_sample()
        workload = "built-in synthetic needle"
    else:
        question, expected = load_row(args.ruler, args.sample)
        workload = f"{args.ruler} sample={args.sample}"
    from freetoken.models.gguf.tokenizer import load_gguf_tokenizer

    tokenizer = load_gguf_tokenizer(args.model)
    prompt, original_tokens, trimmed_tokens = trim_filler(
        tokenizer, question, expected, args.target_prompt_tokens
    )
    if trimmed_tokens + args.decode > args.max_context:
        raise SystemExit(
            f"prompt+decode={trimmed_tokens + args.decode} exceeds context "
            f"{args.max_context}"
        )

    port = common.free_port()
    origin = f"http://127.0.0.1:{port}"
    fd, log_path = tempfile.mkstemp(prefix="bench-long-offload-", suffix=".log")
    cmd = common.serve_cmd(args, "offload", port)
    if args.kv_cache_dtype_k is not None:
        cmd += ["--kv-cache-dtype-k", args.kv_cache_dtype_k]
    if args.kv_cache_dtype_v is not None:
        cmd += ["--kv-cache-dtype-v", args.kv_cache_dtype_v]
    if args.kv_cache_dtype_k is not None or args.kv_cache_dtype_v is not None:
        cmd += ["--attention-backend", "triton"]
    if args.rope_yarn_factor is not None:
        cmd += ["--rope-yarn-factor", str(args.rope_yarn_factor)]
    if args.rope_yarn_original_context is not None:
        cmd += [
            "--rope-yarn-original-context",
            str(args.rope_yarn_original_context),
        ]
    if args.moe_pageable_gpu:
        cmd.append("--moe-pageable-gpu")
    if args.linear_state_slots is not None:
        cmd += ["--linear-state-slots", str(args.linear_state_slots)]
    cmd += ["--host-ram-reserve-gb", str(args.host_ram_reserve_gb)]
    print(
        f"[long] model={args.model}\n"
        f"[long] workload={workload} expected={expected!r}\n"
        f"[long] prompt tokens: original={original_tokens}, trimmed={trimmed_tokens}, "
        f"target={args.target_prompt_tokens}\n"
        f"[long] context={args.max_context} decode={args.decode} "
        f"kv={args.kv_cache_dtype} k={args.kv_cache_dtype_k or 'inherit'} "
        f"v={args.kv_cache_dtype_v or 'inherit'} chunk={args.prefill_chunk} "
        f"ratio={args.mem_ratio}\n"
        f"[long] server log: {log_path}",
        flush=True,
    )

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
            result = stream_completion(origin, model_id, prompt, args.decode)
            stats = common.get_json(f"{origin}/v1/stats")
        finally:
            common.stop_server(proc)
            pump.join(timeout=10)

    stamps = result["stamps"]
    usage = result["usage"]
    if len(stamps) < 2:
        raise SystemExit(f"need >=2 streamed token events, got {len(stamps)}")
    completion = int(usage["completion_tokens"])
    decode_steps = completion - 1
    decode_seconds = stamps[-1] - stamps[0]
    ttft_seconds = stamps[0] - result["t0"]
    prompt_tokens = int(usage["prompt_tokens"])
    row = {
        "model": args.model,
        "workload": workload,
        "ruler": None if args.synthetic_needle else args.ruler,
        "sample": args.sample,
        "expected": expected,
        "expected_found": expected in result["text"],
        "prompt_tokens": prompt_tokens,
        "original_ornith_tokens": original_tokens,
        "target_prompt_tokens": args.target_prompt_tokens,
        "max_context": args.max_context,
        "decode_tokens": completion,
        "ttft_seconds": ttft_seconds,
        "prefill_tok_s": prompt_tokens / ttft_seconds,
        "decode_tok_s": decode_steps / decode_seconds,
        "ms_per_decode_token": decode_seconds / decode_steps * 1000,
        "vram_gib": stats.get("vram_bytes", 0) / 2**30,
        "kv_cache_dtype": args.kv_cache_dtype,
        "kv_cache_dtype_k": args.kv_cache_dtype_k,
        "kv_cache_dtype_v": args.kv_cache_dtype_v,
        "prefill_chunk": args.prefill_chunk,
        "memory_ratio": args.mem_ratio,
        "kv_grow_step_tokens": args.kv_grow_step_tokens,
        "output_sha1": hashlib.sha1(result["text"].encode()).hexdigest()[:12],
        "output": result["text"],
        "server_log": log_path,
    }
    row.update(prefill_rates(log_path))
    failures = acceptance_failures(row, args)
    if not row["expected_found"]:
        failures.insert(0, f"expected answer {expected!r} was absent")
    row["acceptance_failures"] = failures
    row["accepted"] = not failures
    print("\n==== cold long-context result ====", flush=True)
    print(
        f"  prompt/prefill : {prompt_tokens} tokens in {ttft_seconds:.3f} s "
        f"({row['prefill_tok_s']:.2f} tok/s end-to-end)"
    )
    print(
        f"  prefill engine : {row['prefill_instant_tok_s']} instant / "
        f"{row['prefill_average_tok_s']} average tok/s"
    )
    print(
        f"  decode         : {decode_steps} steps in {decode_seconds:.3f} s "
        f"({row['decode_tok_s']:.2f} tok/s)"
    )
    print(f"  expected       : {expected!r}, found={row['expected_found']}")
    print(f"  output         : {result['text']!r}")
    print(f"  VRAM           : {row['vram_gib']:.2f} GiB")
    print(f"  acceptance     : {'PASS' if row['accepted'] else 'FAIL'}")
    for failure in failures:
        print(f"    - {failure}")
    if args.json_out:
        with open(args.json_out, "a") as f:
            f.write(json.dumps(row) + "\n")
    return 0 if row["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
