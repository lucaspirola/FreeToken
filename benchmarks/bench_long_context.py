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
import subprocess
import sys
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
    p.add_argument("--target-prompt-tokens", type=int, default=261_800)
    p.add_argument("--decode", type=int, default=128)
    p.add_argument("--max-context", type=int, default=262_144)
    p.add_argument("--kv-cache-dtype", required=True)
    p.add_argument("--prefill-chunk", type=int, default=8192)
    p.add_argument("--mem-ratio", type=float, default=0.97)
    p.add_argument("--cache-policy", choices=("lru", "lfu"), default="lfu")
    p.add_argument("--server-timeout", type=float, default=1800)
    p.add_argument("--json", dest="json_out")
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


def main() -> int:
    args = parse_args()
    question, expected = load_row(args.ruler, args.sample)
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
    print(
        f"[long] model={args.model}\n"
        f"[long] ruler={args.ruler} sample={args.sample} expected={expected!r}\n"
        f"[long] prompt tokens: original={original_tokens}, trimmed={trimmed_tokens}, "
        f"target={args.target_prompt_tokens}\n"
        f"[long] context={args.max_context} decode={args.decode} "
        f"kv={args.kv_cache_dtype} chunk={args.prefill_chunk} ratio={args.mem_ratio}\n"
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
        "ruler": args.ruler,
        "sample": args.sample,
        "expected": expected,
        "expected_found": expected in result["text"],
        "prompt_tokens": prompt_tokens,
        "original_ornith_tokens": original_tokens,
        "target_prompt_tokens": args.target_prompt_tokens,
        "decode_tokens": completion,
        "ttft_seconds": ttft_seconds,
        "prefill_tok_s": prompt_tokens / ttft_seconds,
        "decode_tok_s": decode_steps / decode_seconds,
        "ms_per_decode_token": decode_seconds / decode_steps * 1000,
        "vram_gib": stats.get("vram_bytes", 0) / 2**30,
        "kv_cache_dtype": args.kv_cache_dtype,
        "prefill_chunk": args.prefill_chunk,
        "memory_ratio": args.mem_ratio,
        "output_sha1": hashlib.sha1(result["text"].encode()).hexdigest()[:12],
        "output": result["text"],
        "server_log": log_path,
    }
    print("\n==== cold long-context result ====", flush=True)
    print(
        f"  prompt/prefill : {prompt_tokens} tokens in {ttft_seconds:.3f} s "
        f"({row['prefill_tok_s']:.2f} tok/s)"
    )
    print(
        f"  decode         : {decode_steps} steps in {decode_seconds:.3f} s "
        f"({row['decode_tok_s']:.2f} tok/s)"
    )
    print(f"  expected       : {expected!r}, found={row['expected_found']}")
    print(f"  output         : {result['text']!r}")
    print(f"  VRAM           : {row['vram_gib']:.2f} GiB")
    if args.json_out:
        with open(args.json_out, "a") as f:
            f.write(json.dumps(row) + "\n")
    return 0 if row["expected_found"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
