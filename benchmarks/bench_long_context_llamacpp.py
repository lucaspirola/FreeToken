"""Cross-engine needle check: the *same* prompt as bench_long_context.py, but graded
through an already-running llama.cpp ``llama-server``.

Why a separate entry point: bench_long_context.py owns the FreeToken server lifecycle
(it builds the ``ft serve`` command line and waits for it). Here the server is started
by hand under ``scripts/gpu_lock.sh`` with llama.cpp flags, so this script only builds
the prompt and grades one chat completion against ``--base-url``.

The prompt is byte-identical to the FreeToken bisect's: same
``synthetic_needle_sample`` + ``trim_filler``, tokenized with the *HuggingFace*
checkpoint tokenizer (``--tokenizer``), so the token positions quoted in
``benchmarks/results/nemotron35_lightning_5080_262k_bisect_2026-09-04.md`` carry over.
llama.cpp retokenizes it with the GGUF vocabulary; the reported ``prompt_tokens`` is
recorded so any drift is visible.

Grading goes through ``/v1/chat/completions`` (never a raw completion, never a grep of
the raw SSE frames): the JSON deltas are concatenated first, then searched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench_long_context import (  # noqa: E402
    load_tokenizer,
    synthetic_needle_sample,
    trim_filler,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://127.0.0.1:8080")
    p.add_argument(
        "--tokenizer",
        default="/home/lucas/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
        help="HF checkpoint dir used to size/trim the prompt (must match the FreeToken run)",
    )
    p.add_argument("--needle-depth", type=float, default=0.5)
    p.add_argument("--target-prompt-tokens", type=int, default=262_144)
    p.add_argument("--decode", type=int, default=48)
    p.add_argument("--timeout", type=float, default=7200)
    p.add_argument("--prompt-cache", help="directory of pre-built prompt .txt files")
    p.add_argument("--json", dest="json_out")
    p.add_argument("--label", default="")
    p.add_argument(
        "--build-only",
        action="store_true",
        help="materialise the prompt into --prompt-cache and exit (no server needed)",
    )
    return p.parse_args()


def build_prompt(args) -> tuple[str, str, int, int]:
    expected = "5663623"
    cache_path = None
    if args.prompt_cache:
        os.makedirs(args.prompt_cache, exist_ok=True)
        cache_path = os.path.join(
            args.prompt_cache,
            f"needle_{args.target_prompt_tokens}_d{args.needle_depth:.2f}.txt",
        )
        meta_path = cache_path + ".json"
        if os.path.exists(cache_path) and os.path.exists(meta_path):
            with open(cache_path) as f:
                prompt = f.read()
            meta = json.load(open(meta_path))
            return prompt, expected, meta["original_tokens"], meta["trimmed_tokens"]

    question, expected = synthetic_needle_sample(args.needle_depth)
    tokenizer = load_tokenizer(args.tokenizer)
    prompt, original_tokens, trimmed_tokens = trim_filler(
        tokenizer, question, expected, args.target_prompt_tokens
    )
    if cache_path:
        with open(cache_path, "w") as f:
            f.write(prompt)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        needle_ids = tokenizer.encode(expected, add_special_tokens=False)
        needle_at = next(
            i
            for i in range(len(ids) - len(needle_ids) + 1)
            if ids[i : i + len(needle_ids)] == needle_ids
        )
        json.dump(
            {
                "original_tokens": original_tokens,
                "trimmed_tokens": trimmed_tokens,
                "needle_token": needle_at,
                "measured_depth": needle_at / trimmed_tokens,
                "sha1": hashlib.sha1(prompt.encode()).hexdigest(),
            },
            open(cache_path + ".json", "w"),
            indent=1,
        )
    return prompt, expected, original_tokens, trimmed_tokens


def chat(base_url: str, model_id: str, prompt: str, decode: int, timeout: float) -> dict:
    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": decode,
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "seed": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    stamps: list[float] = []
    pieces: list[str] = []
    usage = None
    t0 = time.perf_counter()
    try:
        response = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as error:
        raise SystemExit(f"request failed: HTTP {error.code}: {error.read()[:2000]!r}")
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
                delta = choice.get("delta") or {}
                piece = "".join(
                    str(delta[k])
                    for k in ("content", "reasoning_content", "reasoning")
                    if delta.get(k)
                )
                if piece:
                    stamps.append(now)
                    pieces.append(piece)
    return {"t0": t0, "stamps": stamps, "text": "".join(pieces), "usage": usage or {}}


def main() -> int:
    args = parse_args()
    prompt, expected, original_tokens, trimmed_tokens = build_prompt(args)
    print(
        f"[llamacpp] depth={args.needle_depth} target={args.target_prompt_tokens} "
        f"hf-tokens={trimmed_tokens} (orig {original_tokens}) "
        f"sha1={hashlib.sha1(prompt.encode()).hexdigest()[:12]}",
        flush=True,
    )
    if args.build_only:
        return 0

    with urllib.request.urlopen(f"{args.base_url}/v1/models", timeout=60) as r:
        model_id = json.load(r)["data"][0]["id"]
    result = chat(args.base_url, model_id, prompt, args.decode, args.timeout)

    usage = result["usage"]
    stamps = result["stamps"]
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or len(stamps))
    ttft = (stamps[0] - result["t0"]) if stamps else float("nan")
    decode_s = (stamps[-1] - stamps[0]) if len(stamps) > 1 else float("nan")
    row = {
        "engine": "llama.cpp",
        "label": args.label,
        "needle_depth": args.needle_depth,
        "target_prompt_tokens": args.target_prompt_tokens,
        "hf_prompt_tokens": trimmed_tokens,
        "prompt_tokens": prompt_tokens,
        "expected": expected,
        "expected_found": expected in result["text"],
        "decode_tokens": completion,
        "ttft_seconds": ttft,
        "prefill_tok_s": prompt_tokens / ttft if ttft and ttft == ttft else None,
        "decode_tok_s": (len(stamps) - 1) / decode_s if decode_s == decode_s else None,
        "output": result["text"],
        "output_sha1": hashlib.sha1(result["text"].encode()).hexdigest()[:12],
    }
    print(json.dumps({k: v for k, v in row.items() if k != "output"}, indent=1))
    print(f"  output   : {result['text']!r}")
    print(f"  NEEDLE   : {'PASS' if row['expected_found'] else 'FAIL'}")
    if args.json_out:
        with open(args.json_out, "a") as f:
            f.write(json.dumps(row) + "\n")
    return 0 if row["expected_found"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
