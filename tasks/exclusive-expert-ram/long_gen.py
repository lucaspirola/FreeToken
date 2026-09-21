#!/usr/bin/env python3
"""Long generation: the workload this branch has never actually exercised.

Why
---
Every probe in this campaign generates 128 tokens. That is a kernel benchmark,
not the workload these models ship for. Long output is the ONLY case where:

  * the growable KV grows DURING decode (every other probe grows it during
    prefill, then decodes against a fixed arena),
  * the pool's eviction mix reaches steady state -- 128 tokens never gets past
    the experts the warm start already seated, so a 128-token number reports
    the warm start, not the policy,
  * a retained-copy leak or a reserve starvation has time to surface.

So this streams a genuinely long answer and reports decode rate PER 1K-TOKEN
WINDOW rather than one average: the question is not "how fast was it" but
"did it stay that fast to the end". An average hides a cliff at token 30K.

It runs with thinking ON by default, because that is how the server serves
(both chat templates default ``enable_thinking`` true) and because the answer
and the reasoning are billed from the same output budget -- a cap that fits the
answer but not the thinking truncates mid-thought.

Correctness is the output, as always: run the same prompt against the
whole-model arm at temperature 0 and compare. A pool arm whose text degrades
after N tokens is a CORRECTNESS finding, not a performance one, and no counter
will show it.

    long_gen.py --max-tokens 65536
    long_gen.py --max-tokens 16384 --prompt-tokens 240000 --out results/x.json

Accept (plan Phase 7): decode in the last full 1K window within 10 % of the
first, 0 starved writebacks, 0 coverage faults, finish_reason eos (or length at
the cap with coherent text).

Environment: FREETOKEN_URL (default http://127.0.0.1:1920),
FREETOKEN_MODEL_NAME (default nemotron-3.5-lightning).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
import urllib.request

URL = os.environ.get("FREETOKEN_URL", "http://127.0.0.1:1920")
NAME = os.environ.get("FREETOKEN_MODEL_NAME", "nemotron-3.5-lightning")

_WORDS = ("ledger entry audit column revision batch quarter invoice reconcile "
          "variance schedule appendix clause remittance").split()

# A task that legitimately needs a long answer: it has a natural structure the
# model can keep extending, so a truncation at the cap is still coherent text
# rather than a model that ran out of things to say at token 900.
TASK = (
    "Write a complete, self-contained Python module that implements a "
    "fixed-capacity cache with a least-frequently-used eviction policy and an "
    "optional write-back tier, of the kind a GPU expert cache would need. "
    "Include: the full implementation with type hints and docstrings; a "
    "thorough unittest suite covering eviction order, ties, the write-back "
    "path, concurrent access and every boundary condition you can think of; "
    "and then a detailed written walkthrough of the design, the alternatives "
    "you rejected and why, and the failure modes a caller should expect. "
    "Think the design through step by step before you write it, and be "
    "exhaustive -- do not summarize or abbreviate any section."
)


def _stats() -> dict:
    try:
        with urllib.request.urlopen(f"{URL}/v1/stats", timeout=10) as r:
            return json.load(r)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _flatten(stats: dict) -> dict:
    """The few counters this test is about, wherever the build nests them."""
    out: dict = {}
    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if k in ("coverage_faults", "starved_writebacks", "swaps",
                         "retained_rows", "free_eviction_rate",
                         "moe_cache_size", "kv_tokens", "graph_captures"):
                    out.setdefault(k, v)
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
    walk(stats)
    return out


def build_prompt(prompt_tokens: int) -> str:
    """The task, optionally preceded by filler to force a large KV base."""
    if prompt_tokens <= 0:
        return TASK
    rng = random.Random(20260922)
    words = int(prompt_tokens / 1.3)
    filler = " ".join(rng.choice(_WORDS) for _ in range(words))
    return ("Reference log (ignore its contents; it is context only):\n"
            + filler + "\n\n" + TASK)


def run(max_tokens: int, prompt_tokens: int, thinking: bool,
        sample_every: int) -> dict:
    prompt = build_prompt(prompt_tokens)
    payload = {
        "model": NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": bool(thinking)},
    }
    req = urllib.request.Request(
        f"{URL}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})

    result: dict = {"max_tokens": max_tokens, "prompt_tokens_requested":
                    prompt_tokens, "thinking": thinking}
    counters: list[dict] = []
    stop = threading.Event()
    n_tokens = [0]

    def sampler():
        """/v1/stats every ``sample_every`` output tokens.

        A separate thread, because sampling inline between SSE chunks would
        charge its own latency to the decode rate we are measuring.
        """
        next_at = sample_every
        while not stop.wait(0.5):
            if n_tokens[0] >= next_at:
                counters.append({"at_token": n_tokens[0],
                                 **_flatten(_stats())})
                next_at += sample_every

    counters.append({"at_token": 0, **_flatten(_stats())})
    th = threading.Thread(target=sampler, daemon=True)
    th.start()

    answer: list[str] = []
    reasoning: list[str] = []
    stamps: list[float] = []          # wall clock of every output token
    t0 = time.perf_counter()
    ttft = None
    usage = {}
    finish = None
    try:
        with urllib.request.urlopen(req, timeout=86400) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices") or ():
                    delta = choice.get("delta") or {}
                    piece = delta.get("content")
                    think = delta.get("reasoning_content")
                    if think:
                        reasoning.append(think)
                    if piece:
                        answer.append(piece)
                    if piece or think:
                        now = time.perf_counter()
                        if ttft is None:
                            ttft = round(now - t0, 3)
                        stamps.append(now)
                        n_tokens[0] += 1
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
    finally:
        stop.set()
        th.join(timeout=3)

    counters.append({"at_token": n_tokens[0], **_flatten(_stats())})

    # Decode rate per 1K-token window. Windows are measured between token
    # timestamps, so the prefill is excluded from every one of them.
    windows = []
    for start in range(0, len(stamps) - 1, 1000):
        end = min(start + 1000, len(stamps)) - 1
        if end - start < 50:                      # a stub window says nothing
            continue
        span = stamps[end] - stamps[start]
        if span > 0:
            windows.append({"from": start, "to": end,
                            "tok_s": round((end - start) / span, 1)})

    result.update({
        "ttft_s": ttft,
        "total_s": round(time.perf_counter() - t0, 2),
        "stream_tokens": n_tokens[0],
        "usage": usage,
        "finish_reason": finish,
        "reasoning_chars": sum(len(x) for x in reasoning),
        "answer_chars": sum(len(x) for x in answer),
        "windows": windows,
        "first_window_tok_s": windows[0]["tok_s"] if windows else None,
        "last_window_tok_s": windows[-1]["tok_s"] if windows else None,
        "counters": counters,
        "answer_head": "".join(answer)[:600],
        "answer_tail": "".join(answer)[-600:],
    })
    first, last = result["first_window_tok_s"], result["last_window_tok_s"]
    if first and last:
        result["last_vs_first"] = round(last / first, 3)
        result["within_10pct"] = last >= 0.9 * first
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-tokens", type=int, default=65536)
    ap.add_argument("--prompt-tokens", type=int, default=0,
                    help="filler tokens before the task (KV growth during "
                         "decode from a large base)")
    ap.add_argument("--no-thinking", action="store_true",
                    help="measure the kernel, not the served workload")
    ap.add_argument("--sample-every", type=int, default=4096)
    ap.add_argument("--out", help="write the JSON result here")
    args = ap.parse_args()

    res = run(args.max_tokens, args.prompt_tokens, not args.no_thinking,
              args.sample_every)
    text = json.dumps(res, indent=1)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
    print(f"tokens={res['stream_tokens']} finish={res['finish_reason']} "
          f"ttft={res['ttft_s']} first={res['first_window_tok_s']} "
          f"last={res['last_window_tok_s']} "
          f"ratio={res.get('last_vs_first')} "
          f"reasoning_chars={res['reasoning_chars']}", flush=True)
    if not args.out:
        print(text)
    # Non-zero only on the acceptance gate, so a sweep script can branch.
    return 0 if res.get("within_10pct", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
