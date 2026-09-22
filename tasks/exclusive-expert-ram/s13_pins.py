#!/usr/bin/env python3
"""Plan S13 Verify: does a session's haystack survive between its questions?

Run as measure.sh's FT_POST (FREETOKEN_URL / FREETOKEN_MODEL_NAME point at the arm).

    s13_pins.py [haystack_tokens=112000] [handoff_tokens=32000]

Sequence, one lane, nothing interleaved:
  A1  haystack + question 1, session key A   (the uncached prefill)
  A2  haystack + question 2, key A           cached_tokens >= haystack - 8192, TTFT < 3 s
  A3  haystack + question 3, key A           same
  B1  another haystack, key B                (the handoff: B admitted while A's pin is held)
  A4  haystack + question 4, key A           must still hit

Every request is streamed so TTFT is the first content/reasoning delta; the usage chunk
(stream_options.include_usage) carries prompt_tokens_details.cached_tokens
(--enable-cache-report). /v1/stats scheduler.prefix is sampled after every request.
The PASS lines are plan S13's criteria; the output JSON is the record.
Environment: S13_OUT (JSON path), S13_KEY_A / S13_KEY_B (session ids).
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recall import build  # noqa: E402  same deterministic haystack builder

URL = os.environ.get("FREETOKEN_URL", "http://127.0.0.1:1920")
NAME = os.environ.get("FREETOKEN_MODEL_NAME", "nemotron-3.5-lightning")
QUESTIONS = (
    "What is the first authorization code in the text above? Answer with the code only.",
    "What is the second authorization code in the text above? Answer with the code only.",
    "What is the third authorization code in the text above? Answer with the code only.",
    "List the three authorization codes in the text above, in order, and nothing else.",
)


def prefix_stats() -> dict:
    with urllib.request.urlopen(f"{URL}/v1/stats", timeout=30) as r:
        return (json.load(r).get("scheduler") or {}).get("prefix") or {}


def ask(prompt: str, question: str, key: str) -> dict:
    req = urllib.request.Request(
        f"{URL}/v1/chat/completions",
        data=json.dumps({
            "model": NAME,
            "messages": [{"role": "user", "content": prompt + "\n\n" + question}],
            "max_tokens": 48,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode(),
        headers={"Content-Type": "application/json", "x-session-id": key},
    )
    t0 = time.perf_counter()
    ttft = None
    text, usage = [], {}
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                piece = d.get("content") or d.get("reasoning_content")
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    text.append(d.get("content") or "")
    details = usage.get("prompt_tokens_details") or {}
    return {
        "key": key,
        "ttft_s": round(ttft, 3) if ttft is not None else None,
        "total_s": round(time.perf_counter() - t0, 3),
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": details.get("cached_tokens"),
        "answer": "".join(text).strip()[:200],
    }


def main(argv: list[str]) -> int:
    hay_tokens = int(argv[0]) if argv else 112000
    handoff_tokens = int(argv[1]) if len(argv) > 1 else 32000
    key_a = os.environ.get("S13_KEY_A", "s13-session-a")
    key_b = os.environ.get("S13_KEY_B", "s13-session-b")
    hay_a, codes_a = build(hay_tokens, random.Random(hay_tokens))
    hay_b, _ = build(handoff_tokens, random.Random(handoff_tokens + 1))

    steps = []
    def run(label, prompt, q, key):
        r = ask(prompt, q, key)
        r["label"] = label
        r["prefix_after"] = prefix_stats()
        steps.append(r)
        print(json.dumps(r), flush=True)
        return r

    steps_start = prefix_stats()
    a1 = run("A1", hay_a, QUESTIONS[0], key_a)
    run("A2", hay_a, QUESTIONS[1], key_a)
    run("A3", hay_a, QUESTIONS[2], key_a)
    run("B1", hay_b, QUESTIONS[3], key_b)
    run("A4", hay_a, QUESTIONS[3], key_a)

    floor = (a1["prompt_tokens"] or hay_tokens) - 8192
    by = {s["label"]: s for s in steps}
    checks = {
        "A2/A3 cached >= prompt-8192": all((by[k]["cached_tokens"] or 0) >= floor for k in ("A2", "A3")),
        "A2/A3 TTFT < 3 s": all((by[k]["ttft_s"] or 1e9) < 3.0 for k in ("A2", "A3")),
        "A4 after handoff cached >= prompt-8192": (by["A4"]["cached_tokens"] or 0) >= floor,
        "pin_budget_refusals == 0": steps[-1]["prefix_after"].get("pin_budget_refusals", 0)
            - steps_start.get("pin_budget_refusals", 0) == 0,
        "codes": [c in by[k]["answer"] for k, c in (("A1", codes_a[0]), ("A2", codes_a[1]), ("A3", codes_a[2]))],
    }
    for k, v in checks.items():
        print(f"{'PASS' if (all(v) if isinstance(v, list) else v) else 'FAIL'}  {k}  {v if isinstance(v, list) else ''}")
    out = os.environ.get("S13_OUT")
    if out:
        with open(out, "w") as f:
            json.dump({"haystack_tokens": hay_tokens, "planted": codes_a, "start_prefix": steps_start,
                       "steps": steps, "checks": checks}, f, indent=1)
    return 0 if all((all(v) if isinstance(v, list) else v) for v in checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
