#!/usr/bin/env python3
"""Long-context recall against a running FreeToken server.

Correctness under the mirror is not "it did not crash" and it is not "the fault
counters are zero" -- a stale free-row publish once served the WRONG experts
with coverage_faults at 0 (see plan.md). The only witness that the right expert
weights reached the GEMM is the output, so this plants a marker deep inside a
long prompt and asks for it back.

Markers are planted at three depths (start, middle, end) because a cache that
serves wrong experts usually still parrots the most recent tokens.

    recall.py 21000 240000 713000

Environment: FREETOKEN_URL (default http://127.0.0.1:1920),
FREETOKEN_MODEL_NAME (default nemotron-3.5-lightning).
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
import urllib.request

URL = os.environ.get("FREETOKEN_URL", "http://127.0.0.1:1920")
NAME = os.environ.get("FREETOKEN_MODEL_NAME", "nemotron-3.5-lightning")

# Filler that tokenizes densely and carries no answer of its own.
_WORDS = ("ledger entry audit column revision batch quarter invoice reconcile "
          "variance schedule appendix clause remittance").split()


def _marker(rng: random.Random) -> tuple[str, str]:
    code = "-".join((
        rng.choice(["SIERRA", "TANGO", "KILO", "ROMEO", "ZULU"]),
        str(rng.randrange(1000, 9999)),
    ))
    return code, f"The authorization code for this section is {code}."


def build(tokens: int, rng: random.Random) -> tuple[str, list[str]]:
    """A prompt of roughly ``tokens`` tokens with three planted codes."""
    # ~1.3 tokens per word for this filler; overshoot slightly and let the
    # server's own count be the truth (the probe prints prompt_tokens).
    words = int(tokens / 1.3)
    body = [rng.choice(_WORDS) for _ in range(words)]
    codes = []
    for frac in (0.02, 0.5, 0.97):
        code, sentence = _marker(rng)
        codes.append(code)
        body.insert(int(len(body) * frac), sentence)
    return " ".join(body), codes


def ask(prompt: str, codes: list[str]) -> dict:
    question = ("\n\nList the three authorization codes that appear in the text "
                "above, in the order they appear, and nothing else.")
    req = urllib.request.Request(
        f"{URL}/v1/chat/completions",
        data=json.dumps({
            "model": NAME,
            "messages": [{"role": "user", "content": prompt + question}],
            "max_tokens": 64,
            "temperature": 0.0,
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=3600) as r:
        body = json.load(r)
    text = body["choices"][0]["message"]["content"]
    found = [c for c in codes if c in text]
    return {
        "prompt_tokens": body.get("usage", {}).get("prompt_tokens"),
        "seconds": round(time.perf_counter() - t0, 1),
        "planted": codes,
        "recalled": found,
        "all_three": len(found) == 3,
        "answer": text.strip()[:200],
    }


def main(sizes: list[str]) -> int:
    rng = random.Random(20260921)
    bad = 0
    for raw in sizes:
        target = int(raw)
        prompt, codes = build(target, rng)
        try:
            res = {"target": target, **ask(prompt, codes)}
        except Exception as exc:                      # a failure is a result
            res = {"target": target, "error": f"{type(exc).__name__}: {exc}"}
        if not res.get("all_three"):
            bad += 1
        print(json.dumps(res), flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or ["21000", "240000", "713000"]))
