#!/usr/bin/env python3
"""Measure prefill tok/s, TTFT and steady decode tok/s vs prompt size on one lane.

    scripts/probe_decode.py [SIZE ...]        default sizes 8000 32000 80000 128000 256000
    FREETOKEN_URL (default http://127.0.0.1:1919) and FREETOKEN_MODEL_NAME
    (default nemotron-3.5-lightning) select the server. One JSON line per size.
"""
import json
import os
import sys
import time
import urllib.request

URL = os.environ.get("FREETOKEN_URL", "http://127.0.0.1:1919") + "/v1/chat/completions"
MODEL = os.environ.get("FREETOKEN_MODEL_NAME", "nemotron-3.5-lightning")
SENT = "The quick brown fox jumps over the lazy dog near the riverbank while the miller counts his sacks of grain. "
SIZES = [int(x) for x in (sys.argv[1:] or ["8000", "32000", "80000", "128000", "256000"])]
GEN = int(os.environ.get("PROBE_GEN_TOKENS", "128"))


def run(target_tokens: int) -> dict:
    reps = max(1, int(target_tokens / 23))  # ~23 tokens per sentence
    prompt = f"Run {target_tokens}. " + SENT * reps + "\n\nWrite a long story about the fox."
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": GEN, "temperature": 0, "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time(); first = None; n = 0; usage = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            d = json.loads(payload)
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices", []):
                delta = ch.get("delta", {})
                if delta.get("content") or delta.get("reasoning_content"):
                    n += 1
                    if first is None:
                        first = time.time()
    t1 = time.time()
    ttft = (first or t1) - t0
    dec = (t1 - first) if first and n > 1 else 0.0
    pt = (usage or {}).get("prompt_tokens", 0)
    ct = (usage or {}).get("completion_tokens", n)
    return {"prompt_tokens": pt, "ttft_s": round(ttft, 2),
            "prefill_tok_s": round(pt / ttft, 0) if ttft else None,
            "gen_tokens": ct, "decode_tok_s": round((ct - 1) / dec, 1) if dec else None,
            "total_s": round(t1 - t0, 1)}


for s in SIZES:
    print(json.dumps({"target": s, **run(s)}), flush=True)
