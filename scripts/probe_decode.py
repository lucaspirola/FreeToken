#!/usr/bin/env python3
"""Measure prefill tok/s, TTFT and steady decode tok/s vs prompt size on one lane.

    scripts/probe_decode.py [SIZE ...]        default sizes 8000 32000 80000 128000 256000
    FREETOKEN_URL (default http://127.0.0.1:1919) and FREETOKEN_MODEL_NAME
    (default nemotron-3.5-lightning) select the server. One JSON line per size.
"""
import hashlib
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
# PROBE_PASSES=2 runs every size twice, each pass with its own prompt prefix so
# the second is not a radix-cache hit on the first. The point is the growable KV:
# the FIRST request that needs a bigger arena pays one commit plus a decode-graph
# recapture, and where that stall lands decides which number it lands in. On an
# 80K prompt it showed up as ttft 11.94 / decode 127.7 in one arm and ttft 9.90 /
# decode 46.4 in the next, for the same 12.6-12.9 s of total wall clock -- the
# arms differed in whether the stall fell before or after the first token, not in
# how fast they decode. Pass 2 needs no growth, so its decode is the steady one.
PASSES = max(1, int(os.environ.get("PROBE_PASSES", "1")))


def run(target_tokens: int, tag: str = "") -> dict:
    reps = max(1, int(target_tokens / 23))  # ~23 tokens per sentence
    prompt = (f"Run {tag}{target_tokens}. " + SENT * reps
              + "\n\nWrite a long story about the fox.")
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": GEN, "temperature": 0, "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    # Both clocks: time.time() keeps the numbers comparable with every earlier record,
    # time.monotonic() cannot jump (WSL2's wall clock steps ~1.7-2 s every ~34 s, which
    # an 80K TTFT straddles). The *_mono_s fields are the ones to trust.
    t0 = time.time(); m0 = time.monotonic(); first = None; mfirst = None; n = 0; usage = None
    arrivals = []  # monotonic arrival of every content chunk: shows a stall inside decode
    text = []  # the streamed output, hashed into out_sha1 (greedy: same code + lane -> same hash)
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
                    text.append((delta.get("reasoning_content") or "") + (delta.get("content") or ""))
                    n += 1
                    arrivals.append(time.monotonic())
                    if first is None:
                        first = time.time(); mfirst = time.monotonic()
    t1 = time.time(); m1 = time.monotonic()
    ttft = (first or t1) - t0
    ttft_m = (mfirst or m1) - m0
    dec = (t1 - first) if first and n > 1 else 0.0
    dec_m = (m1 - mfirst) if mfirst and n > 1 else 0.0
    pt = (usage or {}).get("prompt_tokens", 0)
    ct = (usage or {}).get("completion_tokens", n)
    return {"prompt_tokens": pt, "ttft_s": round(ttft, 2),
            "prefill_tok_s": round(pt / ttft, 0) if ttft else None,
            # decode from the monotonic clock: a WSL2 wall-clock step inside a 2 s decode
            # window turned 82.7 tok/s into 56.7 (ck4 mirror-1m, 1M pass 2)
            "gen_tokens": ct, "decode_tok_s": round((ct - 1) / dec_m, 1) if dec_m else None,
            "decode_tok_s_wall": round((ct - 1) / dec, 1) if dec else None,
            "total_s": round(t1 - t0, 1),
            "ttft_mono_s": round(ttft_m, 3),
            "prefill_tok_s_mono": round(pt / ttft_m, 0) if ttft_m else None,
            "total_mono_s": round(m1 - m0, 2),
            "t_start_wall": round(t0, 3),
            # sha1 of the streamed output text: lets two arms (KV lanes, residencies)
            # say "identical output" or not without storing the text.
            "out_sha1": hashlib.sha1("".join(text).encode()).hexdigest(),
            # Gap between the first and second streamed chunk (a server-side stall right
            # after the first token, e.g. the dynamic-headroom release, lands here), the
            # largest gap anywhere in decode, and decode excluding that first gap.
            "gap1_ms": round((arrivals[1] - arrivals[0]) * 1e3, 1) if len(arrivals) > 1 else None,
            "max_gap_ms": round(max(b - a for a, b in zip(arrivals, arrivals[1:])) * 1e3, 1)
                          if len(arrivals) > 1 else None,
            "median_gap_ms": round(sorted(b - a for a, b in zip(arrivals, arrivals[1:]))[
                (len(arrivals) - 1) // 2] * 1e3, 1) if len(arrivals) > 1 else None,
            "decode_tok_s_after_gap1": round((ct - 2) / (m1 - arrivals[1]), 1)
                                       if len(arrivals) > 2 and m1 > arrivals[1] else None}


# PROBE_STATS=1 adds each request's delta of the /v1/stats expert-cache counters, so a
# pass's decode can be tied to the traffic it caused: "mirror_delta" (saver: swaps,
# writebacks, free evictions, ...) and "decode_delta" (any offload residency, only with
# --moe-collect-stats: active and missing experts, layer calls). Off by default; the
# counters are read at idle.
STATS = os.environ.get("PROBE_STATS", "").strip() not in ("", "0")


def moe_counters() -> dict:
    try:
        with urllib.request.urlopen(URL.replace("/chat/completions", "/stats"), timeout=10) as r:
            moe = (json.load(r).get("scheduler") or {}).get("moe") or {}
    except Exception:
        return {}
    return {block: {k: v for k, v in (moe.get(block) or {}).items() if isinstance(v, int)}
            for block in ("mirror", "decode")}


for p in range(1, PASSES + 1):
    for s in SIZES:
        before = moe_counters() if STATS else {}
        # The tag is the prefix, so pass 2 misses the prefix cache pass 1 left.
        # PROBE_TAG prefixes every prompt (e.g. a profiler run that must miss the cache
        # of an untraced run before it); empty by default, so records are unchanged.
        tag = os.environ.get("PROBE_TAG", "") + (f"p{p} " if p > 1 else "")
        rec = {"target": s, "pass": p, **run(s, tag)}
        if STATS:
            for block, after in moe_counters().items():
                prior = before.get(block, {})
                rec[f"{block}_delta"] = {k: v - prior.get(k, 0) for k, v in after.items()}
        print(json.dumps(rec), flush=True)
