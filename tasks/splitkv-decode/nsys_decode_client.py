#!/usr/bin/env python3
"""Stream one probe_decode-style request and nsys-capture ONLY its decode.

    nsys_decode_client.py SIZE OUT_PREFIX
    env: NSYS (binary), NSYS_SESSION, FREETOKEN_URL, PROBE_GEN_TOKENS (default 160),
         SKIP_TOKENS (default 16: tokens decoded before the capture starts)

The server runs under ``nsys launch --session-new=$NSYS_SESSION``; this client calls
``nsys start`` once SKIP_TOKENS tokens have streamed (the prefill and the first decode
steps are outside the capture) and ``nsys stop`` when the request ends. Prints one JSON
line with the request's decode tok/s (perturbed by the tracer: use probe_decode.py
numbers for rates, this trace for the per-kernel split).
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

URL = os.environ.get("FREETOKEN_URL", "http://127.0.0.1:1920") + "/v1/chat/completions"
MODEL = os.environ.get("FREETOKEN_MODEL_NAME", "nemotron-3.5-lightning")
SENT = "The quick brown fox jumps over the lazy dog near the riverbank while the miller counts his sacks of grain. "
NSYS = os.environ["NSYS"]
SESSION = os.environ["NSYS_SESSION"]
GEN = int(os.environ.get("PROBE_GEN_TOKENS", "160"))
SKIP = int(os.environ.get("SKIP_TOKENS", "16"))

size, out = int(sys.argv[1]), sys.argv[2]
tag = os.environ.get("PROBE_TAG", "nsys ")
prompt = f"Run {tag}{size}. " + SENT * max(1, size // 23) + "\n\nWrite a long story about the fox."
body = json.dumps({
    "model": MODEL, "messages": [{"role": "user", "content": prompt}],
    "max_tokens": GEN, "temperature": 0, "stream": True,
    "stream_options": {"include_usage": True},
    "chat_template_kwargs": {"enable_thinking": False},
}).encode()
req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
n = 0; started = None; t_first = None; usage = None; n_at_start = 0
t0 = time.monotonic()
with urllib.request.urlopen(req, timeout=7200) as r:
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
                if t_first is None:
                    t_first = time.monotonic()
                if n == SKIP and started is None:
                    subprocess.run([NSYS, "start", f"--session={SESSION}", f"--output={out}",
                                    "--force-overwrite=true"], check=True)
                    started = time.monotonic(); n_at_start = n
t1 = time.monotonic()
if started is not None:
    subprocess.run([NSYS, "stop", f"--session={SESSION}"], check=True)
print(json.dumps({
    "target": size, "prompt_tokens": (usage or {}).get("prompt_tokens"),
    "ttft_s": round((t_first or t1) - t0, 2), "gen_tokens": n,
    "captured_tokens": n - n_at_start if started else 0,
    "traced_decode_tok_s": round((n - n_at_start) / (t1 - started), 1) if started else None,
    "out": out,
}), flush=True)
