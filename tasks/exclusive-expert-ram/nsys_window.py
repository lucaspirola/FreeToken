#!/usr/bin/env python3
"""Stream one probe_decode-style request and trace only a decode window with nsys.

  nsys_window.py SIZE SESSION OUTPUT [SKIP] [WINDOW]

Sends the same prompt probe_decode.py sends for SIZE (pass 1, no tag), max_tokens
SKIP + WINDOW + 64. After the first token and SKIP more, it runs `nsys start`; after WINDOW
further tokens, `nsys stop`. Prefill is never traced, so a 1M request yields a small report.
Prints one JSON line: ttft, the traced token count and its decode rate.
"""
import json, os, subprocess, sys, time, urllib.request

SIZE, SESSION, OUT = int(sys.argv[1]), sys.argv[2], sys.argv[3]
SKIP = int(sys.argv[4]) if len(sys.argv) > 4 else 32
WINDOW = int(sys.argv[5]) if len(sys.argv) > 5 else 200
NSYS = os.environ.get("NSYS", "/usr/local/cuda/bin/nsys")
URL = os.environ.get("FREETOKEN_URL", "http://127.0.0.1:1920") + "/v1/chat/completions"
MODEL = os.environ.get("FREETOKEN_MODEL_NAME", "nemotron-3.5-lightning")
SENT = "The quick brown fox jumps over the lazy dog near the riverbank while the miller counts his sacks of grain. "
prompt = f"Run {SIZE}. " + SENT * max(1, int(SIZE / 23)) + "\n\nWrite a long story about the fox."
body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": SKIP + WINDOW + 64, "temperature": 0, "stream": True,
                   "chat_template_kwargs": {"enable_thinking": False}}).encode()
req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
m0 = time.monotonic(); n = 0; state = "wait"; t_start = t_stop = None; n_start = n_stop = 0; ttft = None
with urllib.request.urlopen(req, timeout=7200) as r:
    for line in r:
        if not line.startswith(b"data:"):
            continue
        p = line[5:].strip()
        if p == b"[DONE]":
            break
        d = json.loads(p)
        for ch in d.get("choices", []):
            if ch.get("delta", {}).get("content"):
                n += 1
                if ttft is None:
                    ttft = time.monotonic() - m0
                if state == "wait" and n >= 1 + SKIP:
                    subprocess.run([NSYS, "start", f"--session={SESSION}", f"--output={OUT}", "--force-overwrite=true"], check=False)
                    state, t_start, n_start = "on", time.monotonic(), n
                elif state == "on" and n >= n_start + WINDOW:
                    t_stop, n_stop = time.monotonic(), n
                    subprocess.run([NSYS, "stop", f"--session={SESSION}"], check=False)
                    state = "off"
if state == "on":
    t_stop, n_stop = time.monotonic(), n
    subprocess.run([NSYS, "stop", f"--session={SESSION}"], check=False)
print(json.dumps({"target": SIZE, "ttft_mono_s": round(ttft or 0, 2), "tokens": n,
                  "traced_tokens": n_stop - n_start,
                  "traced_tok_s": round((n_stop - n_start) / (t_stop - t_start), 1) if t_stop and t_start else None}), flush=True)
