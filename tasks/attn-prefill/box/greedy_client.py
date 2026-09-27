#!/usr/bin/env python3
"""Greedy decode transcripts for the before/after correctness check of decode attention.

    greedy_client.py OUT.jsonl SIZE [SIZE ...]
    env: FREETOKEN_URL, FREETOKEN_MODEL_NAME, GREEDY_TOKENS (default 256), GREEDY_REPEATS (2)

Per size: a long haystack prompt (distinct numbered sentences, so the answer depends on
positions deep in the KV), temperature 0, thinking off, GREEDY_TOKENS decoded tokens.
Each prompt is sent GREEDY_REPEATS times (no session key -> no prefix reuse) so the
arm's own run-to-run determinism is on record next to the cross-arm comparison.
"""
import hashlib
import json
import os
import sys
import urllib.request

URL = os.environ.get("FREETOKEN_URL", "http://127.0.0.1:1920") + "/v1/chat/completions"
MODEL = os.environ.get("FREETOKEN_MODEL_NAME", "nemotron-3.5-lightning")
GEN = int(os.environ.get("GREEDY_TOKENS", "256"))
REP = int(os.environ.get("GREEDY_REPEATS", "2"))
NAMES = ["Ada", "Bruno", "Chiara", "Dmitri", "Elif", "Farid", "Greta", "Hiro", "Ines", "Jonas"]
ITEMS = ["lantern", "compass", "violin", "atlas", "kettle", "telescope", "anchor", "quill"]
CITIES = ["Porto", "Tromso", "Cusco", "Hobart", "Tallinn", "Oaxaca", "Nagoya", "Bergen"]


def prompt(size):
    lines, i = [], 0
    while len(lines) * 16 < size:  # ~16 tokens per line
        lines.append(f"Record {i}: {NAMES[i % 10]} left a {ITEMS[(i * 3) % 8]} in "
                     f"{CITIES[(i * 5) % 8]} on day {(i * 37) % 365}.")
        i += 1
    n = len(lines)
    q = (f"Above are {n} records. Quote records {n // 7}, {n // 2} and {n - 3} exactly, then "
         "write a long story that visits every city mentioned in them, in order.")
    return "\n".join(lines) + "\n\n" + q


def main():
    if sys.argv[1] == "--dump":  # greedy_client.py --dump SIZE FILE: write the prompt text only
        open(sys.argv[3], "w").write(prompt(int(sys.argv[2])))
        sys.exit(0)
    out = open(sys.argv[1], "a")
    for size in map(int, sys.argv[2:]):
        p = prompt(size)
        for rep in range(REP):
            body = json.dumps({
                "model": MODEL, "messages": [{"role": "user", "content": p}],
                "max_tokens": GEN, "temperature": 0, "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
            }).encode()
            req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=7200) as r:
                d = json.load(r)
            msg = d["choices"][0]["message"]
            text = (msg.get("reasoning_content") or "") + "\x00" + (msg.get("content") or "")
            row = {"target": size, "rep": rep, "prompt_tokens": d["usage"]["prompt_tokens"],
                   "completion_tokens": d["usage"]["completion_tokens"],
                   "sha1": hashlib.sha1(text.encode()).hexdigest(), "text": text}
            out.write(json.dumps(row) + "\n"); out.flush()
            print(json.dumps({k: v for k, v in row.items() if k != "text"}), flush=True)


if __name__ == "__main__":
    main()
