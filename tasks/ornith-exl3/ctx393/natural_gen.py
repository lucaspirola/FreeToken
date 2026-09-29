#!/usr/bin/env python3
"""Natural-text decode workload for the Belady headroom check (tasks/harvest/README.md #4).

The 8K/80K probe decodes after a repeated-sentence prompt, whose routing is not what a user's
session looks like. This drives the server with a few ordinary tasks instead: code, an essay,
step-by-step math, a translation, and a summary + critique of a real document (the repo's own
docs, ~15K tokens). Temperature 0, thinking as the server serves it (on). Each request is
non-streaming; /v1/stats is read before the first request and after every request, so the
route trace (FREETOKEN_ROUTE_TRACE) drains at each boundary and the .jsonl sidecar marks
exactly which rows belong to this workload.

    natural_gen.py --doc docs/nemotron.md --doc docs/cli.md --max-tokens 3500 --out x.json
Env: FREETOKEN_URL (default http://127.0.0.1:1920), FREETOKEN_MODEL_NAME.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.request

URL = os.environ.get("FREETOKEN_URL", "http://127.0.0.1:1920")
NAME = os.environ.get("FREETOKEN_MODEL_NAME", "nemotron-3.5-lightning")

TASKS = [
    ("code", "Write a complete, well-documented Python module that implements a thread-safe LRU cache "
             "with per-entry time-to-live, size-based eviction, hit/miss statistics and a decorator for "
             "memoizing functions. Include type hints and a set of unittest test cases."),
    ("essay", "Write a detailed essay on the economic and social causes of the French Revolution, the "
              "role of the Estates-General, and how the events of 1789 changed European politics in the "
              "following decades. Use clear sections and concrete historical examples."),
    ("math", "A tank is filled by two pipes: pipe A alone fills it in 6 hours and pipe B alone in 9 hours. "
             "A drain empties the full tank in 12 hours. Starting empty, A and the drain are opened at 8:00, "
             "B is opened at 9:30, and A is closed at 11:00. When is the tank full? Solve step by step, then "
             "generalise the method and prove that it always gives the right answer."),
    ("translate", "Translate the following into Brazilian Portuguese, then explain each idiom you had to adapt "
                  "and why: 'It was raining cats and dogs, so we decided to call it a day. Our manager said we "
                  "were not out of the woods yet, but that the project was finally on the right track, and "
                  "that we should not count our chickens before they hatch.'"),
]


def post(path, body, timeout=1800):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def drain():
    with urllib.request.urlopen(URL + "/v1/stats", timeout=120) as r:
        doc = json.loads(r.read())
    return (((doc.get("scheduler") or {}).get("moe") or {}).get("route_trace") or {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", action="append", default=[])
    ap.add_argument("--doc-chars", type=int, default=60000)
    ap.add_argument("--max-tokens", type=int, default=3500)
    ap.add_argument("--out", default="")
    ap.add_argument("--texts", default="", help="path prefix: write each full output (reasoning + content)")
    a = ap.parse_args()
    tasks = list(TASKS)
    if a.doc:
        text = "\n\n".join(open(p).read() for p in a.doc)[: a.doc_chars]
        tasks.append(("summary", "Here is an engineering design document.\n\n" + text +
                      "\n\nSummarise its main design decisions, then critique them: which decisions look "
                      "fragile, what measurements are missing, and what would you change first?"))
    rec = {"start_drain": drain(), "tasks": []}
    for name, prompt in tasks:
        t0 = time.time()
        r = post("/v1/chat/completions", {"model": NAME, "messages": [{"role": "user", "content": prompt}],
                                          "max_tokens": a.max_tokens, "temperature": 0})
        dt = time.time() - t0
        u = r.get("usage", {})
        ch = r["choices"][0]
        msg = ch.get("message", {})
        text = (msg.get("content") or "")
        reasoning = (msg.get("reasoning_content") or "")
        full = reasoning + "\n<<content>>\n" + text
        row = {"task": name, "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
               "finish": ch.get("finish_reason"), "s": round(dt, 2), "head": text[:160],
               "md5": hashlib.md5(full.encode()).hexdigest(), "drain": drain()}
        if a.texts:
            with open(f"{a.texts}-{name}.txt", "w") as f:
                f.write(full)
        rec["tasks"].append(row)
        print(json.dumps({k: row[k] for k in ("task", "prompt_tokens", "completion_tokens", "finish", "s", "md5")}), flush=True)
    if a.out:
        json.dump(rec, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
