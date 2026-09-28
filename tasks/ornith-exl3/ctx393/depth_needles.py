#!/usr/bin/env python3
"""Needle recall at fixed depths inside the YaRN-stretched range (262144 < prompt <= 393216).

recall.py sizes its haystack with a words->tokens guess (240000 asked -> 197993 served), which is
useless near a hard ceiling. This builds the filler with the model's own tokenizer, so every prompt
is the requested size to within a few tokens and the needle sits at an exact token depth.

Two kinds of request, all thinking OFF, temperature 0, one fresh prefill each (every prompt starts
with its own header line, so no two share a cached prefix):
  single: one passphrase planted at DEPTH of a SIZE-token haystack, "what is the passphrase?"
  multi:  three codes at 2% / 50% / 97% of one haystack, "list them in order" (recall.py's question)

    depth_needles.py --sizes 300000 380000 390000 --depths 0.10 0.50 0.85 0.98 --multi 300000 380000
Env: FREETOKEN_URL (default http://127.0.0.1:1920), FREETOKEN_MODEL_NAME (default ornith).
One JSON line per request on stdout; exit 1 if any request missed its needle(s).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
import urllib.request

from tokenizers import Tokenizer

URL = os.environ.get("FREETOKEN_URL", "http://127.0.0.1:1920")
NAME = os.environ.get("FREETOKEN_MODEL_NAME", "ornith")
TOK = Tokenizer.from_file(os.path.expanduser(
    os.environ.get("NEEDLE_TOKENIZER", "~/ai/models/Ornith-1.5-35B-A3B-exl3-5.0bpw-hq/tokenizer.json")))
# recall.py's filler words plus more of the same register: dense, answer-free
WORDS = ("ledger entry audit column revision batch quarter invoice reconcile variance schedule appendix "
         "clause remittance account balance credit debit journal posting accrual deferral statement "
         "memo voucher payable receivable fiscal period closing adjustment transfer settlement").split()
# chat template + question overhead is ~40 tokens; keep the body this much under the target
OVERHEAD = 64


def filler(n_tokens: int, rng: random.Random) -> str:
    words = [rng.choice(WORDS) for _ in range(int(n_tokens * 1.1) + 16)]
    text = " ".join(words)
    enc = TOK.encode(text, add_special_tokens=False)
    assert len(enc.ids) >= n_tokens, (len(enc.ids), n_tokens)
    return text[:enc.offsets[n_tokens - 1][1]]


def code(rng: random.Random) -> str:
    return f"{rng.choice(['AMBER', 'COBALT', 'SAFFRON', 'INDIGO', 'VIOLET'])}-" \
           f"{rng.choice(['FALCON', 'OTTER', 'HERON', 'LYNX', 'MARTEN'])}-{rng.randrange(1000, 9999)}"


def haystack(size: int, plants: list[tuple[float, str]], header: str, rng: random.Random):
    """Body of ~size-OVERHEAD tokens with each sentence planted at its token depth."""
    body_tokens = size - OVERHEAD - len(TOK.encode(header, add_special_tokens=False).ids)
    for _, s in plants:
        body_tokens -= len(TOK.encode(" " + s, add_special_tokens=False).ids)
    parts, prev = [header], 0
    for depth, sentence in sorted(plants):
        at = int(body_tokens * depth)
        parts += [filler(at - prev, rng), sentence] if at > prev else [sentence]
        prev = at
    parts.append(filler(body_tokens - prev, rng))
    return " ".join(parts)


def ask(prompt: str, max_tokens: int) -> dict:
    req = urllib.request.Request(
        f"{URL}/v1/chat/completions",
        data=json.dumps({"model": NAME, "messages": [{"role": "user", "content": prompt}],
                         "max_tokens": max_tokens, "temperature": 0.0,
                         "chat_template_kwargs": {"enable_thinking": False}}).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            body = json.load(r)
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read()[:300].decode(errors='replace')}",
                "seconds": round(time.monotonic() - t0, 1)}
    u = body.get("usage") or {}
    return {"answer": (body["choices"][0]["message"].get("content") or "").strip(),
            "finish_reason": body["choices"][0].get("finish_reason"),
            "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
            "seconds": round(time.monotonic() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="*", default=[300000, 380000])
    ap.add_argument("--depths", type=float, nargs="*", default=[0.50, 0.85, 0.98])
    ap.add_argument("--multi", type=int, nargs="*", default=[])
    ap.add_argument("--max-tokens", type=int, default=48)
    a = ap.parse_args()
    bad = 0
    for size in a.sizes:
        for depth in a.depths:
            rng = random.Random(f"ctx393-single-{size}-{depth}")
            pw = code(rng)
            prompt = haystack(size, [(depth, f"The vault passphrase is {pw}.")],
                              f"Archive {size}/{depth}: quarterly ledger extract.", rng)
            prompt += "\n\nWhat is the vault passphrase stated in the text above? Answer with the passphrase only."
            res = {"kind": "single", "size": size, "depth": depth, "planted": pw, **ask(prompt, a.max_tokens)}
            res["pass"] = pw in res.get("answer", "")
            bad += not res["pass"]
            print(json.dumps(res), flush=True)
    for size in a.multi:
        rng = random.Random(f"ctx393-multi-{size}")
        codes = [code(rng) for _ in range(3)]
        prompt = haystack(size, [(d, f"The authorization code for this section is {c}.")
                                 for d, c in zip((0.02, 0.50, 0.97), codes)],
                          f"Archive {size}/multi: quarterly ledger extract.", rng)
        prompt += ("\n\nList the three authorization codes that appear in the text above, "
                   "in the order they appear, and nothing else.")
        res = {"kind": "multi", "size": size, "depth": [0.02, 0.50, 0.97], "planted": codes,
               **ask(prompt, a.max_tokens * 2)}
        ans = res.get("answer", "")
        pos = [ans.find(c) for c in codes]
        res["recalled"] = [c for c in codes if c in ans]
        res["pass"] = all(p >= 0 for p in pos) and pos == sorted(pos)
        bad += not res["pass"]
        print(json.dumps(res), flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
