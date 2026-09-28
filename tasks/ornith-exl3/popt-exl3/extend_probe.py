#!/usr/bin/env python3
"""File-read extends at depth: the owner's omp shape (a tool result of 5-80K tokens appended to a deep,
prefix-cached context).

    extend_probe.py [DEPTH:ADD ...]     default 100000:10000 100000:30000 200000:30000

For every pass (EXT_PASSES, default 2; each pass has its own context prefix, so pass 2 cannot hit
pass 1's cache) and every DEPTH: one context request (DEPTH tokens of prose, 1 generated token; a
fresh prefill, recorded as kind=ctx), then for each ADD at that DEPTH one extend request: the same
context turn + an assistant turn + a user turn holding ADD tokens of code-like "file" text. Only the
file part is new, so TTFT is the extend's prefill at depth. Records prompt/cached tokens (usage),
TTFT and extend tok/s on the monotonic clock, decode tok/s of EXT_GEN tokens, out_sha1.
Env: FREETOKEN_URL, FREETOKEN_MODEL_NAME, EXT_GEN (default 64), EXT_PASSES (default 2), EXT_TAG,
EXT_PRE / EXT_POST: shell commands run just before / after each extend request (profiling:
nsys start/stop), formatted with {depth} {add} {pass}.
"""
import hashlib, json, os, sys, time, urllib.request

URL = os.environ.get("FREETOKEN_URL", "http://127.0.0.1:1920") + "/v1/chat/completions"
MODEL = os.environ.get("FREETOKEN_MODEL_NAME", "ornith")
GEN = int(os.environ.get("EXT_GEN", "64"))
PASSES = int(os.environ.get("EXT_PASSES", "2"))
SENT = "The quick brown fox jumps over the lazy dog near the riverbank while the miller counts his sacks of grain. "
PAIRS = [tuple(int(v) for v in a.split(":")) for a in (sys.argv[1:] or ["100000:10000", "100000:30000", "200000:30000"])]


def ctx_text(depth, tag):
    return f"Session {tag}{depth}. " + SENT * max(1, depth // 23)


def file_text(add, tag):
    # ~14 tokens per line of code-like text, deterministic, distinct lines
    lines = [f"def f{tag}{add}_{i}(x, y):  return (x * {i % 97} + y) // {1 + i % 13}  # line {i}\n"
             for i in range(max(1, add // 26))]
    return "Here is the file you asked for:\n```python\n" + "".join(lines) + "```\nWhat does f_7 return for x=3, y=4?"


def post(messages, gen, session):
    body = json.dumps({"model": MODEL, "messages": messages, "max_tokens": gen, "temperature": 0,
                       "stream": True, "stream_options": {"include_usage": True},
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json",
                                                          "x-session-id": session})
    m0 = time.monotonic(); mfirst = None; n = 0; usage = None; text = []
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            if not line.startswith(b"data:"):
                continue
            p = line[5:].strip()
            if p == b"[DONE]":
                break
            d = json.loads(p)
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices", []):
                de = ch.get("delta", {})
                if de.get("content") or de.get("reasoning_content"):
                    text.append((de.get("reasoning_content") or "") + (de.get("content") or ""))
                    n += 1
                    if mfirst is None:
                        mfirst = time.monotonic()
    m1 = time.monotonic()
    u = usage or {}
    pt = u.get("prompt_tokens", 0)
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    ct = u.get("completion_tokens", n)
    ttft = (mfirst or m1) - m0
    new = pt - (cached or 0)
    return {"prompt_tokens": pt, "cached_tokens": cached, "new_tokens": new, "ttft_mono_s": round(ttft, 3),
            "extend_tok_s": round(new / ttft, 0) if ttft else None, "gen_tokens": ct,
            "decode_tok_s": round((ct - 1) / (m1 - mfirst), 1) if mfirst and ct > 1 and m1 > mfirst else None,
            "out_sha1": hashlib.sha1("".join(text).encode()).hexdigest()}


depths = []
for d, _ in PAIRS:
    if d not in depths:
        depths.append(d)
for p in range(1, PASSES + 1):
    tag = os.environ.get("EXT_TAG", "") + f"e{p} "
    for d in depths:
        session = f"extprobe-{tag.strip()}-{d}"
        ctx = [{"role": "user", "content": ctx_text(d, tag) + "\n\nReply with one word."}]
        rec = post(ctx, 1, session)
        print(json.dumps({"kind": "ctx", "pass": p, "depth": d, **rec}), flush=True)
        for dd, add in PAIRS:
            if dd != d:
                continue
            msgs = ctx + [{"role": "assistant", "content": "Ok."},
                          {"role": "user", "content": file_text(add, tag.strip())}]
            if os.environ.get("EXT_PRE"):
                os.system(os.environ["EXT_PRE"].format(depth=d, add=add, **{"pass": p}))
            rec = post(msgs, GEN, session)
            if os.environ.get("EXT_POST"):
                os.system(os.environ["EXT_POST"].format(depth=d, add=add, **{"pass": p}))
            print(json.dumps({"kind": "extend", "pass": p, "depth": d, "add": add, **rec}), flush=True)
