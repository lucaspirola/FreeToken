#!/usr/bin/env python3
"""Part B of the /clear probe: replay the captured client requests against a real
FreeToken server (FREETOKEN_URL, never :1919) and report, per client, the cached-token
count and TTFT of the first request after /clear once session A has grown to ~30K tokens,
against a cold start of the same request.

Per client, in this order:
  cold      the post-/clear request with a nonce at the head of its system prompt, sent
            before anything of this client reached the server (nothing to reuse)
  A         the pre-/clear main request grown by ~30K tokens of filler (the old session)
  clear     the post-/clear request verbatim (new session id / new prompt_cache_key)
  repeat    the same post-/clear request again (the reuse ceiling)
  next      the client's next turn of the post-/clear conversation (turn N+1 must resume
            turn N's end state)
Only the main-model calls are replayed; the clients' side calls (Claude Code's
0-tool call, omp's judge/title calls, Codex's title call) are left out. The captures
(capture.sh, redact.py) are picked by structure, not by row number: the main calls are the
ones carrying the client's full tool list, and /clear is where their conversation length
drops.
"""
import json, os, sys, time, urllib.request, uuid, copy, glob

HERE = os.path.dirname(os.path.abspath(__file__))
URL = os.environ.get("FREETOKEN_URL", "http://127.0.0.1:1920").rstrip("/")
assert not URL.endswith(":1919"), "never the owner's production unit"
MODEL = os.environ.get("FREETOKEN_MODEL_NAME", "nemotron-3.5-lightning")
GEN = 32
FILLER_CHARS = int(os.environ.get("FILLER_CHARS", "110000"))


def rows(name):
    return [json.loads(l) for l in open(os.path.join(HERE, f"{name}.jsonl"))]


def main_calls(R, items):
    """(last call before /clear, first call after it, second call after it)."""
    ntools = max(len(r["body"].get("tools") or []) for r in R if isinstance(r["body"], dict))
    main = [r for r in R if isinstance(r["body"], dict) and len(r["body"].get("tools") or []) == ntools]
    cut = next(i for i in range(1, len(main)) if len(main[i]["body"][items]) < len(main[i - 1]["body"][items]))
    return main[cut - 1], main[cut], main[cut + 1]


def filler():
    src = sorted(glob.glob(os.path.join(HERE, "..", "..", "python", "freetoken", "server", "*.py")))
    text = "Reference material the user pasted earlier in this session:\n\n"
    for p in src:
        text += f"=== {os.path.basename(p)} ===\n" + open(p).read() + "\n"
        if len(text) >= FILLER_CHARS:
            break
    return text[:FILLER_CHARS]


FILL = filler()
KEEP_HDR = {"x-claude-code-session-id", "x-claude-code-agent-id", "anthropic-version", "anthropic-beta",
            "x-app", "user-agent", "session-id", "thread-id", "x-codex-window-id", "x-codex-turn-metadata",
            "originator", "x-client-request-id", "x-codex-beta-features", "x-session-id"}


def send(path, body, headers):
    h = {k: v for k, v in headers.items() if k.lower() in KEEP_HDR}
    h["Content-Type"] = "application/json"
    h["Authorization"] = "Bearer local"
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers=h, method="POST")
    t0 = time.monotonic(); ttft = None; usage = None; nbytes = 0
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                continue
            try:
                ev = json.loads(data)
            except Exception:
                continue
            typ = ev.get("type", "")
            gen = False
            if typ == "content_block_delta" or (typ.startswith("response.") and typ.endswith(".delta")):
                gen = True
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                if any(d.get(k) for k in ("content", "reasoning_content", "reasoning", "tool_calls")):
                    gen = True
            if gen and ttft is None:
                ttft = time.monotonic() - t0
            if typ == "message_start":
                usage = {**(usage or {}), **ev["message"].get("usage", {})}
            if typ == "message_delta" and ev.get("usage"):
                usage = {**(usage or {}), **ev["usage"]}
            if typ == "response.completed":
                usage = ev["response"].get("usage")
            if ev.get("usage") and "choices" in ev:
                usage = ev["usage"]
    return {"ttft_s": round(ttft, 3) if ttft is not None else None,
            "total_s": round(time.monotonic() - t0, 3), "usage": usage}


def cached(u):
    if not u:
        return None
    if "cache_read_input_tokens" in u:
        return {"prompt": (u.get("input_tokens") or 0) + (u.get("cache_read_input_tokens") or 0),
                "cached": u.get("cache_read_input_tokens")}
    if "input_tokens_details" in u:
        return {"prompt": u.get("input_tokens"), "cached": (u["input_tokens_details"] or {}).get("cached_tokens")}
    return {"prompt": u.get("prompt_tokens"), "cached": ((u.get("prompt_tokens_details") or {}).get("cached_tokens"))}


def claude_code():
    a, c, n = main_calls(rows("claude-code"), "messages")

    def prep(r):
        b = copy.deepcopy(r["body"]); b["model"] = MODEL; b["max_tokens"] = GEN; b["stream"] = True
        return b
    cold = prep(c); cold["system"][0]["text"] = f"nonce {uuid.uuid4()}\n" + cold["system"][0]["text"]
    A = prep(a); last = A["messages"][-1]
    last["content"] = [{"type": "text", "text": FILL}] + (last["content"] if isinstance(last["content"], list)
                                                           else [{"type": "text", "text": last["content"]}])
    return "/v1/messages", cold, A, prep(c), a["headers"], c["headers"], prep(n), n["headers"]


def omp():
    a, c, n = main_calls(rows("omp"), "messages")

    def prep(r):
        b = copy.deepcopy(r["body"]); b["model"] = MODEL; b["max_completion_tokens"] = GEN; b["stream"] = True
        return b
    cold = prep(c); cold["messages"][0]["content"] = f"nonce {uuid.uuid4()}\n" + cold["messages"][0]["content"]
    A = prep(a); last = A["messages"][-1]
    if isinstance(last["content"], list):
        last["content"] = [{"type": "text", "text": FILL}] + last["content"]
    else:
        last["content"] = FILL + "\n\n" + last["content"]
    return "/v1/chat/completions", cold, A, prep(c), a["headers"], c["headers"], prep(n), n["headers"]


def codex():
    a, c, n = main_calls(rows("codex"), "input")

    def prep(r):
        b = copy.deepcopy(r["body"]); b["model"] = MODEL; b["max_output_tokens"] = GEN; b["stream"] = True
        return b
    cold = prep(c); cold["instructions"] = f"nonce {uuid.uuid4()}\n" + cold["instructions"]
    A = prep(a)
    A["input"].insert(len(A["input"]) - 1, {"type": "message", "role": "user",
                                            "content": [{"type": "input_text", "text": FILL}]})
    return "/v1/responses", cold, A, prep(c), a["headers"], c["headers"], prep(n), n["headers"]


out = {}
for name, build in (("claude-code", claude_code), ("omp", omp), ("codex", codex)):
    path, cold, A, clear, ha, hc, nxt, hn = build()
    res = {}
    for step, body, hdr in (("cold", cold, hc), ("A", A, ha), ("clear", clear, hc), ("repeat", clear, hc),
                            ("next", nxt, hn)):
        try:
            r = send(path, body, hdr)
        except Exception as exc:
            r = {"error": repr(exc)}
        r["tokens"] = cached(r.get("usage"))
        res[step] = r
        print(name, step, json.dumps(r), flush=True)
    out[name] = res
json.dump(out, open(os.environ.get("REPLAY_OUT", os.path.join(HERE, "replay-result.json")), "w"), indent=1)
print("replay done")
