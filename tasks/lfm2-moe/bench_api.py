import json, os, sys, time, urllib.request
URL = f"http://127.0.0.1:{os.environ.get('FREETOKEN_PORT', '1921')}/v1/chat/completions"

def call(body, sid):
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "x-session-id": sid})
    return urllib.request.urlopen(req, timeout=1800)

def stream(messages, sid, max_tokens, tools=None):
    body = {"model": "lfm25", "messages": messages, "max_tokens": max_tokens, "stream": True,
            "temperature": 0, "stream_options": {"include_usage": True}}
    if tools: body["tools"] = tools
    t0 = time.monotonic(); first = None; n = 0; usage = None; reasoning = []; content = []; tcalls = []
    for line in call(body, sid):
        line = line.decode().strip()
        if not line.startswith("data:") or line == "data: [DONE]": continue
        d = json.loads(line[5:])
        if d.get("usage"): usage = d["usage"]
        for ch in d.get("choices", []):
            delta = ch.get("delta", {})
            piece = (delta.get("reasoning_content") or "") + (delta.get("content") or "")
            if piece or delta.get("tool_calls"):
                if first is None: first = time.monotonic()
                n += 1
            reasoning.append(delta.get("reasoning_content") or ""); content.append(delta.get("content") or "")
            tcalls += delta.get("tool_calls") or []
    t1 = time.monotonic()
    return dict(ttft=first - t0, total=t1 - t0, chunks=n, usage=usage,
                reasoning="".join(reasoning), content="".join(content), tool_calls=tcalls)

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "smoke":
        r = stream([{"role": "user", "content": "What is 17*23? Answer briefly."}], "smoke-1", 512)
        print("REASONING:", r["reasoning"][:300].replace("\n", " ")); print("CONTENT:", r["content"][:300]); print(r["usage"])
        tools = [{"type": "function", "function": {"name": "get_weather", "description": "Weather for a city",
                  "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
        r = stream([{"role": "user", "content": "What's the weather in Lisbon?"}], "smoke-2", 1024, tools)
        print("TOOL_CALLS:", json.dumps(r["tool_calls"])[:400]); print("CONTENT:", r["content"][:200])
    elif mode == "bench":
        base = open("/home/lucas/ai/FreeToken/CLAUDE.md").read() + open("/home/lucas/ai/FreeToken/README.md").read()
        # BENCH_PASSES runs of all sizes; BENCH_SLEEP idle seconds before each request
        passes, idle = int(os.environ.get("BENCH_PASSES", "1")), float(os.environ.get("BENCH_SLEEP", "0"))
        for p, target in ((p, t) for p in range(passes) for t in map(int, sys.argv[2:])):
            time.sleep(idle)
            text = ""; i = 0
            while len(text) < target * 4: text += f"\n[{i}] " + base; i += 1
            text = text[: target * 4]
            nonce = f"Run {time.time_ns()} {target}.\n"
            msg = [{"role": "user", "content": nonce + text + "\n\nSummarize the text above in one sentence."}]
            r = stream(msg, f"bench-{target}-{time.time()}", 256)
            u = r["usage"]; pt = u["prompt_tokens"]; ct = u["completion_tokens"]
            dec = (ct - 1) / (r["total"] - r["ttft"]) if ct > 1 else 0
            print(json.dumps(dict(pass_=p, target=target, prompt_tokens=pt, ttft_s=round(r["ttft"], 2), prefill_tok_s=round(pt / r["ttft"]),
                                  completion_tokens=ct, decode_tok_s=round(dec, 1), cached=u.get("prompt_tokens_details", {}).get("cached_tokens"))), flush=True)
