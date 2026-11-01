"""Hit a running lfm25 test server: models, chat (non-stream and stream), tools, tool history.

    python api_check.py <base_url>
"""
import json
import sys
import urllib.request

BASE = sys.argv[1]
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                         "required": ["city"]}}}]


def post(path, body, stream=False):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        if not stream:
            return r.status, json.loads(r.read())
        return r.status, [json.loads(l[6:]) for l in r.read().decode().splitlines() if l.startswith("data: {")]


with urllib.request.urlopen(BASE + "/v1/models", timeout=30) as r:
    print("models:", [m["id"] for m in json.loads(r.read())["data"]])
msgs = [{"role": "user", "content": "Say hi."}]
st, r = post("/v1/chat/completions", {"model": "lfm25", "messages": msgs, "max_tokens": 24, "temperature": 0})
m = r["choices"][0]["message"]
print("non-stream:", st, r["choices"][0]["finish_reason"], "usage", r["usage"],
      "| reasoning chars", len(m.get("reasoning_content") or ""), "| content chars", len(m.get("content") or ""))
st, chunks = post("/v1/chat/completions", {"model": "lfm25", "messages": msgs, "max_tokens": 24, "temperature": 0, "stream": True}, stream=True)
rs = "".join((c["choices"][0]["delta"].get("reasoning_content") or "") for c in chunks if c.get("choices"))
cs = "".join((c["choices"][0]["delta"].get("content") or "") for c in chunks if c.get("choices"))
print("stream:", st, len(chunks), "chunks | reasoning == non-stream (stripped):", rs.strip() == (m.get("reasoning_content") or "").strip(),
      "| content == non-stream:", cs == (m.get("content") or ""))
st, r = post("/v1/chat/completions", {"model": "lfm25", "messages": [{"role": "user", "content": "Weather in Paris?"}],
                                       "tools": TOOLS, "max_tokens": 16, "temperature": 0})
print("with tools:", st, r["choices"][0]["finish_reason"], "prompt tokens", r["usage"]["prompt_tokens"])
hist = [{"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": "", "reasoning_content": "Call the tool.",
         "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\", \"days\": 2}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "{\"temp_c\": 18}"}]
st, r = post("/v1/chat/completions", {"model": "lfm25", "messages": hist, "tools": TOOLS, "max_tokens": 16, "temperature": 0})
print("tool history:", st, r["choices"][0]["finish_reason"], "prompt tokens", r["usage"]["prompt_tokens"])
