#!/usr/bin/env python3
"""Greedy two-turn tool-use conversation through /v1/messages in Claude Code's shape: the
environment block arrives as a system message AFTER the first user message, and a system
reminder follows the tool result. Checks the model reads both and calls the tool correctly."""
import json, os, urllib.request
URL = os.environ.get("FREETOKEN_URL", "http://127.0.0.1:1920").rstrip("/")
assert not URL.endswith(":1919")
MODEL = os.environ.get("FREETOKEN_MODEL_NAME", "nemotron-3.5-lightning")
TOOLS = [
    {"name": "Bash", "description": "Run a shell command and return its output.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "Read", "description": "Read a file by absolute path.",
     "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]}},
]
SYSTEM = [{"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.282.abc; cc_entrypoint=cli;"},
          {"type": "text", "text": "You are a coding agent. Use the tools to inspect the user's machine; never guess."}]
ENV = "# Environment\nYou have been invoked in the following environment:\n - Primary working directory: /srv/demo-repo-7731\n - Platform: linux"
REMINDER = "<system-reminder>When you give your final answer, end it with the exact word FINISHED.</system-reminder>"


def call(messages):
    body = {"model": MODEL, "max_tokens": 1500, "temperature": 0, "system": SYSTEM, "tools": TOOLS,
            "messages": messages}
    req = urllib.request.Request(URL + "/v1/messages", data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer local",
                                          "x-claude-code-session-id": "tooluse-probe-" + os.environ.get("ARM", "x")})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


msgs = [{"role": "user", "content": "Which directory am I working in, and what files does it contain? Use the Bash tool with the directory's absolute path."},
        {"role": "system", "content": ENV}]
r1 = call(msgs)
tu = [b for b in r1["content"] if b["type"] == "tool_use"]
print("turn1 stop", r1.get("stop_reason"), "usage", r1.get("usage"))
print("turn1 tool_use", json.dumps([(b["name"], b["input"]) for b in tu]))
print("turn1 text", json.dumps([b.get("text", "")[:300] for b in r1["content"] if b["type"] == "text"]))
ok1 = bool(tu) and tu[0]["name"] == "Bash" and "/srv/demo-repo-7731" in json.dumps(tu[0]["input"])
res = {"turn1_tool_ok": ok1}
if tu:
    msgs.append({"role": "assistant", "content": r1["content"]})
    msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": tu[0]["id"],
                                              "content": "README.md\nsetup.py\nsrc/"}]})
    msgs.append({"role": "system", "content": REMINDER})
    r2 = call(msgs)
    text = "".join(b.get("text", "") for b in r2["content"] if b["type"] == "text")
    print("turn2 stop", r2.get("stop_reason"), "usage", r2.get("usage"))
    print("turn2 text", json.dumps(text[-600:]))
    res.update(turn2_mentions_files="README.md" in text and "setup.py" in text,
               turn2_follows_reminder=text.rstrip().rstrip(".*").endswith("FINISHED"),
               turn2_cached=(r2.get("usage") or {}).get("cache_read_input_tokens"),
               turn2_prompt=((r2.get("usage") or {}).get("input_tokens") or 0) + ((r2.get("usage") or {}).get("cache_read_input_tokens") or 0))
print("TOOLUSE", json.dumps(res))
