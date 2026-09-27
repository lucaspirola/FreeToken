#!/usr/bin/env python3
"""Redact raw capture.sh captures into the <client>.jsonl files replay.py loads.

  redact.py RAWDIR [OUTDIR (default: this directory)]

* Authorization / x-api-key / cookie headers become "<dummy>".
* Claude Code's billing block keeps its "x-anthropic-billing-header:" prefix (the server strips
  blocks by that prefix) but loses its value.
* Every UUID and every hex id of 16+ digits (device, installation, session, thread, turn, window,
  item and request ids) is replaced by a pseudonym derived from a per-run random salt. The mapping
  is consistent inside a run, so which requests share a session / prompt_cache_key is kept.
Fails if anything that looks like a real key survives.
"""
import hashlib, json, os, re, secrets, sys, uuid

RAW = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(__file__))
SALT = secrets.token_bytes(16)
UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
HEXID = re.compile(r"(?<![0-9a-zA-Z])(?=[0-9]*[a-f])(?=[a-f]*[0-9])[0-9a-f]{16,}(?![0-9a-zA-Z])")
SECRET_HDR = {"authorization", "x-api-key", "cookie", "proxy-authorization", "openai-organization",
              "openai-project", "chatgpt-account-id"}
BILLING = re.compile(r"(x-anthropic-billing-header:)[^\"\\]*")
LEAK = re.compile(r"(?<![A-Za-z0-9_-])sk-(?!ant-dummy|dummy)[A-Za-z0-9_-]{12,}|ghp_[A-Za-z0-9]{20,}|eyJ[A-Za-z0-9_-]{20,}\.")


def pseudo(s):
    d = hashlib.sha256(SALT + s.lower().encode()).digest()
    if UUID.fullmatch(s):
        return str(uuid.UUID(bytes=d[:16]))
    return d.hex()[: len(s)]


def scrub(text):
    text = UUID.sub(lambda m: pseudo(m.group(0)), text)
    text = HEXID.sub(lambda m: pseudo(m.group(0)), text)
    return BILLING.sub(r"\1 <redacted>", text)


for name in ("claude-code", "omp", "codex"):
    src = os.path.join(RAW, f"{name}.jsonl")
    if not os.path.exists(src):
        continue
    out = []
    for line in open(src):
        r = json.loads(line)
        r["headers"] = {k: ("<dummy>" if k.lower() in SECRET_HDR else v) for k, v in r["headers"].items()}
        out.append(scrub(json.dumps(r)))
    text = "\n".join(out) + "\n"
    leak = LEAK.search(text)
    assert leak is None, f"{name}: possible secret survives: {leak.group(0)[:12]}..."
    for l in text.splitlines():
        json.loads(l)
    open(os.path.join(OUT, f"{name}.jsonl"), "w").write(text)
    print(name, len(out), "requests ->", os.path.join(OUT, f"{name}.jsonl"))
