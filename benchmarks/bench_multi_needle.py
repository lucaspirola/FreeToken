"""Multi-needle long-context recall against an already-running FreeToken server.

One expensive prefill, many cheap questions. ``bench_long_context.py`` builds a
*single* needle prompt and starts its own server per length, so probing N depths
costs N full prefills -- at 1M tokens that is N x ~30 minutes of GPU. This harness
puts every needle in one haystack and then asks its questions as follow-up *turns*
of the same chat conversation, so turns 2..N hit the prefix cache and cost only
their own question.

What it measures, per question: whether the right code came back, whether a *wrong*
needle's code came back instead, the reported ``cached_tokens`` (the prefix-cache
proof), TTFT and decode rate. Two of the questions are not needles:

* a **control** naming a key that is not in the haystack -- the correct answer is
  that it is absent, and any 7-digit answer is a fabrication;
* a **combined** question that cannot be answered from one needle alone (compare
  two codes and add them), which tests that separate needles are jointly available
  rather than one lucky retrieval.

Conventions inherited from ``bench_long_context.py``: the filler carries **no
digits at all** (a numeric needle in a numeric haystack is a distractor test, not a
retrieval test), grading reads the concatenated SSE ``content`` fields of
``/v1/chat/completions`` and never the raw stream, and thinking is disabled so the
decode budget is not spent inside a reasoning block.

``ignore_eos`` is deliberately NOT used: a forced-length reply ends in tokens the
next turn cannot resend, and the prefix match is exact, so it would throw away the
cache the whole design depends on.

The server is expected to be already serving (started under ``scripts/gpu_lock.sh``
with the 1M profile from ``docs/nemotron.md``); this script only drives it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request

from bench_long_context import load_tokenizer


#: (key, 7-digit code, depth in the haystack). Depths are fractions of the filler
#: body, so 0.05 really is 5 % of a million tokens in and not "just after the
#: instruction": the 262K bisect showed this model's recall is depth-dependent.
NEEDLES: list[tuple[str, str, float]] = [
    ("orchard", "5663623", 0.05),
    ("harbour", "4190877", 0.25),
    ("quarry", "8324516", 0.50),
    ("cavern", "6082735", 0.60),
    ("meadow", "7218459", 0.75),
    ("thicket", "3947162", 0.95),
]

#: A key that appears nowhere in the prompt. Must not collide with the filler's
#: vocabulary either, or the control turns into a distractor test.
CONTROL_KEY = "belfry"

HEAD = (
    "You are auditing a warehouse log. The records below are almost all inactive "
    "seal notices, which carry no information. Scattered among them are a few "
    "lines of the form \"The <name> ledger code is <digits>.\" -- remember every "
    "one of them, together with the name it belongs to, and ignore every inactive "
    "seal notice.\n\n"
)

#: Digit-free filler, the shape bench_long_context.py's synthetic needle uses. It
#: deliberately names no ledger: a filler that mentions the needles' keys (or the
#: control's) would make a miss unreadable.
_SHADES = ("dull", "faint", "pale", "muted", "dim", "worn", "cold", "quiet")
_STUFF = ("copper", "silver", "bronze", "pewter", "amber", "cobalt", "slate", "ivory")


def filler_lines(start: int, count: int) -> str:
    return "".join(
        f"The warden notes that the {_SHADES[i % len(_SHADES)]} "
        f"{_STUFF[(i // len(_SHADES)) % len(_STUFF)]} seal remains inactive.\n"
        for i in range(start, start + count)
    )


class Filler:
    """Deterministic filler sliced to an exact token count."""

    def __init__(self, tok, cursor: int = 0) -> None:
        self.tok = tok
        self.cursor = cursor
        probe = filler_lines(0, 64)
        self.tok_per_line = len(tok.encode(probe, add_special_tokens=False)) / 64.0

    def take(self, target_tokens: int) -> str:
        if target_tokens <= 0:
            return ""
        text = ""
        for _ in range(6):
            need = target_tokens - len(self.tok.encode(text, add_special_tokens=False))
            if need <= 0:
                break
            lines = max(1, int(need / self.tok_per_line) + 2)
            text += filler_lines(self.cursor, lines)
            self.cursor += lines
        ids = self.tok.encode(text, add_special_tokens=False)
        if len(ids) > target_tokens:
            # Slice in token space, then snap back to a line boundary so the decoded
            # text re-encodes to the same ids (no split-token drift between turns).
            text = self.tok.decode(ids[:target_tokens], skip_special_tokens=False)
            text = text[: text.rfind("\n") + 1] or text
        return text


def needle_line(key: str, code: str) -> str:
    return f"\nThe {key} ledger code is {code}.\n"


def build_haystack(tok, target_tokens: int, cursor: int) -> tuple[str, list[dict]]:
    """``target_tokens`` of prompt with every needle at its requested depth.

    Returns the text and, for the record, each needle's realised depth.
    """
    filler = Filler(tok, cursor)
    ntok = lambda s: len(tok.encode(s, add_special_tokens=False))  # noqa: E731

    overhead = ntok(HEAD) + sum(ntok(needle_line(k, c)) for k, c, _ in NEEDLES)
    body_tokens = target_tokens - overhead
    if body_tokens < 4096:
        raise SystemExit("target is too small for the instruction plus the needles")

    ordered = sorted(NEEDLES, key=lambda n: n[2])
    parts = [HEAD]
    placed: list[dict] = []
    previous = 0.0
    consumed = ntok(HEAD)
    for key, code, depth in ordered:
        span = int(round(body_tokens * (depth - previous)))
        chunk = filler.take(span)
        parts.append(chunk)
        consumed += ntok(chunk)
        line = needle_line(key, code)
        parts.append(line)
        placed.append({"key": key, "code": code, "requested_depth": depth,
                       "token_offset": consumed})
        consumed += ntok(line)
        previous = depth
    tail = filler.take(int(round(body_tokens * (1.0 - previous))))
    parts.append(tail)
    text = "".join(parts)
    total = ntok(text)
    for record in placed:
        record["actual_depth"] = record["token_offset"] / total
    for key, code, _ in NEEDLES:
        if needle_line(key, code) not in text:
            raise SystemExit(f"needle for {key} was lost while building the haystack")
    return text, placed


def questions() -> list[dict]:
    """The eight turns, in order. Turn 1 rides along with the haystack."""
    items = [
        {
            "id": f"needle:{key}",
            "kind": "needle",
            "key": key,
            "expect": code,
            "text": (
                f"What is the {key} ledger code? State the seven digits clearly, "
                "then name the ledger they belong to."
            ),
        }
        for key, code, _ in NEEDLES
    ]
    items.append({
        "id": f"control:{CONTROL_KEY}",
        "kind": "control",
        "key": CONTROL_KEY,
        "expect": None,
        "text": (
            f"What is the {CONTROL_KEY} ledger code? If no {CONTROL_KEY} ledger code "
            "appears anywhere in the records, say so plainly instead of guessing."
        ),
    })
    a_key, a_code, _ = NEEDLES[0]
    b_key, b_code, _ = NEEDLES[1]
    items.append({
        "id": "combined",
        "kind": "combined",
        "key": f"{a_key}+{b_key}",
        "expect": str(int(a_code) + int(b_code)),
        "expect_larger": a_key if int(a_code) > int(b_code) else b_key,
        "text": (
            f"Which is larger, the {a_key} ledger code or the {b_key} ledger code? "
            "Name the larger one, then give the exact sum of the two codes as a "
            "single number."
        ),
    })
    return items


_DIGITS = re.compile(r"\d")
_NEGATION = re.compile(
    r"\b(no|not|never|none|absent|does not|doesn't|isn't|is not|cannot|can't|"
    r"nowhere|there is no|do not appear|does not appear)\b",
    re.IGNORECASE,
)


def normalise(text: str) -> str:
    """Strip separators so ``9,854,500`` and ``9 854 500`` match ``9854500``."""
    return re.sub(r"[,\s_.]", "", text)


def grade(item: dict, answer: str) -> dict:
    flat = normalise(answer)
    others = [c for _, c, _ in NEEDLES if c != item.get("expect")]
    wrong = sorted({c for c in others if c in flat})
    if item["kind"] == "needle":
        return {
            "pass": item["expect"] in flat,
            "wrong_codes": wrong,
            "detail": "expected code present" if item["expect"] in flat
            else "expected code absent",
        }
    if item["kind"] == "control":
        fabricated = bool(re.search(r"\d{7}", flat))
        denied = bool(_NEGATION.search(answer))
        return {
            "pass": (not fabricated) and denied,
            "wrong_codes": wrong,
            "detail": ("fabricated a 7-digit code" if fabricated
                       else "denied the key" if denied
                       else "neither denied nor fabricated"),
        }
    # combined: both halves must be right.
    sum_ok = item["expect"] in flat
    larger_ok = item["expect_larger"] in answer.lower()
    return {
        "pass": sum_ok and larger_ok,
        "wrong_codes": [],
        "detail": f"larger={'ok' if larger_ok else 'wrong'}, "
                  f"sum={'ok' if sum_ok else 'wrong'}",
    }


def chat(origin: str, model_id: str, session_id: str, messages: list, decode: int,
         timeout: float) -> dict:
    body = {
        "model": model_id,
        "messages": messages,
        "max_completion_tokens": decode,
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    headers = {"Content-Type": "application/json"}
    if session_id:
        headers["x-switchyard-session-id"] = session_id
    req = urllib.request.Request(
        f"{origin}/v1/chat/completions", data=json.dumps(body).encode(), headers=headers
    )
    stamps: list[float] = []
    pieces: list[str] = []
    usage = None
    t0 = time.perf_counter()
    try:
        response = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as error:
        return {"error": f"HTTP {error.code}: "
                         f"{error.read()[:600].decode(errors='replace')}"}
    with response:
        for raw in response:
            line = raw.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[len(b"data:"):].strip()
            if payload == b"[DONE]":
                break
            now = time.perf_counter()
            chunk = json.loads(payload)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta") or {}
                piece = "".join(
                    str(delta[k]) for k in ("content", "reasoning_content", "reasoning")
                    if delta.get(k)
                )
                if piece:
                    stamps.append(now)
                    pieces.append(piece)
    if usage is None:
        return {"error": "stream ended without a usage chunk"}
    cached = ((usage.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0
    ttft = (stamps[0] - t0) if stamps else None
    decode_rate = None
    if len(stamps) > 1:
        decode_rate = (len(stamps) - 1) / (stamps[-1] - stamps[0])
    fresh = usage["prompt_tokens"] - cached
    return {
        "prompt_tokens": usage["prompt_tokens"],
        "cached_tokens": cached,
        "fresh_tokens": fresh,
        "completion_tokens": usage.get("completion_tokens"),
        "ttft_s": ttft,
        "fresh_prefill_tok_s": (fresh / ttft) if (ttft and fresh > 0) else None,
        "decode_tok_s": decode_rate,
        "wall_s": time.perf_counter() - t0,
        "text": "".join(pieces),
    }


def clean_reply(reply: str) -> str:
    """What can be resent so the next turn's prefix matches byte for byte.

    At long context this model emits a stray ``</think>`` that FreeToken streams as
    ordinary content; echoing it back retokenizes differently from what the server
    generated and costs a full re-prefill.
    """
    for marker in ("<|im_end|>", "<|im_start|>", "<think>", "</think>"):
        if marker in reply:
            reply = reply.split(marker)[0]
    return reply or " "


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://127.0.0.1:8123")
    p.add_argument("--model-dir", required=True,
                   help="checkpoint directory, for the tokenizer only")
    p.add_argument("--target-prompt-tokens", type=int, default=1_044_000)
    p.add_argument("--decode", type=int, default=128)
    p.add_argument("--session-id", default="multineedle-1m")
    p.add_argument("--filler-cursor", type=int, default=0,
                   help="rotate the filler so a previous run's checkpoint cannot match")
    p.add_argument("--timeout", type=float, default=7200.0)
    p.add_argument("--out", help="jsonl of one row per turn")
    p.add_argument("--build-only", action="store_true",
                   help="build and token-count the prompt, contact no server")
    args = p.parse_args()

    tok = load_tokenizer(args.model_dir)
    t_build = time.perf_counter()
    haystack, placed = build_haystack(tok, args.target_prompt_tokens, args.filler_cursor)
    built = len(tok.encode(haystack, add_special_tokens=False))
    print(f"[mn] haystack {built} tokens (target {args.target_prompt_tokens}) "
          f"built in {time.perf_counter() - t_build:.1f}s", flush=True)
    for record in placed:
        print(f"[mn]   {record['key']:8s} {record['code']} at depth "
              f"{record['actual_depth']:.4f} (asked {record['requested_depth']:.2f})",
              flush=True)
    if args.build_only:
        return 0

    origin = args.base_url.rstrip("/")
    with urllib.request.urlopen(f"{origin}/v1/models", timeout=60) as response:
        model_id = json.load(response)["data"][0]["id"]
    print(f"[mn] model={model_id}", flush=True)

    out = open(args.out, "a") if args.out else None
    messages: list[dict] = []
    rows: list[dict] = []
    for index, item in enumerate(questions(), start=1):
        content = (haystack + "\n" + item["text"]) if index == 1 else item["text"]
        messages.append({"role": "user", "content": content})
        row = chat(origin, model_id, args.session_id, messages, args.decode,
                   args.timeout)
        if "error" in row:
            print(f"[mn] turn {index} {item['id']}: ERROR {row['error']}", flush=True)
            if out:
                out.write(json.dumps({"turn": index, **item, **row}) + "\n")
                out.flush()
            return 1
        verdict = grade(item, row["text"])
        row.update({"turn": index, "question_id": item["id"], "kind": item["kind"],
                    "key": item["key"], "expect": item.get("expect"),
                    "prompt_target_tokens": args.target_prompt_tokens,
                    "haystack_tokens": built, "needles": placed,
                    "ts": time.time(), **{f"verdict_{k}": v for k, v in verdict.items()}})
        rows.append(row)
        if out:
            out.write(json.dumps(row) + "\n")
            out.flush()
        print(
            f"[mn] turn {index} {item['id']:18s} "
            f"{'PASS' if verdict['pass'] else 'FAIL'}  "
            f"prompt={row['prompt_tokens']} cached={row['cached_tokens']} "
            f"fresh={row['fresh_tokens']} ttft={row['ttft_s']:.2f}s "
            f"decode={row['decode_tok_s'] and round(row['decode_tok_s'], 2)} tok/s "
            f"[{verdict['detail']}]"
            + (f" wrong_codes={verdict['wrong_codes']}" if verdict["wrong_codes"] else ""),
            flush=True,
        )
        print(f"[mn]      answer: {row['text'].strip()[:400]!r}", flush=True)
        messages.append({"role": "assistant", "content": clean_reply(row["text"])})

    if out:
        out.close()
    passed = sum(1 for r in rows if r["verdict_pass"])
    print(f"\n[mn] {passed}/{len(rows)} questions passed", flush=True)
    return 0 if passed == len(rows) else 2


if __name__ == "__main__":
    sys.exit(main())
