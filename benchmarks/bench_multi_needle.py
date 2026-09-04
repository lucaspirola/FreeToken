"""Multi-needle long-context recall against an already-running OpenAI-compatible server.

One expensive prefill, many cheap questions. ``bench_long_context.py`` builds a
*single* needle prompt and starts its own server per length, so probing N depths
costs N full prefills -- at 1M tokens that is N x ~30 minutes of GPU. This harness
puts every needle in one haystack and then asks its questions as follow-up *turns*
of the same chat conversation, so turns 2..N hit the prefix cache and cost only
their own question.

Why three question shapes per needle
------------------------------------
The 2026-09-04 1M run (``benchmarks/results/nemotron35_lightning_5080_1m_multineedle_
2026-09-04.md``) scored 5/8 and would have been filed as "1M recall fails at depth
0.25" if it had asked one question per needle. It did not: the *direct* question for
the depth-0.25 needle returned the depth-0.05 needle's code, but the *combined*
question ("which is larger, then sum them") produced 9,854,500 -- whose second addend
is the depth-0.25 code, which appears nowhere earlier in the conversation. The needle
was resident; the direct question failed to address it. A single-shape gate files a
false retention bug.

So every needle is asked three ways:

* **direct** -- ``key -> code``;
* **combined** -- paired with a neighbouring needle, requiring both codes at once;
* **reverse** -- ``code -> key``, naming which of the two same-named records carries it.

and every needle has a **near-duplicate distractor**: ``the orchard ledger`` (the
needle) is shadowed by ``the orchard register`` (a different code, placed half the
haystack away). A miss then classifies instead of merely failing:

``retention``    nothing recovers the code -- it is not in state;
``selection``    a leak-free combined/reverse probe recovers it, the direct one does not;
``interference`` a probe returned the near-duplicate's code (``-near``) or another
                 key's code (``-cross``) -- something *was* addressable, the wrong thing;
``incoherent``   the direct answer carried neither a code nor a denial (decode ran off);
``recall`` / ``recall-partial`` -- direct works, all / not all composed probes do.

Leak control
------------
Every turn's text joins the conversation, so a code that has already been printed can
be read out of the transcript instead of the haystack. Questions are ordered
``direct x N -> control -> combined x N -> reverse x N`` (reverse *states* the code, so
it must come last) and each row records ``leak_free``: whether that needle's code had
appeared anywhere in the conversation before the turn ran. Only leak-free probes count
as evidence that a needle is in state.

Conventions inherited from ``bench_long_context.py``: the filler carries **no digits at
all** (a numeric needle in a numeric haystack is a distractor test, not a retrieval
test), grading reads the concatenated SSE ``content`` fields of ``/v1/chat/completions``
and never the raw stream, and thinking is disabled so the decode budget is not spent
inside a reasoning block.

``ignore_eos`` is deliberately NOT used: a forced-length reply ends in tokens the next
turn cannot resend, and the prefix match is exact, so it would throw away the cache the
whole design depends on.

The server is expected to be already serving (started under ``scripts/gpu_lock.sh``
with the 1M profile from ``docs/nemotron.md``); this script only drives it. The same
suite is driven through llama.cpp by ``benchmarks/oracle_cross_engine.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from bench_long_context import load_tokenizer


#: The record word the needle question selects on, and its near-duplicate shadow.
#: Both records share the *key* ("orchard"), so answering needs the second word too.
NEEDLE_KIND = "ledger"
DISTRACTOR_KIND = "register"


@dataclass(frozen=True)
class Needle:
    """A target record and the near-duplicate planted to compete with it."""

    key: str
    code: str
    depth: float
    distractor_code: str

    @property
    def distractor_depth(self) -> float:
        """Half a haystack away from its twin (+3 % so it lands on no other needle).

        A near-duplicate sitting next to its target would test local blur; the point
        here is key *interference* across the whole context, so they are separated.
        """
        return round((self.depth + 0.53) % 1.0, 4) or 0.02

    @property
    def name(self) -> str:
        return f"{self.key} {NEEDLE_KIND}"

    @property
    def distractor_name(self) -> str:
        return f"{self.key} {DISTRACTOR_KIND}"


#: Depths are fractions of the filler body, so 0.05 really is 5 % of a million tokens
#: in and not "just after the instruction": the 262K bisect showed this model's recall
#: is depth-dependent. The six needle codes are unchanged from the 2026-09-04 1M run so
#: its numbers stay comparable.
NEEDLES: list[Needle] = [
    Needle("orchard", "5663623", 0.05, "2871904"),
    Needle("harbour", "4190877", 0.25, "9435216"),
    Needle("quarry", "8324516", 0.50, "1607392"),
    Needle("cavern", "6082735", 0.60, "3518470"),
    Needle("meadow", "7218459", 0.75, "8043961"),
    Needle("thicket", "3947162", 0.95, "5290638"),
]

#: A key that appears nowhere in the prompt. Must not collide with the filler's
#: vocabulary either, or the control turns into a distractor test.
CONTROL_KEY = "belfry"

HEAD = (
    "You are auditing a warehouse log. The records below are almost all inactive "
    "seal notices, which carry no information. Scattered among them are a few "
    "lines of the form \"The <name> ledger code is <digits>.\" and "
    "\"The <name> register code is <digits>.\" -- a ledger and a register with the "
    "same name are DIFFERENT records with DIFFERENT codes. Remember every one of "
    "them, together with the full record name it belongs to, and ignore every "
    "inactive seal notice.\n\n"
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


def record_line(key: str, kind: str, code: str) -> str:
    return f"\nThe {key} {kind} code is {code}.\n"


def planted_records(needles: list[Needle] | None = None) -> list[dict]:
    """Every line the haystack plants, needles and near-duplicates alike, by depth."""
    needles = NEEDLES if needles is None else needles
    records = []
    for needle in needles:
        records.append({
            "key": needle.key, "kind": NEEDLE_KIND, "code": needle.code,
            "role": "needle", "requested_depth": needle.depth,
        })
        records.append({
            "key": needle.key, "kind": DISTRACTOR_KIND, "code": needle.distractor_code,
            "role": "distractor", "requested_depth": needle.distractor_depth,
        })
    return sorted(records, key=lambda r: r["requested_depth"])


def all_codes(needles: list[Needle] | None = None) -> dict[str, tuple[str, str]]:
    """``code -> (key, role)`` for every planted code."""
    return {
        record["code"]: (record["key"], record["role"])
        for record in planted_records(needles)
    }


def build_haystack(tok, target_tokens: int, cursor: int,
                   needles: list[Needle] | None = None) -> tuple[str, list[dict]]:
    """``target_tokens`` of prompt with every record at its requested depth.

    Returns the text and, for the record, each planted line's realised depth.
    """
    needles = NEEDLES if needles is None else needles
    filler = Filler(tok, cursor)
    ntok = lambda s: len(tok.encode(s, add_special_tokens=False))  # noqa: E731

    records = planted_records(needles)
    lines = {id(r): record_line(r["key"], r["kind"], r["code"]) for r in records}
    overhead = ntok(HEAD) + sum(ntok(line) for line in lines.values())
    body_tokens = target_tokens - overhead
    if body_tokens < 4096:
        raise SystemExit("target is too small for the instruction plus the needles")

    parts = [HEAD]
    placed: list[dict] = []
    previous = 0.0
    consumed = ntok(HEAD)
    for record in records:
        span = int(round(body_tokens * (record["requested_depth"] - previous)))
        chunk = filler.take(span)
        parts.append(chunk)
        consumed += ntok(chunk)
        line = lines[id(record)]
        parts.append(line)
        placed.append({**record, "token_offset": consumed})
        consumed += ntok(line)
        previous = record["requested_depth"]
    parts.append(filler.take(int(round(body_tokens * (1.0 - previous)))))
    text = "".join(parts)
    total = ntok(text)
    for entry in placed:
        entry["actual_depth"] = entry["token_offset"] / total
    for record in records:
        if lines[id(record)] not in text:
            raise SystemExit(
                f"record for {record['key']} {record['kind']} was lost while building"
            )
    return text, placed


# --------------------------------------------------------------------- questions


def questions(needles: list[Needle] | None = None) -> list[dict]:
    """The turns, in order: direct x N, control, combined x N, reverse x N.

    Order is load-bearing. A reverse question *states* the code it is asking about, so
    running it early would put that code in the transcript and make every later probe
    for the same needle unreadable. Directs come first for the same reason.
    """
    needles = NEEDLES if needles is None else needles
    items: list[dict] = []

    for needle in needles:
        items.append({
            "id": f"direct:{needle.key}",
            "shape": "direct",
            "kind": "needle",
            "owner": needle.key,
            "key": needle.key,
            "depth": needle.depth,
            "expect": needle.code,
            "text": (
                f"What is the {needle.name} code? Note that a {needle.distractor_name} "
                f"also exists and has a different code; I want the "
                f"{NEEDLE_KIND} one. State the seven digits clearly, then name the "
                "record they belong to."
            ),
        })

    items.append({
        "id": f"control:{CONTROL_KEY}",
        "shape": "control",
        "kind": "control",
        "owner": None,
        "key": CONTROL_KEY,
        "depth": None,
        "expect": None,
        "text": (
            f"What is the {CONTROL_KEY} {NEEDLE_KIND} code? If no {CONTROL_KEY} "
            f"{NEEDLE_KIND} code appears anywhere in the records, say so plainly "
            "instead of guessing."
        ),
    })

    for index, needle in enumerate(needles):
        partner = needles[(index + 1) % len(needles)]
        if partner.key == needle.key:
            continue
        larger = needle if int(needle.code) > int(partner.code) else partner
        items.append({
            "id": f"combined:{needle.key}+{partner.key}",
            "shape": "combined",
            "kind": "combined",
            "owner": needle.key,
            "partner": partner.key,
            "key": f"{needle.key}+{partner.key}",
            "depth": needle.depth,
            "expect": str(int(needle.code) + int(partner.code)),
            "expect_larger": larger.key,
            "operand_codes": [needle.code, partner.code],
            "text": (
                f"Which is larger, the {needle.name} code or the {partner.name} code? "
                "Name the larger one, then give the exact sum of the two "
                f"{NEEDLE_KIND} codes as a single number. Use the {NEEDLE_KIND} "
                f"records, not the {DISTRACTOR_KIND} ones."
            ),
        })

    for needle in needles:
        items.append({
            "id": f"reverse:{needle.key}",
            "shape": "reverse",
            "kind": "reverse",
            "owner": needle.key,
            "key": needle.key,
            "depth": needle.depth,
            "expect": needle.key,
            "expect_kind": NEEDLE_KIND,
            "reject_kind": DISTRACTOR_KIND,
            "probe_code": needle.code,
            "text": (
                f"Exactly one record in the log has the code {needle.code}. Name that "
                f"record in full -- give both words, as \"<name> {NEEDLE_KIND}\" or "
                f"\"<name> {DISTRACTOR_KIND}\"."
            ),
        })

    return items


# ---------------------------------------------------------------------- grading

_NEGATION = re.compile(
    r"\b(no|not|never|none|absent|does not|doesn't|isn't|is not|cannot|can't|"
    r"nowhere|there is no|do not appear|does not appear)\b",
    re.IGNORECASE,
)
#: Separators *between digits* only: "9,854,500" and "9 854 500" are one number, but
#: "code 12. 5663623" must not be glued into a run that no longer matches any code.
_NUM_SEP = re.compile(r"(?<=\d)[,\s_](?=\d)")


def digit_runs(text: str) -> list[str]:
    return re.findall(r"\d+", _NUM_SEP.sub("", text))


def grade(item: dict, answer: str, needles: list[Needle] | None = None) -> dict:
    """Pass/fail plus the wrong codes that came back, which is what classifies a miss."""
    needles = NEEDLES if needles is None else needles
    codes = all_codes(needles)
    runs = set(digit_runs(answer))
    lowered = answer.lower()

    def wrong_codes(exclude: set[str]) -> list[str]:
        return sorted(run for run in runs if run in codes and run not in exclude)

    if item["shape"] == "direct":
        expect = item["expect"]
        found = expect in runs
        wrong = wrong_codes({expect})
        owner = item["owner"]
        near = sorted(c for c in wrong if codes[c] == (owner, "distractor"))
        cross = sorted(c for c in wrong if codes[c][0] != owner)
        detail = "expected code present" if found else "expected code absent"
        if near:
            detail += f"; returned the {owner} {DISTRACTOR_KIND} code"
        elif cross:
            detail += f"; returned {', '.join(sorted({codes[c][0] for c in cross}))}"
        elif not found and not runs and not _NEGATION.search(answer):
            detail = "no code and no denial"
        return {"pass": found, "wrong_codes": wrong, "near_duplicate_codes": near,
                "cross_key_codes": cross, "denied": bool(_NEGATION.search(answer)),
                "any_digits": bool(runs), "detail": detail}

    if item["shape"] == "control":
        fabricated = any(len(run) == 7 for run in runs)
        denied = bool(_NEGATION.search(answer))
        return {
            "pass": (not fabricated) and denied,
            "wrong_codes": wrong_codes(set()),
            "near_duplicate_codes": [], "cross_key_codes": [],
            "denied": denied, "any_digits": bool(runs),
            "detail": ("fabricated a 7-digit code" if fabricated
                       else "denied the key" if denied
                       else "neither denied nor fabricated"),
        }

    if item["shape"] == "combined":
        sum_ok = item["expect"] in runs
        larger_ok = item["expect_larger"] in lowered
        operands = set(item["operand_codes"])
        wrong = wrong_codes(operands | {item["expect"]})
        near = sorted(c for c in wrong if codes[c][1] == "distractor")
        return {
            "pass": sum_ok and larger_ok,
            "wrong_codes": wrong, "near_duplicate_codes": near,
            "cross_key_codes": sorted(set(wrong) - set(near)),
            "denied": bool(_NEGATION.search(answer)), "any_digits": bool(runs),
            "detail": f"larger={'ok' if larger_ok else 'wrong'}, "
                      f"sum={'ok' if sum_ok else 'wrong'}",
        }

    # reverse: the record name must be right in BOTH words.
    key_ok = item["expect"] in lowered
    kind_ok = item["expect_kind"] in lowered
    took_distractor = bool(
        re.search(rf"{re.escape(item['expect'])}\s+{item['reject_kind']}", lowered)
    )
    # Only interference when the *right* key did not come back: an answer that says
    # "the orchard ledger, not the harbour one" names two keys and is still correct.
    other_keys = [] if key_ok else sorted(
        {n.key for n in needles if n.key != item["expect"] and n.key in lowered}
    )
    return {
        "pass": key_ok and kind_ok and not took_distractor,
        "wrong_codes": [], "near_duplicate_codes": [],
        "cross_key_codes": other_keys,
        "denied": bool(_NEGATION.search(answer)), "any_digits": bool(runs),
        "detail": f"key={'ok' if key_ok else 'wrong'}, "
                  f"kind={'ok' if kind_ok and not took_distractor else 'wrong'}"
                  + (f"; named {DISTRACTOR_KIND}" if took_distractor else "")
                  + (f"; also named {', '.join(other_keys)}" if other_keys else ""),
    }


# --------------------------------------------------------------- classification

#: In precedence order. The first label whose evidence is present wins, so a wrong
#: code beats a bare miss: "something was addressable, the wrong thing" is a strictly
#: sharper finding than "nothing came back".
CLASSES = (
    "recall", "recall-partial", "interference-near", "interference-cross",
    "selection", "incoherent", "retention", "unprobed",
)


def classify_needle(key: str, rows: list[dict]) -> dict:
    """Label one needle from all of its probe rows.

    ``rows`` are graded rows (``question_id``/``shape``/``owner``/``verdict_*``/
    ``leak_free``) for every question that touches this needle -- its own three probes
    plus any combined question in which it is the partner.
    """
    mine = [r for r in rows if key in _row_keys(r)]
    direct = next((r for r in mine if r.get("shape") == "direct"
                   and r.get("owner") == key), None)
    if direct is None:
        return {"key": key, "class": "unprobed", "in_state": False,
                "evidence": "no direct probe was asked", "probes": {}}

    probes = {r["shape"]: bool(r.get("verdict_pass")) for r in mine
              if r.get("shape") in ("direct", "combined", "reverse")}
    support = [r for r in mine
               if r.get("shape") in ("combined", "reverse")
               and r.get("verdict_pass") and r.get("leak_free", True)]
    in_state = bool(direct.get("verdict_pass")) or bool(support)

    near = [c for r in mine for c in (r.get("verdict_near_duplicate_codes") or [])]
    cross = [c for r in mine for c in (r.get("verdict_cross_key_codes") or [])]

    if direct.get("verdict_pass"):
        composed = [r for r in mine if r.get("shape") in ("combined", "reverse")]
        failed = [r["question_id"] for r in composed if not r.get("verdict_pass")]
        if failed:
            return {"key": key, "class": "recall-partial", "in_state": True,
                    "evidence": f"direct passed; {', '.join(failed)} did not",
                    "probes": probes}
        return {"key": key, "class": "recall", "in_state": True,
                "evidence": "every probe passed", "probes": probes}

    if near:
        return {"key": key, "class": "interference-near", "in_state": in_state,
                "evidence": f"a probe returned the {key} {DISTRACTOR_KIND} code "
                            f"({', '.join(sorted(set(near)))})",
                "probes": probes}
    if cross:
        return {"key": key, "class": "interference-cross", "in_state": in_state,
                "evidence": f"a probe returned another record's answer "
                            f"({', '.join(sorted(set(cross)))})",
                "probes": probes}
    if support:
        return {"key": key, "class": "selection", "in_state": True,
                "evidence": "direct missed but leak-free "
                            f"{', '.join(r['question_id'] for r in support)} recovered it",
                "probes": probes}
    if not direct.get("verdict_any_digits") and not direct.get("verdict_denied"):
        return {"key": key, "class": "incoherent", "in_state": False,
                "evidence": "the direct answer carried neither a code nor a denial",
                "probes": probes}
    return {"key": key, "class": "retention", "in_state": False,
            "evidence": "no probe recovered the code", "probes": probes}


def _row_keys(row: dict) -> set[str]:
    return {k for k in (row.get("owner"), row.get("partner")) if k}


def classify_all(rows: list[dict], needles: list[Needle] | None = None) -> list[dict]:
    needles = NEEDLES if needles is None else needles
    return [classify_needle(n.key, rows) for n in needles]


# --------------------------------------------------------------------- transport


def chat(origin: str, model_id: str, session_id: str, messages: list, decode: int,
         timeout: float, *, top_logprobs: int = 0, extra_body: dict | None = None) -> dict:
    """One streamed chat completion. ``top_logprobs > 0`` also collects the streamed
    per-position logprobs, when the server emits them."""
    body = {
        "model": model_id,
        "messages": messages,
        "max_completion_tokens": decode,
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "seed": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if top_logprobs > 0:
        body["logprobs"] = True
        body["top_logprobs"] = top_logprobs
    if extra_body:
        body.update(extra_body)
    headers = {"Content-Type": "application/json"}
    if session_id:
        headers["x-switchyard-session-id"] = session_id
    req = urllib.request.Request(
        f"{origin}/v1/chat/completions", data=json.dumps(body).encode(), headers=headers
    )
    stamps: list[float] = []
    pieces: list[str] = []
    logprobs: list[dict] = []
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
                entries = (choice.get("logprobs") or {}).get("content") or []
                logprobs.extend(entries)
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
        "logprobs": logprobs,
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


# ------------------------------------------------------------------------ driver


def run_suite(origin: str, model_id: str, haystack: str, items: list[dict], *,
              session_id: str, decode: int, timeout: float,
              needles: list[Needle] | None = None, top_logprobs: int = 0,
              first_turn_extra: dict | None = None,
              on_row=None) -> tuple[list[dict], dict]:
    """Drive the whole suite as one conversation. Returns ``(rows, first_turn_extra)``.

    Turn 1 carries the haystack; every later turn rides the prefix cache. ``on_row`` is
    called with each finished row so a CLI can print progress without owning the loop.
    """
    needles = NEEDLES if needles is None else needles
    messages: list[dict] = []
    rows: list[dict] = []
    seen_codes: set[str] = set()
    codes = set(all_codes(needles))
    first_extra: dict = {}

    for index, item in enumerate(items, start=1):
        content = (haystack + "\n" + item["text"]) if index == 1 else item["text"]
        messages.append({"role": "user", "content": content})
        # Leak accounting is decided *before* the turn runs: a probe is evidence only
        # if the code it is about had not already been printed in the conversation.
        owner_code = next((n.code for n in needles if n.key == item.get("owner")), None)
        leak_free = owner_code is None or owner_code not in seen_codes
        row = chat(origin, model_id, session_id, messages, decode, timeout,
                   top_logprobs=top_logprobs,
                   extra_body=first_turn_extra if index == 1 else None)
        if "error" in row:
            rows.append({"turn": index, "question_id": item["id"],
                         "shape": item["shape"], "owner": item.get("owner"),
                         "error": row["error"]})
            if on_row:
                on_row(item, rows[-1])
            break
        if index == 1:
            first_extra = {k: row[k] for k in ("prompt_tokens",) if k in row}
        verdict = grade(item, row["text"], needles)
        row.update({
            "turn": index, "question_id": item["id"], "shape": item["shape"],
            "kind": item["kind"], "owner": item.get("owner"),
            "partner": item.get("partner"), "key": item["key"],
            "depth": item.get("depth"), "expect": item.get("expect"),
            "leak_free": leak_free, "ts": time.time(),
            **{f"verdict_{k}": v for k, v in verdict.items()},
        })
        rows.append(row)
        if on_row:
            on_row(item, row)
        seen_codes.update(codes.intersection(digit_runs(item["text"])))
        seen_codes.update(codes.intersection(digit_runs(row["text"])))
        messages.append({"role": "assistant", "content": clean_reply(row["text"])})
    return rows, first_extra


def haystack_digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://127.0.0.1:8123")
    p.add_argument("--model-dir", required=True,
                   help="checkpoint directory, for the tokenizer only")
    p.add_argument("--target-prompt-tokens", type=int, default=1_044_000)
    p.add_argument("--decode", type=int, default=128)
    p.add_argument("--session-id", default="multineedle-1m")
    p.add_argument("--filler-cursor", type=int, default=0,
                   help="rotate the filler so a previous run's checkpoint cannot "
                        "match; the filler repeats every 64 lines, so use a value "
                        "that is NOT a multiple of 64")
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
          f"sha256={haystack_digest(haystack)[:12]} "
          f"built in {time.perf_counter() - t_build:.1f}s", flush=True)
    for record in placed:
        print(f"[mn]   {record['key']:8s} {record['kind']:8s} {record['code']} at depth "
              f"{record['actual_depth']:.4f} (asked {record['requested_depth']:.2f})",
              flush=True)
    if args.build_only:
        return 0

    origin = args.base_url.rstrip("/")
    with urllib.request.urlopen(f"{origin}/v1/models", timeout=60) as response:
        model_id = json.load(response)["data"][0]["id"]
    print(f"[mn] model={model_id}", flush=True)

    out = open(args.out, "a") if args.out else None

    def report(item: dict, row: dict) -> None:
        if "error" in row:
            print(f"[mn] turn {row['turn']} {item['id']}: ERROR {row['error']}",
                  flush=True)
        else:
            print(
                f"[mn] turn {row['turn']:2d} {item['id']:24s} "
                f"{'PASS' if row['verdict_pass'] else 'FAIL'} "
                f"{'leak-free' if row['leak_free'] else 'LEAKED  '} "
                f"prompt={row['prompt_tokens']} cached={row['cached_tokens']} "
                f"fresh={row['fresh_tokens']} ttft={row['ttft_s']:.2f}s "
                f"[{row['verdict_detail']}]", flush=True)
            print(f"[mn]      answer: {row['text'].strip()[:400]!r}", flush=True)
        if out:
            out.write(json.dumps({**row, "needles": placed,
                                  "haystack_tokens": built}) + "\n")
            out.flush()

    rows, _ = run_suite(origin, model_id, haystack, questions(),
                        session_id=args.session_id, decode=args.decode,
                        timeout=args.timeout, on_row=report)
    if out:
        out.close()
    if any("error" in row for row in rows):
        return 1

    print("\n[mn] classification")
    for entry in classify_all(rows):
        print(f"[mn]   {entry['key']:8s} {entry['class']:18s} "
              f"in_state={str(entry['in_state']):5s} {entry['evidence']}", flush=True)
    passed = sum(1 for r in rows if r.get("verdict_pass"))
    print(f"\n[mn] {passed}/{len(rows)} questions passed", flush=True)
    return 0 if passed == len(rows) else 2


if __name__ == "__main__":
    sys.exit(main())
