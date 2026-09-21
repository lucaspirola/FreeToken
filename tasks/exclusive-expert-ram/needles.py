#!/usr/bin/env python3
"""The harder needle battery: seven question types over one long haystack.

Why this exists
---------------
``recall.py`` asks for three planted codes. That is a retrieval test, and a
retrieval test is the easiest thing for a degraded cache to pass: copying a
literal string back out of the KV needs far less of the model than reasoning
over it. A mirror that serves the WRONG experts, or a KV lane quantized past
what the model tolerates, can still parrot a code while getting every question
that needs two facts, a count or an order wrong.

So this plants seven kinds of evidence in ONE haystack and asks seven kinds of
question about it. Types 1 and 7 are retrieval (thinking OFF -- reasoning
would only burn budget at 700K). Types 2-6 need the model to combine, count,
order, reconcile or add facts that sit far apart (thinking ON, the served
default). Every answer is scored 0/1 by a checker that looks for the fact, not
for a phrasing.

Prefill once, ask many
----------------------
A 700K haystack costs a minute of prefill. Sending it seven times would cost
seven minutes per size and measure nothing new. The server keeps a radix prefix
cache, so the haystack is sent ONCE (that request is the honest, uncached
prefill number) and every question afterwards repeats the same haystack
verbatim as its prefix and hits the cache -- seconds per question. Each result
carries ``cached_tokens`` so a miss is visible rather than silently expensive;
if that number collapses to ~0 the questions are re-prefilling and the timings
below are not comparable.

Keep the lane free while this runs: the server is single-lane, and another
prompt interleaved between questions evicts the prefix.

    needles.py 21000 240000 713000 1000000
    FREETOKEN_MODEL_NAME=ornith needles.py 21000 120000 250000

Environment: FREETOKEN_URL (default http://127.0.0.1:1920),
FREETOKEN_MODEL_NAME (default nemotron-3.5-lightning),
NEEDLES_OUT (write the JSON report here as well as stdout),
NEEDLES_MAX_TOKENS (answer budget, default 2048 -- thinking needs room).

Exit status is 1 if any type-1 (codes) question failed at any size: that is the
hard gate, the same one recall.py enforces. The other six are reported and
compared against the whole-model arm, which is the reference -- the reference is
the MODEL, not this harness: a question the whole-model arm also gets wrong is
a limit of the model at that length, not a regression of the pool.
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
import time
import urllib.request

URL = os.environ.get("FREETOKEN_URL", "http://127.0.0.1:1920")
NAME = os.environ.get("FREETOKEN_MODEL_NAME", "nemotron-3.5-lightning")
MAX_TOKENS = int(os.environ.get("NEEDLES_MAX_TOKENS", "2048"))

# Filler that tokenizes densely and answers nothing on its own. Deliberately
# the same vocabulary recall.py uses, so a haystack of a given size costs the
# same prefill in both harnesses.
_WORDS = ("ledger entry audit column revision batch quarter invoice reconcile "
          "variance schedule appendix clause remittance").split()

# ---------------------------------------------------------------- planting


def _plant(body: list[str], frac: float, sentence: str) -> None:
    """Insert ``sentence`` at ``frac`` of the way through ``body``."""
    body.insert(int(len(body) * frac), sentence)


def build(tokens: int, rng: random.Random) -> tuple[str, dict]:
    """One haystack of roughly ``tokens`` tokens carrying all seven needles.

    Returns the text and the ground truth. Insertions are done from the END
    backwards so that an earlier insertion does not shift a later fraction.
    """
    words = int(tokens / 1.3)            # ~1.3 tokens per word for this filler
    body = [rng.choice(_WORDS) for _ in range(words)]
    truth: dict = {}

    # (1) codes -- three authorization codes at 2 / 50 / 97 %.
    codes = []
    for _ in range(3):
        codes.append("-".join((
            rng.choice(["SIERRA", "TANGO", "KILO", "ROMEO", "ZULU"]),
            str(rng.randrange(1000, 9999)),
        )))
    truth["codes"] = codes

    # (2) multi-hop -- the two halves are 50 % of the document apart, so the
    #     model cannot answer from either one alone.
    truth["multihop"] = {"person": "Delia Marsh", "manager": "Ov Petran",
                         "city": "Braunfels"}

    # (3) counting -- a phrase that occurs nowhere in the filler, N times.
    truth["counting"] = {"phrase": "crimson ledger", "n": 7}

    # (4) ordering -- five dated events, planted out of chronological order.
    events = [("the Halberd audit", "1994-03-12"),
              ("the Ostrey merger", "1987-11-02"),
              ("the Pellow recall", "2003-07-19"),
              ("the Innsmouth filing", "1979-01-30"),
              ("the Verrick settlement", "2011-09-08")]
    truth["ordering"] = [e[0] for e in sorted(events, key=lambda e: e[1])]

    # (5) contradiction -- a value stated, then revised much later.
    truth["contradiction"] = {"first": "48,200", "revised": "51,750"}

    # (6) arithmetic -- three quantities scattered apart; the answer is a sum
    #     that appears nowhere in the text.
    quantities = [1420, 385, 2073]
    truth["arithmetic"] = {"parts": quantities, "sum": sum(quantities)}

    # (7) negative -- a code that is NOT planted. Nothing to insert.
    truth["negative"] = {"code": "OSCAR-6621"}

    # Insert from the deepest fraction backwards.
    _plant(body, 0.97, f"The authorization code for this section is {codes[2]}.")
    _plant(body, 0.90, "The crimson ledger was sealed that evening.")
    _plant(body, 0.86, f"Amended figure: the {truth['contradiction']['first']} "
                       f"quoted earlier was wrong; the correct total for the "
                       f"Calloway account is {truth['contradiction']['revised']} "
                       f"units.")
    _plant(body, 0.82, f"{events[4][0].capitalize()} was concluded on {events[4][1]}.")
    _plant(body, 0.78, "A crimson ledger sat unopened on the sill.")
    _plant(body, 0.74, f"The Tarn depot received {quantities[2]} units.")
    _plant(body, 0.70, "They filed the crimson ledger with the clerk.")
    _plant(body, 0.66, f"{events[2][0].capitalize()} was concluded on {events[2][1]}.")
    _plant(body, 0.60, f"{truth['multihop']['manager']}'s office is in "
                       f"{truth['multihop']['city']}.")
    _plant(body, 0.56, "Someone had annotated the crimson ledger in pencil.")
    _plant(body, 0.50, f"The authorization code for this section is {codes[1]}.")
    _plant(body, 0.46, f"The Wexley depot received {quantities[1]} units.")
    _plant(body, 0.42, "The crimson ledger listed no remittance at all.")
    _plant(body, 0.38, f"{events[0][0].capitalize()} was concluded on {events[0][1]}.")
    _plant(body, 0.34, "Every crimson ledger in the room bore the same seal.")
    _plant(body, 0.30, f"The Calloway account total is {truth['contradiction']['first']} units.")
    _plant(body, 0.26, f"{events[3][0].capitalize()} was concluded on {events[3][1]}.")
    _plant(body, 0.22, f"The Harrow depot received {quantities[0]} units.")
    _plant(body, 0.18, "The crimson ledger, finally, was returned to the vault.")
    _plant(body, 0.14, f"{events[1][0].capitalize()} was concluded on {events[1][1]}.")
    _plant(body, 0.10, f"{truth['multihop']['person']}'s manager is "
                       f"{truth['multihop']['manager']}.")
    _plant(body, 0.02, f"The authorization code for this section is {codes[0]}.")

    return " ".join(body), truth


# ---------------------------------------------------------------- scoring

def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).lower()


def _score_codes(ans: str, t: dict) -> bool:
    """All three, in the order they appear in the document."""
    positions = [ans.find(c) for c in t["codes"]]
    return all(p >= 0 for p in positions) and positions == sorted(positions)


def _score_multihop(ans: str, t: dict) -> bool:
    return t["multihop"]["city"].lower() in _norm(ans)


def _score_counting(ans: str, t: dict) -> bool:
    n = t["counting"]["n"]
    low = _norm(ans)
    words = {7: "seven", 6: "six", 8: "eight"}
    # The count must be the answer, not a stray number: accept the digit as a
    # standalone token or the spelled-out word.
    return bool(re.search(rf"\b{n}\b", low)) or words.get(n, "") in low


def _score_ordering(ans: str, t: dict) -> bool:
    """The five labels must appear in chronological order in the answer."""
    low = _norm(ans)
    seen = []
    for label in t["ordering"]:
        key = label.split()[-1].lower()          # Halberd / Ostrey / ...
        pos = low.find(key)
        if pos < 0:
            return False
        seen.append((pos, label))
    return [lbl for _, lbl in sorted(seen)] == t["ordering"]


def _score_contradiction(ans: str, t: dict) -> bool:
    """The revised value, and an acknowledgement that it changed."""
    low = _norm(ans)
    revised = t["contradiction"]["revised"].lower()
    bare = revised.replace(",", "")
    has_value = revised in low or bare in low
    changed = any(w in low for w in
                  ("chang", "correct", "revis", "amend", "updat", "yes"))
    return has_value and changed


def _score_arithmetic(ans: str, t: dict) -> bool:
    total = t["arithmetic"]["sum"]
    low = _norm(ans).replace(",", "")
    return bool(re.search(rf"\b{total}\b", low))


def _score_negative(ans: str, t: dict) -> bool:
    """The right answer is that it is not there -- and it must not hallucinate
    a location for it."""
    low = _norm(ans)
    denies = any(w in low for w in (
        "not present", "does not appear", "doesn't appear", "not found",
        "no such", "not in the text", "not mentioned", "absent",
        "cannot find", "can't find", "does not occur", "no occurrence"))
    return denies


# type -> (question, thinking, scorer)
QUESTIONS: dict[str, tuple[str, bool, object]] = {
    "codes": (
        "List the three authorization codes that appear in the text above, in "
        "the order they appear, and nothing else.",
        False, _score_codes),
    "multihop": (
        "Delia Marsh's manager is named somewhere in the text above, and that "
        "manager's office location is named somewhere else in it. In which "
        "city does Delia Marsh's manager sit? Answer with the city name.",
        True, _score_multihop),
    "counting": (
        "Exactly how many times does the phrase \"crimson ledger\" appear in "
        "the text above? Answer with the number only.",
        True, _score_counting),
    "ordering": (
        "Five events are described in the text above, each with a date. List "
        "all five in chronological order, earliest first, one per line, using "
        "the names given.",
        True, _score_ordering),
    "contradiction": (
        "What is the current total for the Calloway account according to the "
        "text above, and did that figure change anywhere in the text? Answer "
        "with the current value and yes or no.",
        True, _score_contradiction),
    "arithmetic": (
        "Three depots are each described in the text above as receiving some "
        "number of units. What is the sum of those three numbers? Answer with "
        "the total only.",
        True, _score_arithmetic),
    "negative": (
        "What is the authorization code OSCAR-6621 used for in the text above? "
        "If that code does not appear anywhere in the text, say that it is not "
        "present.",
        False, _score_negative),
}


# ---------------------------------------------------------------- transport

def _post(messages: list[dict], thinking: bool, max_tokens: int) -> dict:
    payload = {
        "model": NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": bool(thinking)},
    }
    req = urllib.request.Request(
        f"{URL}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=7200) as r:
        body = json.load(r)
    body["_seconds"] = round(time.perf_counter() - t0, 2)
    return body


def _cached(usage: dict) -> int | None:
    """Cached prompt tokens, however this build reports them."""
    for key in ("cached_tokens", "prompt_cache_hit_tokens"):
        if key in usage:
            return usage[key]
    details = usage.get("prompt_tokens_details") or {}
    return details.get("cached_tokens")


def run_size(target: int, rng: random.Random) -> dict:
    haystack, truth = build(target, rng)
    out: dict = {"target": target, "questions": {}}

    # The prefill of record: the haystack, uncached, asked something trivial so
    # the answer costs nothing. Every question below reuses this prefix.
    warm = _post([{"role": "user", "content": haystack + "\n\nReply with the "
                                               "single word: ready."}],
                 thinking=False, max_tokens=8)
    usage = warm.get("usage", {}) or {}
    out["prompt_tokens"] = usage.get("prompt_tokens")
    out["prefill_seconds"] = warm["_seconds"]
    out["prefill_cached_tokens"] = _cached(usage)

    for kind, (question, thinking, scorer) in QUESTIONS.items():
        try:
            body = _post(
                [{"role": "user", "content": haystack + "\n\n" + question}],
                thinking=thinking, max_tokens=MAX_TOKENS)
        except Exception as exc:                       # a failure is a result
            out["questions"][kind] = {"error": f"{type(exc).__name__}: {exc}",
                                      "thinking": thinking, "correct": False}
            continue
        msg = body["choices"][0]["message"]
        text = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        u = body.get("usage", {}) or {}
        out["questions"][kind] = {
            "thinking": thinking,
            "correct": bool(scorer(text, truth)),
            "seconds": body["_seconds"],
            "cached_tokens": _cached(u),
            "completion_tokens": u.get("completion_tokens"),
            # Reasoning is billed as completion; report it so a thinking-ON
            # column can be read against a thinking-OFF one.
            "reasoning_chars": len(reasoning),
            "finish_reason": body["choices"][0].get("finish_reason"),
            "answer": text.strip()[:400],
        }
    out["truth"] = truth
    out["score"] = sum(1 for q in out["questions"].values() if q.get("correct"))
    return out


def main(sizes: list[str]) -> int:
    rng = random.Random(20260922)
    report = {"url": URL, "model": NAME, "max_tokens": MAX_TOKENS, "sizes": []}
    hard_failure = 0
    for raw in sizes:
        res = run_size(int(raw), rng)
        report["sizes"].append(res)
        if not res["questions"].get("codes", {}).get("correct"):
            hard_failure = 1
        line = " ".join(f"{k}={'ok' if v.get('correct') else 'FAIL'}"
                        for k, v in res["questions"].items())
        print(f"[{res['target']} -> {res.get('prompt_tokens')} tok] "
              f"{res['score']}/7  {line}", flush=True)
    text = json.dumps(report, indent=1)
    dest = os.environ.get("NEEDLES_OUT")
    if dest:
        with open(dest, "w") as fh:
            fh.write(text + "\n")
        print(f"wrote {dest}", flush=True)
    else:
        print(text, flush=True)
    return hard_failure


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or ["21000", "240000", "713000"]))
