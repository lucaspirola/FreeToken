"""Needle haystacks for teacher-forced retrieval scoring.

``_WORDS``, ``_plant`` and ``build`` are copied verbatim from FreeToken's
tasks/exclusive-expert-ram/needles.py (lines 84-171, FreeToken-wt/reorg 33dc23e): the same
filler, the same seven planted evidence types at the same depths. The questions are the
subset with a short, deterministic answer, re-phrased to pin the answer FORMAT (thinking
off), because here the answer is not sampled: it is teacher-forced and scored by its
log-probability, per-token rank and whether greedy decoding would reproduce it exactly.
"""
from __future__ import annotations

import random

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



# ---------------------------------------------------------------- teacher-forced questions
# name -> (question, answer(truth) -> str). Answers are what a correct, terse reply is.
TF_QUESTIONS = {
    "codes": (
        "List the three authorization codes that appear in the text above, in the order "
        "they appear, separated by commas, and nothing else.",
        lambda t: ", ".join(t["codes"])),
    "multihop": (
        "Delia Marsh's manager is named somewhere in the text above, and that manager's "
        "office location is named somewhere else in it. In which city does Delia Marsh's "
        "manager sit? Answer with the city name only.",
        lambda t: t["multihop"]["city"]),
    "contradiction": (
        "What is the current total for the Calloway account according to the text above? "
        "Answer with the number only.",
        lambda t: t["contradiction"]["revised"]),
    "arithmetic": (
        "Three depots are each described in the text above as receiving some number of "
        "units. What is the sum of those three numbers? Answer with the total only.",
        lambda t: str(t["arithmetic"]["sum"])),
    "counting": (
        "Exactly how many times does the phrase \"crimson ledger\" appear in the text "
        "above? Answer with the number only.",
        lambda t: str(t["counting"]["n"])),
}
