"""Needle-suite construction, grading and miss classification, with no server.

The classification ladder is the part that turns a red cell into a filed bug, so it is
tested against the exact failure shapes the 2026-09-04 1M run produced: a direct
question answered with a *different* needle's code while a combined question proved the
needle was in state.
"""

from __future__ import annotations

import sys
from pathlib import Path


BENCH = Path(__file__).parents[2] / "benchmarks"
sys.path.insert(0, str(BENCH))
import bench_multi_needle as mn  # noqa: E402


class CharTokenizer:
    """One token per character: exact, deterministic, and re-encode stable."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)


# ------------------------------------------------------------------ the haystack


def test_every_record_is_planted_and_the_twins_are_far_apart():
    text, placed = mn.build_haystack(CharTokenizer(), 40_000, cursor=0)

    assert len(placed) == 2 * len(mn.NEEDLES)
    for needle in mn.NEEDLES:
        assert mn.record_line(needle.key, mn.NEEDLE_KIND, needle.code) in text
        assert mn.record_line(needle.key, mn.DISTRACTOR_KIND,
                              needle.distractor_code) in text
        depths = {r["role"]: r["actual_depth"] for r in placed if r["key"] == needle.key}
        assert abs(depths["needle"] - depths["distractor"]) > 0.3


def test_the_filler_carries_no_digits_outside_the_planted_records():
    text, placed = mn.build_haystack(CharTokenizer(), 40_000, cursor=0)
    planted = {record["code"] for record in placed}
    assert set(mn.digit_runs(text)) == planted


def test_planted_codes_are_all_distinct():
    codes = [record["code"] for record in mn.planted_records()]
    assert len(set(codes)) == len(codes)
    assert all(len(code) == 7 for code in codes)
    assert mn.CONTROL_KEY not in {needle.key for needle in mn.NEEDLES}


def test_distractor_depths_land_on_no_needle_depth():
    needle_depths = {needle.depth for needle in mn.NEEDLES}
    for needle in mn.NEEDLES:
        assert needle.distractor_depth not in needle_depths
        assert 0.0 < needle.distractor_depth < 1.0


def test_the_haystack_is_deterministic_and_digest_stable():
    a, _ = mn.build_haystack(CharTokenizer(), 40_000, cursor=0)
    b, _ = mn.build_haystack(CharTokenizer(), 40_000, cursor=0)
    c, _ = mn.build_haystack(CharTokenizer(), 40_000, cursor=7)
    assert mn.haystack_digest(a) == mn.haystack_digest(b)
    assert mn.haystack_digest(a) != mn.haystack_digest(c)


def test_the_filler_cursor_repeats_every_64_lines():
    """A cursor that is a multiple of 64 rotates nothing -- 8 shades x 8 materials.

    ``--filler-cursor`` exists so a previous run's session checkpoint cannot match the
    new prompt. Passing 64 (or 128, or 1024) silently produces a byte-identical
    haystack and the old checkpoint matches after all, so the runbook tells callers to
    use a cursor that is not a multiple of 64.
    """
    assert mn.filler_lines(0, 4) == mn.filler_lines(64, 4)
    assert mn.filler_lines(0, 4) != mn.filler_lines(7, 4)


# ------------------------------------------------------------------- the questions


def test_reverse_questions_come_after_every_probe_that_they_would_leak_into():
    items = mn.questions()
    shapes = [item["shape"] for item in items]
    last_direct = max(i for i, s in enumerate(shapes) if s == "direct")
    last_combined = max(i for i, s in enumerate(shapes) if s == "combined")
    first_reverse = min(i for i, s in enumerate(shapes) if s == "reverse")
    assert first_reverse > last_combined > last_direct

    # Only the reverse questions may state a code, and each states only its own.
    for item in items:
        stated = set(mn.digit_runs(item["text"])) & set(mn.all_codes())
        if item["shape"] == "reverse":
            assert stated == {item["probe_code"]}
        else:
            assert stated == set()


def test_each_needle_is_asked_three_ways_plus_one_control():
    items = mn.questions()
    for needle in mn.NEEDLES:
        owned = [i["shape"] for i in items if i.get("owner") == needle.key]
        assert sorted(owned) == ["combined", "direct", "reverse"]
    assert [i["id"] for i in items].count(f"control:{mn.CONTROL_KEY}") == 1
    assert len({item["id"] for item in items}) == len(items)


def test_direct_questions_name_the_near_duplicate_so_a_miss_is_a_choice():
    for item in mn.questions():
        if item["shape"] == "direct":
            assert mn.DISTRACTOR_KIND in item["text"]
            assert mn.NEEDLE_KIND in item["text"]


# --------------------------------------------------------------------- grading


def _item(shape: str, key: str = "harbour"):
    return next(i for i in mn.questions()
                if i["shape"] == shape and i.get("owner") == key)


def test_thousands_separators_are_one_number_but_sentence_breaks_are_not():
    assert mn.digit_runs("the sum is 9,854,500.") == ["9854500"]
    assert mn.digit_runs("9 854 500") == ["9854500"]
    # "...12. 5663623..." must not glue into 125663623 and lose the code.
    assert "5663623" in mn.digit_runs("record 12. 5663623 follows")


def test_direct_grade_separates_near_duplicate_from_cross_key_interference():
    item = _item("direct", "harbour")
    near = mn.grade(item, "The harbour register code is 9435216.")
    assert not near["pass"] and near["near_duplicate_codes"] == ["9435216"]
    assert near["cross_key_codes"] == []

    cross = mn.grade(item, "The orchard ledger code is 5663623.")
    assert not cross["pass"] and cross["cross_key_codes"] == ["5663623"]
    assert cross["near_duplicate_codes"] == []

    good = mn.grade(item, "The code is 4190877, the harbour ledger.")
    assert good["pass"] and good["wrong_codes"] == []


def test_direct_grade_flags_an_answer_with_no_code_and_no_denial():
    verdict = mn.grade(_item("direct"), "The warden notes that the seal remains quiet.")
    assert not verdict["pass"]
    assert not verdict["any_digits"] and not verdict["denied"]


def test_control_passes_only_on_an_explicit_denial():
    control = next(i for i in mn.questions() if i["shape"] == "control")
    assert mn.grade(control, "No belfry ledger code appears in the records.")["pass"]
    assert not mn.grade(control, "The belfry ledger code is 1234567.")["pass"]
    assert not mn.grade(control, "The belfry ledger.")["pass"]


def test_combined_grade_needs_both_the_larger_name_and_the_exact_sum():
    item = _item("combined", "orchard")
    assert item["expect"] == str(int("5663623") + int("4190877"))
    assert mn.grade(item, "The orchard code is larger. The sum is 9,854,500.")["pass"]
    assert not mn.grade(item, "The harbour code is larger. The sum is 9854500.")["pass"]
    assert not mn.grade(item, "The orchard code is larger. The sum is 9854501.")["pass"]


def test_reverse_grade_rejects_the_register_and_tolerates_naming_both_words():
    item = _item("reverse", "harbour")
    assert mn.grade(item, "That is the harbour ledger.")["pass"]
    assert not mn.grade(item, "That is the harbour register.")["pass"]
    assert not mn.grade(item, "That is the orchard ledger.")["pass"]
    # Naming a rival while still answering correctly is not interference.
    both = mn.grade(item, "The harbour ledger, not the orchard ledger.")
    assert both["pass"] and both["cross_key_codes"] == []


# -------------------------------------------------------------- classification


def _row(question_id, shape, owner, passed, *, leak_free=True, near=(), cross=(),
         digits=True, denied=False, partner=None):
    return {"question_id": question_id, "shape": shape, "owner": owner,
            "partner": partner, "leak_free": leak_free, "verdict_pass": passed,
            "verdict_near_duplicate_codes": list(near),
            "verdict_cross_key_codes": list(cross),
            "verdict_any_digits": digits, "verdict_denied": denied}


def test_all_probes_passing_is_recall():
    rows = [_row("direct:quarry", "direct", "quarry", True),
            _row("combined:quarry+cavern", "combined", "quarry", True,
                 partner="cavern"),
            _row("reverse:quarry", "reverse", "quarry", True)]
    entry = mn.classify_needle("quarry", rows)
    assert entry["class"] == "recall" and entry["in_state"]


def test_a_working_direct_probe_with_a_failing_composed_one_is_recall_partial():
    rows = [_row("direct:quarry", "direct", "quarry", True),
            _row("combined:quarry+cavern", "combined", "quarry", False,
                 partner="cavern"),
            _row("reverse:quarry", "reverse", "quarry", True)]
    entry = mn.classify_needle("quarry", rows)
    assert entry["class"] == "recall-partial" and entry["in_state"]
    assert "combined:quarry+cavern" in entry["evidence"]


def test_the_1m_regression_shape_classifies_as_interference_not_retention():
    """Direct returned another needle's code; the combined question recovered it.

    This is exactly the 2026-09-04 1M depth-0.25 result. It must never come back as
    ``retention`` -- that reading filed a false bug once already.
    """
    rows = [_row("direct:harbour", "direct", "harbour", False, cross=["5663623"]),
            _row("combined:harbour+quarry", "combined", "harbour", True,
                 partner="quarry"),
            _row("reverse:harbour", "reverse", "harbour", True)]
    entry = mn.classify_needle("harbour", rows)
    assert entry["class"] == "interference-cross"
    assert entry["in_state"] is True


def test_the_near_duplicate_twin_outranks_a_cross_key_miss():
    rows = [_row("direct:harbour", "direct", "harbour", False,
                 near=["9435216"], cross=["5663623"]),
            _row("combined:harbour+quarry", "combined", "harbour", False,
                 partner="quarry"),
            _row("reverse:harbour", "reverse", "harbour", False)]
    entry = mn.classify_needle("harbour", rows)
    assert entry["class"] == "interference-near"
    assert not entry["in_state"]


def test_a_clean_miss_that_a_leak_free_probe_recovers_is_selection():
    rows = [_row("direct:meadow", "direct", "meadow", False, denied=True),
            _row("combined:meadow+thicket", "combined", "meadow", True,
                 partner="thicket"),
            _row("reverse:meadow", "reverse", "meadow", False)]
    entry = mn.classify_needle("meadow", rows)
    assert entry["class"] == "selection" and entry["in_state"]


def test_a_leaked_probe_is_not_evidence_that_the_needle_was_in_state():
    rows = [_row("direct:meadow", "direct", "meadow", False, denied=True),
            _row("combined:meadow+thicket", "combined", "meadow", True,
                 leak_free=False, partner="thicket"),
            _row("reverse:meadow", "reverse", "meadow", False, leak_free=False)]
    entry = mn.classify_needle("meadow", rows)
    assert entry["class"] == "retention" and not entry["in_state"]


def test_a_direct_answer_with_neither_code_nor_denial_is_incoherent():
    rows = [_row("direct:cavern", "direct", "cavern", False, digits=False),
            _row("combined:cavern+meadow", "combined", "cavern", False,
                 partner="meadow"),
            _row("reverse:cavern", "reverse", "cavern", False)]
    assert mn.classify_needle("cavern", rows)["class"] == "incoherent"


def test_a_partner_only_row_still_counts_toward_the_partners_classification():
    rows = [_row("direct:thicket", "direct", "thicket", False, denied=True),
            _row("combined:meadow+thicket", "combined", "meadow", True,
                 partner="thicket")]
    entry = mn.classify_needle("thicket", rows)
    assert entry["class"] == "selection" and entry["in_state"]


def test_a_needle_with_no_direct_probe_is_unprobed_rather_than_a_pass():
    entry = mn.classify_needle("quarry", [])
    assert entry["class"] == "unprobed" and not entry["in_state"]


def test_classify_all_covers_every_needle_and_only_uses_known_labels():
    entries = mn.classify_all([])
    assert [e["key"] for e in entries] == [n.key for n in mn.NEEDLES]
    assert all(e["class"] in mn.CLASSES for e in entries)
