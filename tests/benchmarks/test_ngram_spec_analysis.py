"""CPU-only coverage for the prompt-lookup speculative-decoding analysis.

No GPU and no model: the drafter, the acceptance replay and the cost model are pure
functions of a token sequence. They are worth testing because the 2026-09-05 go/no-go
rests on the numbers they produce -- in particular on the drafter actually firing, which
an earlier version did not (it indexed the query n-gram itself, so every lookup found a
self-match and no draft was ever issued; `test_drafter_does_not_self_match` is that bug).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
import ngram_spec_analysis as nsa  # noqa: E402


def _repeating(prompt_len: int = 200, span: int = 120) -> tuple[list[int], int]:
    """A transcript whose generation copies a span of its own prompt verbatim."""
    prompt = [(i * 7919) % 5000 for i in range(prompt_len)]
    output = prompt[20 : 20 + span]
    return prompt + output, prompt_len


def _novel(prompt_len: int = 200, out_len: int = 120) -> tuple[list[int], int]:
    tokens = [i for i in range(prompt_len + out_len)]  # every token unique -> no n-gram repeats
    return tokens, prompt_len


def test_drafter_does_not_self_match():
    """The n-gram ending at the cursor is the query, not a prediction."""
    tokens = list(range(50))
    d = nsa.NgramDrafter([3])
    d.observe(tokens, 40)
    # tokens[37:40] occurs exactly once, at the cursor itself: nothing to draft from.
    assert d.draft(tokens, 40, 4) == []


def test_drafter_finds_earlier_occurrence():
    tokens = [1, 2, 3, 9, 9, 9, 9, 1, 2, 3]
    d = nsa.NgramDrafter([3])
    d.observe(tokens, 10)
    # The 3-gram (1,2,3) at the cursor recurs from index 0, which predicts 9, 9, 9.
    assert d.draft(tokens, 10, 3) == [9, 9, 9]


def test_drafter_prefers_the_longest_n():
    # A 3-gram match points at 100; a 5-gram match points at 200. Longest wins.
    tokens = [5, 6, 7, 8, 9, 200, 0, 0, 7, 8, 9, 100, 0, 0, 5, 6, 7, 8, 9]
    d = nsa.NgramDrafter([5, 3])
    d.observe(tokens, len(tokens))
    assert d.draft(tokens, len(tokens), 1) == [200]
    # With only the short n available, the same cursor takes the more recent, worse match.
    short = nsa.NgramDrafter([3])
    short.observe(tokens, len(tokens))
    assert short.draft(tokens, len(tokens), 1) == [100]


@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_replay_reproduces_the_transcript_exactly(k):
    """Greedy equivalence in miniature: speculation changes the step count, never the
    token count. Every accepted token came from the recorded greedy continuation."""
    for tokens, start in (_repeating(), _novel()):
        sim = nsa.simulate(tokens, start, [5, 4, 3], k, adaptive=False)
        assert sim["generated"] == len(tokens) - start
        assert sim["accepted_total"] <= sim["drafted_total"]


def test_copy_heavy_transcript_accepts_almost_everything():
    tokens, start = _repeating()
    sim = nsa.simulate(tokens, start, [5, 4, 3], 4, adaptive=False)
    proj = nsa.project(sim, 0.5)
    assert proj["draft_rate"] > 0.8
    assert proj["accept_rate"] > 0.9
    assert proj["lambda"] > 3.0
    assert proj["speedup"] > 1.0


def test_novel_transcript_never_drafts_and_never_speeds_up():
    tokens, start = _novel()
    sim = nsa.simulate(tokens, start, [5, 4, 3], 8, adaptive=False)
    proj = nsa.project(sim, 0.63)
    assert proj["draft_rate"] == 0.0
    assert proj["lambda"] == 1.0
    assert proj["speedup"] == 1.0


def test_widths_account_for_every_verify_step():
    tokens, start = _repeating()
    sim = nsa.simulate(tokens, start, [5, 4, 3], 6, adaptive=True)
    assert sum(sim["widths"].values()) == sim["steps_drafted"]
    # A verify forward covers the draft plus the token it extends.
    assert all(2 <= m <= 7 for m in sim["widths"])


def test_cost_from_routing_interpolates_the_two_anchor_points():
    routing = {"1": 6.0, "2": 11.61, "9": 33.23}
    cost = nsa.cost_from_routing(routing, (6.0, 11.61, 1.0, 1.63))
    assert cost[1] == pytest.approx(1.0)
    assert cost[2] == pytest.approx(1.63)
    # Slope is 0.63 per 5.61 extra experts; 33.23 experts is (33.23-6)/5.61 steps out.
    assert cost[9] == pytest.approx(1.0 + 0.63 * (33.23 - 6.0) / (11.61 - 6.0))


def test_measured_cost_table_beats_the_flat_model_when_routing_overlaps():
    """The flat model charges every extra token the first one's price; the measured
    table charges the shared experts once, so it must project at least as well."""
    tokens, start = _repeating()
    sim = nsa.simulate(tokens, start, [5, 4, 3], 4, adaptive=False)
    table = nsa.cost_from_routing(
        {"1": 6.0, "2": 10.56, "3": 14.57, "4": 18.24, "5": 21.64},
        (6.0, 11.61, 1.0, 1.63),
    )
    assert nsa.project(sim, table)["speedup"] > nsa.project(sim, 0.63)["speedup"]
