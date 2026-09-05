"""`--speculative ngram` emits the same greedy tokens as the ordinary decode path.

The correctness claim of speculative decoding is that it changes the number of forward
steps and nothing else. Here that is checked against the engine itself: one model load, two
arms toggled on the live scheduler, a copy-heavy prompt (which is what makes the drafter
fire at all), and an assertion that the token ids agree.

The verify forward takes the EXTEND kernels where a decode step takes the graphed decode
kernels, so this is agreement, not bitwise equality -- the same standard the 2026-09-04
extend-tile and 2026-09-05 extend-MoE changes were held to. A divergence, if it ever
appears, is a near-tie in the logits and shows up as a late ``first_diff``, not as garbage.

Gated behind ``needs_weights``:

  FREETOKEN_SPEC_TEST_MODEL   model dir (falls back to FREETOKEN_TEST_MODEL)
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.needs_weights


def _model_dir() -> Path | None:
    value = os.environ.get("FREETOKEN_SPEC_TEST_MODEL") or os.environ.get("FREETOKEN_TEST_MODEL")
    return Path(value).expanduser() if value else None


@pytest.fixture(scope="module")
def llm():
    model = _model_dir()
    if model is None or not model.exists():
        pytest.skip("set FREETOKEN_SPEC_TEST_MODEL (or FREETOKEN_TEST_MODEL)")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from freetoken.llm.llm import LLM

    engine = LLM(
        model_path=str(model),
        dtype=torch.bfloat16,
        max_running_req=1,
        cuda_graph_max_bs=1,
        speculative="ngram",
        spec_ngram_n=8,
        spec_draft_len=8,
    )
    yield engine
    engine.shutdown()


def _copy_heavy_prompt(tok) -> list[int]:
    """A block the model is asked to reproduce: most output tokens are prompt tokens."""
    block = "\n".join(
        f"line {i:03d}: the quick brown fox jumps over the lazy dog number {i}"
        for i in range(40)
    )
    content = (
        "Here is a file:\n\n```\n" + block + "\n```\n\n"
        "Output the COMPLETE file again inside one ``` fence, unchanged. "
        "Do not abbreviate or elide any part of it."
    )
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=True
    )
    if hasattr(rendered, "tolist"):
        rendered = rendered.tolist()
    while isinstance(rendered, (list, tuple)) and rendered and isinstance(rendered[0], (list, tuple)):
        rendered = rendered[0]
    if isinstance(rendered, str):
        rendered = tok(rendered, add_special_tokens=False)["input_ids"]
    return [int(x) for x in rendered]


def test_speculation_is_greedy_equivalent_and_actually_fires(llm):
    from freetoken.core import SamplingParams
    from freetoken.scheduler.spec_ngram import SpecStats

    spec = llm._spec
    assert spec is not None, "the engine refused speculation on this model"

    ids = _copy_heavy_prompt(llm.tokenizer)
    warm = SamplingParams(temperature=0.0, max_tokens=1)
    params = SamplingParams(temperature=0.0, max_tokens=256)

    # Warm the prefix tree so both arms start decoding from the same cached prefix.
    llm._spec = None
    llm.generate([list(ids)], warm)

    baseline = llm.generate([list(ids)], params)[0]["token_ids"]

    llm._spec = spec
    spec.stats = SpecStats()
    spec._state = None
    speculative = llm.generate([list(ids)], params)[0]["token_ids"]

    assert spec.stats.verify_steps > 0, "the drafter never fired; the test proves nothing"
    assert len(baseline) >= 200, f"only {len(baseline)} baseline tokens"
    assert speculative == baseline, (
        f"diverged at token "
        f"{next((i for i, (a, b) in enumerate(zip(baseline, speculative)) if a != b), None)}"
        f" of {len(baseline)}"
    )


def test_a_sampling_request_falls_back_to_the_ordinary_path(llm):
    """Speculation is greedy-only in v1: a temperature request must not take the path."""
    from freetoken.core import SamplingParams
    from freetoken.scheduler.spec_ngram import SpecStats

    spec = llm._spec
    assert spec is not None
    spec.stats = SpecStats()
    spec._state = None
    out = llm.generate(
        [_copy_heavy_prompt(llm.tokenizer)],
        SamplingParams(temperature=0.8, max_tokens=32),
    )[0]
    assert len(out["token_ids"]) > 0
    assert spec.stats.verify_steps == 0
