"""``NemotronHBackbone.forward``'s hidden-state hook.

What the router consumes is the *post-block residual stream*: the value each block
leaves behind after adding its mixer output, before the next block's input norm and
before ``norm_f``. This drives the real ``NemotronHBackbone.forward`` (bound to a
stand-in ``self`` -- the loop and the hook are the code under test, not the mixers) and
checks three things: the captured value equals the block's own output, all 52 blocks are
offered regardless of kind, and an absent sink changes nothing.
"""

from __future__ import annotations

import torch
from freetoken.core import Context, get_global_ctx, set_global_ctx
from freetoken.hidden_states import HiddenStateCapture, HiddenStateSink, HiddenStateSpec
from freetoken.models.nemotron_h.model import NemotronHBackbone
from types import SimpleNamespace

#: The Nemotron-3.5-Lightning geometry: 52 blocks -- 23 mamba, 23 MoE, 6 attention --
#: all one residual stream. The exact interleave does not matter here (the hook is
#: kind-blind by construction); the counts and the depth do.
_ATTENTION_IDS = (5, 13, 21, 29, 37, 45)


def _lightning_kinds() -> list[str]:
    kinds: list[str] = []
    mixed = 0
    for layer_id in range(52):
        if layer_id in _ATTENTION_IDS:
            kinds.append("attention")
            continue
        kinds.append("mamba" if mixed % 2 == 0 else "moe")
        mixed += 1
    return kinds


LIGHTNING_KINDS = _lightning_kinds()
HIDDEN = 6


def _ctx() -> Context:
    try:
        return get_global_ctx()
    except AssertionError:
        set_global_ctx(Context(page_size=1))
        return get_global_ctx()


class _Block:
    """``x + mixer(norm(x))`` with a mixer that returns 1.

    The residual after block ``i`` is therefore ``i + 1`` and before it ``i`` -- so an
    off-by-one hook (capturing the block's input instead of its output) is caught, and
    every value stays an integer bf16 represents exactly.
    """

    def __init__(self, layer_id: int, kind: str):
        self.layer_id = layer_id
        self.kind = kind

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1.0


class _Backbone:
    """A ``NemotronHBackbone``-shaped stand-in for the real forward to run against."""

    def __init__(self, kinds):
        self.embeddings = SimpleNamespace(
            forward=lambda ids: torch.zeros(len(ids), HIDDEN, dtype=torch.float32)
        )
        self.layers = SimpleNamespace(
            op_list=[_Block(i, kind) for i, kind in enumerate(kinds)]
        )
        # norm_f must NOT be part of what is captured.
        self.norm_f = SimpleNamespace(forward=lambda x: x * 1000.0)


def _run(kinds, sink):
    ctx = _ctx()
    backbone = _Backbone(kinds)
    ctx.hidden_state_sink = sink
    try:
        return NemotronHBackbone.forward(backbone, torch.tensor([1, 2, 3]))
    finally:
        ctx.hidden_state_sink = None


def test_capture_is_the_post_block_residual_not_the_final_norm():
    capture = HiddenStateCapture(
        HiddenStateSpec(directory="/tmp", layer_ids=list(range(len(LIGHTNING_KINDS)))),
        hidden_size=HIDDEN,
    )
    capture.begin_chunk(torch.tensor([1, 2, 3], dtype=torch.int32))
    out = _run(LIGHTNING_KINDS, HiddenStateSink([(capture, 0, 3)]))

    hidden, tokens = capture.finish()
    assert hidden.shape == (3, len(LIGHTNING_KINDS), HIDDEN)
    assert torch.equal(tokens, torch.tensor([1, 2, 3], dtype=torch.int64))
    # Block i leaves i+1 behind; norm_f's x1000 must be nowhere in the artifact.
    for layer_id in range(len(LIGHTNING_KINDS)):
        assert torch.allclose(
            hidden[:, layer_id].float(), torch.full((3, HIDDEN), float(layer_id + 1))
        )
    assert torch.allclose(out, torch.full((3, HIDDEN), 52_000.0))


def test_every_block_kind_is_offered_to_the_sink():
    seen: list[int] = []
    sink = SimpleNamespace(capture=lambda layer_id, hidden: seen.append(layer_id))
    _run(LIGHTNING_KINDS, sink)
    assert seen == list(range(52))
    assert LIGHTNING_KINDS.count("mamba") == 23
    assert LIGHTNING_KINDS.count("moe") == 23
    assert LIGHTNING_KINDS.count("attention") == 6


def test_a_layer_subset_drops_the_rest():
    capture = HiddenStateCapture(
        HiddenStateSpec(directory="/tmp", layer_ids=[0, 1, 2]), hidden_size=HIDDEN
    )
    capture.begin_chunk(torch.tensor([1, 2, 3], dtype=torch.int32))
    _run(LIGHTNING_KINDS, HiddenStateSink([(capture, 0, 3)]))
    hidden, _ = capture.finish()
    assert hidden.shape == (3, 3, HIDDEN)
    assert torch.allclose(hidden[:, 2].float(), torch.full((3, HIDDEN), 3.0))


def test_no_sink_leaves_the_forward_unchanged():
    with_sink = _run(LIGHTNING_KINDS, None)
    ctx = _ctx()
    assert ctx.hidden_state_sink is None
    assert torch.allclose(with_sink, torch.full((3, HIDDEN), 52_000.0))


def test_sink_slices_only_its_own_rows_out_of_a_mixed_batch():
    """Two requests share one prefill forward; each capture owns a row range."""
    first = HiddenStateCapture(
        HiddenStateSpec(directory="/tmp", layer_ids=[0]), hidden_size=HIDDEN
    )
    second = HiddenStateCapture(
        HiddenStateSpec(directory="/tmp", layer_ids=[0]), hidden_size=HIDDEN
    )
    first.begin_chunk(torch.tensor([10], dtype=torch.int32))
    second.begin_chunk(torch.tensor([20, 21], dtype=torch.int32))
    sink = HiddenStateSink([(first, 0, 1), (second, 1, 3)])

    hidden = torch.arange(3 * HIDDEN, dtype=torch.float32).reshape(3, HIDDEN)
    sink.capture(0, hidden)

    assert torch.equal(first.finish()[0][:, 0], hidden[:1].to(torch.bfloat16))
    assert torch.equal(second.finish()[0][:, 0], hidden[1:].to(torch.bfloat16))
