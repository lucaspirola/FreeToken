"""MRotaryEmbedding + yarn against HF's Qwen3.5-MoE text rotary (the real ROPE_INIT_FUNCTIONS
path, not a duct-typed shim): same Ornith-shaped config (head_dim 256, partial_rotary_factor
0.25 -> rotary_dim 64, rope_theta 1e7, mrope_section [11,11,10], interleaved), yarn factor 4.0
over an original context of 262144. Runs on CPU; no GPU/model server touched."""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest
import torch

transformers = pytest.importorskip("transformers")

HEAD_DIM = 256
ROTARY_DIM = 64  # head_dim * partial_rotary_factor(0.25)
BASE = 1e7
MROPE_SECTION = (11, 11, 10)
ORIG_MAX_POS = 262144
YARN_FACTOR = 4.0
MAX_POSITION = int(round(ORIG_MAX_POS * YARN_FACTOR))


def _hf_cos_sin(pos3: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """HF Qwen3_5MoeTextRotaryEmbedding cos/sin for positions [3, n] (already axis-selected
    and mrope-interleaved). Returns the un-duplicated half [n, rotary_dim//2] each, since HF's
    emb is cat(freqs, freqs) -- both halves carry the same values."""
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextRotaryEmbedding

    cfg = Qwen3_5MoeTextConfig(
        head_dim=HEAD_DIM,
        partial_rotary_factor=0.25,
        rope_theta=BASE,
        max_position_embeddings=ORIG_MAX_POS,
        num_hidden_layers=2,
        rope_parameters={
            "rope_type": "yarn",
            "factor": YARN_FACTOR,
            "original_max_position_embeddings": ORIG_MAX_POS,
            "mrope_section": list(MROPE_SECTION),
            "mrope_interleaved": True,
        },
    )
    emb = Qwen3_5MoeTextRotaryEmbedding(cfg)
    x = torch.zeros(1, 1)
    position_ids = pos3.unsqueeze(1).float()  # [3, 1, n]
    cos, sin = emb.forward(x, position_ids)  # [1, n, rotary_dim]
    half = ROTARY_DIM // 2
    return cos[0, :, :half], sin[0, :, :half]


def _freetoken_cos_sin(pos3: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from freetoken.layers.rotary import get_rope, mrope_cos_sin_rows

    get_rope.cache_clear()
    rope = get_rope(
        head_dim=HEAD_DIM,
        rotary_dim=ROTARY_DIM,
        max_position=MAX_POSITION,
        base=BASE,
        rope_scaling=(
            ("rope_type", "yarn"),
            ("factor", YARN_FACTOR),
            ("original_max_position_embeddings", ORIG_MAX_POS),
        ),
        mrope_section=MROPE_SECTION,
        mrope_layout="interleaved",
    )
    rows = mrope_cos_sin_rows(rope._cos_sin_cache, pos3.long(), rope._section_table)
    half = ROTARY_DIM // 2
    return rows[:, :half], rows[:, half:]


@pytest.mark.parametrize(
    "pos3",
    [
        # text positions: all 3 axes equal (the degenerate case every model actually hits
        # outside true multimodal spans)
        torch.tensor([[0, 1000, 300000, 900000]] * 3, dtype=torch.long),
        # distinct t/h/w axes, still spanning small/medium/large positions
        torch.tensor(
            [
                [0, 1000, 300000, 900000],
                [0, 1200, 300500, 899000],
                [0, 900, 299000, 901500],
            ],
            dtype=torch.long,
        ),
    ],
    ids=["text-positions-equal-axes", "distinct-axes"],
)
def test_mrope_yarn_matches_hf_reference(pos3):
    hf_cos, hf_sin = _hf_cos_sin(pos3)
    ft_cos, ft_sin = _freetoken_cos_sin(pos3)

    max_cos_diff = (hf_cos - ft_cos).abs().max().item()
    max_sin_diff = (hf_sin - ft_sin).abs().max().item()
    assert max_cos_diff < 2e-5, max_cos_diff
    assert max_sin_diff < 2e-5, max_sin_diff
    torch.testing.assert_close(ft_cos, hf_cos, rtol=0, atol=2e-5)
    torch.testing.assert_close(ft_sin, hf_sin, rtol=0, atol=2e-5)


def test_no_yarn_mrope_cache_unchanged_when_factor_is_none():
    """factor=None (today's default path) must stay byte-identical: no scaling kwargs reach
    MRotaryEmbedding when rope_scaling is None."""
    from freetoken.layers.rotary import get_rope

    get_rope.cache_clear()
    rope_no_scaling = get_rope(
        head_dim=HEAD_DIM, rotary_dim=ROTARY_DIM, max_position=4096, base=BASE,
        rope_scaling=None, mrope_section=MROPE_SECTION, mrope_layout="interleaved",
    )
    get_rope.cache_clear()
    rope_default = get_rope(
        head_dim=HEAD_DIM, rotary_dim=ROTARY_DIM, max_position=4096, base=BASE,
        rope_scaling=(("rope_type", "default"),), mrope_section=MROPE_SECTION,
        mrope_layout="interleaved",
    )
    torch.testing.assert_close(
        rope_no_scaling._cos_sin_cache, rope_default._cos_sin_cache, rtol=0, atol=0
    )
    # plain inv_freq, no post_process/attention_factor: cos at position 0 is exactly 1
    half = ROTARY_DIM // 2
    torch.testing.assert_close(
        rope_no_scaling._cos_sin_cache[0, :half], torch.ones(half), rtol=0, atol=0
    )
