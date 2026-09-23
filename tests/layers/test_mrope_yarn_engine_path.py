"""--rope-yarn-factor reaches Ornith's (qwen3_5_moe) rotary end to end, and the applied q/k
rotation equals HF transformers' for the same config.

Path under test: EngineConfig(rope_yarn_factor=4.0, rope_yarn_original_context=262144)
.model_config -> FullAttentionGroupConfig/ModelConfig.rotary_config -> Qwen3_5Attention's
get_rope call -> MRotaryEmbedding (vision active, 3-axis positions) or RotaryEmbedding
(text-only, 1-D positions). The reference is HF's Qwen3_5MoeTextRotaryEmbedding +
apply_rotary_pos_emb built from a yarn rope_parameters dict. CPU only (torch fallback of the
mrope apply; the triton kernel gathers the same _cos_sin_cache rows)."""
from __future__ import annotations

import pytest
import torch

transformers = pytest.importorskip("transformers")

HEAD_DIM = 256
PARTIAL = 0.25
ROTARY_DIM = int(HEAD_DIM * PARTIAL)
BASE = 1e7
MROPE_SECTION = [11, 11, 10]
ORIG = 262144
FACTOR = 4.0
POSITIONS = [0, 1000, 300000, 900000]


def _hf_text_config(rope_type: str):
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig

    rope = {
        "rope_type": rope_type,
        "rope_theta": BASE,
        "partial_rotary_factor": PARTIAL,
        "mrope_section": list(MROPE_SECTION),
        "mrope_interleaved": True,
    }
    if rope_type == "yarn":
        rope.update(factor=FACTOR, original_max_position_embeddings=ORIG)
    cfg = Qwen3_5MoeConfig(
        text_config=dict(
            head_dim=HEAD_DIM,
            partial_rotary_factor=PARTIAL,
            num_hidden_layers=4,  # layer_types -> 3 linear_attention + 1 full_attention
            hidden_size=256,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=ORIG,
            rope_parameters=rope,
        )
    )
    cfg.architectures = ["Qwen3_5MoeForConditionalGeneration"]
    return cfg


def _engine_model_config(monkeypatch, *, vision: bool):
    import freetoken.engine.config as engine_config
    from freetoken.distributed import DistributedInfo

    cfg = _hf_text_config("default")  # the checkpoint's own rope: no scaling
    if not vision:
        cfg.vision_config = None
    monkeypatch.setattr(engine_config, "cached_load_hf_config", lambda path: cfg)
    engine = engine_config.EngineConfig(
        model_path="/nonexistent-ornith-checkpoint",
        tp_info=DistributedInfo(0, 1),
        dtype=torch.float32,
        rope_yarn_factor=FACTOR,
        rope_yarn_original_context=ORIG,
    )
    return engine.model_config


def _ornith_attention_rotary(model_config):
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.layers.rotary import get_rope
    from freetoken.models.config import FullAttentionGroupConfig
    from freetoken.models.qwen3_5_moe.attention import Qwen3_5Attention

    if try_get_tp_info() is None:
        set_tp_info(0, 1)
    get_rope.cache_clear()
    full = next(g for g in model_config.attention_groups if isinstance(g, FullAttentionGroupConfig))
    return Qwen3_5Attention(model_config, full.layer_ids[0], prefix="t").rotary


def _hf_rotate(q, k, pos3):
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeTextRotaryEmbedding,
        apply_rotary_pos_emb,
    )

    emb = Qwen3_5MoeTextRotaryEmbedding(_hf_text_config("yarn").text_config)
    cos, sin = emb.forward(torch.zeros(1, dtype=torch.float32), pos3.unsqueeze(1))  # [1, n, rot]
    # HF layout [bs, heads, n, head_dim]
    qh, kh = q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0)
    qo, ko = apply_rotary_pos_emb(qh, kh, cos, sin)
    return qo[0].transpose(0, 1), ko[0].transpose(0, 1), cos[0], sin[0]


def _qk(n: int, heads_q: int = 2, heads_k: int = 1):
    g = torch.Generator().manual_seed(0)
    q = torch.randn(n, heads_q, HEAD_DIM, generator=g)
    k = torch.randn(n, heads_k, HEAD_DIM, generator=g)
    return q, k


def test_override_keeps_mrope_and_sets_yarn(monkeypatch):
    mc = _engine_model_config(monkeypatch, vision=True)
    from freetoken.models.config import FullAttentionGroupConfig

    full = next(g for g in mc.attention_groups if isinstance(g, FullAttentionGroupConfig))
    assert full.rotary_config is mc.rotary_config
    rc = mc.rotary_config
    assert rc.max_position == int(ORIG * FACTOR) == 1048576
    assert rc.scaling == {
        "rope_type": "yarn", "factor": FACTOR, "original_max_position_embeddings": ORIG,
    }
    assert list(rc.mrope_section) == MROPE_SECTION and rc.mrope_layout == "interleaved"
    # the linear-attention (GDN) group carries no rotary at all
    others = [g for g in mc.attention_groups if not isinstance(g, FullAttentionGroupConfig)]
    assert others and all(getattr(g, "rotary_config", None) is None for g in others)


@pytest.mark.parametrize(
    "pos3",
    [
        torch.tensor([POSITIONS] * 3),
        torch.tensor([POSITIONS, [0, 1200, 300500, 899000], [0, 900, 299000, 901500]]),
    ],
    ids=["text-equal-axes", "distinct-axes"],
)
def test_mrope_yarn_applied_qk_matches_hf(monkeypatch, pos3):
    from freetoken.layers.rotary import MRotaryEmbedding, _mrope_torch, mrope_cos_sin_rows

    rotary = _ornith_attention_rotary(_engine_model_config(monkeypatch, vision=True))
    assert isinstance(rotary, MRotaryEmbedding)

    q, k = _qk(pos3.shape[1])
    hf_q, hf_k, hf_cos, hf_sin = _hf_rotate(q, k, pos3)

    rows = mrope_cos_sin_rows(rotary._cos_sin_cache, pos3, rotary._section_table)
    half = ROTARY_DIM // 2
    cos_diff = (rows[:, :half] - hf_cos[:, :half]).abs().max().item()
    sin_diff = (rows[:, half:] - hf_sin[:, :half]).abs().max().item()

    fq, fk = q.reshape(q.shape[0], -1).clone(), k.reshape(k.shape[0], -1).clone()
    _mrope_torch(pos3, fq, fk, HEAD_DIM, rotary._cos_sin_cache, rotary._section_table)
    q_diff = (fq.view_as(q) - hf_q).abs().max().item()
    k_diff = (fk.view_as(k) - hf_k).abs().max().item()
    print(f"max|diff| cos={cos_diff:.3g} sin={sin_diff:.3g} q={q_diff:.3g} k={k_diff:.3g}")
    assert max(cos_diff, sin_diff) <= 1e-5
    assert max(q_diff, k_diff) <= 1e-5
    # the triton module's pure-torch fallback is the same rotation (both used to alias fp32
    # inputs: the second half read the already-rotated first half)
    from freetoken.kernel.triton.rope import apply_mrope_torch_fallback

    gq, gk = q.reshape(q.shape[0], -1).clone(), k.reshape(k.shape[0], -1).clone()
    apply_mrope_torch_fallback(pos3, gq, gk, HEAD_DIM, rotary._cos_sin_cache, rotary._section_table)
    assert torch.equal(gq, fq) and torch.equal(gk, fk)
    # bf16 activations (what the server feeds): rounding-level agreement with HF in fp32
    bq, bk = q.reshape(q.shape[0], -1).bfloat16(), k.reshape(k.shape[0], -1).bfloat16()
    _mrope_torch(pos3, bq, bk, HEAD_DIM, rotary._cos_sin_cache, rotary._section_table)
    assert (bq.float().view_as(q) - hf_q).abs().max().item() <= 5e-2
    assert (bk.float().view_as(k) - hf_k).abs().max().item() <= 5e-2
    # mscale really applied: yarn attention_factor 0.1*ln(4)+1 at position 0
    assert rows[0, 0].item() == pytest.approx(0.1 * torch.log(torch.tensor(FACTOR)).item() + 1.0)


def test_text_only_yarn_is_1d_and_matches_hf(monkeypatch):
    """Text-only serving (no vision tower) keeps the 1-D rope; it must carry the same yarn."""
    from freetoken.layers.rotary import MRotaryEmbedding, RotaryEmbedding

    rotary = _ornith_attention_rotary(_engine_model_config(monkeypatch, vision=False))
    assert type(rotary) is RotaryEmbedding and not isinstance(rotary, MRotaryEmbedding)
    pos = torch.tensor(POSITIONS)
    _, _, hf_cos, hf_sin = _hf_rotate(*_qk(len(POSITIONS)), torch.stack([pos] * 3))
    half = ROTARY_DIM // 2
    rows = rotary._cos_sin_cache[pos]
    assert (rows[:, :half] - hf_cos[:, :half]).abs().max().item() <= 1e-5
    assert (rows[:, half:] - hf_sin[:, :half]).abs().max().item() <= 1e-5


def test_mrope_and_1d_share_the_scaled_table():
    """Generic invariant (any mrope model, any scaling): mrope only selects axes, so its
    cos/sin cache is bit-identical to the 1-D cache of the same rope_scaling."""
    from freetoken.layers.rotary import get_rope

    scalings = [
        None,
        (("rope_type", "yarn"), ("factor", FACTOR), ("original_max_position_embeddings", ORIG)),
        (("rope_type", "llama3"), ("factor", 8.0), ("low_freq_factor", 1.0),
         ("high_freq_factor", 4.0), ("original_max_position_embeddings", 8192)),
    ]
    for scaling in scalings:
        get_rope.cache_clear()
        one_d = get_rope(HEAD_DIM, ROTARY_DIM, 4096, BASE, scaling)
        m = get_rope(HEAD_DIM, ROTARY_DIM, 4096, BASE, scaling,
                     mrope_section=tuple(MROPE_SECTION), mrope_layout="interleaved")
        assert torch.equal(one_d._cos_sin_cache, m._cos_sin_cache), scaling
