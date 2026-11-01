"""lfm2_moe (LFM2.5-24B-A2B): config parsing, the router, and the EXL3 checkpoint reader.

The synthetic checkpoint has the exllamav3 export's key names, dtypes and ``tensor_storage``
table at a tiny size: 4 layers (conv, conv, attention, conv), 2 dense then MoE.
"""

from __future__ import annotations

import json
import math

import pytest
import torch
from safetensors.torch import save_file

from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.layers.quantization import set_quant_config
from freetoken.models.lfm2_moe.config import parse_config
from freetoken.models.lfm2_moe.weight import iter_weights
from freetoken.models.register import checkpoint_quant_config, get_model_spec
from freetoken.utils import cached_load_hf_config

H, V, NH, NKV, HD = 128, 256, 2, 1, 64
INTER, MOE_INTER, E, TOPK = 256, 128, 4, 2
KINDS = ["conv", "conv", "full_attention", "conv"]
MUL1 = 0x83DCD12D

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.fixture(scope="module", autouse=True)
def _tp_info():
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    yield
    set_quant_config(None)


def _config_dict(**over) -> dict:
    cfg = {
        "architectures": ["Lfm2MoeForCausalLM"], "model_type": "lfm2_moe", "hidden_size": H,
        "intermediate_size": INTER, "moe_intermediate_size": MOE_INTER, "num_attention_heads": NH,
        "num_key_value_heads": NKV, "num_hidden_layers": len(KINDS), "layer_types": KINDS,
        "num_dense_layers": 2, "num_experts": E, "num_experts_per_tok": TOPK, "vocab_size": V,
        "max_position_embeddings": 131072, "norm_eps": 1e-5, "conv_L_cache": 3, "conv_bias": False,
        "norm_topk_prob": True, "routed_scaling_factor": 1.0, "use_expert_bias": True,
        "tie_word_embeddings": True, "rope_parameters": {"rope_theta": 1e7, "rope_type": "default"},
    }
    cfg.update(over)
    return cfg


def _write_exl3(folder, *, head_bits: int = 6) -> dict[str, torch.Tensor]:
    """An EXL3 export as exllamav3 writes it: every Linear as trellis/suh/svh/mul1, the rest bf16,
    routers fp16. ``head_bits=16`` leaves out ``lm_head`` (the export then ties the embeddings)."""
    g = torch.Generator().manual_seed(3)
    tensors: dict[str, torch.Tensor] = {}
    storage: dict[str, dict] = {}

    def exl3(name, k, n, bits):
        parts = {
            "trellis": torch.randint(-(1 << 15), 1 << 15, (k // 16, n // 16, 16 * bits), generator=g, dtype=torch.int32).to(torch.int16),
            "suh": torch.randn(k, generator=g).half(), "svh": (torch.randn(n, generator=g) / math.sqrt(k)).half(),
            "mul1": torch.tensor(MUL1 - (1 << 32), dtype=torch.int32),
        }
        entry = {"stored_tensors": {}, "quant_format": "exl3", "bits_per_weight": bits}
        for kind, v in parts.items():
            tensors[f"{name}.{kind}"] = v
            entry["stored_tensors"][f"{name}.{kind}"] = {"shape": list(v.shape), "dtype": str(v.dtype)}
        storage[name] = entry

    def plain(name, value):
        tensors[name] = value
        module = name.rsplit(".", 1)[0]
        storage.setdefault(module, {"stored_tensors": {}})["stored_tensors"][name] = {"shape": list(value.shape)}

    bf16 = lambda *s: torch.randn(*s, generator=g).to(torch.bfloat16)  # noqa: E731
    plain("model.embed_tokens.weight", bf16(V, H))
    plain("model.embedding_norm.weight", bf16(H))
    for layer, kind in enumerate(KINDS):
        p = f"model.layers.{layer}"
        plain(f"{p}.operator_norm.weight", bf16(H))
        plain(f"{p}.ffn_norm.weight", bf16(H))
        if kind == "conv":
            exl3(f"{p}.conv.in_proj", H, 3 * H, 7)
            exl3(f"{p}.conv.out_proj", H, H, 7)
            plain(f"{p}.conv.conv.weight", bf16(H, 1, 3))
        else:
            for proj, n in (("q_proj", NH * HD), ("k_proj", NKV * HD), ("v_proj", NKV * HD)):
                exl3(f"{p}.self_attn.{proj}", H, n, 7)
            exl3(f"{p}.self_attn.out_proj", NH * HD, H, 7)
            plain(f"{p}.self_attn.q_layernorm.weight", bf16(HD))
            plain(f"{p}.self_attn.k_layernorm.weight", bf16(HD))
        ff = f"{p}.feed_forward"
        if layer < 2:
            exl3(f"{ff}.w1", H, INTER, 6)
            exl3(f"{ff}.w3", H, INTER, 6)
            exl3(f"{ff}.w2", INTER, H, 6)
        else:
            plain(f"{ff}.gate.weight", torch.randn(E, H, generator=g).half())
            tensors[f"{ff}.expert_bias"] = torch.randn(E, generator=g)
            for e in range(E):
                exl3(f"{ff}.experts.{e}.w1", H, MOE_INTER, 5)
                exl3(f"{ff}.experts.{e}.w3", H, MOE_INTER, 5)
                exl3(f"{ff}.experts.{e}.w2", MOE_INTER, H, 5)
    if head_bits != 16:
        exl3("lm_head", H, V, head_bits)

    names = sorted(tensors)
    weight_map = {}
    for i in range(2):
        shard = f"model-{i:05d}-of-00002.safetensors"
        save_file({n: tensors[n].contiguous() for n in names[i::2]}, str(folder / shard))
        weight_map.update({n: shard for n in names[i::2]})
    (folder / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    (folder / "quantization_config.json").write_text(json.dumps({"tensor_storage": storage}))
    quant = {"quant_method": "exl3", "version": "1.6.6", "bits": 5.0, "head_bits": head_bits, "codebook": "mul1"}
    (folder / "config.json").write_text(json.dumps(_config_dict(quantization_config=quant)))
    return tensors


def _install(folder: str):
    hf = cached_load_hf_config(folder)
    quant = checkpoint_quant_config(folder, hf, get_model_spec(hf.architectures[0]))
    set_quant_config(quant)
    return hf, quant


def _meta_state_dict(folder: str) -> dict[str, torch.Tensor]:
    """State dict of the model the engine builds for ``folder`` (experts offloaded), on meta."""
    from freetoken.engine.config import EngineConfig
    from freetoken.engine.engine import _decode_target
    from freetoken.layers import rotary
    from freetoken.models import create_model
    from freetoken.utils.torch_utils import torch_dtype

    config = EngineConfig(model_path=folder, tp_info=try_get_tp_info(), dtype=torch.bfloat16, moe_strategy="offload")
    object.__setattr__(config.model_config, "moe_strategy", "offload")
    object.__setattr__(config.model_config, "decode_target", _decode_target(config))
    saved = rotary._ROPE_DEVICE
    rotary.set_rope_device(torch.device("cpu"))
    rotary.get_rope.cache_clear()
    try:
        with torch.device("meta"), torch_dtype(torch.bfloat16):
            return create_model(config.model_config).state_dict()
    finally:
        rotary.set_rope_device(saved)
        rotary.get_rope.cache_clear()


@pytest.fixture(scope="module")
def exl3_ckpt(tmp_path_factory):
    folder = tmp_path_factory.mktemp("lfm2_exl3")
    return str(folder), _write_exl3(folder)


# --------------------------------------------------------------------------- config


def test_parse_config_matches_the_released_geometry():
    from transformers import Lfm2MoeConfig

    kinds = ["full_attention" if i % 4 == 2 else "conv" for i in range(40)]
    hf = Lfm2MoeConfig(**{k: v for k, v in _config_dict(
        hidden_size=2048, intermediate_size=11776, moe_intermediate_size=1536, num_attention_heads=32,
        num_key_value_heads=8, num_hidden_layers=40, layer_types=kinds, num_experts=64,
        num_experts_per_tok=4, vocab_size=128000,
    ).items() if k not in ("architectures", "model_type")})
    hf.architectures = ["Lfm2MoeForCausalLM"]
    mc = parse_config(hf)
    conv, full = mc.linear_attention_group(), mc.attention_groups[1]
    assert conv.state_layout == "short_conv" and len(conv.layer_ids) == 30
    assert conv.num_value_heads * conv.value_head_dim == 2048 and conv.conv_kernel_dim == 3
    assert full.layer_ids == tuple(range(2, 40, 4)) and full.num_kv_heads == 8 and full.head_dim == 64
    assert mc.rotary_config.base == 1e7 and mc.rotary_config.rotary_dim == 64
    assert mc.first_k_dense_replace == 2 and mc.num_moe_layers == 38
    assert mc.tie_word_embeddings and mc.expert_quant == "none"
    assert mc.rms_norm_eps == 1e-5 and mc.intermediate_size == 11776


def test_layer_types_fall_back_to_every_fourth_layer():
    from types import SimpleNamespace

    from freetoken.models.lfm2_moe.config import layer_types

    assert layer_types(SimpleNamespace(num_hidden_layers=8)) == ("conv", "conv", "conv", "full_attention") * 2
    with pytest.raises(ValueError, match="unknown layer types"):
        layer_types(SimpleNamespace(num_hidden_layers=2, layer_types=["conv", "mamba"]))


@pytest.mark.parametrize("head_bits, tied", [(6, False), (16, True)])
def test_a_stored_lm_head_unties_the_embeddings(tmp_path, head_bits, tied):
    _write_exl3(tmp_path, head_bits=head_bits)
    hf, _ = _install(str(tmp_path))
    mc = parse_config(hf)
    assert mc.tie_word_embeddings is tied and mc.expert_quant == "exl3"


# --------------------------------------------------------------------------- router


def test_router_matches_hf():
    from transformers import Lfm2MoeConfig
    from transformers.models.lfm2_moe.modeling_lfm2_moe import Lfm2MoeTopKRouter

    from freetoken.models.lfm2_moe.moe import Lfm2MoeSparseBlock

    torch.manual_seed(0)
    hf = Lfm2MoeTopKRouter(Lfm2MoeConfig(hidden_size=H, num_experts=16, num_experts_per_tok=4, routed_scaling_factor=1.5))
    hf.weight.data.normal_()
    bias = torch.randn(16) * 0.5
    block = object.__new__(Lfm2MoeSparseBlock)
    block.top_k, block.norm_topk_prob, block.routed_scaling_factor = 4, True, 1.5
    block.gate = type("G", (), {"weight": hf.weight.data})()
    block.expert_bias = bias
    x = torch.randn(33, H)
    _, ref_w, ref_ids = hf(x, bias)
    w, ids = block.route(x)
    assert torch.equal(ids.long().sort(-1).values, ref_ids.sort(-1).values)
    order_ref, order = ref_ids.argsort(-1), ids.long().argsort(-1)
    torch.testing.assert_close(w.gather(-1, order), ref_w.gather(-1, order_ref))


# --------------------------------------------------------------------------- the EXL3 reader


def test_emitted_keys_are_the_model_state_dict(exl3_ckpt):
    folder, _raw = exl3_ckpt
    _install(folder)
    loaded = {n: t for n, t in iter_weights(folder, torch.device("cpu"), include_moe_experts=False, include_non_moe=True)}
    state = _meta_state_dict(folder)
    assert set(loaded) == set(state)
    for key, tensor in loaded.items():
        assert tensor.shape == state[key].shape, key
    assert not any(".experts." in k for k in loaded)


def test_fused_parts_keep_their_own_input_signs(exl3_ckpt):
    folder, raw = exl3_ckpt
    _install(folder)
    loaded = dict(iter_weights(folder, torch.device("cpu"), include_moe_experts=False, include_non_moe=True))
    qkv = "model.layers.2.self_attn.qkv_proj"
    for i, proj in enumerate(("q_proj", "k_proj", "v_proj")):
        assert torch.equal(loaded[f"{qkv}.suh"][i], raw[f"model.layers.2.self_attn.{proj}.suh"])
    w13 = "model.layers.0.feed_forward.w13"
    assert torch.equal(loaded[f"{w13}.svh"], torch.cat([raw["model.layers.0.feed_forward.w1.svh"], raw["model.layers.0.feed_forward.w3.svh"]]))
    assert torch.equal(loaded["model.layers.2.self_attn.o_proj.trellis"].reshape(-1), raw["model.layers.2.self_attn.out_proj.trellis"].reshape(-1))
    assert torch.equal(loaded["model.layers.2.self_attn.q_norm.weight"], raw["model.layers.2.self_attn.q_layernorm.weight"])


def test_expert_requests_are_refused(exl3_ckpt):
    folder, _raw = exl3_ckpt
    _install(folder)
    with pytest.raises(NotImplementedError, match="offload cache"):
        list(iter_weights(folder, torch.device("cpu"), include_moe_experts=True, include_non_moe=False))


def test_exl3_expert_banks_read_w1_w3_w2_after_the_dense_layers(exl3_ckpt):
    from freetoken.models.exl3_banks import expert_facts_for, iter_exl3_expert_pieces

    folder, raw = exl3_ckpt
    hf, _ = _install(folder)
    mc = parse_config(hf)
    assert expert_facts_for(mc) == (5, "mul1")
    pieces = list(iter_exl3_expert_pieces(folder, mc, batch=E))
    assert [(layer, e0, e1) for layer, e0, e1, _ in pieces] == [(0, 0, E), (1, 0, E)]
    for bank_layer, _e0, _e1, piece in pieces:
        layer = bank_layer + 2
        for e in range(E):
            for role, proj in (("gate", "w1"), ("up", "w3"), ("down", "w2")):
                for kind in ("trellis", "suh", "svh"):
                    want = raw[f"model.layers.{layer}.feed_forward.experts.{e}.{proj}.{kind}"]
                    assert torch.equal(piece[f"{role}_{kind}"][e], want)


def test_registry_entry():
    spec = get_model_spec("Lfm2MoeForCausalLM")
    assert spec.module == "freetoken.models.lfm2_moe"
    assert dict(spec.packed_modules_mapping)["w13"] == ("w1", "w3")


# --------------------------------------------------------------------------- the short conv op

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
CONV_H, CONV_K = 256, 3


def _conv_setup(num_slots=8):
    import freetoken.core as core
    from freetoken.core import Context, set_global_ctx
    from freetoken.kvcache.linear_state_pool import LinearStatePool
    from freetoken.models.config import LinearGatedDeltaGroupConfig
    from freetoken.models.lfm2_moe.short_conv import Lfm2ShortConv

    group = LinearGatedDeltaGroupConfig(
        name="conv", layer_ids=(0,), num_key_heads=4, num_value_heads=4, key_head_dim=0,
        value_head_dim=CONV_H // 4, conv_kernel_dim=CONV_K, output_gate="none", state_layout="short_conv",
    )
    dev = torch.device("cuda")
    pool = LinearStatePool(group, num_slots, torch.float32, dev, tp_size=1)
    core._GLOBAL_CTX = None  # test-only: the op reads the pool and the batch off the ctx
    ctx = Context(page_size=64, linear_state_pool=pool)
    set_global_ctx(ctx)
    g = torch.Generator().manual_seed(5)
    with torch.device(dev):
        op = Lfm2ShortConv(CONV_H, CONV_K, 0)
    weights = {
        "in": (torch.randn(3 * CONV_H, CONV_H, generator=g) / CONV_H**0.5).to(dev),
        "conv": torch.randn(CONV_H, 1, CONV_K, generator=g).to(dev),
        "out": (torch.randn(CONV_H, CONV_H, generator=g) / CONV_H**0.5).to(dev),
    }
    op.in_proj.weight, op.conv.weight, op.out_proj.weight = weights["in"], weights["conv"], weights["out"]
    return ctx, pool, op, weights


def _conv_req(slot, cached_len, extend_len, *, ping_pong=None):
    from types import SimpleNamespace

    return SimpleNamespace(
        table_idx=slot, linear_slot_idx=slot, cached_len=cached_len, extend_len=extend_len,
        mamba_ping_pong=ping_pong, mamba_next_track_idx=0, mamba_last_track_seqlen=None,
    )


def _conv_run(ctx, op, reqs, x, *, decode=False):
    from types import SimpleNamespace

    from freetoken.attention.linear import build_fla_metadata

    batch = SimpleNamespace(
        padded_reqs=reqs, reqs=reqs, is_decode=decode, is_prefill=not decode, fla_metadata=None,
        linear_table_idx=torch.tensor([r.linear_slot_idx for r in reqs], dtype=torch.int32, device=x.device),
    )
    batch.fla_metadata = build_fla_metadata(batch, x.device)
    with ctx.forward_batch(batch):
        return op.forward(x)


def _ref(w, x, state=None):
    from freetoken.models.lfm2_moe.short_conv import short_conv_reference

    return short_conv_reference(x, w["in"], w["conv"], w["out"], state)


@requires_cuda
def test_short_conv_prefill_then_decode_matches_one_shot():
    ctx, pool, op, w = _conv_setup()
    lens, slots = [37, 5], [2, 6]
    xs = [torch.randn(n + 4, CONV_H, device="cuda") for n in lens]
    full = [_ref(w, x) for x in xs]

    # a fresh prefill, a continued prefill of 2 tokens, then 2 decode steps
    out = _conv_run(ctx, op, [_conv_req(s, 0, n) for s, n in zip(slots, lens)], torch.cat([x[:n] for x, n in zip(xs, lens)]))
    for i, part in enumerate(out.split(lens)):
        torch.testing.assert_close(part, full[i][0][: lens[i]], rtol=1e-4, atol=1e-4)
    out = _conv_run(ctx, op, [_conv_req(s, n, 2) for s, n in zip(slots, lens)], torch.cat([x[n : n + 2] for x, n in zip(xs, lens)]))
    for i, part in enumerate(out.split([2, 2])):
        torch.testing.assert_close(part, full[i][0][lens[i] : lens[i] + 2], rtol=1e-4, atol=1e-4)
    for step in (2, 3):
        out = _conv_run(ctx, op, [_conv_req(s, n + step, 1) for s, n in zip(slots, lens)],
                        torch.stack([x[n + step] for x, n in zip(xs, lens)]), decode=True)
        for i in range(2):
            torch.testing.assert_close(out[i], full[i][0][lens[i] + step], rtol=1e-4, atol=1e-4)
    for i, s in enumerate(slots):
        torch.testing.assert_close(pool.conv_states[0, s], full[i][1], rtol=1e-5, atol=1e-5)
    assert pool.recurrent_states.numel() == 0


@requires_cuda
def test_short_conv_track_snapshot_is_the_state_at_the_boundary():
    ctx, pool, op, w = _conv_setup()
    chunk = 64
    live, dst = 1, 5
    x = torch.randn(chunk + 6, CONV_H, device="cuda")
    req = _conv_req(live, 0, len(x), ping_pong=(dst, 6))
    _conv_run(ctx, op, [req], x)
    assert req.mamba_last_track_seqlen == chunk
    torch.testing.assert_close(pool.conv_states[0, dst], _ref(w, x[:chunk])[1], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(pool.conv_states[0, live], _ref(w, x)[1], rtol=1e-5, atol=1e-5)
