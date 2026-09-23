"""S8's model conformance check for a bare ``.gguf`` checkpoint.

Companion to ``test_expert_source_conformance.py`` (the NVFP4-safetensors path):
writes tiny, real qwen35moe GGUF files with gguf-py's ``GGUFWriter`` -- uniform and
mixed per-layer expert quant types, plus an MTP predictor block that must be counted
in the total but excluded from serving -- and exercises
``freetoken.models.check_experts.check_experts`` end to end (real file on disk, no
mocking of the GGUF reader or config parser), the same header-only, no-GPU contract
as the NVFP4 path.
"""

from __future__ import annotations

import numpy as np
import pytest

import gguf

import freetoken.distributed.info as di
from freetoken.models.check_experts import CheckFailed, GgufExpertConformanceReport, check_experts

# Tiny geometry: block size 32 for both Q8_0 and Q4_0, so H=I=32 is exactly one block
# per row -- no padding arithmetic to reason about when hand-checking byte counts.
H, I, E = 32, 32, 4
MAIN_LAYERS = 3


@pytest.fixture(autouse=True)
def _tp1():
    try:
        di.get_tp_info()
    except RuntimeError:
        di.set_tp_info(0, 1)


def _write_qwen35moe_gguf(
    path: str,
    *,
    expert_types: list[int],  # one ggml type per MAIN layer (gate/up/down all this type)
    mtp: bool,
    full_attention_interval: int = 3,
) -> None:
    """A real, minimal qwen35moe GGUF: enough metadata for ``parse_gguf_config`` and
    enough tensors (``token_embd.weight`` + each layer's three expert banks) for
    ``check_gguf_experts`` -- nothing else ``parse_gguf_config`` reads only from
    metadata, not tensors, is written.
    """
    total_blocks = MAIN_LAYERS + (1 if mtp else 0)
    w = gguf.GGUFWriter(path, "qwen35moe")
    w.add_block_count(total_blocks)
    if mtp:
        w.add_uint32("qwen35moe.nextn_predict_layers", 1)
    w.add_embedding_length(H)
    w.add_context_length(4096)
    w.add_head_count(4)
    w.add_head_count_kv(2)
    w.add_key_length(16)
    w.add_uint32("qwen35moe.full_attention_interval", full_attention_interval)
    w.add_rope_dimension_count(16)
    w.add_rope_freq_base(10000.0)
    w.add_layer_norm_rms_eps(1e-6)
    w.add_uint32("qwen35moe.ssm.state_size", 8)
    w.add_uint32("qwen35moe.ssm.inner_size", 16)
    w.add_uint32("qwen35moe.ssm.group_count", 2)
    w.add_uint32("qwen35moe.ssm.conv_kernel", 4)
    w.add_expert_count(E)
    w.add_expert_used_count(2)
    w.add_expert_feed_forward_length(I)
    w.add_expert_shared_feed_forward_length(I)

    rng = np.random.default_rng(0)

    def quant(name: str, rows: int, cols: int, ggml_type: int) -> None:
        data = rng.standard_normal((rows, cols)).astype(np.float32)
        w.add_tensor(name, gguf.quants.quantize(data, ggml_type), raw_dtype=ggml_type)

    quant("token_embd.weight", 8, H, gguf.GGMLQuantizationType.Q8_0)

    types_by_block = list(expert_types)
    if mtp:
        types_by_block = types_by_block + [types_by_block[-1]]  # MTP reuses the last type
    for layer, ggml_type in enumerate(types_by_block):
        p = f"blk.{layer}."
        for role, rows, cols in (
            ("ffn_gate_exps", I, H),
            ("ffn_up_exps", I, H),
            ("ffn_down_exps", H, I),
        ):
            data = rng.standard_normal((E, rows, cols)).astype(np.float32)
            quantized = gguf.quants.quantize(data.reshape(E * rows, cols), ggml_type)
            w.add_tensor(
                p + role + ".weight",
                quantized.reshape(E, rows, -1),
                raw_dtype=ggml_type,
            )

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def test_uniform_gguf_checkpoint_reports_one_size_class(tmp_path):
    path = str(tmp_path / "uniform.gguf")
    q8 = gguf.GGMLQuantizationType.Q8_0
    _write_qwen35moe_gguf(path, expert_types=[int(q8)] * MAIN_LAYERS, mtp=True)

    report = check_experts(path)
    assert isinstance(report, GgufExpertConformanceReport)
    assert report.hidden_size == H
    assert report.moe_intermediate_size == I
    assert report.num_experts == E
    assert report.num_served_layers == MAIN_LAYERS
    assert report.mtp_blocks == 1
    assert report.num_expert_blocks == MAIN_LAYERS + 1
    assert report.quant_types == ("Q8_0",)
    unique = tuple(dict.fromkeys(report.signatures))
    assert len(unique) == 1, report.signatures

    row_bytes = gguf.GGML_QUANT_SIZES[q8][1] * (H // gguf.GGML_QUANT_SIZES[q8][0])
    gu_bytes = 2 * I * row_bytes  # 64B-aligned; already a multiple of 64 here
    dn_bytes = H * row_bytes
    assert unique[0] == (gu_bytes, dn_bytes)

    # total bank bytes = real on-disk tensor bytes over EVERY expert block, MTP included
    per_block_bytes = E * (2 * I * row_bytes + H * row_bytes)
    assert report.total_expert_bank_bytes == per_block_bytes * (MAIN_LAYERS + 1)

    rendered = report.render()
    assert rendered.startswith("OK  ")
    assert "single arena" in rendered
    assert "1 MTP" in rendered
    assert "REFUSED: requires nvfp4" in rendered  # --expert-residency mirror line


def test_mixed_gguf_checkpoint_reports_per_class_arena(tmp_path):
    path = str(tmp_path / "mixed.gguf")
    q8, q4 = gguf.GGMLQuantizationType.Q8_0, gguf.GGMLQuantizationType.Q4_0
    _write_qwen35moe_gguf(path, expert_types=[int(q8), int(q4), int(q8)], mtp=False)

    report = check_experts(path)
    assert report.num_served_layers == MAIN_LAYERS
    assert report.mtp_blocks == 0
    assert report.num_expert_blocks == MAIN_LAYERS
    assert set(report.quant_types) == {"Q8_0", "Q4_0"}
    unique = tuple(dict.fromkeys(report.signatures))
    assert len(unique) == 2, report.signatures

    rendered = report.render()
    assert rendered.startswith("OK  ")
    assert "per-class arena, 2 classes" in rendered
    assert "S12b merge 1704620" in rendered
    assert "distinct size classes  2" in rendered


def test_no_expert_tensors_is_refused_not_traceback(tmp_path):
    """A qwen35moe GGUF whose config claims MAIN_LAYERS layers but is missing one
    layer's expert tensors entirely must REFUSE, not raise from deep inside the
    reader or crash on a KeyError."""
    path = str(tmp_path / "truncated.gguf")
    q8 = gguf.GGMLQuantizationType.Q8_0
    _write_qwen35moe_gguf(path, expert_types=[int(q8)] * MAIN_LAYERS, mtp=False)

    # Corrupt the checkpoint after writing by monkeypatching the config to claim
    # more served layers than the file actually has expert tensors for.
    from freetoken.models.check_experts import check_gguf_experts
    from freetoken.models.gguf.config import build_gguf_shim
    from freetoken.models.qwen3_5_moe import gguf as qwen_gguf

    shim = build_gguf_shim(path)
    real_expert_types = qwen_gguf._expert_types(shim)
    assert len(real_expert_types) == MAIN_LAYERS

    import freetoken.models.qwen3_5_moe.gguf as gguf_mod

    original = gguf_mod._main_layer_count
    try:
        gguf_mod._main_layer_count = lambda shim: MAIN_LAYERS + 5  # claims 5 layers that don't exist
        with pytest.raises(CheckFailed):
            check_gguf_experts(path)
    finally:
        gguf_mod._main_layer_count = original


def test_unrecognized_checkpoint_type_is_refused_cleanly(tmp_path):
    """Neither a HF config dir nor a .gguf file: a clean REFUSED, not a traceback."""
    empty_dir = tmp_path / "not_a_checkpoint"
    empty_dir.mkdir()
    (empty_dir / "readme.txt").write_text("nothing here")

    with pytest.raises(CheckFailed):
        check_experts(str(empty_dir))
