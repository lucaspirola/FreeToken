from __future__ import annotations

from types import SimpleNamespace

from freetoken.models.gguf.dequant import (
    GGML_BF16,
    GGML_Q4_0,
    GGML_Q4_K,
    GGML_Q6_K,
    expert_bank_geometry,
)
from freetoken.moe.expert_banks import bank_bytes_estimate


def _align(n: int) -> int:
    return (n + 63) // 64 * 64


def _cfg(
    expert_quant: str,
    types: tuple[tuple[int, int], ...] | None,
    *,
    H: int = 2048,
    I: int = 512,
    E: int = 256,
    layers: int = 40,
    fmt: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        expert_quant=expert_quant,
        moe_weight_format=fmt or ("bf16" if expert_quant == "none" else expert_quant),
        num_moe_layers=layers,
        num_experts=E,
        hidden_size=H,
        moe_intermediate_size=I,
        gguf_expert_types=types,
    )


def test_mixed_gguf_geometry_uses_global_max_down_stride() -> None:
    """Mixed Q4_K/Q6_K: gate_up all Q4_K, down Q4_K x20 + Q6_K x20. The uniform
    flat-slot stride must be the global max (the Q6_K down stride), so the Q4_K
    down layers are padded to it."""
    cfg = _cfg(
        "gguf",
        types=((GGML_Q4_K, GGML_Q6_K),) * 20 + ((GGML_Q4_K, GGML_Q4_K),) * 20,
    )
    gu, down = expert_bank_geometry(cfg)
    assert gu == _align(2 * 512 * (2048 // 256 * 144))  # Q4_K gate_up row payload
    q6_down = _align(2048 * (512 // 256 * 210))
    q4_down = _align(2048 * (512 // 256 * 144))
    assert down == q6_down
    assert down > q4_down  # max down stride wins globally


def test_estimate_is_layers_times_experts_times_aligned_strides() -> None:
    """The bank footprint is allocation (aligned flat slots), not payload bytes:
    layers x experts x (gate_up_stride + down_stride)."""
    cfg = _cfg(
        "gguf",
        types=((GGML_Q4_K, GGML_Q6_K),) * 20 + ((GGML_Q4_K, GGML_Q4_K),) * 20,
    )
    gu, down = expert_bank_geometry(cfg)
    assert bank_bytes_estimate(cfg) == 40 * 256 * (gu + down)
    # Same geometry as the validated Ornith checkpoint.
    assert bank_bytes_estimate(cfg) == 20_887_633_920


def test_homogeneous_gguf_geometry() -> None:
    cfg = _cfg("gguf", types=((GGML_Q4_K, GGML_Q4_K),) * 40)
    gu, down = expert_bank_geometry(cfg)
    assert down == _align(2048 * (512 // 256 * 144))
    assert bank_bytes_estimate(cfg) == 40 * 256 * (gu + down)


def test_unknown_quant_returns_none_not_an_estimate() -> None:
    cfg = _cfg("gguf", types=((999, GGML_Q4_K),) * 40)
    assert expert_bank_geometry(cfg) is None
    assert bank_bytes_estimate(cfg) is None


def test_missing_or_empty_metadata_returns_none() -> None:
    assert expert_bank_geometry(_cfg("gguf", types=None)) is None
    assert bank_bytes_estimate(_cfg("gguf", types=None)) is None
    assert bank_bytes_estimate(_cfg("gguf", types=(), layers=0)) is None


def test_malformed_type_pair_returns_none() -> None:
    """A (gate_up, down) pair that is not a 2-tuple must not crash the estimator."""
    cfg = _cfg("gguf", types=((GGML_Q4_K,),) * 40)  # type: ignore[arg-type]
    assert expert_bank_geometry(cfg) is None
    assert bank_bytes_estimate(cfg) is None


def test_non_gguf_path_unchanged() -> None:
    bf16 = _cfg("none", types=None, fmt="bf16")
    assert bank_bytes_estimate(bf16) == 40 * 256 * (3 * 512 * 2048 * 2)


def test_model_geometry_wrappers_share_the_helper() -> None:
    """The qwen35moe and laguna GGUF geometry wrappers and the estimator must all
    resolve to the shared helper (no duplicated uniform-max formulas)."""
    import freetoken.models.laguna.gguf as laguna
    import freetoken.models.qwen3_5_moe.gguf as qwen35

    qwen_cfg = _cfg("gguf", types=((GGML_Q4_K, GGML_Q6_K),) * 40)
    laguna_cfg = _cfg("gguf", types=((GGML_Q4_0, GGML_BF16),) * 40, H=4096, I=1024)

    assert qwen35._expert_bank_geometry(qwen_cfg) == expert_bank_geometry(qwen_cfg)
    assert laguna._expert_bank_geometry(laguna_cfg) == expert_bank_geometry(laguna_cfg)
    gu, down = expert_bank_geometry(qwen_cfg)
    assert bank_bytes_estimate(qwen_cfg) == 40 * 256 * (gu + down)
