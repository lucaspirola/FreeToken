"""AOT shape table: per-entry derivations that a drifting config would silently break.

A wrong row here does not fail loudly -- the prebuilt cache just misses by spec name and
the kernel falls back to JIT (which needs nvcc). CPU-only.
"""

from __future__ import annotations

import pytest

from freetoken.kernel.aot_models import (
    SUPPORTED_MODELS,
    aggregate_fast_index_copy_feature_sizes,
    expert_bank_row_bytes,
    fast_index_copy_feature_sizes,
)


def _entry(architecture: str):
    return next(m for m in SUPPORTED_MODELS if m.architecture == architecture)


def test_every_bank_row_is_fused_copy_aligned():
    """The fused multi-bank copy only engages on 16-byte multiples."""
    for model in SUPPORTED_MODELS:
        for size in fast_index_copy_feature_sizes(model):
            assert size % 16 == 0, (model.name, size)


def test_ungated_experts_halve_the_gate_up_bank():
    gated = expert_bank_row_bytes("nvfp4", 2688, 1856)
    ungated = expert_bank_row_bytes("nvfp4", 2688, 1856, gated=False)
    assert ungated["gate_up_packed"] * 2 == gated["gate_up_packed"]
    assert ungated["gate_up_scale"] * 2 == gated["gate_up_scale"]
    assert ungated["gate_up_global"] * 2 == gated["gate_up_global"]
    # The down bank is [H, I] either way.
    assert ungated["down_packed"] == gated["down_packed"]
    assert ungated["down_global"] == gated["down_global"]


def test_gating_defaults_to_true_for_every_other_family():
    ungated = {m.name for m in SUPPORTED_MODELS if not m.expert_gated}
    assert ungated == {"nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"}


def test_nemotron_3_5_lightning_entry():
    entry = _entry("NemotronHForCausalLM")
    assert entry.hidden_size == 2688
    assert entry.kv_groups == ((2, 128),)  # 6 full-attention layers, 2 kv heads x 128
    assert entry.top_k == 6
    assert entry.moe_intermediate_size == 1856
    assert entry.expert_formats == ("nvfp4",)
    assert entry.expert_gated is False
    banks = expert_bank_row_bytes(
        "nvfp4", entry.hidden_size, entry.moe_intermediate_size, gated=False
    )
    # 128 routed experts x 23 MoE layers == the 15.4 GiB of host banks the launch
    # profiles in docs/models.md budget for.
    total = sum(banks.values()) * 128 * 23
    assert total / 2**30 == pytest.approx(15.41, abs=0.02)
    assert set(fast_index_copy_feature_sizes(entry)) <= set(
        aggregate_fast_index_copy_feature_sizes()
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
