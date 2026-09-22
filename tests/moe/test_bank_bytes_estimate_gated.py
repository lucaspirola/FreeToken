"""S5b: ``_BANK_BYTES_PER_EXPERT["nvfp4"]`` honours ``gated``.

The table used to assume a gated ``2*I`` gate|up bank for every NVFP4 model.
``bank_bytes_estimate`` never reached it for Nemotron-H (an ungated guard in
``moe/expert_banks.py`` special-cases it first), so this is hygiene: the table
entry, the guard and the single-source row layout must agree, and the gated
answer must not move.
"""
from __future__ import annotations

import types

from freetoken.models.nvfp4_banks import nvfp4_expert_row_layout
from freetoken.moe.expert_banks import bank_bytes_estimate
from freetoken.moe.offload_cache import _BANK_BYTES_PER_EXPERT

# NVIDIA-Nemotron-3.5-Lightning-30B-A3B geometry (relu2 experts: ungated).
H, I, E, L = 2688, 1856, 128, 23


def _config(*, gated: bool):
    return types.SimpleNamespace(
        expert_quant="nvfp4", moe_weight_format=None, num_moe_layers=L,
        num_experts=E, hidden_size=H, expert_hidden_size=None,
        moe_intermediate_size=I, expert_gated=gated, gguf_expert_types=None,
    )


def test_bank_bytes_estimate_gated_ungated_nemotron_matches_row_layout():
    row = nvfp4_expert_row_layout(H, I, gated=False).row_bytes
    assert bank_bytes_estimate(_config(gated=False)) == L * E * row
    # the table entry itself now agrees (it used to say the gated size)
    assert _BANK_BYTES_PER_EXPERT["nvfp4"](H, I, gated=False) == row
    # the real host footprint of the banks, 15.41 GiB -- not 23.12
    assert round(L * E * row / 2**30, 2) == 15.41


def test_bank_bytes_estimate_gated_case_unchanged():
    row = nvfp4_expert_row_layout(H, I, gated=True).row_bytes
    legacy = 2 * I * (H // 2 + H // 16 + 2) + H * (I // 2 + I // 16 + 2)
    assert row == legacy
    assert _BANK_BYTES_PER_EXPERT["nvfp4"](H, I) == legacy  # default stays gated
    assert _BANK_BYTES_PER_EXPERT["nvfp4"](H, I, gated=True) == legacy
    assert bank_bytes_estimate(_config(gated=True)) == L * E * legacy
    assert round(L * E * legacy / 2**30, 2) == 23.12
