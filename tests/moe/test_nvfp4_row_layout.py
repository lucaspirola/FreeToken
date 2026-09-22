"""Byte-equality between ``nvfp4_expert_row_layout`` and the real loader.

S5a (``tasks/exclusive-expert-ram/reviews/2026-09-22-refactor-plan-final.md``
section 3.4): the NVFP4 expert-row byte layout used to be stated in four
independent places. ``models.nvfp4_banks.nvfp4_expert_row_layout`` is the new
single source of truth; ``_alloc_nvfp4_host_banks`` already consumes it
(``test_alloc_host_banks_matches_layout`` below). This file is "the test that
matters": it proves the layout function's addressing is not just internally
consistent but actually agrees, tensor for tensor, with what
``load_nvfp4_expert_source_banks`` puts in a real host bank for the same
on-disk checkpoint bytes -- the invariant ``mirror_pool.py`` claims in prose
("The mirror's rows must be byte-identical to what the regular loader
produces") and nothing tested before this file.

CPU only: three synthetic checkpoints (``tests/moe/_mirror_checkpoint.py``),
no GPU, no server, no real model. Small enough (2 layers x 3 experts x 32x32)
to enumerate every tensor by hand.
"""
from __future__ import annotations

import re
import types

import pytest
import torch

from freetoken.models.nvfp4_banks import (
    Nvfp4ExpertSourceSpec,
    load_nvfp4_expert_source_banks,
    nvfp4_expert_row_layout,
)
from freetoken.moe.host_banks import alloc_layer_banks
from freetoken.moe.mirror_pool import nvfp4_bank_shapes

from ._mirror_checkpoint import (
    _KIND_SUFFIX_OF_NAMING,
    _PROJ_OF_ROLE,
    row_tensor_byte,
    row_tensor_global,
    write_generic_nvfp4_checkpoint,
)

LAYERS, EXPERTS, H, I = 2, 3, 32, 32
_CANONICAL_KINDS = ("weight", "weight_scale", "weight_scale_2")


def _synthetic_spec(gated: bool, naming: str) -> Nvfp4ExpertSourceSpec:
    """A throwaway ``Nvfp4ExpertSourceSpec`` for one (gated, naming) combination.

    Not a real model's spec: the point of this test is to isolate ``gated``
    and the on-disk tensor-kind naming as the only two variables, so the key
    format (``backbone.layers.N.mixer.experts.E.{proj}.{kind}``) is fixed to
    what ``write_generic_nvfp4_checkpoint`` writes.
    """
    roles = ("gate", "up", "down") if gated else ("up", "down")
    kind_suffix = _KIND_SUFFIX_OF_NAMING[naming]
    suffixes = {kind_suffix[k] for k in _CANONICAL_KINDS}
    proj_alt = "|".join(_PROJ_OF_ROLE[r] for r in roles)
    kind_alt = "|".join(re.escape(s) for s in suffixes)
    key_pattern = re.compile(
        rf"^backbone\.layers\.(?P<layer>\d+)\.mixer\.experts\.(?P<expert>\d+)\."
        rf"(?P<proj>{proj_alt})\.(?P<kind>{kind_alt})$"
    )
    return Nvfp4ExpertSourceSpec(
        key_pattern=key_pattern,
        proj_to_role={_PROJ_OF_ROLE[r]: r for r in roles},
        layer_to_bank=lambda layer, config: layer,
        desc=f"synthetic {naming} ({'gated' if gated else 'ungated'})",
        gated=gated,
    )


def _synthetic_config() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        num_experts=EXPERTS, hidden_size=H, moe_intermediate_size=I,
        num_moe_layers=LAYERS,
    )


def _itemsize(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _assert_rows_byte_equal(root: str, *, gated: bool, naming: str) -> None:
    """Load the checkpoint for real, then check every tensor's placement that
    ``nvfp4_expert_row_layout`` computes against what the loader actually put
    in the bank at that (bank, byte_offset, shape)."""
    spec = _synthetic_spec(gated, naming)
    config = _synthetic_config()
    layout = nvfp4_expert_row_layout(H, I, gated=gated)
    # layer_sink=no-op: skip the default PinPipeline path, which needs the
    # cudaHostRegister CUDA extension this CPU test environment does not have
    # built. This test only cares whether the bytes landed correctly, not
    # whether the tensors ended up pinned (a GPU-only concern).
    banks = load_nvfp4_expert_source_banks(
        root, config, spec, drop_page_cache=lambda _p: None, primary=False,
        layer_sink=lambda _layer_id, _banks: None,
    )
    roles = ("gate", "up", "down") if gated else ("up", "down")

    for layer in range(LAYERS):
        for expert in range(EXPERTS):
            flat = layer * EXPERTS + expert
            for role in roles:
                for kind in ("weight", "weight_scale"):
                    placement = layout.tensors[(role, kind)]
                    bank_dtype = layout.bank_shapes[placement.bank][1]
                    rows, width = placement.checkpoint_shape
                    # byte_offset is into the bank row FLATTENED (rows*width
                    # elements); the bank tensor is 2D, so the row-index slice
                    # needs a further /width.
                    row_start = placement.byte_offset // _itemsize(bank_dtype) // width
                    got = banks[placement.bank][layer][expert, row_start:row_start + rows, :]
                    want = torch.tensor(
                        [row_tensor_byte(flat, role, kind, k) for k in range(rows * width)],
                        dtype=torch.uint8,
                    ).view(rows, width)
                    assert torch.equal(got.view(torch.uint8).cpu(), want), (
                        naming, gated, layer, expert, role, kind
                    )

                # weight_scale_2: on disk a scalar FP32, on the host a
                # per-output-row FP16 broadcast -- byte-identical to the
                # loader's own conversion, not to the on-disk bytes.
                placement = layout.tensors[(role, "weight_scale_2")]
                bank_dtype = layout.bank_shapes[placement.bank][1]
                row_start = placement.byte_offset // _itemsize(bank_dtype)
                rows = I if role != "down" else H
                got = banks[placement.bank][layer][expert, row_start:row_start + rows]
                scalar = torch.tensor(row_tensor_global(flat, role), dtype=torch.float32)
                want = torch.full((rows,), scalar.to(torch.float16).item(), dtype=torch.float16)
                assert torch.equal(got.cpu(), want), (naming, gated, layer, expert, role)


@pytest.mark.parametrize("gated", [False, True], ids=["ungated", "gated"])
def test_row_layout_equivalence_modelopt(tmp_path, gated):
    """ungated modelopt and gated modelopt: rows built from
    ``nvfp4_expert_row_layout`` are byte-equal to the loader's own rows."""
    write_generic_nvfp4_checkpoint(
        str(tmp_path), LAYERS, EXPERTS, H, I, gated=gated, naming="modelopt",
    )
    _assert_rows_byte_equal(str(tmp_path), gated=gated, naming="modelopt")


def test_row_layout_equivalence_compressed_tensors_gated_is_refused_today(tmp_path):
    """gated compressed-tensors (``weight_packed`` / ``weight_global_scale``
    naming): the genuine finding.

    ``load_nvfp4_expert_source_banks``'s kind handling is hardcoded to
    modelopt's three names (``kind == "weight_scale_2"`` / ``kind in
    {"weight", "weight_scale"}`` / else-raise) -- it does not yet accept a
    ``kind_map`` to canonicalise a different on-disk naming, because
    ``Nvfp4ExpertSourceSpec`` has no ``kind_map`` field until the S0 merge
    lands that upstream spec field. So this checkpoint, which
    ``nvfp4_expert_row_layout`` and ``write_generic_nvfp4_checkpoint`` both
    already handle correctly at the byte-layout level (the layout does not
    depend on on-disk naming at all -- see
    ``test_layout_is_naming_agnostic`` below), is refused by the loader today
    with exactly the "unknown tensor kind" error the refactor plan (section
    3.4) names. This is not a bug in ``nvfp4_expert_row_layout``: it is the
    reason the plan defers ``kind_map`` wiring to after the merge.
    """
    write_generic_nvfp4_checkpoint(
        str(tmp_path), LAYERS, EXPERTS, H, I, gated=True, naming="compressed_tensors",
    )
    spec = _synthetic_spec(gated=True, naming="compressed_tensors")
    config = _synthetic_config()
    with pytest.raises(ValueError, match="unknown NVFP4 expert tensor kind"):
        load_nvfp4_expert_source_banks(
            str(tmp_path), config, spec, drop_page_cache=lambda _p: None, primary=False,
        )


def test_layout_is_naming_agnostic():
    """The layout depends only on (H, I, gated), never on what a checkpoint
    calls a tensor -- which is exactly what makes ``kind_map`` (S5a part 2,
    after the merge) a pure canonicalisation step in front of this function
    rather than a change to it."""
    gated_layout = nvfp4_expert_row_layout(H, I, gated=True)
    assert nvfp4_expert_row_layout(H, I, gated=True, kind_map={"weight_packed": "weight"}) == gated_layout


def test_alloc_host_banks_matches_layout():
    """``_alloc_nvfp4_host_banks`` no longer carries its own copy of the shapes:
    it must allocate exactly what ``nvfp4_expert_row_layout`` says."""
    from freetoken.models.nvfp4_banks import _alloc_nvfp4_host_banks

    for gated in (False, True):
        layout = nvfp4_expert_row_layout(H, I, gated=gated)
        banks = _alloc_nvfp4_host_banks(LAYERS, EXPERTS, H, I, gated=gated)
        assert set(banks) == set(layout.bank_shapes)
        for name, (shape, dtype) in layout.bank_shapes.items():
            layer_bank = banks[name][0]
            assert layer_bank.tensor.shape == (EXPERTS, *shape)
            assert layer_bank.tensor.dtype == dtype


def test_layout_matches_the_mirror_pools_own_shapes_today():
    """Cross-check against ``moe.mirror_pool.nvfp4_bank_shapes`` (the
    still-separate copy this lane does not touch): the two must currently
    agree, since S5a part 2 (wiring the mirror pool onto this function) has
    not landed. If they ever disagree, one of the two copies drifted and
    that is exactly the failure mode S5a exists to end."""
    for gated in (False, True):
        layout = nvfp4_expert_row_layout(H, I, gated=gated)
        mirror_shapes = nvfp4_bank_shapes(H, I, gated=gated)
        for name, (shape, dtype) in layout.bank_shapes.items():
            assert mirror_shapes[name] == (shape, dtype), (name, gated)


def test_row_bytes_accounts_for_gating():
    """The four-copies bug the plan calls out: ``_BANK_BYTES_PER_EXPERT["nvfp4"]``
    assumed gated ``2*I`` regardless of ``expert_gated``. This function must not
    repeat it: an ungated row is smaller than a gated one at the same H, I, and
    the difference is exactly one extra ``gate_up_*`` share (the ``down_*``
    banks, which do not depend on gating, must be identical either way)."""
    down_names = {"down_packed", "down_scale", "down_global"}
    layout_u = nvfp4_expert_row_layout(H, I, gated=False)
    layout_g = nvfp4_expert_row_layout(H, I, gated=True)
    for name in down_names:
        assert layout_u.bank_shapes[name] == layout_g.bank_shapes[name]

    gate_up_names = {"gate_up_packed", "gate_up_scale", "gate_up_global"}
    ungated_gate_up_bytes = sum(
        _itemsize(dt) * _prod(shape)
        for name, (shape, dt) in layout_u.bank_shapes.items()
        if name in gate_up_names
    )
    assert layout_g.row_bytes == layout_u.row_bytes + ungated_gate_up_bytes


def _prod(shape: tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n
