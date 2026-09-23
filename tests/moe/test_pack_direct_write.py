"""The generic NVFP4 Triton pack path writes gate/up pieces straight into their destination
bank slices (``write_fused_piece`` / ``write_fused_global``) instead of concatenating a
fused piece and copying it (``fused_piece`` / ``fused_global``). This must be byte-identical
to the old cat-then-copy code for every input shape the pack path sees:

  - "gated": synthetic pieces carry separate ``gate`` / ``up`` (+ ``_scale`` / ``_global``)
    tensors, exercising the split-and-copy branch (the one being optimized).
  - "ungated" (relu2-style): synthetic pieces already carry a single fused ``gate_up``
    (+ ``_scale`` / ``_global``) tensor, exercising the pass-through branch (unchanged,
    a single copy either way).

Also exercises the real pack path end to end (``TritonNvfp4MoEKernel.pack``) against a
reference built the old way, and CPU only (no CUDA needed for either).
"""

from __future__ import annotations

import torch

from freetoken.layers.quantization.moe.base import (
    fused_global,
    fused_piece,
    write_fused_global,
    write_fused_piece,
)
from freetoken.layers.quantization.moe.nvfp4 import TritonNvfp4MoEKernel

E, H, I = 4, 64, 32  # experts, hidden, moe intermediate (small synthetic sizes)


def _gated_pieces(seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {
        "gate": torch.randint(0, 256, (E, I, H // 2), dtype=torch.uint8, generator=g),
        "up": torch.randint(0, 256, (E, I, H // 2), dtype=torch.uint8, generator=g),
        "gate_scale": (torch.rand(E, I, H // 16, generator=g) * 1.5 + 0.25).to(torch.float8_e4m3fn),
        "up_scale": (torch.rand(E, I, H // 16, generator=g) * 1.5 + 0.25).to(torch.float8_e4m3fn),
        "gate_global": torch.rand(E, 1, generator=g).to(torch.float16),
        "up_global": torch.rand(E, 1, generator=g).to(torch.float16),
        "down": torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, generator=g),
        "down_scale": (torch.rand(E, H, I // 16, generator=g) * 1.5 + 0.25).to(torch.float8_e4m3fn),
        "down_global": torch.rand(E, 1, generator=g).to(torch.float16),
    }


def _ungated_pieces(seed: int = 1) -> dict[str, torch.Tensor]:
    """A relu2-style piece stream: gate_up already arrives fused (nothing to concatenate)."""
    g = torch.Generator().manual_seed(seed)
    return {
        "gate_up": torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, generator=g),
        "gate_up_scale": (torch.rand(E, 2 * I, H // 16, generator=g) * 1.5 + 0.25).to(torch.float8_e4m3fn),
        "gate_up_global": torch.rand(E, 1, generator=g).to(torch.float16),
        "down": torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, generator=g),
        "down_scale": (torch.rand(E, H, I // 16, generator=g) * 1.5 + 0.25).to(torch.float8_e4m3fn),
        "down_global": torch.rand(E, 1, generator=g).to(torch.float16),
    }


def _old_pack(pieces, out) -> None:
    """The pre-optimization TritonNvfp4MoEKernel.pack body: cat a fused piece, then copy_."""
    out["gate_up"].copy_(fused_piece(pieces, "gate_up"))
    out["gate_up_scale"].copy_(fused_piece(pieces, "gate_up_scale"))
    out["gate_up_global"].copy_(fused_global(pieces, I))


def _new_pack(pieces, out) -> None:
    write_fused_piece(out["gate_up"], pieces, "gate_up")
    write_fused_piece(out["gate_up_scale"], pieces, "gate_up_scale")
    write_fused_global(out["gate_up_global"], pieces, I)


def _empty_out():
    return {
        "gate_up": torch.zeros(E, 2 * I, H // 2, dtype=torch.uint8),
        "gate_up_scale": torch.zeros(E, 2 * I, H // 16, dtype=torch.float8_e4m3fn),
        "gate_up_global": torch.zeros(E, 2 * I, dtype=torch.float16),
    }


def _check_equal(pieces) -> None:
    ref = _empty_out()
    _old_pack(pieces, ref)

    got = _empty_out()
    _new_pack(pieces, got)

    for role in ref:
        assert torch.equal(ref[role], got[role]), f"bank {role!r} diverged between old (cat) and new (direct-write) pack"


def test_direct_write_matches_cat_gated():
    _check_equal(_gated_pieces())


def test_direct_write_matches_cat_ungated_fused_piece():
    _check_equal(_ungated_pieces())


def _full_out():
    kernel = TritonNvfp4MoEKernel()
    return {
        "gate_up": torch.zeros(E, 2 * I, H // 2, dtype=torch.uint8),
        "gate_up_scale": torch.zeros(E, 2 * I, H // 16, dtype=torch.float8_e4m3fn),
        "gate_up_global": torch.zeros(E, 2 * I, dtype=torch.float16),
        "down": torch.zeros(E, H, I // 2, dtype=torch.uint8),
        "down_scale": torch.zeros(E, H, I // 16, dtype=torch.float8_e4m3fn),
        "down_global": torch.zeros(E, H, dtype=torch.float16),
    }, kernel


class _Cfg:
    intermediate = I
    hidden = H


def _reference_full_pack(pieces, out) -> None:
    """The pre-optimization TritonNvfp4MoEKernel.pack, in full (including down)."""
    out["gate_up"].copy_(fused_piece(pieces, "gate_up"))
    out["gate_up_scale"].copy_(fused_piece(pieces, "gate_up_scale"))
    out["gate_up_global"].copy_(fused_global(pieces, I))
    out["down"].copy_(pieces["down"])
    out["down_scale"].copy_(pieces["down_scale"])
    from freetoken.layers.quantization.moe.base import global_rows

    out["down_global"].copy_(global_rows(pieces["down_global"], H))


def test_kernel_pack_matches_reference_gated():
    pieces = _gated_pieces()
    ref, _ = _full_out()
    _reference_full_pack(pieces, ref)

    out, kernel = _full_out()
    kernel.pack(pieces, _Cfg(), out)

    for role in ref:
        assert torch.equal(ref[role], out[role]), f"bank {role!r} diverged: TritonNvfp4MoEKernel.pack vs reference"


def test_kernel_pack_matches_reference_ungated():
    pieces = _ungated_pieces()
    ref, _ = _full_out()
    _reference_full_pack(pieces, ref)

    out, kernel = _full_out()
    kernel.pack(pieces, _Cfg(), out)

    for role in ref:
        assert torch.equal(ref[role], out[role]), f"bank {role!r} diverged: TritonNvfp4MoEKernel.pack vs reference"
