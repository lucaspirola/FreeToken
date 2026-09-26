"""EXL3 routed experts: the checkpoint's own trellis / suh / svh bytes in six raw-row banks.

The banks are the RAM saver's rows unchanged (``BankSpec.raw_row``): ``pack`` copies each
projection's tensors to the byte offsets ``models/exl3_banks.Exl3ExpertIndex`` places them at,
which is also where the mirror pool reads them from disk. The forward is
``freetoken.moe.fused_exl3``; decode and prefill read the same banks.
"""

from __future__ import annotations

import torch

from ..registry import LayerKind, register_method
from ..scheme import QuantKind
from .base import BankSpec, ExpertView, MoEConfig, MoEKernel, MoEMethod, gated_epilogue_reason, limit_or_inf


def _facts(cfg: MoEConfig) -> tuple[int, str]:
    from ..linear.exl3 import exl3_facts

    return exl3_facts(cfg.scheme)


class TritonExl3MoEKernel(MoEKernel):
    """Trellis decode inside Triton GEMV (decode) / grouped GEMM (prefill) kernels."""

    name = "triton"

    def unusable_reason(self, cfg: MoEConfig) -> str | None:
        reason = self._common_reject(cfg, resident_ok=False, tp_ok=False, cpu_ok=False, plain_silu_only=False)
        if reason:
            return reason
        reason = gated_epilogue_reason(cfg)
        if reason:
            return f"triton exl3 MoE kernel: {reason}"
        if cfg.apply_router_weight_on_input:
            return "triton exl3 MoE kernel weights the routes on the output only"
        if cfg.hidden % 128 or cfg.intermediate % 128:
            return f"EXL3 experts need 128-aligned hidden/intermediate, got {cfg.hidden}/{cfg.intermediate}"
        return None

    def layout(self, cfg: MoEConfig) -> dict[str, BankSpec]:
        from freetoken.models.exl3_banks import exl3_bank_shapes

        bits, _ = _facts(cfg)
        # raw_row: pack() writes the checkpoint bytes at the offsets the mirror pool reads them
        # to (both from Exl3ExpertIndex.placement); tests/moe/test_exl3_banks.py proves it.
        return {
            name: BankSpec(shape, dtype, raw_row=True)
            for name, (shape, dtype) in exl3_bank_shapes(cfg.hidden, cfg.intermediate, bits).items()
        }

    def pack(self, pieces, cfg: MoEConfig, out):
        for kind in ("trellis", "suh", "svh"):
            out[f"gate_up_{kind}"][:, 0].copy_(pieces[f"gate_{kind}"])
            out[f"gate_up_{kind}"][:, 1].copy_(pieces[f"up_{kind}"])
            out[f"down_{kind}"].copy_(pieces[f"down_{kind}"])
        return {}

    def apply(self, layer, x, topk_weights, topk_ids, view: ExpertView, *, is_prefill: bool):
        from freetoken.moe.fused_exl3 import fused_experts_exl3

        bits, codebook = _facts(layer.quant_method.cfg)
        t = view.tensors
        banks = tuple(t[n] for n in ("gate_up_trellis", "gate_up_suh", "gate_up_svh", "down_trellis", "down_suh", "down_svh"))
        return fused_experts_exl3(
            x, banks, topk_weights, topk_ids, bits=bits, codebook=codebook,
            activation=layer.activation, alpha=float(layer.alpha), limit=limit_or_inf(layer),
            is_prefill=is_prefill, num_experts=view.n,
        )


@register_method(QuantKind.EXL3, LayerKind.MOE)
class Exl3MoEMethod(MoEMethod):
    candidates = (TritonExl3MoEKernel,)

    def create_weights(self, layer) -> None:
        raise NotImplementedError("EXL3 experts are served from the offload cache (--moe-strategy offload), not resident")

    def resident_view(self, layer) -> ExpertView:
        raise NotImplementedError("EXL3 experts are not resident")


__all__ = ["Exl3MoEMethod", "TritonExl3MoEKernel"]
