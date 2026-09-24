"""FreeToken KV-cache quantization, vendored standalone (no freetoken import).

``quantize`` / ``dequantize`` are copied verbatim from FreeToken's pure-torch reference
(python/freetoken/kvcache/quant.py, KVQuantSpec.quantize lines 123-207 and .dequantize lines
209-250 at FreeToken-wt/reorg 33dc23e), which is the oracle its Triton store kernel
(kernel/triton/kv_quant.py) and attention loaders (kernel/triton/attention.py::_load_kv) are
tested against (tests/kernels/test_kv_quant.py). Specs (lines 253-267) are copied too:
BLOCK=32 elements per scale along head_dim, fp16 scales.

``fake_quant`` is what the attention kernel actually sees for a stored element: FreeToken's
``_load_kv`` computes ``(code as fp32) * (fp16 scale as fp32)`` and casts to the query dtype
(``.to(out_dtype)``, attention.py:349/401/480/493), i.e. bf16. So the emulated value is
``dequantize(quantize(x)).to(x.dtype)`` with ``x`` the bf16 K or V FreeToken hands to
``store_kv`` (quantize itself runs on ``x.float()``, as the store kernel loads the bf16 source
and works in fp32). dryrun/parity_check.py proves these functions bit-identical to FreeToken's.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

BLOCK = 32
SCALE_DTYPE = torch.float16


@dataclass(frozen=True)
class KVQuantSpec:
    name: str
    storage_dtype: torch.dtype | None
    max_magnitude: float
    bits: int = 8

    @property
    def enabled(self) -> bool:
        return self.storage_dtype is not None

    @property
    def is_integer(self) -> bool:
        return self.storage_dtype in (torch.int8, torch.uint8)

    def logical_dim(self, storage_dim: int) -> int:
        payload_bits = storage_dim * 8
        if payload_bits % self.bits:
            raise ValueError(f"storage dim {storage_dim} does not represent whole {self.name} values")
        return payload_bits // self.bits

    # ---- verbatim copy of FreeToken's reference implementations ----

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.enabled, "quantize() on an unquantized spec"
        blocks = x.float().unflatten(-1, (x.shape[-1] // BLOCK, BLOCK))
        if self.bits == 4:
            extreme = blocks.gather(
                -1, blocks.abs().argmax(dim=-1, keepdim=True)
            ).squeeze(-1)
            scales = torch.where(extreme != 0, extreme / -8.0, torch.ones_like(extreme))
            scales = scales.to(SCALE_DTYPE)
            q = torch.floor(blocks / scales.float().unsqueeze(-1) + 8.5).clamp_(0, 15)
            q = q.flatten(-2).to(torch.uint8)
            even = q[..., 0::2]
            odd = q[..., 1::2]
            return even | (odd << 4), scales

        if self.bits == 5:
            abs_blocks = blocks.abs()
            extreme = blocks.gather(
                -1, abs_blocks.argmax(dim=-1, keepdim=True)
            ).squeeze(-1)
            scales = torch.where(extreme != 0, extreme / -16.0, torch.ones_like(extreme))
            scales = scales.to(SCALE_DTYPE)
            codes = torch.floor(blocks / scales.float().unsqueeze(-1) + 16.5).clamp_(0, 31)
            codes = codes.flatten(-2).to(torch.uint8)
            lo = (codes[..., 0::2] & 0x0F) | ((codes[..., 1::2] & 0x0F) << 4)
            hi = sum(
                (((codes[..., lane::8] >> 4) & 0x01) << lane)
                for lane in range(8)
            )
            return torch.cat((lo, hi), dim=-1), scales

        if self.bits == 6:
            abs_blocks = blocks.abs()
            extreme = blocks.gather(
                -1, abs_blocks.argmax(dim=-1, keepdim=True)
            ).squeeze(-1)
            initial = torch.where(extreme != 0, extreme / -32.0, torch.ones_like(extreme))
            codes = torch.floor(blocks / initial.unsqueeze(-1) + 32.5).clamp_(0, 63)
            signed = codes - 32.0
            weights = blocks.square()
            sumqx = (weights * signed * blocks).sum(dim=-1)
            sumq2 = (weights * signed.square()).sum(dim=-1)
            scales = torch.where(sumq2 > 0, sumqx / sumq2, initial).to(SCALE_DTYPE)
            codes = codes.flatten(-2).to(torch.uint8)
            lo = (codes[..., 0::2] & 0x0F) | ((codes[..., 1::2] & 0x0F) << 4)
            hi = (
                ((codes[..., 0::4] >> 4) & 0x03)
                | (((codes[..., 1::4] >> 4) & 0x03) << 2)
                | (((codes[..., 2::4] >> 4) & 0x03) << 4)
                | (((codes[..., 3::4] >> 4) & 0x03) << 6)
            )
            return torch.cat((lo, hi), dim=-1), scales

        amax = blocks.abs().amax(dim=-1)
        scales = torch.where(amax > 0, amax / self.max_magnitude, torch.ones_like(amax))
        scales = scales.to(SCALE_DTYPE)
        q = blocks / scales.float().unsqueeze(-1)
        if self.is_integer:
            q = torch.where(q >= 0, (q + 0.5).floor(), (q - 0.5).ceil())
            q = q.clamp_(-self.max_magnitude, self.max_magnitude)
        return q.flatten(-2).to(self.storage_dtype), scales

    def dequantize(self, q: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        assert self.enabled, "dequantize() on an unquantized spec"
        if self.bits == 4:
            logical_d = self.logical_dim(q.shape[-1])
            nblock = logical_d // BLOCK
            codes = q.to(torch.uint8)
            blocks = codes.unflatten(-1, (nblock, BLOCK // 2))
            values = torch.stack([blocks & 0x0F, blocks >> 4], dim=-1)
            values = values.reshape(*blocks.shape[:-1], BLOCK).float()
            values = values - 8.0
        elif self.bits == 5:
            logical_d = self.logical_dim(q.shape[-1])
            nblock = logical_d // BLOCK
            payload = q.to(torch.uint8)
            lo, hi = payload[..., : logical_d // 2], payload[..., logical_d // 2 :]
            lower = torch.stack((lo & 0x0F, lo >> 4), dim=-1).flatten(-2)
            upper = torch.stack(
                tuple((hi >> lane) & 0x01 for lane in range(8)), dim=-1
            ).flatten(-2)
            values = (lower | (upper << 4)).float().unflatten(
                -1, (nblock, BLOCK)
            ) - 16.0
        elif self.bits == 6:
            logical_d = self.logical_dim(q.shape[-1])
            nblock = logical_d // BLOCK
            payload = q.to(torch.uint8)
            lo, hi = payload[..., : logical_d // 2], payload[..., logical_d // 2 :]
            lower = torch.stack((lo & 0x0F, lo >> 4), dim=-1).flatten(-2)
            upper = torch.stack(
                (hi & 0x03, (hi >> 2) & 0x03, (hi >> 4) & 0x03, hi >> 6),
                dim=-1,
            ).flatten(-2)
            values = (lower | (upper << 4)).float().unflatten(
                -1, (nblock, BLOCK)
            ) - 32.0
        else:
            values = q.float().unflatten(-1, (q.shape[-1] // BLOCK, BLOCK))
        return (values * scales.float().unsqueeze(-1)).flatten(-2)


Q8_0 = KVQuantSpec(name="q8_0", storage_dtype=torch.int8, max_magnitude=127.0)
FP8_E4M3 = KVQuantSpec(name="fp8_e4m3", storage_dtype=torch.float8_e4m3fn, max_magnitude=448.0)
INT4 = KVQuantSpec(name="int4", storage_dtype=torch.uint8, max_magnitude=8.0, bits=4)
Q6_0 = KVQuantSpec(name="q6_0", storage_dtype=torch.uint8, max_magnitude=32.0, bits=6)
Q5_0 = KVQuantSpec(name="q5_0", storage_dtype=torch.uint8, max_magnitude=16.0, bits=5)
NONE = KVQuantSpec(name="auto", storage_dtype=None, max_magnitude=0.0)

BY_NAME = {s.name: s for s in (NONE, Q8_0, FP8_E4M3, INT4, Q5_0, Q6_0)}
BY_NAME["q4_0"] = INT4
BY_NAME["bf16"] = NONE


def quantize_as_kernel(spec: KVQuantSpec, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The Triton store kernel's result (kernel/triton/kv_quant.py). Identical to the torch
    oracle ``spec.quantize`` except in ONE case, found by dryrun/parity_check.py: for fp8_e4m3
    the kernel clamps ``x / scale`` to +-448 before rounding (kv_quant.py:130
    ``round_e4m3(tl.minimum(tl.maximum(q, -MAX_MAG), MAX_MAG))``) while the oracle casts
    unclamped. When a block's fp16 scale ``amax/448`` is subnormal (amax < ~0.027) or
    underflows to 0, ``x/scale`` can exceed the e4m3 range and the oracle yields NaN; the
    kernel yields +-448. The served numerics are the kernel's, so fp8 is re-done here with the
    clamp (RNE fp32->e4m3 cast after clamping == round_e4m3, kv_quant.py:126-130). 0/0 (a zero
    element in a block whose scale underflowed to 0) is mapped to code 0; it dequantizes to 0
    either way. All other formats: the oracle already clamps and is used verbatim."""
    if spec.storage_dtype is not torch.float8_e4m3fn:
        return spec.quantize(x)
    blocks = x.float().unflatten(-1, (x.shape[-1] // BLOCK, BLOCK))
    amax = blocks.abs().amax(dim=-1)
    scales = torch.where(amax > 0, amax / spec.max_magnitude, torch.ones_like(amax)).to(SCALE_DTYPE)
    q = blocks / scales.float().unsqueeze(-1)
    q = torch.nan_to_num(q, nan=0.0).clamp(-spec.max_magnitude, spec.max_magnitude)
    return q.flatten(-2).to(spec.storage_dtype), scales


def fake_quant(spec: KVQuantSpec, x: torch.Tensor) -> torch.Tensor:
    """What FreeToken's attention reads back for a stored bf16 element (see module doc)."""
    if not spec.enabled:
        return x
    return spec.dequantize(*quantize_as_kernel(spec, x)).to(x.dtype)


# Lanes: name -> (K format, V format, servable-in-FreeToken note). Order = run priority.
# FreeToken's config gate (engine/engine.py:293-305) accepts q5_0/q6_0 only in the validated
# pairs (q8_0 K, q6_0 V) and (q6_0 K, q5_0 V), and different K/V formats only for those pairs;
# the other mixed lanes here are diagnostics that FreeToken cannot serve today.
LANES = {
    "bf16":          ("bf16", "bf16", "reference (unquantized pool)"),
    "q4_0":          ("q4_0", "q4_0", "servable (--kv-cache-dtype q4_0)"),
    "q8_0":          ("q8_0", "q8_0", "servable (default serve profile)"),
    "fp8_e4m3":      ("fp8_e4m3", "fp8_e4m3", "servable"),
    "q6_0K_q5_0V":   ("q6_0", "q5_0", "servable (validated pair)"),
    "q8_0K_q6_0V":   ("q8_0", "q6_0", "servable (validated pair)"),
    "q5_0":          ("q5_0", "q5_0", "emulated only: FreeToken rejects q5_0/q5_0 today"),
    "q4_0K_bf16V":   ("q4_0", "bf16", "diagnostic only (K side alone)"),
    "bf16K_q4_0V":   ("bf16", "q4_0", "diagnostic only (V side alone)"),
}
