"""KV-cache quantization schemes with a per-block scale.

The KV pool normally stores K/V in the model's compute dtype (bf16). Quantized pools
store a compact payload plus a parallel scale tensor holding one fp16 scale per
:data:`BLOCK` elements along ``head_dim`` -- the same block geometry GGUF's Q8_0 uses.
The block is small because KV outliers (mostly in the keys) concentrate in a few
channels, and a block of 32 keeps an outlier from stretching the scale of the whole
head.

The 8-bit schemes store one element per byte; ``int4`` stores two signed values in each
``uint8`` byte using llama.cpp/GGML Q4_0 scale selection. Q5 and Q6 use cache-native
contiguous bit planes with GGML Q5_0/Q6_0 scale selection. All share the store kernel and
the dequant path in the attention kernels; the format only changes payload layout and
the divisor mapping a block's extreme onto its representable range. The scale varies
along ``head_dim``, the reduction dimension of ``q @ k``, so attention dequantizes
before the dot. This saves storage bandwidth, not tensor-core compute.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Elements per scale, along head_dim. Matches GGUF Q8_0's block.
BLOCK = 32
# One fp16 scale per block.
SCALE_DTYPE = torch.float16


@dataclass(frozen=True)
class KVQuantSpec:
    """How a KV pool stores its K/V elements.

    ``name`` is the ``--kv-cache-dtype`` value. ``storage_dtype`` is None for the
    unquantized pool, in which case the pool allocates in the compute dtype and no
    scale tensor exists.

    ``bits`` is the payload width. torch has no sub-byte integer dtypes, so formats
    below eight bits use a byte slab whose last dimension is ``D * bits // 8``.
    """

    name: str
    storage_dtype: torch.dtype | None
    # Max-abs of a block maps to this magnitude in the storage dtype.
    max_magnitude: float
    bits: int = 8

    @property
    def enabled(self) -> bool:
        return self.storage_dtype is not None

    @property
    def is_integer(self) -> bool:
        """Integer schemes round; float ones just divide."""
        return self.storage_dtype in (torch.int8, torch.uint8)

    @property
    def packed(self) -> bool:
        """True when the slab packs sub-byte integer values."""
        return self.bits < 8

    @property
    def elements_per_byte(self) -> int:
        """Compatibility helper for the byte-aligned q8/q4 formats.

        Q5/Q6 have fractional byte ratios and therefore no integral elements-per-byte
        ratio. New code should use :meth:`storage_dim` and :meth:`logical_dim`; asking
        for this legacy ratio on either format is an error.
        """
        if 8 % self.bits:
            raise ValueError(f"{self.name} has a fractional elements-per-byte ratio")
        return 8 // self.bits

    def bytes_per_element(self, compute_dtype: torch.dtype) -> float:
        """Storage bytes per K/V element, scales amortized over the block.

        Unquantized: the compute dtype's itemsize. 8-bit: 1 byte + 2/32 for the fp16
        scale = 1.0625. int4: half a byte + 2/32 = 0.5625.
        """
        if not self.enabled:
            return float(compute_dtype.itemsize)
        return self.bits / 8.0 + SCALE_DTYPE.itemsize / BLOCK

    def storage_dim(self, logical_dim: int) -> int:
        """Physical bytes required for one logical head."""
        payload_bits = logical_dim * self.bits
        if payload_bits % 8:
            raise ValueError(
                f"head_dim {logical_dim} does not produce whole bytes for {self.name}"
            )
        return payload_bits // 8

    def logical_dim(self, storage_dim: int) -> int:
        """Logical elements represented by a physical byte extent."""
        payload_bits = storage_dim * 8
        if payload_bits % self.bits:
            raise ValueError(
                f"storage dim {storage_dim} does not represent whole {self.name} values"
            )
        return payload_bits // self.bits

    def storage_shape(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        """Element-storage slab shape for a logical KV shape (last dim halves when packed)."""
        if self.packed:
            return (*shape[:-1], self.storage_dim(shape[-1]))
        return shape

    def scale_shape(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        """Scale-tensor shape for a *logical* KV shape: last dim divided by the block.

        ``shape`` is the element-counted (unpacked) KV geometry, so the scale extent is
        the same whether the slab is 8-bit (element-shaped) or int4 (byte-packed).
        """
        if shape[-1] % BLOCK:
            raise ValueError(
                f"head_dim {shape[-1]} is not a multiple of the KV quant block {BLOCK}"
            )
        return (*shape[:-1], shape[-1] // BLOCK)

    # ---- reference implementations (correctness oracle for the Triton kernels) ----

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``x[..., D]`` (float) -> ``(quantized[..., D // epb], scales[..., D // BLOCK])``.

        The quantized tensor has the *storage* shape: element-shaped for 8-bit,
        byte-packed (last dim halves) for int4.
        """
        assert self.enabled, "quantize() on an unquantized spec"
        blocks = x.float().unflatten(-1, (x.shape[-1] // BLOCK, BLOCK))
        if self.bits == 4:
            # Match GGML Q4_0. The signed value with the greatest magnitude selects a
            # possibly-negative scale; codes 0..15 then represent (code - 8) * scale.
            # This uses all 16 nibble values and preserves the block's largest-magnitude
            # element exactly, unlike the former symmetric [-7, 7] scheme.
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
            # GGML Q5_0 scale/codes with a cache-native plane layout: adjacent low
            # nibbles first, then one packed high-bit byte per eight logical values.
            # Keeping both planes contiguous avoids GGML's per-block metadata stride
            # in the attention hot path.
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
            # Use GGML Q6_0's quantizer with a cache-native plane layout: all adjacent
            # low-nibble pairs first, then all adjacent upper-two-bit quads. Keeping
            # each plane contiguous lets attention unpack it without integer divides.
            abs_blocks = blocks.abs()
            extreme = blocks.gather(
                -1, abs_blocks.argmax(dim=-1, keepdim=True)
            ).squeeze(-1)
            initial = torch.where(extreme != 0, extreme / -32.0, torch.ones_like(extreme))
            codes = torch.floor(blocks / initial.unsqueeze(-1) + 32.5).clamp_(0, 63)

            # GGML Q6_0 refines the scale after selecting codes. Preserve that useful
            # quality detail; codes stay fixed and only the stored delta changes.
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
        # Round the scale to its stored precision BEFORE dividing, so quantize and
        # dequantize use the identical value.
        scales = scales.to(SCALE_DTYPE)
        q = blocks / scales.float().unsqueeze(-1)
        if self.is_integer:
            # Half away from zero, matching the store kernel. ``Tensor.round`` is
            # half-to-even and would disagree on ties.
            q = torch.where(q >= 0, (q + 0.5).floor(), (q - 0.5).ceil())
            q = q.clamp_(-self.max_magnitude, self.max_magnitude)
        return q.flatten(-2).to(self.storage_dtype), scales

    def dequantize(self, q: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`quantize`, in float32 (logical element shape)."""
        assert self.enabled, "dequantize() on an unquantized spec"
        if self.bits == 4:
            logical_d = self.logical_dim(q.shape[-1])
            nblock = logical_d // BLOCK
            # Each block of BLOCK elements occupies BLOCK // elements_per_byte bytes;
            # split each byte's low (even element) and high (odd) nibbles. Operate on
            # the integer codes (the caller may pass the packed tensor already floated).
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


# int8 symmetric: a block's max-abs maps to 127.
Q8_0 = KVQuantSpec(name="q8_0", storage_dtype=torch.int8, max_magnitude=127.0)
# e4m3: 4-bit exponent, 3-bit mantissa, max finite magnitude 448.
FP8_E4M3 = KVQuantSpec(name="fp8_e4m3", storage_dtype=torch.float8_e4m3fn, max_magnitude=448.0)
# GGML Q4_0, two values per byte: a block's signed extreme selects a scale and all
# 16 codes represent [-8, 7] times that (possibly negative) scale.
INT4 = KVQuantSpec(
    name="int4", storage_dtype=torch.uint8, max_magnitude=8.0, bits=4
)
Q6_0 = KVQuantSpec(name="q6_0", storage_dtype=torch.uint8, max_magnitude=32.0, bits=6)
Q5_0 = KVQuantSpec(name="q5_0", storage_dtype=torch.uint8, max_magnitude=16.0, bits=5)
NONE = KVQuantSpec(name="auto", storage_dtype=None, max_magnitude=0.0)

_BY_NAME = {spec.name: spec for spec in (NONE, Q8_0, FP8_E4M3, INT4, Q5_0, Q6_0)}
_BY_NAME["q4_0"] = INT4
KV_CACHE_DTYPES = tuple(_BY_NAME)


def resolve_kv_quant(name: str | None) -> KVQuantSpec:
    """``--kv-cache-dtype`` value -> spec. ``None``/``"auto"`` means unquantized."""
    if name is None:
        return NONE
    try:
        return _BY_NAME[name]
    except KeyError:
        raise ValueError(
            f"unknown --kv-cache-dtype {name!r}; choose from {', '.join(KV_CACHE_DTYPES)}"
        ) from None


__all__ = [
    "BLOCK",
    "SCALE_DTYPE",
    "KVQuantSpec",
    "KV_CACHE_DTYPES",
    "Q8_0",
    "FP8_E4M3",
    "INT4",
    "Q5_0",
    "Q6_0",
    "NONE",
    "resolve_kv_quant",
]
