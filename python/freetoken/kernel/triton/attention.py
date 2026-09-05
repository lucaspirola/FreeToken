from __future__ import annotations

import functools
import os

import torch
import triton
import triton.language as tl


_MAX_KV_SPLITS = 8
_MIN_BLOCK_KV = 32

# Grid-filling decode fallback. Stage 1 launches ``batch * head_blocks * kv_splits``
# CTAs and ``head_blocks`` is fixed by the head geometry, so ``kv_splits`` is the only
# term that scales the decode grid with the GPU. One CTA per SM at batch one is what the
# 2026-09-04 RTX 5080 sweep measured (benchmarks/bench_decode_launch.py): for 32Q/2KV/D128
# at 131K-1M, 64 splits (128 CTAs on 84 SMs) beat both 32 and 128 at every length.
# ``_MAX_AUTO_KV_SPLITS`` caps the stage-2 reduction and the fp32 scratch.
_DECODE_CTAS_PER_SM = 1
_MAX_AUTO_KV_SPLITS = 128

# The cache-native Q8 score path is independently switchable so its numerical and
# performance gates can be compared against the dequantize-to-BF16 implementation.
_Q8_NATIVE_QK = os.getenv("FREETOKEN_Q8_NATIVE_QK", "1").strip() != "0"


@functools.lru_cache(maxsize=1)
def _decode_launch_env_override() -> tuple[int | None, int | None, int | None]:
    """``(kv_splits, block_n, num_warps)`` forced by the environment, or ``(None,)*3``.

    Exists so a launch configuration can be A/B'd end to end against a live server
    (``FREETOKEN_DECODE_KV_SPLITS=8`` reproduces the pre-2026-09-04 fallback) without
    rebuilding: the split count is baked into the CUDA-graph grid and the fp32 scratch
    at capture time, so it cannot be varied inside one process.
    """
    def _get(name: str) -> int | None:
        raw = os.getenv(name, "").strip()
        return int(raw) if raw else None

    return (
        _get("FREETOKEN_DECODE_KV_SPLITS"),
        _get("FREETOKEN_DECODE_BLOCK_N"),
        _get("FREETOKEN_DECODE_NUM_WARPS"),
    )


def _decode_head_blocks(num_q_heads: int, num_kv_heads: int) -> int:
    """Stage-1 grid extent along the head axis -- ``cdiv(num_q_heads, valid_block_h)``.

    Mirrors ``decode_paged_attention``'s ``valid_block_h = min(16, group)`` so the
    launch heuristic reasons about the same grid the kernel is actually given. It is 2
    for both tuned geometries here (16Q/2KV and 32Q/2KV), which is exactly why the
    split count is the only free parameter left to fill the GPU with.
    """
    group = max(1, num_q_heads // max(1, num_kv_heads))
    return -(-num_q_heads // min(16, group))


def _grid_filling_splits(*, num_q_heads: int, num_kv_heads: int, sm_count: int) -> int:
    """Split count that keeps every SM busy for a geometry with no measured tuning.

    The historical fallback was a flat 8 splits. On a 32Q/2KV head shape that is a
    16-CTA grid on an 84-SM part, with every CTA walking ``seq_len / 8`` tokens
    serially -- so single-stream decode slows down roughly linearly with context
    (Nemotron 3.5 Lightning: 72 tok/s at 131K, 32 at 524K) while 5/6 of the GPU idles.
    Splitting the KV axis further is close to free: stage 1's total work is
    ``batch * heads * seq_len`` regardless, empty splits exit before loading anything,
    and only the stage-2 reduction (``splits`` fp32 rows per head) grows.
    """
    head_blocks = _decode_head_blocks(num_q_heads, num_kv_heads)
    target = -(-(sm_count * _DECODE_CTAS_PER_SM) // head_blocks)
    splits = 1 << max(0, (target - 1).bit_length())
    return max(_MAX_KV_SPLITS, min(splits, _MAX_AUTO_KV_SPLITS))


def decode_launch_config(
    *,
    quant_name: str | None,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    compute_capability: tuple[int, int] | None = None,
    sm_count: int | None = None,
) -> tuple[int, int, int]:
    """Return ``(kv_splits, block_n, num_warps)`` for grouped decode attention.

    Quantized caches are dequantized before each tensor-core dot and need more
    parallel KV partitions than the bf16 path. The Ornith/Qwen3.5 geometry has
    measured launch configurations for consumer Ada and Blackwell; any other shape
    falls back to ``_grid_filling_splits`` when the caller knows the GPU's SM count,
    and to the historical conservative constant when it does not.
    """
    splits, block_n, num_warps = _tuned_decode_launch_config(
        quant_name=quant_name,
        head_dim=head_dim,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        compute_capability=compute_capability,
        sm_count=sm_count,
    )
    env_splits, env_block_n, env_warps = _decode_launch_env_override()
    return (
        env_splits or splits,
        env_block_n or block_n,
        env_warps or num_warps,
    )


def _tuned_decode_launch_config(
    *,
    quant_name: str | None,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    compute_capability: tuple[int, int] | None,
    sm_count: int | None,
) -> tuple[int, int, int]:
    if (
        quant_name == "q8_q6"
        and head_dim == 256
        and num_q_heads == 16
        and num_kv_heads == 2
        and compute_capability is not None
        and compute_capability >= (12, 0)
    ):
        # Mixed formats increase register pressure. At 262K on RTX 5080, 32-token
        # tiles / 8 warps cut the Q6-value attention toll by 25% versus Q8's 64/4
        # launch (0.492 vs 0.655 ms per layer); 128 splits regressed.
        return 64, 32, 8
    if (
        quant_name == "q6_q5"
        and head_dim == 256
        and num_q_heads == 16
        and num_kv_heads == 2
        and compute_capability is not None
        and compute_capability >= (12, 0)
    ):
        # The denser Q6-K/Q5-V unpack needs more independent partitions but fewer
        # warps than Q8-K/Q6-V. At 262K on RTX 5080, 128/32/4 reached 0.480 ms per
        # layer versus 0.546 for the inherited 64/32/8 launch and 0.494 for Q8/Q6.
        return 128, 32, 4
    if (
        quant_name == "int4"
        and head_dim == 256
        and num_q_heads == 16
        and num_kv_heads == 2
    ):
        if compute_capability is not None and compute_capability >= (12, 0):
            # RTX 5080 / Triton 3.6: 64 splits and 64-token tiles are 57% faster
            # than the sm_89 launch at 262K (0.356 vs 0.822 ms per layer).
            return 64, 64, 8
        # BLOCK_N=16 is not safe for the packed-byte loader at this geometry on
        # Triton 3.6/sm_89; it silently corrupts attention output. BLOCK_N=32 has
        # a numerical regression test and is also the fastest correct 200K launch.
        return 32, 32, 4
    if (
        quant_name in {"q8_0", "quant8"}
        and head_dim == 256
        and num_q_heads == 16
        and num_kv_heads == 2
    ):
        # The backend sees the public name (q8_0) while the low-level kernel
        # infers quant8 from the cache tensors. Accept both so CUDA-graph scratch
        # is allocated for all 64 splits and the kernel can actually select them.
        if compute_capability == (8, 9):
            # RTX 2000 Ada sweep: 16 splits is 2-5% faster than 64 at
            # 50K/254K and statistically tied at 170K. The 64-token tile and
            # four-warps geometry remain the fastest correct configuration.
            return 16, 64, 4
        return 64, 64, 4
    if sm_count:
        # Untuned geometry on a known GPU (Nemotron 3.5 Lightning's 32Q/2KV/D128 is the
        # motivating one): size the grid to the part instead of to a constant. The tile
        # follows head_dim -- at D128 the 64-token tile with 8 warps was fastest at every
        # length measured (0.117/0.213/0.402/0.790 ms per layer at 131K/262K/524K/1M),
        # while at D256 the same tile doubles the register/shared footprint and the
        # 32-token/4-warp launch wins (bf16 Ornith 16Q/2KV/D256: 0.312 vs 0.332 ms).
        block_n, num_warps = (64, 8) if head_dim <= 128 else (32, 4)
        return (
            _grid_filling_splits(
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                sm_count=sm_count,
            ),
            block_n,
            num_warps,
        )
    return _MAX_KV_SPLITS, 32, 4


def decode_runtime_splits(
    *,
    preferred_splits: int,
    scratch_splits: int,
    batch: int,
    quant_name: str | None,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    compute_capability: tuple[int, int] | None,
) -> int:
    """Select the realized split count once decode batch size is known.

    The RTX 2000 Ada Ornith sweep shows that batch two already supplies enough
    independent stage-1 blocks at 16 splits; keeping 32 adds reduction work and
    is about 9% slower at 262K. Batch one and batch four both remain fastest at
    32. Keep this deliberately narrow so unrelated GQA shapes and Blackwell's
    separately tuned launch are unaffected.
    """
    splits = min(preferred_splits, scratch_splits)
    if (
        compute_capability == (8, 9)
        and batch == 2
        and quant_name == "int4"
        and head_dim == 256
        and num_q_heads == 16
        and num_kv_heads == 2
    ):
        splits = min(splits, 16)
    return max(splits, 1)


@triton.jit
def _load_kv(
    ptr,
    scale_ptr,
    base,  # slot/head byte base (already carries the broadcast), per-row/per-col
    elem,  # within-head element index (the heads dim), broadcast to match ``base``
    scale_offsets,
    mask,
    scale_mask,
    out_dtype: tl.constexpr,
    FORMAT: tl.constexpr,
    QBLOCK: tl.constexpr,
    D_ON_ROWS: tl.constexpr,
    PACKED_PRESCALE: tl.constexpr,
):
    """Load a K or V tile, dequantizing it when the pool stores quantized values.

    ``base + elem`` addresses the tile in the KV buffer; ``scale_offsets`` addresses the
    matching scales, whose ``head_dim`` extent is ``QBLOCK`` times smaller -- one scale
    per block, loaded once and broadcast across the block rather than re-read per
    element.

    ``base`` is kept separate from ``elem`` because packed storage changes only the
    within-head addressing; slot/head strides in ``base`` are already byte counts.

    ``D_ON_ROWS`` says which way the tile is laid out: the dot-product kernels want K as
    ``[D, N]`` and V as ``[N, D]``, and the block axis has to be expanded along whichever
    one carries ``head_dim``.

    The scale varies along ``head_dim``, the reduction dimension of ``q @ k``, so it
    cannot be folded in after the dot -- the tile is dequantized into ``out_dtype`` (the
    query's dtype) and fed to the tensor cores like the bf16 path does.
    """
    if FORMAT == 4:
        # Cache-native Q5 planes: D/2 adjacent low-nibble pairs followed by D/8
        # bytes carrying the high bit of eight consecutive logical values.
        if D_ON_ROWS:
            q5_nb: tl.constexpr = scale_offsets.shape[0]
            q5_n: tl.constexpr = scale_offsets.shape[1]
            q5_d: tl.constexpr = q5_nb * QBLOCK
            low_elem = tl.arange(0, q5_d // 2)[:, None]
            high_elem = q5_d // 2 + tl.arange(0, q5_d // 8)[:, None]
            low_mask = tl.broadcast_to(
                scale_mask[:, None, :], (q5_nb, QBLOCK // 2, q5_n)
            ).reshape(q5_d // 2, q5_n)
            high_mask = tl.broadcast_to(
                scale_mask[:, None, :], (q5_nb, QBLOCK // 8, q5_n)
            ).reshape(q5_d // 8, q5_n)
        else:
            q5_n: tl.constexpr = scale_offsets.shape[0]
            q5_nb: tl.constexpr = scale_offsets.shape[1]
            q5_d: tl.constexpr = q5_nb * QBLOCK
            low_elem = tl.arange(0, q5_d // 2)[None, :]
            high_elem = q5_d // 2 + tl.arange(0, q5_d // 8)[None, :]
            low_mask = tl.broadcast_to(
                scale_mask[:, :, None], (q5_n, q5_nb, QBLOCK // 2)
            ).reshape(q5_n, q5_d // 2)
            high_mask = tl.broadcast_to(
                scale_mask[:, :, None], (q5_n, q5_nb, QBLOCK // 8)
            ).reshape(q5_n, q5_d // 8)
        low = tl.load(ptr + base + low_elem, mask=low_mask, other=0)
        high = tl.load(ptr + base + high_elem, mask=high_mask, other=0)
        if D_ON_ROWS:
            lower = tl.interleave((low & 15).trans(), (low >> 4).trans()).trans()
            high_t = high.trans()
            planes = tl.broadcast_to(high_t[:, :, None], (q5_n, q5_d // 8, 8))
            lane = tl.arange(0, 8)[None, None, :]
            upper = ((planes >> lane) & 1).reshape(q5_n, q5_d).trans()
        else:
            lower = tl.interleave(low & 15, low >> 4)
            planes = tl.broadcast_to(high[:, :, None], (q5_n, q5_d // 8, 8))
            lane = tl.arange(0, 8)[None, None, :]
            upper = ((planes >> lane) & 1).reshape(q5_n, q5_d)
        vals = ((lower | (upper << 4)).to(tl.float32) - 16.0)
        scale = tl.load(scale_ptr + scale_offsets, mask=scale_mask, other=0.0)
        if D_ON_ROWS:
            wide = tl.broadcast_to(
                scale[:, None, :], (q5_nb, QBLOCK, q5_n)
            ).reshape(q5_d, q5_n)
        else:
            wide = tl.broadcast_to(
                scale[:, :, None], (q5_n, q5_nb, QBLOCK)
            ).reshape(q5_n, q5_d)
        return (vals * wide.to(tl.float32)).to(out_dtype)

    if FORMAT == 3:
        # Cache-native Q6 planes: D/2 adjacent low-nibble pairs, then D/4 adjacent
        # upper-two-bit quads. Load every payload byte once and unpack in registers.
        if D_ON_ROWS:
            q6_nb: tl.constexpr = scale_offsets.shape[0]
            q6_n: tl.constexpr = scale_offsets.shape[1]
            q6_d: tl.constexpr = q6_nb * QBLOCK
            low_elem = tl.arange(0, q6_d // 2)[:, None]
            high_elem = q6_d // 2 + tl.arange(0, q6_d // 4)[:, None]
            low_mask = tl.broadcast_to(
                scale_mask[:, None, :], (q6_nb, QBLOCK // 2, q6_n)
            ).reshape(q6_d // 2, q6_n)
            high_mask = tl.broadcast_to(
                scale_mask[:, None, :], (q6_nb, QBLOCK // 4, q6_n)
            ).reshape(q6_d // 4, q6_n)
        else:
            q6_n: tl.constexpr = scale_offsets.shape[0]
            q6_nb: tl.constexpr = scale_offsets.shape[1]
            q6_d: tl.constexpr = q6_nb * QBLOCK
            low_elem = tl.arange(0, q6_d // 2)[None, :]
            high_elem = q6_d // 2 + tl.arange(0, q6_d // 4)[None, :]
            low_mask = tl.broadcast_to(
                scale_mask[:, :, None], (q6_n, q6_nb, QBLOCK // 2)
            ).reshape(q6_n, q6_d // 2)
            high_mask = tl.broadcast_to(
                scale_mask[:, :, None], (q6_n, q6_nb, QBLOCK // 4)
            ).reshape(q6_n, q6_d // 4)
        low = tl.load(ptr + base + low_elem, mask=low_mask, other=0)
        high = tl.load(ptr + base + high_elem, mask=high_mask, other=0)
        if D_ON_ROWS:
            lower = tl.interleave((low & 15).trans(), (low >> 4).trans()).trans()
            high_t = high.trans()
            planes = tl.broadcast_to(
                high_t[:, :, None], (q6_n, q6_d // 4, 4)
            )
            lane = tl.arange(0, 4)[None, None, :]
            upper = ((planes >> (lane * 2)) & 3).reshape(q6_n, q6_d).trans()
        else:
            lower = tl.interleave(low & 15, low >> 4)
            planes = tl.broadcast_to(
                high[:, :, None], (q6_n, q6_d // 4, 4)
            )
            lane = tl.arange(0, 4)[None, None, :]
            upper = ((planes >> (lane * 2)) & 3).reshape(q6_n, q6_d)
        vals = ((lower | (upper << 4)).to(tl.float32) - 32.0)
        scale = tl.load(scale_ptr + scale_offsets, mask=scale_mask, other=0.0)
        if D_ON_ROWS:
            wide = tl.broadcast_to(
                scale[:, None, :], (q6_nb, QBLOCK, q6_n)
            ).reshape(q6_d, q6_n)
        else:
            wide = tl.broadcast_to(
                scale[:, :, None], (q6_n, q6_nb, QBLOCK)
            ).reshape(q6_n, q6_d)
        return (vals * wide.to(tl.float32)).to(out_dtype)

    if FORMAT == 2:
        # Nibble-packed GGML Q4_0: uint8 byte per element pair; low nibble = even
        # element, high nibble = odd element, and value = (nibble - 8) * signed scale.
        # Build a byte-sized tile and interleave its nibbles after the load.  The old
        # logical-element tile addressed ``elem // 2`` and therefore fetched every byte
        # twice.  Keeping packed space until unpack halves KV load instructions/traffic.
        if D_ON_ROWS:
            nb_p: tl.constexpr = scale_offsets.shape[0]
            n_p: tl.constexpr = scale_offsets.shape[1]
            packed_d: tl.constexpr = nb_p * QBLOCK // 2
            packed_elem = tl.arange(0, packed_d)[:, None]
            packed_mask = tl.broadcast_to(
                scale_mask[:, None, :], (nb_p, QBLOCK // 2, n_p)
            ).reshape(packed_d, n_p)
        else:
            n_p: tl.constexpr = scale_offsets.shape[0]
            nb_p: tl.constexpr = scale_offsets.shape[1]
            packed_d: tl.constexpr = nb_p * QBLOCK // 2
            packed_elem = tl.arange(0, packed_d)[None, :]
            packed_mask = tl.broadcast_to(
                scale_mask[:, :, None], (n_p, nb_p, QBLOCK // 2)
            ).reshape(n_p, packed_d)
        packed = tl.load(ptr + base + packed_elem, mask=packed_mask, other=0)
        # Triton 3.6 lowers the integer bit operations directly; unlike modulo and
        # division they do not introduce integer arithmetic on every KV element.
        lo = (packed & 15).to(tl.float32)
        hi = (packed >> 4).to(tl.float32)
        scale = tl.load(scale_ptr + scale_offsets, mask=scale_mask, other=0.0)
        if PACKED_PRESCALE:
            # Decode/paged attention benefits from applying one scale to each
            # packed 16-byte Q4_0 group before nibble interleave. Long-prefix
            # extend is register-bound and uses the post-interleave form below.
            if D_ON_ROWS:
                packed_scale = tl.broadcast_to(
                    scale[:, None, :], (nb_p, QBLOCK // 2, n_p)
                ).reshape(packed_d, n_p)
                lo = (lo - 8.0) * packed_scale
                hi = (hi - 8.0) * packed_scale
                vals = tl.interleave(lo.trans(), hi.trans()).trans()
            else:
                packed_scale = tl.broadcast_to(
                    scale[:, :, None], (n_p, nb_p, QBLOCK // 2)
                ).reshape(n_p, packed_d)
                lo = (lo - 8.0) * packed_scale
                hi = (hi - 8.0) * packed_scale
                vals = tl.interleave(lo, hi)
        else:
            if D_ON_ROWS:
                vals = tl.interleave(lo.trans(), hi.trans()).trans()
            else:
                vals = tl.interleave(lo, hi)
        # Masked lanes must read as 0 (like the ``other=0.0`` loads in the element path),
        # not as the -7 bias, or they would poison the dot when head_dim is not a power
        # of two and BLOCK_D probes past D.
        if D_ON_ROWS:
            logical_mask = tl.broadcast_to(
                scale_mask[:, None, :], (nb_p, QBLOCK, n_p)
            ).reshape(nb_p * QBLOCK, n_p)
        else:
            logical_mask = tl.broadcast_to(
                scale_mask[:, :, None], (n_p, nb_p, QBLOCK)
            ).reshape(n_p, nb_p * QBLOCK)
        if PACKED_PRESCALE:
            return tl.where(logical_mask, vals, 0.0).to(out_dtype)
        vals = tl.where(logical_mask, vals - 8.0, 0.0)
        if D_ON_ROWS:
            wide = tl.broadcast_to(scale[:, None, :], (nb_p, QBLOCK, n_p)).reshape(
                nb_p * QBLOCK, n_p
            )
        else:
            wide = tl.broadcast_to(scale[:, :, None], (n_p, nb_p, QBLOCK)).reshape(
                n_p, nb_p * QBLOCK
            )
        return (vals * wide.to(tl.float32)).to(out_dtype)

    vals = tl.load(ptr + base + elem, mask=mask, other=0.0)
    if FORMAT != 0:
        scale = tl.load(scale_ptr + scale_offsets, mask=scale_mask, other=0.0)
        if D_ON_ROWS:
            nb: tl.constexpr = scale.shape[0]
            n: tl.constexpr = scale.shape[1]
            wide = tl.broadcast_to(scale[:, None, :], (nb, QBLOCK, n)).reshape(nb * QBLOCK, n)
        else:
            n: tl.constexpr = scale.shape[0]
            nb: tl.constexpr = scale.shape[1]
            wide = tl.broadcast_to(scale[:, :, None], (n, nb, QBLOCK)).reshape(n, nb * QBLOCK)
        return (vals.to(tl.float32) * wide.to(tl.float32)).to(out_dtype)
    # Both branches must yield the same type for Triton to compile the function, so the
    # unquantized path casts too. Callers pass the dtype the tile already has there
    # (float32 for the fp32 kernel, the cache's own dtype elsewhere), so it is a no-op.
    return vals.to(out_dtype)


def _slot_offsets_need_int64(*pools) -> bool:
    """True when ``slot_id * stride`` can overflow int32 for any of these pools.

    Triton evaluates ``ptr + offset`` in the offset's own width, so the 32-bit
    ``slots * stride_ks`` in the KV gathers wraps once a pool holds 2**31 elements or
    more. That is 8.4M slots at Nemotron's 2 KV heads x 128 dim, but only 1.05M at
    8 heads x 256 -- inside the 1M-token profile. The KV *store* path already widens
    unconditionally (``kv_quant.py``); the load path pays 64-bit address math, so it
    widens only when the pool is actually that large.
    """
    return any(pool is not None and pool.numel() >= 2**31 for pool in pools)


@functools.cache
def _sm_count(device_index: int) -> int:
    """Streaming-multiprocessor count of a CUDA device (0 if unavailable)."""
    props = torch.cuda.get_device_properties(device_index)
    return int(getattr(props, "multi_processor_count", 0))


@functools.lru_cache(maxsize=None)
def _optin_smem_bytes(device_index: int) -> int:
    """Per-block opt-in shared-memory budget for a CUDA device (0 if unavailable)."""
    props = torch.cuda.get_device_properties(device_index)
    return int(getattr(props, "shared_memory_per_block_optin", 0))


@functools.lru_cache(maxsize=1)
def _extend_launch_env_override() -> tuple[int | None, int | None, int | None, int | None]:
    """``(block_m, block_n, num_warps, num_stages)`` forced by the environment.

    The extend/prefill launch is picked from ``head_dim`` and the device's opt-in
    shared memory alone (:func:`_select_extend_tile`, :func:`extend_launch_config`),
    so an end-to-end A/B of a launch change would otherwise need a rebuild. The
    decode path has had ``FREETOKEN_DECODE_*`` since 2026-09-04; these are the
    prefill twins. Unset entries keep the computed value.
    """

    def _get(name: str) -> int | None:
        raw = os.getenv(name, "").strip()
        return int(raw) if raw else None

    return (
        _get("FREETOKEN_EXTEND_BLOCK_M"),
        _get("FREETOKEN_EXTEND_BLOCK_N"),
        _get("FREETOKEN_EXTEND_NUM_WARPS"),
        _get("FREETOKEN_EXTEND_NUM_STAGES"),
    )


def _select_extend_tile(head_dim: int, block_d: int, smem_optin: int) -> tuple[int, int]:
    """Pick ``(BLOCK_M, BLOCK_N)`` for the extend/prefill kernel, shared-memory aware.

    Larger tiles run materially faster (~2x for head_dim 512 on H100) but their bf16
    q/k/v tiles need about ``(BLOCK_M + 2 * BLOCK_N) * BLOCK_D * 2`` bytes of shared
    memory, which overflows consumer GPUs (sm_89 ~99KB opt-in) once head_dim >= 256.
    Keep the fast tiles where the device's opt-in shared memory fits them (datacenter
    A100/H100); shrink only where it does not. ``smem_optin == 0`` (unknown) conservatively
    selects the small tiles, i.e. the prior consumer-safe behavior.
    """
    budget = smem_optin * 0.8  # headroom for scores/acc/alignment/triton scratch

    def fits(block_m: int, block_n: int) -> bool:
        return (block_m + 2 * block_n) * block_d * 2 <= budget

    if head_dim <= 128:
        return 128, 64
    if head_dim <= 256:
        return (128, 64) if fits(128, 64) else (64, 32)
    if head_dim <= 384:
        return (32, 64) if fits(32, 64) else (32, 32)
    return (32, 64) if fits(32, 64) else (16, 16)


# The extend kernel's fp32 accumulator is `BLOCK_M x BLOCK_DV`, spread over
# `32 * num_warps` lanes. Past ~64 registers per thread for the accumulator alone
# Triton spills it to local memory and the kernel collapses -- measured 2.46x on an
# RTX 5080 at 32Q/2KV/D128 (see benchmarks/results/
# nemotron35_lightning_5080_prefill_profile_2026-09-05.md).
_EXTEND_ACC_REGS = 64


def _extend_block_m_cap(block_dv: int, num_warps: int) -> int:
    """Largest ``BLOCK_M`` whose fp32 accumulator stays inside the register budget.

    Derived from the kernel's own accumulator shape rather than from a measured
    constant, so it follows the warp count instead of pinning one head shape:
    ``acc`` is ``BLOCK_M x BLOCK_DV`` fp32 over ``32 * num_warps`` lanes.
    """
    return max(16, _EXTEND_ACC_REGS * 32 * num_warps // block_dv)


def extend_launch_config(
    *,
    head_dim: int,
    block_d: int,
    smem_optin: int,
    capability: tuple[int, int],
    k_format: int = 0,
    v_format: int = 0,
) -> tuple[int, int, int, int]:
    """``(block_m, block_n, num_warps, num_stages)`` for the extend/prefill kernels.

    Split out of :func:`extend_paged_attention` so the choice is testable and
    overridable (``FREETOKEN_EXTEND_*``) rather than inline in the launch.
    """
    block_m, block_n = _select_extend_tile(head_dim, block_d, smem_optin)
    # On sm_89 the consumer-safe 64x32 tile for D=256 still has room for a
    # second software-pipeline stage.  It halves cold-chunk time (68 -> 35 ms
    # per Ornith full-attention layer at 8K) and remains ~10% faster once an
    # 8K quantized prefix is present. Larger-D fallback tiles stay at one stage.
    num_stages = 2 if (head_dim, block_m, block_n) == (256, 64, 32) else 1
    # Swept on RTX 5080 (sm_120): 4 warps beat 8 at the consumer (64, 32) D=256
    # tile for both extend kernels (2.01x cold prefill, 1.12x long-Q4-prefix
    # extension); sm_89 measured faster with 8. BLOCK_N=16 corrupts the packed
    # loader in the extend kernels too and must never be selected here.
    num_warps = 4 if capability >= (12, 0) else 8
    if (
        (k_format, v_format) == (3, 4)
        and capability >= (12, 0)
        and (head_dim, block_m, block_n) == (256, 64, 32)
    ):
        # Q6/Q5 benefits from the extra unpacking lanes: 8 warps / 2 stages cut the
        # 8K-prefix + 2K-chunk kernel from 12.31 to 9.69 ms on RTX 5080. BLOCK_N=16
        # failed the numerical oracle and 64 overflowed consumer Blackwell shared memory.
        num_warps = 8
        num_stages = 2
    if head_dim <= 128:
        # The <=128 arm of _select_extend_tile returns a hard-coded 128-row tile that
        # was never measured against the warp count chosen just above. On sm_120 that
        # is 4 warps, i.e. 128 accumulator registers per thread and a spilling kernel:
        # 29.3 TFLOP/s against 70.4 for the 64-row tile at a 131K prefix, and the whole
        # 2.46x of the 2026-09-05 prefill profile. Devices that take 8 warps still
        # admit BLOCK_M=128 and are left exactly as they were. Head dims above 128 keep
        # their measured tiles (D=256 consumer 64x32, gemma4's D>=384 fallbacks).
        block_m = min(block_m, _extend_block_m_cap(block_d, num_warps))
    env_m, env_n, env_warps, env_stages = _extend_launch_env_override()
    return (
        env_m or block_m,
        env_n or block_n,
        env_warps or num_warps,
        env_stages or num_stages,
    )


@triton.jit
def _paged_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    ks_ptr,
    vs_ptr,
    o_ptr,
    indptr_ptr,
    indices_ptr,
    q_to_req_ptr,
    q_pos_ptr,
    sm_scale,
    sinks_ptr,
    stride_qt,
    stride_qh,
    stride_ks,
    stride_kh,
    stride_vs,
    stride_vh,
    stride_kss,
    stride_ksh,
    stride_vss,
    stride_vsh,
    stride_ot,
    stride_oh,
    GROUP: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    HAS_SINKS: tl.constexpr,
    K_FORMAT: tl.constexpr,
    V_FORMAT: tl.constexpr,
    QBLOCK: tl.constexpr,
    SLOT_I64: tl.constexpr,
):
    q_tok = tl.program_id(0)
    q_head = tl.program_id(1)
    kv_head = q_head // GROUP

    req = tl.load(q_to_req_ptr + q_tok)
    kv_start = tl.load(indptr_ptr + req)
    kv_end = tl.load(indptr_ptr + req + 1)
    kv_len = kv_end - kv_start
    q_pos = tl.load(q_pos_ptr + q_tok)

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    # One scale per QBLOCK elements of head_dim: the tile's block axis.
    offs_nb = tl.arange(0, BLOCK_D // QBLOCK)
    mask_nb = offs_nb < D // QBLOCK
    q = tl.load(
        q_ptr + q_tok * stride_qt + q_head * stride_qh + offs_d,
        mask=mask_d,
        other=0.0,
    ).to(tl.float32)

    if HAS_SINKS:
        m_i = tl.load(sinks_ptr + q_head).to(tl.float32)
        l_i = 1.0
    else:
        m_i = -float("inf")
        l_i = 0.0
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for start in range(0, kv_len, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < kv_len
        k_pos = offs_n
        causal_mask = k_pos <= q_pos
        if SLIDING_WINDOW > 0:
            causal_mask = causal_mask & ((k_pos + SLIDING_WINDOW) > q_pos)
        mask_n = mask_n & causal_mask

        skip_tile = tl.max(mask_n.to(tl.int32), axis=0) == 0
        if not skip_tile:
            slots = tl.load(indices_ptr + kv_start + offs_n, mask=offs_n < kv_len, other=0)
            if SLOT_I64:
                # Pools above 2**31 elements overflow a 32-bit slot*stride offset.
                slots = slots.to(tl.int64)
            kv_mask = (offs_n[:, None] < kv_len) & mask_d[None, :]
            kv_scale_mask = (offs_n[:, None] < kv_len) & mask_nb[None, :]
            k = _load_kv(
                k_ptr,
                ks_ptr,
                slots[:, None] * stride_ks + kv_head * stride_kh,
                offs_d[None, :],
                slots[:, None] * stride_kss + kv_head * stride_ksh + offs_nb[None, :],
                kv_mask,
                kv_scale_mask,
                tl.float32,
                K_FORMAT,
                QBLOCK,
                False,
                True,
            ).to(tl.float32)
            scores = tl.sum(q[None, :] * k, axis=1) * sm_scale
            scores = tl.where(mask_n, scores, -float("inf"))

            row_max = tl.max(scores, axis=0)
            row_max_fixed = tl.where(row_max == -float("inf"), -1e20, row_max)
            m_new = tl.maximum(row_max_fixed, m_i)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new)

            v = _load_kv(
                v_ptr,
                vs_ptr,
                slots[:, None] * stride_vs + kv_head * stride_vh,
                offs_d[None, :],
                slots[:, None] * stride_vss + kv_head * stride_vsh + offs_nb[None, :],
                kv_mask,
                kv_scale_mask,
                tl.float32,
                V_FORMAT,
                QBLOCK,
                False,
                True,
            ).to(tl.float32)
            acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
            l_i = l_i * alpha + tl.sum(p, axis=0)
            m_i = m_new

    out = tl.where(l_i == 0.0, 0.0, acc / l_i)
    tl.store(
        o_ptr + q_tok * stride_ot + q_head * stride_oh + offs_d,
        out.to(o_ptr.dtype.element_ty),
        mask=mask_d,
    )


@triton.jit
def _decode_grouped_stage1_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    ks_ptr,
    vs_ptr,
    sm_scale,
    indptr_ptr,
    indices_ptr,
    q_pos_ptr,
    mid_o_ptr,
    mid_lse_ptr,
    num_kv_splits_ptr,
    stride_qt,
    stride_qh,
    stride_ks,
    stride_kh,
    stride_vs,
    stride_vh,
    stride_kss,
    stride_ksh,
    stride_vss,
    stride_vsh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_lse_b,
    stride_lse_h,
    stride_lse_s,
    GROUP: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    VALID_BLOCK_H: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    D: tl.constexpr,
    DV: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    K_FORMAT: tl.constexpr,
    V_FORMAT: tl.constexpr,
    QBLOCK: tl.constexpr,
    KV_SPLITS: tl.constexpr,
    Q8_NATIVE_QK: tl.constexpr,
    SLOT_I64: tl.constexpr,
):
    batch_id = tl.program_id(0)
    head_block_id = tl.program_id(1)
    split_id = tl.program_id(2)

    # VALID_BLOCK_H == min(cap, GROUP) is the number of query heads actually handled per program;
    # BLOCK_H is the power-of-two tile size for the head axis (tl.arange requires a power of two),
    # so a non-power-of-two GQA group (e.g. 24/4 == 6) rounds the tile up and masks the extra
    # lanes. Each kv head spans cdiv(GROUP, VALID_BLOCK_H) head blocks.
    kv_head = head_block_id // tl.cdiv(GROUP, VALID_BLOCK_H)
    q_heads = head_block_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = q_heads < (head_block_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (q_heads < NUM_Q_HEADS)

    offs_d = tl.arange(0, BLOCK_D)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < D
    mask_dv = offs_dv < DV
    offs_nb = tl.arange(0, BLOCK_D // QBLOCK)
    offs_nbv = tl.arange(0, BLOCK_DV // QBLOCK)
    mask_nb = offs_nb < D // QBLOCK
    mask_nbv = offs_nbv < DV // QBLOCK

    kv_start = tl.load(indptr_ptr + batch_id)
    kv_len = tl.load(indptr_ptr + batch_id + 1) - kv_start
    q_pos = tl.load(q_pos_ptr + batch_id)
    effective_end = tl.minimum(kv_len, q_pos + 1)
    effective_start = 0
    if SLIDING_WINDOW > 0:
        effective_start = tl.maximum(0, q_pos - SLIDING_WINDOW + 1)
    effective_len = tl.maximum(0, effective_end - effective_start)

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(effective_len, KV_SPLITS), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_start = kv_len_per_split * split_id
    split_end = tl.minimum(split_start + kv_len_per_split, effective_len)

    m_i = tl.zeros((BLOCK_H,), dtype=tl.float32) - float("inf")
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_DV), dtype=tl.float32)

    q_offsets = batch_id * stride_qt + q_heads[:, None] * stride_qh + offs_d[None, :]
    k_base_offsets = kv_head * stride_kh
    v_base_offsets = kv_head * stride_vh
    ks_base_offsets = kv_head * stride_ksh + offs_nb[:, None]
    vs_base_offsets = kv_head * stride_vsh + offs_nbv[None, :]

    if split_end > split_start:
        q = tl.load(q_ptr + q_offsets, mask=mask_h[:, None] & mask_d[None, :], other=0.0)
        if K_FORMAT == 0:
            # Unquantized: match the cache's dtype as before. Quantized: the cache is
            # int8/fp8 and casting q into it would destroy the query -- the dequantized
            # K/V tiles are produced in q's dtype instead.
            q = q.to(k_ptr.dtype.element_ty)

        for rel_start in tl.range(split_start, split_end, BLOCK_N):
            rel_offs = rel_start + tl.arange(0, BLOCK_N)
            mask_n = rel_offs < split_end
            logical_offs = effective_start + rel_offs
            slots = tl.load(indices_ptr + kv_start + logical_offs, mask=mask_n, other=0)
            if SLOT_I64:
                # Pools above 2**31 elements overflow a 32-bit slot*stride offset.
                slots = slots.to(tl.int64)

            if Q8_NATIVE_QK:
                # Q8 K is already in the integer domain. Quantize each 32-wide
                # query block once, perform eight native int8 dot products, then
                # apply the query/key block scales to the int32 partials. This
                # avoids expanding K to BF16 and preserves Q8_0's per-block scale.
                scores = tl.zeros((BLOCK_H, BLOCK_N), dtype=tl.float32)
                for block_id in tl.static_range(BLOCK_D // QBLOCK):
                    block_offs = block_id * QBLOCK + tl.arange(0, QBLOCK)
                    q_block = tl.load(
                        q_ptr
                        + batch_id * stride_qt
                        + q_heads[:, None] * stride_qh
                        + block_offs[None, :],
                        mask=mask_h[:, None] & (block_offs[None, :] < D),
                        other=0.0,
                    ).to(tl.float32)
                    q_amax = tl.max(tl.abs(q_block), axis=1)
                    q_scale = tl.where(q_amax > 0, q_amax / 127.0, 1.0)
                    q_scaled = q_block / q_scale[:, None]
                    q_int = tl.where(
                        q_scaled >= 0,
                        tl.floor(q_scaled + 0.5),
                        tl.ceil(q_scaled - 0.5),
                    ).to(tl.int8)
                    k_int = tl.load(
                        k_ptr
                        + slots[None, :] * stride_ks
                        + k_base_offsets
                        + block_offs[:, None],
                        mask=(block_offs[:, None] < D) & mask_n[None, :],
                        other=0,
                    ).to(tl.int8)
                    k_scale = tl.load(
                        ks_ptr
                        + slots * stride_kss
                        + kv_head * stride_ksh
                        + block_id,
                        mask=mask_n,
                        other=0.0,
                    ).to(tl.float32)
                    scores += tl.dot(q_int, k_int, out_dtype=tl.int32).to(
                        tl.float32
                    ) * (q_scale[:, None] * k_scale[None, :])
                scores *= sm_scale
            else:
                k = _load_kv(
                    k_ptr,
                    ks_ptr,
                    slots[None, :] * stride_ks + k_base_offsets,
                    offs_d[:, None],
                    slots[None, :] * stride_kss + ks_base_offsets,
                    mask_n[None, :] & mask_d[:, None],
                    mask_n[None, :] & mask_nb[:, None],
                    q.dtype,
                    K_FORMAT,
                    QBLOCK,
                    True,
                    True,
                )
                scores = tl.dot(q, k) * sm_scale
            scores = tl.where(mask_h[:, None] & mask_n[None, :], scores, -float("inf"))

            v = _load_kv(
                v_ptr,
                vs_ptr,
                slots[:, None] * stride_vs + v_base_offsets,
                offs_dv[None, :],
                slots[:, None] * stride_vss + vs_base_offsets,
                mask_n[:, None] & mask_dv[None, :],
                mask_n[:, None] & mask_nbv[None, :],
                q.dtype,
                V_FORMAT,
                QBLOCK,
                False,
                True,
            )

            m_new = tl.maximum(tl.max(scores, axis=1), m_i)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new[:, None])
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_new

        out = acc / l_i[:, None]
        mid_offsets = (
            batch_id * stride_mid_ob
            + q_heads[:, None] * stride_mid_oh
            + split_id * stride_mid_os
            + offs_dv[None, :]
        )
        tl.store(mid_o_ptr + mid_offsets, out, mask=mask_h[:, None] & mask_dv[None, :])

        lse_offsets = (
            batch_id * stride_lse_b
            + q_heads * stride_lse_h
            + split_id * stride_lse_s
        )
        tl.store(mid_lse_ptr + lse_offsets, m_i + tl.log(l_i), mask=mask_h)


@triton.jit
def _decode_stage2_kernel(
    mid_o_ptr,
    mid_lse_ptr,
    o_ptr,
    indptr_ptr,
    q_pos_ptr,
    num_kv_splits_ptr,
    sinks_ptr,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_lse_b,
    stride_lse_h,
    stride_lse_s,
    stride_ot,
    stride_oh,
    MAX_KV_SPLITS: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    DV: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    HAS_SINKS: tl.constexpr,
    KV_SPLITS: tl.constexpr,
):
    batch_id = tl.program_id(0)
    q_head = tl.program_id(1)

    kv_len = tl.load(indptr_ptr + batch_id + 1) - tl.load(indptr_ptr + batch_id)
    q_pos = tl.load(q_pos_ptr + batch_id)
    effective_end = tl.minimum(kv_len, q_pos + 1)
    effective_start = 0
    if SLIDING_WINDOW > 0:
        effective_start = tl.maximum(0, q_pos - SLIDING_WINDOW + 1)
    effective_len = tl.maximum(0, effective_end - effective_start)

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(effective_len, KV_SPLITS), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < DV
    if HAS_SINKS:
        m_i = tl.load(sinks_ptr + q_head).to(tl.float32)
        l_i = 1.0
    else:
        m_i = -float("inf")
        l_i = 0.0
    acc = tl.zeros((BLOCK_DV,), dtype=tl.float32)

    mid_base = batch_id * stride_mid_ob + q_head * stride_mid_oh + offs_d
    lse_base = batch_id * stride_lse_b + q_head * stride_lse_h

    for split_id in tl.range(0, MAX_KV_SPLITS, num_stages=2):
        split_start = kv_len_per_split * split_id
        split_end = tl.minimum(split_start + kv_len_per_split, effective_len)

        if split_end > split_start:
            partial = tl.load(
                mid_o_ptr + mid_base + split_id * stride_mid_os,
                mask=mask_d,
                other=0.0,
            )
            partial_lse = tl.load(mid_lse_ptr + lse_base + split_id * stride_lse_s)
            m_new = tl.maximum(partial_lse, m_i)
            alpha = tl.exp(m_i - m_new)
            beta = tl.exp(partial_lse - m_new)
            acc = acc * alpha + partial * beta
            l_i = l_i * alpha + beta
            m_i = m_new

    out = tl.where(l_i == 0.0, 0.0, acc / l_i)
    tl.store(
        o_ptr + batch_id * stride_ot + q_head * stride_oh + offs_d,
        out.to(o_ptr.dtype.element_ty),
        mask=mask_d,
    )


def _cache_format(cache, scale, head_dim: int) -> int:
    """Infer the compile-time storage format from logical and physical geometry.

    0=unquantized, 1=byte-per-value quantized, 2=Q4_0, 3=Q6_0, 4=Q5_0.
    """
    if scale is None:
        assert cache.shape[-1] == head_dim
        return 0
    if cache.shape[-1] == head_dim:
        return 1
    if cache.shape[-1] * 2 == head_dim:
        return 2
    if cache.shape[-1] * 4 == head_dim * 3:
        return 3
    if cache.shape[-1] * 8 == head_dim * 5:
        return 4
    raise AssertionError(
        f"KV storage dim {cache.shape[-1]} is incompatible with logical head_dim {head_dim}"
    )


def _kv_scale_args(k_cache, v_cache, k_scale, v_scale, head_dim: int):
    """Scale pointers, strides and independent K/V format ids for a kernel launch.

    Unquantized pools pass ``k_scale=None``; the kernels then never touch the scale
    pointer, so the KV buffer itself stands in and ``FORMAT=0`` compiles dequant away.
    dequant away entirely -- the bf16 path emits the same code it did before.

    K and V may differ: this is what permits Q8 keys with Q6 values without padding the
    value slab back to eight bits.
    """
    k_format = _cache_format(k_cache, k_scale, head_dim)
    v_format = _cache_format(v_cache, v_scale, head_dim)
    scales = [s for s in (k_scale, v_scale) if s is not None]
    block = 1
    if scales:
        assert all(s.dim() == 3 for s in scales), "scales are [slots, heads, D // block]"
        block = head_dim // scales[0].shape[-1]
        assert all(s.shape[-1] * block == head_dim for s in scales)
    ks = k_cache if k_scale is None else k_scale
    vs = v_cache if v_scale is None else v_scale
    return (
        ks,
        vs,
        0 if k_scale is None else k_scale.stride(0),
        0 if k_scale is None else k_scale.stride(1),
        0 if v_scale is None else v_scale.stride(0),
        0 if v_scale is None else v_scale.stride(1),
        k_format,
        v_format,
        block,
    )


def decode_paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indptr: torch.Tensor,
    indices: torch.Tensor,
    q_positions: torch.Tensor,
    attn_logits: torch.Tensor,
    attn_lse: torch.Tensor,
    num_kv_splits: torch.Tensor,
    max_kv_splits: int,
    sm_scale: float,
    sliding_window: int | None = None,
    sinks: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """SGLang-style split-k grouped decode attention for one query per request."""

    assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda
    assert q.dim() == 3 and k_cache.dim() == 3 and v_cache.dim() == 3
    batch, num_q_heads, head_dim = q.shape
    ks, vs, s_kss, s_ksh, s_vss, s_vsh, k_format, v_format, qblock = _kv_scale_args(
        k_cache, v_cache, k_scale, v_scale, head_dim
    )
    slot_i64 = _slot_offsets_need_int64(k_cache, v_cache, k_scale, v_scale)
    num_kv_heads = k_cache.shape[1]
    assert batch == indptr.numel() - 1
    assert v_cache.shape[1] == num_kv_heads
    assert num_q_heads % num_kv_heads == 0
    assert attn_logits.shape[0] >= batch
    assert attn_logits.shape[1] >= num_q_heads
    assert attn_logits.shape[2] >= max_kv_splits
    assert attn_logits.shape[3] >= head_dim
    assert attn_lse.shape[0] >= batch
    assert attn_lse.shape[1] >= num_q_heads
    assert attn_lse.shape[2] >= max_kv_splits
    if sinks is not None:
        assert sinks.is_cuda
        assert sinks.dim() == 1
        assert sinks.numel() >= num_q_heads
        sinks = sinks.contiguous()

    o = out if out is not None else torch.empty_like(q)
    sinks_arg = sinks if sinks is not None else q
    group = num_q_heads // num_kv_heads
    quant_name = (
        "q8_q6" if (k_format, v_format) == (1, 3)
        else "q6_q5" if (k_format, v_format) == (3, 4)
        else "int4" if k_format == 2
        else "quant8" if k_format == 1
        else None
    )
    capability = torch.cuda.get_device_capability(q.device)
    q8_native_qk = (
        _Q8_NATIVE_QK
        and k_format == 1
        and k_cache.dtype == torch.int8
        and capability == (8, 9)
        and qblock == 32
        and head_dim == 256
        and num_q_heads == 16
        and num_kv_heads == 2
    )
    preferred_splits, block_n, num_warps = decode_launch_config(
        quant_name=quant_name,
        head_dim=head_dim,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        compute_capability=capability,
        sm_count=_sm_count(q.device.index if q.device.index is not None else 0),
    )
    # Direct kernel callers and older capture buffers may provide less scratch;
    # retain correctness and use every split they made available. The backend
    # allocates the preferred capacity for new captures.
    launch_splits = decode_runtime_splits(
        preferred_splits=preferred_splits,
        scratch_splits=max_kv_splits,
        batch=batch,
        quant_name=quant_name,
        head_dim=head_dim,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        compute_capability=capability,
    )
    # valid_block_h = heads computed per program (drives the grid + head indexing); block_h =
    # power-of-two tile size for tl.arange. They differ only for non-power-of-two GQA groups
    # (e.g. 6), where block_h rounds up and the kernel masks the extra lanes.
    valid_block_h = min(16, group)
    block_h = triton.next_power_of_2(valid_block_h)
    block_d = triton.next_power_of_2(head_dim)
    block_dv = triton.next_power_of_2(head_dim)

    _decode_grouped_stage1_kernel[
        (batch, triton.cdiv(num_q_heads, valid_block_h), launch_splits)
    ](
        q,
        k_cache,
        v_cache,
        ks,
        vs,
        sm_scale,
        indptr,
        indices,
        q_positions,
        attn_logits,
        attn_lse,
        num_kv_splits,
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        v_cache.stride(0),
        v_cache.stride(1),
        s_kss,
        s_ksh,
        s_vss,
        s_vsh,
        attn_logits.stride(0),
        attn_logits.stride(1),
        attn_logits.stride(2),
        attn_lse.stride(0),
        attn_lse.stride(1),
        attn_lse.stride(2),
        GROUP=group,
        NUM_Q_HEADS=num_q_heads,
        BLOCK_D=block_d,
        BLOCK_DV=block_dv,
        BLOCK_N=block_n,
        BLOCK_H=block_h,
        VALID_BLOCK_H=valid_block_h,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        D=head_dim,
        DV=head_dim,
        SLIDING_WINDOW=sliding_window or 0,
        K_FORMAT=k_format,
        V_FORMAT=v_format,
        QBLOCK=qblock,
        SLOT_I64=slot_i64,
        KV_SPLITS=launch_splits,
        Q8_NATIVE_QK=q8_native_qk,
        num_warps=num_warps,
        num_stages=2,
    )
    _decode_stage2_kernel[(batch, num_q_heads)](
        attn_logits,
        attn_lse,
        o,
        indptr,
        q_positions,
        num_kv_splits,
        sinks_arg,
        attn_logits.stride(0),
        attn_logits.stride(1),
        attn_logits.stride(2),
        attn_lse.stride(0),
        attn_lse.stride(1),
        attn_lse.stride(2),
        o.stride(0),
        o.stride(1),
        MAX_KV_SPLITS=launch_splits,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        BLOCK_DV=block_dv,
        DV=head_dim,
        SLIDING_WINDOW=sliding_window or 0,
        HAS_SINKS=sinks is not None,
        KV_SPLITS=launch_splits,
        num_warps=4,
        num_stages=2,
    )
    return o


@triton.jit
def _extend_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    ks_ptr,
    vs_ptr,
    o_ptr,
    qo_indptr_ptr,
    kv_indptr_ptr,
    kv_indices_ptr,
    prefix_lens_ptr,
    sm_scale,
    sinks_ptr,
    stride_qt,
    stride_qh,
    stride_ks,
    stride_kh,
    stride_vs,
    stride_vh,
    stride_kss,
    stride_ksh,
    stride_vss,
    stride_vsh,
    stride_ot,
    stride_oh,
    GROUP: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    HAS_SINKS: tl.constexpr,
    K_FORMAT: tl.constexpr,
    V_FORMAT: tl.constexpr,
    QBLOCK: tl.constexpr,
    SLOT_I64: tl.constexpr,
):
    seq_id = tl.program_id(0)
    q_head = tl.program_id(1)
    block_m_id = tl.program_id(2)
    kv_head = q_head // GROUP

    q_start = tl.load(qo_indptr_ptr + seq_id)
    q_end = tl.load(qo_indptr_ptr + seq_id + 1)
    q_len = q_end - q_start
    kv_start = tl.load(kv_indptr_ptr + seq_id)
    kv_len = tl.load(kv_indptr_ptr + seq_id + 1) - kv_start
    prefix_len = tl.load(prefix_lens_ptr + seq_id)

    offs_m = block_m_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_m = offs_m < q_len
    mask_d = offs_d < D
    mask_dv = offs_dv < D
    offs_nb = tl.arange(0, BLOCK_D // QBLOCK)
    offs_nbv = tl.arange(0, BLOCK_DV // QBLOCK)
    mask_nb = offs_nb < D // QBLOCK
    mask_nbv = offs_nbv < D // QBLOCK
    q_abs_pos = prefix_len + offs_m
    block_q_end = tl.minimum(q_len, (block_m_id + 1) * BLOCK_M)
    kv_loop_end = tl.minimum(kv_len, prefix_len + block_q_end)

    q = tl.load(
        q_ptr + (q_start + offs_m[:, None]) * stride_qt + q_head * stride_qh + offs_d[None, :],
        mask=mask_m[:, None] & mask_d[None, :],
        other=0.0,
    )

    if HAS_SINKS:
        sink = tl.load(sinks_ptr + q_head).to(tl.float32)
        m_i = tl.full((BLOCK_M,), sink, dtype=tl.float32)
        l_i = tl.full((BLOCK_M,), 1.0, dtype=tl.float32)
    else:
        m_i = tl.zeros((BLOCK_M,), dtype=tl.float32) - float("inf")
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_DV), dtype=tl.float32)

    for start_n in tl.range(0, kv_loop_end, BLOCK_N):
        kv_offsets = start_n + offs_n
        mask_n = kv_offsets < kv_len
        key_pos = kv_offsets
        causal_mask = key_pos[None, :] <= q_abs_pos[:, None]
        if SLIDING_WINDOW > 0:
            causal_mask = causal_mask & ((key_pos[None, :] + SLIDING_WINDOW) > q_abs_pos[:, None])
        final_mask = mask_m[:, None] & mask_n[None, :] & causal_mask

        skip_tile = tl.max(tl.max(final_mask.to(tl.int32), axis=1), axis=0) == 0
        if not skip_tile:
            slots = tl.load(kv_indices_ptr + kv_start + kv_offsets, mask=mask_n, other=0)
            if SLOT_I64:
                # Pools above 2**31 elements overflow a 32-bit slot*stride offset.
                slots = slots.to(tl.int64)
            k = _load_kv(
                k_ptr,
                ks_ptr,
                slots[None, :] * stride_ks + kv_head * stride_kh,
                offs_d[:, None],
                slots[None, :] * stride_kss + kv_head * stride_ksh + offs_nb[:, None],
                mask_n[None, :] & mask_d[:, None],
                mask_n[None, :] & mask_nb[:, None],
                q.dtype,
                K_FORMAT,
                QBLOCK,
                True,
                False,
            )
            scores = tl.dot(q.to(k.dtype), k) * sm_scale
            scores = tl.where(final_mask, scores, -float("inf"))

            row_max = tl.max(scores, axis=1)
            row_max_fixed = tl.where(row_max == -float("inf"), -1e20, row_max)
            m_new = tl.maximum(row_max_fixed, m_i)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new[:, None])

            v = _load_kv(
                v_ptr,
                vs_ptr,
                slots[:, None] * stride_vs + kv_head * stride_vh,
                offs_dv[None, :],
                slots[:, None] * stride_vss + kv_head * stride_vsh + offs_nbv[None, :],
                mask_n[:, None] & mask_dv[None, :],
                mask_n[:, None] & mask_nbv[None, :],
                q.dtype,
                V_FORMAT,
                QBLOCK,
                False,
                False,
            )
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_new

    out = tl.where(l_i[:, None] == 0.0, 0.0, acc / l_i[:, None])
    tl.store(
        o_ptr
        + (q_start + offs_m[:, None]) * stride_ot
        + q_head * stride_oh
        + offs_dv[None, :],
        out.to(o_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_dv[None, :],
    )


@triton.jit
def _extend_attention_split_kernel(
    q_ptr,
    k_extend_ptr,
    v_extend_ptr,
    k_cache_ptr,
    v_cache_ptr,
    ks_ptr,
    vs_ptr,
    o_ptr,
    qo_indptr_ptr,
    kv_indptr_ptr,
    kv_indices_ptr,
    prefix_lens_ptr,
    sm_scale,
    sinks_ptr,
    stride_qt,
    stride_qh,
    stride_ket,
    stride_keh,
    stride_vet,
    stride_veh,
    stride_kcs,
    stride_kch,
    stride_vcs,
    stride_vch,
    stride_kss,
    stride_ksh,
    stride_vss,
    stride_vsh,
    stride_ot,
    stride_oh,
    GROUP: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    HAS_SINKS: tl.constexpr,
    K_FORMAT: tl.constexpr,
    V_FORMAT: tl.constexpr,
    QBLOCK: tl.constexpr,
    SLOT_I64: tl.constexpr,
):
    seq_id = tl.program_id(0)
    q_head = tl.program_id(1)
    block_m_id = tl.program_id(2)
    kv_head = q_head // GROUP

    q_start = tl.load(qo_indptr_ptr + seq_id)
    q_len = tl.load(qo_indptr_ptr + seq_id + 1) - q_start
    kv_start = tl.load(kv_indptr_ptr + seq_id)
    prefix_len = tl.load(prefix_lens_ptr + seq_id)

    offs_m = block_m_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_m = offs_m < q_len
    mask_d = offs_d < D
    mask_dv = offs_dv < D
    offs_nb = tl.arange(0, BLOCK_D // QBLOCK)
    offs_nbv = tl.arange(0, BLOCK_DV // QBLOCK)
    mask_nb = offs_nb < D // QBLOCK
    mask_nbv = offs_nbv < D // QBLOCK
    q_abs_pos = prefix_len + offs_m

    q = tl.load(
        q_ptr
        + (q_start + offs_m[:, None]) * stride_qt
        + q_head * stride_qh
        + offs_d[None, :],
        mask=mask_m[:, None] & mask_d[None, :],
        other=0.0,
    )

    if HAS_SINKS:
        sink = tl.load(sinks_ptr + q_head).to(tl.float32)
        m_i = tl.full((BLOCK_M,), sink, dtype=tl.float32)
        l_i = tl.full((BLOCK_M,), 1.0, dtype=tl.float32)
    else:
        m_i = tl.zeros((BLOCK_M,), dtype=tl.float32) - float("inf")
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_DV), dtype=tl.float32)

    for start_n in tl.range(0, prefix_len, BLOCK_N):
        kv_offsets = start_n + offs_n
        mask_n = kv_offsets < prefix_len
        key_pos = kv_offsets
        final_mask = mask_m[:, None] & mask_n[None, :]
        if SLIDING_WINDOW > 0:
            window_mask = (key_pos[None, :] + SLIDING_WINDOW) > q_abs_pos[:, None]
            final_mask = final_mask & window_mask

        skip_tile = False
        if SLIDING_WINDOW > 0:
            skip_tile = tl.max(tl.max(final_mask.to(tl.int32), axis=1), axis=0) == 0

        if not skip_tile:
            slots = tl.load(kv_indices_ptr + kv_start + kv_offsets, mask=mask_n, other=0)
            if SLOT_I64:
                # Pools above 2**31 elements overflow a 32-bit slot*stride offset.
                slots = slots.to(tl.int64)
            k = _load_kv(
                k_cache_ptr,
                ks_ptr,
                slots[None, :] * stride_kcs + kv_head * stride_kch,
                offs_d[:, None],
                slots[None, :] * stride_kss + kv_head * stride_ksh + offs_nb[:, None],
                mask_n[None, :] & mask_d[:, None],
                mask_n[None, :] & mask_nb[:, None],
                q.dtype,
                K_FORMAT,
                QBLOCK,
                True,
                False,
            )
            scores = tl.dot(q.to(k.dtype), k) * sm_scale
            scores = tl.where(final_mask, scores, -float("inf"))

            row_max = tl.max(scores, axis=1)
            row_max_fixed = tl.where(row_max == -float("inf"), -1e20, row_max)
            m_new = tl.maximum(row_max_fixed, m_i)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new[:, None])

            v = _load_kv(
                v_cache_ptr,
                vs_ptr,
                slots[:, None] * stride_vcs + kv_head * stride_vch,
                offs_dv[None, :],
                slots[:, None] * stride_vss + kv_head * stride_vsh + offs_nbv[None, :],
                mask_n[:, None] & mask_dv[None, :],
                mask_n[:, None] & mask_nbv[None, :],
                q.dtype,
                V_FORMAT,
                QBLOCK,
                False,
                False,
            )
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_new

    current_end = tl.minimum(q_len, (block_m_id + 1) * BLOCK_M)
    for start_n in tl.range(0, current_end, BLOCK_N):
        local_kv_offsets = start_n + offs_n
        mask_n = local_kv_offsets < current_end
        local_q_pos = offs_m
        causal_mask = local_kv_offsets[None, :] <= local_q_pos[:, None]
        if SLIDING_WINDOW > 0:
            causal_mask = causal_mask & (
                (local_kv_offsets[None, :] + SLIDING_WINDOW) > local_q_pos[:, None]
            )
        final_mask = mask_m[:, None] & mask_n[None, :] & causal_mask

        skip_tile = False
        if SLIDING_WINDOW > 0:
            skip_tile = tl.max(tl.max(final_mask.to(tl.int32), axis=1), axis=0) == 0

        if not skip_tile:
            k = tl.load(
                k_extend_ptr
                + (q_start + local_kv_offsets[None, :]) * stride_ket
                + kv_head * stride_keh
                + offs_d[:, None],
                mask=mask_n[None, :] & mask_d[:, None],
                other=0.0,
            )
            scores = tl.dot(q.to(k.dtype), k) * sm_scale
            scores = tl.where(final_mask, scores, -float("inf"))

            row_max = tl.max(scores, axis=1)
            row_max_fixed = tl.where(row_max == -float("inf"), -1e20, row_max)
            m_new = tl.maximum(row_max_fixed, m_i)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new[:, None])

            v = tl.load(
                v_extend_ptr
                + (q_start + local_kv_offsets[:, None]) * stride_vet
                + kv_head * stride_veh
                + offs_dv[None, :],
                mask=mask_n[:, None] & mask_dv[None, :],
                other=0.0,
            )
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_new

    out = tl.where(l_i[:, None] == 0.0, 0.0, acc / l_i[:, None])
    tl.store(
        o_ptr
        + (q_start + offs_m[:, None]) * stride_ot
        + q_head * stride_oh
        + offs_dv[None, :],
        out.to(o_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_dv[None, :],
    )


def extend_paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_q_len: int,
    sm_scale: float,
    sliding_window: int | None = None,
    sinks: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    k_extend: torch.Tensor | None = None,
    v_extend: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Block-tiled causal prefill/extend attention over paged KV cache."""

    assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda
    assert q.dim() == 3 and k_cache.dim() == 3 and v_cache.dim() == 3
    num_q_tokens, num_q_heads, head_dim = q.shape
    ks, vs, s_kss, s_ksh, s_vss, s_vsh, k_format, v_format, qblock = _kv_scale_args(
        k_cache, v_cache, k_scale, v_scale, head_dim
    )
    slot_i64 = _slot_offsets_need_int64(k_cache, v_cache, k_scale, v_scale)
    num_kv_heads = k_cache.shape[1]
    assert qo_indptr.numel() == kv_indptr.numel()
    assert prefix_lens.numel() == qo_indptr.numel() - 1
    assert v_cache.shape[1] == num_kv_heads
    assert num_q_heads % num_kv_heads == 0
    if sinks is not None:
        assert sinks.is_cuda
        assert sinks.dim() == 1
        assert sinks.numel() >= num_q_heads
        sinks = sinks.contiguous()

    o = out if out is not None else torch.empty_like(q)
    sinks_arg = sinks if sinks is not None else q
    block_d = triton.next_power_of_2(head_dim)
    block_dv = triton.next_power_of_2(head_dim)
    # Tile size is shared-memory bound: keep the fast (large) tiles on GPUs whose opt-in
    # shared memory fits them, shrink on consumer GPUs (sm_89 ~99KB) where the default
    # 128x64 overflows once head_dim >= 256 (e.g. gemma4: SWA 256, full-attention 512).
    block_m, block_n, num_warps, num_stages = extend_launch_config(
        head_dim=head_dim,
        block_d=block_d,
        smem_optin=_optin_smem_bytes(q.device.index),
        capability=torch.cuda.get_device_capability(q.device),
        k_format=k_format,
        v_format=v_format,
    )
    grid = (qo_indptr.numel() - 1, num_q_heads, triton.cdiv(max_q_len, block_m))
    if k_extend is not None or v_extend is not None:
        assert k_extend is not None and v_extend is not None
        assert k_extend.is_cuda and v_extend.is_cuda
        assert k_extend.dim() == 3 and v_extend.dim() == 3
        assert k_extend.shape[0] == num_q_tokens and v_extend.shape[0] == num_q_tokens
        assert k_extend.shape[1] == num_kv_heads and v_extend.shape[1] == num_kv_heads
        assert k_extend.shape[-1] == head_dim and v_extend.shape[-1] == head_dim
        _extend_attention_split_kernel[grid](
            q,
            k_extend,
            v_extend,
            k_cache,
            v_cache,
            ks,
            vs,
            o,
            qo_indptr,
            kv_indptr,
            kv_indices,
            prefix_lens,
            sm_scale,
            sinks_arg,
            q.stride(0),
            q.stride(1),
            k_extend.stride(0),
            k_extend.stride(1),
            v_extend.stride(0),
            v_extend.stride(1),
            k_cache.stride(0),
            k_cache.stride(1),
            v_cache.stride(0),
            v_cache.stride(1),
            s_kss,
            s_ksh,
            s_vss,
            s_vsh,
            o.stride(0),
            o.stride(1),
            GROUP=num_q_heads // num_kv_heads,
            D=head_dim,
            BLOCK_D=block_d,
            BLOCK_DV=block_dv,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            SLIDING_WINDOW=sliding_window or 0,
            HAS_SINKS=sinks is not None,
            K_FORMAT=k_format,
            V_FORMAT=v_format,
            QBLOCK=qblock,
            SLOT_I64=slot_i64,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return o

    _extend_attention_kernel[grid](
        q,
        k_cache,
        v_cache,
        ks,
        vs,
        o,
        qo_indptr,
        kv_indptr,
        kv_indices,
        prefix_lens,
        sm_scale,
        sinks_arg,
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        v_cache.stride(0),
        v_cache.stride(1),
        s_kss,
        s_ksh,
        s_vss,
        s_vsh,
        o.stride(0),
        o.stride(1),
        GROUP=num_q_heads // num_kv_heads,
        D=head_dim,
        BLOCK_D=block_d,
        BLOCK_DV=block_dv,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        SLIDING_WINDOW=sliding_window or 0,
        HAS_SINKS=sinks is not None,
        K_FORMAT=k_format,
        V_FORMAT=v_format,
        QBLOCK=qblock,
        SLOT_I64=slot_i64,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return o


def paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indptr: torch.Tensor,
    indices: torch.Tensor,
    q_to_req: torch.Tensor,
    q_positions: torch.Tensor,
    sm_scale: float,
    sliding_window: int | None = None,
    sinks: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    block_n: int = 32,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Paged causal attention for one layer.

    ``q`` is ``[num_query_tokens, num_q_heads, head_dim]``. KV cache tensors are
    flattened to ``[num_slots, num_kv_heads, head_dim]``. ``indptr`` and
    ``indices`` describe each request's logical KV slots in order.
    """

    assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda
    assert q.dim() == 3 and k_cache.dim() == 3 and v_cache.dim() == 3
    num_tokens, num_q_heads, head_dim = q.shape
    ks, vs, s_kss, s_ksh, s_vss, s_vsh, k_format, v_format, qblock = _kv_scale_args(
        k_cache, v_cache, k_scale, v_scale, head_dim
    )
    slot_i64 = _slot_offsets_need_int64(k_cache, v_cache, k_scale, v_scale)
    num_kv_heads = k_cache.shape[1]
    assert v_cache.shape[1] == num_kv_heads
    assert num_q_heads % num_kv_heads == 0
    if sinks is not None:
        assert sinks.is_cuda
        assert sinks.dim() == 1
        assert sinks.numel() >= num_q_heads
        sinks = sinks.contiguous()

    o = out if out is not None else torch.empty_like(q)
    sinks_arg = sinks if sinks is not None else q
    block_d = triton.next_power_of_2(head_dim)
    grid = (num_tokens, num_q_heads)
    _paged_attention_kernel[grid](
        q,
        k_cache,
        v_cache,
        ks,
        vs,
        o,
        indptr,
        indices,
        q_to_req,
        q_positions,
        sm_scale,
        sinks_arg,
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        v_cache.stride(0),
        v_cache.stride(1),
        s_kss,
        s_ksh,
        s_vss,
        s_vsh,
        o.stride(0),
        o.stride(1),
        GROUP=num_q_heads // num_kv_heads,
        D=head_dim,
        BLOCK_D=block_d,
        BLOCK_N=block_n,
        SLIDING_WINDOW=sliding_window or 0,
        HAS_SINKS=sinks is not None,
        K_FORMAT=k_format,
        V_FORMAT=v_format,
        QBLOCK=qblock,
        SLOT_I64=slot_i64,
        num_warps=8 if head_dim >= 256 else 4,
        num_stages=2,
    )
    return o
