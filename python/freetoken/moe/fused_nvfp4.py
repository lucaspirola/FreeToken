"""Host orchestration for the inline-dequant NVFP4 fused-MoE path.

Mirrors :mod:`freetoken.moe.fused` (gemm1 -> act -> gemm2 -> sum-reduce) but the two
grouped GEMMs read the NVFP4 expert cache directly and dequantize inside the K-loop,
so no BF16 copy of the experts is ever materialized.

For ungated ReLU^2 experts (Nemotron 3.5 Lightning) the middle step collapses: the
activation rides gemm1's ``ACT`` epilogue, so the ``[M*top_k, I]`` intermediate is never
stored, re-read and re-stored in bf16 -- gemm2 reads gemm1's output buffer directly.
Gated activations mix the two halves of ``gate_up`` and keep the separate op.

Kernel geometry comes from two tuned tables, both with the pre-existing heuristics as
fallbacks so an untuned (shape, device) still runs:
  * decode: :func:`decode_marlin_config`, keyed by ``(N, K, top_k, sm_count)`` (a Python
    dict -- decode is CUDA-graph captured, so the config must be fixed at trace time).
  * prefill: :func:`nvfp4_moe_config`, keyed by ``(num_experts, N, K, device)`` from
    ``configs/triton_<ver>/nvfp4,E=..,N=..,K=..,device_name=...json`` (M-bucketed),
    written by ``benchmarks/tune_nvfp4_moe.py --write``.
"""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path
from typing import Any, Dict

import torch
import triton
import triton.language as tl

from freetoken.kernel import moe_sum_reduce_triton
from freetoken.kernel.triton.e4m3_compat import e4m3_kernel_view
from freetoken.kernel.triton.nvfp4_fused_moe import (
    _decode_nvfp4_marlin_kernel,
    _decode_nvfp4_moe_kernel,
    _e2m1_lut,
    _prefill_nvfp4_moe_kernel,
)
from freetoken.layers import (
    gelu_and_mul,
    gelu_tanh_and_mul,
    silu_and_mul,
    swigluoai_and_mul,
)
from freetoken.moe.fused import moe_align_block_size

_ACT = {"silu": silu_and_mul, "gelu": gelu_and_mul, "gelu_tanh": gelu_tanh_and_mul}

# gemm1 epilogue codes, mirroring the kernels' ``ACT`` constexpr: 0 none, 1 relu(x)**2.
# Only elementwise activations over the *whole* gemm1 row can be fused; the gated kinds
# (silu/gelu/swigluoai) need both halves of a row at once and stay on ``_run_act``.
_FUSED_ACT_CODE = {"relu2": 1}


def _act_code(activation: str) -> int:
    """``ACT`` constexpr for gemm1, or 0 when the activation needs the separate pass."""
    return _FUSED_ACT_CODE.get(activation, 0)


def _run_act(
    activation: str,
    gate_up: torch.Tensor,
    out: torch.Tensor,
    act_alpha: float,
    act_limit: float,
) -> None:
    """gemm1 -> gemm2 activation dispatch. ``swigluoai`` (MiniMax-M3, clamped
    gpt-oss swiglu over the banks' uninterleaved [gate; up] halves) carries the
    per-model ``act_alpha``/``act_limit`` scalars; the plain *_and_mul kinds
    ignore them."""
    if activation == "relu2":
        torch.square(torch.relu(gate_up), out=out)
        return
    if activation == "swigluoai":
        swigluoai_and_mul(gate_up, out, alpha=act_alpha, limit=act_limit)
        return
    _ACT[activation](gate_up, out)

# Decode is captured into a CUDA graph, so the config must be fixed (no triton.autotune,
# which benchmarks at run time). Tuned offline against the NVFP4 decode kernels.
# These drive the original LUT-gather decode (_decode_gemm), kept only for A/B.
_DECODE_BLOCK_N = 64
_DECODE_BLOCK_KB = 128
_DECODE_WARPS = 4

# Marlin-style decode config (int32 wide loads + deferred reduction). Offline sweep over
# the qwen35/qwen3moe (I=512/768) decode shapes picked BLOCK_N=16, BLOCK_KW=16 (== 128
# k-values/iter), 4 warps -- the wide load lifts the gate/up GEMM ~43%->~51% of peak BW.
_DECODE_MARLIN_BLOCK_N = 16
_DECODE_MARLIN_BLOCK_KW = 16
_DECODE_MARLIN_WARPS = 4

# Per-shape decode overrides, keyed by (N, K, top_k, sm_count). Swept cold-L2 (routings
# rotated past the L2, as benchmarks/bench_nvfp4_moe_kernels.py does) with
# ``benchmarks/tune_nvfp4_moe.py --decode``; the pick minimizes the M in {1, 2, 4} total,
# the batch range where the Triton path beats b12x. Anything not listed uses the
# generic constants above.
_DECODE_MARLIN_CONFIGS: Dict[tuple, Dict[str, int]] = {
    # Nemotron 3.5 Lightning (H=2688, I=1856, top-6) on the RTX 5080 (GB203, 84 SMs).
    # Both GEMMs want a 256-k-value K-iter (BLOCK_KW=32) rather than the generic 128:
    # twice the in-flight weight bytes per program, which is what a GEMV with 6 routes
    # and 84 SMs is short of. gate_up additionally prefers 2 warps -- 16 output columns
    # over 4 warps leaves each warp a quarter-tile and the deferred reduction dominates.
    # down is a two-way tie with (BLOCK_N=8, warps=2) across two independent sweeps
    # (<1% apart at every M); 16/4 is kept as the steadier of the two at M=8/16.
    (1856, 2688, 6, 84): {"BLOCK_SIZE_N": 16, "BLOCK_SIZE_KW": 32, "num_warps": 2},
    (2688, 1856, 6, 84): {"BLOCK_SIZE_N": 16, "BLOCK_SIZE_KW": 32, "num_warps": 4},
}


@functools.lru_cache(maxsize=8)
def _sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def decode_marlin_config(N: int, K: int, top_k: int, sm_count: int) -> Dict[str, int]:
    """Tuned ``(BLOCK_SIZE_N, BLOCK_SIZE_KW, num_warps)`` for the decode GEMV.

    Falls back to the generic constants for an unswept ``(N, K, top_k, sm_count)``."""
    cfg = _DECODE_MARLIN_CONFIGS.get((N, K, top_k, sm_count))
    if cfg is not None:
        return dict(cfg)
    return {
        "BLOCK_SIZE_N": _DECODE_MARLIN_BLOCK_N,
        "BLOCK_SIZE_KW": _DECODE_MARLIN_BLOCK_KW,
        "num_warps": _DECODE_MARLIN_WARPS,
    }


def _tl_dtype(dt: torch.dtype):
    if dt == torch.bfloat16:
        return tl.bfloat16
    if dt == torch.float16:
        return tl.float16
    return tl.float32


def _decode_gemm(
    a: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    glob: torch.Tensor,
    c: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    mul_routed_weight: bool,
    a_row_is_route: bool,
    act: int = 0,
) -> None:
    M, top_k = topk_ids.shape
    N = packed.shape[1]
    K = packed.shape[2] * 2
    scale = e4m3_kernel_view(scale)
    total_routes = M * top_k
    grid = (total_routes, triton.cdiv(N, _DECODE_BLOCK_N))
    _decode_nvfp4_moe_kernel[grid](
        a, packed, scale, glob, c, topk_weights, topk_ids,
        _e2m1_lut(a.device.index),
        total_routes, N, K,
        a.stride(0), a.stride(1),
        packed.stride(0), packed.stride(1), packed.stride(2),
        scale.stride(0), scale.stride(1), scale.stride(2),
        glob.stride(0), glob.stride(1),
        c.stride(0), c.stride(1), c.stride(2),
        topk_weights.stride(0), topk_weights.stride(1),
        topk_ids.stride(0), topk_ids.stride(1),
        BLOCK_SIZE_N=_DECODE_BLOCK_N,
        BLOCK_SIZE_KB=_DECODE_BLOCK_KB,
        TOP_K=top_k,
        A_ROW_IS_ROUTE=a_row_is_route,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        ACT=act,
        compute_type=_tl_dtype(c.dtype),
        num_warps=_DECODE_WARPS,
    )


def _decode_gemm_marlin(
    a: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    glob: torch.Tensor,
    c: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    mul_routed_weight: bool,
    a_row_is_route: bool,
    act: int = 0,
    cfg: Dict[str, int] | None = None,
) -> None:
    """Marlin-style decode GEMV: int32 wide loads + deferred reduction
    (:func:`_decode_nvfp4_marlin_kernel`). ``packed`` is the uint8 ``[S, N, K//2]`` bank;
    it is reinterpreted as int32 ``[S, N, K//8]`` (contiguous, K%8==0 for NVFP4).

    ``cfg`` overrides the :func:`decode_marlin_config` lookup (the offline sweep passes
    candidates in); production callers leave it ``None``."""
    M, top_k = topk_ids.shape
    N = packed.shape[1]
    K = packed.shape[2] * 2
    if cfg is None:
        cfg = decode_marlin_config(N, K, top_k, _sm_count(a.device.index))
    packed_i32 = packed.view(torch.int32)  # [S, N, K // 8]
    scale = e4m3_kernel_view(scale)
    total_routes = M * top_k
    grid = (total_routes, triton.cdiv(N, cfg["BLOCK_SIZE_N"]))
    _decode_nvfp4_marlin_kernel[grid](
        a, packed_i32, scale, glob, c, topk_weights, topk_ids,
        _e2m1_lut(a.device.index),
        total_routes, N, K,
        a.stride(0), a.stride(1),
        packed_i32.stride(0), packed_i32.stride(1), packed_i32.stride(2),
        scale.stride(0), scale.stride(1), scale.stride(2),
        glob.stride(0), glob.stride(1),
        c.stride(0), c.stride(1), c.stride(2),
        topk_weights.stride(0), topk_weights.stride(1),
        topk_ids.stride(0), topk_ids.stride(1),
        BLOCK_SIZE_N=cfg["BLOCK_SIZE_N"],
        BLOCK_SIZE_KW=cfg["BLOCK_SIZE_KW"],
        TOP_K=top_k,
        A_ROW_IS_ROUTE=a_row_is_route,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        ACT=act,
        compute_type=_tl_dtype(c.dtype),
        num_warps=cfg["num_warps"],
    )


def _fused_experts_decode_nvfp4(
    gemm_fn,
    hidden_states: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    apply_router_weight_on_input: bool,
    act_alpha: float = 1.702,
    act_limit: float = 7.0,
) -> torch.Tensor:
    """Shared decode body (gemm1 -> act -> gemm2 -> sum-reduce); ``gemm_fn`` is either
    the marlin-style int32 GEMV (:func:`_decode_gemm_marlin`) or the original LUT-gather
    GEMV (:func:`_decode_gemm`), both with the same calling convention."""
    M, H = hidden_states.shape
    top_k = topk_ids.shape[1]
    two_i = gate_up_packed.shape[1]
    inter = two_i if activation == "relu2" else two_i // 2
    dev, dt = hidden_states.device, hidden_states.dtype
    act = _act_code(activation)

    ic1 = torch.empty((M, top_k, two_i), device=dev, dtype=dt)
    gemm_fn(
        hidden_states, gate_up_packed, gate_up_scale, gate_up_global,
        ic1, topk_weights, topk_ids, apply_router_weight_on_input, False, act,
    )
    if act:
        # gemm1's epilogue already applied it in fp32; ic1 IS the activated [M*top_k, I].
        ic2 = ic1.view(-1, two_i)
    else:
        ic2 = torch.empty((M * top_k, inter), device=dev, dtype=dt)
        _run_act(activation, ic1.view(-1, two_i), ic2, act_alpha, act_limit)
    ic3 = torch.empty((M, top_k, H), device=dev, dtype=dt)
    gemm_fn(
        ic2, down_packed, down_scale, down_global,
        ic3, topk_weights, topk_ids, not apply_router_weight_on_input, True, 0,
    )
    out = torch.empty_like(hidden_states)
    moe_sum_reduce_triton(ic3, out)
    return out


def fused_experts_decode_nvfp4_marlin(
    hidden_states: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    act_alpha: float = 1.702,
    act_limit: float = 7.0,
) -> torch.Tensor:
    """Decode inline-NVFP4 MoE using the Marlin-style int32 wide-load GEMV."""
    return _fused_experts_decode_nvfp4(
        _decode_gemm_marlin,
        hidden_states, gate_up_packed, gate_up_scale, gate_up_global,
        down_packed, down_scale, down_global,
        topk_weights, topk_ids, activation, apply_router_weight_on_input,
        act_alpha, act_limit,
    )


def fused_experts_decode_nvfp4_serial(
    hidden_states: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    act_alpha: float = 1.702,
    act_limit: float = 7.0,
) -> torch.Tensor:
    """Original LUT-gather decode (one program per route, full K reduction). Retained for
    A/B benchmarking against the marlin decode path; not on the production decode path."""
    return _fused_experts_decode_nvfp4(
        _decode_gemm,
        hidden_states, gate_up_packed, gate_up_scale, gate_up_global,
        down_packed, down_scale, down_global,
        topk_weights, topk_ids, activation, apply_router_weight_on_input,
        act_alpha, act_limit,
    )


# M buckets the prefill JSON configs are keyed by (``benchmarks/tune_nvfp4_moe.py``).
PREFILL_M_BUCKETS = (16, 64, 256, 1024, 4096, 8192)

PREFILL_CONFIG_KEYS = (
    "BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_KB", "GROUP_SIZE_M", "num_warps", "num_stages"
)


def _prefill_config_default(M: int) -> Dict[str, int]:
    # ``BLOCK_SIZE_M`` is coupled to host-side ``moe_align_block_size`` (token padding),
    # so it cannot be picked by triton.autotune; these were chosen by an offline sweep
    # over (BLOCK_M, BLOCK_N, BLOCK_KB, num_warps, num_stages) for the MiniMax-M2 shapes.
    # This is the fallback whenever no tuned JSON exists for the (E, N, K, device).
    if M <= 64:
        return dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_KB=32,
                    GROUP_SIZE_M=1, num_warps=8, num_stages=4)
    return dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=64, BLOCK_SIZE_KB=32,
                GROUP_SIZE_M=8, num_warps=8, num_stages=4)


def nvfp4_config_filename(num_experts: int, N: int, K: int, device_name: str) -> str:
    """``configs/triton_<ver>/`` basename for a tuned prefill table.

    Mirrors :func:`freetoken.moe.fused._get_tuned_moe_configs`' naming, with the
    ``nvfp4,`` prefix (these drive a different kernel) and ``K`` in the key (the NVFP4
    grouped GEMM is tuned per GEMM: gate_up is ``[I, H]``, down is ``[H, I]``)."""
    return f"nvfp4,E={num_experts},N={N},K={K},device_name={device_name}.json"


def _device_name(device: Any) -> str | None:
    """``torch.device`` / index / literal name / None -> the config-file device name."""
    if isinstance(device, str):
        return device.replace(" ", "_")
    if not torch.cuda.is_available():
        return None
    if isinstance(device, torch.device):
        device = device.index
    return torch.cuda.get_device_name(device).replace(" ", "_")


@functools.lru_cache(maxsize=32)
def _load_nvfp4_moe_configs(
    num_experts: int, N: int, K: int, device_name: str, triton_version: str
) -> Dict[int, Dict[str, int]] | None:
    file_name = nvfp4_config_filename(num_experts, N, K, device_name)
    version_dir = f"triton_{triton_version.replace('.', '_')}"
    roots = []
    if env_dir := os.environ.get("FREETOKEN_MOE_CONFIG_DIR"):
        roots.append(Path(env_dir))
    roots.append(Path(__file__).with_name("configs"))
    for root in roots:
        path = root / version_dir / file_name
        if not path.exists():
            continue
        with path.open() as f:
            configs = json.load(f)
        return {int(bucket): cfg for bucket, cfg in configs.items()}
    return None


def nvfp4_moe_config(
    M: int, N: int, K: int, device: Any = None, num_experts: int = 128
) -> Dict[str, int]:
    """Prefill grouped-GEMM config for one NVFP4 expert GEMM of shape ``[N, K]``.

    Reads the tuned M-bucketed JSON for ``(num_experts, N, K, device, triton version)``
    and returns the nearest bucket; falls back to :func:`_prefill_config_default` when
    the file, the device or the triton version has no table. ``device`` accepts a
    ``torch.device``, a device index, a literal device name, or ``None`` (current
    device)."""
    name = _device_name(device)
    if name is not None:
        configs = _load_nvfp4_moe_configs(num_experts, N, K, name, triton.__version__)
        if configs:
            bucket = min(configs, key=lambda b: abs(b - M))
            return dict(configs[bucket])
    return _prefill_config_default(M)


def _prefill_gemm(
    a: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    glob: torch.Tensor,
    c: torch.Tensor,
    topk_weights_flat: torch.Tensor,
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    num_valid_tokens: int,
    kernel_top_k: int,
    mul_routed_weight: bool,
    cfg: Dict[str, Any],
    act: int = 0,
) -> None:
    N = packed.shape[1]
    K = packed.shape[2] * 2
    EM = sorted_ids.shape[0]
    scale = e4m3_kernel_view(scale)
    grid = lambda META: (  # noqa: E731
        triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    _prefill_nvfp4_moe_kernel[grid](
        a, packed, scale, glob, c, topk_weights_flat, sorted_ids, expert_ids,
        num_tokens_post_padded,
        _e2m1_lut(a.device.index),
        N, K, EM, num_valid_tokens,
        a.stride(0), a.stride(1),
        packed.stride(0), packed.stride(1), packed.stride(2),
        scale.stride(0), scale.stride(1), scale.stride(2),
        glob.stride(0), glob.stride(1),
        c.stride(1), c.stride(2),
        topk_weights_flat.stride(0),
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=kernel_top_k,
        ACT=act,
        compute_type=_tl_dtype(c.dtype),
        **cfg,
    )


def fused_experts_nvfp4(
    hidden_states: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    act_alpha: float = 1.702,
    act_limit: float = 7.0,
) -> torch.Tensor:
    """Prefill inline-NVFP4 MoE. ``topk_ids`` index rows of the bank tensors in
    ``[0, num_experts)``: full-layer banks with position == expert id (the
    materialized ``[:E]`` slot view or the overlap double buffer), raw ids."""
    M, H = hidden_states.shape
    top_k = topk_ids.shape[1]
    two_i = gate_up_packed.shape[1]
    inter = two_i if activation == "relu2" else two_i // 2
    dev, dt = hidden_states.device, hidden_states.dtype
    act = _act_code(activation)

    # Each GEMM gets its own tuned tile, but ``BLOCK_SIZE_M`` is host-coupled through
    # ``moe_align_block_size`` (one sorted-token padding shared by both), so gate_up's
    # choice wins -- which is what the tuner optimizes the pair under.
    cfg = nvfp4_moe_config(M, two_i, H, dev, num_experts)
    cfg_down = nvfp4_moe_config(M, down_packed.shape[1], inter, dev, num_experts)
    cfg_down["BLOCK_SIZE_M"] = cfg["BLOCK_SIZE_M"]

    sorted_ids, expert_ids, ntpp = moe_align_block_size(topk_ids, cfg["BLOCK_SIZE_M"], num_experts)
    tw = topk_weights.reshape(-1).contiguous()
    num_valid = topk_ids.numel()

    ic1 = torch.empty((M, top_k, two_i), device=dev, dtype=dt)
    _prefill_gemm(
        hidden_states, gate_up_packed, gate_up_scale, gate_up_global, ic1,
        tw, sorted_ids, expert_ids, ntpp, num_valid, top_k,
        apply_router_weight_on_input, cfg, act,
    )
    if act:
        ic2 = ic1.view(-1, two_i)  # activated in gemm1's epilogue, in fp32
    else:
        ic2 = torch.empty((M * top_k, inter), device=dev, dtype=dt)
        _run_act(activation, ic1.view(-1, two_i), ic2, act_alpha, act_limit)
    ic3 = torch.empty((M, top_k, H), device=dev, dtype=dt)
    _prefill_gemm(
        ic2, down_packed, down_scale, down_global, ic3,
        tw, sorted_ids, expert_ids, ntpp, num_valid, 1,
        not apply_router_weight_on_input, cfg_down,
    )
    out = torch.empty_like(hidden_states)
    moe_sum_reduce_triton(ic3, out)
    return out


__all__ = [
    "PREFILL_M_BUCKETS",
    "decode_marlin_config",
    "nvfp4_config_filename",
    "nvfp4_moe_config",
    "fused_experts_decode_nvfp4_marlin",
    "fused_experts_decode_nvfp4_serial",
    "fused_experts_nvfp4",
]
