from __future__ import annotations

import math
import os
from functools import lru_cache
from typing import TYPE_CHECKING

import torch

from .utils import load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi import Module


DEFAULT_NUM_BLOCKS = 4
SKIP_FAST_INDEX_COPY_ENV = "FREETOKEN_SKIP_FAST_INDEX_COPY"
_TRUE_VALUES = {"1", "true", "yes", "on"}


def _skip_fast_index_copy_enabled() -> bool:
    return os.getenv(SKIP_FAST_INDEX_COPY_ENV, "").strip().lower() in _TRUE_VALUES


@lru_cache(maxsize=None)
def _jit_update_flag_module() -> Module:
    return load_jit(
        "fast_index_copy_update_flag",
        cuda_files=["fast_index_copy.cuh"],
        cuda_wrappers=[("update_copy_flag", "update_copy_flag")],
    )


@lru_cache(maxsize=None)
def _jit_fast_index_copy_module(
    *,
    feature_size: int,
    worker_threads: int,
    worker_feature_size: int,
    num_block: int,
) -> Module:
    args = make_cpp_args(
        feature_size,
        worker_threads,
        worker_feature_size,
        1024,
        num_block,
        1,
    )
    return load_jit(
        "fast_index_copy",
        *args,
        cuda_files=["fast_index_copy.cuh"],
        cuda_wrappers=[
            ("launch", f"&FastIndexCopyKernel<{args}>::run"),
            ("launch_high", f"&FastIndexCopyKernel<{args}>::run_high"),
            ("launch_normal", f"&FastIndexCopyKernel<{args}>::run_normal"),
        ],
    )


def _default_worker_threads(feature_size: int) -> int:
    if feature_size <= 1024:
        return 8
    if feature_size <= 2048:
        return 16
    return 32


def _shrink_worker_feature_size(feature_size: int, worker_feature_size: int) -> int:
    if feature_size < worker_feature_size:
        worker_feature_size = feature_size
    while feature_size % worker_feature_size != 0 and worker_feature_size > 128:
        worker_feature_size //= 2
    return worker_feature_size


def default_worker_args(feature_size: int) -> tuple[int, int, int]:
    """(worker_threads, worker_feature_size, num_block) of the default call path.

    The AOT spec builder (kernel/aot.py) compiles with exactly these values, so a
    default ``fast_index_copy_jit`` call hits the prebuilt cache by name.
    """
    return (
        _default_worker_threads(feature_size),
        _shrink_worker_feature_size(feature_size, 2048),
        DEFAULT_NUM_BLOCKS,
    )


def fast_index_copy_jit(
    dst: torch.Tensor,
    dst_indices: torch.Tensor,
    src: torch.Tensor,
    src_indices: torch.Tensor,
    num_indices: torch.Tensor | None = None,
    *,
    worker_threads: int | None = None,
    worker_feature_size: int = 2048,
    num_block: int | None = None,
    priority: str | None = None,
    sync_flag: torch.Tensor | None = None,
) -> None:
    num_dst_feature = math.prod(dst.shape[1:])
    num_src_feature = math.prod(src.shape[1:])
    assert num_src_feature == num_dst_feature

    # Debug/perf ablation: keep miss bookkeeping intact, but make the copy free to
    # approximate a zero-copy-miss runtime. Outputs are only meaningful if callers
    # already have valid cache contents for the requested indices.
    if _skip_fast_index_copy_enabled():
        return

    dst = dst.as_strided(size=(dst.size(0), num_dst_feature), stride=(num_dst_feature, 1))
    src = src.as_strided(size=(src.size(0), num_src_feature), stride=(num_src_feature, 1))

    feature_size = dst.size(-1) * dst.element_size()
    num_block = num_block or DEFAULT_NUM_BLOCKS
    worker_threads = worker_threads or _default_worker_threads(feature_size)
    worker_feature_size = _shrink_worker_feature_size(feature_size, worker_feature_size)
    assert worker_threads in (8, 16, 32)
    assert feature_size % worker_feature_size == 0

    module = _jit_fast_index_copy_module(
        feature_size=feature_size,
        worker_threads=worker_threads,
        worker_feature_size=worker_feature_size,
        num_block=num_block,
    )
    if priority is None:
        module.launch(dst, dst_indices, src, src_indices, num_indices)
        return

    assert priority in ("high", "normal")
    assert sync_flag is not None
    if priority == "high":
        module.launch_high(dst, dst_indices, src, src_indices, num_indices, sync_flag)
        return
    module.launch_normal(dst, dst_indices, src, src_indices, num_indices, sync_flag)


@lru_cache(maxsize=None)
def _jit_fast_index_copy_multi_module(*, num_threads: int, blocks_per_bank: int) -> Module:
    args = make_cpp_args(num_threads, blocks_per_bank)
    return load_jit(
        "fast_index_copy_multi",
        *args,
        cuda_files=["fast_index_copy.cuh"],
        cuda_wrappers=[("launch", f"&MultiIndexCopyKernel<{args}>::run")],
    )


@lru_cache(maxsize=None)
def _default_pcie_blocks_per_bank(device_index: int) -> int:
    # The RTX 2000 Ada's mapped-host gather reaches its 12.8 GB/s WSL H2D roofline
    # with 16 blocks/bank. Eight leaves small one-expert copies 10-12% below it.
    # Retain the established geometry on Blackwell and unmeasured architectures.
    return 16 if torch.cuda.get_device_capability(device_index) == (8, 9) else 8


def fast_index_copy_multi_jit(
    dst_ptrs: torch.Tensor,
    src_ptrs: torch.Tensor,
    feat_bytes: torch.Tensor,
    dst_indices: torch.Tensor,
    src_indices: torch.Tensor,
    num_indices: torch.Tensor | None = None,
    *,
    num_threads: int = 1024,
    blocks_per_bank: int | None = None,
) -> None:
    """Fused multi-bank index copy: copy the same rows for every bank in ONE launch.

    The copy is host->device PCIe-bandwidth bound. Datacenter GPUs use the established
    8x1024 launch; Ada sm_89 uses 16x1024 because its small one-to-two-expert decode
    copies otherwise underfill the WSL PCIe path. Launch count remains one regardless
    of grid width, preserving the win over the legacy per-bank kernel.

    ``dst_ptrs``/``src_ptrs``/``feat_bytes`` are int64 [num_banks] device tensors built
    once by the caller (the per-bank slot-cache base addr, host-source base addr, and
    per-row byte size). Every bank's per-row byte size must be a multiple of 16, and the
    base addresses 16-byte aligned (true for contiguous torch allocations of these banks).
    """
    if _skip_fast_index_copy_enabled():
        return
    if blocks_per_bank is None:
        device_index = dst_ptrs.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        blocks_per_bank = _default_pcie_blocks_per_bank(device_index)
    module = _jit_fast_index_copy_multi_module(
        num_threads=num_threads, blocks_per_bank=blocks_per_bank
    )
    module.launch(dst_ptrs, src_ptrs, feat_bytes, dst_indices, src_indices, num_indices)


@lru_cache(maxsize=None)
def _jit_fast_index_copy_multi_strided_module(
    *, num_threads: int, blocks_per_bank: int
) -> Module:
    args = make_cpp_args(num_threads, blocks_per_bank)
    return load_jit(
        "fast_index_copy_multi_strided",
        *args,
        cuda_files=["fast_index_copy.cuh"],
        cuda_wrappers=[("launch", f"&MultiIndexCopyStridedKernel<{args}>::run")],
    )


def fast_index_copy_multi_strided_jit(
    dst_ptrs: torch.Tensor,
    src_ptrs: torch.Tensor,
    feat_bytes: torch.Tensor,
    dst_strides: torch.Tensor,
    src_strides: torch.Tensor,
    dst_indices: torch.Tensor,
    src_indices: torch.Tensor,
    num_indices: torch.Tensor | None = None,
    *,
    num_threads: int = 1024,
    blocks_per_bank: int | None = None,
) -> None:
    """Fused index copy where compact source rows occupy a larger cache-row prefix."""
    if _skip_fast_index_copy_enabled():
        return
    if blocks_per_bank is None:
        device_index = dst_ptrs.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        blocks_per_bank = _default_pcie_blocks_per_bank(device_index)
    module = _jit_fast_index_copy_multi_strided_module(
        num_threads=num_threads, blocks_per_bank=blocks_per_bank
    )
    module.launch(
        dst_ptrs,
        src_ptrs,
        feat_bytes,
        dst_strides,
        src_strides,
        dst_indices,
        src_indices,
        num_indices,
    )


def update_copy_flag_jit(sync_flag: torch.Tensor, delta: int) -> None:
    assert sync_flag.is_cuda
    assert sync_flag.numel() == 1
    assert sync_flag.dtype == torch.int32
    _jit_update_flag_module().update_copy_flag(sync_flag, delta)
