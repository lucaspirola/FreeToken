"""Stable-address CUDA virtual-memory tensors for incrementally committed KV slabs."""

from __future__ import annotations

import functools
import hashlib
import logging
import pathlib
import time

import torch

_CSRC = pathlib.Path(__file__).parent / "csrc" / "vmm_tensor.cpp"

logger = logging.getLogger(__name__)

# A build of this extension takes well under a minute; a lock older than this belongs to
# a build that was killed (torch's FileBaton only checks that the file exists, so every
# later start would otherwise wait on it forever -- 2026-09-24, every local start hung).
_STALE_LOCK_SECONDS = 15 * 60


def _extension_name() -> str:
    """One build per source file and content.

    Worktrees share ~/.cache/torch_extensions; under a single name each worktree rewrote
    the other's build.ninja (absolute source path) and forced a rebuild on every switch.
    """
    digest = hashlib.sha1(str(_CSRC.resolve()).encode() + _CSRC.read_bytes()).hexdigest()
    return f"freetoken_vmm_tensor_{digest[:12]}"


def _clear_stale_lock(name: str) -> None:
    from torch.utils.cpp_extension import _get_build_directory

    lock = pathlib.Path(_get_build_directory(name, verbose=False)) / "lock"
    try:
        age = time.time() - lock.stat().st_mtime
    except FileNotFoundError:
        return
    if age > _STALE_LOCK_SECONDS:
        logger.warning("Removing stale torch extension lock %s (%.0f min old)", lock, age / 60)
        lock.unlink(missing_ok=True)


@functools.cache
def _module():
    from torch.utils.cpp_extension import CUDA_HOME, load

    if CUDA_HOME is None:
        raise RuntimeError("CUDA_HOME is required to build the VMM tensor extension")
    cuda_root = pathlib.Path(CUDA_HOME)
    target_lib = cuda_root / "targets" / "x86_64-linux" / "lib"
    runtime_lib = target_lib if target_lib.is_dir() else cuda_root / "lib64"

    name = _extension_name()
    _clear_stale_lock(name)
    return load(
        name=name,
        sources=[str(_CSRC)],
        extra_include_paths=[str(cuda_root / "include")],
        extra_cflags=["-O3", "-std=c++17"],
        extra_ldflags=[
            f"-L{cuda_root / 'lib64' / 'stubs'}",
            f"-L{runtime_lib}",
            "-lcuda",
            "-lcudart",
        ],
        verbose=True,
    )


_VMM_RETRIES = 6


def allocation_granularity(device: torch.device) -> int:
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(f"CUDA VMM requires a CUDA device, got {device}")
    index = device.index if device.index is not None else torch.cuda.current_device()
    return int(_module().allocation_granularity(index))


class VMMTensor:
    """A CUDA tensor with a stable reserved address and explicitly mapped ranges."""

    _DTYPE_NAMES = {
        torch.uint8: "uint8",
        torch.int8: "int8",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
        # NVFP4 expert-bank scales (see parse_dtype in csrc/vmm_tensor.cpp).
        torch.float8_e4m3fn: "float8_e4m3fn",
        torch.float8_e5m2: "float8_e5m2",
        # b12x/flashinfer packs its NVFP4 codes into an int32 bank; growable KV makes the
        # MoE device banks VMM-backed, so that pairing needs the integer dtypes too.
        torch.int16: "int16",
        torch.int32: "int32",
        torch.int64: "int64",
    }

    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype,
        device: torch.device,
        reserved_bytes: int | None = None,
        initial_ranges: list[tuple[int, int]] | None = None,
    ) -> None:
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError(f"VMMTensor requires a CUDA device, got {device}")
        index = device.index if device.index is not None else torch.cuda.current_device()
        try:
            dtype_name = self._DTYPE_NAMES[dtype]
        except KeyError:
            raise ValueError(f"unsupported VMM tensor dtype: {dtype}") from None
        tensor_bytes = int(torch.empty((), dtype=dtype).element_size())
        for dim in shape:
            tensor_bytes *= int(dim)
        module = _module()
        if initial_ranges is None:
            granularity = int(module.allocation_granularity(index))
            initial_ranges = [(0, granularity)]
        elif not initial_ranges:
            raise ValueError("VMMTensor needs at least one initially mapped range")
        allocation = _module().VMMAllocation(
            list(shape),
            dtype_name,
            index,
            reserved_bytes or tensor_bytes,
            initial_ranges,
        )
        self._allocation = allocation
        self.tensor: torch.Tensor = allocation.tensor

    @property
    def granularity(self) -> int:
        return int(self._allocation.granularity)

    @property
    def reserved_bytes(self) -> int:
        return int(self._allocation.reserved_bytes)

    @property
    def mapped_bytes(self) -> int:
        return int(self._allocation.mapped_bytes)

    def commit_ranges(self, ranges: list[tuple[int, int]]) -> None:
        """Map physical pages into the reserved VA, retrying while the device is busy.

        cuMemSetAccess is issued from the host, but it is ordered against work the
        device has already been handed: when a range is (re-)mapped right after the
        growable KV released pages, the driver can answer CUDA_ERROR_NOT_READY --
        "ask again", not "this failed". It is not sticky and it does not poison the
        context, so the only correct response is to drain the device and retry.
        Raising instead kills the scheduler worker, which is how a 1M request that
        prefilled and decoded perfectly still took the server down on teardown
        (2026-09-22, tasks/exclusive-expert-ram/results/nemotron-1m-only-journal.txt).
        """
        self._retry_while_not_ready(self._allocation.commit_ranges, ranges)

    @staticmethod
    def _retry_while_not_ready(fn, ranges: list[tuple[int, int]]) -> None:
        # Six attempts over ~1.3 s. The wait is a device synchronize first (the
        # cheap, correct barrier) and only then wall-clock backoff.
        delay = 0.01
        for attempt in range(_VMM_RETRIES):
            try:
                fn(ranges)
                return
            except RuntimeError as exc:
                if "CUDA_ERROR_NOT_READY" not in str(exc) or attempt == _VMM_RETRIES - 1:
                    raise
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                time.sleep(delay)
                delay *= 2

    def uncommit_ranges(self, ranges: list[tuple[int, int]]) -> None:
        """Unmap fully committed, granularity-aligned ranges without moving the tensor.

        Same NOT_READY retry as commit_ranges: the unmap is the other half of the
        arena/KV handover and can meet the device equally busy.
        """
        self._retry_while_not_ready(self._allocation.uncommit_ranges, ranges)


__all__ = ["VMMTensor", "allocation_granularity"]
