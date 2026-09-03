"""Stable-address CUDA virtual-memory tensors for incrementally committed KV slabs."""

from __future__ import annotations

import functools
import pathlib

import torch

_CSRC = pathlib.Path(__file__).parent / "csrc" / "vmm_tensor.cpp"


@functools.cache
def _module():
    from torch.utils.cpp_extension import CUDA_HOME, load

    if CUDA_HOME is None:
        raise RuntimeError("CUDA_HOME is required to build the VMM tensor extension")
    cuda_root = pathlib.Path(CUDA_HOME)
    target_lib = cuda_root / "targets" / "x86_64-linux" / "lib"
    runtime_lib = target_lib if target_lib.is_dir() else cuda_root / "lib64"

    return load(
        name="freetoken_vmm_tensor",
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
        self._allocation.commit_ranges(ranges)

    def uncommit_ranges(self, ranges: list[tuple[int, int]]) -> None:
        """Unmap fully committed, granularity-aligned ranges without moving the tensor."""
        self._allocation.uncommit_ranges(ranges)


__all__ = ["VMMTensor", "allocation_granularity"]
