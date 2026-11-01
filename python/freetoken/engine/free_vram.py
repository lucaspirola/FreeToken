"""Free VRAM as the engine plans with it: ``cudaMemGetInfo``, corrected where it reads 0.

On the owner's WSL2 host (RTX 5080, 2026-11-02) ``cudaMemGetInfo`` reports 0 free once
its own figure would drop below about 3.4 GiB. NVML still reports the physical free
memory, and new allocations still land in VRAM at full bandwidth: 1 GiB steps read
4.54, 3.54, then 0.00 GiB while NVML read 5.26, 4.26, 3.26, and tensors up to 14 GiB
held all ran at ~900 GB/s. Above the cliff the two differ by a constant (0.72 GiB).

Startup and growable-KV decisions read free VRAM through ``Engine._sync_get_memory``,
so they saw 0 where about 2 GiB was free: the RAM-saver start of LFM2.5-24B planned its
KV ceiling against that 0 and was refused.

``FreeVramReader.read`` returns the CUDA figure whenever it is positive, which changes
nothing on a host without the cliff, and records the NVML-minus-CUDA offset at each such
reading. When CUDA reads 0 after an offset was recorded, it returns NVML free minus that
offset: the CUDA scale, continued below the cliff.
"""
from __future__ import annotations

import ctypes
import os

import torch

from freetoken.utils import init_logger, mem_GB

logger = init_logger(__name__)


class _NvmlMemory(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


def _open_nvml_device(uuid: "str | None"):
    """(library, handle) for the GPU with ``uuid``, or None when NVML cannot serve it."""
    if not uuid:
        return None
    names = ["nvml.dll"] if os.name == "nt" else ["libnvidia-ml.so.1"]
    for name in names:
        try:
            lib = ctypes.CDLL(name)
            break
        except OSError:
            continue
    else:
        return None
    try:
        if lib.nvmlInit_v2() != 0:
            return None
        handle = ctypes.c_void_p()
        if lib.nvmlDeviceGetHandleByUUID(uuid.encode("ascii"), ctypes.byref(handle)) != 0:
            return None
        return lib, handle
    except (OSError, AttributeError):
        return None


class FreeVramReader:
    """Free VRAM on one device; see the module docstring."""

    def __init__(self, device: torch.device, nvml=None) -> None:
        self.device = device
        self._nvml = nvml
        self._nvml_opened = nvml is not None
        self.offset: "int | None" = None
        self._logged = False

    def _cuda_free(self) -> int:
        return int(torch.cuda.mem_get_info(self.device)[0])

    def _nvml_free(self) -> "int | None":
        if not self._nvml_opened:
            self._nvml_opened = True
            from freetoken.gpu_select import gpu_identity

            index = self.device.index if self.device.index is not None else torch.cuda.current_device()
            self._nvml = _open_nvml_device(gpu_identity(index)["uuid"])
        if self._nvml is None:
            return None
        if callable(self._nvml):
            return self._nvml()
        lib, handle = self._nvml
        mem = _NvmlMemory()
        if lib.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(mem)) != 0:
            return None
        return int(mem.free)

    def read(self) -> int:
        cuda_free = self._cuda_free()
        nvml_free = self._nvml_free()
        if nvml_free is None:
            return cuda_free
        if cuda_free > 0:
            self.offset = max(nvml_free - cuda_free, 0)
            return cuda_free
        if self.offset is None:
            return cuda_free
        free = max(nvml_free - self.offset, 0)
        if free > 0 and not self._logged:
            self._logged = True
            logger.info(
                "cudaMemGetInfo reports 0 free, NVML %s: planning with %s "
                "(NVML minus the %s offset measured while CUDA still reported)",
                mem_GB(nvml_free), mem_GB(free), mem_GB(self.offset),
            )
        return free
