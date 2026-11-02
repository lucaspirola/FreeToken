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

The physical limit is lower than the cliff: past about 0.2-0.3 GiB of NVML free,
cuMemSetAccess returns CUDA_ERROR_NOT_READY, which breaks torch's caching allocator for
the life of the process. Every reading is therefore also capped at NVML free minus
``nvml_floor_bytes()`` (0.40 GiB on WSL2), whatever offset the host shows.
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


# NVML free VRAM no plan may go below on WSL2. The cliff map (.work/lfm25/notready,
# 120 runs) saw the first CUDA_ERROR_NOT_READY at 0.19-0.30 GiB NVML free, probing in
# 64 MiB steps: 0.30 + 0.0625 rounds up to 0.40 GiB.
WSL_NVML_FLOOR_BYTES = 400 * 1024 * 1024


def _is_wsl() -> bool:
    return os.path.exists("/dev/dxg")


def nvml_floor_bytes() -> int:
    """``FREETOKEN_NVML_FLOOR_MB`` if set, else ``WSL_NVML_FLOOR_BYTES`` under WSL2, else 0."""
    raw = os.environ.get("FREETOKEN_NVML_FLOOR_MB", "").strip()
    if raw:
        return int(float(raw) * 1024 * 1024)
    return WSL_NVML_FLOOR_BYTES if _is_wsl() else 0


class FreeVramReader:
    """Free VRAM on one device; see the module docstring."""

    def __init__(self, device: torch.device, nvml=None, floor: "int | None" = None) -> None:
        self.device = device
        self.floor = nvml_floor_bytes() if floor is None else int(floor)
        self._nvml = nvml
        self._nvml_opened = nvml is not None
        self.offset: "int | None" = None
        self._logged = False
        self._floor_logged = False

    def _cuda_free(self) -> int:
        return int(torch.cuda.mem_get_info(self.device)[0])

    def nvml_free(self) -> "int | None":
        """NVML free bytes, or None when NVML cannot serve this device."""
        return self._nvml_free()

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
        ceiling = max(nvml_free - self.floor, 0)
        if cuda_free > 0:
            self.offset = max(nvml_free - cuda_free, 0)
            if ceiling < cuda_free and not self._floor_logged:
                self._floor_logged = True
                logger.info(
                    "Free VRAM capped by NVML: CUDA %s, NVML %s minus the %s floor = %s",
                    mem_GB(cuda_free), mem_GB(nvml_free), mem_GB(self.floor), mem_GB(ceiling),
                )
            return min(cuda_free, ceiling)
        if self.offset is None:
            return cuda_free
        free = min(max(nvml_free - self.offset, 0), ceiling)
        if free > 0 and not self._logged:
            self._logged = True
            logger.info(
                "cudaMemGetInfo reports 0 free, NVML %s: planning with %s "
                "(NVML minus the %s offset measured while CUDA still reported)",
                mem_GB(nvml_free), mem_GB(free), mem_GB(self.offset),
            )
        return free


class PrefillMemoryProbe:
    """``FREETOKEN_PREFILL_MEM_PROBE=1``: per eager prefill forward, the caching
    allocator's reserved rise and the minimum NVML free VRAM (a 1 ms sampling thread).
    Diagnostic only: it synchronizes before and after each prefill forward."""

    enabled = os.environ.get("FREETOKEN_PREFILL_MEM_PROBE", "0").strip() not in (
        "", "0", "false", "no", "off",
    )

    def __init__(self, reader: FreeVramReader) -> None:
        self.reader = reader
        self._state = None

    def begin(self, tokens: int, cached: int) -> None:
        import threading

        dev = self.reader.device
        torch.cuda.synchronize(dev)
        nvml0 = self.reader.nvml_free()
        state = {
            "tokens": tokens, "cached": cached, "nvml0": nvml0, "nvml_min": nvml0,
            "reserved0": int(torch.cuda.memory_reserved(dev)),
            "allocated0": int(torch.cuda.memory_allocated(dev)),
            "stop": threading.Event(),
        }
        torch.cuda.reset_peak_memory_stats(dev)

        def sample() -> None:
            while not state["stop"].wait(0.001):
                free = self.reader.nvml_free()
                if free is not None and (state["nvml_min"] is None or free < state["nvml_min"]):
                    state["nvml_min"] = free

        state["thread"] = threading.Thread(target=sample, daemon=True)
        state["thread"].start()
        self._state = state

    def end(self) -> None:
        state, self._state = self._state, None
        if state is None:
            return
        dev = self.reader.device
        torch.cuda.synchronize(dev)
        state["stop"].set()
        state["thread"].join()
        nvml1 = self.reader.nvml_free()
        gib = lambda b: "n/a" if b is None else f"{b / 2**30:.3f}"  # noqa: E731
        logger.info(
            "Prefill memory probe: %d new tokens after %d cached; reserved %s -> peak +%s GiB, "
            "allocated peak +%s GiB; NVML free %s -> min %s -> %s GiB; cuda free %s GiB",
            state["tokens"], state["cached"], gib(state["reserved0"]),
            gib(torch.cuda.max_memory_reserved(dev) - state["reserved0"]),
            gib(torch.cuda.max_memory_allocated(dev) - state["allocated0"]),
            gib(state["nvml0"]), gib(state["nvml_min"]), gib(nvml1),
            gib(int(torch.cuda.mem_get_info(dev)[0])),
        )
