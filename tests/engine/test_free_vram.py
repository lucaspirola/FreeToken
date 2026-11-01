"""Free VRAM below the WSL2 cudaMemGetInfo cliff (engine/free_vram.py).

The numbers are the 1 GiB steps measured on the RTX 5080 WSL2 host on 2026-11-02:
CUDA 4.54 / 3.54 / 0.00 GiB free while NVML read 5.26 / 4.26 / 3.26. With the 0
reading, the bounded-mirror start of LFM2.5-24B planned its 131072-token KV ceiling
against no free VRAM and was refused. CPU-only: torch.cuda is patched.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.engine.engine import Engine
from freetoken.engine.free_vram import FreeVramReader

GIB = 2**30
CUDA = [int(4.54 * GIB), int(3.54 * GIB), 0]
NVML = [int(5.26 * GIB), int(4.26 * GIB), int(3.26 * GIB)]
OFFSET = NVML[0] - CUDA[0]


def _patch_cuda(monkeypatch, cuda_values):
    values = iter(cuda_values)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (next(values), 16 * GIB))
    for name in ("synchronize", "empty_cache", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, lambda *a, **k: None)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *a, **k: None)


def _nvml(values):
    it = iter(values)
    return lambda: next(it)


def test_engine_plans_with_the_physical_free_vram_below_the_cliff(monkeypatch):
    _patch_cuda(monkeypatch, CUDA)
    engine = SimpleNamespace(device=torch.device("cuda", 0), tp_cpu_group=None)
    engine._free_vram = FreeVramReader(engine.device, nvml=_nvml(NVML))
    readings = [Engine._sync_get_memory(engine)[0] for _ in CUDA]
    assert readings[:2] == CUDA[:2]
    assert readings[2] == NVML[2] - OFFSET > 2 * GIB


def test_a_positive_cuda_reading_is_returned_unchanged(monkeypatch):
    _patch_cuda(monkeypatch, [7 * GIB, 1 * GIB])
    reader = FreeVramReader(torch.device("cuda", 0), nvml=_nvml([9 * GIB, 2 * GIB]))
    assert [reader.read(), reader.read()] == [7 * GIB, 1 * GIB]


def test_zero_stays_zero_without_nvml_or_without_a_calibration(monkeypatch):
    _patch_cuda(monkeypatch, [0, 0])
    no_nvml = FreeVramReader(torch.device("cuda", 0), nvml=lambda: None)
    assert no_nvml.read() == 0
    uncalibrated = FreeVramReader(torch.device("cuda", 0), nvml=_nvml([3 * GIB]))
    assert uncalibrated.read() == 0
