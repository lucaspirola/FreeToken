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
    reader = FreeVramReader(torch.device("cuda", 0), nvml=_nvml([9 * GIB, 2 * GIB]), floor=0)
    assert [reader.read(), reader.read()] == [7 * GIB, 1 * GIB]


def test_zero_stays_zero_without_nvml_or_without_a_calibration(monkeypatch):
    _patch_cuda(monkeypatch, [0, 0])
    no_nvml = FreeVramReader(torch.device("cuda", 0), nvml=lambda: None)
    assert no_nvml.read() == 0
    uncalibrated = FreeVramReader(torch.device("cuda", 0), nvml=_nvml([3 * GIB]))
    assert uncalibrated.read() == 0


# The cliff map (.work/lfm25/notready/cliff): the first CUDA_ERROR_NOT_READY came at
# 0.19-0.30 GiB NVML free. A reading must never let a plan spend below the floor.
FLOOR = 400 * 2**20


def test_reading_never_exceeds_nvml_free_minus_the_floor(monkeypatch):
    # A host whose CUDA figure tracks NVML closely (offset 0.05 GiB, below the floor).
    _patch_cuda(monkeypatch, [int(2.95 * GIB), 0])
    reader = FreeVramReader(
        torch.device("cuda", 0), nvml=_nvml([3 * GIB, int(1.0 * GIB)]), floor=FLOOR
    )
    first, below_cliff = reader.read(), reader.read()
    assert first == 3 * GIB - FLOOR
    assert below_cliff == int(1.0 * GIB) - FLOOR


def test_the_floor_does_not_cut_a_reading_that_is_already_stricter(monkeypatch):
    # This host: CUDA = NVML - 0.72 GiB, which is below NVML - floor.
    _patch_cuda(monkeypatch, CUDA)
    reader = FreeVramReader(torch.device("cuda", 0), nvml=_nvml(NVML), floor=FLOOR)
    assert [reader.read() for _ in CUDA] == [CUDA[0], CUDA[1], NVML[2] - OFFSET]


def test_nvml_below_the_floor_reads_zero(monkeypatch):
    _patch_cuda(monkeypatch, [int(0.1 * GIB)])
    reader = FreeVramReader(torch.device("cuda", 0), nvml=_nvml([int(0.3 * GIB)]), floor=FLOOR)
    assert reader.read() == 0


def test_floor_default_is_wsl_only(monkeypatch):
    from freetoken.engine import free_vram

    monkeypatch.delenv("FREETOKEN_NVML_FLOOR_MB", raising=False)
    monkeypatch.setattr(free_vram, "_is_wsl", lambda: True)
    assert free_vram.nvml_floor_bytes() == free_vram.WSL_NVML_FLOOR_BYTES
    monkeypatch.setattr(free_vram, "_is_wsl", lambda: False)
    assert free_vram.nvml_floor_bytes() == 0
    monkeypatch.setenv("FREETOKEN_NVML_FLOOR_MB", "256")
    assert free_vram.nvml_floor_bytes() == 256 * 2**20


def test_prefill_reserve_keeps_the_nvml_floor_when_the_offset_is_small(monkeypatch):
    """The crash run's decode level (2224 slots) read 0.14 GiB on the CUDA scale while
    NVML showed 0.44 GiB free: an offset near 0.3 GiB, not 0.72. A reserve planned on
    the CUDA scale then put the chunk's peak at NVML offset + margin, at the
    NOT_READY limit. The reserve must leave NVML free minus the transient above the floor."""
    from freetoken.engine import growable_kv
    from freetoken.engine.growable_kv import GrowableKvController

    offset = int(0.25 * GIB)
    floor = 400 * 2**20
    monkeypatch.setenv("FREETOKEN_NVML_FLOOR_MB", "400")
    monkeypatch.setattr(
        growable_kv, "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(synchronize=lambda *_: None, empty_cache=lambda: None),
            inference_mode=lambda: (lambda f: f),
        ),
    )
    ledger = {"cuda": int(0.125 * GIB)}
    row = 2 * 2**20

    class Arena:
        num_experts = 8
        arena_layout = (2048, 16)
        bank_row_bytes = [row]
        prefill_overlap = True
        residency = SimpleNamespace(min_gpu_slots=lambda: 0)
        cache_size = 1024

        def set_usable_slots(self, n):
            delta = (self.cache_size - n) * row
            ledger["cuda"] += delta
            self.cache_size = n
            return abs(delta)

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (ledger["cuda"], 16 * GIB))
    nvml = lambda: ledger["cuda"] + offset  # noqa: E731
    reader = FreeVramReader(torch.device("cuda", 0), nvml=nvml)
    moe = Arena()
    transient = 1 * GIB
    engine = SimpleNamespace(
        moe_offload_cache=moe, device="cuda", prefill_transient_bytes=transient,
        _growable_moe_ceiling=moe.cache_size, _free_vram=reader,
        config=SimpleNamespace(kv_grow_step_tokens=65536, moe_cache_size=moe.cache_size,
                               tp_info=SimpleNamespace(size=1)),
        _sync_get_memory=lambda: (reader.read(), reader.read()),
    )
    ctl = GrowableKvController(engine)
    engine.growable_kv = ctl
    ctl._decode_level = True
    ctl.reserve_prefill_headroom()
    assert nvml() - transient >= floor
