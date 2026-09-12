"""CPU-only contract tests for opt-in allocator snapshots."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace


_PATH = Path(__file__).resolve().parents[2] / "python/freetoken/engine/cuda_memory.py"
_SPEC = importlib.util.spec_from_file_location("cuda_memory_telemetry", _PATH)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
PEAK_SCOPE = _MODULE.PEAK_SCOPE
allocator_snapshot = _MODULE.allocator_snapshot

_STATS_PATH = Path(__file__).resolve().parents[2] / "python/freetoken/server/stats.py"
_STATS_SPEC = importlib.util.spec_from_file_location("server_stats", _STATS_PATH)
assert _STATS_SPEC and _STATS_SPEC.loader
_STATS_MODULE = importlib.util.module_from_spec(_STATS_SPEC)
_STATS_SPEC.loader.exec_module(_STATS_MODULE)
StatsTracker = _STATS_MODULE.StatsTracker


class FakeCuda:
    def __init__(self):
        self.calls = []

    def memory_allocated(self, device):
        self.calls.append("allocated")
        return 11

    def memory_reserved(self, device):
        self.calls.append("reserved")
        return 22

    def max_memory_allocated(self, device):
        self.calls.append("peak_allocated")
        return 33

    def max_memory_reserved(self, device):
        self.calls.append("peak_reserved")
        return 44

    def mem_get_info(self, device):
        self.calls.append("mem_get_info")
        return 55, 66


def test_disabled_does_not_touch_cuda():
    cuda = FakeCuda()
    assert allocator_snapshot(SimpleNamespace(cuda=cuda), SimpleNamespace(type="cuda"), False) is None
    assert cuda.calls == []


def test_cuda_snapshot_has_allocator_driver_and_monotonic_timestamp_values():
    cuda = FakeCuda()
    snapshot = allocator_snapshot(SimpleNamespace(cuda=cuda), SimpleNamespace(type="cuda"), True)
    assert snapshot["available"] is True
    assert snapshot["peak_scope"] == PEAK_SCOPE
    assert snapshot["request_scoped"] is False
    assert snapshot["current_allocated_bytes"] == 11
    assert snapshot["current_reserved_bytes"] == 22
    assert snapshot["peak_allocated_bytes"] == 33
    assert snapshot["peak_reserved_bytes"] == 44
    assert snapshot["driver_free_bytes"] == 55
    assert snapshot["driver_total_bytes"] == 66
    assert isinstance(snapshot["sample_timestamp_ns"], int)
    assert snapshot["driver_free_is_sampled_bound"] is True
    assert snapshot["pytorch_allocator_excludes_external_cuda_allocations"] is True
    assert cuda.calls == ["mem_get_info", "allocated", "reserved", "peak_allocated", "peak_reserved"]


def test_cpu_snapshot_never_touches_cuda():
    cuda = FakeCuda()
    snapshot = allocator_snapshot(SimpleNamespace(cuda=cuda), SimpleNamespace(type="cpu"), True)
    assert snapshot == {"available": False, "peak_scope": PEAK_SCOPE, "request_scoped": False}
    assert cuda.calls == []


def test_stats_counts_one_batch_snapshot_once_and_keeps_min_driver_free():
    tracker = StatsTracker()
    first = {"sample_timestamp_ns": 1, "driver_free_bytes": 55}
    same_batch_reply = {"sample_timestamp_ns": 1, "driver_free_bytes": 55}
    second = {"sample_timestamp_ns": 2, "driver_free_bytes": 40}
    tracker.observe(SimpleNamespace(cuda_memory=first))
    tracker.observe(SimpleNamespace(cuda_memory=same_batch_reply))
    tracker.observe(SimpleNamespace(cuda_memory=second))
    assert tracker.cuda_memory_sample_count == 2
    assert tracker.cuda_memory_min_driver_free_bytes == 40
    assert tracker.cuda_memory == second
