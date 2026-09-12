"""Cheap opt-in CUDA allocator telemetry at scheduler reply-batch boundaries.

This module never synchronizes CUDA or resets allocator peak counters. PyTorch allocator
values exclude allocations outside PyTorch. ``driver_free_bytes`` is an observation at a
sample instant, not a true driver-free minimum over the workload.
"""

from __future__ import annotations

import time
from typing import Any


PEAK_SCOPE = "since_last_torch_cuda_reset_peak_memory_stats"


def allocator_snapshot(torch_module: Any, device: Any, enabled: bool) -> dict | None:
    """Return one no-sync snapshot, or ``None`` without touching CUDA when disabled."""
    if not enabled:
        return None
    if getattr(device, "type", None) != "cuda":
        return {"available": False, "peak_scope": PEAK_SCOPE, "request_scoped": False}
    try:
        cuda = torch_module.cuda
        driver_free, driver_total = cuda.mem_get_info(device)
        return {
            "available": True,
            "peak_scope": PEAK_SCOPE,
            "request_scoped": False,
            "sample_timestamp_ns": time.monotonic_ns(),
            "current_allocated_bytes": int(cuda.memory_allocated(device)),
            "current_reserved_bytes": int(cuda.memory_reserved(device)),
            "peak_allocated_bytes": int(cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(cuda.max_memory_reserved(device)),
            "driver_free_bytes": int(driver_free),
            "driver_total_bytes": int(driver_total),
            "driver_free_is_sampled_bound": True,
            "pytorch_allocator_excludes_external_cuda_allocations": True,
        }
    except Exception:  # optional telemetry must never affect serving replies
        return {"available": False, "peak_scope": PEAK_SCOPE, "request_scoped": False}
