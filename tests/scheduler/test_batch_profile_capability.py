"""The GPU batch profile must not make a Scheduler unconstructible without a GPU."""

from __future__ import annotations

from types import SimpleNamespace

import torch

import freetoken.scheduler.scheduler as scheduler_module
from freetoken.scheduler.scheduler import (
    _auto_small_prompt_group_tokens,
    _device_compute_capability,
)


def _config(max_prefill_seqs=None):
    return SimpleNamespace(max_prefill_seqs=max_prefill_seqs)


def test_capability_of_a_cpu_device_is_none_without_touching_cuda(monkeypatch):
    def explode(*_args, **_kwargs):  # a CPU device must never reach the CUDA query
        raise AssertionError("get_device_capability called for a non-CUDA device")

    monkeypatch.setattr(torch.cuda, "get_device_capability", explode)
    assert _device_compute_capability(torch.device("cpu")) is None


def test_capability_is_none_when_cuda_is_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("queried a missing GPU")),
    )
    assert _device_compute_capability(torch.device("cuda:0")) is None


def test_capability_is_reported_verbatim_when_cuda_is_present(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (8, 9))
    assert _device_compute_capability(torch.device("cuda:0")) == (8, 9)


def test_ada_crossover_is_unchanged_and_no_gpu_disables_grouping():
    assert _auto_small_prompt_group_tokens(_config(), 1, (8, 9)) == 1536  # Ada
    assert _auto_small_prompt_group_tokens(_config(), 1, (12, 0)) == 1280  # Blackwell
    assert _auto_small_prompt_group_tokens(_config(), 1, None) == 0  # no GPU
    # An explicit config or a multi-sequence prefill limit still wins over the profile.
    assert _auto_small_prompt_group_tokens(_config(max_prefill_seqs=2), 2, (8, 9)) == 0


def test_scheduler_init_reads_the_capability_through_the_guard():
    """Regression: the profile lookup was an unconditional torch.cuda call."""
    source = scheduler_module.Scheduler.__init__.__code__.co_names
    assert "_device_compute_capability" in source
    assert "get_device_capability" not in source
