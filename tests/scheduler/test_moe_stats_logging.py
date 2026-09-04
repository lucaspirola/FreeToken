"""run_when_idle dumps the full per-layer MoE decode stats as parseable json.

CPU-only: run_when_idle is exercised as an unbound method over a stub scheduler, so no
engine, no GPU and no cache manager are needed."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


class _Recorder:
    def __init__(self):
        self.lines: list[str] = []

    def info_rank0(self, fmt, *args):
        self.lines.append(fmt % args)


def _per_layer(pageable_calls=0):
    return [
        {
            "layer": L,
            "steps": 64,
            "active_per_step": 8.0,
            "missing_per_step": 1.5,
            "miss_rate": 0.1875,
            "fetched_per_step": 1.5,
            "pageable_stage_calls": pageable_calls,
            "pageable_rows": 0,
            "pageable_plan_wait_seconds": 0.0,
            "pageable_gather_seconds": 0.0,
        }
        for L in range(3)
    ]


def _stub_scheduler(per_layer, collect_stats=True, calls=64):
    moe = SimpleNamespace(
        decode_miss_stats=lambda: {"layer_calls": calls, "miss_rate": 0.1875},
        decode_miss_stats_per_layer=lambda: {"per_layer": per_layer},
    )
    return SimpleNamespace(
        engine=SimpleNamespace(moe_offload_cache=moe),
        config=SimpleNamespace(moe_collect_stats=collect_stats),
        _last_moe_stats_calls=0,
        cache_manager=SimpleNamespace(check_integrity=lambda: None),
        _maybe_retune_pageable_layers=lambda rows: None,
    )


def _run(monkeypatch, stub):
    from freetoken.scheduler import scheduler as sched

    rec = _Recorder()
    monkeypatch.setattr(sched, "logger", rec)
    sched.Scheduler.run_when_idle(stub)
    return rec.lines


@pytest.mark.parametrize("pageable_calls", [0, 5])
def test_per_layer_stats_are_logged_as_json_for_every_layer(monkeypatch, pageable_calls):
    per_layer = _per_layer(pageable_calls)
    lines = _run(monkeypatch, _stub_scheduler(per_layer))

    prefix = "MoE decode miss stats per layer: "
    dumps = [ln for ln in lines if ln.startswith(prefix)]
    assert len(dumps) == 1, lines
    # Parses with json.loads (not repr) and covers all layers, pageable or not.
    assert json.loads(dumps[0][len(prefix):]) == per_layer
    # The aggregate line is still a python repr on its own line.
    assert any(ln.startswith("MoE decode miss stats: {") for ln in lines)


def test_nothing_is_logged_without_moe_collect_stats(monkeypatch):
    stub = _stub_scheduler(_per_layer(), collect_stats=False)
    lines = _run(monkeypatch, stub)
    assert not any("MoE decode miss stats" in ln for ln in lines)
    assert lines == ["Scheduler is idle, waiting for new reqs..."]


def test_repeat_idle_without_new_calls_does_not_redump(monkeypatch):
    stub = _stub_scheduler(_per_layer())
    stub._last_moe_stats_calls = 64  # same as the cache's layer_calls
    lines = _run(monkeypatch, stub)
    assert not any("MoE decode miss stats" in ln for ln in lines)
