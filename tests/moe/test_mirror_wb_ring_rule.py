"""The DMA-writeback staging ring's size rule (residency.wb_stage_rows), CPU only."""
from __future__ import annotations

import pytest

from freetoken.moe.residency import wb_stage_rows


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("FREETOKEN_MIRROR_WB_STAGE_MB", raising=False)
    monkeypatch.delenv("FREETOKEN_MIRROR_WB_STAGE_ROWS", raising=False)


# The two production geometries at their box warm-start arenas (2026-09-24 logs):
# per-step writebacks measured at the 99th percentile were 75 and 20.
EXL3 = dict(top_k=8, moe_layers=40, experts=256, gpu_slots=6076, reserve_rows=768)
NEMOTRON = dict(top_k=6, moe_layers=23, experts=128, gpu_slots=2296, reserve_rows=256)


def test_rule_on_the_measured_geometries():
    assert wb_stage_rows(**EXL3) == 66        # ceil(320 * 0.4066 / 2)
    assert wb_stage_rows(**NEMOTRON) == 16    # ceil(138 * 0.2201 / 2)


def test_more_of_the_model_off_the_gpu_needs_a_bigger_ring():
    small_arena = {**NEMOTRON, "gpu_slots": 1480}
    assert wb_stage_rows(**small_arena) > wb_stage_rows(**NEMOTRON)


def test_whole_model_on_the_gpu_gets_the_floor():
    assert wb_stage_rows(**{**EXL3, "gpu_slots": 40 * 256}) == 4


def test_batch_scales_it_and_the_reserve_caps_it():
    assert wb_stage_rows(**NEMOTRON, batch=2) == 31
    assert wb_stage_rows(**NEMOTRON, batch=64) == 256 // 3


def test_overrides(monkeypatch):
    monkeypatch.setenv("FREETOKEN_MIRROR_WB_STAGE_ROWS", "32")
    assert wb_stage_rows(**EXL3) == 32
    monkeypatch.setenv("FREETOKEN_MIRROR_WB_STAGE_ROWS", "1000")
    assert wb_stage_rows(**EXL3) == 768 // 3      # still capped
    monkeypatch.setenv("FREETOKEN_MIRROR_WB_STAGE_MB", "0")
    assert wb_stage_rows(**EXL3) == 0             # off wins
