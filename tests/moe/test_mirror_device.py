"""Device-level mirror tests: the byte-exact swap lifecycle, run for real.

These wrap the two scenario scripts that found every serious bug in this
design (ownership-map divergence, arena-shrink refill, the graphs-mode
boundary) as permanent regressions. They are marked CUDA-required: the value
is exercising the real triton kernels and copy hardware, notAnimating not the
host model.
"""
import os

import subprocess
import sys

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="device tests need CUDA"
)

_TASKS = os.path.join(os.path.dirname(__file__), "..", "..",
                      "tasks", "exclusive-expert-ram")

def _run_script(name: str) -> None:
    path = os.path.join(_TASKS, name)
    if not os.path.exists(path):
        pytest.skip(f"{name} not present (tasks/ assets are dev-only)")
    env = dict(os.environ, FREETOKEN_EXPERT_ARENA="1")
    # The scenarios pin their own sys.path from __file__, so cwd is free.
    result = subprocess.run(
        [sys.executable, path],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"{name} failed\n--- stdout tail ---\n{result.stdout[-2000:]}"
        f"\n--- stderr tail ---\n{result.stderr[-2000:]}"
    )
    assert "PASS" in result.stdout, result.stdout[-2000:]

def test_swap_lifecycle_byte_exact():
    """Prefill sweeps, routed decode and arena shrinks keep every served row
    byte-identical to the checkpoint (the corruption regressions)."""
    _run_script("swap_smoke.py")

def test_graph_replay_boundary():
    """Replayed decode after an eager prefill sweep is byte-correct: the
    scheduler boundary feeds the replay a full mirror (the graphs-mode
    regression)."""
    _run_script("graph_race_repro.py")
