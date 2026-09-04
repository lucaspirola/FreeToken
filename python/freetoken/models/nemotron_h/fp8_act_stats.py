"""Debug hook: how far the Mamba FP8 W8A8 activations run past their calibrated scale.

Enabled by pointing ``FREETOKEN_DEBUG_FP8_ACT_STATS`` at a JSON path. When unset the whole
module costs one ``is None`` test per Mamba mixer forward -- nothing is imported or launched.

Nemotron-3.5-Lightning ships ``backbone.layers.*.mixer.{in_proj,out_proj}`` as modelopt
FP8 W8A8 with a **static** per-tensor ``input_scale`` (one fp32 scalar calibrated offline).
``fp8_pertensor_linear`` therefore computes ``clamp(x / input_scale, -448, 448)`` on every
activation. If real long-context activations exceed ``448 * input_scale`` the clamp silently
saturates them, and nothing in the serving path reports it. This records, per module and
per phase (prefill / decode):

    amax          max |x| seen
    scale         the checkpoint's input_scale
    limit         448 * input_scale (the representable ceiling)
    clipped       number of elements with |x| > limit
    elements      elements seen
    calls         forward calls

Accumulation is done on the GPU with no host sync; the single sync is in ``flush()``.
"""

from __future__ import annotations

import json
import os

import torch

__all__ = ["ACT_STATS_PATH", "flush", "record"]

ACT_STATS_PATH = os.environ.get("FREETOKEN_DEBUG_FP8_ACT_STATS") or None

# name -> [amax, clipped, elements, calls] fp64 on the activation's device.
_ACC: dict[str, torch.Tensor] = {}
_SCALE: dict[str, float] = {}

_E4M3_MAX = 448.0


def record(name: str, x: torch.Tensor, input_scale: torch.Tensor | None) -> None:
    """Fold one activation tensor into the running per-module statistics."""
    if input_scale is None:
        return
    flat = x.detach().reshape(-1).float().abs()
    limit = input_scale.detach().float().reshape(()) * _E4M3_MAX
    row = _ACC.get(name)
    if row is None:
        row = torch.zeros(4, dtype=torch.float64, device=flat.device)
        _ACC[name] = row
        _SCALE[name] = float(input_scale.detach().float().reshape(()).item())
    row[0] = torch.maximum(row[0], flat.amax().double())
    row[1] += (flat > limit).sum().double()
    row[2] += float(flat.numel())
    row[3] += 1.0


def flush() -> None:
    """Serialize the accumulators to ``ACT_STATS_PATH`` (one host sync per module)."""
    if not _ACC:
        return
    out = {}
    for name, row in _ACC.items():
        amax, clipped, elements, calls = (float(v) for v in row.tolist())
        scale = _SCALE[name]
        limit = scale * _E4M3_MAX
        out[name] = {
            "amax": amax,
            "input_scale": scale,
            "limit": limit,
            "amax_over_limit": (amax / limit) if limit else float("inf"),
            "clipped": clipped,
            "elements": elements,
            "clipped_frac": (clipped / elements) if elements else 0.0,
            "calls": calls,
        }
    assert ACT_STATS_PATH is not None
    tmp = f"{ACT_STATS_PATH}.tmp"
    with open(tmp, "w") as handle:
        json.dump(out, handle, indent=1, sort_keys=True)
    os.replace(tmp, ACT_STATS_PATH)
