"""Mamba-2 gated group RMSNorm -- ``norm(x * silu(z))`` per group.

Task 2A3 of `tasks/nemotron35-plan.md`. A thin, name-stable wrapper over
`freetoken.kernel.fla.layernorm_gated.rms_norm_gated`, pinned to the Mamba-2
semantics so no call site has to remember the four flags that make it match
HF's ``MambaRMSNormGated``:

    norm_before_gate=False   ->  norm(x * silu(z)), not norm(x) * silu(z)
    activation="silu"
    is_rms_norm=True         ->  no mean subtraction
    group_size=D // n_groups

which is exactly

    y = (x * silu(z)) * rsqrt(mean((x * silu(z))^2, per group) + eps) * weight

with the reduction taken over each of the ``D // group_size`` contiguous groups
(Nemotron-3.5 Lightning: intermediate 4096, ``n_groups`` 8, group_size 512).
"""

from __future__ import annotations

import torch

from freetoken.kernel.fla.layernorm_gated import rms_norm_gated

__all__ = ["mamba2_gated_rmsnorm"]


def mamba2_gated_rmsnorm(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    group_size: int = 512,
) -> torch.Tensor:
    """Grouped RMSNorm of ``x * silu(gate)``.

    Args:
        x: ``[..., D]``.
        gate: ``[..., D]``, same shape as ``x`` (the ``z`` branch).
        weight: ``[D]`` per-channel scale.
        eps: variance epsilon.
        group_size: channels per normalisation group; ``D`` must be a multiple.

    Returns:
        ``[..., D]`` in ``x``'s dtype. The reduction runs in fp32 regardless of
        the input dtype.
    """
    assert gate.shape == x.shape, (
        f"gate is {tuple(gate.shape)}, expected {tuple(x.shape)}"
    )
    dim = x.shape[-1]
    assert weight.shape == (dim,), f"weight is {tuple(weight.shape)}, expected {(dim,)}"
    assert dim % group_size == 0, f"{dim} channels is not a multiple of {group_size}"
    return rms_norm_gated(
        x=x,
        weight=weight,
        bias=None,
        z=gate,
        eps=eps,
        group_size=group_size,
        norm_before_gate=False,
        is_rms_norm=True,
        activation="silu",
    )
