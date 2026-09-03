"""fp64 sequential Mamba-2 SSD recurrence -- the numerical gold for the kernel tests.

The chunked HF reference (`transformers...nemotron_h.mamba2_chunk_scan`) is *not*
a gold: it is a different, also-approximate factorisation of the same recurrence
(segment sums of `dt*A`, an exp of a difference of cumsums, a `[chunk, chunk]`
attention-like matrix), and in fp32 it drifts from the exact answer by more than
the Triton kernels do. This module evaluates the definition directly, one token
at a time, in float64:

    dt_t   = clamp(softplus(dt_raw_t + dt_bias), *dt_limit)
    h_t    = exp(dt_t * A) * h_{t-1} + dt_t * x_t (x) B_t
    y_t    = C_t . h_t + D * x_t

with grouped B/C: head ``h`` reads group ``h // (H // G)``. State layout is
``[H, P, N]`` (the pool layout), matching the kernels.
"""

from __future__ import annotations

import torch

__all__ = ["gold_ssd"]


def _apply_dt(dt_raw, dt_bias, dt_softplus, dt_limit):
    dt = dt_raw.to(torch.float64)
    if dt_bias is not None:
        dt = dt + dt_bias.to(torch.float64)
    if dt_softplus:
        dt = torch.where(dt <= 20.0, torch.log1p(torch.exp(dt)), dt)
    lo, hi = dt_limit
    if lo != 0.0 or hi != float("inf"):
        dt = dt.clamp(lo, hi)
    else:
        dt = dt.clamp_min(0.0)
    return dt


def gold_ssd(
    x,
    dt_raw,
    B,
    C,
    A,
    D=None,
    dt_bias=None,
    initial=None,
    *,
    dt_softplus: bool = True,
    dt_limit: tuple[float, float] = (0.0, float("inf")),
    chunk_size: int | None = None,
):
    """Exact fp64 scan over one sequence.

    Args:
        x: ``[T, H, P]``; dt_raw: ``[T, H]``; B/C: ``[T, G, N]``;
        A/D/dt_bias: ``[H]``; initial: ``[H, P, N]`` or None.
        chunk_size: when given, also return the state after every ``chunk_size``
            tokens (last chunk included, even when short) as ``[nchunks, H, P, N]``.

    Returns:
        ``(out, final_state)`` in float64, or ``(out, final_state, chunk_states)``
        when ``chunk_size`` is given.
    """
    T, H, P = x.shape
    _, G, N = B.shape
    rep = H // G
    dev = x.device

    xf = x.to(torch.float64)
    Bf = B.to(torch.float64)
    Cf = C.to(torch.float64)
    Af = A.to(torch.float64)
    dt = _apply_dt(dt_raw, dt_bias, dt_softplus, dt_limit)

    h = (
        torch.zeros(H, P, N, device=dev, dtype=torch.float64)
        if initial is None
        else initial.to(torch.float64).clone()
    )
    out = torch.empty(T, H, P, device=dev, dtype=torch.float64)
    chunks = []
    for t in range(T):
        dt_t = dt[t]  # [H]
        decay = torch.exp(dt_t * Af)  # [H]
        b_t = Bf[t].repeat_interleave(rep, dim=0)  # [H, N]
        c_t = Cf[t].repeat_interleave(rep, dim=0)  # [H, N]
        h = h * decay[:, None, None] + (dt_t[:, None] * xf[t])[:, :, None] * b_t[:, None, :]
        y = (h * c_t[:, None, :]).sum(-1)  # [H, P]
        if D is not None:
            y = y + D.to(torch.float64)[:, None] * xf[t]
        out[t] = y
        if chunk_size is not None and ((t + 1) % chunk_size == 0 or t == T - 1):
            chunks.append(h.clone())

    if chunk_size is not None:
        return out, h, torch.stack(chunks) if chunks else torch.zeros(
            0, H, P, N, device=dev, dtype=torch.float64
        )
    return out, h
