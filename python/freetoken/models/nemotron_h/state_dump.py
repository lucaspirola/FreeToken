"""Debug hook: dump a request's end-of-prefill Mamba-2 state (A/B only).

Enabled by pointing ``FREETOKEN_MAMBA2_STATE_DUMP`` at a directory. When unset the
whole module costs one module-level ``if`` per forward in
``NemotronHForCausalLM.forward`` -- nothing is imported, allocated or synchronized.

It exists to answer one question the serving path cannot answer from the outside:
does the *same* prompt leave the *same* recurrent + conv state behind when the engine
splits it into a different number of prefill chunks (``--max-prefill-length``)?  The
kernels are bit-exact chunk-invariant in isolation, so any divergence here is in the
integration (metadata, track snapshot, conv carry), not in the scan.

Written on the last prefill forward of a request (the one that samples -- a
``ChunkedReq`` continuation reports ``can_decode`` False and is skipped), one
``torch.save`` per request:

    recurrent  [L, H, P, N] fp32   the live slot's SSM state, all Mamba-2 layers
    conv       [L, conv_dim, K-1] fp32
    logits     [vocab] fp32        the sampled position's logits
    layer_ids  the model layer id of each row of the two stacks
"""

from __future__ import annotations

import os

import torch

__all__ = ["STATE_DUMP_DIR", "dump_prefill_state"]

STATE_DUMP_DIR = os.environ.get("FREETOKEN_MAMBA2_STATE_DUMP") or None


def dump_prefill_state(logits: torch.Tensor) -> None:
    """Save the end-of-prefill recurrent/conv state + logits of every sampling request."""
    from freetoken.core import get_global_ctx

    ctx = get_global_ctx()
    batch = ctx.batch
    pool = ctx.linear_state_pool
    if pool is None or not batch.is_prefill:
        return
    os.makedirs(STATE_DUMP_DIR, exist_ok=True)
    for index, req in enumerate(batch.reqs):
        if req.uid < 0:
            continue  # engine warmup / profiling dummy batch
        if not req.can_decode:
            continue  # chunked continuation: this is not the end of the prefill
        slot = req.linear_slot_idx if req.linear_slot_idx is not None else req.table_idx
        record = {
            "uid": req.uid,
            "slot": int(slot),
            "cached_len": int(req.cached_len),
            "device_len": int(req.device_len),
            "extend_len": int(req.extend_len),
            "layer_ids": list(pool.group.layer_ids),
            "recurrent": pool.recurrent_states[:, slot].float().cpu(),
            "conv": pool.conv_states[:, slot].float().cpu(),
            "logits": logits[index].float().cpu(),
        }
        path = os.path.join(STATE_DUMP_DIR, f"prefill_uid{req.uid}_len{req.device_len}.pt")
        torch.save(record, path)
