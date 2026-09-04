"""Chunked-prefill invariance for the Nemotron-H Mamba-2 mixer.

A long prompt is prefilled in `--max-prefill-length` chunks: each chunk reads the
previous chunk's recurrent + conv state out of the ``LinearStatePool`` and writes its
own back. On top of that the forward freezes a donatable mid-chunk snapshot at the
deepest interior hybrid-radix track boundary (a multiple of the group's
``track_chunk_size`` -- for Mamba-2 that is the SSD chunk, 128, so the snapshot is a row
of the per-chunk state block the scan already produced).

Both handoffs must be invisible: the layer output and the carried state have to match a
single-chunk prefill of the same tokens. This pins that invariant against the pure-Torch
chunk scan and against the Triton SSD kernels that replace it, because a state-handoff
regression here shows up only as slowly degrading long-context quality (a wrong digit in
a needle answer), never as a crash.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.distributed import get_tp_info, set_tp_info
from freetoken.models.nemotron_h.config import parse_config
from freetoken.utils import torch_dtype

from .test_nemotron_h import _hf_config

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

_TOKENS = 1024
_UNQUANTIZED = {
    "quant_algo": "MIXED_PRECISION",
    "quant_method": "modelopt",
    "quantized_layers": {},
}


def _small_config():
    try:
        get_tp_info()
    except RuntimeError:
        set_tp_info(0, 1)
    hf = _hf_config(
        ["mamba"],
        hidden_size=256,
        mamba_num_heads=8,
        mamba_head_dim=32,
        ssm_state_size=32,
        n_groups=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        intermediate_size=128,
        moe_intermediate_size=128,
        moe_shared_expert_intermediate_size=128,
        n_routed_experts=8,
        num_experts_per_tok=2,
        vocab_size=512,
        quantization_config=_UNQUANTIZED,
    )
    return parse_config(hf)


def _prefill_req(extend_len: int, cached_len: int, slot: int, ping_pong, next_idx: int):
    """Only the fields the mixer's prefill path reads (Req itself needs a full scheduler)."""
    return SimpleNamespace(
        extend_len=extend_len,
        cached_len=cached_len,
        table_idx=0,
        linear_slot_idx=slot,
        mamba_last_track_seqlen=None,
        mamba_ping_pong=ping_pong,
        mamba_next_track_idx=next_idx,
    )


def _run_chunked(mixer, ctx, hidden, chunk_len, slot, ping_pong):
    """Push ``hidden`` through the mixer in consecutive prefill batches of ``chunk_len``."""
    from freetoken.core import Batch

    outputs, cached, next_idx = [], 0, 0
    total = hidden.shape[0]
    while cached < total:
        length = min(chunk_len, total - cached)
        req = _prefill_req(length, cached, slot, ping_pong, next_idx)
        batch = Batch(reqs=[req], phase="prefill")
        batch.padded_reqs = batch.reqs
        with ctx.forward_batch(batch), torch.inference_mode():
            outputs.append(mixer.forward(hidden[cached : cached + length]).float())
        next_idx = req.mamba_next_track_idx
        cached += length
    return torch.cat(outputs, dim=0)


def _relative_error(reference: torch.Tensor, actual: torch.Tensor) -> float:
    reference, actual = reference.float(), actual.float()
    denominator = float(torch.linalg.vector_norm(reference))
    return float(torch.linalg.vector_norm(reference - actual)) / max(denominator, 1e-12)


@cuda_only
@pytest.mark.parametrize("chunk_len", [512, 256, 128])
@pytest.mark.parametrize("track", [False, True])
def test_chunked_prefill_matches_single_pass(chunk_len, track):
    import freetoken.core as core
    from freetoken.core import Context, set_global_ctx
    from freetoken.kvcache.linear_state_pool import LinearStatePool
    from freetoken.models.nemotron_h.model import NemotronHMamba2Mixer

    device = torch.device("cuda")
    config = _small_config()
    pool = LinearStatePool(
        group=config.linear_attention_group(),
        num_slots=8,
        dtype=torch.bfloat16,
        device=device,
        tp_size=1,
    )
    core._GLOBAL_CTX = None
    ctx = Context(page_size=1, linear_state_pool=pool)
    set_global_ctx(ctx)

    torch.manual_seed(1234)
    with torch.device(device), torch_dtype(torch.bfloat16):
        mixer = NemotronHMamba2Mixer(config, 0)
    state = {}
    for name, buffer in mixer.state_dict().items():
        # Keep the scan's scalars in a sane range: A = -exp(A_log) and softplus(dt+dt_bias)
        # are what set the per-token decay, and a wild draw makes every chunking identical
        # by saturating the state to zero.
        scale = 0.5 if buffer.dtype.is_floating_point else 1.0
        state[name] = (torch.randn(buffer.shape, device=device) * scale).to(buffer.dtype)
    mixer.load_state_dict(state)

    hidden = (torch.randn(_TOKENS, config.hidden_size, device=device) * 0.5).to(torch.bfloat16)
    layer = pool.local_index(0)

    single = _run_chunked(mixer, ctx, hidden, _TOKENS, slot=1, ping_pong=None)
    single_recurrent = pool.recurrent_states[layer, 1].clone()
    single_conv = pool.conv_states[layer, 1].clone()

    chunked = _run_chunked(
        mixer, ctx, hidden, chunk_len, slot=2, ping_pong=(5, 6) if track else None
    )

    assert _relative_error(single, chunked) < 2e-3          # bf16 output rounding floor
    assert _relative_error(single_recurrent, pool.recurrent_states[layer, 2]) < 1e-5
    assert torch.equal(single_conv, pool.conv_states[layer, 2])


@cuda_only
def test_prefill_scan_splits_at_the_track_boundary():
    """The x128 track snapshot is exercised, and exact against a standalone prefill."""
    import freetoken.core as core
    from freetoken.core import Batch, Context, set_global_ctx
    from freetoken.kvcache.linear_state_pool import LinearStatePool
    from freetoken.models.nemotron_h.model import NemotronHMamba2Mixer

    device = torch.device("cuda")
    config = _small_config()
    pool = LinearStatePool(
        group=config.linear_attention_group(),
        num_slots=8,
        dtype=torch.bfloat16,
        device=device,
        tp_size=1,
    )
    core._GLOBAL_CTX = None
    ctx = Context(page_size=1, linear_state_pool=pool)
    set_global_ctx(ctx)
    torch.manual_seed(7)
    with torch.device(device), torch_dtype(torch.bfloat16):
        mixer = NemotronHMamba2Mixer(config, 0)
    mixer.load_state_dict(
        {
            name: (torch.randn(buf.shape, device=device) * 0.5).to(buf.dtype)
            for name, buf in mixer.state_dict().items()
        }
    )

    hidden = (torch.randn(320, config.hidden_size, device=device) * 0.5).to(torch.bfloat16)
    req = _prefill_req(320, 0, slot=1, ping_pong=(5, 6), next_idx=0)
    batch = Batch(reqs=[req], phase="prefill")
    batch.padded_reqs = batch.reqs
    with ctx.forward_batch(batch), torch.inference_mode():
        mixer.forward(hidden)
    # 320 tokens -> deepest strictly-interior x128 boundary is 256, and the frozen snapshot
    # lands in ping-pong slot 5 (mamba_next_track_idx flipped to 1 while building metadata).
    assert req.mamba_last_track_seqlen == 256
    assert req.mamba_next_track_idx == 1
    layer = pool.local_index(0)
    assert pool.recurrent_states[layer, 5].abs().sum() > 0

    # The frozen snapshot must equal a standalone prefill of the first 256 tokens.
    req2 = _prefill_req(256, 0, slot=2, ping_pong=None, next_idx=0)
    batch2 = Batch(reqs=[req2], phase="prefill")
    batch2.padded_reqs = batch2.reqs
    with ctx.forward_batch(batch2), torch.inference_mode():
        mixer.forward(hidden[:256])
    assert _relative_error(pool.recurrent_states[layer, 2], pool.recurrent_states[layer, 5]) < 1e-5
