"""Task 2A4: the Nemotron-H mixer on the Triton SSD kernels vs the pure-Torch reference.

``FREETOKEN_MAMBA2_REF=1`` swaps the whole scan (prefill, decode, gated norm) for
``models/nemotron_h/mamba2_reference.py``, the Phase 1 path. These tests drive the SAME
mixer object both ways and require the two to agree, which is what makes the env var a
usable A/B switch on a real server: if it ever silently diverges, the reference stops
being evidence about the kernels.

They also pin the state-layout contract itself -- the pool slot is the SSD-native
``[H, P, N]`` block, so no scan input or output is transposed anywhere -- and the
prefill/decode handoff (``prefill(T) + decode(1)`` must equal ``prefill(T+1)``).
"""

from __future__ import annotations

import pytest
import torch

from freetoken.utils import torch_dtype

from .test_nemotron_h_chunked_prefill import (
    _prefill_req,
    _relative_error,
    _small_config,
    cuda_only,
)


def _mixer_and_pool(device, num_slots=8, seed=11):
    import freetoken.core as core
    from freetoken.core import Context, set_global_ctx
    from freetoken.kvcache.linear_state_pool import LinearStatePool
    from freetoken.models.nemotron_h.model import NemotronHMamba2Mixer

    config = _small_config()
    pool = LinearStatePool(
        group=config.linear_attention_group(),
        num_slots=num_slots,
        dtype=torch.bfloat16,
        device=device,
        tp_size=1,
    )
    core._GLOBAL_CTX = None
    ctx = Context(page_size=1, linear_state_pool=pool)
    set_global_ctx(ctx)
    torch.manual_seed(seed)
    with torch.device(device), torch_dtype(torch.bfloat16):
        mixer = NemotronHMamba2Mixer(config, 0)
    mixer.load_state_dict(
        {
            name: (torch.randn(buf.shape, device=device) * 0.5).to(buf.dtype)
            for name, buf in mixer.state_dict().items()
        }
    )
    return config, ctx, pool, mixer


def _prefill(mixer, ctx, hidden, *, slot, cached_len=0, ping_pong=None, next_idx=0):
    from freetoken.core import Batch

    req = _prefill_req(hidden.shape[0], cached_len, slot, ping_pong, next_idx)
    batch = Batch(reqs=[req], phase="prefill")
    batch.padded_reqs = batch.reqs
    with ctx.forward_batch(batch), torch.inference_mode():
        out = mixer.forward(hidden)
    return out, req


def _decode(mixer, ctx, hidden, slots):
    from freetoken.core import Batch

    reqs = [_prefill_req(1, 0, slot, None, 0) for slot in slots]
    batch = Batch(reqs=reqs, phase="decode")
    batch.padded_reqs = batch.reqs
    batch.linear_table_idx = torch.tensor(
        slots, dtype=torch.int32, device=hidden.device
    )
    with ctx.forward_batch(batch), torch.inference_mode():
        return mixer.forward(hidden)


# ------------------------------------------------------------------ layout contract
def test_mamba2_state_layout_is_native_hpn():
    """[H, P, N] pool + a 6144-wide conv stream for the real Lightning geometry."""
    from freetoken.kvcache.linear_state_pool import _linear_local_dims

    config = _small_config()
    group = config.linear_attention_group()
    assert group.state_layout == "mamba2"
    assert group.track_chunk_size == 128            # the SSD chunk, not the FLA 64
    # small fixture: H=8, P=32, N=32, G=2  ->  conv = H*P + 2*G*N = 256 + 128
    assert _linear_local_dims(group, 1) == (1, 384, 8)
    assert (group.num_value_heads, group.key_head_dim, group.value_head_dim) == (8, 32, 32)

    from freetoken.models.config import LinearGatedDeltaGroupConfig

    lightning = LinearGatedDeltaGroupConfig(
        name="mamba", layer_ids=tuple(range(23)),
        num_key_heads=8, num_value_heads=64, key_head_dim=64, value_head_dim=128,
        conv_kernel_dim=4, output_gate=True, state_layout="mamba2", track_chunk_size=128,
    )
    assert _linear_local_dims(lightning, 1) == (23, 6144, 64)


def test_gdn_layout_is_unchanged():
    """Qwen3.5 / Ornith GDN keeps the "kv" layout and the 64-token track chunk."""
    from freetoken.kvcache.linear_state_pool import _linear_local_dims
    from freetoken.models.config import LinearGatedDeltaGroupConfig

    gdn = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0, 1), num_key_heads=16, num_value_heads=32,
        key_head_dim=128, value_head_dim=128, conv_kernel_dim=4, output_gate=True,
    )
    assert (gdn.state_layout, gdn.track_chunk_size) == ("kv", 64)
    assert _linear_local_dims(gdn, 1) == (2, 2 * 16 * 128 + 32 * 128, 32)


def test_track_chunk_size_must_be_a_power_of_two():
    from freetoken.models.config import LinearGatedDeltaGroupConfig

    with pytest.raises(ValueError, match="power of two"):
        LinearGatedDeltaGroupConfig(
            name="m", layer_ids=(0,), num_key_heads=1, num_value_heads=1,
            key_head_dim=1, value_head_dim=1, conv_kernel_dim=4, output_gate=True,
            track_chunk_size=96,
        )


# ------------------------------------------------------------------ kernel vs reference
@cuda_only
@pytest.mark.parametrize("tokens", [1, 127, 300, 1024])
def test_prefill_matches_the_pure_torch_reference(monkeypatch, tokens):
    device = torch.device("cuda")
    config, ctx, pool, mixer = _mixer_and_pool(device)
    layer = pool.local_index(0)
    hidden = (torch.randn(tokens, config.hidden_size, device=device) * 0.5).to(
        torch.bfloat16
    )

    monkeypatch.delenv("FREETOKEN_MAMBA2_REF", raising=False)
    kernel_out, _ = _prefill(mixer, ctx, hidden, slot=1)
    kernel_state = pool.recurrent_states[layer, 1].clone()

    monkeypatch.setenv("FREETOKEN_MAMBA2_REF", "1")
    ref_out, _ = _prefill(mixer, ctx, hidden, slot=2)
    ref_state = pool.recurrent_states[layer, 2].clone()

    # bf16 IO floor. The SSD kernels round B*decay and CB to the input dtype before
    # each tl.dot, which tests/kernels/test_mamba2_ssd.py measures at 1.7e-3 state /
    # 1.8e-3 output against an fp64 gold; the reference carries its own error, and the
    # kernel additionally writes the scan output in bf16 before the norm + out_proj.
    # Measured here: 2.0-3.5e-3 state, 4.7e-3 output, flat in T.
    assert _relative_error(ref_out.float(), kernel_out.float()) < 1e-2
    assert _relative_error(ref_state, kernel_state) < 6e-3


@cuda_only
@pytest.mark.parametrize("bs", [1, 3])
def test_decode_matches_the_pure_torch_reference(monkeypatch, bs):
    device = torch.device("cuda")
    config, ctx, pool, mixer = _mixer_and_pool(device, num_slots=16)
    layer = pool.local_index(0)

    # Give every slot a non-trivial carried state, then run one decode step twice from
    # the same starting point (slots 1..bs for the kernel, 1+bs.. for the reference).
    prompt = (torch.randn(200, config.hidden_size, device=device) * 0.5).to(torch.bfloat16)
    monkeypatch.delenv("FREETOKEN_MAMBA2_REF", raising=False)
    for slot in range(1, 2 * bs + 1):
        _prefill(mixer, ctx, prompt, slot=slot)

    step = (torch.randn(bs, config.hidden_size, device=device) * 0.5).to(torch.bfloat16)
    kernel_slots = list(range(1, bs + 1))
    ref_slots = list(range(bs + 1, 2 * bs + 1))
    conv_before = pool.conv_states[layer].clone()

    kernel_out = _decode(mixer, ctx, step, kernel_slots).float().clone()
    kernel_state = pool.recurrent_states[layer, kernel_slots].clone()

    pool.conv_states[layer].copy_(conv_before)   # the kernel step advanced the conv window
    monkeypatch.setenv("FREETOKEN_MAMBA2_REF", "1")
    ref_out = _decode(mixer, ctx, step, ref_slots).float().clone()
    ref_state = pool.recurrent_states[layer, ref_slots].clone()

    assert _relative_error(ref_out, kernel_out) < 5e-3
    assert _relative_error(ref_state, kernel_state) < 5e-3


@cuda_only
def test_prefill_then_decode_equals_prefill_of_one_more_token():
    """The prefill and decode kernels must agree on the recurrence at the seam."""
    device = torch.device("cuda")
    config, ctx, pool, mixer = _mixer_and_pool(device, seed=3)
    layer = pool.local_index(0)
    hidden = (torch.randn(301, config.hidden_size, device=device) * 0.5).to(torch.bfloat16)

    _prefill(mixer, ctx, hidden, slot=1)                       # all 301 tokens at once
    full_state = pool.recurrent_states[layer, 1].clone()

    _prefill(mixer, ctx, hidden[:300], slot=2)                 # 300 + one decode step
    _decode(mixer, ctx, hidden[300:301], [2])

    assert _relative_error(full_state, pool.recurrent_states[layer, 2]) < 5e-3


@cuda_only
def test_decode_leaves_untouched_slots_alone():
    device = torch.device("cuda")
    config, ctx, pool, mixer = _mixer_and_pool(device, num_slots=16, seed=5)
    layer = pool.local_index(0)
    prompt = (torch.randn(200, config.hidden_size, device=device) * 0.5).to(torch.bfloat16)
    for slot in (1, 2, 7):
        _prefill(mixer, ctx, prompt, slot=slot)
    untouched = pool.recurrent_states[layer, 7].clone()

    step = (torch.randn(2, config.hidden_size, device=device) * 0.5).to(torch.bfloat16)
    _decode(mixer, ctx, step, [1, 2])

    assert torch.equal(untouched, pool.recurrent_states[layer, 7])


@cuda_only
def test_varlen_batch_matches_per_sequence_prefill():
    """One SSD launch over a mixed-length batch == the same sequences run alone."""
    from freetoken.core import Batch

    device = torch.device("cuda")
    config, ctx, pool, mixer = _mixer_and_pool(device, num_slots=16, seed=9)
    layer = pool.local_index(0)
    lengths = [1, 129, 256, 300]
    hidden = [
        (torch.randn(n, config.hidden_size, device=device) * 0.5).to(torch.bfloat16)
        for n in lengths
    ]

    alone = [_prefill(mixer, ctx, h, slot=8 + i)[0].float() for i, h in enumerate(hidden)]
    alone_states = [pool.recurrent_states[layer, 8 + i].clone() for i in range(len(lengths))]

    reqs = [_prefill_req(n, 0, 1 + i, None, 0) for i, n in enumerate(lengths)]
    batch = Batch(reqs=reqs, phase="prefill")
    batch.padded_reqs = batch.reqs
    with ctx.forward_batch(batch), torch.inference_mode():
        out = mixer.forward(torch.cat(hidden, dim=0)).float()

    offset = 0
    for i, n in enumerate(lengths):
        assert _relative_error(alone[i], out[offset : offset + n]) < 1e-5
        assert _relative_error(alone_states[i], pool.recurrent_states[layer, 1 + i]) < 1e-5
        offset += n


@cuda_only
def test_track_snapshot_row_is_the_state_after_the_boundary(monkeypatch):
    """The kernel path's track_h_row must select the SAME state the reference's second
    scan produces -- an off-by-one chunk here silently corrupts every prefix-cache hit."""
    device = torch.device("cuda")
    config, ctx, pool, mixer = _mixer_and_pool(device, num_slots=16, seed=13)
    layer = pool.local_index(0)
    hidden = (torch.randn(400, config.hidden_size, device=device) * 0.5).to(torch.bfloat16)

    monkeypatch.delenv("FREETOKEN_MAMBA2_REF", raising=False)
    _, req = _prefill(mixer, ctx, hidden, slot=1, ping_pong=(5, 6))
    assert req.mamba_last_track_seqlen == 384       # deepest interior x128 boundary
    kernel_snapshot = pool.recurrent_states[layer, 5].clone()

    monkeypatch.setenv("FREETOKEN_MAMBA2_REF", "1")
    _, ref_req = _prefill(mixer, ctx, hidden, slot=2, ping_pong=(11, 12))
    assert ref_req.mamba_last_track_seqlen == 384
    assert _relative_error(pool.recurrent_states[layer, 11], kernel_snapshot) < 2e-3

    # ... and both equal a standalone prefill of exactly the first 384 tokens.
    monkeypatch.delenv("FREETOKEN_MAMBA2_REF", raising=False)
    _prefill(mixer, ctx, hidden[:384], slot=3)
    assert _relative_error(pool.recurrent_states[layer, 3], kernel_snapshot) < 1e-5


@cuda_only
def test_decode_out_buffer_is_stable_per_batch_size():
    """A captured decode graph bakes in the `out` address it saw, so the buffer for a
    batch size must never be replaced -- not even by a later, WIDER eager decode (which
    is exactly what an elastic capacity raise produces: batches above the largest
    captured size run eagerly). A grow-only buffer would free the block those graphs
    still write to on every replay, silently corrupting whatever the allocator hands out
    next."""
    device = torch.device("cuda")
    _, _, _, mixer = _mixer_and_pool(device)
    shape = (mixer.num_heads, mixer.head_dim)

    def probe(bs):
        return mixer._decode_out(
            torch.empty(bs, *shape, dtype=torch.bfloat16, device=device)
        )

    small = probe(4)
    assert small.shape == (4, *shape) and small.is_contiguous()
    address = small.data_ptr()
    probe(15)                                  # wider eager decode
    probe(1)
    assert probe(4).data_ptr() == address      # bs=4 buffer must not have moved
    assert probe(15).shape == (15, *shape)
