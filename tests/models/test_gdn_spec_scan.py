"""GDN speculative-verify capture/commit against a synthetic pool.

The live gate (tests/e2e/test_spec_ngram_equivalence.py) proves the fused commit
bit-exact on Ornith, but it needs a checkpoint. This test drives
``GdnSpecScanCapture`` against a two-layer synthetic ``LinearStatePool`` so the
three commit invariants are covered without weights:

* a full acceptance copies the scratch slot back wholesale (no replay);
* a partial acceptance replays exactly ``n`` tokens into the live slot and matches
  an independent per-layer reference commit;
* the fused (layer-folded) and per-layer paths agree, and the conv window slides.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

# Small GDN geometry (head_k_dim == head_v_dim is required by the pool + kernel).
HK, HV, DK, DV = 4, 8, 32, 32
CONV_KERNEL = 4
LAYER_IDS = (0, 1)
NUM_SLOTS = 4


class _FakeGdn:
    def __init__(self, layer_id: int, device, seed: int):
        g = torch.Generator(device=device).manual_seed(seed)
        self.layer_id = layer_id
        self.num_k_heads = HK
        self.num_v_heads = HV
        self.head_k_dim = DK
        self.head_v_dim = DV
        self.conv_dim = 2 * HK * DK + HV * DV
        # Negative-ish A_log (log-decay base) and a small dt_bias, fp32 like production.
        self.A_log = torch.randn(HV, device=device, generator=g, dtype=torch.float32).abs() * -0.5
        self.dt_bias = torch.randn(HV, device=device, generator=g, dtype=torch.float32) * 0.1


def _pool(device) -> "LinearStatePool":
    from freetoken.kvcache.linear_state_pool import LinearStatePool
    from freetoken.models.config import LinearGatedDeltaGroupConfig

    group = LinearGatedDeltaGroupConfig(
        "gdn",
        layer_ids=LAYER_IDS,
        num_key_heads=HK,
        num_value_heads=HV,
        key_head_dim=DK,
        value_head_dim=DV,
        conv_kernel_dim=CONV_KERNEL,
        output_gate=True,
        state_layout="kv",
        track_chunk_size=64,
    )
    return LinearStatePool(group, num_slots=NUM_SLOTS, dtype=torch.bfloat16, device=device, tp_size=1)


def _capture(gdns, m: int, device, seed: int):
    from freetoken.models.qwen3_5_moe.spec_scan import GdnSpecScanCapture

    cap = GdnSpecScanCapture(m, fused=True)
    g = torch.Generator(device=device).manual_seed(seed)
    conv_dim = 2 * HK * DK + HV * DV
    for gdn in gdns:
        q = torch.randn(1, m, HK, DK, device=device, generator=g, dtype=torch.bfloat16)
        k = torch.randn(1, m, HK, DK, device=device, generator=g, dtype=torch.bfloat16)
        v = torch.randn(1, m, HV, DV, device=device, generator=g, dtype=torch.bfloat16)
        gg = torch.rand(1, m, HV, device=device, generator=g, dtype=torch.float32) * -0.2
        beta = torch.rand(1, m, HV, device=device, generator=g, dtype=torch.float32)
        conv_in = torch.randn(m, conv_dim, device=device, generator=g, dtype=torch.bfloat16)
        cap.record(gdn, q, k, v, gg, beta, conv_in)
    return cap


def _freshen(pool, device, seed: int) -> None:
    """Give the pool non-trivial live + scratch state so copies/slides are observable."""
    g = torch.Generator(device=device).manual_seed(seed)
    pool.recurrent_states.normal_(generator=g)
    pool.conv_states.normal_(generator=g)


def test_full_acceptance_copies_scratch_back():
    device = torch.device("cuda")
    pool = _pool(device)
    _freshen(pool, device, 0)
    live, scratch = 1, 2
    gdns = [_FakeGdn(i, device, 10 + i) for i in LAYER_IDS]
    cap = _capture(gdns, 4, device, 100)
    # Distinct scratch content; commit must clone it onto live without a replay.
    pool.conv_states[:, scratch] += 1.0
    pool.recurrent_states[:, scratch] += 2.0
    before_live = pool.recurrent_states[:, live].clone()
    cap.commit(pool, live, scratch, n=4)
    assert torch.equal(pool.recurrent_states[:, scratch], pool.recurrent_states[:, live])
    assert not torch.equal(pool.recurrent_states[:, live], before_live)


def test_partial_acceptance_matches_per_layer_reference():
    device = torch.device("cuda")
    n = 3
    # Two identical pools: fused commit on one, per-layer reference on the other.
    fused = _pool(device)
    ref = _pool(device)
    _freshen(fused, device, 1)
    _freshen(ref, device, 1)
    live = 1
    gdns_f = [_FakeGdn(i, device, 20 + i) for i in LAYER_IDS]
    gdns_r = [_FakeGdn(i, device, 20 + i) for i in LAYER_IDS]
    # Same recorded inputs on both captures.
    cap_f = _capture(gdns_f, 5, device, 200)
    cap_r = _capture(gdns_r, 5, device, 200)
    cap_f.commit(fused, live, scratch_slot=2, n=n)           # fused path
    cap_r._commit_per_layer(ref, live, n)                    # reference path
    torch.testing.assert_close(
        fused.recurrent_states[:, live], ref.recurrent_states[:, live],
        rtol=2e-2, atol=2e-2,
    )
    torch.testing.assert_close(
        fused.conv_states[:, live].float(), ref.conv_states[:, live].float(),
        rtol=0, atol=0,
    )


def test_partial_commit_differs_from_full_replay():
    """Replaying n < m tokens must NOT equal replaying all m (guards a dropped prefix)."""
    device = torch.device("cuda")
    pool = _pool(device)
    _freshen(pool, device, 2)
    live = 1
    gdns = [_FakeGdn(i, device, 30 + i) for i in LAYER_IDS]
    cap = _capture(gdns, 5, device, 300)
    state_full = None
    # Replay 5 tokens into a copy of live, then 3 tokens from the same start.
    pool.copy_from(live, 3)
    cap.commit(pool, 3, scratch_slot=2, n=5)
    state_full = pool.recurrent_states[:, 3].clone()
    cap.commit(pool, live, scratch_slot=2, n=3)
    assert not torch.allclose(
        pool.recurrent_states[:, live], state_full, rtol=1e-3, atol=1e-3
    )
