"""Write-back DMA gated on decode attention (MirrorResidency.attention_gate_hook).

The ring -> pool DMA is split into one chunk per decode attention layer and
each chunk waits for the event recorded in front of that layer's attention.
Three things must hold: the proxy gates decode calls only and is otherwise the
inner backend; a chunk really waits for its gate, also when the gate is an
external-event node inside a CUDA graph (one capture, replayed); and the pool
stays byte-exact with the gates in the step (the DMA only moves in time).
"""
from __future__ import annotations

import tempfile
import types

import pytest
import torch

from freetoken.attention.base import DecodeGatedBackend


class _Inner:
    def __init__(self):
        self.calls, self.flag = [], 1

    def forward(self, q, k, v, layer_id, batch, attn_spec=None):
        self.calls.append(("forward", layer_id))
        return q

    def mla_forward(self, q_nope, q_pe, c_kv, k_rope, layer_id, batch, indexer_inputs=None):
        self.calls.append(("mla", layer_id))
        return q_nope

    def prepare_metadata(self, batch):
        self.calls.append(("meta", None))


def test_proxy_gates_decode_only_and_passes_everything_through():
    inner, gated = _Inner(), []
    b = DecodeGatedBackend(inner, gated.append)
    dec, pre = types.SimpleNamespace(is_prefill=False), types.SimpleNamespace(is_prefill=True)
    assert b.forward(1, 2, 3, 5, dec) == 1
    assert b.forward(1, 2, 3, 7, pre) == 1
    assert b.forward(1, 2, 3, layer_id=9, batch=dec) == 1
    assert b.mla_forward(4, 0, 0, 0, 11, dec) == 4
    b.prepare_metadata(dec)
    assert gated == [5, 9, 11]
    assert inner.calls == [("forward", 5), ("forward", 7), ("forward", 9), ("mla", 11),
                           ("meta", None)]
    assert b.flag == 1
    b.flag = 2                                   # writes go to the inner backend
    assert inner.flag == 2 and not hasattr(b, "bsa_forward")


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@cuda
def test_side_stream_waits_for_a_gate_recorded_inside_a_graph():
    """External-event record node in a captured graph: a copy issued AFTER the
    launch on another stream, behind a wait on that event, starts only once the
    graph reaches the node (after the long kernel in front of it)."""
    main, side = torch.cuda.Stream(), torch.cuda.Stream()
    gate = torch.cuda.Event(external=True)
    x = torch.zeros(1, device="cuda")
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(main):
        torch.cuda._sleep(1000)                  # warm the sleep kernel
        main.synchronize()
        with torch.cuda.graph(g, stream=main):
            torch.cuda._sleep(50_000_000)        # tens of ms on any GPU
            gate.record()
            x.add_(1)
    torch.cuda.synchronize()
    host = torch.zeros(1, pin_memory=True)
    for _ in range(2):                           # one capture, two replays
        t0, t1, t_gate = (torch.cuda.Event(enable_timing=True) for _ in range(3))
        with torch.cuda.stream(main):
            t0.record()
            g.replay()
            t1.record()
        side.wait_event(gate)
        with torch.cuda.stream(side):
            host.copy_(x, non_blocking=True)
            t_gate.record()
        torch.cuda.synchronize()
        graph_ms = t0.elapsed_time(t1)
        assert graph_ms > 5.0
        assert t0.elapsed_time(t_gate) > 0.9 * graph_ms, (graph_ms, t0.elapsed_time(t_gate))


@cuda
def test_gated_writebacks_stay_byte_exact(monkeypatch):
    from tests.moe import test_mirror_dma_writeback as T

    monkeypatch.setattr(T, "_decode", _gated_decode(T))
    with tempfile.TemporaryDirectory() as root:
        T.write_nvfp4_checkpoint(root, T.LAYERS, T.EXPERTS, T.H, T.ISZ)
        golden = T._golden(root)
        from freetoken.moe import offload_cache as oc
        from freetoken.moe import offload_kernels as ok
        prev = (oc.FREETOKEN_EXPERT_ARENA, ok.FREETOKEN_EXPERT_ARENA)
        oc.FREETOKEN_EXPERT_ARENA = ok.FREETOKEN_EXPERT_ARENA = True
        cache, pool = T._cache(root, monkeypatch, T._DEFAULT)
        try:
            hook = cache.residency.attention_gate_hook()
            assert hook is not None
            bad = T._decode(cache, golden, steps=240, service_every=1, gate=hook)
            assert bad == []
            assert len(cache.residency._attn_gate_list) == 3
            st = cache.mirror_stats()
            assert st["writebacks"] > 20 and st["staged_writebacks"] > 0
            assert st["coverage_faults"] == 0 and st["starved_writebacks"] == 0
            cache.residency.drain_writebacks()
            assert T._holes(cache) == [] and T._pool_bad(cache, pool, golden) == []
            st = cache.mirror_stats()
            assert st["dma_writebacks"] == st["staged_writebacks"]
        finally:
            pool.close()
            oc.FREETOKEN_EXPERT_ARENA, ok.FREETOKEN_EXPERT_ARENA = prev


@cuda
def test_gate_switch_off(monkeypatch):
    monkeypatch.setenv("FREETOKEN_MIRROR_WB_AT_ATTENTION", "0")
    from freetoken.moe.residency import MirrorResidency

    r = MirrorResidency.__new__(MirrorResidency)
    r._wb = {"rows": 1}
    assert r.attention_gate_hook() is None
    monkeypatch.setenv("FREETOKEN_MIRROR_WB_AT_ATTENTION", "1")
    assert r.attention_gate_hook() is not None
    r._wb = None
    assert r.attention_gate_hook() is None


def _gated_decode(T):
    def run(cache, golden, *, steps, service_every, gate, seed=0):
        res = cache.residency
        torch.manual_seed(seed)
        bad = []
        for step in range(steps):
            layer = step % T.LAYERS
            if layer % 2 == 0:                   # an "attention layer" before 0, 2, 4
                gate(layer)
                torch.cuda._sleep(20_000)
            perm = torch.randperm(T.EXPERTS, device="cuda")[:4].to(torch.int32)
            want = perm.tolist()
            ids = perm.clone().reshape(1, 4)
            cache.ensure_experts(layer, ids)
            cache.copy_missing()
            if step % service_every == 0:
                res.service_writebacks()
            torch.cuda.synchronize()
            for e, slot in zip(want, ids.reshape(-1).tolist()):
                if not T._slot_ok(cache, golden, layer * T.EXPERTS + e, slot):
                    bad.append((step, layer * T.EXPERTS + e, slot))
            cache.mirror_fault_check()
        torch.cuda.synchronize()
        return bad
    return run
