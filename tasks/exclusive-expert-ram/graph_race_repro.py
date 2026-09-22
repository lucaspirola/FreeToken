"""Graph-replay race repro for the mirror swap path.

Mimics the real system's structure: the decode step (ensure_experts +
copy_missing) is CAPTURED into a CUDA graph, replayed after eager prefill
sweeps, with the routing tensor rewritten between replays exactly like
GraphCaptureBuffer.copy_from does. Byte-checks every routed expert after each
replay.

If this reproduces the corruption seen on the live server (graphs on ->
degenerate output, graphs off -> correct), the race can be debugged locally.
"""
import json, os, struct, sys, tempfile
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "python"))
os.environ.setdefault("FREETOKEN_EXPERT_ARENA", "1")

from swap_smoke import write_ckpt, L, E, H, ISZ
from freetoken.moe.offload_cache import OffloadMoeCache
from freetoken.moe.offload_cache import set_expert_arena

# S7: the arena gate is a config value; publish the env the device test passes.
set_expert_arena(os.environ["FREETOKEN_EXPERT_ARENA"] == "1")
from freetoken.moe.mirror_pool import MirrorExpertPool, plan_capacity
import types

from freetoken.models.nemotron_h.weight import (
    NVFP4_EXPERT_SOURCE_SPEC as NEMOTRON_SPEC,
)


def main():
    dev = torch.device("cuda")
    failures = []
    with tempfile.TemporaryDirectory() as root:
        write_ckpt(root)
        total = L * E
        gpu = int(total * 0.74)
        cap = plan_capacity(L, E, gpu)
        cfg = types.SimpleNamespace(moe_layer_ids=list(range(L)))
        pool = MirrorExpertPool(root, L, E, cap, hidden_size=H,
                                intermediate_size=ISZ, spec=NEMOTRON_SPEC,
                                config=cfg, device=dev)
        full = MirrorExpertPool(root, L, E, total, hidden_size=H,
                                intermediate_size=ISZ, spec=NEMOTRON_SPEC,
                                config=cfg, device=dev)
        full.load_initial(set())
        golden = {f: full.banks["gate_up_packed"][full.pool_row_of_id[f]].clone()
                  for f in range(total)}
        full.close()

        c = OffloadMoeCache(num_layers=L, num_experts=E, cache_size=gpu,
                            slot_capacity=gpu, arena_step_slots=4,
                            device=dev, quant_format="nvfp4",
                            cache_policy="lfu", prefill_overlap=True)
        c.direct_device_banks = True
        c.attach_mirror_pool(pool)
        c.mirror_warm_start()

        # The captured decode step for ONE layer, exactly like the real one:
        # ensure_experts rewrites ids in place to slot ids, copy_missing moves
        # rows. The routing tensor is a persistent buffer whose CONTENT changes
        # per replay (GraphCaptureBuffer.copy_from semantics).
        layer = 0
        ids_buf = torch.zeros((1, 6), dtype=torch.int32, device=dev)

        # Eager warmup first (the real system runs model.forward once before
        # capture -- first-use kernels, allocator activity).
        ids_buf.copy_(torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.int32))
        c.ensure_experts(layer, ids_buf)
        c.copy_missing()
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            c.ensure_experts(layer, ids_buf)
            c.copy_missing()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        with torch.cuda.graph(g, stream=s):
            c.ensure_experts(layer, ids_buf)
            c.copy_missing()
        torch.cuda.synchronize()

        # Replay with fresh routing, faithfullly: copy_from writes into the
        # graph's static buffer but expects readers to consume it via tensor
        # IDENTITY -- so mirror that: per-step buffer + captured alias.
        decode_buf = torch.zeros((1, 6), dtype=torch.int32, device=dev)
        g2 = torch.cuda.CUDAGraph()
        s2 = torch.cuda.Stream()
        s2.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s2):
            c.ensure_experts(layer, decode_buf)
            c.copy_missing()
        torch.cuda.current_stream().wait_stream(s2)
        torch.cuda.synchronize()
        with torch.cuda.graph(g2, stream=s2):
            c.ensure_experts(layer, decode_buf)
            c.copy_missing()
        torch.cuda.synchronize()

        torch.manual_seed(0)
        for step in range(60):
            perm = torch.randperm(E, device=dev)[:6].to(torch.int32)
            want = perm.tolist()
            # GraphCaptureBuffer.copy_from semantics: buffer write on the
            # CURRENT stream, replay ordered after it.
            with torch.cuda.stream(torch.cuda.current_stream()):
                decode_buf.copy_(perm.reshape(1, 6).contiguous())
            g2.replay()
            torch.cuda.synchronize()
            # what did the buffer say the GEMM should read?
            buf_after = ids_buf if step < 0 else decode_buf
            buf_vals = buf_after.reshape(-1).cpu().tolist()
            slots = c.slot_for_id[layer].cpu().tolist()  # authoritative
            wrong_in_buf = []
            for e in want:
                sl = slots[e]
                if sl < 0: wrong_in_buf.append((step, e, 'absent')); continue
                got = c.bank_caches["gate_up_packed"][sl].view(torch.uint8).cpu()
                if not torch.equal(got, golden[layer * E + e].view(torch.uint8).cpu()):
                    wrong_in_buf.append((step, e, sl))
                if len(wrong_in_buf) <= 2 and wrong_in_buf:
                    print(f'   step{step}: want {want}')
                    print(f'      buf_after_replay={buf_vals}  sfi={slots}')
            failures.extend(wrong_in_buf)

        print(f"graph replay decode-only: {len(failures)} wrong experts over 60 replays")

        # Now the full pattern: eager prefill sweep, then replayed decode --
        # this is where the live server corrupts. The prefill sweep itself
        # must go through the real overlap choreography (begin_prefill /
        # prefetch_prefill_layer / wait_prefill_layer / release_prefill_layer
        # -- see tests/moe/test_mirror_prefill.py's _sweep): materialize_layer
        # is not a legal prefill path under the mirror (it schedules a copy
        # for every expert of the layer including already-resident ones,
        # which would need the whole layer staged into the mirror first).
        #
        # The old eager-vs-replay branch on a "_mirror_needs_coverage" flag
        # is gone along with the flag itself: that flag stood for a
        # batch-boundary coverage restore materialize_layer's whole-layer
        # invalidation used to require. Under the overlap path, decode
        # coverage is maintained by construction (ensure_experts's docstring
        # in offload_cache.py) -- there is nothing to restore, so every
        # step below is a plain graph replay. That keeps the boundary this
        # script exists to race: g's replay still runs immediately after a
        # real prefill sweep touched the mirror's residency maps through the
        # copy stream, on statically captured addresses whose CONTENT this
        # loop rewrites every step -- exactly the graphs-mode corruption
        # this reproduces.
        for rnd in range(3):
            c.begin_prefill()
            for lid in range(L):
                c.prefetch_prefill_layer(lid)
                c.prefetch_prefill_layer(lid + 1)
                c.wait_prefill_layer(lid)
                c.release_prefill_layer(lid)
            torch.cuda.synchronize()
            for step in range(20):
                perm = torch.randperm(E, device=dev)[:6].to(torch.int32)
                want = perm.tolist()
                ids_buf.copy_(perm.reshape(1, 6).contiguous())
                g.replay()
                torch.cuda.synchronize()
                slots = ids_buf.reshape(-1).tolist()
                for e, sl in zip(want, slots):
                    got = c.bank_caches["gate_up_packed"][sl].view(torch.uint8).cpu()
                    if not torch.equal(got, golden[layer * E + e].view(torch.uint8).cpu()):
                        failures.append((f"r{rnd}s{step}", e, sl))
        print(f"after 3x(eager prefill sweep + 20 replayed decode steps): "
              f"{len(failures)} wrong experts total")
        print("stats:", c.mirror_stats())
        pool.close()

    if failures:
        print("FAIL: first failures:", failures[:5])
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
