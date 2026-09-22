"""End-to-end check of the device swap path against a real OffloadMoeCache.

Proves what the unit tests cannot: that `resolve_swaps` picks the right pool
rows, that the D2H-then-H2D ordering leaves both the GPU slot and the mirror row
holding the correct bytes, and that coverage survives prefill, a long eviction
chain and an arena shrink.

Run:  FREETOKEN_EXPERT_ARENA=1 python tasks/exclusive-expert-ram/swap_smoke.py
"""
import json, os, struct, sys, tempfile
import torch

# Resolve the branch's python/ regardless of cwd: the CUDA extensions are built
# in the main checkout's venv, so this is usually run as
#   cd ~/ai/FreeToken && PYTHONPATH=<branch>/python .venv/bin/python <this file>
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "python"))
os.environ.setdefault("FREETOKEN_EXPERT_ARENA", "1")

from freetoken.moe.mirror_pool import MirrorExpertPool, nvfp4_bank_shapes, plan_capacity
import types

from freetoken.models.nemotron_h.weight import (
    NVFP4_EXPERT_SOURCE_SPEC as NEMOTRON_SPEC,
)
from freetoken.moe.offload_cache import OffloadMoeCache

L, E, H, ISZ = 3, 8, 32, 32
SUFFIX = {
    "gate_up_packed": ("up_proj.weight", "U8"),
    "gate_up_scale": ("up_proj.weight_scale", "F8_E4M3"),
    "gate_up_global": ("up_proj.weight_scale_2", "F32"),
    "down_packed": ("down_proj.weight", "U8"),
    "down_scale": ("down_proj.weight_scale", "F8_E4M3"),
    "down_global": ("down_proj.weight_scale_2", "F32"),
}


def write_ckpt(root):
    shapes = nvfp4_bank_shapes(H, ISZ)
    header, blob, off = {}, bytearray(), 0
    for layer in range(L):
        for e in range(E):
            flat = layer * E + e
            for name, (tail, _dt) in shapes.items():
                suffix, dt = SUFFIX[name]
                key = f"backbone.layers.{layer}.mixer.experts.{e}.{suffix}"
                if dt == "F32":
                    payload, shape = struct.pack("<f", float(flat + 1)), []
                else:
                    n = 1
                    for d in tail:
                        n *= d
                    payload = bytes(((flat * 13 + k) % 251) + 1 for k in range(n))
                    shape = list(tail)
                header[key] = {"dtype": dt, "shape": shape,
                               "data_offsets": [off, off + len(payload)]}
                blob += payload
                off += len(payload)
    head = json.dumps(header).encode()
    with open(os.path.join(root, "model.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", len(head))); f.write(head); f.write(blob)
    with open(os.path.join(root, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": {k: "model.safetensors" for k in header}}, f)
    with open(os.path.join(root, "config.json"), "w") as f:
        json.dump({"layers_block_type": ["moe"] * L}, f)


def holes(cache):
    fwd = cache._mirror["pool_row_of_id"].cpu().tolist()
    live = set(f for f in cache.id_of_slot[: cache.cache_size].cpu().tolist() if f >= 0)
    return [f for f in range(L * E) if f not in live and fwd[f] < 0]


def main():
    dev = torch.device("cuda")
    failures = []
    with tempfile.TemporaryDirectory() as root:
        write_ckpt(root)
        # Under prefill_overlap the double buffer owns [0, 2E) outright, so
        # set_usable_slots' floor is 2*E = 16 (see offload_cache.py's
        # _mirror_prefill_base / the "floor" comment in set_usable_slots) --
        # start above it so the arena-shrink check below still has room to
        # shrink into and land above the floor.
        gpu = 20
        cfg = types.SimpleNamespace(moe_layer_ids=list(range(L)))
        pool = MirrorExpertPool(root, L, E, plan_capacity(L, E, gpu),
                                hidden_size=H, intermediate_size=ISZ,
                                spec=NEMOTRON_SPEC, config=cfg, device=dev)
        full = MirrorExpertPool(root, L, E, L * E,
                                hidden_size=H, intermediate_size=ISZ,
                                spec=NEMOTRON_SPEC, config=cfg, device=dev)
        full.load_initial(set())
        golden = {f: {n: full.banks[n][full.pool_row_of_id[f]].clone()
                      for n in full.shapes} for f in range(L * E)}
        full.close()

        cache = OffloadMoeCache(
            num_layers=L, num_experts=E, cache_size=gpu, slot_capacity=gpu,
            arena_step_slots=4, device=dev, quant_format="nvfp4", cache_policy="lfu",
            prefill_overlap=True,
        )
        cache.direct_device_banks = True
        cache.attach_mirror_pool(pool)
        info = cache.mirror_warm_start()
        print(f"warm start: {info}")
        if holes(cache):
            failures.append(f"{len(holes(cache))} coverage holes after warm start")

        def check(flat, slot, where):
            for name in pool.shapes:
                got = cache.bank_caches[name][slot].view(torch.uint8).cpu()
                want = golden[flat][name].view(torch.uint8).cpu()
                if not torch.equal(got, want):
                    failures.append(f"{where}: expert {flat} bank {name} mismatch")
                    return False
            return True

        def check_view(flat, views, e, where):
            for name, view in zip(cache.bank_schema, views):
                got = view[e].view(torch.uint8).cpu()
                want = golden[flat][name].view(torch.uint8).cpu()
                if not torch.equal(got, want):
                    failures.append(f"{where}: expert {flat} bank {name} mismatch")
                    return False
            return True

        # Prefill: every layer assembled through the real overlap choreography
        # (begin_prefill / prefetch_prefill_layer / wait_prefill_layer /
        # release_prefill_layer -- see tests/moe/test_mirror_prefill.py's
        # _sweep), every expert byte-exact. Under the bounded mirror
        # materialize_layer is not a legal prefill path (it schedules a copy
        # for every expert of the layer including already-resident ones,
        # which would need the whole layer staged into the mirror at once);
        # the buffer views wait_prefill_layer hands back are the only place
        # prefill's bytes live, so the check reads those, not a cache slot.
        cache.begin_prefill()
        for lid in range(L):
            cache.prefetch_prefill_layer(lid)
            cache.prefetch_prefill_layer(lid + 1)
            views = cache.wait_prefill_layer(lid)
            torch.cuda.synchronize()
            for e in range(E):
                check_view(lid * E + e, views, e, f"prefill({lid})")
            cache.release_prefill_layer(lid)
            # No coverage assertion here: prefill deliberately drops displaced
            # experts (preserving them would need the whole model mirrored).
            # Coverage is re-established at the prefill -> decode boundary and
            # asserted per decode step below.
        torch.cuda.synchronize()
        print(f"prefill: {L} layers assembled via overlap choreography, "
              f"faults={int(cache._mirror['stats'][3])}")

        # Decode: distinct experts per step so slot<->expert pairing is unambiguous.
        torch.manual_seed(0)
        checked = 0
        for step in range(40):
            lid = step % L
            perm = torch.randperm(E, device=dev)[:4].to(torch.int32)
            ids = perm.clone().reshape(1, 4)
            want = perm.tolist()
            cache.ensure_experts(lid, ids)
            cache.copy_missing()
            torch.cuda.synchronize()
            for e, slot in zip(want, ids.reshape(-1).tolist()):
                check(lid * E + e, slot, f"decode step {step}")
                checked += 1
            if holes(cache):
                failures.append(f"coverage holes at decode step {step}")
        print(f"decode: {checked} expert rows verified over 40 routed steps")

        # Arena shrink: slots handed to the KV cache must not strand experts.
        cache.set_usable_slots(16)
        if holes(cache):
            failures.append("coverage holes after arena shrink")

        stats = cache.mirror_stats()
        if stats["coverage_faults"]:
            failures.append(f"{stats['coverage_faults']} coverage faults")
        if stats["starved_writebacks"]:
            failures.append(f"{stats['starved_writebacks']} starved writebacks")
        print(f"stats: {stats}")

        # Mirror rows still hold the checkpoint's bytes for whatever they own.
        inv = cache._mirror["id_of_pool_row"].cpu().tolist()
        for row, flat in enumerate(inv):
            if flat < 0:
                continue
            for name in pool.shapes:
                if not torch.equal(pool.banks[name][row].view(torch.uint8),
                                   golden[flat][name].view(torch.uint8).cpu()):
                    failures.append(f"mirror row {row} (expert {flat}) corrupted")
        pool.close()

    if failures:
        print("\nFAIL:")
        for f in failures[:10]:
            print(f"  - {f}")
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
