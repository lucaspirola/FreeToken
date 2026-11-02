"""One EXL3 MoE layer prefill (fused_experts_exl3, 8192 tokens, uniform random routing) vs the
decoded W_hat scratch group, on LFM2.5-24B-A2B and Ornith-1.5-35B-A3B expert shapes.

    PYTHONPATH=python python tasks/lfm25-scratch/bench_decode_group.py [--reps N]

Two sweeps per model: "gu" forces only the gate|up group (down keeps the L2 rule), "both" forces
both. Reps are interleaved across group sizes. Every output is checked bitwise against group 32.
Scratch = decoded fp16 W_hat of one gate|up group + one down group. Peak = allocator peak of the
call above the inputs.
"""
import sys

import torch

import freetoken.moe.fused_exl3 as fe
from freetoken.models.exl3_banks import exl3_bank_shapes

DEV = torch.device("cuda")
M, BITS = 8192, 5
REPS = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 15
MODELS = {"lfm25": (2048, 1536, 64, 4), "ornith": (2048, 512, 256, 8)}
GROUPS = (2, 4, 8, 16, 32)
ORIG = fe._decode_group


def make(H, I, E, K):
    torch.manual_seed(0)
    B = []
    for name, (shape, dtype) in exl3_bank_shapes(H, I, BITS).items():
        if dtype == torch.int16:
            B.append(torch.randint(-(1 << 15), 1 << 15, (E, *shape), dtype=torch.int32, device=DEV).to(torch.int16))
        else:
            B.append((torch.randn((E, *shape), device=DEV) * (0.5 if name.endswith("suh") else 0.05)).half())
    x = torch.randn(M, H, dtype=torch.bfloat16, device=DEV)
    w = torch.softmax(torch.randn(M, K, device=DEV), -1)
    ids = torch.stack([torch.randperm(E, device=DEV)[:K] for _ in range(M)]).to(torch.int32)
    return lambda: fe.fused_experts_exl3(x, tuple(B), w, ids, bits=BITS, codebook="mul1", is_prefill=True, num_experts=E)


def force(mode, g, H):
    def pick(parts, num_experts, device):
        if mode == "both" or parts.k == H:  # gate|up is [H -> 2I], down is [I -> H]
            return min(g, num_experts)
        return ORIG(parts, num_experts, device)
    fe._decode_group = pick


def groups_used(run):
    seen = []
    def spy(parts, num_experts, device):
        g = fe._decode_group_inner(parts, num_experts, device)
        seen.append((parts.k, parts.n, g))
        return g
    fe._decode_group_inner, fe._decode_group = fe._decode_group, spy
    run(); torch.cuda.synchronize()
    fe._decode_group = fe._decode_group_inner
    return seen


def main():
    l2 = torch.cuda.get_device_properties(0).L2_cache_size
    print(f"GPU {torch.cuda.get_device_name(0)}  L2 {l2 / 2**20:.0f} MiB  M={M}  reps={REPS}")
    for model, (H, I, E, K) in MODELS.items():
        gu_b, dn_b = H * 2 * I * 2, I * H * 2
        run = make(H, I, E, K)
        fe._decode_group = ORIG
        default = groups_used(run)
        print(f"\n{model}: H={H} I={I} E={E} top-{K}; gate|up {gu_b / 2**20:.0f} MiB/expert, down {dn_b / 2**20:.0f} MiB/expert;"
              f" shipped groups (k, n, group) = {default}")
        force("both", 32, H)
        ref = run(); torch.cuda.synchronize()
        cfgs = [("default", None)] + [(m, g) for m in ("gu", "both") for g in GROUPS]
        ts = {c: [] for c in cfgs}
        peak = {}
        for rep in range(REPS):
            for c in cfgs:
                if c[0] == "default":
                    fe._decode_group = ORIG
                else:
                    force(*c, H)
                run(); torch.cuda.synchronize()
                if rep == 0:
                    torch.cuda.reset_peak_memory_stats()
                    base = torch.cuda.memory_allocated()
                a, b = torch.cuda.Event(True), torch.cuda.Event(True)
                a.record(); o = run(); b.record(); b.synchronize()
                ts[c].append(a.elapsed_time(b))
                if rep == 0:
                    peak[c] = (torch.cuda.max_memory_allocated() - base) / 2**20
                    assert torch.equal(o, ref), c
                    del o
                    c_groups = groups_used(run)
                    scr = sum(k * n * 2 * g for k, n, g in c_groups) / 2**20
                    peak[c] = (peak[c], scr, [g for _, _, g in c_groups])
        fe._decode_group = ORIG
        t0 = sorted(ts[("default", None)])[REPS // 2]
        print(f"{'config':14s} {'groups gu/dn':>12s} {'median ms':>10s} {'min ms':>8s} {'vs shipped':>10s} {'scratch MiB':>11s} {'peak MiB':>9s}")
        for c in cfgs:
            v = sorted(ts[c]); med = v[REPS // 2]
            pk, scr, gs = peak[c]
            name = c[0] if c[1] is None else f"{c[0]} g{c[1]}"
            print(f"{name:14s} {'/'.join(map(str, gs)):>12s} {med:10.3f} {v[0]:8.3f} {100 * (med / t0 - 1):+9.1f}% {scr:11.0f} {pk:9.0f}")
        del run, ref
        torch.cuda.empty_cache()
    print("\nall outputs bitwise equal to group 32")


if __name__ == "__main__":
    main()
