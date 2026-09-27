"""Interleaved A/B of fused MoE prefill variants (module switches) at Ornith shapes, uniform and
skewed (Zipf-like, s=1) routing; asserts every variant's output is bitwise the first's.
Usage: python ab_moe.py 'NAME:mod.ATTR=VAL[,mod.ATTR=VAL]' ...   (first spec = reference)"""
import sys
import torch
import freetoken.kernel.triton.exl3 as k3
import freetoken.moe.fused_exl3 as fe
from freetoken.models.exl3_banks import exl3_bank_shapes

DEV = torch.device("cuda")
H, I, E, TOPK, BITS = 2048, 512, 256, 8, 5
MODS = {"fe": fe, "k3": k3}
torch.manual_seed(0)
B = []
for name, (shape, dtype) in exl3_bank_shapes(H, I, BITS).items():
    if dtype == torch.int16:
        B.append(torch.randint(-(1 << 15), 1 << 15, (E, *shape), dtype=torch.int32, device=DEV).to(torch.int16))
    else:
        B.append((torch.randn((E, *shape), device=DEV) * (0.5 if name.endswith("suh") else 0.05)).half())
B = tuple(B)
specs = []
for a in sys.argv[1:]:
    nm, _, kv = a.partition(":")
    sets = []
    for item in filter(None, kv.split(",")):
        k, v = item.split("=")
        m, attr = k.split(".")
        sets.append((MODS[m], attr, eval(v)))
    specs.append((nm, sets))
defaults = {(id(m), attr): getattr(m, attr) for _, ss in specs for m, attr, _ in ss}


def apply(sets):
    for (mid, attr), v in defaults.items():
        setattr(fe if id(fe) == mid else k3, attr, v)
    for m, attr, v in sets:
        setattr(m, attr, v)


def routing(M, skew):
    if not skew:
        return torch.stack([torch.randperm(E, device=DEV)[:TOPK] for _ in range(M)]).to(torch.int32)
    p = 1.0 / torch.arange(1, E + 1, device=DEV, dtype=torch.float32)
    p = p[torch.randperm(E, device=DEV)]
    return torch.multinomial(p.expand(M, E), TOPK).to(torch.int32)


for M in (8192, 2048, 512):
    for skew in (False, True):
        x = torch.randn(M, H, dtype=torch.bfloat16, device=DEV)
        w = torch.softmax(torch.randn(M, TOPK, device=DEV), -1)
        ids = routing(M, skew)
        run = lambda: fe.fused_experts_exl3(x, B, w, ids, bits=BITS, codebook="mul1", is_prefill=True, num_experts=E)
        ts = {nm: [] for nm, _ in specs}
        ref = None
        for rep in range(15):
            for nm, sets in specs:
                apply(sets)
                if rep == 0:
                    o = run(); torch.cuda.synchronize()
                    if ref is None:
                        ref = o
                    assert torch.equal(o, ref), f"{nm} differs at M={M} skew={skew}"
                a, b = torch.cuda.Event(True), torch.cuda.Event(True)
                a.record(); run(); b.record(); b.synchronize(); ts[nm].append(a.elapsed_time(b))
        base = sorted(ts[specs[0][0]])[7]
        print(f"M={M:5d} {'skew' if skew else 'unif'} " + "  ".join(
            f"{nm} {sorted(v)[7]:.3f} ms ({(sorted(v)[7] / base - 1) * 100:+.1f}%)" for nm, v in ts.items()), flush=True)
apply([])
print("all outputs bitwise equal")
