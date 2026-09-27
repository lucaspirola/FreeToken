"""Median device time (CUDA events) + sha1 of the output, EXL3 prefill paths at Ornith shapes, for a
tree A/B (run with PYTHONPATH=<tree>/python): fused MoE prefill (256 experts, top-8, 5-bit mul1) at
8192 / 2048 / 512 / 64 tokens, uniform and skewed (Zipf s=1) routing; dense exl3_forward at 8192 rows
(2048 -> 1024|512, and 2048 -> 12288 in 6 parts ~ GDN in_proj). Usage: python bench_moe.py TAG"""
import hashlib, sys
import torch
import freetoken.kernel.triton.exl3 as k3
import freetoken.moe.fused_exl3 as fe
from freetoken.layers.quantization.linear.exl3 import exl3_forward
from freetoken.models.exl3_banks import exl3_bank_shapes

DEV = torch.device("cuda")
H, I, E, TOPK, BITS = 2048, 512, 256, 8, 5
torch.manual_seed(0)
B = []
for name, (shape, dtype) in exl3_bank_shapes(H, I, BITS).items():
    if dtype == torch.int16:
        B.append(torch.randint(-(1 << 15), 1 << 15, (E, *shape), dtype=torch.int32, device=DEV).to(torch.int16))
    else:
        B.append((torch.randn((E, *shape), device=DEV) * (0.5 if name.endswith("suh") else 0.05)).half())
B = tuple(B)


def med(fn, reps=20):
    fn(); fn(); torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record(); b.synchronize(); ts.append(a.elapsed_time(b))
    return sorted(ts)[reps // 2]


def sha(t):
    return hashlib.sha1(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:10]


row = [sys.argv[1]]
g = torch.Generator(device="cpu").manual_seed(1)
for M in (8192, 2048, 512, 64):
    for skew in (0, 1):
        x = torch.randn(M, H, generator=g).to(torch.bfloat16).to(DEV)
        w = torch.softmax(torch.randn(M, TOPK, generator=g), -1).to(DEV)
        if skew:
            p = 1.0 / torch.arange(1, E + 1, dtype=torch.float32)
            p = p[torch.randperm(E, generator=g)]
            ids = torch.multinomial(p.expand(M, E), TOPK, generator=g).to(torch.int32).to(DEV)
        else:
            ids = torch.stack([torch.randperm(E, generator=g)[:TOPK] for _ in range(M)]).to(torch.int32).to(DEV)
        f = lambda: fe.fused_experts_exl3(x, B, w, ids, bits=BITS, codebook="mul1", is_prefill=True, num_experts=E)
        row.append(f"moe{M}{'s' if skew else 'u'}={med(f):.4f}:{sha(f())}")
for sizes in ((1024, 512), (2048,) * 6):
    parts = k3.Exl3Parts.build(2048, sizes, BITS, "mul1", DEV)
    tr = torch.randint(-(1 << 15), 1 << 15, (parts.words * 2,), generator=g, dtype=torch.int32).to(torch.int16).to(DEV)
    suh = (torch.randn(len(sizes) * 2048, generator=g) * 0.5).half().to(DEV)
    svh = (torch.randn(sum(sizes), generator=g) * 0.05).half().to(DEV)
    x = torch.randn(8192, 2048, generator=g).to(torch.bfloat16).to(DEV)
    f = lambda: exl3_forward(x, tr, suh, svh, parts, torch.bfloat16)
    row.append(f"dense{sum(sizes)}={med(f):.4f}:{sha(f())}")
print("\t".join(row), flush=True)
