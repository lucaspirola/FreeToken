"""Host cost of cudaGraphLaunch vs graph content (box microbench).
A base graph of 800 tiny torch kernels (the decode graph has ~800 nodes), plus 92 extra nodes of
one kind: tiny torch kernels, single-CTA Triton kernels with 4 args, or with 30 args (like
resolve_swaps). Host time of graph.replay() after a sync, median of 300."""
import time, statistics as S
import torch, triton, triton.language as tl

@triton.jit
def k4(a, b, n, BLOCK: tl.constexpr):
    o = tl.arange(0, BLOCK)
    tl.store(a + o, tl.load(b + o, mask=o < n), mask=o < n)

@triton.jit
def k30(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18,
        s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, BLOCK: tl.constexpr):
    o = tl.arange(0, BLOCK)
    tl.store(p0 + o, tl.load(p1 + o, mask=o < s0), mask=o < s0)

x = torch.zeros(1024, device="cuda", dtype=torch.int32)
y = torch.ones(1024, device="cuda", dtype=torch.int32)
ptrs = [torch.zeros(64, device="cuda", dtype=torch.int32) for _ in range(19)]

def base():
    for _ in range(800):
        x.add_(1)

def extra(kind):
    for _ in range(92):
        if kind == "torch":
            y.add_(1)
        elif kind == "triton4":
            k4[(1,)](x, y, 256, BLOCK=256)
        elif kind == "triton30":
            k30[(1,)](*ptrs, 16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, BLOCK=64)

for kind in ("none", "torch", "triton4", "triton30", "none"):
    fn = (lambda: base()) if kind == "none" else (lambda k=kind: (base(), extra(k)))
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn(); fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    t = []
    for i in range(300):
        torch.cuda.synchronize()
        t0 = time.perf_counter(); g.replay(); t.append(time.perf_counter() - t0)
    torch.cuda.synchronize()
    print(f"{kind:9s} replay() host {S.median(t)*1e6:7.1f} us (p90 {sorted(t)[270]*1e6:.1f})", flush=True)
