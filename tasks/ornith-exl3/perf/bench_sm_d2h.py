"""Can SM stores into pinned host memory reach DMA speed on this host? 2 MB device -> pinned copies
with contiguous per-program chunks, several grid sizes and store cache modifiers."""
import torch
import triton
import triton.language as tl
import triton.testing as tt


@triton.jit
def _copy(src, dst, n, CHUNK: tl.constexpr, CM: tl.constexpr):
    base = tl.program_id(0).to(tl.int64) * CHUNK
    for o in range(0, CHUNK, 4096):
        i = base + o + tl.arange(0, 4096)
        m = i < n
        v = tl.load(src + i, mask=m)
        if CM == 1:
            tl.store(dst + i, v, mask=m, cache_modifier=".wt")
        elif CM == 2:
            tl.store(dst + i, v, mask=m, cache_modifier=".cs")
        else:
            tl.store(dst + i, v, mask=m)


n = 2 * 1024 * 1024 // 4
g = torch.randn(n, device="cuda")
h = torch.empty(n).pin_memory()
t = tt.do_bench(lambda: h.copy_(g, non_blocking=True), rep=100)
print(f"DMA D2H: {n * 4 / t / 1e6:.1f} GB/s")
for chunk in (4096, 16384, 65536, 262144):
    for cm, name in ((0, "default"), (1, ".wt"), (2, ".cs")):
        for nw in (4, 8):
            grid = (triton.cdiv(n, chunk),)
            t = tt.do_bench(lambda: _copy[grid](g, h, n, CHUNK=chunk, CM=cm, num_warps=nw), rep=50)
            ok = torch.equal(h, g.cpu())
            print(f"SM D2H chunk {chunk * 4 // 1024:5d} KiB/program ({grid[0]:4d} programs) {name:8s} w{nw}: {n * 4 / t / 1e6:5.1f} GB/s {'ok' if ok else 'MISMATCH'}")
