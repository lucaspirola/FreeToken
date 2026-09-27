"""GDN prefill copies at 8192 tokens (Ornith: in_proj row 12352 wide, conv_dim 8192 = q 2048 | k 2048 |
v 4096): torch transpose-contiguous vs the tiled Triton transpose, bitwise + time, tile sweep."""
import itertools, torch
import freetoken.kernel.triton.transpose as tp
from bench_moe import med
T = 8192
proj = torch.randn(T, 12352, device="cuda").to(torch.bfloat16)
conv_in = proj[:, :8192]
ref1 = conv_in.transpose(0, 1).contiguous()
out = torch.randn(8192, T, device="cuda").to(torch.bfloat16)
refq = [out[a:b].t().contiguous() for a, b in ((0, 2048), (2048, 4096), (4096, 8192))]
print(f"torch: conv_in^T {med(lambda: conv_in.transpose(0, 1).contiguous()) * 1000:.1f} us, "
      f"q/k/v^T {med(lambda: [out[a:b].t().contiguous() for a, b in ((0, 2048), (2048, 4096), (4096, 8192))]) * 1000:.1f} us")
for br, bc, nw in itertools.product((32, 64, 128), (32, 64, 128), (4, 8)):
    tp.TRANSPOSE_TILE = dict(BR=br, BC=bc, num_warps=nw)
    a = tp.transpose(conv_in)
    qs = [tp.transpose(out[x:y]) for x, y in ((0, 2048), (2048, 4096), (4096, 8192))]
    eq = torch.equal(a, ref1) and all(torch.equal(u, v) for u, v in zip(qs, refq))
    t1 = med(lambda: tp.transpose(conv_in)); t2 = med(lambda: [tp.transpose(out[x:y]) for x, y in ((0, 2048), (2048, 4096), (4096, 8192))])
    print(f"tile {br}x{bc} w{nw}: conv_in^T {t1 * 1000:.1f} us, q/k/v^T {t2 * 1000:.1f} us, equal={eq}", flush=True)
