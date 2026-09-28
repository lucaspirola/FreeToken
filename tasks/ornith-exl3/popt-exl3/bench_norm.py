"""GDN output norm at 8192 tokens (Ornith: 32 v-heads x 128, z a column slice of the 12352-wide in_proj
output): row-per-head launch on a contiguous copy of z (757caee) vs the strided-z heads launch."""
import torch
from bench_moe import med
from freetoken.kernel.fla import rms_norm_gated, rms_norm_gated_heads
T, H, D = 8192, 32, 128
x = torch.randn(T, H, D, device="cuda").to(torch.bfloat16)
proj = torch.randn(T, 12352, device="cuda").to(torch.bfloat16)
z = proj[:, 8192:12288].reshape(T, H, D)
w = torch.randn(D, device="cuda").to(torch.bfloat16)
old = lambda: rms_norm_gated(x=x.reshape(-1, D), weight=w, bias=None, z=z.reshape(-1, D), eps=1e-6, is_rms_norm=True, norm_before_gate=True, activation="silu")
new = lambda: rms_norm_gated_heads(x=x, weight=w, z=z, eps=1e-6, activation="silu")
print("equal", torch.equal(old(), new().reshape(-1, D)))
for r in range(3):
    print(f"round {r}: rows+copy {med(old) * 1000:.1f} us, heads {med(new) * 1000:.1f} us, heads {med(new) * 1000:.1f} us, rows+copy {med(old) * 1000:.1f} us", flush=True)
