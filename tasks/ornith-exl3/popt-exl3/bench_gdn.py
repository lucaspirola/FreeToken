"""One GDN prefill layer at 8192 tokens (Ornith dims), conv-in to chunk output: torch strided copies
(757caee) vs tiled transposes. Uses the test's run() for identical inputs; prints median ms, equal."""
import sys, torch
sys.path.insert(0, "/home/lucas/ai/FreeToken-wt/popt-exl3/tests/kernels")
from bench_moe import med
import test_transpose as tt
from freetoken.kernel.triton.transpose import transpose
from freetoken.kernel.causal_conv1d import causal_conv1d_varlen
from freetoken.models.qwen3_5_moe.gdn_kernels import gdn_prefill_chunk_fla
torch.manual_seed(0)
H, D, HV, K, total = 16, 128, 32, 4, 8192
kd, vd = H * D, HV * D; conv_dim = 2 * kd + vd
proj = torch.randn(total, conv_dim + vd + 2 * HV, device="cuda").to(torch.bfloat16)
conv_in = proj[:, :conv_dim]
w = (torch.randn(conv_dim, K, device="cuda") * 0.5).to(torch.bfloat16)
cu = torch.tensor([0, total], device="cuda"); idx = torch.tensor([1], device="cuda", dtype=torch.int32)
init = torch.tensor([True], device="cuda")
cs = torch.randn(3, conv_dim, K - 1, device="cuda").to(torch.bfloat16); rs = torch.randn(3, HV, D, D, device="cuda") * 0.1
g = -torch.rand(1, total, HV, device="cuda"); beta = torch.rand(1, total, HV, device="cuda")
def run(new):
    x = transpose(conv_in) if new else conv_in.transpose(0, 1).contiguous()
    out = causal_conv1d_varlen(x, w, cs, cu, idx, init)
    if new: q, k, v = transpose(out[:kd]), transpose(out[kd:2 * kd]), transpose(out[2 * kd:])
    else: q, k, v = torch.split(out.transpose(0, 1), [kd, kd, vd], dim=-1)
    q, k, v = (t.reshape(1, total, -1, D) for t in (q, k, v))
    return gdn_prefill_chunk_fla(q, k, v, g, beta, state_source=rs, indices=idx, cu_seqlens=cu, scale=D ** -0.5)
for r in range(3):
    for new in (False, True, True, False):
        print(f"round {r} {'tiled ' if new else 'strided'}: {med(lambda: run(new)):.3f} ms", flush=True)
