"""The tiled transpose is a pure copy: bitwise equal to ``x.t().contiguous()``, and the GDN prefill
path built on it (conv in, q/k/v out) gives the same conv states, recurrent states and output as the
torch-strided-copy path it replaced."""

import pytest
import torch

from freetoken.kernel.triton.transpose import transpose

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(1, 1), (7, 3), (65, 129), (300, 8192), (8192, 64), (0, 5)])
def test_transpose_equals_torch(shape, dtype):
    base = torch.randn(shape[0], shape[1] + 37, device="cuda").to(dtype)
    for x in (base[:, : shape[1]], base[:, 37:], base[:, : shape[1]].contiguous()):
        assert torch.equal(transpose(x), x.t().contiguous())


@cuda
@pytest.mark.parametrize("lens", [[8192], [300, 1, 4000, 777], [64]])
def test_gdn_prefill_transposes_equal_strided_copies(lens):
    from freetoken.kernel.causal_conv1d import causal_conv1d_varlen
    from freetoken.models.qwen3_5_moe.gdn_kernels import gdn_prefill_chunk_fla

    torch.manual_seed(0)
    H, D, HV, K = 16, 128, 32, 4
    kd, vd = H * D, HV * D
    conv_dim, total, n = 2 * kd + vd, sum(lens), len(lens)
    proj = torch.randn(total, conv_dim + vd + 2 * HV, device="cuda").to(torch.bfloat16)
    conv_in = proj[:, :conv_dim]
    w = (torch.randn(conv_dim, K, device="cuda") * 0.5).to(torch.bfloat16)
    cu = torch.tensor([0] + lens, device="cuda").cumsum(0)
    idx = torch.arange(n, device="cuda", dtype=torch.int32) + 1
    init = torch.tensor([i % 2 == 1 for i in range(n)], device="cuda")
    conv0 = torch.randn(n + 2, conv_dim, K - 1, device="cuda").to(torch.bfloat16)
    rec0 = torch.randn(n + 2, HV, D, D, device="cuda") * 0.1
    rec0[idx[~init].long()] = 0
    g = -torch.rand(1, total, HV, device="cuda")
    beta = torch.rand(1, total, HV, device="cuda")

    def run(new):
        cs, rs = conv0.clone(), rec0.clone()
        x = transpose(conv_in) if new else conv_in.transpose(0, 1).contiguous()
        out = causal_conv1d_varlen(x, w, cs, cu, idx, init)
        if new:
            q, k, v = transpose(out[:kd]), transpose(out[kd:2 * kd]), transpose(out[2 * kd:])
        else:
            q, k, v = torch.split(out.transpose(0, 1), [kd, kd, vd], dim=-1)
        q, k, v = (t.reshape(1, total, -1, D) for t in (q, k, v))
        o = gdn_prefill_chunk_fla(q, k, v, g, beta, state_source=rs, indices=idx,
                                  cu_seqlens=cu, scale=D ** -0.5)
        return q, k, v, o, cs, rs

    for a, b in zip(run(False), run(True)):
        assert torch.equal(a, b)
