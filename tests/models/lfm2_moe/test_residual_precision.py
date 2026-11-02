"""lfm2_moe keeps its residual stream in fp32 (exllamav3 does too); the normed outputs stay in the
model dtype. CPU tests of the torch path, plus a CUDA check of the Triton path against it."""

from __future__ import annotations

import pytest
import torch
from freetoken.models.lfm2_moe.model import Lfm2MoeDecoderLayer, Lfm2MoeModel, add_rmsnorm_fp32

H = 64
EPS = 1e-5


class _Norm:
    def __init__(self, weight: torch.Tensor):
        self.weight, self.eps = weight, EPS


def _ref_norm(r: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    r = r.double()
    return r * torch.rsqrt(r.pow(2).mean(-1, keepdim=True) + EPS) * w.double()


def _weight(device="cpu") -> torch.Tensor:
    return (1 + 0.1 * torch.randn(H, generator=torch.Generator().manual_seed(1))).to(torch.bfloat16).to(device)


def test_first_call_starts_an_fp32_stream_from_x():
    norm = _Norm(_weight())
    x = torch.randn(5, H).to(torch.bfloat16)
    y, r = add_rmsnorm_fp32(norm, x, None)
    assert r.dtype == torch.float32 and y.dtype == torch.bfloat16
    assert torch.equal(r, x.float())
    torch.testing.assert_close(y.double(), _ref_norm(x.float(), norm.weight).to(torch.bfloat16).double())


def test_stream_keeps_updates_a_bf16_stream_drops():
    # 64 updates of 0.1 on a stream at 100: bf16 spacing there is 0.5, so a bf16 stream stays at 100
    norm = _Norm(_weight())
    start = torch.full((3, H), 100.0)
    step = torch.full((3, H), 0.1).to(torch.bfloat16)
    _, r = add_rmsnorm_fp32(norm, start.to(torch.bfloat16), None)
    r_bf16 = start.to(torch.bfloat16)
    for _ in range(64):
        y, r = add_rmsnorm_fp32(norm, step.clone(), r)
        r_bf16 = (r_bf16.float() + step.float()).to(torch.bfloat16)
    assert r.dtype == torch.float32 and y.dtype == torch.bfloat16
    exact = 100.0 + 64 * float(step[0, 0])
    assert (r - exact).abs().max() < 1e-3
    assert (r_bf16.float() - 100.0).abs().max() == 0
    torch.testing.assert_close(y.double(), _ref_norm(r, norm.weight).to(torch.bfloat16).double())


def test_model_threads_the_fp32_residual_through_its_layers():
    torch.manual_seed(0)
    deltas = [torch.randn(4, H) * 0.01 for _ in range(6)]
    seen = []

    def make_layer(i):
        layer = object.__new__(Lfm2MoeDecoderLayer)
        layer._layer_id, layer._is_conv = i, True
        layer.operator_norm, layer.ffn_norm = _Norm(_weight()), _Norm(_weight())

        def mix(x, d=deltas[2 * i]):
            return (x.float() * 0 + d).to(x.dtype)

        def ffn(x, d=deltas[2 * i + 1]):
            return (x.float() * 0 + d).to(x.dtype)

        layer.conv = type("C", (), {"forward": staticmethod(mix)})()
        layer.feed_forward = type("F", (), {"forward": staticmethod(ffn)})()
        orig = layer.forward

        def fwd(x, residual=None):
            y, res = orig(x, residual)
            seen.append(res.dtype)
            return y, res

        layer.forward = fwd
        return layer

    model = object.__new__(Lfm2MoeModel)
    emb = torch.randn(4, H).to(torch.bfloat16)
    model.embed_tokens = type("E", (), {"forward": staticmethod(lambda ids: emb.clone())})()
    model.layers = type("L", (), {"op_list": [make_layer(i) for i in range(3)]})()
    model.embedding_norm = _Norm(_weight())
    out = model.forward(torch.zeros(4, dtype=torch.int32))
    assert seen == [torch.float32] * 3 and out.dtype == torch.bfloat16
    stream = emb.double() + sum(d.to(torch.bfloat16).double() for d in deltas)
    torch.testing.assert_close(out.double(), _ref_norm(stream, model.embedding_norm.weight), atol=2e-2, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_triton_path_matches_the_torch_path(dtype):
    norm_gpu, norm_cpu = _Norm(_weight("cuda").to(dtype)), _Norm(_weight().to(dtype))
    g = torch.Generator().manual_seed(2)
    x0, x1 = (torch.randn(7, H, generator=g).to(dtype) for _ in range(2))
    yc, rc = add_rmsnorm_fp32(norm_cpu, x0.clone(), None)
    yc, rc = add_rmsnorm_fp32(norm_cpu, x1.clone(), rc)
    yg, rg = add_rmsnorm_fp32(norm_gpu, x0.cuda(), None)
    yg, rg = add_rmsnorm_fp32(norm_gpu, x1.cuda(), rg)
    assert rg.dtype == torch.float32 and yg.dtype == dtype
    torch.testing.assert_close(rg.cpu(), rc, atol=0, rtol=0)
    torch.testing.assert_close(yg.cpu().float(), yc.float(), atol=1e-2, rtol=1e-2)
