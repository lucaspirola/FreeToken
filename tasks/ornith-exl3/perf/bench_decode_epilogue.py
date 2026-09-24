"""One Ornith MoE decode layer (EXL3 5.0bpw mul1, H 2048, I 512, top-8) and one dense GEMV
(o_proj 4096 -> 2048), replayed in a CUDA graph: split-K reduced by torch.sum vs inside the
GEMV (FREETOKEN_EXL3_SPLITK_INKERNEL), MoE epilogues unfused vs fused
(FREETOKEN_EXL3_FUSED_EPILOGUE), input rotation as its own had_rows launch vs inside the GEMV
prologue (FREETOKEN_EXL3_GEMV_PREROT). Prints us per replay and the max difference to the unfused
path. A GDN in_proj-shaped dense GEMV (2048 -> 12288) is timed too."""
import os

import torch
import triton.testing as tt

from freetoken.kernel.triton.exl3 import Exl3Parts
from freetoken.layers.quantization.linear.exl3 import exl3_forward
from freetoken.models.exl3_banks import exl3_bank_shapes
from freetoken.moe.fused_exl3 import fused_experts_exl3

H, I, TOPK, SLOTS, BITS = 2048, 512, 8, 512, 5
dev = torch.device("cuda")
g = torch.Generator(device=dev).manual_seed(0)
banks = []
for name, (shape, dtype) in exl3_bank_shapes(H, I, BITS).items():
    if dtype == torch.int16:
        banks.append(torch.randint(-(1 << 15), 1 << 15, (SLOTS, *shape), generator=g, device=dev,
                                   dtype=torch.int32).to(torch.int16))
    else:
        scale = 0.5 if name.endswith("suh") else 0.05
        banks.append((torch.randn((SLOTS, *shape), generator=g, device=dev) * scale).half())
banks = tuple(banks)
x = torch.randn(1, H, generator=g, device=dev).to(torch.bfloat16)
ids = torch.randperm(SLOTS, generator=g, device=dev)[:TOPK].to(torch.int32).view(1, TOPK)
w = torch.softmax(torch.randn(1, TOPK, generator=g, device=dev), dim=-1)

k, n = 4096, 2048
parts = Exl3Parts.build(k, (n,), BITS, "mul1", "cuda")
words = (k // 16) * (n // 16) * 8 * BITS  # int32 words: 16 * BITS int16 per 16x16 tile
tr = torch.randint(-(1 << 31), (1 << 31) - 1, (words,), generator=g, device=dev, dtype=torch.int32)
suh = (torch.randn(k, generator=g, device=dev) * 0.5).half()
svh = (torch.randn(n, generator=g, device=dev) * 0.05).half()
xd = torch.randn(1, k, generator=g, device=dev).to(torch.bfloat16)


k2, n2 = 2048, 12288
parts2 = Exl3Parts.build(k2, (n2,), BITS, "mul1", "cuda")
tr2 = torch.randint(-(1 << 31), (1 << 31) - 1, ((k2 // 16) * (n2 // 16) * 8 * BITS,), generator=g, device=dev, dtype=torch.int32)
suh2 = (torch.randn(1, k2, generator=g, device=dev) * 0.5).half()
svh2 = (torch.randn(n2, generator=g, device=dev) * 0.05).half()
xd2 = torch.randn(1, k2, generator=g, device=dev).to(torch.bfloat16)


def moe():
    return fused_experts_exl3(x, banks, w, ids, bits=BITS, codebook="mul1", is_prefill=False)


def dense():
    return exl3_forward(xd, tr, suh, svh, parts, torch.bfloat16)


def dense_in():
    return exl3_forward(xd2, tr2, suh2, svh2, parts2, torch.bfloat16)


def graphed(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    return graph, out


ref = {}
for label, env in (("unfused, torch.sum", {"FREETOKEN_EXL3_SPLITK_INKERNEL": "0", "FREETOKEN_EXL3_FUSED_EPILOGUE": "0", "FREETOKEN_EXL3_GEMV_PREROT": "0"}),
                   ("in-kernel sum", {"FREETOKEN_EXL3_SPLITK_INKERNEL": "1", "FREETOKEN_EXL3_FUSED_EPILOGUE": "0", "FREETOKEN_EXL3_GEMV_PREROT": "0"}),
                   ("in-kernel sum + fused", {"FREETOKEN_EXL3_SPLITK_INKERNEL": "1", "FREETOKEN_EXL3_FUSED_EPILOGUE": "1", "FREETOKEN_EXL3_GEMV_PREROT": "0"}),
                   ("+ gemv pre-rotation", {"FREETOKEN_EXL3_SPLITK_INKERNEL": "1", "FREETOKEN_EXL3_FUSED_EPILOGUE": "1", "FREETOKEN_EXL3_GEMV_PREROT": "1"})):
    os.environ.update(env)
    for name, fn in (("moe layer", moe), ("dense o_proj", dense), ("dense in_proj", dense_in)):
        graph, out = graphed(fn)
        graph.replay()
        torch.cuda.synchronize()
        base = ref.setdefault(name, out.clone())
        err = (out.float() - base.float()).abs().max().item()
        t = tt.do_bench(graph.replay, rep=300) * 1e3
        print(f"{name:12s} {label:24s} {t:7.1f} us/replay  max|d| vs unfused {err:.3g}", flush=True)
