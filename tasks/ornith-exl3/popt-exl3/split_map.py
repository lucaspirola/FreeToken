"""Where does cuBLAS (torch.mm fp16) stop being a sequential-k fp32 GEMM? Compare it bitwise with
the Triton sequential-k GEMM for M = 256..8192 on the dense EXL3 folded shapes."""
import torch
from tri_vs_cublas import mm
torch.manual_seed(0)
for K, N in ((2048, 2048), (4096, 2048), (2048, 1024), (512, 2048), (2048, 512), (2048, 1536)):
    w = (torch.randn(K, N, device="cuda") * 0.03).half()
    diff = []
    for M in list(range(256, 8193, 256)) + [1000, 1808, 3001, 5000, 7777]:
        x = torch.randn(M, K, device="cuda").to(torch.bfloat16)
        ref = torch.mm(x.to(torch.float16), w)
        out = torch.empty(M, N, dtype=torch.float16, device="cuda")
        mm(x, w, out, 64, 128, 32, 3, 4)
        if not torch.equal(out.view(torch.int16), ref.view(torch.int16)):
            diff.append(M)
    print(f"K{K} N{N}: cuBLAS differs from sequential-k at M = {diff}", flush=True)
