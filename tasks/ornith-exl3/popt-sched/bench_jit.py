#!/usr/bin/env python3
"""Wall cost of the shared-expert epilogue on a prompt length the process has not seen, old
(n_elements constexpr: a compile per length) vs new (runtime arg). Fresh TRITON_CACHE_DIR, so a
novel length compiles as it does on the owner's file reads (every extend has a new length)."""
import os, sys, tempfile, time, json
os.environ["TRITON_CACHE_DIR"] = tempfile.mkdtemp(prefix="jitbench-", dir=os.path.dirname(os.path.abspath(__file__)))
import torch, triton, triton.language as tl
from freetoken.kernel.triton.shared_expert import fused_shared_expert_add_

@triton.jit
def _old(routed_ptr, shared_ptr, gate_ptr, hidden: tl.constexpr, n_elements: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    token = offsets // hidden
    routed = tl.load(routed_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    shared = tl.load(shared_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + token, mask=mask, other=0.0).to(tl.float32)
    tl.store(routed_ptr + offsets, routed + shared * tl.sigmoid(gate), mask=mask)

def old(r, s, g):
    n = r.numel()
    _old[(triton.cdiv(n, 256),)](r, s, g, hidden=r.shape[1], n_elements=n, BLOCK=256, num_warps=4)

H = 2048
res = {}
for name, fn in (("old", old), ("new", fused_shared_expert_add_)):
    ts = []
    for tokens in (8192, 1017, 14493, 4481, 12345):
        r = torch.randn(tokens, H, device="cuda", dtype=torch.bfloat16); s = torch.randn_like(r)
        g = torch.randn(tokens, 1, device="cuda", dtype=torch.bfloat16)
        torch.cuda.synchronize(); t0 = time.perf_counter(); fn(r, s, g); torch.cuda.synchronize()
        ts.append(round((time.perf_counter() - t0) * 1e3, 2))
    res[name] = ts
    print(name, "first call per new length, ms:", ts, flush=True)
print(json.dumps(res))
import shutil; shutil.rmtree(os.environ["TRITON_CACHE_DIR"], ignore_errors=True)
