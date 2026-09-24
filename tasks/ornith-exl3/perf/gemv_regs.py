"""Registers and spills of the decode GEMV (`_exl3_gemv_kernel`) per variant, from ptxas -v on sm_120.
CPU only: compiles through triton for the target, no GPU launch (Nsight Compute's counters are
blocked on the Vast box, ERR_NVGPUCTRPERM).

    CUDA_VISIBLE_DEVICES= PYTHONPATH=python python tasks/ornith-exl3/perf/gemv_regs.py
"""
import inspect
import os
import subprocess
import tempfile

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from freetoken.kernel.triton import exl3 as K

PTXAS = os.environ.get("PTXAS", "/usr/local/cuda-13.0/bin/ptxas")
kern = K._exl3_gemv_kernel
names = list(inspect.signature(kern.fn).parameters)
CONST = {"BITS", "CB", "HAS_EXPERT", "REDUCE", "PRE_ROT", "SRC_DIV", "KB", "K_BLOCKS", "BANDS"} & set(names)
FP16 = {"xh_ptr", "svh_ptr", "had_ptr", "x_ptr", "suh_ptr", "rot_ptr"}
FP32 = {"out_ptr", "final_ptr"}


def variant(maxnreg=None, **consts):
    sig = {n: "constexpr" if n in CONST else ("*fp16" if n in FP16 else "*fp32" if n in FP32 else "*i32")
           if n.endswith("_ptr") or n in CONST else "i32" for n in names}
    c = dict(BITS=5, CB=1, HAS_EXPERT=False, REDUCE=True, PRE_ROT=False, SRC_DIV=1, KB=1, K_BLOCKS=16, BANDS=1)
    c.update(consts)
    c = {k: v for k, v in c.items() if k in CONST}
    opts = dict(num_warps=1, **({"maxnreg": maxnreg} if maxnreg else {}))
    ck = triton.compile(ASTSource(fn=kern, signature=sig, constexprs=c), target=GPUTarget("cuda", 120, 32), options=opts)
    d = tempfile.mkdtemp()
    open(os.path.join(d, "k.ptx"), "w").write(ck.asm["ptx"])
    r = subprocess.run([PTXAS, "-arch=sm_120a", "-v", os.path.join(d, "k.ptx"), "-o", os.path.join(d, "k.cubin")],
                       capture_output=True, text=True)
    info = " | ".join(line.split(":", 1)[-1].strip() for line in r.stderr.splitlines() if "registers" in line or "spill" in line)
    print(f"maxnreg {maxnreg} {consts}: {info}", flush=True)


for cap in (None, getattr(K, "GEMV_MAXNREG", None)):
    variant(cap)
    variant(cap, REDUCE=False)
    variant(cap, PRE_ROT=True, KB=2, HAS_EXPERT=True)
