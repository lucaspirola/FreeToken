"""Host cost of cudaGraphLaunch: which graph property makes the decode graph's launch slow?

Box nsys (ft-g5, L5 e7e029f, 8K decode): cudaGraphLaunch takes ~176 us on the host for the
pool graph and ~63 us for the whole-model graph, while a straight-line graph of 892 tiny
kernels launches in ~3 us (bench_graph_launch.py). Each variant below adds ONE property to
that 800-kernel base graph and times replay() on the host (after a sync, median of 300):

  distinct   92 kernels that are 92 DISTINCT Triton functions (the decode graph has many)
  pinned     92 Triton kernels reading a 4 GiB cudaHostAlloc (pinned, mapped) tensor
  hostreg    92 Triton kernels reading an 8 GiB cudaHostRegister'd buffer (the expert banks
             are mlock'd + registered)
  hostreg_w  as hostreg, but the kernels WRITE the registered buffer
  bigparam   92 Triton kernels with 60 pointer args
  manyreg    92 Triton kernels, each on its own 64 MiB cudaHostRegister'd buffer (many
             distinct registrations referenced by one graph)
"""
import ctypes, statistics as S, sys, time

import torch
import triton
import triton.language as tl

N_EXTRA = 92


@triton.jit
def k_rd(dst, src, n, BLOCK: tl.constexpr):
    o = tl.arange(0, BLOCK)
    tl.store(dst + o, tl.load(src + o, mask=o < n), mask=o < n)


@triton.jit
def k_wr(dst, n, BLOCK: tl.constexpr):
    o = tl.arange(0, BLOCK)
    tl.store(dst + o, o, mask=o < n)


def make_distinct(i):
    src = f'''
import triton, triton.language as tl
@triton.jit
def kd{i}(dst, src, n, BLOCK: tl.constexpr):
    o = tl.arange(0, BLOCK)
    tl.store(dst + o, tl.load(src + o, mask=o < n) + {i}, mask=o < n)
'''
    path = f"/tmp/_kd{i}.py"
    open(path, "w").write(src)
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"_kd{i}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, f"kd{i}")


args60 = ", ".join(f"p{i}" for i in range(60))
src60 = f'''
import triton, triton.language as tl
@triton.jit
def k60({args60}, n, BLOCK: tl.constexpr):
    o = tl.arange(0, BLOCK)
    tl.store(p0 + o, tl.load(p1 + o, mask=o < n), mask=o < n)
'''
open("/tmp/_k60.py", "w").write(src60)
import importlib.util
_spec = importlib.util.spec_from_file_location("_k60", "/tmp/_k60.py")
_m60 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m60)
k60 = _m60.k60

cudart = ctypes.CDLL("libcudart.so" if len(sys.argv) < 2 else sys.argv[1])


def host_register(nbytes):
    t = torch.empty(nbytes, dtype=torch.uint8)
    t.fill_(1)
    rc = cudart.cudaHostRegister(ctypes.c_void_p(t.data_ptr()), ctypes.c_size_t(nbytes), 0)
    assert rc == 0, rc
    return t


x = torch.zeros(1024, device="cuda", dtype=torch.int32)
y = torch.zeros(1024, device="cuda", dtype=torch.int32)
ptrs = [torch.zeros(64, device="cuda", dtype=torch.int32) for _ in range(60)]
distinct = [make_distinct(i) for i in range(N_EXTRA)]
pinned = torch.ones((4 << 30) // 4, dtype=torch.int32).pin_memory()
hostreg = host_register(8 << 30)
hostreg_i = hostreg.view(torch.int32)
manyreg = [host_register(64 << 20).view(torch.int32) for _ in range(N_EXTRA)]


def base():
    for _ in range(800):
        x.add_(1)


def extra(kind):
    for i in range(N_EXTRA):
        if kind == "distinct":
            distinct[i][(1,)](y, x, 256, BLOCK=256)
        elif kind == "pinned":
            k_rd[(1,)](y, pinned[i * 4096:], 256, BLOCK=256)
        elif kind == "hostreg":
            k_rd[(1,)](y, hostreg_i[i * (1 << 20):], 256, BLOCK=256)
        elif kind == "hostreg_w":
            k_wr[(1,)](hostreg_i[i * (1 << 20):], 256, BLOCK=256)
        elif kind == "bigparam":
            k60[(1,)](*ptrs, 16, BLOCK=64)
        elif kind == "manyreg":
            k_rd[(1,)](y, manyreg[i], 256, BLOCK=256)
        elif kind == "same":
            k_rd[(1,)](y, x, 256, BLOCK=256)


for kind in ("none", "same", "distinct", "pinned", "hostreg", "hostreg_w", "bigparam", "manyreg", "none"):
    fn = (lambda: base()) if kind == "none" else (lambda k=kind: (base(), extra(k)))
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn(); fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    t = []
    for i in range(300):
        torch.cuda.synchronize()
        t0 = time.perf_counter(); g.replay(); t.append(time.perf_counter() - t0)
    torch.cuda.synchronize()
    print(f"{kind:10s} replay() host {S.median(t)*1e6:7.1f} us (p90 {sorted(t)[270]*1e6:.1f})", flush=True)
    del g
