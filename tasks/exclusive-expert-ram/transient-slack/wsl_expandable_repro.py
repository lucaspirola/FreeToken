"""Why does torch's expandable_segments fail on WSL while our VMM arena works?

torch 2.11 (c10/cuda/CUDACachingAllocator.cpp @70d99e9, ExpandableSegment::map, lines ~409-465)
calls cuMemCreate with prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC first (a non-OOM
error falls back), then CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, which is checked hard
(C10_CUDA_DRIVER_CHECK). It sets allocFlags.gpuDirectRDMACapable when the device reports it.
TORCH_CUDA_EXPANDABLE_SEGMENTS_IPC=0 leaves requestedHandleTypes = 0 (NONE). Our arena
(kernel/csrc/vmm_tensor.cpp) uses CU_MEM_HANDLE_TYPE_NONE.

Part 1 calls cuMemCreate directly with each handle type (NONE / POSIX_FD / FABRIC).
Part 2 runs torch in a fresh process per allocator setting: allocations with churn, a
CUDA graph capture + replay, and set_per_process_memory_fraction.

Small: < 1 GiB of VRAM, a few seconds. Usage: python wsl_expandable_repro.py
"""
import ctypes, json, os, subprocess, sys

CU_MEM_HANDLE = {"NONE": 0, "POSIX_FD": 1, "FABRIC": 8}


class CUmemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class AllocFlags(ctypes.Structure):
    _fields_ = [("compressionType", ctypes.c_ubyte), ("gpuDirectRDMACapable", ctypes.c_ubyte),
                ("usage", ctypes.c_ushort), ("reserved", ctypes.c_ubyte * 4)]


class CUmemAllocationProp(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("requestedHandleTypes", ctypes.c_int),
                ("location", CUmemLocation), ("win32HandleMetaData", ctypes.c_void_p),
                ("allocFlags", AllocFlags)]


def part1():
    cu = ctypes.CDLL("libcuda.so.1")
    def name(rc):
        s = ctypes.c_char_p(); cu.cuGetErrorName(rc, ctypes.byref(s))
        return f"{rc} {s.value.decode() if s.value else '?'}"
    assert cu.cuInit(0) == 0
    dev = ctypes.c_int(); cu.cuDeviceGet(ctypes.byref(dev), 0)
    ctx = ctypes.c_void_p(); assert cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev) == 0
    cu.cuCtxSetCurrent(ctx)
    rdma = ctypes.c_int()
    # CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED = 116
    cu.cuDeviceGetAttribute(ctypes.byref(rdma), 116, dev)
    ver = ctypes.c_int(); cu.cuDriverGetVersion(ctypes.byref(ver))
    out = {"driver": ver.value, "rdma_vmm_attr": rdma.value}
    for ht, v in CU_MEM_HANDLE.items():
        for rd in sorted({0, rdma.value}):
            prop = CUmemAllocationProp()
            prop.type = 1  # PINNED
            prop.requestedHandleTypes = v
            prop.location.type = 1; prop.location.id = dev.value  # DEVICE
            prop.allocFlags.gpuDirectRDMACapable = rd
            gran = ctypes.c_size_t()
            cu.cuMemGetAllocationGranularity(ctypes.byref(gran), ctypes.byref(prop), 0)
            h = ctypes.c_ulonglong()
            size = max(gran.value, 2 << 20)
            rc = cu.cuMemCreate(ctypes.byref(h), ctypes.c_size_t(size), ctypes.byref(prop), ctypes.c_ulonglong(0))
            r = {"cuMemCreate": name(rc)}
            if rc == 0:
                va = ctypes.c_ulonglong()
                rc2 = cu.cuMemAddressReserve(ctypes.byref(va), ctypes.c_size_t(size), ctypes.c_size_t(0), ctypes.c_ulonglong(0), ctypes.c_ulonglong(0))
                rc3 = cu.cuMemMap(va, ctypes.c_size_t(size), ctypes.c_size_t(0), h, ctypes.c_ulonglong(0)) if rc2 == 0 else -1
                class Acc(ctypes.Structure):
                    _fields_ = [("location", CUmemLocation), ("flags", ctypes.c_int)]
                acc = Acc(); acc.location.type = 1; acc.location.id = dev.value; acc.flags = 3
                rc4 = cu.cuMemSetAccess(va, ctypes.c_size_t(size), ctypes.byref(acc), ctypes.c_size_t(1)) if rc3 == 0 else -1
                r.update(reserve=name(rc2), map=name(rc3) if rc3 != -1 else "-", setAccess=name(rc4) if rc4 != -1 else "-")
                if rc3 == 0: cu.cuMemUnmap(va, ctypes.c_size_t(size))
                if rc2 == 0: cu.cuMemAddressFree(va, ctypes.c_size_t(size))
                cu.cuMemRelease(h)
            out[f"{ht}/rdma{rd}"] = r
    # torch reserves 1 1/8 of total VRAM of address space per expandable segment
    total = ctypes.c_size_t(); cu.cuDeviceTotalMem_v2(ctypes.byref(total), dev)
    seg = 20 << 20   # torch's large-pool segment_size_; max_handles_ = ceil(1.125 * VRAM / seg)
    va = ctypes.c_ulonglong(); n = -(-(total.value + total.value // 8) // seg) * seg
    rc = cu.cuMemAddressReserve(ctypes.byref(va), ctypes.c_size_t(n), ctypes.c_size_t(0), ctypes.c_ulonglong(0), ctypes.c_ulonglong(0))
    out["reserve_1.125x_vram"] = name(rc)
    if rc == 0: cu.cuMemAddressFree(va, ctypes.c_size_t(n))
    print(json.dumps(out, indent=1))


def part2_child():
    import torch
    if os.environ.get("REPRO_RUNTIME_API"):
        # the engine's path: the runtime API before the first CUDA allocation
        torch.cuda.memory._set_allocator_settings(os.environ["REPRO_RUNTIME_API"])
    r = {"conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"), "runtime": os.environ.get("REPRO_RUNTIME_API"), "ipc": os.environ.get("TORCH_CUDA_EXPANDABLE_SEGMENTS_IPC")}
    try:
        xs = [torch.empty(n << 20, dtype=torch.uint8, device="cuda") for n in (16, 128, 96, 64)]
        del xs
        big = torch.empty(252 << 20, dtype=torch.uint8, device="cuda"); del big
        r["peak_alloc_MiB"] = torch.cuda.max_memory_allocated() >> 20
        r["peak_reserved_MiB"] = torch.cuda.max_memory_reserved() >> 20
        x = torch.zeros(1 << 20, device="cuda")
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        g = torch.cuda.CUDAGraph()
        with torch.cuda.stream(s):
            y = x * 2
        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(g):
            y = x * 2 + 1
        g.replay(); torch.cuda.synchronize()
        r["graph_ok"] = bool((y == 1).all().item())
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.5)
        z = torch.empty(64 << 20, dtype=torch.uint8, device="cuda"); del z
        torch.cuda.set_per_process_memory_fraction(1.0)
        r["segments"] = torch.cuda.memory_stats().get("segment.all.current")
        r["ok"] = True
    except Exception as exc:
        r["ok"] = False; r["error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
    print(json.dumps(r))


if __name__ == "__main__":
    if sys.argv[1:] == ["child"]:
        part2_child(); sys.exit(0)
    print("== part 1: cuMemCreate by handle type"); sys.stdout.flush()
    try:
        part1()
    except Exception as exc:
        print("part 1 failed:", exc)
    print("== part 2: torch expandable segments, one process per setting"); sys.stdout.flush()
    for conf, ipc in (("expandable_segments:False", None), ("expandable_segments:True", None),
                      ("expandable_segments:True", "0"), ("expandable_segments:True", "1"),
                      ("runtime:expandable_segments:True", None), ("runtime:expandable_segments:True", "0")):
        env = dict(os.environ)
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None); env.pop("REPRO_RUNTIME_API", None)
        if conf.startswith("runtime:"):
            env["REPRO_RUNTIME_API"] = conf[len("runtime:"):]
        else:
            env["PYTORCH_CUDA_ALLOC_CONF"] = conf
        env.pop("TORCH_CUDA_EXPANDABLE_SEGMENTS_IPC", None)
        if ipc is not None:
            env["TORCH_CUDA_EXPANDABLE_SEGMENTS_IPC"] = ipc
        p = subprocess.run([sys.executable, __file__, "child"], env=env, capture_output=True, text=True, timeout=300)
        print(p.stdout.strip() or f"rc={p.returncode} stderr: {p.stderr.strip()[-600:]}")
        sys.stdout.flush()
