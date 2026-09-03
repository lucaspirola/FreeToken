"""Probe: does flashinfer.mamba.selective_state_update JIT-build and run on sm_120?

Task T0 of tasks/nemotron35-plan.md (Phase 2).

Geometry mirrors Nemotron-3.5-Lightning-30B Mamba-2:
    nheads H=64, head_dim P=64, d_state N=128, n_groups G=8, conv 4.

Checks
  1. JIT build wall time (first call) + the exact nvcc arch flags used.
  2. Numerical parity vs transformers' pure-PyTorch
     ``nemotron_h.modeling_nemotron_h.mamba2_selective_state_update`` (output +
     updated state), for bs in {1, 8, 16}.
  3. ``pad_slot_id`` rows: cache slots referenced by a padded index, and slots
     not referenced at all, must be bit-identical after the call.
  4. Permuted / non-identity ``state_batch_indices`` gather+scatter correctness.
  5. Per-call latency (CUDA events, 20 iters after warmup) at each batch size.
  6. CUDA-graph capturability: capture with a preallocated ``out``, replay 3x,
     compare against eager.

Run:
    CUDA_VISIBLE_DEVICES=0 \
        uv run python benchmarks/probe_flashinfer_ssu_sm120.py
"""

from __future__ import annotations

import os
import sys
import time
import traceback

import torch

# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
H = 64          # nheads
P = 64          # head_dim  (== "dim" in the flashinfer contract)
N = 128         # d_state   (== "dstate")
G = 8           # n_groups
CACHE = 32      # state cache slots
BATCHES = (1, 8, 16)
PAD_SLOT_ID = -1

DEV = torch.device("cuda")
X_DT = torch.bfloat16     # x / B / C / z / out  -> input_dtype
W_DT = torch.bfloat16     # dt / D / dt_bias     -> weight_dtype
S_DT = torch.float32      # state                -> state_dtype
A_DT = torch.float32      # A                    -> matrixA_dtype

ITERS = 20
WARMUP = 5


def _hdr(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def _relerr(got: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    """Return (max_abs_diff, max_abs_diff / max|ref|)."""
    g = got.detach().float()
    r = ref.detach().float()
    diff = (g - r).abs()
    denom = r.abs().max().clamp_min(1e-12)
    return diff.max().item(), (diff.max() / denom).item()


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------
def make_static_params(seed: int = 0):
    """Per-layer parameters, in the exact strided layout the kernel demands."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    A_log = torch.rand(H, generator=g, dtype=torch.float32).to(DEV) * 2.0 + 0.1
    A_vec = -torch.exp(A_log)                                  # (H,) fp32
    D_vec = torch.randn(H, generator=g, dtype=torch.float32).to(DEV).to(W_DT)
    dtb_vec = (torch.rand(H, generator=g, dtype=torch.float32).to(DEV) - 4.0).to(W_DT)

    # A: (H, P, N) strides (1, 0, 0)
    A = A_vec[:, None, None].expand(H, P, N)
    # D: (H, P) strides (1, 0)
    D = D_vec[:, None].expand(H, P)
    # dt_bias: (H, P) strides (1, 0)
    dt_bias = dtb_vec[:, None].expand(H, P)

    assert A.stride() == (1, 0, 0), A.stride()
    assert D.stride() == (1, 0), D.stride()
    assert dt_bias.stride() == (1, 0), dt_bias.stride()
    return A, D, dt_bias


def make_batch(bs: int, seed: int = 1):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = (torch.randn(bs, H, P, generator=g, dtype=torch.float32) * 0.5).to(DEV).to(X_DT)
    # dt is stored per (batch, head) and broadcast over head_dim -> stride(2) == 0
    dt_raw = (torch.randn(bs, H, generator=g, dtype=torch.float32) * 0.5).to(DEV).to(W_DT)
    dt = dt_raw[:, :, None].expand(bs, H, P)
    B = (torch.randn(bs, G, N, generator=g, dtype=torch.float32) * 0.5).to(DEV).to(X_DT)
    C = (torch.randn(bs, G, N, generator=g, dtype=torch.float32) * 0.5).to(DEV).to(X_DT)
    assert dt.stride() == (H, 1, 0), dt.stride()
    return x, dt, B, C


def make_cache(seed: int = 2) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    st = (torch.randn(CACHE, H, P, N, generator=g, dtype=torch.float32) * 0.1).to(DEV)
    return st.contiguous()


# --------------------------------------------------------------------------
# Reference
# --------------------------------------------------------------------------
def reference(state_cache, idx, x, dt, A, B, C, D, dt_bias):
    """HF pure-PyTorch reference, driven through a cache gather/scatter.

    Returns (out_ref [bs,H,P], state_cache_ref [CACHE,H,P,N]).
    Rows whose index == PAD_SLOT_ID are excluded (kernel skips their state).
    """
    from transformers.models.nemotron_h.modeling_nemotron_h import (
        mamba2_selective_state_update,
    )

    idx_cpu = idx.tolist()
    live = [i for i, s in enumerate(idx_cpu) if s != PAD_SLOT_ID]
    slots = [idx_cpu[i] for i in live]
    out_ref = torch.zeros_like(x)
    cache_ref = state_cache.clone()
    if not live:
        return out_ref, cache_ref, live

    sel = torch.tensor(live, device=DEV, dtype=torch.long)
    sub_state = cache_ref[torch.tensor(slots, device=DEV, dtype=torch.long)].clone()
    o = mamba2_selective_state_update(
        sub_state,
        x.index_select(0, sel).contiguous(),
        dt.index_select(0, sel).contiguous(),
        A,
        B.index_select(0, sel).contiguous(),
        C.index_select(0, sel).contiguous(),
        D,
        dt_bias=dt_bias,
        dt_softplus=True,
        z=None,
    )
    out_ref[sel] = o.to(out_ref.dtype)
    # sub_state was mutated in place by the reference
    for k, slot in enumerate(slots):
        cache_ref[slot] = sub_state[k]
    return out_ref, cache_ref, live



def gold(state_cache, idx, x, dt, A, B, C, D, dt_bias):
    """Float64 gold recurrence (same math, no bf16 rounding inside)."""
    import torch.nn.functional as F

    idx_cpu = idx.tolist()
    live = [i for i, s in enumerate(idx_cpu) if s != PAD_SLOT_ID]
    slots = [idx_cpu[i] for i in live]
    out_g = torch.zeros(x.shape, device=DEV, dtype=torch.float64)
    cache_g = state_cache.double().clone()
    if not live:
        return out_g, cache_g, live
    sel = torch.tensor(live, device=DEV, dtype=torch.long)
    sl = torch.tensor(slots, device=DEV, dtype=torch.long)

    xs = x.index_select(0, sel).double()                     # (b,H,P)
    dts = dt.index_select(0, sel).double()                   # (b,H,P)
    Bs = B.index_select(0, sel).double()                     # (b,G,N)
    Cs = C.index_select(0, sel).double()
    st = cache_g.index_select(0, sl)                         # (b,H,P,N)

    dtv = F.softplus(dts + dt_bias.double())                 # (b,H,P)
    dA = torch.exp(dtv[..., None] * A.double())              # (b,H,P,N)
    rep = H // G
    Be = Bs.repeat_interleave(rep, dim=1)[:, :, None, :]     # (b,H,1,N)
    Ce = Cs.repeat_interleave(rep, dim=1)                    # (b,H,N)
    dBx = dtv[..., None] * Be * xs[..., None]
    new = st * dA + dBx
    o = (new * Ce[:, :, None, :]).sum(-1) + xs * D.double()
    out_g[sel] = o
    cache_g[sl] = new
    return out_g, cache_g, live


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> int:
    _hdr("environment")
    print(f"torch            {torch.__version__}")
    print(f"torch cuda       {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("no CUDA device; aborting")
        return 2
    cap = torch.cuda.get_device_capability(0)
    print(f"device           {torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]}")
    import flashinfer

    print(f"flashinfer       {flashinfer.__version__}")

    from flashinfer.compilation_context import CompilationContext
    from flashinfer.jit import env as jit_env
    from flashinfer.jit.mamba.selective_state_update import (
        get_selective_state_update_uri,
    )

    try:
        nvcc_flags = CompilationContext().get_nvcc_flags_list(
            supported_major_versions=[10, 11, 12]
        )
        print(f"nvcc arch flags  {nvcc_flags}")
    except Exception as exc:  # pragma: no cover
        print(f"nvcc arch flags  FAILED: {exc}")
        return 3
    print(f"CUDA_HOME        {os.environ.get('CUDA_HOME', '(unset)')}")
    print(f"jit cache dir    {jit_env.FLASHINFER_JIT_DIR}")

    uri = (
        get_selective_state_update_uri(
            S_DT, X_DT, W_DT, A_DT, torch.int32, None, P, N, 1,
            torch.int32, torch.int64, 0,
        )
        + "_sm100"
    )
    cached = (jit_env.FLASHINFER_JIT_DIR / uri).exists()
    print(f"module uri       {uri}")
    print(f"already cached   {cached}  (build time below is "
          f"{'a cache hit' if cached else 'a COLD build'})")

    from flashinfer.mamba import selective_state_update

    A, D, dt_bias = make_static_params()

    # ---------------------------------------------------------------- JIT
    _hdr("1. JIT build (first call)")
    x, dt, B, C = make_batch(1)
    state = make_cache()
    idx = torch.tensor([0], device=DEV, dtype=torch.int32)
    t0 = time.perf_counter()
    try:
        selective_state_update(
            state, x, dt, A, B, C, D, z=None, dt_bias=dt_bias,
            dt_softplus=True, state_batch_indices=idx, pad_slot_id=PAD_SLOT_ID,
        )
        torch.cuda.synchronize()
    except Exception:
        print("JIT / launch FAILED\n")
        traceback.print_exc()
        print(f"\nnvcc flags used: {nvcc_flags}")
        print(f"nvcc binary: {os.environ.get('CUDA_HOME', '/usr/local/cuda')}/bin/nvcc")
        return 1
    build_s = time.perf_counter() - t0
    print(f"first call wall time: {build_s:.2f} s")

    # -------------------------------------------------------- correctness
    _hdr("2. numerical parity vs transformers reference (all-live indices)")
    ok = True
    for bs in BATCHES:
        x, dt, B, C = make_batch(bs, seed=10 + bs)
        base = make_cache(seed=20 + bs)
        idx = torch.arange(bs, device=DEV, dtype=torch.int32)
        st = base.clone()
        out = selective_state_update(
            st, x, dt, A, B, C, D, z=None, dt_bias=dt_bias, dt_softplus=True,
            state_batch_indices=idx, pad_slot_id=PAD_SLOT_ID,
        )
        torch.cuda.synchronize()
        out_ref, st_ref, _ = reference(base, idx, x, dt, A, B, C, D, dt_bias)
        out_g, st_g64, _ = gold(base, idx, x, dt, A, B, C, D, dt_bias)
        oa, orl = _relerr(out, out_ref)
        sa, srl = _relerr(st, st_ref)
        _, orl_g = _relerr(out, out_g)
        _, srl_g = _relerr(st, st_g64)
        _, orl_r = _relerr(out_ref, out_g)
        _, srl_r = _relerr(st_ref, st_g64)
        # Pass if flashinfer is no worse than the HF reference against fp64 gold
        # (both consume the same bf16 inputs, so bf16 eps ~ 7.8e-3 is the floor).
        good = orl_g <= max(2.0 * orl_r, 2e-2) and srl_g <= max(2.0 * srl_r, 2e-2)
        ok &= good
        print(f"bs={bs:<3} vs HF ref   out: max_abs={oa:.3e} max_rel={orl:.3e} | "
              f"state: max_abs={sa:.3e} max_rel={srl:.3e}")
        print(f"      vs fp64 gold  flashinfer out={orl_g:.3e} state={srl_g:.3e} | "
              f"HF ref out={orl_r:.3e} state={srl_r:.3e}   {'OK' if good else 'MISMATCH'}")

    # ------------------------------------------------ permuted + pad slots
    _hdr("3. permuted state_batch_indices + pad_slot_id rows")
    for bs in (8, 16):
        x, dt, B, C = make_batch(bs, seed=30 + bs)
        base = make_cache(seed=40 + bs)
        gp = torch.Generator(device="cpu").manual_seed(7 + bs)
        perm = torch.randperm(CACHE, generator=gp)[:bs].to(torch.int32)
        pad_rows = [1, bs - 2] if bs >= 4 else [0]
        for r in pad_rows:
            perm[r] = PAD_SLOT_ID
        idx = perm.to(DEV)
        st = base.clone()
        out = selective_state_update(
            st, x, dt, A, B, C, D, z=None, dt_bias=dt_bias, dt_softplus=True,
            state_batch_indices=idx, pad_slot_id=PAD_SLOT_ID,
        )
        torch.cuda.synchronize()
        out_ref, st_ref, live = reference(base, idx, x, dt, A, B, C, D, dt_bias)

        out_g64, st_g64, _ = gold(base, idx, x, dt, A, B, C, D, dt_bias)
        sel = torch.tensor(live, device=DEV, dtype=torch.long)
        oa, orl = _relerr(out.index_select(0, sel), out_ref.index_select(0, sel))
        sa, srl = _relerr(st, st_ref)
        _, orl_g = _relerr(out.index_select(0, sel), out_g64.index_select(0, sel))
        _, srl_g = _relerr(st, st_g64)

        touched = {int(v) for v in idx.tolist() if v != PAD_SLOT_ID}
        untouched = sorted(set(range(CACHE)) - touched)
        ut = torch.tensor(untouched, device=DEV, dtype=torch.long)
        untouched_bitexact = bool(
            torch.equal(st.index_select(0, ut), base.index_select(0, ut))
        )
        good = orl_g < 2e-2 and srl_g < 2e-2 and untouched_bitexact
        ok &= good
        print(f"bs={bs:<3} slots={sorted(touched)}")
        print(f"      pad rows (idx=={PAD_SLOT_ID}) at batch positions "
              f"{[r for r in pad_rows]}")
        print(f"      live out vs HF ref: max_abs={oa:.3e} max_rel={orl:.3e} | "
              f"state: max_abs={sa:.3e} max_rel={srl:.3e}")
        print(f"      vs fp64 gold: out={orl_g:.3e} state={srl_g:.3e}")
        print(f"      {len(untouched)} unreferenced cache slots bit-identical: "
              f"{untouched_bitexact}   {'OK' if good else 'MISMATCH'}")

    # ------------------------------------------------------------ latency
    _hdr("4. per-call latency (CUDA events)")
    for bs in BATCHES:
        x, dt, B, C = make_batch(bs, seed=50 + bs)
        st = make_cache(seed=60 + bs)
        idx = torch.arange(bs, device=DEV, dtype=torch.int32)
        out = torch.empty(bs, H, P, device=DEV, dtype=X_DT)
        for _ in range(WARMUP):
            selective_state_update(
                st, x, dt, A, B, C, D, z=None, dt_bias=dt_bias, dt_softplus=True,
                state_batch_indices=idx, pad_slot_id=PAD_SLOT_ID, out=out,
            )
        torch.cuda.synchronize()
        ev0 = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
        ev1 = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
        for i in range(ITERS):
            ev0[i].record()
            selective_state_update(
                st, x, dt, A, B, C, D, z=None, dt_bias=dt_bias, dt_softplus=True,
                state_batch_indices=idx, pad_slot_id=PAD_SLOT_ID, out=out,
            )
            ev1[i].record()
        torch.cuda.synchronize()
        us = sorted(ev0[i].elapsed_time(ev1[i]) * 1e3 for i in range(ITERS))
        mean = sum(us) / len(us)
        bytes_rw = 2 * bs * H * P * N * 4  # state read + write, fp32
        gbs = bytes_rw / (mean * 1e-6) / 1e9
        print(f"bs={bs:<3} mean={mean:8.2f} us  min={us[0]:8.2f}  "
              f"p50={us[len(us) // 2]:8.2f}  max={us[-1]:8.2f}   "
              f"state traffic {bytes_rw / 2**20:.1f} MiB  -> {gbs:.0f} GB/s")

    # --------------------------------------------------------- CUDA graph
    _hdr("5. CUDA graph capture / replay")
    bs = 16
    x, dt, B, C = make_batch(bs, seed=99)
    base = make_cache(seed=98)
    idx = torch.arange(bs, device=DEV, dtype=torch.int32)

    st_eager = base.clone()
    out_eager = torch.empty(bs, H, P, device=DEV, dtype=X_DT)
    selective_state_update(
        st_eager, x, dt, A, B, C, D, z=None, dt_bias=dt_bias, dt_softplus=True,
        state_batch_indices=idx, pad_slot_id=PAD_SLOT_ID, out=out_eager,
    )
    torch.cuda.synchronize()

    st_g = base.clone()
    out_g = torch.empty(bs, H, P, device=DEV, dtype=X_DT)

    def _call():
        selective_state_update(
            st_g, x, dt, A, B, C, D, z=None, dt_bias=dt_bias, dt_softplus=True,
            state_batch_indices=idx, pad_slot_id=PAD_SLOT_ID, out=out_g,
        )

    graph_ok = True
    try:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                _call()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _call()
        torch.cuda.synchronize()
        print("capture: OK")

        for r in range(3):
            st_g.copy_(base)
            out_g.zero_()
            graph.replay()
            torch.cuda.synchronize()
            oa, orl = _relerr(out_g, out_eager)
            sa, srl = _relerr(st_g, st_eager)
            bit_out = bool(torch.equal(out_g, out_eager))
            bit_state = bool(torch.equal(st_g, st_eager))
            good = bit_out and bit_state
            graph_ok &= good
            print(f"replay {r}: out bitexact={bit_out} state bitexact={bit_state} "
                  f"| out max_abs={oa:.3e} state max_abs={sa:.3e}  "
                  f"{'OK' if good else 'MISMATCH'}")

        # graph latency
        for _ in range(WARMUP):
            graph.replay()
        torch.cuda.synchronize()
        e0 = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
        e1 = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
        for i in range(ITERS):
            e0[i].record()
            graph.replay()
            e1[i].record()
        torch.cuda.synchronize()
        us = sorted(e0[i].elapsed_time(e1[i]) * 1e3 for i in range(ITERS))
        print(f"graph replay bs={bs}: mean={sum(us) / len(us):.2f} us "
              f"min={us[0]:.2f} max={us[-1]:.2f}")
    except Exception:
        graph_ok = False
        print("CUDA graph capture FAILED")
        traceback.print_exc()

    ok &= graph_ok

    _hdr("verdict")
    print(f"JIT build            : OK ({build_s:.1f} s "
          f"{'cold' if not cached else 'cached'})")
    print(f"numerics             : {'OK' if ok else 'see failures above'}")
    print(f"cuda graph capturable: {graph_ok}")
    print(f"OVERALL              : {'USABLE' if ok else 'NOT USABLE'} "
          f"for the Nemotron decode path")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
