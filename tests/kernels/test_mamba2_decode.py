"""Parity tests for the Mamba-2 decode step (task 2A3).

`mamba2_decode` is one token of the SSD recurrence with the recurrent pool
updated in place, behind two backends (`FREETOKEN_MAMBA2_DECODE`): flashinfer's
`selective_state_update` CUDA kernel and the vendored Triton port. Both are
checked against the same oracle the prefill kernels use --
:func:`tests.kernels.mamba2_gold.gold_ssd`, the recurrence evaluated in float64
-- run with ``T = 1``.

What is covered
  * bs in {1, 7, 16}, both backends, a random duplicate-free slot permutation:
    live outputs and the written state vs the fp64 gold, pad rows
    (``indices == -1``) leaving their slot untouched, and every unreferenced
    slot bit-identical afterwards.
  * `mamba2_prefill(T)` then `mamba2_decode(1)` == `mamba2_prefill(T + 1)` --
    the plan's cross-check that the two kernels share one state convention.
  * the gated RMSNorm wrapper vs the model's `_MambaGatedRMSNorm` and vs HF's
    `Zamba2RMSNormGated` (which is what nemotron_h instantiates).
  * CUDA-graph capture + 3 replays == eager, for both backends.
  * flashinfer and Triton agreeing with each other.

Error floors (measured on an RTX 5080, Nemotron-3.5 Lightning geometry): the
inputs are bf16, so both backends sit at the bf16 rounding floor against the
fp64 gold. flashinfer additionally takes ``D``/``dt_bias`` in ``dt``'s dtype
(bf16) while the Triton path keeps them fp32, which is why the cross-backend
tolerance is looser than either backend's own agreement with the gold.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.kernel.triton.mamba2 import (
    build_mamba2_metadata,
    mamba2_decode,
    mamba2_gated_rmsnorm,
    mamba2_prefill,
    warm_mamba2_decode,
)
from freetoken.kernel.triton.mamba2 import selective_state_update as _ssu
from freetoken.kernel.triton.mamba2.selective_state_update import _flashinfer_ssu

from .mamba2_gold import gold_ssd

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

# Nemotron-3.5 Lightning geometry.
H, P, N, G, CHUNK = 64, 64, 128, 8, 128
SLOTS = 24
PAD = -1

# Measured on an RTX 5080 at this geometry, RMS-relative against the fp64 gold
# (`_err_ratio`), with the elementwise max|diff| that goes with it. `out` is
# stored in bf16, so ~1.6e-3 is its floor for any backend.
#
#   backend      out rms-rel   out max|d|   state rms-rel   state max|d|
#   triton       1.7e-3        1.5e-2       2.6e-7          3.6e-7
#   flashinfer   2.2e-3        2.7e-2       1.0e-3          2.1e-3
#
# (|out| <= 5.4, rms 0.70; |state| <= 0.52, rms 0.093.)
#
# The state gap is not a kernel bug: flashinfer's contract takes `D` and
# `dt_bias` in `dt`'s dtype, i.e. bf16, while the Triton path keeps them fp32.
# `dt_bias` enters as `exp(A * softplus(dt + dt_bias))`, so its 2e-3 relative
# representation error moves the decay by ~|A * dt| x 2e-3. Every bar below is
# ~2x the measured value, so a real regression still trips it.
GOLD_BARS = {
    #             out rms   out (rtol, atol)   state rms   state (rtol, atol)
    "triton": (3e-3, (3e-2, 2e-2), 1e-6, (1e-4, 1e-5)),
    "flashinfer": (5e-3, (5e-2, 4e-2), 2e-3, (5e-2, 4e-3)),
}
# flashinfer vs Triton: the sum of the two rows above.
XB_OUT_RTOL, XB_OUT_ATOL = 6e-2, 5e-2
XB_STATE_RTOL, XB_STATE_ATOL = 6e-2, 5e-3


def _err_ratio(ref: torch.Tensor, got: torch.Tensor) -> float:
    """RMS(got - ref) / RMS(ref), in fp64."""
    r, g = ref.double(), got.double()
    return ((g - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt()).item()


def _backends() -> list[str]:
    if not torch.cuda.is_available():
        return ["triton"]
    return ["triton", "flashinfer"] if _flashinfer_ssu() is not None else ["triton"]


BACKENDS = _backends()


@pytest.fixture(params=BACKENDS)
def backend(request, monkeypatch):
    monkeypatch.setenv("FREETOKEN_MAMBA2_DECODE", request.param)
    return request.param


def _params(device="cuda", seed=0):
    """A, D, dt_bias at the scales the checkpoint uses: A in [-16, -1],
    softplus(dt + dt_bias) in ~[1e-3, 1e-1]."""
    g = torch.Generator(device=device).manual_seed(seed)
    f32 = {"device": device, "dtype": torch.float32, "generator": g}
    A = -torch.exp(torch.rand(H, **f32) * 2.8)
    D = torch.randn(H, **f32)
    dt_bias = torch.rand(H, **f32) * 2.0 - 5.0
    return A, D, dt_bias


def _step_inputs(bs, device="cuda", seed=1):
    g = torch.Generator(device=device).manual_seed(seed)
    bf = {"device": device, "dtype": torch.bfloat16, "generator": g}
    x = torch.randn(bs, H, P, **bf) * 0.5
    dt = torch.randn(bs, H, **bf) * 0.5
    B = torch.randn(bs, G, N, **bf) * 0.5
    C = torch.randn(bs, G, N, **bf) * 0.5
    return x, dt, B, C


def _pool(device="cuda", seed=2):
    g = torch.Generator(device=device).manual_seed(seed)
    return (
        torch.randn(SLOTS, H, P, N, device=device, dtype=torch.float32, generator=g)
        * 0.1
    ).contiguous()


def _slots(bs, pad_rows=(), seed=3):
    """A duplicate-free random permutation of pool slots, with `pad_rows` set to
    the pad id."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(SLOTS, generator=g)[:bs].to(torch.int32)
    for r in pad_rows:
        idx[r] = PAD
    return idx


def _gold_step(pool, idx, x, dt, B, C, A, D, dt_bias):
    """Per-row fp64 single-step recurrence. Returns (out[bs,H,P] f64,
    pool_after[SLOTS,H,P,N] f64, live row list)."""
    live = [i for i, s in enumerate(idx.tolist()) if s != PAD]
    out = torch.zeros(x.shape, device=x.device, dtype=torch.float64)
    after = pool.double().clone()
    for r in live:
        slot = int(idx[r])
        y, h = gold_ssd(
            x[r : r + 1],
            dt[r : r + 1],
            B[r : r + 1],
            C[r : r + 1],
            A,
            D=D,
            dt_bias=dt_bias,
            initial=pool[slot],
            dt_softplus=True,
        )
        out[r] = y[0]
        after[slot] = h
    return out, after, live


# ---------------------------------------------------------------- gold parity
@cuda_only
@pytest.mark.parametrize("bs", [1, 7, 16])
def test_decode_matches_fp64_gold(backend, bs):
    x, dt, B, C = _step_inputs(bs, seed=10 + bs)
    A, D, dt_bias = _params(seed=20 + bs)
    base = _pool(seed=30 + bs)
    pad_rows = () if bs == 1 else (1, bs - 2)
    idx = _slots(bs, pad_rows, seed=40 + bs).cuda()

    pool = base.clone()
    out = mamba2_decode(
        x, dt, B, C, A=A, D=D, dt_bias=dt_bias, state_source=pool, indices=idx
    )
    assert out.shape == x.shape and out.dtype == x.dtype

    gold_out, gold_pool, live = _gold_step(base, idx.cpu(), x, dt, B, C, A, D, dt_bias)
    assert len(live) == bs - len(pad_rows)

    out_rms, out_tol, state_rms, state_tol = GOLD_BARS[backend]

    sel = torch.tensor(live, device="cuda")
    got_out = out.index_select(0, sel).float()
    want_out = gold_out.index_select(0, sel).float()
    assert _err_ratio(want_out, got_out) < out_rms
    torch.testing.assert_close(got_out, want_out, rtol=out_tol[0], atol=out_tol[1])

    touched = torch.tensor(
        sorted({int(idx[r]) for r in live}), device="cuda", dtype=torch.long
    )
    got_state = pool.index_select(0, touched)
    want_state = gold_pool.index_select(0, touched).float()
    assert _err_ratio(want_state, got_state) < state_rms
    torch.testing.assert_close(
        got_state, want_state, rtol=state_tol[0], atol=state_tol[1]
    )

    # Every slot the batch did not name -- including the ones behind pad rows --
    # must come out bit-identical.
    untouched = sorted(set(range(SLOTS)) - {int(v) for v in touched.tolist()})
    ut = torch.tensor(untouched, device="cuda", dtype=torch.long)
    assert torch.equal(pool.index_select(0, ut), base.index_select(0, ut))


@cuda_only
def test_pad_rows_leave_their_slot_untouched(backend):
    """A row whose index is the pad id must not read or write the pool, even
    when a *different* live row happens to point at a neighbouring slot."""
    bs = 8
    x, dt, B, C = _step_inputs(bs, seed=77)
    A, D, dt_bias = _params(seed=78)
    base = _pool(seed=79)
    idx = torch.full((bs,), PAD, dtype=torch.int32, device="cuda")
    pool = base.clone()
    mamba2_decode(
        x, dt, B, C, A=A, D=D, dt_bias=dt_bias, state_source=pool, indices=idx
    )
    assert torch.equal(pool, base)


# ------------------------------------------------- prefill / decode agreement
@cuda_only
def test_prefill_then_decode_equals_prefill_of_one_more_token(backend):
    """The plan's cross-check: the two kernels must share one state convention.
    T straddles the 128-token chunk boundary so the decode step lands on a fresh
    chunk in the T+1 prefill."""
    total = 129
    g = torch.Generator(device="cuda").manual_seed(5)
    bf = {"device": "cuda", "dtype": torch.bfloat16, "generator": g}
    x = torch.randn(total, H, P, **bf) * 0.5
    dt = torch.randn(total, H, **bf) * 0.5
    B = torch.randn(total, G, N, **bf) * 0.5
    C = torch.randn(total, G, N, **bf) * 0.5
    A, D, dt_bias = _params(seed=6)
    idx1 = torch.zeros(1, dtype=torch.int32, device="cuda")
    common = dict(A=A, D=D, dt_bias=dt_bias, indices=idx1)

    # (a) prefill T-1 tokens, then one decode step on token T-1.
    pool_a = torch.zeros(1, H, P, N, device="cuda", dtype=torch.float32)
    meta = build_mamba2_metadata([0, total - 1], CHUNK, device="cuda")
    cu = torch.tensor([0, total - 1], dtype=torch.int32, device="cuda")
    out_a, _ = mamba2_prefill(
        x[:-1],
        dt[:-1],
        B[:-1],
        C[:-1],
        meta=meta,
        cu_seqlens=cu,
        state_source=pool_a,
        **common,
    )
    last = mamba2_decode(x[-1:], dt[-1:], B[-1:], C[-1:], state_source=pool_a, **common)

    # (b) one prefill over all T tokens.
    pool_b = torch.zeros(1, H, P, N, device="cuda", dtype=torch.float32)
    meta_b = build_mamba2_metadata([0, total], CHUNK, device="cuda")
    cu_b = torch.tensor([0, total], dtype=torch.int32, device="cuda")
    out_b, _ = mamba2_prefill(
        x, dt, B, C, meta=meta_b, cu_seqlens=cu_b, state_source=pool_b, **common
    )

    # The two paths differ by the chunked SSD factorisation's own bf16 rounding
    # (`tests/kernels/test_mamba2_ssd.py` measures it at 1.8e-3 RMS-relative on
    # the output and 1.7e-3 on the state), which is the bar here.
    assert _err_ratio(out_b[:-1].float(), out_a.float()) < 5e-3
    assert _err_ratio(out_b[-1].float(), last[0].float()) < 5e-3
    assert _err_ratio(pool_b, pool_a) < 5e-3
    torch.testing.assert_close(pool_a, pool_b, rtol=5e-2, atol=5e-3)


# ------------------------------------------------------------- CUDA graph
@cuda_only
def test_cuda_graph_replay_equals_eager(backend):
    bs = 16
    x, dt, B, C = _step_inputs(bs, seed=90)
    A, D, dt_bias = _params(seed=91)
    base = _pool(seed=92)
    idx = _slots(bs, (0, 5), seed=93).cuda()

    assert warm_mamba2_decode(base, bs, ngroups=G) == backend

    pool_e = base.clone()
    out_e = torch.empty(bs, H, P, device="cuda", dtype=torch.bfloat16)
    mamba2_decode(
        x,
        dt,
        B,
        C,
        A=A,
        D=D,
        dt_bias=dt_bias,
        state_source=pool_e,
        indices=idx,
        out=out_e,
    )
    torch.cuda.synchronize()

    pool_g = base.clone()
    out_g = torch.empty_like(out_e)

    def _call():
        mamba2_decode(
            x,
            dt,
            B,
            C,
            A=A,
            D=D,
            dt_bias=dt_bias,
            state_source=pool_g,
            indices=idx,
            out=out_g,
        )

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

    for _ in range(3):
        pool_g.copy_(base)
        out_g.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out_g, out_e)
        assert torch.equal(pool_g, pool_e)


@cuda_only
def test_steady_state_call_allocates_nothing(backend):
    """With `out` given, a decode step must not allocate: that is what makes the
    expanded A / D / dt_bias views worth caching and graph capture safe."""
    bs = 16
    x, dt, B, C = _step_inputs(bs, seed=120)
    A, D, dt_bias = _params(seed=121)
    pool = _pool(seed=122)
    idx = _slots(bs, seed=123).cuda()
    out = torch.empty(bs, H, P, device="cuda", dtype=torch.bfloat16)

    def _call():
        mamba2_decode(
            x,
            dt,
            B,
            C,
            A=A,
            D=D,
            dt_bias=dt_bias,
            state_source=pool,
            indices=idx,
            out=out,
        )

    for _ in range(3):
        _call()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    for _ in range(5):
        _call()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before


# --------------------------------------------------------- backend agreement
@cuda_only
@pytest.mark.skipif(len(BACKENDS) < 2, reason="flashinfer not installed")
@pytest.mark.parametrize("bs", [1, 16])
def test_backends_agree(monkeypatch, bs):
    x, dt, B, C = _step_inputs(bs, seed=110 + bs)
    A, D, dt_bias = _params(seed=111 + bs)
    base = _pool(seed=112 + bs)
    pad_rows = () if bs == 1 else (2,)
    idx = _slots(bs, pad_rows, seed=113 + bs).cuda()

    results = {}
    for name in ("flashinfer", "triton"):
        monkeypatch.setenv("FREETOKEN_MAMBA2_DECODE", name)
        pool = base.clone()
        out = mamba2_decode(
            x, dt, B, C, A=A, D=D, dt_bias=dt_bias, state_source=pool, indices=idx
        )
        results[name] = (out, pool)

    live = torch.tensor(
        [i for i, s in enumerate(idx.tolist()) if s != PAD], device="cuda"
    )
    fo, fp = results["flashinfer"]
    to, tp = results["triton"]
    torch.testing.assert_close(
        fo.index_select(0, live).float(),
        to.index_select(0, live).float(),
        rtol=XB_OUT_RTOL,
        atol=XB_OUT_ATOL,
    )
    torch.testing.assert_close(fp, tp, rtol=XB_STATE_RTOL, atol=XB_STATE_ATOL)


def test_env_var_rejects_unknown_backend(monkeypatch):
    monkeypatch.setenv("FREETOKEN_MAMBA2_DECODE", "cublas")
    with pytest.raises(ValueError, match="FREETOKEN_MAMBA2_DECODE"):
        _ssu.resolve_decode_backend()


def test_auto_prefers_flashinfer_when_it_is_available(monkeypatch):
    monkeypatch.setenv("FREETOKEN_MAMBA2_DECODE", "auto")
    monkeypatch.setattr(_ssu, "_flashinfer_broken", None)
    assert _ssu.resolve_decode_backend() == (
        "flashinfer" if len(BACKENDS) > 1 else "triton"
    )


@cuda_only
@pytest.mark.skipif(len(BACKENDS) < 2, reason="flashinfer not installed")
def test_auto_demotes_to_triton_when_flashinfer_fails(monkeypatch):
    """A flashinfer JIT build failure must demote the process once and then run
    the Triton kernel -- not propagate, and not retry on every step."""
    bs = 4
    x, dt, B, C = _step_inputs(bs, seed=131)
    A, D, dt_bias = _params(seed=132)
    base = _pool(seed=133)
    idx = _slots(bs, seed=134).cuda()
    call = dict(A=A, D=D, dt_bias=dt_bias, indices=idx)

    monkeypatch.setenv("FREETOKEN_MAMBA2_DECODE", "triton")
    want_pool = base.clone()
    want = mamba2_decode(x, dt, B, C, state_source=want_pool, **call)

    monkeypatch.setenv("FREETOKEN_MAMBA2_DECODE", "auto")
    monkeypatch.setattr(_ssu, "_flashinfer_broken", None)
    calls = []

    def _boom(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("nvcc not found")

    monkeypatch.setattr(_ssu, "_decode_flashinfer", _boom)
    got_pool = base.clone()
    got = mamba2_decode(x, dt, B, C, state_source=got_pool, **call)
    assert torch.equal(got, want) and torch.equal(got_pool, want_pool)
    assert _ssu._flashinfer_broken is not None
    assert _ssu.resolve_decode_backend() == "triton"

    # Demoted for good: the second step never touches flashinfer again.
    mamba2_decode(x, dt, B, C, state_source=got_pool, **call)
    assert len(calls) == 1


@cuda_only
@pytest.mark.skipif(len(BACKENDS) < 2, reason="flashinfer not installed")
def test_explicit_flashinfer_does_not_silently_fall_back(monkeypatch):
    bs = 2
    x, dt, B, C = _step_inputs(bs, seed=141)
    A, D, dt_bias = _params(seed=142)
    pool = _pool(seed=143)
    idx = _slots(bs, seed=144).cuda()

    monkeypatch.setenv("FREETOKEN_MAMBA2_DECODE", "flashinfer")
    monkeypatch.setattr(
        _ssu,
        "_decode_flashinfer",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("nvcc not found")),
    )
    with pytest.raises(RuntimeError, match="nvcc"):
        mamba2_decode(
            x, dt, B, C, A=A, D=D, dt_bias=dt_bias, state_source=pool, indices=idx
        )


# ---------------------------------------------------------------- gated norm
@cuda_only
@pytest.mark.parametrize("tokens", [1, 16, 257])
def test_gated_rmsnorm_matches_the_model_reference(tokens):
    """`mamba2_gated_rmsnorm` must reproduce `_MambaGatedRMSNorm` (which is
    itself HF's `Zamba2RMSNormGated`, the module nemotron_h instantiates):
    norm(x * silu(z)) over 8 groups of 512."""
    from transformers.models.nemotron_h.modeling_nemotron_h import Zamba2RMSNormGated

    from freetoken.models.nemotron_h.model import _MambaGatedRMSNorm

    dim, groups, eps = H * P, G, 1e-5
    g = torch.Generator(device="cuda").manual_seed(4)
    bf = {"device": "cuda", "dtype": torch.bfloat16, "generator": g}
    x = torch.randn(tokens, dim, **bf)
    z = torch.randn(tokens, dim, **bf)
    weight = torch.randn(dim, **bf) * 0.2 + 1.0

    ref = _MambaGatedRMSNorm(dim, groups, eps)
    ref.weight = weight
    want = ref.forward(x, z)

    hf = Zamba2RMSNormGated(dim, group_size=dim // groups, eps=eps).cuda().bfloat16()
    with torch.no_grad():
        hf.weight.copy_(weight)
    want_hf = hf(x, z)
    torch.testing.assert_close(want.float(), want_hf.float(), rtol=1e-2, atol=1e-2)

    got = mamba2_gated_rmsnorm(x, z, weight, eps, group_size=dim // groups)
    assert got.shape == x.shape and got.dtype == x.dtype
    torch.testing.assert_close(got.float(), want.float(), rtol=1e-2, atol=1e-2)


@cuda_only
def test_gated_rmsnorm_keeps_3d_shapes():
    dim, eps = H * P, 1e-5
    g = torch.Generator(device="cuda").manual_seed(8)
    bf = {"device": "cuda", "dtype": torch.bfloat16, "generator": g}
    x = torch.randn(3, 5, dim, **bf)
    z = torch.randn(3, 5, dim, **bf)
    w = torch.ones(dim, device="cuda", dtype=torch.bfloat16)
    got = mamba2_gated_rmsnorm(x, z, w, eps)
    flat = mamba2_gated_rmsnorm(x.reshape(-1, dim), z.reshape(-1, dim), w, eps)
    assert got.shape == x.shape
    assert torch.equal(got.reshape(-1, dim), flat)
