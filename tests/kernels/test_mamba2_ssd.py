"""Parity tests for the vendored Triton Mamba-2 SSD prefill kernels.

The oracle is :func:`tests.kernels.mamba2_gold.gold_ssd`: the SSD recurrence
itself, evaluated one token at a time in float64. HF's
``mamba2_chunk_scan`` is *a* second implementation, not a gold -- it is the same
chunked factorisation the kernels use, so it cannot expose a shared algorithmic
mistake, and materialising its ``[nchunks, chunk, H, P, N]`` intermediate costs
~9 GB at T=4097. `test_hf_chunk_scan_agrees_with_the_gold` pins it at a length
where it is cheap; everything else compares to the gold directly.

Measured against the gold on an RTX 5080 (all lengths 1..4097, with and without
a carried state), the kernel's RMS-relative error is entirely input-dtype
rounding -- the kernel rounds ``B * decay`` (chunk state) and ``CB`` (chunk scan)
to the input dtype before each ``tl.dot``, and the residual tracks the mantissa
exactly:

    input dtype   mantissa bits   state err   output err
    bf16          8               1.7e-3      1.8e-3
    fp16          11              2.1e-4      2.2e-4     (8x = 2^3 bits)
    fp32          24              1.4e-6      3.2e-7

For reference, HF's fp32 chunk scan sits at ~7e-7 / 6e-8 against the same gold.
`test_error_floor_tracks_input_mantissa` keeps that ladder honest: a real bug
(bad masking, a dropped decay, a mis-scattered state) would not shrink 8x when
three mantissa bits are added.

Coverage: single-sequence lengths that straddle the 128 chunk boundary, zero and
random carried state, a varlen batch with permuted pool slots and an untouched
pad slot, the `chunk_offsets[i] + c` intermediate-state contract, and chunk
continuation (T1 then T2 carrying state == one T1+T2 scan).

Everything that touches Triton needs CUDA; the metadata tests are pure host code
and run everywhere.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.kernel.triton.mamba2 import build_mamba2_metadata, mamba2_prefill

from .mamba2_gold import gold_ssd

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

# Nemotron-3.5 Lightning geometry.
H, P, N, G, CHUNK = 64, 64, 128, 8, 128

# bf16 in, fp32 accumulate. The measured floor is 1.7e-3 / 1.8e-3 (state /
# output) and is flat in T, so 3e-3 leaves ~1.7x headroom without hiding a
# regression.
OUT_ERR_RATIO = 3e-3
STATE_ERR_RATIO = 3e-3
# Elementwise guard against a localised blow-up the RMS ratio would average
# away: measured max|diff| on the state is 2.4e-3 on values up to 0.83.
STATE_RTOL, STATE_ATOL = 2e-2, 2e-3
# With fp32 inputs there is no rounding through the dots at all (`tl.dot` is
# pinned to `input_precision="ieee"`; the Triton default of TF32 would put this
# path at 4e-4, i.e. fp16 grade). Measured: 1.4e-6 state, 3.2e-7 output.
FP32_STATE_ERR_RATIO = 1e-5
FP32_STATE_RTOL, FP32_STATE_ATOL = 1e-4, 1e-5
FP32_OUT_ERR_RATIO = 5e-6


def _assert_state(got, want, msg="", *, fp32=False):
    bar = FP32_STATE_ERR_RATIO if fp32 else STATE_ERR_RATIO
    rtol, atol = (
        (FP32_STATE_RTOL, FP32_STATE_ATOL) if fp32 else (STATE_RTOL, STATE_ATOL)
    )
    got, want = got.float(), want.float()
    assert _err_ratio(want, got) < bar, f"state rms error {msg}"
    torch.testing.assert_close(got, want, rtol=rtol, atol=atol)


def _inputs(total: int, device="cuda", seed=0):
    """Nemotron-like SSM inputs. A in [-16, -1], dt in [1e-3, 1e-1] after softplus."""
    g = torch.Generator(device=device).manual_seed(seed)
    bf = {"device": device, "dtype": torch.bfloat16, "generator": g}
    f32 = {"device": device, "dtype": torch.float32, "generator": g}
    x = torch.randn(total, H, P, **bf)
    B = torch.randn(total, G, N, **bf) * 0.5
    C = torch.randn(total, G, N, **bf) * 0.5
    # dt_bias is the inverse-softplus of U(1e-3, 1e-1), matching HF init.
    tgt = torch.rand(H, **f32) * 0.099 + 1e-3
    dt_bias = torch.log(torch.expm1(tgt))
    dt = torch.randn(total, H, **bf) * 0.3
    A = -torch.exp(torch.rand(H, **f32) * 2.77)  # A_log ~ U(0, log 16)
    D = torch.randn(H, **f32)
    return x, dt, B, C, A, D, dt_bias


def _gold(x, dt, B, C, A, D, dt_bias, initial, chunk_size=None):
    """fp64 sequential recurrence. initial/final states are [H, P, N]."""
    return gold_ssd(
        x, dt, B, C, A, D, dt_bias, initial, chunk_size=chunk_size
    )


def _hf_reference(x, dt, B, C, A, D, dt_bias, initial, chunk_size=CHUNK):
    """HF fp32 chunk scan. Memory-hungry: only used on short sequences."""
    from transformers.models.nemotron_h.modeling_nemotron_h import mamba2_chunk_scan

    out, final = mamba2_chunk_scan(
        x[None].float(),
        dt[None].float(),
        A.float(),
        B[None].float(),
        C[None].float(),
        chunk_size=chunk_size,
        D=D.float(),
        dt_bias=dt_bias.float(),
        initial_states=initial[None].float(),
        dt_softplus=True,
        dt_limit=(0.0, float("inf")),
        return_final_states=True,
    )
    return out[0], final[0]


def _err_ratio(ref, got):
    from freetoken.kernel.fla.utils import get_err_ratio

    return get_err_ratio(ref.float(), got.float())


def _run(x, dt, B, C, A, D, dt_bias, *, lens, pool, indices, has_init=None, **kw):
    cu = [0]
    for n in lens:
        cu.append(cu[-1] + n)
    cu_seqlens = torch.tensor(cu, dtype=torch.int32, device=x.device)
    meta = build_mamba2_metadata(cu, CHUNK, device=x.device)
    return meta, mamba2_prefill(
        x, dt, B, C,
        A=A, D=D, dt_bias=dt_bias, meta=meta, cu_seqlens=cu_seqlens,
        state_source=pool, indices=indices, has_initial_state=has_init, **kw,
    )


# --------------------------------------------------------------------------- #
# metadata (host only)
# --------------------------------------------------------------------------- #

def test_metadata_is_sequence_aligned():
    meta = build_mamba2_metadata([0, 1, 201, 329, 1106], CHUNK, device="cpu")
    # lens 1, 200, 128, 777 -> 1 + 2 + 1 + 7 chunks
    assert meta.num_chunks == 11
    assert meta.num_seqs == 4
    assert meta.cu_chunk_seqlens.tolist() == [
        0, 1, 129, 201, 329, 457, 585, 713, 841, 969, 1097, 1106
    ]
    assert meta.seq_idx.tolist() == [0, 1, 1, 2, 3, 3, 3, 3, 3, 3, 3]
    assert meta.last_chunk_indices.tolist() == [0, 2, 3, 10]
    assert meta.chunk_offsets.tolist() == [0, 1, 3, 4]
    # every chunk fits and none straddles a sequence
    cu = meta.cu_chunk_seqlens.tolist()
    assert all(0 < cu[i + 1] - cu[i] <= CHUNK for i in range(meta.num_chunks))


def test_metadata_rejects_empty_sequences():
    with pytest.raises(AssertionError):
        build_mamba2_metadata([0, 5, 5, 9], CHUNK, device="cpu")


def test_metadata_accepts_a_tensor():
    a = build_mamba2_metadata([0, 300], CHUNK, device="cpu")
    b = build_mamba2_metadata(
        torch.tensor([0, 300], dtype=torch.int64), CHUNK, device="cpu"
    )
    assert a.cu_chunk_seqlens.tolist() == b.cu_chunk_seqlens.tolist()
    assert a.num_chunks == b.num_chunks == 3


# --------------------------------------------------------------------------- #
# the oracle itself
# --------------------------------------------------------------------------- #

@cuda_only
@pytest.mark.parametrize("T", [129])
def test_hf_chunk_scan_agrees_with_the_gold(T):
    """HF's fp32 chunk scan is ~1e-6 from the fp64 recurrence.

    Kept short on purpose: HF materialises `[nchunks, chunk, H, P, N]` fp32,
    i.e. ~0.5 GB per 128 tokens at this geometry (1.1 GB peak at T=129).
    """
    x, dt, B, C, A, D, dt_bias = _inputs(T, seed=T)
    g = torch.Generator(device="cuda").manual_seed(1234)
    initial = torch.randn(H, P, N, device="cuda", generator=g) * 0.1

    hf_out, hf_state = _hf_reference(x, dt, B, C, A, D, dt_bias, initial)
    g_out, g_state = _gold(x, dt, B, C, A, D, dt_bias, initial)
    assert _err_ratio(g_state, hf_state) < 1e-5
    assert _err_ratio(g_out, hf_out) < 1e-5


@cuda_only
def test_error_floor_tracks_input_mantissa():
    """The residual vs the gold is input rounding, not a kernel mistake.

    bf16 (8 mantissa bits) -> fp16 (11) must buy ~2^3, and fp32 (24) must be
    ~fp32-exact. A structural bug would be dtype-independent.
    """
    T = 1024
    ref = _inputs(T, seed=T)
    errs = {}
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        x, dt, B, C = (t.to(dtype) for t in ref[:4])
        A, D, dt_bias = ref[4:]
        pool = torch.zeros(1, H, P, N, device="cuda", dtype=torch.float32)
        idx = torch.tensor([0], dtype=torch.int32, device="cuda")
        _, (out, _) = _run(
            x, dt, B, C, A, D, dt_bias, lens=[T], pool=pool, indices=idx,
            has_init=torch.tensor([False], device="cuda"),
        )
        g_out, g_state = _gold(x, dt, B, C, A, D, dt_bias, None)
        errs[dtype] = (_err_ratio(g_state, pool[0]), _err_ratio(g_out, out))

    bf16, fp16, fp32 = (errs[d] for d in (torch.bfloat16, torch.float16, torch.float32))
    for i, what in enumerate(("state", "output")):
        assert fp16[i] < bf16[i] / 4, f"{what}: fp16 {fp16[i]:.2e} vs bf16 {bf16[i]:.2e}"
        assert fp32[i] < fp16[i] / 20, f"{what}: fp32 {fp32[i]:.2e} vs fp16 {fp16[i]:.2e}"


# --------------------------------------------------------------------------- #
# single sequence
# --------------------------------------------------------------------------- #

@cuda_only
@pytest.mark.parametrize("T", [1, 5, 127, 128, 129, 300, 1024, 4097])
@pytest.mark.parametrize("with_initial_state", [False, True])
def test_single_sequence_matches_the_gold(T, with_initial_state):
    x, dt, B, C, A, D, dt_bias = _inputs(T, seed=T)
    pool = torch.zeros(3, H, P, N, device="cuda", dtype=torch.float32)
    if with_initial_state:
        g = torch.Generator(device="cuda").manual_seed(1234)
        pool[1] = torch.randn(H, P, N, device="cuda", generator=g) * 0.1
    indices = torch.tensor([1], dtype=torch.int32, device="cuda")
    initial = pool[1].clone()

    _, (out, inter) = _run(
        x, dt, B, C, A, D, dt_bias, lens=[T], pool=pool, indices=indices,
        has_init=torch.tensor([with_initial_state], device="cuda"),
    )
    assert inter is None
    assert out.shape == (T, H, P) and out.dtype == torch.bfloat16

    ref_out, ref_state = _gold(
        x, dt, B, C, A, D, dt_bias, initial if with_initial_state else None
    )
    assert _err_ratio(ref_out, out) < OUT_ERR_RATIO
    _assert_state(pool[1], ref_state)
    # the scan must not have touched the neighbouring slots
    assert pool[0].abs().max() == 0 and pool[2].abs().max() == 0


@cuda_only
def test_has_initial_state_false_ignores_a_dirty_slot():
    """A fresh sequence starts from zeros even when its pool slot holds garbage."""
    T = 300
    x, dt, B, C, A, D, dt_bias = _inputs(T, seed=7)
    pool = torch.randn(2, H, P, N, device="cuda", dtype=torch.float32)
    indices = torch.tensor([0], dtype=torch.int32, device="cuda")

    _, (out, _) = _run(
        x, dt, B, C, A, D, dt_bias, lens=[T], pool=pool.clone(), indices=indices,
        has_init=torch.tensor([False], device="cuda"),
    )
    ref_out, _ = _gold(x, dt, B, C, A, D, dt_bias, None)
    assert _err_ratio(ref_out, out) < OUT_ERR_RATIO


# --------------------------------------------------------------------------- #
# varlen
# --------------------------------------------------------------------------- #

@cuda_only
def test_varlen_batch_with_permuted_slots_and_a_pad_row():
    lens = [1, 200, 128, 777]
    total = sum(lens)
    x, dt, B, C, A, D, dt_bias = _inputs(total, seed=99)

    slots = 6
    g = torch.Generator(device="cuda").manual_seed(5)
    pool = torch.randn(slots, H, P, N, device="cuda", generator=g) * 0.1
    # deliberately out of order, and slots 0 / 5 are never named: slot 5 is the
    # pad row a padded batch would carry.
    indices = torch.tensor([4, 1, 3, 2], dtype=torch.int32, device="cuda")
    has_init = torch.tensor([True, False, True, False], device="cuda")
    before = pool.clone()

    meta, (out, inter) = _run(
        x, dt, B, C, A, D, dt_bias, lens=lens, pool=pool, indices=indices,
        has_init=has_init, return_intermediate_states=True,
    )
    assert inter.shape == (meta.num_chunks, H, P, N)

    off = 0
    for i, L in enumerate(lens):
        sl = slice(off, off + L)
        slot = int(indices[i])
        init = before[slot] if has_init[i] else None
        ref_out, ref_state = _gold(
            x[sl], dt[sl], B[sl], C[sl], A, D, dt_bias, init
        )
        assert _err_ratio(ref_out, out[sl]) < OUT_ERR_RATIO, f"seq {i}"
        _assert_state(pool[slot], ref_state, f"seq {i}")
        off += L

    # unnamed slots (including the pad row) are byte-identical
    torch.testing.assert_close(pool[0], before[0], rtol=0, atol=0)
    torch.testing.assert_close(pool[5], before[5], rtol=0, atol=0)


# --------------------------------------------------------------------------- #
# intermediate states / continuation
# --------------------------------------------------------------------------- #

@cuda_only
@pytest.mark.parametrize("c", [0, 3, 7])
def test_intermediate_state_indexing(c):
    """`intermediate[chunk_offsets[i] + c]` is the state after (c+1)*CHUNK tokens."""
    T = 1024
    x, dt, B, C, A, D, dt_bias = _inputs(T, seed=11)
    pool = torch.zeros(2, H, P, N, device="cuda", dtype=torch.float32)
    indices = torch.tensor([0], dtype=torch.int32, device="cuda")

    meta, (_, inter) = _run(
        x, dt, B, C, A, D, dt_bias, lens=[T], pool=pool, indices=indices,
        return_intermediate_states=True,
    )
    assert inter.shape == (T // CHUNK, H, P, N)

    _, _, chunk_states = _gold(
        x, dt, B, C, A, D, dt_bias, None, chunk_size=CHUNK
    )
    assert chunk_states.shape == (T // CHUNK, H, P, N)
    row = int(meta.chunk_offsets[0]) + c
    _assert_state(inter[row], chunk_states[c])


@cuda_only
def test_intermediate_offsets_are_per_sequence():
    lens = [300, 128, 700]
    total = sum(lens)
    x, dt, B, C, A, D, dt_bias = _inputs(total, seed=13)
    pool = torch.zeros(3, H, P, N, device="cuda", dtype=torch.float32)
    indices = torch.tensor([2, 0, 1], dtype=torch.int32, device="cuda")

    meta, (_, inter) = _run(
        x, dt, B, C, A, D, dt_bias, lens=lens, pool=pool, indices=indices,
        return_intermediate_states=True,
    )
    # 3 + 1 + 6 chunks; each sequence's last intermediate == its scattered state
    assert meta.chunk_offsets.tolist() == [0, 3, 4]
    off = 0
    for i, L in enumerate(lens):
        last = int(meta.chunk_offsets[i]) + (L + CHUNK - 1) // CHUNK - 1
        assert last == int(meta.last_chunk_indices[i])
        torch.testing.assert_close(inter[last], pool[int(indices[i])], rtol=0, atol=0)
        # ... and every chunk boundary of that sequence matches the gold
        sl = slice(off, off + L)
        _, _, chunk_states = _gold(
            x[sl], dt[sl], B[sl], C[sl], A, D, dt_bias, None, chunk_size=CHUNK
        )
        for c in range(chunk_states.shape[0]):
            _assert_state(inter[int(meta.chunk_offsets[i]) + c], chunk_states[c],
                          f"seq {i} chunk {c}")
        off += L


@cuda_only
@pytest.mark.parametrize("T1,T2", [(128, 128), (200, 328), (1, 512)])
def test_chunk_continuation_equals_one_pass(T1, T2):
    T = T1 + T2
    x, dt, B, C, A, D, dt_bias = _inputs(T, seed=17)
    idx = torch.tensor([0], dtype=torch.int32, device="cuda")

    one = torch.zeros(1, H, P, N, device="cuda", dtype=torch.float32)
    _, (out_one, _) = _run(
        x, dt, B, C, A, D, dt_bias, lens=[T], pool=one, indices=idx,
        has_init=torch.tensor([False], device="cuda"),
    )

    two = torch.zeros(1, H, P, N, device="cuda", dtype=torch.float32)
    _, (out_a, _) = _run(
        x[:T1], dt[:T1], B[:T1], C[:T1], A, D, dt_bias, lens=[T1], pool=two,
        indices=idx, has_init=torch.tensor([False], device="cuda"),
    )
    _, (out_b, _) = _run(
        x[T1:], dt[T1:], B[T1:], C[T1:], A, D, dt_bias, lens=[T2], pool=two,
        indices=idx, has_init=torch.tensor([True], device="cuda"),
    )

    assert _err_ratio(out_one[:T1], out_a) < OUT_ERR_RATIO
    assert _err_ratio(out_one[T1:], out_b) < OUT_ERR_RATIO
    _assert_state(two[0], one[0])
    # both paths must also land on the gold, not merely on each other
    ref_out, ref_state = _gold(x, dt, B, C, A, D, dt_bias, None)
    assert _err_ratio(ref_out, out_one) < OUT_ERR_RATIO
    _assert_state(two[0], ref_state)


@cuda_only
@pytest.mark.parametrize("T", [129, 1024])
def test_fp32_inputs_hit_the_tight_state_tolerance(T):
    """No rounding through the dots in the fp32 path, so 1e-4 / 1e-5 holds.

    This separates "the kernel is right" from "bf16 has 8 mantissa bits": the
    bf16 cases above use a looser RMS-relative bar instead. It also pins
    `input_precision="ieee"` on `tl.dot` -- Triton's TF32 default silently
    demotes this path to 4e-4, i.e. fp16 grade.
    """
    x, dt, B, C, A, D, dt_bias = _inputs(T, seed=T + 1)
    x, dt, B, C = (t.float() for t in (x, dt, B, C))
    pool = torch.zeros(1, H, P, N, device="cuda", dtype=torch.float32)
    indices = torch.tensor([0], dtype=torch.int32, device="cuda")

    _, (out, _) = _run(
        x, dt, B, C, A, D, dt_bias, lens=[T], pool=pool, indices=indices,
        has_init=torch.tensor([False], device="cuda"),
    )
    ref_out, ref_state = _gold(x, dt, B, C, A, D, dt_bias, None)
    assert out.dtype == torch.float32
    assert _err_ratio(ref_out, out) < FP32_OUT_ERR_RATIO
    _assert_state(pool[0], ref_state, fp32=True)
