"""Arch-aware GGUF kernel dispatch thresholds (sm_120 vs Ada/default).

The threshold *functions* are pure and tested on both archs without a GPU; the
dispatch-site tests stub the CUDA kernels and fake the device capability, so
they only need any CUDA device (branch selection, not numerics).
"""

from __future__ import annotations

import freetoken.layers.gguf as layers_gguf
import freetoken.kernel.gguf as kernel_gguf
import freetoken.moe.fused_gguf as moe_gguf
import pytest
import torch
from freetoken.layers.gguf import dequant_gemm_min_rows, mma_mmq_row_band
from freetoken.models.gguf.dequant import BLOCK_SHAPE, GGML_Q4_K, GGML_Q6_K
from freetoken.moe.fused_gguf import mma_moe_token_range, mmq_min_tokens


@pytest.mark.parametrize(
    "capability,expected",
    [(None, 32), ((8, 9), 32), ((9, 0), 32), ((12, 0), 24), ((12, 1), 24)],
)
def test_dequant_gemm_min_rows(capability, expected):
    assert dequant_gemm_min_rows(capability) == expected


@pytest.mark.parametrize(
    "capability,expected",
    [(None, 32), ((8, 9), 32), ((9, 0), 32), ((12, 0), 16), ((12, 1), 16)],
)
def test_mmq_min_tokens(capability, expected):
    assert mmq_min_tokens(capability) == expected


@pytest.mark.parametrize(
    "qtype,capability,out_features,in_features,expected",
    [
        (GGML_Q4_K, (8, 9), 8192, 2048, (2, 512)),
        (GGML_Q4_K, (8, 9), 12352, 2048, (4, 512)),
        (GGML_Q4_K, (8, 9), 4096, 2048, None),
        (GGML_Q6_K, (8, 9), 8192, 2048, (2, 448)),
        (GGML_Q6_K, (8, 9), 2048, 512, (8, 64)),
        (GGML_Q6_K, (8, 9), 2048, 4096, None),
        (GGML_Q6_K, (8, 9), 9216, 2048, (2, 256)),
        (GGML_Q6_K, (8, 9), 12352, 2048, (2, 256)),
        (GGML_Q6_K, (8, 9), 248320, 2048, (2, 8)),
        (GGML_Q6_K, (8, 9), 152064, 2048, None),
        (GGML_Q4_K, (12, 0), 8, 2048, (7, None)),
        (GGML_Q4_K, (12, 0), 2048, 4096, (7, None)),
        (GGML_Q4_K, (12, 0), 1024, 2048, (7, None)),
        (GGML_Q4_K, (12, 0), 2048, 512, (7, None)),
        (GGML_Q6_K, (12, 0), 2048, 4096, (7, 2048)),
        (GGML_Q6_K, (12, 0), 4096, 2048, (7, 2048)),
        (GGML_Q6_K, (12, 0), 1024, 2048, (7, 23)),
        (GGML_Q6_K, (12, 0), 512, 2048, None),
        (GGML_Q6_K, (12, 1), 12352, 2048, (7, None)),
        (GGML_Q4_K, (9, 0), 8192, 2048, None),
        (2, (8, 9), 8192, 2048, None),
    ],
)
def test_mma_mmq_row_band(qtype, capability, out_features, in_features, expected):
    assert mma_mmq_row_band(qtype, capability, out_features, in_features) == expected


def test_sm120_uncapped_ab_control(monkeypatch):
    monkeypatch.setenv("FREETOKEN_GGUF_SM120_DENSE_UNCAPPED", "1")
    assert mma_mmq_row_band(GGML_Q6_K, (12, 0), 2048, 512) == (7, None)


@pytest.mark.parametrize(
    "qtype,capability,expected",
    [
        (GGML_Q4_K, (8, 9), (272, 16384)),
        (GGML_Q6_K, (8, 9), (272, 16384)),
        (GGML_Q4_K, (12, 0), (272, 16384)),
        (GGML_Q4_K, (9, 0), None),
        (2, (8, 9), None),
    ],
)
def test_mma_moe_token_range(qtype, capability, expected):
    assert mma_moe_token_range(qtype, capability) == expected


@pytest.mark.parametrize(
    "capability,gate_up,expected",
    [
        ((8, 9), True, True),
        ((8, 9), False, True),
        ((9, 0), False, False),
        ((12, 0), True, True),
        ((12, 0), False, True),
    ],
)
def test_shared_vec_multiwarp(capability, gate_up, expected):
    assert kernel_gguf.shared_vec_multiwarp(capability, gate_up=gate_up) is expected


@pytest.mark.parametrize(
    "capability,gate_up,expected",
    [
        ((8, 9), True, 2),
        ((8, 9), False, 4),
        ((9, 0), True, 1),
        ((12, 0), True, 4),
    ],
)
def test_shared_vec_warps(capability, gate_up, expected):
    assert kernel_gguf.shared_vec_warps(capability, gate_up=gate_up) == expected


def test_shared_vec_warps_override(monkeypatch):
    monkeypatch.setenv("FREETOKEN_GGUF_SHARED_GATE_WARPS", "2")
    assert kernel_gguf.shared_vec_warps((8, 9), gate_up=True) == 2
    monkeypatch.setenv("FREETOKEN_GGUF_SHARED_GATE_WARPS", "3")
    with pytest.raises(ValueError, match="must be 1, 2, or 4"):
        kernel_gguf.shared_vec_warps((8, 9), gate_up=True)


@pytest.fixture
def cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")


@pytest.mark.parametrize(
    "capability,rows,expected_branch",
    [
        ((12, 0), 24, "dequant"),
        ((12, 0), 23, "mmq"),
        ((8, 9), 24, "mmq"),
        ((8, 9), 32, "dequant"),
    ],
)
def test_dense_dispatch_branch(cuda, monkeypatch, capability, rows, expected_branch):
    import freetoken.kernel.gguf as kernel_gguf

    monkeypatch.setattr(layers_gguf, "_device_capability", lambda i: capability)
    # The MMA path (tested separately) sits above the dequant/DP4A crossover.
    monkeypatch.setattr(
        layers_gguf, "_use_mma_mmq", lambda qt, cc, rows, out, in_features: False
    )
    block, type_size = BLOCK_SHAPE[GGML_Q4_K]
    in_features, out_features = 256, 8
    qweight = torch.zeros(
        (out_features, in_features // block * type_size),
        dtype=torch.uint8,
        device="cuda",
    )
    x = torch.zeros((rows, in_features), dtype=torch.float16, device="cuda")
    called = []
    monkeypatch.setattr(
        kernel_gguf,
        "ggml_dequantize",
        lambda *a, **k: (
            called.append("dequant")
            or torch.zeros((out_features, in_features), dtype=x.dtype, device="cuda")
        ),
    )
    monkeypatch.setattr(
        kernel_gguf,
        "ggml_mul_mat_a8",
        lambda *a, **k: (
            called.append("mmq")
            or torch.zeros((rows, out_features), dtype=x.dtype, device="cuda")
        ),
    )
    out = layers_gguf.fused_mul_mat_gguf(x, qweight, GGML_Q4_K)
    assert called == [expected_branch]
    assert out.shape == (rows, out_features)


@pytest.mark.parametrize(
    "capability,qtype,rows,out_features,in_features,expect_mma",
    [
        ((12, 0), GGML_Q4_K, 64, 8, 256, True),
        ((12, 1), GGML_Q6_K, 2048, 8, 256, True),
        ((12, 0), GGML_Q6_K, 2048, 2048, 4096, True),
        ((12, 0), GGML_Q6_K, 2049, 2048, 4096, False),
        ((12, 0), GGML_Q6_K, 64, 512, 2048, False),
        ((8, 9), GGML_Q4_K, 2, 8192, 2048, True),
        ((8, 9), GGML_Q4_K, 4, 12352, 2048, True),
        ((8, 9), GGML_Q4_K, 512, 8192, 2048, True),
        ((8, 9), GGML_Q4_K, 513, 8192, 2048, False),
        ((8, 9), GGML_Q4_K, 32, 4096, 2048, False),
        ((8, 9), GGML_Q6_K, 448, 8192, 2048, True),
        ((8, 9), GGML_Q6_K, 449, 8192, 2048, False),
        ((8, 9), GGML_Q6_K, 64, 2048, 512, True),
        ((8, 9), GGML_Q6_K, 65, 2048, 512, False),
        ((9, 0), GGML_Q4_K, 64, 8192, 2048, False),
    ],
)
def test_dense_dispatch_mma_branch(
        cuda, monkeypatch, capability, qtype, rows, out_features, in_features, expect_mma
):
    """Blackwell is uncapped; Ada MMA is bounded before cuBLAS retakes the lead."""
    import freetoken.kernel.gguf as kernel_gguf

    monkeypatch.setattr(layers_gguf, "_device_capability", lambda i: capability)
    monkeypatch.setattr(layers_gguf, "_mma_mmq_ok", lambda: True)
    block, type_size = BLOCK_SHAPE[qtype]
    qweight = torch.zeros(
        (out_features, in_features // block * type_size),
        dtype=torch.uint8,
        device="cuda",
    )
    x = torch.zeros((rows, in_features), dtype=torch.float16, device="cuda")
    called = []
    monkeypatch.setattr(
        kernel_gguf,
        "ggml_mul_mat_a8_mma",
        lambda *a, **k: (
            called.append("mma")
            or torch.zeros((rows, out_features), dtype=torch.float32, device="cuda")
        ),
    )
    monkeypatch.setattr(
        kernel_gguf,
        "ggml_dequantize",
        lambda *a, **k: (
            called.append("dequant")
            or torch.zeros((out_features, in_features), dtype=x.dtype, device="cuda")
        ),
    )
    out = layers_gguf.fused_mul_mat_gguf(x, qweight, qtype)
    assert called == (["mma"] if expect_mma else ["dequant"])
    assert out.dtype == x.dtype


@pytest.mark.parametrize(
    "capability,tokens,expect_mmq",
    [
        ((12, 0), 16, True),
        ((12, 0), 15, False),
        ((8, 9), 16, False),
        ((8, 9), 32, True),
    ],
)
def test_moe_dispatch_branch(cuda, monkeypatch, capability, tokens, expect_mmq):
    import freetoken.kernel.gguf as kernel_gguf

    monkeypatch.setattr(layers_gguf, "_device_capability", lambda i: capability)
    monkeypatch.setattr(moe_gguf, "_use_mma_moe", lambda *a: False)
    considered = []
    # Returning 0 makes the MMQ branch fall through to the (stubbed) vec path,
    # so the sentinel records threshold crossing without running any kernel.
    monkeypatch.setattr(
        kernel_gguf,
        "ggml_moe_get_block_size",
        lambda qt: considered.append("mmq") or 0,
    )
    sentinel = torch.zeros((tokens, 4))
    monkeypatch.setattr(moe_gguf, "_moe_vec_chunked", lambda *a, **k: sentinel)
    x = torch.zeros((tokens, 8), dtype=torch.float16, device="cuda")
    topk_ids = torch.zeros((tokens, 2), dtype=torch.int32, device="cuda")
    weight = torch.zeros((4, 16), dtype=torch.uint8, device="cuda")
    out = moe_gguf._moe_matmul(
        x, weight, topk_ids, 2, GGML_Q4_K, rows=4, tokens=tokens, stride=16
    )
    assert (considered == ["mmq"]) is expect_mmq
    assert out is sentinel


def test_moe_use_mma_gate(monkeypatch):
    """_use_mma_moe: measured arch/range + supported type + aligned stride."""
    monkeypatch.setattr(layers_gguf, "_mma_mmq_ok", lambda: True)
    type_size = BLOCK_SHAPE[GGML_Q4_K][1]
    assert moe_gguf._use_mma_moe(GGML_Q4_K, 4 * type_size, (12, 0), 272, 4, True, 2)
    assert moe_gguf._use_mma_moe(GGML_Q4_K, 4 * type_size, (8, 9), 272, 1024, True, 8)
    assert moe_gguf._use_mma_moe(
        GGML_Q6_K, 4 * BLOCK_SHAPE[GGML_Q6_K][1], (8, 9), 272, 1024, True, 8
    )
    assert moe_gguf._use_mma_moe(
        GGML_Q6_K, 4 * BLOCK_SHAPE[GGML_Q6_K][1], (8, 9), 272, 2048, False, 8
    )
    assert not moe_gguf._use_mma_moe(
        GGML_Q4_K, 4 * type_size, (8, 9), 256, 1024, True, 8
    )
    assert not moe_gguf._use_mma_moe(
        GGML_Q4_K, 4 * type_size, (8, 9), 272, 512, True, 8
    )
    assert not moe_gguf._use_mma_moe(
        GGML_Q4_K, 4 * type_size + 1, (12, 0), 320, 4, True, 2
    )
    assert not moe_gguf._use_mma_moe(
        2, 64, (12, 0), 320, 4, True, 2
    )  # Q4_0: not instantiated


@pytest.mark.parametrize(
    "tokens,broadcast,expect_mma",
    [(320, True, True), (320, False, True), (319, True, False), (16385, True, False)],
)
def test_moe_dispatch_mma_branch(cuda, monkeypatch, tokens, broadcast, expect_mma):
    import freetoken.kernel.gguf as kernel_gguf

    monkeypatch.setattr(layers_gguf, "_device_capability", lambda i: (12, 0))
    monkeypatch.setattr(moe_gguf, "_use_mma_moe", lambda *a: expect_mma)
    top_k = 2
    called = []
    mma_out = torch.zeros(tokens * top_k, 4)
    monkeypatch.setattr(
        kernel_gguf, "ggml_moe_a8_mma", lambda *a, **k: called.append("mma") or mma_out
    )
    vec_out = torch.zeros(tokens * top_k, 4)
    monkeypatch.setattr(moe_gguf, "_moe_vec_chunked", lambda *a, **k: vec_out)
    import freetoken.kernel.gguf as kg

    monkeypatch.setattr(kg, "ggml_moe_get_block_size", lambda qt: 0)
    rows_x = tokens if broadcast else tokens * top_k
    x = torch.zeros((rows_x, 8), dtype=torch.float16, device="cuda")
    topk_ids = torch.zeros((tokens, top_k), dtype=torch.int32, device="cuda")
    weight = torch.zeros((4, 16), dtype=torch.uint8, device="cuda")
    out = moe_gguf._moe_matmul(
        x,
        weight,
        topk_ids,
        top_k,
        GGML_Q4_K,
        rows=4,
        tokens=tokens,
        stride=16,
        broadcast=broadcast,
    )
    assert (called == ["mma"]) is expect_mma
    if expect_mma:
        assert out.dtype == x.dtype  # fp32 kernel output cast back
    else:
        assert out is vec_out
