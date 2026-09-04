"""Dense NVFP4 (W4A16) linear on the Nemotron-3.5-Lightning shapes.

Guards the sm_120 launch-config tuning (task 2B3): the per-arch table in
``kernel/triton/nvfp4_linear.py`` changes tile sizes / split-K / the in-kernel-vs-scratch
crossover per device, and every one of those choices must stay numerically equivalent to
the ``FREETOKEN_DEBUG_DENSE_NVFP4_REF=1`` reference (``dequant_nvfp4`` to bf16 + matmul)
and to the H100 fallback constants. Bit-for-bit equality across launch configs is not
required -- split-K changes the fp32 summation order -- but the outputs must agree to
cosine > 0.9999 and the lm_head argmax must not move.

Shapes are Lightning's three dense NVFP4 projections:
``shared_experts.up_proj`` 3712x2688, ``shared_experts.down_proj`` 2688x3712,
``lm_head`` 131072x2688; M covers decode (1), a decode batch (8) and prefill (8192).

The ``needs_weights`` tests run the same checks on the real checkpoint tensors pointed at
by ``FREETOKEN_NEMOTRON_LIGHTNING_PATH``.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="NVFP4 W4A16 kernels are CUDA-only"
)

SHARED_UP = (3712, 2688)
SHARED_DOWN = (2688, 3712)
LM_HEAD = (131072, 2688)
_REAL_PATH_ENV = "FREETOKEN_NEMOTRON_LIGHTNING_PATH"


@pytest.fixture(autouse=True)
def _deterministic_activations():
    """Every test here draws its activations from the global RNG and then asserts on an
    argmax over 131072 near-tied logits; leaving that unseeded makes the whole file a
    coin flip that fails once every few runs on a different row."""
    torch.manual_seed(20250904)


def _free_gib() -> float:
    torch.cuda.empty_cache()  # the caching allocator's freed blocks are usable here
    return torch.cuda.mem_get_info()[0] / 1024**3


def _need_vram(n: int, k: int, m: int, *, outputs: int = 1, bf16_weight: bool = False) -> None:
    """Skip unless the point fits: NVFP4 weight (row-major + K-major resident copy), the
    activation, ``outputs`` [M, N] results, and optionally the bf16 dequant reference
    weight (704 MB for the lm_head), plus 0.7 GiB of allocator/context slack. The GPU is
    shared, so ``lm_head`` at prefill M (2.1 GB of logits per copy) skips itself here."""
    need = (n * k * 0.5 * 2 + n * k / 16 * 2 + m * k * 2 + outputs * m * n * 2
            + (n * k * 2 if bf16_weight else 0)) / 1024**3 + 0.7
    if _free_gib() < need:
        pytest.skip(f"needs {need:.1f} GiB free VRAM, have {_free_gib():.1f}")


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().flatten(), b.float().flatten()
    return float(torch.dot(a, b) / (a.norm() * b.norm()))


def assert_top1_agrees(got: torch.Tensor, ref: torch.Tensor, *, exact: bool) -> None:
    """The sampler only ever sees the argmax, so that is the invariant that matters.

    Two claims, in decreasing strength:

    * Always: the token the kernel picks must be worth as much as the reference's best,
      to within bf16 rounding. Both sides are bf16, so the budget is two ulps of *that
      row's* own magnitude -- one for each side's rounding of its fp32 accumulator. The
      budget has to be row-relative: over 256 rows the widest-gap row is not the
      largest-logit row.
    * ``exact``: the token *id* must not move either -- but only on rows where bf16 can
      actually tell the top two apart. The activations here are random, so a row's top
      131072 logits are near-uniform noise and its top two are routinely inside that same
      two-ulp budget; on those rows no summation order is more right than another, and
      demanding a particular winner would test the RNG rather than the kernel."""
    fref = ref.float()
    picked = fref.gather(-1, got.argmax(-1, keepdim=True)).squeeze(-1)
    best, runner_up = fref.topk(2, dim=-1).values.unbind(-1)
    ulp2 = 2 * 2.0**-7 * best.abs()  # bf16 keeps 8 significand bits
    assert bool(((best - picked) <= ulp2).all())
    if exact:
        decisive = (best - runner_up) > ulp2
        assert bool((got.argmax(-1) == ref.argmax(-1))[decisive].all())


def _synthetic(n: int, k: int, seed: int = 0):
    """A random NVFP4 weight in checkpoint (row-major) layout: packed uint8 [N, K//2],
    fp8-e4m3 block scales [N, K//16], fp16 per-row global scale [N]."""
    from freetoken.kernel.triton.nvfp4_linear import FP8

    g = torch.Generator(device="cuda").manual_seed(seed)
    packed = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda", generator=g)
    scale = (torch.rand((n, k // 16), device="cuda", generator=g) * 1.5 + 0.25).to(FP8)
    gscale = torch.full((n,), 0.03, dtype=torch.float16, device="cuda")
    return packed, scale, gscale


def _reference(x, packed, scale, gscale):
    """Exactly what ``FREETOKEN_DEBUG_DENSE_NVFP4_REF=1`` runs for the row-major entry
    point: ``dequant_nvfp4`` to bf16 then a plain matmul."""
    from freetoken.kernel.triton import nvfp4_linear as nl

    return nl._ref(x, packed, scale, gscale, x.dtype)


def _resident(packed, scale):
    from freetoken.kernel.triton.nvfp4_linear import nvfp4_transpose_resident

    return nvfp4_transpose_resident(packed, scale)


@pytest.mark.parametrize("shape,m", [
    (SHARED_UP, 1), (SHARED_UP, 8), (SHARED_UP, 8192),
    (SHARED_DOWN, 1), (SHARED_DOWN, 8), (SHARED_DOWN, 8192),
    (LM_HEAD, 1), (LM_HEAD, 8), (LM_HEAD, 8192),
], ids=lambda v: f"{v[0]}x{v[1]}" if isinstance(v, tuple) else f"M{v}")
def test_lightning_shapes_match_the_debug_reference(shape, m):
    """Both storage layouts, every dispatch branch (GEMV / in-kernel dot GEMM / dequant +
    cuBLAS scratch), against the dequant reference."""
    from freetoken.kernel.triton.nvfp4_linear import nvfp4_dense_linear, nvfp4_dense_linear_t

    n, k = shape
    _need_vram(n, k, m, outputs=3, bf16_weight=True)

    packed, scale, gscale = _synthetic(n, k)
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda") * 0.05
    ref = _reference(x, packed, scale, gscale)

    got_row = nvfp4_dense_linear(x, packed, scale, gscale)
    assert got_row.shape == (m, n)
    assert cosine(got_row, ref) > 0.9999
    del got_row

    weight_t, scale_t = _resident(packed, scale)
    del packed
    got_t = nvfp4_dense_linear_t(x, weight_t, scale_t, gscale)
    assert cosine(got_t, ref) > 0.9999
    if n > 4096:  # lm_head: the argmax is what the sampler actually consumes
        assert_top1_agrees(got_t, ref, exact=False)


def test_gemv_path_is_taken_for_a_1d_activation():
    """M == 1 (decode) must hit the GEMV, not the GEMM -- a [K] activation and a [1, K]
    one have to agree, and the [K] form is what the decode graph captures."""
    from freetoken.kernel.triton.nvfp4_linear import nvfp4_dense_linear_t

    n, k = SHARED_UP
    packed, scale, gscale = _synthetic(n, k, seed=3)
    weight_t, scale_t = _resident(packed, scale)
    x = torch.randn(k, dtype=torch.bfloat16, device="cuda") * 0.05
    flat = nvfp4_dense_linear_t(x, weight_t, scale_t, gscale)
    batched = nvfp4_dense_linear_t(x.reshape(1, k), weight_t, scale_t, gscale)
    assert flat.shape == (n,)
    assert cosine(flat, batched[0]) > 0.9999


@pytest.mark.parametrize("m", [1, 8, 64, 256, 8192])
def test_fused_relu2_epilogue_matches_the_eager_activation(m):
    """``act="relu2"`` (Nemotron's ungated shared expert) must equal ``relu(y)**2`` on
    every dispatch branch. The fused form applies the activation to the fp32 accumulator
    instead of the rounded bf16 output, so it is a hair *more* accurate, not identical."""
    from freetoken.kernel.triton.nvfp4_linear import nvfp4_dense_linear_t

    n, k = SHARED_UP
    _need_vram(n, k, m, outputs=3)
    packed, scale, gscale = _synthetic(n, k, seed=5)
    weight_t, scale_t = _resident(packed, scale)
    del packed
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda") * 0.05
    a = x[0] if m == 1 else x
    plain = nvfp4_dense_linear_t(a, weight_t, scale_t, gscale)
    fused = nvfp4_dense_linear_t(a, weight_t, scale_t, gscale, act="relu2")
    assert cosine(fused, torch.relu(plain).square()) > 0.9999


def test_unknown_fused_activation_is_rejected():
    from freetoken.kernel.triton.nvfp4_linear import Nvfp4DenseLinear

    with pytest.raises(AssertionError):
        Nvfp4DenseLinear(16, 16, act="gelu")


@pytest.mark.parametrize("shape", [SHARED_UP, SHARED_DOWN, LM_HEAD],
                         ids=lambda v: f"{v[0]}x{v[1]}")
@pytest.mark.parametrize("m", [1, 8, 64, 256])
def test_arch_tuning_agrees_with_the_fallback_constants(shape, m):
    """The sm_120 table and the H100 fallback pick different tiles / split-K / crossover;
    the results must still agree (fp32 reduction order is all that differs).

    M=256 straddles the tuned crossover on purpose: a shared-expert weight is past
    ``gemm_max_inkernel_m`` (dequant + cuBLAS) while the lm_head, being wider than
    ``gemm_wide_n``, is still in-kernel there -- so both sides of the wide/narrow split
    are exercised against the same fallback, which sends everything to the scratch."""
    from freetoken.kernel.triton import nvfp4_linear as nl

    n, k = shape
    _need_vram(n, k, m, outputs=2)
    packed, scale, gscale = _synthetic(n, k, seed=1)
    weight_t, scale_t = _resident(packed, scale)
    del packed
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda") * 0.05
    a = x[0] if m == 1 else x

    saved = dict(nl._TUNING_CACHE)
    try:
        nl._TUNING_CACHE.clear()
        arch = nl.nvfp4_dense_linear_t(a, weight_t, scale_t, gscale)
        nl._TUNING_CACHE.clear()
        nl._TUNING_CACHE[torch.cuda.current_device()] = nl._resolve_tuning(
            torch.cuda.current_device(), nl._DEFAULT_TUNING
        )
        fallback = nl.nvfp4_dense_linear_t(a, weight_t, scale_t, gscale)
    finally:
        nl._TUNING_CACHE.clear()
        nl._TUNING_CACHE.update(saved)

    assert cosine(arch, fallback) > 0.9999
    if n > 4096:
        assert_top1_agrees(arch, fallback, exact=False)


def test_tuning_table_is_keyed_by_sm_version_and_falls_back():
    """Unknown archs must keep the pre-tuning (H100) constants, and this device must be
    resolved through the table it is keyed by."""
    from freetoken.kernel.triton import nvfp4_linear as nl

    props = torch.cuda.get_device_properties(0)
    assert nl._ARCH_TUNING.get((99, 9)) is None
    fallback = nl._resolve_tuning(0, nl._ARCH_TUNING.get((99, 9), nl._DEFAULT_TUNING))
    assert fallback.gemm_max_inkernel_m == nl._DEFAULT_TUNING.gemm_max_inkernel_m
    assert fallback.wave > 0
    live = nl._tuning(torch.device("cuda", 0))
    expected = nl._ARCH_TUNING.get((props.major, props.minor), nl._DEFAULT_TUNING)
    assert dataclasses.replace(live, wave=0) == expected
    assert live.wave == props.multi_processor_count * max(1, min(
        props.regs_per_multiprocessor // expected.gemm_block_regs,
        props.shared_memory_per_multiprocessor // expected.gemm_block_smem,
        expected.gemm_max_blocks_per_sm,
    ))


# ------------------------------------------------------------------ real checkpoint
def _real_path() -> Path:
    raw = os.environ.get(_REAL_PATH_ENV)
    if not raw or not Path(raw).is_dir():
        pytest.skip(f"set {_REAL_PATH_ENV} to a local Nemotron-3.5-Lightning checkpoint")
    return Path(raw)


def _load_real(name: str):
    """(packed, block scale, per-row global, per-tensor global) for one real projection."""
    import safetensors
    from freetoken.models.qwen3_5_moe.weight import _nvfp4_parts

    path = _real_path()
    index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    with safetensors.safe_open(
        str(path / index[name + ".weight"]), framework="pt", device="cuda"
    ) as f:
        w, scale, glob = _nvfp4_parts(f, name)
        scale_2 = f.get_tensor(name + ".weight_scale_2")
    return w, scale, glob, scale_2


@pytest.mark.needs_weights
@pytest.mark.parametrize("name,shape", [
    ("backbone.layers.1.mixer.shared_experts.up_proj", SHARED_UP),
    ("backbone.layers.1.mixer.shared_experts.down_proj", SHARED_DOWN),
    ("lm_head", LM_HEAD),
])
@pytest.mark.parametrize("m", [1, 8])
def test_real_lightning_tensors_match_the_debug_reference(name, shape, m):
    from freetoken.kernel.triton.nvfp4_linear import Nvfp4DenseLinear

    n, k = shape
    _need_vram(n, k, m, outputs=2, bf16_weight=True)
    w, scale, glob, _scale_2 = _load_real(name)
    assert w.shape == (n, k // 2)

    linear = Nvfp4DenseLinear(k, n)
    linear.load_state_dict({"weight": w, "weight_scale": scale, "weight_global": glob})
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda") * 0.05
    got = linear.forward(x)
    ref = _reference(x, w, scale, glob)
    assert got.shape == ref.shape
    assert cosine(got, ref) > 0.9999
    if n > 4096:  # real logits: the sampled token id must not move at all
        assert_top1_agrees(got, ref, exact=True)


@pytest.mark.needs_weights
def test_real_lm_head_argmax_survives_the_tuned_launch_configs():
    """The end-to-end invariant that matters for greedy decoding: the sampled token id."""
    from freetoken.kernel.triton import nvfp4_linear as nl

    n, k = LM_HEAD
    _need_vram(n, k, 16, outputs=3, bf16_weight=True)
    w, scale, glob, _s2 = _load_real("lm_head")
    head = nl.Nvfp4LMHead(n, k)
    head.load_state_dict({"weight": w, "weight_scale": scale, "weight_global": glob})
    x = torch.randn(16, k, dtype=torch.bfloat16, device="cuda") * 0.05
    ref = _reference(x, w, scale, glob)
    del w, scale

    saved = dict(nl._TUNING_CACHE)
    try:
        for m in (1, 8, 16):
            a = x[:m]
            nl._TUNING_CACHE.clear()
            arch = nl.nvfp4_dense_linear_t(a, head.weight, head.weight_scale, head.weight_global)
            nl._TUNING_CACHE.clear()
            nl._TUNING_CACHE[torch.cuda.current_device()] = nl._resolve_tuning(
                torch.cuda.current_device(), nl._DEFAULT_TUNING
            )
            fall = nl.nvfp4_dense_linear_t(a, head.weight, head.weight_scale, head.weight_global)
            assert cosine(arch, ref[:m]) > 0.9999
            assert_top1_agrees(arch, ref[:m], exact=True)
            assert_top1_agrees(arch, fall, exact=True)
    finally:
        nl._TUNING_CACHE.clear()
        nl._TUNING_CACHE.update(saved)
