# MoE prefill GEMM — Nemotron-3.5-Lightning-30B-A3B-NVFP4 on one RTX 5080 (2026-09-05)

Ticket §9.2 of `nemotron35_lightning_5080_prefill_profile_2026-09-05.md`: *"MoE prefill is
now the flat term and it is 33 TFLOP/s — 29.46 ms per layer at M=8192 (981 GFLOP), 79 % of
the position-independent cost, and the whole of a short-prompt prefill."*

**Answer: 29.47 → 16.95 ms per layer, 1.74x, 33 → 57.9 TFLOP/s (28 % → 49 % of the card's
measured Triton `tl.dot` ceiling). It was not the tile. The K-loop was loading one e4m3
block scale *per packed byte*, i.e. every scale eight times — one scale covers 16 k-values
= 8 bytes — and that single redundancy was 1.73x on its own.** Loading the distinct
`[BLOCK_KB/8, BLOCK_N]` scale rows and broadcasting them also drops the kernel's shared
memory from 28 KB to 12 KB, which is what makes a much wider `BLOCK_N` affordable: the
tuner, re-run with `BLOCK_N=256` and `BLOCK_KB=16` newly reachable, moves the M=8192 tile
from `128/128/32/1/4/4` to `64/256/16/8/4/3`.

A second, smaller term: `cvt.rn.f16x2.e2m1x2`, the Blackwell hardware FP4 decode — the same
instruction flashinfer's b12x kernel uses — replaces `_e2m1_decode`'s ~14-op bit
construction. It is **bit-identical for all 256 packed byte values** and worth a further
1.04x. Every configuration in every sweep below reproduces the tree kernel's output
**exactly** (`max|Δ| = 0.000e+00`), so this whole change is a scheduling change.

The b12x "nan at M=8192" in the ticket is a **reporting artifact**, not a bug: the
comparison table in `bench_nvfp4_moe_kernels.py` prints `nan` in every column when a backend
contributed no rows. Re-run on this host b12x is fine — and the retuned Triton kernel now
**beats it at M=256 (1.43x) and at every decode width (1.6–1.9x)**, and is 1.34x behind only
at M=8192. Triton stays the single recommended backend; §6.

GPU 0 MiB at the end. Nothing committed.

---

## 1. The denominators

From `nemotron35_lightning_5080_prefill_q8_2026-09-05.md` §1, measured on this card:
**123.0 TFLOP/s** cuBLAS bf16, **118.4 TFLOP/s** Triton `tl.dot`, 128.0 TOP/s int8. The
225 TFLOP/s of the spec sheet is not a number this GPU produces in any dtype.

The pair of expert GEMMs at M=8192, top-6, H=2688, I=1856 is `2 · 8192 · 6 · 2 · H · I` =
**981 GFLOP**. `moe_align_block_size` pads to `sum_e ceil(n_e/BLOCK_M)·BLOCK_M`, +8.5 % at
`BLOCK_M=64` and +17.2 % at 128, so the kernel's own arithmetic is ~1.09x that.

`benchmarks/bench_moe_prefill_gemm.py` (new) reports TFLOP/s against those ceilings.
`bench_nvfp4_moe_kernels.py` reports % of the 960 GB/s **HBM roofline**, which is the right
figure of merit for decode (weight-stream bound: an M=1 step reads 32 MiB of experts and
does 12 GFLOP) and a misleading one at M=8192, where each expert is read once per M-block
and the kernel is arithmetic bound. Both are kept.

## 2. The tile space was already exhausted

54 configurations of the **unmodified** kernel at M=8192 (`BLOCK_M` 64/128/256 x `BLOCK_N`
64/128/256 x `BLOCK_KB` 32/64/128 x 4/8 warps, uniform routing, median of 5, every one
verified against the shipped tile); 14 exceeded the 101,376-byte opt-in shared memory and
did not compile, 40 were timed:

| BLOCK_M/N/KB/G/warps/stages | ms | TFLOP/s | % of `tl.dot` |
|---|---:|---:|---:|
| **128/256/32/1/8/3** | **26.43** | 37.1 | 31 % |
| 64/256/64/1/8/3 | 28.40 | 34.5 | 29 % |
| 64/256/32/1/8/3 | 29.23 | 33.6 | 28 % |
| 128/128/32/1/4/3 *(the shipped tile at 3 stages)* | 29.81 | 32.9 | 28 % |
| 64/128/64/1/4/3 | 30.21 | 32.5 | 27 % |
| … 35 more, down to 351 ms | | | |

The shipped tile itself (`128/128/32/1/4/4`) measures **29.79 ms** in the same session and
29.47 in the isolated decomposition below, so **1.12x is all the tile space had**, against
the 3.6x the ceiling says is missing. The tuner's grid (`BLOCK_N ≤ 128`,
`BLOCK_KB ∈ {32,64}`) had already found its best point inside it.

Decomposing the shipped launch (`scratchpad/moe/probe.py`): `moe_align_block_size` 0.019 ms,
gemm1 14.86, gemm2 14.43, `moe_sum_reduce` 0.344 — **the two grouped GEMMs are 99 % of it**,
and they are symmetric, so there is one problem, not two.

## 3. What it actually was — an ablation, not a hypothesis

Four independent changes to a standalone copy of the kernel (`scratchpad/moe/variants.py`),
selected by a `FAST` bitmask, at the tile that turned out to win (`64/256/16/1/4/3`),
M=8192, pair ms, median of 9:

| FAST | what | pair ms | vs the same tile | max abs Δ | n_regs / spills / smem |
|---:|---|---:|---:|---:|---|
| 0 | copy of the tree kernel (control) | 29.87 | 1.00x | 0.000e+00 | 255 / 0 / 28,672 |
| 1 | native `cvt.rn.f16x2.e2m1x2` | 26.93 | 1.11x | 0.000e+00 | 239 / 0 / 28,672 |
| **2** | **scale loaded per 8-byte group, broadcast** | **17.31** | **1.73x** | 0.000e+00 | 241 / 0 / **12,288** |
| **3** | **both** | **16.67** | **1.79x** | 0.000e+00 | 246 / 0 / 12,288 |
| 7 | + the `e2m1 · scale` product in fp16 | 16.80 | 1.78x | 0.000e+00 | 237 / 0 / 12,288 |
| 15 | + `b_lo`/`b_hi` dequantized one at a time | 16.80 | 1.78x | 0.000e+00 | 237 / 0 / 12,288 |
| **16** | **no scale applied at all (ABLATION, wrong)** | **17.41** | 1.72x | 6.9e+03 | 246 / 0 / 12,288 |

Two things fall straight out of the last row. **FAST=3 is already at the bound where the
scale is free** — 16.67 ms against 17.41 ms for a kernel that does not load or apply a scale
at all — so there is nothing left on that axis. And the 1.73x was never *arithmetic*: it was
the 8x-redundant **load**, which Triton was staging through 16 KB of shared memory per CTA.

The redundancy, in the code that was there:

```python
sblk = byte_idx // 8                     # byte_idx is [BLOCK_KB]
s_ptrs = scale_base + sblk[:, None] * stride_sblk
scale = tl.load(s_ptrs, ...)             # [BLOCK_KB, BLOCK_N]  <- 8 identical rows in every 8
```

and what replaces it:

```python
NGRP: tl.constexpr = BLOCK_SIZE_KB // 8
grp_idx = kb * NGRP + tl.arange(0, NGRP)
s_grp = tl.load(scale_base + grp_idx[:, None] * stride_sblk, ...)   # [NGRP, BLOCK_N]
scale = tl.reshape(tl.broadcast_to(s_grp[:, None, :], (NGRP, 8, BLOCK_SIZE_N)),
                   (BLOCK_SIZE_KB, BLOCK_SIZE_N))
```

*(The general rule, and it is the same one the q8 attention write-up needed from the other
side: **a per-block quantization scale is a `K/QBLOCK`-sized tensor, so load it at that size
and broadcast — never at the tile's K.** The dequant multiply has to happen at the tile's K;
the load does not.)*

### 3.1 The hardware FP4 decode

`cvt.rn.f16x2.e2m1x2` (PTX ISA 8.6, Blackwell — sm_100 datacenter and sm_120 consumer alike)
turns one packed NVFP4 byte into the `f16x2` pair of its two codes in a single instruction:

```
{ .reg .b8 b0; .reg .b32 h2;
  cvt.u8.u32 b0, $2;
  cvt.rn.f16x2.e2m1x2 h2, b0;
  mov.b32 {$0, $1}, h2; }
```

Checked exhaustively against `_e2m1_decode` over all 256 byte values, bitwise **and** on the
sign bit (code 8 must stay `-0.0`): identical. So `e2m1_native_cvt_cx()` is a pure scheduling
branch, and `FREETOKEN_NVFP4_NO_NATIVE_CVT=1` forces the arithmetic form for an A/B. Every
value the kernel can produce is unchanged, because the product `e2m1 · e4m3` needs 7
significant bits and the operand is rounded to bf16 (8) either way.

## 4. The retuned tiles

`benchmarks/tune_nvfp4_moe.py` with `BLOCK_N=256` and `BLOCK_KB=16` added to the grid (both
unreachable before the scale fix: at `BLOCK_KB=16` the old kernel loaded a `[16, N]` scale
tile for 2 distinct rows) and `--merge` so the table can be rebuilt in passes. It picks
`64/256/16/8/4/3` for M=4096 and M=8192 **independently of the hand sweep above**, which is
the check that this is a property of the kernel and not a curve fit.

Shipped table, before → after, and what the tuner now writes:

| M bucket | tile before | tile after | ms before | ms after | after TFLOP/s | % `tl.dot` |
|---:|---|---|---:|---:|---:|---:|
| 256 | 16/64/64/1/4/3 | 16/64/64/1/4/4 | 1.82 | **1.275** | 24.0 | 20 % |
| 1024 | 64/128/64/1/4/3 | 16/256/64/8/4/3 | — | **2.942** | 41.7 | 35 % |
| 2048 *(snaps to 1024)* | 64/128/64/1/4/3 | 16/256/64/8/4/3 | 8.47 | **5.107** | 48.0 | 41 % |
| 4096 | 64/128/64/1/4/3 | **64/256/16/8/4/3** | — | **9.203** | 53.3 | 45 % |
| **8192** | 128/128/32/1/4/4 | **64/256/16/8/4/3** | **29.47** | **16.95** | **57.9** | **49 %** |

(“ms before” at 256/2048/8192 are the figures recorded in `bench_nvfp4_moe_kernels.py`'s
docstring and `nvfp4_moe_kernels_5080.jsonl` for the same geometry and routing draw count.)

### 4.1 Routing skew does not move the tile

The ticket asked for the real routing distribution. It does not matter at this M, and that
is measurable rather than assumed — Dirichlet-α expert probabilities at M=8192, shipped tile:

| α | distinct experts | padded rows | ms |
|---:|---:|---:|---:|
| 0.25 (heavily skewed) | 116 | 53,632 (+9.1 %) | 16.97 |
| 1.0 | 128 | 53,760 (+9.4 %) | 17.10 |
| 4.0 | 128 | 53,056 (+7.9 %) | 16.91 |
| uniform | 128 | 53,312 (+8.5 %) | 17.01 |

**0.7 % spread.** At 49,152 routed rows over 128 experts every expert gets hundreds of rows
whatever the skew, so `sum_e ceil(n_e/64)·64` barely moves and no CTA is short of work. The
`--routing-file` replay path is in the harness for the small-M buckets, where it can matter.

## 5. End to end — 131K and 262K, two servers, one A/B

Two servers, same flags, same prompts, same box minutes apart; the **before** arm is a
detached `git worktree` at `e4070da` (`PYTHONPATH=<worktree>/python`, the built `kernel/*.so`
copied in), so it carries the old kernel *and* the old config table. Synthetic needle at
depth 0.50, chat endpoint, `--max-prefill-length 8192`, `--kv-cache-dtype q8_0`,
`--memory-ratio 0.80`, `--nvfp4-backend triton`, one request per length, cold prefix cache.

| prompt tokens | chunks | TTFT before | TTFT after | prefill before | prefill after | speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 131,088 | 16 | 26.69 s | **22.29 s** | 4,910.8 tok/s | **5,881.7 tok/s** | **1.198x** |
| 262,160 | 32 | 75.31 s | **66.60 s** | 3,481.3 tok/s | **3,936.3 tok/s** | **1.131x** |

**The engine sees exactly the kernel's number.** Δ TTFT / chunk count is
4.40 s / 16 = **275 ms** at 131K and 8.71 s / 32 = **272 ms** at 262K, against the kernel
microbench's prediction of `23 × (29.47 − 16.95) = 288 ms` per chunk — 95 % of it, with no
in-engine instrumentation. The residual is inside run-to-run spread on a single request.

The flat per-chunk term of the 2026-09-05 profile therefore goes **861 → ~586 ms**, of which
MoE is now 390 ms (67 %) instead of 678 (79 %).

### 5.1 Quality

| | before | after |
|---|---|---|
| needle @131,088 (depth 0.50) | recalled | recalled |
| needle @262,160 (depth 0.50) | recalled | recalled |
| answer text, 131K and 262K | — | **byte-identical to the before arm** |
| 64-token greedy continuation, short prompt | — | **byte-identical** |

Byte-identical output is the *expected* result here, not a lucky one: the ablation table in
§3 shows `max|Δ| = 0.000e+00` against the old kernel at every tile, because the scale
broadcast reads the same values and the hardware FP4 decode produces the same 16 codes. This
change is not held to an "agreement" gate the way the 2026-09-05 extend-tile change was — the
tile changes the order of nothing, since each `tl.dot` still reduces the same K in the same
order and the accumulator is per-CTA.

### 5.2 Where the win lands

The MoE GEMM is the *flat* term, so the speedup is inversely proportional to prompt length:

| prompt | flat share before | speedup |
|---|---:|---:|
| ≤ 8,192 (one chunk) | ~100 % | **~1.47x** (projected) |
| 131,088 | 52 % | **1.198x** (measured) |
| 262,160 | 37 % | **1.131x** (measured) |
| 1,040,016 | 14 % | ~1.05x (projected: 795.8 → ~761 s) |

## 6. `--nvfp4-backend`: triton stays the default, and now by a wider margin

Re-measured on this host after the retune, `bench_nvfp4_moe_kernels.py --routings 8`
(the b12x rows are unchanged from 2026-09-04 within noise, as they must be):

| regime | M | triton before | **triton after** | b12x | after vs b12x |
|---|---:|---:|---:|---:|---:|
| decode | 1 | 57.8 us | 57.7 us | 94.5 us | **triton 1.64x** |
| decode | 8 | — | 301.1 us | 555.3 us | **triton 1.85x** |
| prefill | 256 | 1,819 us | **1,257 us** | 1,797 us | **triton 1.43x** |
| prefill | 2,048 | 8,466 us | **5,023 us** | 3,879 us | b12x 1.29x |
| prefill | 8,192 | 29,472 us | **16,954 us** | 12,618 us | b12x 1.34x |

b12x's prefill lead at M=8192 went **2.34x → 1.34x**, it now *loses* at M=256, and it loses
decode by 1.6–1.9x. Taking it would cost the decode path and the whole bank layout (the
choice is made once at bank-load time — `moe/expert_banks.py:180` — because it pins the
on-GPU weight format; you cannot have both resident). The 2B1 gate in
`bench_nvfp4_moe_kernels.py` ("b12x ≥ 2x triton at M=8/16") is now inverted and stale.

**There was never a nan.** The ticket's "`--nvfp4-backend b12x` did not produce a number in
this sweep (flashinfer path returned `nan`)" is the summary table's `nan` fill-in for a
backend that contributed no rows (`speed = t["us"]/b["us"] if (t and b) else float("nan")`,
`bench_nvfp4_moe_kernels.py:290`). On this host `_b12x_unusable_reason((12, 0))` is `None`,
flashinfer 0.6.17 imports, and the b12x arm runs clean at every M above. Nothing to fix.

## 7. `--nvfp4-backend flashinfer` + `--kv-grow-step-tokens` — the VMM int32 bank, fixed

`benchmarks/results/nemotron35_lightning_5080_cache_study_2026-09-04.md` recorded that the
pairing dies at init with `ValueError: unsupported VMM tensor dtype: torch.int32`: growable
KV sets `cache.direct_device_banks`, so the expert slot cache is allocated as `VMMTensor`s,
and the repacked b12x banks include an int32 bank. It is exactly the two-line fix that write-up
named — `int16`/`int32`/`int64` in `VMMTensor._DTYPE_NAMES` and in `parse_dtype`
(`kernel/csrc/vmm_tensor.cpp`), which must stay in step or the failure is identical.

Verified end to end: `--nvfp4-backend flashinfer --kv-grow-step-tokens 32768
--kv-cache-dtype q8_0 --num-tokens 131072` now starts and answers (*"The three primary
colors are **red**, **blue**, and **yellow**."*, 38.3 s including the b12x repack and the
CuTe-DSL JIT on the first request). `tests/kernels/test_vmm_tensor.py` pins the three
integer dtypes through a real reserve/commit/write cycle.

This does not change the recommendation in §6 — triton is still the pick — but the
combination is no longer a startup error, and the doc note that said it "cannot be
combined" is now wrong and has been updated.

## 8. Files

- `python/freetoken/kernel/triton/nvfp4_fused_moe.py` — the scale broadcast in
  `_prefill_nvfp4_moe_kernel`'s K-loop, `_e2m1_decode_pair_native`, `e2m1_native_cvt_cx`,
  `FREETOKEN_NVFP4_NO_NATIVE_CVT`.
- `python/freetoken/moe/fused_nvfp4.py` — `_PREFILL_ENV_KEYS` /
  `_prefill_launch_env_override` (`FREETOKEN_NVFP4_PREFILL_*`).
- `python/freetoken/moe/configs/triton_3_6_0/nvfp4,E=128,N={1856,2688},K={2688,1856},
  device_name=NVIDIA_GeForce_RTX_5080.json` — retuned, all six M buckets.
- `python/freetoken/kernel/vmm.py`, `python/freetoken/kernel/csrc/vmm_tensor.cpp` — the
  integer VMM dtypes.
- `benchmarks/bench_moe_prefill_gemm.py` (new) — the sweep harness used here.
- `benchmarks/tune_nvfp4_moe.py` — `BLOCK_N=256`, `BLOCK_KB=16`, `--merge`.
- `tests/moe/test_nvfp4_backends.py` — the native-cvt exhaustive check, the dequant
  reference at `BLOCK_KB` 16/32/64 (`NGRP` 2/4/8).
- `tests/moe/test_nvfp4_triton_tuning.py` — the env override, and `BLOCK_SIZE_KB % 8 == 0`
  and `>= 16` for every shipped table and the heuristic fallback.
- `tests/kernels/test_vmm_tensor.py` — the integer dtypes.
- `docs/nemotron.md`, `tasks/todo.md`, `tasks/lessons.md`.

193 tests pass across `tests/moe` + `tests/kernels/test_vmm_tensor.py` (5 skipped) on the 5080.

## 9. Reproduction

```
# the tile sweep and the shipped-config table (never pipe gpu_lock.sh -- redirect)
PYTHONPATH=python scripts/gpu_lock.sh .venv/bin/python -u \
  benchmarks/bench_moe_prefill_gemm.py --m 256 1024 2048 4096 8192 --grid shipped \
  > shipped.log 2>&1
PYTHONPATH=python scripts/gpu_lock.sh .venv/bin/python -u \
  benchmarks/bench_moe_prefill_gemm.py --m 8192 --verify --grid-json \
  '{"BLOCK_SIZE_M":[64,128,256],"BLOCK_SIZE_N":[64,128,256],"BLOCK_SIZE_KB":[16,32,64],
    "GROUP_SIZE_M":[1],"num_warps":[4,8],"num_stages":[3]}' > sweep.log 2>&1

# re-tune (about 16 minutes; --merge lets it be split across passes)
PYTHONPATH=python scripts/gpu_lock.sh .venv/bin/python -u benchmarks/tune_nvfp4_moe.py \
  --prefill --prefill-m 8192 4096 1024 256 64 16 --merge --write > tune.log 2>&1

# force the pre-change tile inside the new binary (the K-loop rewrite needs the worktree)
FREETOKEN_NVFP4_PREFILL_BLOCK_M=128 FREETOKEN_NVFP4_PREFILL_BLOCK_N=128 \
FREETOKEN_NVFP4_PREFILL_BLOCK_KB=32 FREETOKEN_NVFP4_PREFILL_GROUP_M=1 \
FREETOKEN_NVFP4_PREFILL_NUM_WARPS=4 FREETOKEN_NVFP4_PREFILL_NUM_STAGES=4 ft serve ...
# force the arithmetic FP4 decode:  FREETOKEN_NVFP4_NO_NATIVE_CVT=1
```

Drivers: `scratchpad/moe/{probe.py,variants.py,runv.py,e2e.py,fi.py,ptxtest.py}` under
`/tmp/claude-1000/-home-lucas-ai-FreeToken/f4e2e9e3-f4f5-40d0-9980-b3b09d1ef47d/scratchpad/`.

**Do not pipe `scripts/gpu_lock.sh`.** Its exit trap runs `pkill -9 -g $$`; it also kills the
job before Python flushes a buffered stdout, so the wrapped script must `exec >` its own log
(the first tuner run completed and wrote its tables but left a 0-byte log).

## 10. Still open — tickets

1. **b12x is still 1.34x on the M=8192 grouped GEMM (12.62 vs 16.95 ms).** The remaining gap
   is the operand path, not the dequant: the `FAST=16` ablation (no scale loaded or applied
   at all) is 17.41 ms, i.e. *slower* than the shipped kernel, so nothing is left on that
   axis. b12x reads a **pre-swizzled tensor-core fragment layout**; the Triton kernel reads
   ModelOpt rows and pays a shared-memory staging pass for the B operand. Closing it inside
   Triton means changing the bank layout, which is a load-time global decision that also
   costs the decode GEMV. Not worth it at the measured 1.34x on a term that is 67 % of a
   flat cost that is itself 14 % of a 1M prefill.
2. **The activation operand is loaded twice, strided.** `a_ptrs_lo`/`a_ptrs_hi` read the even
   and odd k of the same span at a 2-element stride, so neither load vectorizes. A one-off
   deinterleave of `A` per GEMM (44 MB for gemm1, 182 MB for gemm2 — ~0.5 ms at 960 GB/s)
   would make both contiguous. Untested; it is the most likely remaining Triton-side win and
   it needs no layout change to the *weights*.
3. **The decode GEMVs load their block scale the same redundant way.** `_decode_nvfp4_moe_kernel`
   and the marlin variant were not touched. Decode is HBM-bound with idle LSU capacity (the
   `_e2m1_decode` docstring measured the arithmetic decode as a *loss* there), so the fix may
   be worth nothing — but it is the same three lines and it is now measurable in one run.
4. **The 2B1 gate in `bench_nvfp4_moe_kernels.py` is stale and inverted.** It still asserts
   "b12x ≥ 70 % of roofline at M=1" and "b12x ≥ 2x triton at M=8/16"; triton now wins both by
   1.6–1.9x, so `--gate` fails on a healthy tree. Rewrite it as a triton floor.
5. **Small-M buckets are the weakest.** M=256 runs at 20 % of the `tl.dot` ceiling with
   +53 % padding waste at `BLOCK_M=16`. The scheduler's interleave share produces 512-token
   chunks under load, so this bucket is not hypothetical; `--routing-file` in the new harness
   exists to replay a real capture there, which is where routing skew *can* matter.
