# Native-Q8 extend attention — the ticket, measured and closed NEGATIVE (2026-09-05)

Ticket (§9.1 of `nemotron35_lightning_5080_prefill_profile_2026-09-05.md`): *"the extend
kernel is at 31 % of peak even after the fix … the q8_0 prefix is dequantized to bf16 inside
the inner loop … a native-Q8 QK for extend is plausibly worth another 1.5–2x, i.e. 1M TTFT
~470–530 s."*

**Answer: no. The kernel is at 57 % of this card's real bf16 peak, not 31 %; int8 tensor
throughput on sm_120 is 1.04x bf16, not 2x; and removing the q8_0 dequant *entirely* — measured
by running the same kernel over a bf16 KV pool — is worth 1.206x, at every prefix length. A
native-Q8 QK can reach only part of that, and the scale-fold arithmetic it requires is 2x more
work than the dequant it removes at this tile. The kernel is unchanged.**

The best combination of dequant-side optimisations I could find is **1.14x**, below the 1.2x
bar, and it is **2.0x worse against the fp32 oracle** than the kernel in the tree
(7.6e-4 vs 3.7e-4). It is recorded in §5 with its code, and not shipped.

GPU 0 MiB at the end. Nothing committed. No model was loaded: every number here is a kernel
microbenchmark, which is why this cost ~35 GPU minutes instead of ~2 GPU hours.

---

## 1. The ticket's denominator was the marketing number

`225 TFLOP/s bf16` is the RTX 5080's spec-sheet FP16/BF16 tensor figure. What the part
actually delivers, measured (`scratchpad/q8/peak.py`, `tritonpeak.py`):

| what | measured |
|---|---:|
| cuBLAS bf16 GEMM, n=8192 | **123.0 TFLOP/s** |
| cuBLAS fp16 GEMM, n=8192 | 122.6 TFLOP/s |
| cuBLAS **int8** GEMM (`torch._int_mm`), n=8192 | **128.0 TOP/s** |
| Triton `tl.dot` bf16 GEMM, best of 5 configs (64/64/64, 4 warps, 2 stages) | **118.4 TFLOP/s** |
| Triton `tl.dot` bf16, 128/128/64, 8 warps, 3 stages | 106.7 TFLOP/s |

So the honest denominators are 123 TFLOP/s (hardware) and 118 TFLOP/s (Triton's own codegen,
the like-for-like ceiling for this kernel). Against those:

| kernel | TFLOP/s @131K | of cuBLAS peak | of Triton `tl.dot` peak |
|---|---:|---:|---:|
| extend, q8_0 KV (tree HEAD) | 70.7 | 57 % | **60 %** |
| extend, bf16 KV (same kernel, no dequant) | 85.6 | 70 % | **72 %** |

A flash-attention kernel at 72 % of the achievable GEMM rate is at its structural limit: the
softmax tax (`exp`, row max, row sum, accumulator rescale on `BLOCK_M x BLOCK_N` elements
while the dots do `BLOCK_M x BLOCK_N x D`) is intrinsic and does not go away. **There was
never a 1.5–2x here.** The "31 % of peak" in the ticket is 70.4/225, and 225 is not a number
this GPU produces in any dtype through any library.

**int8 is not faster than bf16 on this part.** 128.0 TOP/s vs 123.0 TFLOP/s = **1.04x**. The
2x int8:bf16 tensor ratio the ticket implicitly assumes is a datacenter-part property; the
consumer Blackwell `mma.sync` path Triton emits has essentially the same rate for both. So a
native int8 QK cannot make the dot cheaper — the *only* thing it can win is dequant ALU.

## 2. How much dequant ALU there is to win: 1.206x, and it is flat in length

The decisive control needs no new kernel: run **the same production kernel, same launch, same
shapes, over a bf16 KV pool**. A bf16 pool does zero dequantization (it has 2x the bytes, and
the microbench shows 0.6 GB/s of unique DRAM traffic on a 960 GB/s part, so the extra traffic
is free). The gap is therefore the entire cost of the q8_0 path — an upper bound on any
native-Q8 or dequant-elimination work, including one that is *perfect*.

`bench_prefill_attention.py` geometry (32Q / 2KV / D128, chunk 8192, tile 64/64/4/1),
ms per attention layer, median of 5:

| prefix | q8_0 (tree HEAD) | bf16 KV | **q8 / bf16** | q8 TFLOP/s | bf16 TFLOP/s |
|---:|---:|---:|---:|---:|---:|
| 131,072 | 256.55 | 212.01 | **1.210** | 70.7 | 85.6 |
| 262,144 | 504.62 | 419.77 | **1.202** | 70.8 | 85.1 |
| 524,288 | 1007.65 | 834.08 | **1.208** | 70.4 | 85.0 |
| 1,048,576 | 2005.08 | 1662.74 | **1.206** | 70.5 | 85.0 |

**1.206 ± 0.004x, dead flat in prefix.** That is the ceiling. The ticket asked for 1.5–2x.

## 3. Why a native-Q8 extend kernel cannot even reach that ceiling

q8_0 stores **one fp16 scale per 32 elements along `head_dim`**, per token, per head
(`kvcache/quant.py:26`, `:109-119`; pool `[slots, heads, 128]` int8 + `[slots, heads, 4]`
fp16). With `D = 128` that is 4 scale blocks per (token, head).

### 3.1 QK — the scale sits inside the reduction, so folding it costs *more* than dequantizing

```
scores[m,n] = SUM_b  s_k[n,b] * SUM_{d in block b} q[m,d] * k_int[d,n]
```

The scale varies along `d`, the reduction dimension, so it cannot be pulled out of a single
`tl.dot`. The only native form is `D/QBLOCK` separate int8 dots with the scale applied to each
`[BLOCK_M, BLOCK_N]` int32 partial — exactly what `_Q8_NATIVE_QK` does in decode
(`attention.py:847-890`). Per KV tile:

| | multiplies per KV tile | at 64/64/4/1, D=128 |
|---|---|---:|
| dequantize K in place (today) | `BLOCK_D * BLOCK_N` | **8,192** |
| fold the scale after `D/QBLOCK` int8 dots | `BLOCK_M * BLOCK_N * BLOCK_D/QBLOCK` | **16,384** |

Folding is cheaper **iff `BLOCK_M < QBLOCK`**. Decode has `BLOCK_M = BLOCK_H = 16` query heads
against `QBLOCK = 32` — folding is 2x cheaper there, which is why `_Q8_NATIVE_QK` pays and was
built. The extend kernel's `BLOCK_M` is **64** tokens, so folding is **2x more expensive**, and
it also replaces 8,192 int8→fp32 converts with 16,384 int32→fp32 converts. The decode path's
absence from extend is not an oversight; the arithmetic reverses at `BLOCK_M > QBLOCK`.

*(This is the general rule the ticket needed: **fold a per-block scale after the dot only when
the tile's row count is below the quant block size.**)*

### 3.2 PV — the V scale varies along the reduction dimension and cannot be folded at all

```
acc[m,d] = SUM_n  p[m,n] * v_int[n,d] * s_v[n, d/32]
```

`s_v` depends on `n`, which *is* the PV reduction axis. There is no post-dot fold. The only
restructuring is to pre-scale `p` once per d-block — `BLOCK_M * BLOCK_N * D/QBLOCK` = 16,384
multiplies against 8,192 for dequantizing V. Same 2x loss. **The V half of the 1.206x is
unreachable by any native-Q8 formulation.**

So a native-Q8 extend kernel would address at best the K half of a 1.206x gap while adding
arithmetic — i.e. under 1.10x if it broke even on everything else, and plausibly a loss.
**Option (a) of the ticket is dead on the desk.**

### 3.3 Option (b), software-pipelined dequant, is dead too

`num_stages > 1` was already 0-for-66 in the 2026-09-04 sweep, but that sweep ran with
loop-variant masks on every KV load, which inhibits Triton's pipeliner. §5 re-swept the whole
launch space **with the masks removed** (the pipeliner's precondition): `num_stages = 1` still
wins at every one of the 18 tiles that compile, most of them by 20–60 %. It is a shared-memory
and occupancy trade, not a latency one — 64/64 at 2 stages needs 57 KB of the 100 KB opt-in
budget and drops the SM to one CTA.

## 4. What the 1M projection would have been

From the profile's decomposition (flat term 109 s, attention term 689 s after `4a99e34`):

| scenario | attention term | 1M TTFT | vs today |
|---|---:|---:|---:|
| today (`4a99e34`, measured) | 689 s | **795.8 s** | 1.00x |
| §5's 1.14x variant (accuracy-regressing, not shipped) | 604 s | 713 s | 1.12x |
| the 1.206x ceiling — *bf16 KV, i.e. 2x the KV memory* | 571 s | 680 s | 1.17x |
| **the ticket's claim** | 345–460 s | **470–530 s** | 1.5–1.7x |

Even a *perfect* native-Q8 kernel lands at 680 s, and the ticket's 470–530 s requires the
kernel to exceed the achievable bf16 GEMM rate of the card.

## 5. The 1.14x that does exist, and why it is not shipped

Three changes to the prefix loop, tried as a standalone copy of
`_extend_attention_split_kernel` (`scratchpad/q8/variants.py`, q8_0 only, no sinks, no SWA):

- **`split`** — split the prefix loop into `prefix_len // BLOCK_N` **fully-valid** tiles plus a
  ragged tail. Full tiles need no `mask_n`, no `tl.where(final_mask, scores, -inf)`, no
  `row_max == -inf` fixup and no masked loads. (Rows past `q_len` need no mask either: `q` is
  loaded with `other=0.0`, so an invalid row scores 0 and is discarded at the store.)
- **`exp2`** — carry `m_i` in the log2 domain and use `tl.exp2`, folding `sm_scale * log2(e)`
  into the multiply `tl.exp` performs internally anyway.
- **`bf16deq`** — dequantize int8 → bf16 with a bf16 multiply, instead of
  int8 → fp32 → fp32 multiply → bf16 (two quarter-rate converts and an fp32 multiply per element).
- **`qfold`** — (needs `exp2`) scale `q` by `sm_scale*log2(e)` once, so the score tile is never
  scaled.

Prefix 131,072, chunk 8,192, tile 64/64/4/1, median of 7. `max|Δ|` is against the production
kernel's own output:

| variant | ms | speedup | max abs Δ vs prod | n_regs | n_spills |
|---|---:|---:|---:|---:|---:|
| production (tree HEAD) | 256.87 | 1.00x | — | 255 | 28 |
| copy of it (`FAST=0`, control) | 260.55 | 0.99x | 0.00e+00 | 255 | 28 |
| `split` | 255.70 | 1.00x | 1.22e-04 | 255 | 30 |
| `exp2` | 254.80 | 1.01x | 1.22e-04 | 255 | 28 |
| `bf16deq` | 248.41 | 1.03x | 1.83e-04 | 255 | 18 |
| `split+exp2` | 244.62 | 1.05x | 1.22e-04 | 255 | 32 |
| `exp2+bf16deq` | 241.25 | 1.06x | 1.83e-04 | 255 | 22 |
| `split+bf16deq` | 231.42 | 1.11x | 1.83e-04 | 255 | 26 |
| `split+exp2+bf16deq` | 228.28 | 1.13x | 1.83e-04 | 255 | 16 |
| **`split+exp2+bf16deq+qfold`** | **225.86** | **1.14x** | 1.53e-04 | 255 | 26 |

Flat in prefix, like the ceiling:

| prefix | prod ms | `FAST=15` ms | speedup | TFLOP/s |
|---:|---:|---:|---:|---:|
| 131,072 | 256.55 | 225.12 | 1.140x | 80.6 |
| 262,144 | 504.62 | 443.14 | 1.139x | 80.6 |
| 524,288 | 1007.65 | 881.51 | 1.143x | 80.5 |
| 1,048,576 | 2005.08 | 1754.79 | 1.143x | 80.5 |

**It fails the accuracy gate.** Against an fp32 oracle built by dequantizing the pool and doing
the whole attention in fp32 torch (prefix 8,192, chunk 1,024, 32Q/2KV/D128):

| kernel | max abs Δ vs fp32 oracle | mean abs Δ |
|---|---:|---:|
| **production (tree HEAD)** | **3.7293e-04** | 4.3262e-05 |
| copy of it (`FAST=0`) | 3.7293e-04 | 4.3262e-05 |
| `split+exp2+bf16deq` | 7.5779e-04 | 5.6347e-05 |
| `split+exp2+bf16deq+qfold` | 7.9688e-04 | 5.9462e-05 |

The gate was *"max |Δ| vs the dequantized-pool fp32 oracle no worse than the current kernel's"*.
2.0x worse fails it. The cost is `bf16deq`: q8_0's scale is fp16 (10 mantissa bits) and bf16
holds 7, so rounding the scale into bf16 puts a systematic ~0.4 % relative error on a whole
32-element block, on top of the product rounding the fp32 path pays once. The accuracy-neutral
subset (`split+exp2`, max |Δ| 1.22e-04 vs production) is worth **1.05x** on its own — not
enough to justify touching a kernel that six tests and a 1M needle run depend on.

### 5.1 The launch space is exhausted, with or without the masks

54 configurations of the lean (`FAST=15`) kernel, prefix 131,072, against production's 256.84 ms.
Best 8:

| BLOCK_M/N/warps/stages | ms | vs prod | n_spills | smem |
|---|---:|---:|---:|---:|
| **64/64/4/1** | **226.15** | **1.14x** | 26 | 40,960 |
| 32/128/4/1 | 258.43 | 0.99x | 34 | 49,152 |
| 128/64/8/2 | 303.74 | 0.85x | 74 | 82,432 |
| 64/128/8/2 | 305.63 | 0.84x | 10 | 98,304 |
| 64/128/8/1 | 307.43 | 0.84x | 8 | 65,536 |
| 128/64/8/3 | 315.33 | 0.81x | 54 | 98,304 |
| 64/32/4/2 | 316.44 | 0.81x | 8 | 45,056 |
| 128/32/8/2 | 326.68 | 0.79x | 8 | 73,728 |

`64/64/4/1` — the tile `4a99e34` picked — wins by 14 % over the runner-up even after the
kernel body changed, `num_stages > 1` never wins, and every 8-warp configuration loses. The
2026-09-05 register-budget rule survives a re-sweep of a differently-shaped kernel, which is
the check that it is a rule and not a curve fit.

## 6. What is actually left on this kernel

`70.5 → 85.0 TFLOP/s` (dequant removed) `→ 118 TFLOP/s` (Triton's own bf16 GEMM ceiling). The
remaining 1.39x between the bf16-KV kernel and a pure GEMM is softmax and occupancy, and the
occupancy half is not addressable by tiling: at `BLOCK_M=64 / 4 warps` the kernel uses 255
registers and 40 KB of shared memory per CTA, so 2 CTAs (8 warps) fit per SM — **2 warps per
scheduler, no latency hiding** — and every configuration that lowers register pressure lowers
rows-per-warp faster (§5.1, and the 66-config sweep of 2026-09-05).

Things that were considered and rejected on the desk, recorded so they are not re-litigated:

- **GQA head merging** (16 q heads share each KV head; a CTA that handled 2 q heads would
  halve the per-tile KV load and dequant). The load+dequant is ~810 SM-cycles of the ~5,900
  per tile after §5's changes; halving it is ~7 %, and it doubles the accumulator.
- **Bigger `BLOCK_M` to amortize the dequant** (dequant per tile is `BLOCK_D x BLOCK_N`,
  independent of `BLOCK_M`; tensor work is proportional to it). Measured: 128-row tiles spill
  250–806 slots at 4 warps and lose 1.4–2.8x. This is the 2026-09-05 finding, re-confirmed.
- **`ncu` pipe-utilisation counters.** Unavailable: `ERR_NVGPUCTRPERM` on this host (profiling
  counters are admin-restricted and there is no root). The bf16-KV control in §2 answers the
  same question with a wall clock, and more directly.

The honest ranking of what is left on a 1M prefill after `4a99e34`:

| term | 1M seconds | note |
|---|---:|---|
| attention, at 60 % of the achievable Triton bf16 rate | 689 | ≤ 118 s recoverable, all of it hard |
| MoE routed-expert GEMMs, 23 layers x 29.5 ms x 127 chunks | 86 | **33 TFLOP/s of 123 = 27 % — the open ticket** |
| everything else (flat) | 23 | |

**Ticket §9.2 (MoE prefill at 33 TFLOP/s) is 3.7x off the same measured peak that attention is
1.7x off.** That is where the next prefill work belongs, not here.

## 7. Files

Nothing in the tree changed. Scratchpad drivers under
`/tmp/claude-1000/-home-lucas-ai-FreeToken/f4e2e9e3-f4f5-40d0-9980-b3b09d1ef47d/scratchpad/q8/`:
`peak.py` (cuBLAS/int8 peak), `tritonpeak.py` (Triton `tl.dot` peak), `variants.py` (the
`FAST` bitmask kernel), `run_variants.py`, `sweep_variants.py`, `lengths.py`, `oracle.py`.

`tests/kernels/test_triton_attention.py`: 55 passed on the unmodified tree.

## 8. Reproduction

```
# the ceiling: same kernel, same launch, q8_0 vs bf16 KV
PYTHONPATH=python python benchmarks/bench_prefill_attention.py \
    --q-heads 32 --kv-heads 2 --head-dim 128 --quant q8_0 --layers 6 \
    --prefix-lens 131072 262144 --chunk 8192 --iters 5 --skip-verify
PYTHONPATH=python python benchmarks/bench_prefill_attention.py \
    --q-heads 32 --kv-heads 2 --head-dim 128 --quant bf16 --layers 6 \
    --prefix-lens 131072 262144 --chunk 8192 --iters 5 --skip-verify
```

Under `scripts/gpu_lock.sh`, redirected to a file — never piped (its exit trap
`pkill -9 -g $$` kills the reader).
