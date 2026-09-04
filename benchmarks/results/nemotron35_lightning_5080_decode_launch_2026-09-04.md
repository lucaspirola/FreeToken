# Decode launch config — Nemotron 3.5 Lightning 30B-A3B-NVFP4 on one RTX 5080 (2026-09-04/05)

Ticket: the `decode_launch_config` item from
`benchmarks/results/nemotron35_lightning_5080_262k_bisect_2026-09-04.md` §"Adjacent
findings" — *"no Nemotron head-shape branch, falls back to `kv_splits=8`: 16 CTAs on 84 SMs
at 262K"*, plus the int64 slot-id widening item next to it.

**Result: single-stream decode is now flat with context.** Measured end to end on one RTX
5080; the first three rows are paired arms of one A/B — the same server binary twice, only
the launch configuration differing — and the 1M row is against the figure recorded on
2026-09-04 (§4):

| prompt tokens | decode before | decode after | speedup |
|---:|---:|---:|---:|
| 131,088 | 82.8 tok/s | **145.3 tok/s** | 1.75x |
| 262,160 | 58.7 | **132.4** | 2.26x |
| 524,304 | 35.4 | **113.6** | 3.21x |
| 1,040,016 | ~20 (recorded, see below) | **95.8** | ~4.8x |

For scale: this model's *short-context* single-stream decode is 143.2 tok/s, so a 131K-token
prompt now costs essentially nothing in decode rate. Prefill is unchanged (within 3 %, both
directions) — the change is confined to the decode kernel launch. The attention kernel itself
is 8.3-9.7x faster per layer and its KV read went from 74 GB/s to 609-722 GB/s of a ~960 GB/s
part.

## 1. Why 8 splits was the whole problem

Nemotron 3.5 Lightning's attention is **32 query heads / 2 KV heads / head_dim 128**, on 6
of its 52 layers (`hybrid_override_pattern` indices 5, 12, 19, 26, 33, 42). The Triton
split-K decode kernel launches

```
grid = (batch, cdiv(num_q_heads, min(16, group)), kv_splits)     # group = q/kv = 16
     = (batch, 2, kv_splits)
```

so the **only** term that scales the decode grid with the GPU is `kv_splits`. Every tuned
branch in `decode_launch_config` required `head_dim == 256, 16 q heads, 2 kv heads` (the
Ornith/Qwen3.5 geometry), so Nemotron took the `_MAX_KV_SPLITS = 8` fallback:

| context | CTAs at 8 splits | SMs busy | KV tokens walked per CTA |
|---:|---:|---:|---:|
| 131,072 | 16 | 16 / 84 | 16,384 |
| 262,144 | 16 | 16 / 84 | 32,768 |
| 524,288 | 16 | 16 / 84 | 65,536 |
| 1,048,576 | 16 | 16 / 84 | 131,072 |

The CTA count is constant, so the per-CTA walk — and therefore the decode step — grows
linearly with context while 81 % of the GPU idles. That is exactly the reported
72 → 56 → 34 → 20 tok/s curve.

The KV traffic bound says how much was left on the table: one attention layer at 1M tokens
reads 1,048,576 × 2 heads × 128 dim × (1 B K + 1 B V) plus q8_0 scales ≈ 290 MiB. At 8
splits the kernel sustained **74 GB/s** of a ~960 GB/s part.

## 2. The rule

`_grid_filling_splits` (new) sizes the split count to the GPU instead of to a constant:

```python
head_blocks = cdiv(num_q_heads, min(16, num_q_heads // num_kv_heads))   # the kernel's own grid
target      = cdiv(sm_count * _DECODE_CTAS_PER_SM, head_blocks)         # _DECODE_CTAS_PER_SM = 1
splits      = clamp(next_pow2(target), _MAX_KV_SPLITS, _MAX_AUTO_KV_SPLITS)   # 8 .. 128
```

On an 84-SM RTX 5080 with 2 head blocks that is **64 splits = 128 CTAs**. It applies only
where no measured branch matched, and only when the caller knows the SM count — direct
kernel callers and CPU devices keep the historical `(8, 32, 4)`, so every existing pin holds.

The tile follows `head_dim`: `(block_n, num_warps) = (64, 8)` at `head_dim <= 128`,
`(32, 4)` above it (measured, §3.3).

Two sanity checks that the rule is not curve-fitted to one model: it reproduces the *measured*
Ornith Blackwell split count (16Q/2KV/D256 also has 2 head blocks → 64 splits, which is what
the `q8_0` and `int4` branches were independently tuned to), and it falls back to the old
constant for MHA-shaped grids (one head block per query head already fills the machine).

**Environment override.** The split count is baked into the CUDA-graph grid and the fp32
scratch at capture time, so it cannot be varied inside a live process.
`FREETOKEN_DECODE_KV_SPLITS` / `_BLOCK_N` / `_NUM_WARPS` force it at startup; setting
`8 / 32 / 4` reproduces the pre-fix fallback exactly, which is how the A/B in §4 was run.

## 3. Kernel microbenchmark

`benchmarks/bench_decode_launch.py` (new) sweeps the three knobs against the production
`decode_paged_attention` entry point — no server, no weights — for an arbitrary head shape.
Median of 15–30 timed calls, clocks settled with `torch.cuda._sleep` before each.

### 3.1 Nemotron geometry, q8_0 KV, batch 1 — ms per attention layer

| splits/block_n/warps | CTAs | 131K | 262K | 524K | 1M |
|---|---:|---:|---:|---:|---:|
| 8 / 32 / 4 | 16 | 0.970 | 1.941 | 3.861 | 7.643 |
| 8 / 32 / 8 | 16 | 0.997 | 1.945 | 3.879 | 7.746 |
| 16 / 32 / 4 | 32 | 0.484 | 0.964 | 1.894 | 3.765 |
| 16 / 32 / 8 | 32 | 0.509 | 0.978 | 1.933 | 3.873 |
| 32 / 32 / 4 | 64 | 0.252 | 0.488 | 0.949 | 1.881 |
| 32 / 32 / 8 | 64 | 0.267 | 0.502 | 0.972 | 1.947 |
| 64 / 32 / 4 | 128 | 0.148 | 0.277 | 0.521 | 1.024 |
| 64 / 32 / 8 | 128 | 0.167 | 0.304 | 0.574 | 1.128 |
| 128 / 32 / 4 | 256 | 0.150 | 0.312 | 0.588 | 1.134 |
| 128 / 32 / 8 | 256 | 0.168 | 0.312 | 0.598 | 1.146 |
| 8 / 64 / 4 | 16 | 0.845 | 1.683 | 3.345 | 6.694 |
| 8 / 64 / 8 | 16 | 0.574 | 1.142 | 2.263 | 4.549 |
| 16 / 64 / 4 | 32 | 0.426 | 0.839 | 1.665 | 3.339 |
| 16 / 64 / 8 | 32 | 0.293 | 0.578 | 1.138 | 2.292 |
| 32 / 64 / 4 | 64 | 0.226 | 0.435 | 0.851 | 1.691 |
| 32 / 64 / 8 | 64 | 0.164 | 0.312 | 0.600 | 1.212 |
| 64 / 64 / 4 | 128 | 0.139 | 0.258 | 0.488 | 0.952 |
| **64 / 64 / 8** | 128 | 0.117 | 0.213 | 0.402 | 0.790 |
| 128 / 64 / 4 | 256 | 0.148 | 0.269 | 0.515 | 0.980 |
| 128 / 64 / 8 | 256 | 0.127 | 0.232 | 0.436 | 0.827 |

**64 / 64 / 8 wins at every length**, by 8.3× / 9.1× / 9.6× / 9.7× over the old
`8 / 32 / 4` fallback, and reaches 609 / 669 / 710 / 722 GB/s of KV — 63–75 % of the part's
bandwidth, against 73–74 GB/s (7.7 %) before. Note the two effects are separable: splits
alone (8→64 at block_n 32) is worth 6.5–7.5×, and the wider tile plus 8 warps adds the last
20–25 % on top.

### 3.2 Batch sensitivity (131K)

| batch | old 8/32/4 | new 64/64/8 | best in sweep |
|---:|---:|---:|---|
| 1 | 0.970 ms | 0.117 ms (8.29x) | 64/64/8 — 0.117 ms |
| 4 | 0.938 ms | 0.416 ms (2.26x) | 16/64/8 — 0.393 ms |
| 16 | 2.239 ms | 1.369 ms (1.64x) | 64/64/8 — 1.369 ms |

The chosen configuration is the best of the 20 at batch 1 and batch 16, and 6 % off the best
at batch 4 (16/64/8, 0.393 ms) — an acceptable price for a launch that cannot depend on the
batch size chosen at graph-capture time.

### 3.3 The other geometry the rule now reaches (16Q / 2KV / D256, Ornith)

| pool | config | 131K | 262K |
|---|---|---:|---:|
| bf16 (had no branch) | old fallback 8/32/4 | 0.811 | 1.605 |
| bf16 (had no branch) | **rule 64/32/4** | **0.312** | **0.602** |
| bf16 (had no branch) | sweep best 32/64/8 | 0.309 | 0.598 |
| bf16 (had no branch) | 64/64/8 (the D128 answer) | 0.324 | 0.615 |
| q8_0 (tuned branch) | old fallback 8/32/4 | 1.330 | 2.649 |
| q8_0 (tuned branch) | **tuned 64/64/4 (kept)** | **0.187** | **0.348** |
| q8_0 (tuned branch) | rule 64/32/4 | 0.217 | 0.410 |
| q8_0 (tuned branch) | 64/64/8 (the D128 answer) | 0.258 | 0.500 |

The `q8_0` row is the *tuned* Ornith Blackwell branch and this sweep independently confirms
it is optimal — the rule is not allowed to touch it. The bf16 pool had no branch and took the
same 8-split fallback Nemotron did; the rule's `(64, 32, 4)` lands within 1 % of the sweep's
best (`32/64/8`, 0.309 / 0.598) and 2.6× ahead of the fallback. `(64, 64, 8)` — the right
answer at head_dim 128 — is **43 % slower** than `(64, 64, 4)` on the quantized D256 pool
(0.500 vs 0.348 ms at 262K), which is why the tile is keyed on `head_dim`.

### 3.4 Numerical agreement

Split-K changes the *order* of the log-sum-exp reduction, so a different split count is
**not** bit-identical to the old one and cannot be: the gate is agreement, not equality.
Every configuration in every sweep above was diffed against the 8-split baseline at the same
context and against the dequantized-pool oracle:

- max |Δ| vs the 8-split baseline: **1.22e-04** over all **204** configurations timed here
  (bf16 output whose values are O(1); one bf16 ulp near 1.0 is 7.8e-03), and every one also
  passes `assert_close(rtol=atol=2e-2)`, the kernel's own regression tolerance.
- max |Δ| vs the dequantized oracle (q8_0 pool vs the same kernel fed dequantized bf16 K/V),
  over the 20 configurations where the oracle fits: **6.10e-05** — the new launch is no
  further from the oracle than the old one, and at 131K the 64-split configurations are
  *closer* to it (3.05e-05) than the 8-split fallback (6.10e-05).

End to end, the 131K/262K/524K needle is recalled correctly in every turn of §4.

## 4. End-to-end single-stream decode

The same server twice, only the launch configuration differing: the *before* arm is the
identical binary started with
`FREETOKEN_DECODE_KV_SPLITS=8 FREETOKEN_DECODE_BLOCK_N=32 FREETOKEN_DECODE_NUM_WARPS=4`,
which is bit-for-bit the pre-fix fallback. Both arms log the configuration they took at
startup (`Triton decode launch: kv_splits=... block_n=... warps=...`).

```
FREETOKEN_PIN_BUDGET_GB=17 ft serve --model .../NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --max-running-requests 1 --max-seq-len-override 1048576 --num-tokens 655360 \
  --kv-grow-step-tokens 131072 --kv-cache-dtype q8_0 --attention-backend triton \
  --moe-backend offload --moe-cache-auto --linear-state-slots 5 \
  --memory-ratio 0.85 --max-prefill-length 8192 --host-ram-reserve-gb 6
```

Four turns per length through `/v1/chat/completions` (greedy, `enable_thinking: false`,
`ignore_eos`, 128 decode tokens), on the byte-identical synthetic-needle prompt built once
on the CPU and reused by both arms. Decode rate is the inter-token rate from the SSE
timestamps, so TTFT is excluded.

| prompt tokens | decode before (mean of 4) | decode after (mean of 4) | speedup | prefill before -> after | needle |
|---:|---:|---:|---:|---:|---|
| 131,088 | 82.8 tok/s (78.1-86.6) | **145.3** (143.4-147.1) | **1.75x** | 3 181 -> 3 290 tok/s | 8/8 pass |
| 262,160 | 58.7 (57.3-59.6) | **132.4** (126.4-135.5) | **2.26x** | 1 934 -> 1 965 | 8/8 pass |
| 524,304 | 35.4 (33.7-36.7) | **113.6** (109.9-115.8) | **3.21x** | 1 073 -> 1 063 | 8/8 pass |
| 1,040,016 | *not re-run — 19.2-20.3 recorded* | **95.8** (single sample) | ~4.8x | 575 tok/s (TTFT 1 810 s) | pass |

The 1M row is not a paired arm: a cold 1.04M prefill costs 30 minutes, so the *before* side
is the figure recorded for this model on 2026-09-04 (19.2-20.3 tok/s over eight cached
follow-up turns of a 1,039,994-token session,
`benchmarks/results/nemotron35_lightning_5080_1m_multineedle_2026-09-04.md`) rather than a
run of the `FREETOKEN_DECODE_KV_SPLITS=8` arm. The *after* side is one fresh 1,040,016-token
prompt on the `--num-tokens 1048576` profile, 192 decode tokens, needle recalled. Treat the
~4.8x as indicative and the three paired rows above as the measurement.

Two things to read off this table:

1. **Prefill is untouched** (within 1-3 %, both directions), as it must be — the change is
   confined to the decode launch. Any claim that this trades prefill for decode is refuted
   by the middle column.
2. **The curve is flat, not merely shifted.** The 2B4 short-context single-stream figure for
   this model is 143.2 tok/s (`docs/nemotron.md`, "Measured throughput"); decode at 131,088
   tokens is now 145.3 tok/s, i.e. indistinguishable from decoding with no context at all.
   The residual slope from 131K to 524K (145 -> 114 tok/s) is the KV read itself: 6 layers x
   0.117 -> 0.402 ms per token of unavoidable bandwidth, matching §3.1 to within the noise.

Per-turn rows, both arms, in `scratchpad/decode_ab/results.jsonl`.

## 5. int64 slot ids on the KV load path

The Triton KV loaders widened slot ids to int64 on **store** (`kv_quant.py:47,168`) but not
on **load**: `slots[None, :] * stride_ks` was 32-bit in all four gather kernels
(`_paged_attention_kernel`, `_decode_grouped_stage1_kernel`, and both extend kernels). Triton
evaluates `ptr + offset` in the offset's own width, so the product wraps once a pool holds
2**31 elements or more. At Nemotron's 2 KV heads × 128 dim that is 8.4M slots (safe), but at
8 heads × 256 it is exactly 1,048,576 slots — the 1M profile.

Fixed with a compile-time `SLOT_I64` constexpr, set host-side by `_slot_offsets_need_int64`
from the pools' own `numel()`, so the widening costs nothing on the geometries that do not
need it (the Nemotron and Ornith runs above all compile `SLOT_I64=False` and are unchanged).

## 6. Files

- `python/freetoken/kernel/triton/attention.py` — `_grid_filling_splits`,
  `_decode_head_blocks`, `_sm_count`, `_slot_offsets_need_int64`,
  `_decode_launch_env_override`, `SLOT_I64` in the four gather kernels.
- `python/freetoken/attention/triton.py` — passes `sm_count`, logs the chosen launch once.
- `tests/kernels/test_triton_attention.py` — 5 new tests (tuned branches pinned unchanged
  with and without `sm_count`, the grid rule, the env override, the int64 predicate, and a
  CUDA decode-vs-reference check at the Nemotron head shape).
- `benchmarks/bench_decode_launch.py` — the sweep harness used here.
- `docs/nemotron.md` — decode numbers.

## 7. Reproduction

```
PYTHONPATH=python python benchmarks/bench_decode_launch.py \
    --q-heads 32 --kv-heads 2 --head-dim 128 --quant q8_0 \
    --ctx-lens 131072 262144 524288 1048576 --splits 8 16 32 64 128 \
    --block-n 32 64 --warps 4 8
```

The end-to-end A/B is the same server twice, the second arm with
`FREETOKEN_DECODE_KV_SPLITS=8 FREETOKEN_DECODE_BLOCK_N=32 FREETOKEN_DECODE_NUM_WARPS=4`.
Drivers: `scratchpad/decode_ab/{build_prompts.py,drive.py,run.sh}` under
/tmp/claude-1000/-home-lucas-ai-FreeToken/f4e2e9e3-f4f5-40d0-9980-b3b09d1ef47d/scratchpad/.
