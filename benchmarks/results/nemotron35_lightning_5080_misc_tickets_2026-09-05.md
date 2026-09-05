# Four ranked tickets, one GPU session — 5080 / Nemotron 3.5 Lightning NVFP4

Date 2026-09-05. Base `d960467` (after `c8d42c9`). Card RTX 5080 16 GB, WSL host 34 GiB.
Every GPU job under `scripts/gpu_lock.sh`, never piped, attended to exit.
Serving profile throughout: `FREETOKEN_PIN_BUDGET_GB=17`, `--mem-ratio 0.85`, 8K prefill
chunks, q8_0 KV, Triton attention, `--nvfp4-backend triton`, LFU.

| # | ticket | verdict | number |
|---|---|---|---|
| 1 | non-elastic `_determine_cuda_graph_bs` padding | **SHIPPED**, gated to offload-MoE | **1.074x** decode at 12 lanes (dense 1..16 vs the shipped sparse set) |
| 2 | NVFP4 MoE prefill A-operand deinterleave | **SHIPPED**, on by default | **1.215x** on the GEMM pair at M=8192, bit-exact; **1.074x** on 131K prefill end to end |
| 3 | `--moe-extend-cache-tokens` vs 512-token chunks | **no change (64 confirmed)** + a crash guard | crossover measured **between 64 and 80**; the cached path **cannot run** at m ≥ 256 |
| 4 | `--spec-draft-len 16` as the default | **NO-GO**, stays 8 | k=16 is **0.870x** of spec-off at 131K (k=8 0.898x); criterion was ±2 % |

---

## 1. `graph.py::_determine_cuda_graph_bs` — the non-elastic ladder

### The defect
`_elastic_graph_batch_sizes` went dense to 16 in `14c1bd8` (`_DENSE_GRAPH_BS`), but the
**non-elastic** helper still built `[1, 2, 4] + range(8, max_bs + 1, 8)`. On a server at
`--cuda-graph-max-bs 16` a 12-lane decode batch therefore replayed the **bs-16 graph with
four dummy rows**, and a dummy row on an offload-MoE model is not free: it carries a hidden
state, routes its own top-6 experts and adds `top_k` rows to every expert GEMV.

### What was run
`benchmarks/decode16/phaseE.sh` (already written, priced *padding*: eager-at-12 vs
padded-to-16) and then `benchmarks/decode16/phaseE2.sh` (new — prices *the fix*: the shipped
sparse set vs dense 1..16). Both non-elastic, `--max-running-requests 16`, 12 clients,
256 decode tokens, `benchmarks/bench_decode_moe.py`.

phaseE, re-run at `d960467` (aggregate decode tok/s):

| arm | graph set | 2026-09-05 earlier run | this run |
|---|---|---:|---:|
| E1 `--cuda-graph-max-bs 8` | `[1,2,4,8]`, 12 runs eager | 148.75 | 145.42 |
| E2 `--cuda-graph-max-bs 16` | `[1,2,4,8,16]`, 12 pads to 16 | 139.16 | 144.95 |

Padding never wins, but the penalty reads anywhere from −6.7 % to −0.3 %: **two repeats of
one arm cannot resolve this**, which is why phaseE2 alternates three of each.

phaseE2 — three alternating repeats of each arm out of **one binary**
(`FREETOKEN_GRAPH_DENSE_BS=0|1`), same server flags in both:

| arm | graph set | r1 | r2 | r3 | mean | event-gap p50 |
|---|---|---:|---:|---:|---:|---:|
| `s16` (shipped) | `[1,2,4,8,16]` | 138.69 | 144.86 | 137.74 | **140.43** | 83.0 / 83.8 / 87.0 ms |
| `d16` (dense) | `[1..16]` | 149.83 | 150.33 | 152.55 | **150.90** | 79.3 / 77.2 / 78.2 ms |

**1.074x**, and every dense run is above every sparse run (perfect separation, the smallest
p a 3-vs-3 rank test can produce). Identical to the elastic tier's own 1.074x at 12 lanes.
Capture cost: 11 extra graphs, ~80 MiB, ~0.8 s of startup.

### Shipped
`python/freetoken/engine/graph.py`: `_determine_cuda_graph_bs` takes `offload_moe` and, when
`_dense_small_graph_bs(offload_moe)` is true, unions `range(1, min(max_bs, 16) + 1)` into the
ladder (sorted, deduped); `GraphRunner.__init__` passes `offload_moe=moe_offload_cache is not
None`, so **only offload-MoE models change** — a dense model keeps the historical list
byte-for-byte, and that is pinned by a test. `FREETOKEN_GRAPH_DENSE_BS=0|1` forces either
arm (a garbage value warns and is ignored). Tests:
`tests/engine/test_elastic_graph_sizes.py` (8 new, the non-elastic helper had none).

---

## 2. NVFP4 MoE prefill GEMM — the A operand

### The defect
`_prefill_nvfp4_moe_kernel` walks K in **packed bytes** (two e2m1 codes per byte) and issues
one `tl.dot` per nibble, so against a plain `[M, K]` bf16 activation it builds
`a_ptrs_lo`/`a_ptrs_hi` at `2*offs_kb` and `2*offs_kb + 1`: both A gathers are stride-2 on
the contiguous axis, so neither vectorizes and A's K span is read twice per K-block.

### The change
A `DEINTERLEAVED_A: tl.constexpr` arm in the kernel plus a host prepass that rewrites A into
an **even-k plane followed by an odd-k plane** (`a.view(M, K//2, 2).permute(0,2,1)`), after
which both gathers are unit-stride. The per-`tl.dot` reduction order is unchanged, so this is
a numerics no-op — and it measures as one.

`benchmarks/bench_moe_prefill_gemm.py --m 256 1024 2048 4096 8192 --variant tree deint
prepass --grid shipped --verify` (synthetic banks, shipped tiles, 3 warmup + 9 timed; the
`deint` arm **includes** its prepass, the `prepass` arm times only the two rewrites):

| M | tree (shipped) | deint | speedup | of which prepass | deint TFLOP/s | max abs diff |
|---:|---:|---:|---:|---:|---:|---|
| 256 | 1.271 ms | **1.091** | 1.165x | 0.015 ms | 28.1 | 0.000e+00 |
| 1024 | 2.937 | **2.595** | 1.132x | 0.039 | 47.3 | 0.000e+00 |
| 2048 | 5.103 | **4.459** | 1.144x | 0.142 | 55.0 | 0.000e+00 |
| 4096 | 9.171 | **7.782** | 1.178x | 0.279 | 63.0 | 0.000e+00 |
| **8192** | **16.960** | **13.961** | **1.215x** | 0.551 | **70.3 (59 % of `tl.dot`)** | 0.000e+00 |

Gate was ≥1.08x at M=8192 and exact: **1.215x, bit-exact at every M.** The remaining gap to
b12x at M=8192 falls from 1.34x to **1.10x** (12,618 µs, `bench_nvfp4_moe_kernels.py`).

### End to end (the number that matters)
`benchmarks/bench_long_context.py --synthetic-needle --target-prompt-tokens 130000`, cold
single request through the serving API, two pairs, `FREETOKEN_NVFP4_PREFILL_DEINTERLEAVE_A`
as the only difference:

| arm | prefill end-to-end tok/s | engine average tok/s | decode tok/s | needle |
|---|---:|---:|---:|---|
| off r1 / r2 | 6008.62 / 6240.71 | 5582.62 / 5874.55 | 140.83 / 140.54 | PASS |
| **on r1 / r2** | **6573.10 / 6582.48** | **6079.85 / 6275.37** | 140.94 / 140.86 | PASS |
| mean | 6124.7 → **6577.8 (1.074x)** | 5728.6 → **6177.6 (1.078x)** | unchanged | — |

131K TTFT 21.6 s → 19.8 s. Decode is untouched (the decode GEMV kernels were not modified).

### Shipped
`python/freetoken/kernel/triton/nvfp4_fused_moe.py` (constexpr arm),
`python/freetoken/moe/fused_nvfp4.py` (`deinterleave_a`, module flag
`NVFP4_PREFILL_DEINTERLEAVE_A`, **on by default**; `FREETOKEN_NVFP4_PREFILL_DEINTERLEAVE_A=0`
is the hatch), `benchmarks/bench_moe_prefill_gemm.py` (`deint` / `prepass` variants, and
`--verify` now always references the production `tree` arm and prints the delta).
Test: `tests/moe/test_nvfp4_backends.py::test_deinterleaved_a_is_bit_identical_to_the_
interleaved_kernel` (`torch.equal`, m ∈ {1, 8, 33}). Decode kernels and the fp8 sibling
untouched.

**Left on the table:** gemm2's A *is* gemm1's output, so its 182 MB prepass is removable by
having gemm1's store emit the two k-planes directly (~0.3 ms of the 0.551 at M=8192).

---

## 3. `--moe-extend-cache-tokens` vs the scheduler's 512-token chunks

New harness: `benchmarks/bench_extend_moe_threshold.py` + `benchmarks/extend_moe/
run_threshold.sh`. One model load; per (m, arm) 7 timed extend forwards on a 4,096-token
base with a **fresh tail per call** (without it the radix tree serves the prompt and an
m-token extend is silently measured as a 1-token one); a `use_cached_extend` proxy records
`(layer_id, num_tokens, decision)` so the arm is proven per row (23/23 vs 0/23) and a
mislabelled run fails loudly; 32 decode steps after each cell to price the cached arm's
damage to the following decode's working set.

| m | arm | wall ms | GPU ms | host ms | MoE ms/layer | missing expert rows/layer | next decode ms/step |
|---:|---|---:|---:|---:|---:|---:|---:|
| 64 | stream | 281.1 | 299.8 | 268.9 | 10.813 | — | 9.06 |
| 64 | **cached** | **249.4** | 272.3 | 50.5 | **0.463** | 68.6 (miss 0.81) | 6.92 |
| 80 | stream | **285.3** | 301.2 | 272.6 | 10.906 | — | 8.12 |
| 80 | cached | 294.8 | 312.0 | 58.8 | 1.459 | 95.6 | 7.10 |
| 96 | stream | **284.0** | 299.5 | 271.3 | 10.893 | — | 6.88 |
| 96 | cached | 330.5 | 348.4 | 63.9 | 1.595 | 101.6 | 7.48 |
| 128 | stream | **274.1** | 299.9 | 261.9 | 10.523 | — | 6.71 |
| 128 | cached | 370.3 | 382.1 | 70.4 | 1.792 | 107.8 (miss 1.00) | 7.54 |
| 256 | stream | 271.9 | 300.1 | 259.7 | 10.381 | — | 6.68 |
| 256 | **cached** | **CRASH** | | | | | |
| 512 | (not reached) | | | | | | |

**Crossover is between 64 and 80** — the shipped default of 64 sits exactly at it and does
not move. The mechanism is the miss column: the cached path fetches only the *routed*
experts, but by m=128 it is fetching 107.8 of 128 rows per layer anyway, at the scattered
gather rate (52.9 GB/s) instead of the contiguous stream rate (61.9 GB/s), and it pays a
decode GEMV where the stream pays a grouped GEMM. The decode burst after each cell shows no
eviction penalty worth the name (6.7–7.5 ms/step either way), so the forward decides it.

**Read the wall/GPU columns, not the host column.** The arms hide their cost in different
places: the stream arm blocks the host on its PCIe copies (host ≈ wall), while the cached
arm returns in ~60 ms and leaves the GPU gathering scattered rows for another ~250 ms.
Scored on host time the harness reported a crossover at 96; the summary now scores wall
(the CUDA-event GPU span agrees with it to ~2 %).

### The crash (a live bug, now guarded)
At m=256 the cached path does not merely lose, it **cannot execute**:

```
layers/moe.py:_prefill_routed -> _cached_extend_routed -> _decode_routed
  -> offload_cache.ensure_experts -> offload_kernels.py:57 lru_ensure -> _seq
  -> ValueError('numel (4194304) exceeds triton maximum tensor numel (1048576)')
```

flashlib's `_seq`/`_insert` both launch `_phase1` with `BLOCK_K = next_pow2(query.numel())`
and build the `[BLOCK_K, BLOCK_K]` dedup matrix `q[:, None] == q[None, :]`. Triton caps a
tensor at 1,048,576 elements, so `BLOCK_K > 1024` cannot compile, and the query is
`num_tokens * top_k` ids — a hard ceiling of **m ≤ 170 at top_k = 6**. `--moe-extend-cache-
tokens` is a user-settable flag with no upper bound, so `512` crashes the engine mid-forward.
Even the widths that do compile are pathological: the m=128 cell (`BLOCK_K = 1024`) spent
**22 minutes of one-off Triton JIT** before it ran.

Guard: `use_cached_extend(layer_id, num_tokens, num_routed=None)` refuses when
`num_routed > _MAX_ENSURE_QUERY (1024)`; the one production call site passes
`topk_ids.numel()`. Refusing costs nothing — the stream is already faster at every width
above the crossover. Tests: three new in `tests/moe/test_extend_cache.py`, plus
`test_every_copy_of_the_default_agrees`, which pins the four hardcoded copies of the default
(`moe/offload_cache.py`, `engine/config.py`, `engine/engine.py::_DENSE_MOE_SETTINGS`,
`ServerArgs`) against each other — the `_DENSE_MOE_SETTINGS` copy had no test.

**131K needle prefill unchanged:** four cold 131K runs above, needle PASS in all four,
prefill improved by item 2 and decode flat. The extend gate is not on that path (the 8,192-
token chunks are far above any threshold considered here).

---

## 4. `--spec-draft-len 16` as the default — NO-GO

`benchmarks/probe_spec_ngram_impl.py --only copy --sweep-k 8 16 --max-tokens 1024
--needle-max-tokens 256`, one model load, greedy, arms in order `off / n8k8 / n8k16 / off2`
per class.

### 131K, non-copy (needle, 123,612 prompt tokens, 79 generated)

| arm | tok/s | vs off | verify steps | tokens/verify | `declined_uneconomic` | verify step: total / GPU forward |
|---|---:|---:|---:|---:|---:|---|
| off | 95.97 | — | — | — | — | decode step ≈ 10.4 ms |
| n8k8 | 86.22 | **0.898x** | 2 | 6.50 | 16 | 81.5 / 89.0 ms |
| n8k16 | 83.52 | **0.870x** | 3 | 9.00 | **0** | 92.9 / 101.3 ms |
| off2 | 95.02 | 0.990x | — | — | — | (control spread 1 %) |

The criterion was "raise the default to 16 only if the 131K non-copy case stays within 2 % of
off". It is **13 % below** — k=16 is a NO-GO, and k=8 is not innocent either (−10 %).

**The probe is not the only cost, and at k=16 the gate stops protecting.** A gate probe at
131K costs `_GATE_MIN_SAMPLES = 2` verify steps: **163 ms at k=8, 279 ms at k=16** (3 steps
were spent) against a 10.4 ms decode step. But the decisive difference is that at k=16 the
gate **never closed** (`declined_uneconomic` 0 of 55 peeks, vs 16 at k=8): a longer draft
raises `emit` (9.0 tokens/verify) about as fast as it raises `verify_ms`, so
`emit * 1.25 > verify_ms / decode_ms` (8.93) stays true and the drafter keeps paying
near-break-even steps for the whole generation. Raising k weakens the one mechanism that is
supposed to make long-context speculation free.

### Copy class, short context (1,129-token prompt, 1,023 generated)

| arm | tok/s | draft rate | tokens/verify | `declined_uneconomic` | verify step total / GPU |
|---|---:|---:|---:|---:|---|
| off | 138.74 | — | — | — | — |
| n8k8 | 135.62 | 0.037 | 4.94 | 52 | 35.8 / 34.7 ms |
| n8k16 | 135.84 | 0.020 | 4.63 | 16 | 49.8 / 52.7 ms |
| off2 | 143.97 | — | — | — | — |

Both arms land at 0.98x of off with draft rates of 0.02–0.04 — i.e. this run drew the low end
of the copy-class lottery documented in `ngram_spec_fast_2026-09-05.md` §1 (arms of the same
binary spanned 1.04x–1.67x because speculation perturbs its own token stream). **Do not read
a copy-class verdict off end-to-end arms**; the fixed-transcript replay
(`benchmarks/spec_engage_replay.py`, §8 of that write-up) is the authority there, and it puts
k=16 at 1.88x against k=8's 1.61x. What this run does contribute is the *step cost* at short
context: 35.8 ms at k=8 vs 49.8 ms at k=16, a 1.39x more expensive probe.

### Decision
`spec_draft_len` stays **8** (`python/freetoken/scheduler/config.py:33`), now pinned by
`tests/scheduler/test_spec_ngram.py::test_spec_draft_len_default_stays_8` with the numbers
above in the docstring, plus
`test_the_gate_does_not_close_at_the_k16_operating_point`. Copy-heavy agent traffic should
still pass `--spec-draft-len 16` explicitly. Speculation itself remains opt-in
(`--speculative ngram`).

---

## Reproduction

```bash
# 1  non-elastic graph ladder, 3 alternating repeats per arm (~22 min)
scripts/gpu_lock.sh benchmarks/decode16/phaseE2.sh benchmarks/decode16/runs/phaseE2

# 2  A-operand deinterleave, both arms + prepass cost + exactness (~1 min)
PYTHONPATH=python scripts/gpu_lock.sh .venv/bin/python -u \
  benchmarks/bench_moe_prefill_gemm.py --m 256 1024 2048 4096 8192 \
  --variant tree deint prepass --grid shipped --verify > deint.log 2>&1
# ...and end to end (FREETOKEN_NVFP4_PREFILL_DEINTERLEAVE_A=0|1 selects the arm)
uv run python -u benchmarks/bench_long_context.py --model $M --synthetic-needle \
  --needle-depth 0.5 --target-prompt-tokens 130000 --max-context 131072 --decode 128 \
  --kv-cache-dtype q8_0 --prefill-chunk 8192 --mem-ratio 0.85 --cache-policy lfu \
  --nvfp4-backend triton --host-ram-reserve-gb 6 --kv-grow-step-tokens 65536

# 3  extend-cache threshold (~5 min for m<=96; m>=256 crashes by design)
FREETOKEN_GPU_LOCK_WAIT=7200 scripts/gpu_lock.sh benchmarks/extend_moe/run_threshold.sh
#   NB: the first m whose m*top_k crosses 512 -> 1024 costs ~22 min of Triton JIT.

# 4  spec draft length (~2 min)
PYTHONPATH=python .venv/bin/python -u benchmarks/probe_spec_ngram_impl.py --model $M \
  --moe-cache-auto --only copy --sweep-k 8 16 --max-tokens 1024 --needle-max-tokens 256 \
  --out spec_k8_k16.json
```

Raw artifacts (gitignored): `benchmarks/decode16/runs/phaseE2/`,
`benchmarks/decode16/runs/deint131k/`, `benchmarks/extend_moe/runs/threshold/`,
`benchmarks/spec_k/runs/`.

## Verification
`ruff check .` clean; `pytest tests/scheduler tests/server tests/kvcache tests/dsv4
tests/models/test_nemotron_h*.py` 1,375 passed / 3 skipped; `pytest tests/moe tests/engine
tests/kernels` 847 passed / 13 skipped (both with the deinterleave default ON);
`benchmarks/scheduler_replay.py --gate` passed. GPU at 0 MiB at the end of the session.
