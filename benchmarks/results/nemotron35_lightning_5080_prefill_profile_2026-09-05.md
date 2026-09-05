# Prefill profile — Nemotron 3.5 Lightning 30B-A3B-NVFP4 on one RTX 5080 (2026-09-05)

Queue item Q5: *"profile the prefill curve (3,200 tok/s @131K → 1,065 @524K → 575 @1M):
attention vs SSD scan vs KV grow; fix or file."*

**Answer: (a) the extend/prefill attention kernel, and essentially nothing else — and the
whole superlinear term was a launch configuration, not an algorithm.** The kernel took a
`BLOCK_M = 128` tile with `num_warps = 4`, which spills its fp32 accumulator (**396 spill
slots per thread against 14** at `BLOCK_M = 64`) and runs at 29 TFLOP/s instead of 70. One
line — cap `BLOCK_M` by the accumulator's register budget — is worth **2.46x on the kernel
at every KV length** and this end to end:

| prompt tokens | prefill before | prefill after | speedup | TTFT before → after |
|---:|---:|---:|---:|---:|
| 131,088 | 3,230 tok/s | **5,288 tok/s** | **1.64x** | 40.6 s → **24.8 s** |
| 262,160 | 1,965 | **3,683** | **1.87x** | 133.4 s → **71.2 s** |
| 1,040,016 | 573–576 (recorded 3x) | **1,307** | **2.28x** | 1,810–1,824 s → **795.8 s** |

The first two rows are paired arms of one A/B — the same binary twice, only the launch
differing. The needle is recalled in every run of both arms. Decode is unchanged (73–82
tok/s at 131K/262K in both arms; the decode kernel was not touched).

GPU 0 MiB at the end. Nothing here is committed.

---

## 1. The shape of the curve, before any measurement

The four recorded whole-prompt averages fit a two-term model to within 4 %:

```
t(n) = 16.2 * n  +  26.5 * n^2          seconds, n in units of 131,072 tokens
```

| tokens | measured s | fit s | linear term | quadratic term | quadratic share |
|---:|---:|---:|---:|---:|---:|
| 131,072 | 41.0 | 42.7 | 16.2 | 26.5 | 62 % |
| 262,144 | 134.4 | 138.4 | 32.5 | 105.9 | 77 % |
| 524,288 | 492.3 | 488.6 | 65.0 | 423.6 | 87 % |
| 1,048,576 | 1,823.6 | 1,824.3 | 130.0 | 1,694.3 | **93 %** |

So 93 % of a 1M TTFT is the quadratic term, and the linear-only ceiling is 8,069 tok/s.
Attributing the quadratic term to attention implies the prefill attention kernel runs at
**31.9 TFLOP/s** — identical at all four lengths, and ~14 % of the RTX 5080's ~225 TFLOP/s
bf16 peak. That number is what made "the kernel, not the algorithm" the first hypothesis.

## 2. Why the prefill attention kernel is the only quadratic term

`extend_paged_attention` launches

```
grid = (batch, num_q_heads, cdiv(chunk, BLOCK_M))       # NO KV-split dimension
```

and `_extend_attention_split_kernel` walks the whole prefix serially inside each program
(`for start_n in tl.range(0, prefix_len, BLOCK_N)`, `attention.py:1395`). Every one of the
`cdiv(8192, BLOCK_M)` q-blocks therefore re-reads the entire prefix: cost is
`O(chunk x prefix)` per chunk, `O(n^2)` over a prompt. **Occupancy is not the problem** —
unlike the 2026-09-04 decode case, this grid is 2,048–8,192 CTAs on 84 SMs at the Nemotron
shape. The per-CTA inner loop is.

Everything else on the prefill path is provably flat in position:

- **SSD scan (b).** `mamba2_prefill` is handed this chunk's tokens plus one `[H,P,N]`
  carried state (`kernel/triton/mamba2/__init__.py:141` gathers it, `:179-180` writes it
  back); `Mamba2Metadata` is built from `extend_len` alone (`attention/linear.py:82`), and
  `tests/kernels/test_mamba2_ssd.py::test_chunk_continuation_equals_one_pass` already pins
  the property. Nothing re-scans prior tokens. Measured below: **1.068 ms per layer per 8K
  chunk, constant.**
- **MoE / dense (c).** Static per-block dispatch, no position input at all.
- **Engine (d).** Intermediate chunks are *not* inserted into the prefix cache
  (`scheduler.py:668-678`: `if isinstance(req, ChunkedReq): ... continue` — the whole prompt
  is inserted once, on the last chunk), `allocate_paged` allocates only the new pages, and
  no `.item()` runs on a `ChunkedReq`. The one genuinely `O(prefix)` item per chunk is the
  attention backend's page-index build,
  `indices = torch.cat([page_table[req.table_idx, : req.device_len] for req in reqs])`
  (`attention/triton.py:299`) — a 4 MB int32 device tensor at 1M. §5 bounds its cost at
  **zero, ±2 %.**

## 3. Kernel microbenchmark — `benchmarks/bench_prefill_attention.py` (new)

The prefill twin of `bench_decode_launch.py`: production `extend_paged_attention`, no
server, no weights, one 8,192-token chunk against a quantized paged prefix. Median of 3–5
timed calls, clocks settled with `torch.cuda._sleep`, every configuration verified against
the tree's default launch at `rtol=atol=2e-2` before it is timed.

### 3.1 Nemotron geometry (32Q / 2KV / D128, q8_0 KV), chunk 8192 — ms per attention layer

| BLOCK_M/N/warps/stages | CTAs | prefix 0 | 131,072 | 262,144 | 524,288 | 1,048,576 | TFLOP/s | projected 1M attention |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **64 / 64 / 4 / 1** | 4096 | 8.78 | **256.97** | **508.36** | **1006.53** | **2006.66** | **70.4** | **767 s** |
| 32 / 128 / 4 / 1 | 8192 | 8.22 | 272.34 | 541.21 | 1074.04 | 2140.88 | 66.0 | 818 s |
| 32 / 64 / 4 / 1 | 8192 | 7.78 | 386.72 | 765.12 | 1521.91 | 3035.53 | 46.5 | 1,160 s |
| 64 / 128 / 4 / 1 | 4096 | 12.43 | 518.31 | 1025.33 | 2037.60 | 4062.79 | 34.8 | 1,553 s |
| 128 / 64 / 4 / 1 *(tree default)* | 2048 | 9.77 | 619.94 | 1239.34 | 2471.96 | 4936.73 | 28.6 | 1,883 s |
| 128 / 128 / 4 / 1 | 2048 | 9.66 | 723.95 | 1443.19 | 2857.37 | 5690.39 | 24.8 | 2,176 s |

`64/64/4/1` wins by **2.41x / 2.44x / 2.46x / 2.46x** over the default at 131K / 262K /
524K / 1M. The ratio is flat in prefix, and each column is linear in prefix to 4 digits —
exactly the `O(chunk x prefix)` shape.

The projection column (whole-prompt attention seconds, 6 layers, 8K chunks, the measured
`ms(prefix)` curve integrated over 128 chunks) reads **1,883 s for the default** against the
**1,694 s** the two-term fit assigned to the quadratic term and the 1,824 s of the entire
measured 1M prefill. Attention is the whole thing.

### 3.2 The wider sweep at a 131,072-token prefix (66 configurations)

Best 12, one attention layer, ms:

| config | ms | | config | ms |
|---|---:|---|---|---:|
| **64/64/4/1** | **257.8** | | 64/32/4/2 | 382.0 |
| 32/128/4/1 | 274.3 | | 32/32/4/1 | 383.5 |
| 64/128/8/1 | 329.2 | | 128/32/8/3 | 383.7 |
| 32/64/2/1 | 347.7 | | 32/64/4/1 | 386.8 |
| 32/32/2/1 | 350.8 | | 128/32/8/1 | 394.5 |
| 64/32/4/1 | 352.7 | | 64/64/8/1 | 435.6 |

`128/64/4/1`, the tree default, is 26th of 66 at 621.3 ms. More software-pipeline stages
never helped (`num_stages=1` wins at every tile that fits); `BLOCK_N = 128` overflows the
101,376-byte opt-in shared memory at 3 stages and loses at 2.

### 3.3 The mechanism: a spilling accumulator

Compiled `n_spills` (spill slots per thread) straight out of the Triton JIT cache, same
kernel, same shapes:

| BLOCK_M / warps | n_regs | **n_spills** | smem | ms @131K prefix |
|---|---:|---:|---:|---:|
| 64 / 4  | 255 | **14** | 40,960 | **257.8** |
| 32 / 4  | 255 | 0 | 28,672 | 386.8 |
| 64 / 8  | 248 | 0 | 40,960 | 435.6 |
| 128 / 8 | 255 | 148 | 73,728 | 485.3 |
| **128 / 4** *(tree default)* | 255 | **396** | 73,728 | **619.9** |

The kernel's accumulator is `BLOCK_M x BLOCK_DV` fp32 over `32 * num_warps` lanes: 64
registers per thread at `BLOCK_M=64`, **128** at `BLOCK_M=128` with 4 warps. Fewest spills
is not the same as fastest (`32/4` and `64/8` spill nothing and are slower — too few rows
per warp), but the default's 396-slot spill is what turns 70.4 TFLOP/s into 28.6.

## 4. The fix

`_select_extend_tile`'s `head_dim <= 128` arm returned a hard-coded `(128, 64)` that had
never been measured against the warp count chosen next to it. On sm_120 that warp count is
4 — a value swept for the *D=256* consumer `(64, 32)` tile in an earlier effort and applied
to every geometry, exactly the failure mode of the 2026-09-04 decode fallback.

`extend_launch_config` (new, split out of the launcher so it is testable) now caps `BLOCK_M`
by the kernel's own accumulator shape:

```python
_EXTEND_ACC_REGS = 64
def _extend_block_m_cap(block_dv, num_warps):
    return max(16, _EXTEND_ACC_REGS * 32 * num_warps // block_dv)
...
if head_dim <= 128:
    block_m = min(block_m, _extend_block_m_cap(block_d, num_warps))
```

What moves and what does not:

| head_dim | device | before | after |
|---:|---|---|---|
| 128 | sm_120, 4 warps | 128 / 64 / 4 / 1 | **64 / 64 / 4 / 1** |
| 128 | sm_89, sm_90, 8 warps | 128 / 64 / 8 / 1 | 128 / 64 / 8 / 1 (unchanged) |
| 256 | consumer (measured branch) | 64 / 32 / 4 / 2 | unchanged |
| 256 | consumer q6/q5 (measured) | 64 / 32 / 8 / 2 | unchanged |
| 256 | datacenter | 128 / 64 / 8 / 1 | unchanged |
| 512 | consumer (gemma4) | 16 / 16 / 4 / 1 | unchanged |

Only the arm that was never measured moves. `FREETOKEN_EXTEND_BLOCK_M / _BLOCK_N /
_NUM_WARPS / _NUM_STAGES` (new, the prefill twins of `FREETOKEN_DECODE_*`) force any launch
at startup, which is how the A/B below was run as two invocations of one binary; the server
now logs `Triton extend launch: block_m=... block_n=... warps=... stages=...` once, so a run
proves which arm it took.

**The gate is agreement, not bitwise equality** — changing `BLOCK_M` changes the order of
the flash accumulation. Every configuration in every sweep above passed
`assert_close(rtol=atol=2e-2)` against the previous launch, and the new CUDA test also
checks both against the dense reference.

## 5. End to end — per-chunk cost as a function of position

`input throughput (token/s): X instant` on each `Prefill batch` line is
`#new-token / (wall time since the previous batch)`, measured after the forward's drain
barrier, so `#new-token / X` is that chunk's cost and the running sum of `#new-token` is its
prefix. Regressing chunk cost on prefix (chunk 0 excluded — see below):

| run | chunks | total | ms per 8K chunk | r² |
|---|---:|---:|---|---:|
| before, 131,088 | 16 | 39.6 s | `872 + 25.40e-3 * prefix` | 0.9992 |
| before, 262,160 | 32 | 132.9 s | `836 + 25.79e-3 * prefix` | 0.9985 |
| after, 131,088 | 16 | 24.1 s | `842 + 10.22e-3 * prefix` | 0.9892 |
| after, 262,160 | 32 | 70.8 s | `831 + 10.60e-3 * prefix` | 0.9962 |
| **after, 1,040,016** | **127** | **795.9 s** | **`861 + 10.42e-3 * prefix`** | **0.9990** |

Two things fall straight out:

1. **The intercept does not move** (836 → 831 ms). The change is confined to the
   position-dependent term, as it must be.
2. **The slope ratio is 25.79 / 10.60 = 2.43**, against the kernel microbench's 2.44 at the
   same prefix. The engine sees exactly the kernel's speedup.

### 5.1 Bounding the non-attention position-dependent cost (d)

Write `slope = s_att + s_other`. Two engine measurements (before, after) plus the kernel's
own ratio `R = 2.46` give three equations and solve for `s_other` without needing any
in-engine per-kernel instrumentation:

| pair | s_att (after) | **s_other** | share of slope |
|---|---:|---:|---:|
| 131K arms | 10.41e-3 | −0.19e-3 | −1.9 % |
| 262K arms | 10.41e-3 | +0.18e-3 | +1.7 % |
| 262K before vs 1M after | 10.51e-3 | −0.09e-3 | −0.9 % |

**`s_other = 0 ± 0.2e-3 ms per token of prefix per chunk**, i.e. ±2 % of the slope and
**±13 s of a 1M prefill.** The `torch.cat` page-index build, the page allocation, the KV
grow steps and every other host-side `O(prefix)` item on the chunked path are together
unmeasurable at this resolution. **(d) is not the problem, and neither is KV grow.**

### 5.2 Per-position breakdown, the 1M run (after the fix)

`attn` from the engine-derived slope (§5.1) plus the microbench's 6 x 8.78 ms at prefix 0;
`MoE` = 23 layers x 29.46 ms (§5.3); `SSD` = 23 x 1.068 ms (§5.3); `rest` is the residual.

| chunk | prefix | chunk ms | attn ms (share) | MoE ms | SSD ms | rest ms | before-arm ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 4,799 | 53 (1 %) | 678 | 25 | 4,044 | 2,184 |
| 8 | 65,536 | 1,502 | 742 (49 %) | 678 | 25 | 58 | 2,465 |
| 16 | 131,072 | 2,545 | 1,431 (56 %) | 678 | 25 | 412 | 4,462 |
| 24 | 196,608 | 2,923 | 2,120 (73 %) | 678 | 25 | 101 | 5,919 |
| 32 | 262,144 | 3,737 | 2,809 (75 %) | 678 | 25 | 226 | – |
| 48 | 393,216 | 5,064 | 4,187 (83 %) | 678 | 25 | 175 | – |
| 64 | 524,288 | 6,405 | 5,565 (87 %) | 678 | 25 | 138 | – |
| 80 | 655,360 | 7,752 | 6,943 (90 %) | 678 | 25 | 106 | – |
| 96 | 786,432 | 9,056 | 8,321 (92 %) | 678 | 25 | 33 | – |
| 112 | 917,504 | 10,507 | 9,699 (92 %) | 678 | 25 | 105 | – |
| 126 | 1,032,192 | 11,980 | 10,905 (91 %) | 678 | 25 | 372 | – |

`rest` is 100 ± 200 ms with **no trend in position** — the router, the shared expert, the
conv1d, the Mamba in/out projections, the gated norms, the LM head and all engine work,
together, and flat. Attention goes from 1 % of a chunk at the head of the prompt to 91 % at
the tail even *after* the fix.

**Chunk 0 is an outlier in both arms** (4,799 ms after, 2,184 ms before): first-touch Triton
autotuning of the SSD kernels and the MoE offload cache filling. It is one chunk of 127 and
is excluded from every regression above.

### 5.3 The flat term, decomposed

| component | per 8K chunk | source |
|---|---:|---|
| MoE routed-expert GEMMs, 23 layers x 29.46 ms | **677.7 ms (79 %)** | `bench_nvfp4_moe_kernels.py --prefill-m 8192 --backend triton` |
| attention at prefix 0, 6 layers x 8.78 ms | 52.7 ms (6 %) | §3.1 |
| Mamba-2 SSD scan, 23 layers x 1.068 ms | 24.6 ms (3 %) | `bench_mamba2_ssd.py --case single-8192` (166 MB transient, PASS) |
| everything else (router, shared expert, conv1d, projections, norms, engine) | ~106 ms (12 %) | residual against the 861 ms intercept |

**The SSD scan is 3 % of the flat term and 0.2 % of a 1M prefill, and it is flat in
position — (b) is exonerated.** The MoE is 79 % of it: the flat term is essentially the MoE
expert GEMMs, and they are now the binding constraint below ~200K tokens (at 131K, after the
fix, 16 x 861 ms = 13.8 s of a 24.8 s TTFT).

## 6. Projected and measured 1M TTFT

Built from the 262K before-arm slope, the kernel ratio, and the 861 ms intercept:

| term | before | after |
|---|---:|---:|
| flat, 127 x 861 ms | 109 s | 109 s |
| attention, `slope x sum(prefix)`, sum = 65,544,192 | 1,690 s | 689 s |
| non-attention position-dependent (§5.1) | 0 ± 13 s | 0 ± 13 s |
| **predicted total** | **1,799 s** | **798 s** |
| **measured** | **1,810–1,824 s** (3 recordings) | **795.8 s** |

## 7. Files

- `python/freetoken/kernel/triton/attention.py` — `extend_launch_config`,
  `_extend_block_m_cap`, `_EXTEND_ACC_REGS`, `_extend_launch_env_override`; the launcher now
  calls `extend_launch_config` instead of inlining the choice.
- `python/freetoken/attention/triton.py` — the `Triton extend launch: ...` startup line.
- `tests/kernels/test_triton_attention.py` — 4 new tests (the register-budget rule, the
  launch table incl. every measured branch pinned unchanged, the env override, and a CUDA
  agreement check of the new tile against the old one *and* the dense reference at the
  Nemotron head shape). 55 pass on the 5080.
- `benchmarks/bench_prefill_attention.py` — the sweep harness used here.
- `docs/nemotron.md` — prefill numbers.

## 8. Reproduction

```
PYTHONPATH=python python benchmarks/bench_prefill_attention.py \
    --q-heads 32 --kv-heads 2 --head-dim 128 --quant q8_0 --layers 6 \
    --prefix-lens 0 131072 262144 524288 1048576 --chunk 8192 \
    --block-m 32 64 128 --block-n 64 128 --warps 4 --stages 1
```

The end-to-end A/B is the same server twice, the "before" arm started with
`FREETOKEN_EXTEND_BLOCK_M=128 FREETOKEN_EXTEND_BLOCK_N=64 FREETOKEN_EXTEND_NUM_WARPS=4
FREETOKEN_EXTEND_NUM_STAGES=1`, which is bit-for-bit the pre-fix launch. Drivers:
`scratchpad/prefill_ab/{run.sh,drive.py,chunks.py}` and `scratchpad/prefill/regs.py` under
/tmp/claude-1000/-home-lucas-ai-FreeToken/f4e2e9e3-f4f5-40d0-9980-b3b09d1ef47d/scratchpad/.

**Do not pipe `scripts/gpu_lock.sh` into another command.** Its exit trap runs
`pkill -9 -g $$`, which kills the whole pipeline including the shell reading it; the run
completes but the caller sees exit 137. Redirect to a file instead.

## 9. Still open — tickets

1. **The extend kernel is at 31 % of peak even after the fix.** 70.4 TFLOP/s of the 5080's
   ~225 TFLOP/s bf16, and the q8_0 prefix is dequantized to bf16 inside the inner loop, once
   per q-block per layer — `_load_kv`'s fallback branch (`attention.py:440-451`), a
   `tl.load` + scale broadcast + `tl.dot`. The decode path already has a cache-native
   integer-dot alternative (`_Q8_NATIVE_QK`, `attention.py:755-800`) that the extend kernels
   do not. A pipelined rewrite or a native-Q8 QK for extend is plausibly worth another
   1.5–2x, i.e. **1M TTFT ~470–530 s (2,000–2,200 tok/s)**. Estimated effort: kernel work,
   not a launch change.
2. **MoE prefill is now the flat term and it is 33 TFLOP/s.** 29.46 ms per layer at M=8192
   (981 GFLOP), 79 % of the position-independent cost, and the whole of a short-prompt
   prefill. `--nvfp4-backend b12x` did not produce a number in this sweep (flashinfer path
   returned `nan`); worth re-running the M=8192 arm of `bench_nvfp4_moe_kernels.py` with a
   working b12x before assuming triton is optimal at prefill widths — the 2026-09-04 cache
   study chose triton on *decode* evidence.
3. **`128 / 64 / 8 / 1` still shows 148 spill slots**, and that is the launch every non-sm_120
   device keeps. `64/64/8/1` was 435.6 ms here against `128/64/8/1`'s 485.3 ms — but on
   *this* card, whose warp count the rule does not apply there. Sweep the extend tile on an
   sm_89/sm_90 part before extending the cap above 4 warps; this run has no such GPU and the
   change is deliberately not made blind.
4. **Prefill chunk size is not a lever.** Total prefix work is `~N^2/2` independent of chunk
   size, and the flat term is dominated by MoE work that is proportional to tokens, so
   raising `--max-prefill-length` above 8192 buys only the ~106 ms/chunk of fixed overhead —
   under 2 % of a 1M prefill. Recorded so it is not re-litigated.
