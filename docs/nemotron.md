# NVIDIA Nemotron on FreeToken

Serving notes for the `NemotronHForCausalLM` family. The checkpoint rows live in
[`docs/models.md`](models.md); the Switchyard router contract lives in
[`docs/switchyard.md`](switchyard.md).

## Status

| Phase | State |
|---|---|
| Phase 1 — bring-up | done |
| Phase 2 — kernels (Mamba-2 SSD + sm_120 MoE fast path) | in progress |
| Phase 3 — Switchyard compliance | done |
| Phase 3 — soak run | pending |

## Nemotron 3 Super

Nemotron 3 Super uses its native hybrid Mamba-2 / full-attention / latent-MoE
architecture. The NVFP4 release needs about 60 GiB of host RAM for expert banks and
10.3 GiB of resident GPU weights. Its Mamba-2 recurrent state is ~160 MiB per
sequence, so FreeToken serves one concurrent Super session (`single_stream_only`,
which forces `--max-running-requests 1` and a bs=1 decode graph). On WSL,
`--moe-pageable-gpu` keeps the pin-budget overflow banks pageable, stages only their
routed misses through a small pinned buffer, and still executes every ReLU² expert on
GPU. Decode gathers are CUDA-graph host nodes and overlap the shared expert
calculation; idle telemetry saves a model-scoped time-cost ranking that is applied on
the next clean start. A minimal all-GPU-compute launch is:

```
ft serve --model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
  --max-running-requests 1 --moe-backend offload --moe-cpu-layers 0 \
  --moe-pageable-gpu --moe-cache-auto
```

## Nemotron 3.5 Lightning

Nemotron 3.5 Lightning (30B-A3B) is the same `NemotronHForCausalLM` family with a
smaller, non-latent geometry: 52 layers (23 Mamba-2, 23 MoE, 6 full-attention),
hidden 2688, 128 routed experts at top-6 with **ungated ReLU²** experts (up+down
only, I=1856) plus one shared expert.

### Sizing on a 16 GB GPU

- **Host RAM**: the NVFP4 routed-expert banks are 15.4 GiB (≈16.5 GB). With
  `--host-ram-reserve-gb 6` and the ~4 GiB non-bank process footprint, plan on
  ≥ 23 GiB of MemAvailable. Run `python benchmarks/preflight_nemotron_host.py`
  first — it reports MemAvailable/SwapFree, the pin budget, the pinned/pageable
  layer split, VRAM holders, stale `~/.cache/torch_extensions` locks and stray
  workers, and exits non-zero when the host cannot take the load.
- **VRAM**: ~2.3 GiB of resident weights, 47 MiB of Mamba-2 state per sequence
  slot, ≤ 0.8 GiB of `q8_0` KV at 256K tokens, and the rest to the MoE slot cache.
- **Context**: `max_position_embeddings` is 1,048,576, but the tokenizer's
  `model_max_length` is 262,144 — treat 262K as the served ceiling and pin the
  working window with `--max-seq-len-override`.
- **Concurrency**: the 47 MiB state fits many slots, so Lightning is *not*
  `single_stream_only`: up to 16 concurrent requests, with
  `--elastic-initial-requests` starting the recurrent-state/graph working set
  small and growing it on demand.
- **WSL pin quota**: the CUDA host-registration budget is 0.4 × RAM. Below the
  15.4 GiB of banks, the overflow layers need `--moe-pageable-gpu` (which disables
  the decode CUDA graphs). Raising `FREETOKEN_PIN_BUDGET_GB` to ≥ 17 (backed by a
  `.wslconfig` `memory=` large enough to hold it) pins every layer and keeps the
  graphs; the preflight script prints which side of the line this host is on.
- **Expert GEMM**: keep the **`triton`** default — do not pass `--nvfp4-backend auto`
  or `flashinfer` for this checkpoint. Since the 2026-09-05 prefill-GEMM rewrite
  (`benchmarks/results/nemotron35_lightning_5080_moe_prefill_gemm_2026-09-05.md`) Triton
  wins the kernel microbenchmark outright at every width except the widest prefill chunk
  (per MoE layer, cold L2, `--routings 8`): decode **1.64× b12x at M=1, 1.85× at M=8**,
  prefill **1.43× at M=256**, and b12x keeps only 1.29× at M=2048 and **1.34× at M=8192**
  (was 2.3×; the A-operand deinterleave of 2026-09-05 takes that last residual to **1.10×** —
  see *MoE prefill GEMM* below). `auto` still resolves to b12x here (sm_120, ungated relu2,
  `moe_intermediate_size` 1856 ≥ the 1024 threshold) and that resolution is now wrong on
  this geometry. End to end on the offload path b12x already lost before the rewrite
  (task 2B4, `benchmarks/results/nemotron35_lightning_5080_cache_study_2026-09-04.md`):
  32K prefill 5 623–5 777 tok/s on Triton vs 4 528–4 843 on b12x (Triton +19–24 %, two
  rounds), decode +4 % at bs=1, +18 % at bs=8, tied at bs=2/16. On the offload path the
  experts arrive by DMA and are read L2-warm, and 25–88 % of every decode step is expert
  PCIe traffic, so the tensor-core advantage applies only to the shrinking remainder while
  b12x's launch overhead applies to every call. The backend is chosen once, at bank-load
  time (`moe/expert_banks.py`), because it pins the on-GPU weight layout — you cannot take
  b12x for prefill and Triton for decode. (b12x's remaining prefill lead is its
  pre-swizzled tensor-core fragment layout; the ablation that removes the NVFP4 scale from
  the Triton kernel *entirely* is slower than the shipped kernel, so the gap is the operand
  path, not the dequant.) `--nvfp4-backend flashinfer` **can** now be combined with
  `--kv-grow-step-tokens` (the b12x int32 bank was rejected by `VMMTensor`; the integer
  dtypes were added 2026-09-05). `--nvfp4-backend marlin` is rejected at config
  time (its fused kernel assumes a gated `[2I, H]` bank and a silu epilogue), and
  `--moe-backend cpu`/`hybrid` pins the layout back to `triton` because CPU decode reads
  the native ModelOpt rows.
- **MoE backend**: `offload`. `cpu`/`hybrid` *are* available for this checkpoint — the
  CPU MoE executor handles ungated ReLU² NVFP4 banks on plain AVX2+VNNI (no AVX-512
  needed), and `ft bench bw --model nemotron3.5-lightning` measures the kernel at
  66.9 GB/s against a 52.9 GB/s PCIe gather — but the 1.26× ratio is below the 2× hybrid
  threshold, and measured end to end `hybrid` decodes **3.6× slower** than `offload`
  (32.9 vs 118.1 tok/s at bs=1).
- **Expert-cache policy**: `--moe-cache-policy lfu` for the 16-way profile,
  the `lru` default for single-stream. At bs=16 the decode working set is ~61 distinct
  experts per layer (~1 414 across 23 layers) against the ~1 063 slots left after the
  4.45 GiB recurrent-state pool, and LRU degenerates to a **99.6 % miss rate**; LFU pins
  the hot experts, halves that to 51 %, and is worth **1.80×** aggregate throughput
  (93.7 → 168.2 tok/s). Rule of thumb: LFU above a ~15 % miss rate, LRU below it — read
  the rate off the scheduler's idle `MoE decode miss stats` line under
  `--moe-collect-stats`. `FREETOKEN_MAMBA_SSM_DTYPE=bfloat16` is **not** an option for
  shrinking the state pool: the Mamba-2 SSD kernels require an fp32 state pool and reject
  it at config time.

### Launch profiles

P1 — bring-up profile (single stream, no quantized KV):

```
ft serve --model ~/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --max-running-requests 1 --moe-backend offload --moe-pageable-gpu --moe-cache-auto \
  --num-tokens 65536 --memory-ratio 0.85 --max-prefill-length 8192 --host-ram-reserve-gb 6
```

P2 — serving profile (16 concurrent, elastic KV, prefix cache, quantized KV):

```
FREETOKEN_PIN_BUDGET_GB=17 \
ft serve --model ~/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --max-running-requests 16 --elastic-initial-requests 4 --kv-grow-step-tokens 65536 \
  --num-tokens 262144 --max-seq-len-override 131072 --kv-cache-dtype q8_0 \
  --attention-backend triton --moe-backend offload --moe-cache-auto \
  --moe-cache-policy lfu \
  --memory-ratio 0.85 --max-prefill-length 8192 --host-ram-reserve-gb 6 --enable-cache-report
```

`--moe-cache-policy lfu` is the 2B4 recommendation and is worth 1.80× aggregate decode at
16 concurrent requests. With `FREETOKEN_PIN_BUDGET_GB=17` every expert layer is pinned and
`--moe-pageable-gpu` is not needed (keeping the decode CUDA graphs).

**Decode CUDA graphs on this profile (fixed 2026-09-05).** `--elastic-initial-requests`
recaptures the decode graphs at every capacity tier, and the tier's set used to stop at 8
(`_elastic_graph_batch_sizes` returned `(1,2,3,4,8)`); `can_use_cuda_graph` gates on the
largest captured size, so **every decode batch of 9-16 lanes ran eager** — 314 of the 427
decode batches of the `13af13d` soak, 421 of which were taken at elastic capacity 16. The
set is now **dense to 16** (then a 1.33-1.5x ladder, with the tier's capacity always
appended), because on an offload-MoE model a *padded* row is not free — it routes its own
top-6 experts — and padding a 12-lane batch up to a bs-16 graph measured **6.7 % slower than
running it eagerly**, while an exact graph is **7.4 % faster**. The dense set costs 80 MiB
and under a second per resize and does not shrink the expert cache (976 slots either way).
Full study: `benchmarks/results/nemotron35_lightning_5080_decode16_2026-09-05.md`.

Check it in any server log: `Start capturing CUDA graphs with sizes:` must reach the number
`Elastic capacity ... -> N requests` last reported, with no gaps below 16.
`FREETOKEN_ELASTIC_GRAPH_MAX_BS` caps the set for an A/B (`=8` is the old ceiling).

**The non-elastic ladder too (2026-09-05).** Without `--elastic-initial-requests` the graph set
came from `_determine_cuda_graph_bs`, which still built `[1, 2, 4] + range(8, max_bs + 1, 8)`, so
at `--cuda-graph-max-bs 16` a 12-lane batch replayed the bs-16 graph with four dummy rows — and a
dummy row routes its own top-6 experts. `_determine_cuda_graph_bs` now unions
`range(1, min(max_bs, 16) + 1)` into the ladder **for offload-MoE models only**
(`GraphRunner.__init__` passes `offload_moe=moe_offload_cache is not None`); a dense model keeps
the historical list byte-for-byte, and that is pinned by a test. Three alternating repeats of each
arm out of **one binary** at 12 lanes: sparse `[1,2,4,8,16]` **140.43** tok/s aggregate (event-gap
p50 83.0–87.0 ms) against dense `[1..16]` **150.90** (77.2–79.3 ms) — **1.074x**, every dense run
above every sparse run, the same 1.074x the elastic tier measures at 12 lanes. Cost: 11 extra
graphs, ~80 MiB, ~0.8 s of startup. `FREETOKEN_GRAPH_DENSE_BS=0|1` forces either arm.
`benchmarks/results/nemotron35_lightning_5080_misc_tickets_2026-09-05.md` §1.

**Where a 16-lane decode step goes** (same study, §2). It is movement-bound and nothing else
is close: **74 % PCIe expert misses** (23 MoE layers × 31.45 misses × 5.612 MB at a measured
51-52 GB/s gather — the PCIe roofline, so the copy kernel has no headroom), 12 % expert GEMV,
8 % attention at 131K, 1 % Mamba-2, ~4 % launch. The decode attention split rule from
`acc91e9` is still optimal at batch 16 (64 splits, 80 % of the card's bandwidth), the decode
GEMV needs no kernel switch at m≥8 (its grid is 11k CTAs at m=16), and Mamba-2 is batched in
one launch. Reducing the 74 % means moving fewer bytes — the step's working set is ~1,417
expert-layer slots against the 976 the pool leaves at capacity 16 — not a decode-path change.

Quantized KV requires `--attention-backend triton`; bf16 KV with the FlashInfer
backend is the fallback (KV is only +0.75 GiB at 262K). `--tool-call-parser auto`
resolves to `qwen3_coder` and `--reasoning-parser auto` to `qwen3` for this
checkpoint.

Add the Switchyard serving-compliance flags from [`docs/switchyard.md`](switchyard.md)
when FreeToken is fronted by the router.

### KV dtype

Phase 1 A/B (2026-09-04) chose `q8_0` — `fp8_e4m3` flipped first tokens on
cached-prefix reuse 3/6 runs; equal VRAM and reasoning score. See
`benchmarks/results/nemotron35_lightning_5080_2026-09-04.md`.

### Prefill chunk size

`--max-prefill-length 8192` with `--memory-ratio 0.85`. The SSD kernels are in; a 32 768-token
synthetic needle prefills at 5 623–5 777 tok/s end to end and decodes at ~115 tok/s.

Chunk size is **not** a throughput lever: total prefix work is `~N^2/2` whatever the chunk,
and the per-chunk flat cost is dominated by MoE work proportional to the tokens in it, so
raising it above 8192 buys only ~106 ms per chunk of fixed overhead (< 2 % of a 1M prefill).

### Measured throughput (2026-09-04, task 2B4)

Decode through `/v1/chat/completions`, `--moe-cache-auto`, Triton expert GEMM:

| running requests | expert slots | decode miss rate | per-stream tok/s | aggregate tok/s |
|---:|---:|---:|---:|---:|
| 1 | 1 832 | 12.0 % | 143.2 | 143.2 |
| 2 | 1 797 | 16.6 % | 87.9 | 175.3 |
| 8 | 1 483 | 35.3 % | 21.2 | 169.7 |
| 16, `lru` | 1 063 | 99.6 % | 5.5 | 87.4 |
| 16, **`lfu`** | 1 063 | 51.1 % | 10.5 | **168.2** |

### Decode with context (2026-09-05)

Single-stream decode used to fall off a cliff with context because
`kernel/triton/attention.py::decode_launch_config` had no branch for this head shape
(32 Q heads / 2 KV heads / head_dim 128) and took the `kv_splits = 8` fallback: a 16-CTA
grid on an 84-SM RTX 5080, constant in context, so each CTA walked `seq_len / 8` tokens.
The grid now scales with the GPU (64 splits / `BLOCK_N` 64 / 8 warps here). Same server
twice, four turns per length, `ignore_eos`, 128 decode tokens, needle recalled every turn
(`benchmarks/results/nemotron35_lightning_5080_decode_launch_2026-09-04.md`):

| prompt tokens | decode before | decode after | prefill (unchanged) |
|---:|---:|---:|---:|
| 131,088 | 82.8 tok/s | **145.3** tok/s (1.75x) | 3 181 -> 3 290 tok/s |
| 262,160 | 58.7 | **132.4** (2.26x) | 1 934 -> 1 965 |
| 524,304 | 35.4 | **113.6** (3.21x) | 1 073 -> 1 063 |
| 1,040,016 | ~20 (recorded 2026-09-04, unpaired) | **95.8** (single sample) | 575 tok/s, TTFT 1 810 s |

Decode at 131K is now the same rate as the short-context single-stream figure in the table
above (143.2 tok/s), i.e. the curve is flat rather than context-bound, and what remains is
the KV read itself (the kernel sustains 609-722 GB/s of a ~960 GB/s part). To reproduce the
old behaviour for an A/B, start the server with
`FREETOKEN_DECODE_KV_SPLITS=8 FREETOKEN_DECODE_BLOCK_N=32 FREETOKEN_DECODE_NUM_WARPS=4`.

### Prefill with context (2026-09-05)

Prefill was quadratic in context for the same *kind* of reason decode was linear in it: a
launch constant measured on another geometry. `_select_extend_tile`'s `head_dim <= 128` arm
returned a hard-coded `BLOCK_M = 128` tile, while sm_120 takes `num_warps = 4` (a value swept
for the D=256 consumer tile), so the extend kernel's `BLOCK_M x BLOCK_DV` fp32 accumulator
needed 128 registers per thread and **spilled — 396 spill slots against 14 at `BLOCK_M = 64`**,
28.6 TFLOP/s against 70.4. `extend_launch_config` now caps `BLOCK_M` by that accumulator's
register budget (`_extend_block_m_cap`), which moves only the arm that was never measured;
every `head_dim >= 256` branch and every 8-warp device keeps its launch exactly.

Same server twice, cold prefill, needle recalled in every run
(`benchmarks/results/nemotron35_lightning_5080_prefill_profile_2026-09-05.md`):

| prompt tokens | prefill before | prefill after | TTFT before -> after |
|---:|---:|---:|---:|
| 131,088 | 3 230 tok/s | **5 288** tok/s (1.64x) | 40.6 s -> **24.8 s** |
| 262,160 | 1 965 | **3 683** (1.87x) | 133.4 s -> **71.2 s** |
| 1,040,016 | 573–576 (recorded 3x) | **1 307** (2.28x) | 1 810–1 824 s -> **795.8 s** |

Per 8K chunk the cost is `861 ms + 10.4e-3 * prefix` after the fix (r2 0.999 over 127 chunks
of the 1M run) against `836 + 25.8e-3` before — the **intercept does not move**, so the change
is confined to attention, and the non-attention position-dependent term solves to
**0 ± 0.2e-3 ms/token, i.e. ±13 s of a 1M prefill**: KV grow, page allocation and the
`O(prefix)` page-index build are not measurable. The Mamba-2 SSD scan is flat in position by
construction (1.068 ms/layer/8K chunk, 0.2 % of a 1M prefill); the flat term was **79 % MoE
expert GEMMs** (23 layers x 29.5 ms at M=8192), which is what a short-prompt prefill costs.
Attention is still 91 % of the last chunk at 1M. The MoE half of that flat term was cut on
2026-09-05 — see *MoE prefill GEMM* below — taking the per-chunk flat cost from ~861 to
~586 ms.

The follow-up ticket — *"the extend kernel is at 31 % of peak, dequantizes q8_0 in the inner
loop, and a native-Q8 QK is worth another 1.5–2x"* — was measured on 2026-09-05 and **closed
negative** (`benchmarks/results/nemotron35_lightning_5080_prefill_q8_2026-09-05.md`). The
5080 does **123.0 TFLOP/s** bf16 through cuBLAS and 118.4 through Triton's own `tl.dot`, not
the 225 of the spec sheet, so the extend kernel is at **57–60 % of the achievable rate**, not
31 %; its int8 GEMM rate is **128.0 TOP/s = 1.04x bf16**, so a native int8 dot cannot make the
QK cheaper. Removing the q8_0 dequant *entirely* — the same kernel over a bf16 KV pool — is
worth **1.206x, flat at 131K/262K/524K/1M**, and that is the hard ceiling. A native-Q8 QK
cannot reach it: q8_0's scale is per 32 elements of `head_dim`, so folding it after the dot
costs `BLOCK_M * BLOCK_N * D/QBLOCK` multiplies against `BLOCK_D * BLOCK_N` for dequantizing in
place — cheaper only when `BLOCK_M < QBLOCK`, which is why decode's `_Q8_NATIVE_QK`
(`BLOCK_M = 16` heads) pays and extend's 64-token tile does not; and the V scale sits on the
PV *reduction* axis, where no fold exists. The kernel is unchanged. The next prefill target was
the MoE (33 TFLOP/s of the same 123, i.e. 3.7x off, against attention's 1.7x) — closed below.

To reproduce the old behaviour for an A/B, start the server with
`FREETOKEN_EXTEND_BLOCK_M=128 FREETOKEN_EXTEND_BLOCK_N=64 FREETOKEN_EXTEND_NUM_WARPS=4
FREETOKEN_EXTEND_NUM_STAGES=1`. Both the decode and the extend launch are logged once at
startup (`Triton decode launch: ...`, `Triton extend launch: ...`).

### MoE prefill GEMM — 1.74x on the kernel, 1.20x end to end at 131K (2026-09-05)

Ticket §9.2 of the prefill profile ("MoE prefill is the flat term and it is 33 TFLOP/s")
is closed: **29.47 -> 16.95 ms per MoE layer at M=8192, 57.9 TFLOP/s = 49 % of the card's
118 TFLOP/s Triton `tl.dot` ceiling** (was 28 %).
`benchmarks/results/nemotron35_lightning_5080_moe_prefill_gemm_2026-09-05.md`.

It was **not** the tile — the whole tuner grid was worth 1.12x. `_prefill_nvfp4_moe_kernel`
loaded one e4m3 block scale **per packed byte**, and one scale covers 16 k-values = 8 bytes,
so every value was fetched eight times through 16 KB of shared memory per CTA. Loading the
distinct `[BLOCK_SIZE_KB/8, BLOCK_SIZE_N]` rows and broadcasting is **1.73x on its own** and
drops the kernel's shared memory 28 KB -> 12 KB, which is what makes `BLOCK_N = 256`
affordable; the retuned M=8192 tile is `64/256/16/8/4/3`. A second term, `cvt.rn.f16x2.e2m1x2`
(the Blackwell hardware FP4 decode, gated by `e2m1_native_cvt_cx()`), replaces the ~14-op bit
construction for a further 1.04x. Both are **exact**: the output is bit-identical
(`0.000e+00`) to the old kernel at every tile swept, and the native decode matches
`_e2m1_decode` on all 256 packed byte values including `-0.0`.

The general rule, worth carrying to any other block-quantized kernel: **a per-block scale is
a `K/QBLOCK`-sized tensor — load it at that size and broadcast, never at the tile's K.**

Two servers, same flags, same needle prompts, the before arm a worktree at `e4070da`:

| prompt tokens | TTFT before -> after | prefill before -> after | speedup |
|---:|---|---|---:|
| 131,088 | 26.69 s -> **22.29 s** | 4,911 -> **5,882 tok/s** | **1.198x** |
| 262,160 | 75.31 s -> **66.60 s** | 3,481 -> **3,936 tok/s** | **1.131x** |

The needle is recalled in every run of both arms and the answers are byte-identical, as is a
64-token greedy continuation of a short prompt. Δ TTFT per chunk is 275 / 272 ms against the
microbench's predicted `23 x (29.47 - 16.95) = 288` ms — the engine sees 95 % of the kernel.
Because the MoE GEMM is the *flat* term the win scales inversely with prompt length:
~1.47x projected on a single 8K chunk, 1.20x at 131K, 1.13x at 262K, ~1.05x at 1M.

`FREETOKEN_NVFP4_PREFILL_{BLOCK_M,BLOCK_N,BLOCK_KB,GROUP_M,NUM_WARPS,NUM_STAGES}` force the
prefill tile (the twins of `FREETOKEN_EXTEND_*` / `FREETOKEN_DECODE_*`), and
`FREETOKEN_NVFP4_NO_NATIVE_CVT=1` forces the arithmetic FP4 decode.

#### The A operand — a further 1.215x on the kernel, 1.074x at 131K (2026-09-05)

The follow-up ticket ("neither activation load vectorizes") is closed and **shipped on by
default**. `_prefill_nvfp4_moe_kernel` walks K in *packed bytes* and issues one `tl.dot` per
nibble, so against a plain `[M, K]` bf16 activation both `a_ptrs_lo` / `a_ptrs_hi` are stride-2 on
the contiguous axis — neither gather vectorizes and A's K span is read twice per K-block. A
`DEINTERLEAVED_A: tl.constexpr` arm plus a host prepass that rewrites A into an **even-k plane
followed by an odd-k plane** (`a.view(M, K//2, 2).permute(0,2,1)`) makes both gathers unit-stride.
The per-`tl.dot` reduction order is unchanged, so it is a numerics no-op and measures as one.

| M | tree (old) | deint (incl. prepass) | speedup | of which prepass | max abs diff |
|---:|---:|---:|---:|---:|---|
| 256 | 1.271 ms | **1.091** | 1.165x | 0.015 ms | 0.000e+00 |
| 1024 | 2.937 | **2.595** | 1.132x | 0.039 | 0.000e+00 |
| 2048 | 5.103 | **4.459** | 1.144x | 0.142 | 0.000e+00 |
| 4096 | 9.171 | **7.782** | 1.178x | 0.279 | 0.000e+00 |
| **8192** | **16.960** | **13.961** | **1.215x** | 0.551 | 0.000e+00 |

At M=8192 that is **70.3 TFLOP/s, 59 % of the card's Triton `tl.dot` ceiling**, and the residual
gap to b12x falls **1.34x → 1.10x**. End to end on a cold 131K synthetic-needle request (two pairs,
`FREETOKEN_NVFP4_PREFILL_DEINTERLEAVE_A` the only difference): prefill **6,124.7 → 6,577.8 tok/s
(1.074x)**, engine average 5,728.6 → 6,177.6 (1.078x), **TTFT 21.6 s → 19.8 s**, decode unchanged
(140.5–140.9 tok/s — the decode GEMV kernels were not touched, nor was the fp8 sibling), needle
PASS in all four runs. `FREETOKEN_NVFP4_PREFILL_DEINTERLEAVE_A=0` disables it for an A/B.
Still on the table: gemm2's A *is* gemm1's output, so its prepass is removable by having gemm1's
store emit the two k-planes directly (~0.3 ms of the 0.551 at M=8192).
`benchmarks/results/nemotron35_lightning_5080_misc_tickets_2026-09-05.md` §2.

### Speculative decoding — `--speculative ngram`, shipped 2026-09-05

Prompt-lookup (n-gram) speculative decoding was measured and refused
(`benchmarks/results/nemotron35_lightning_5080_ngram_spec_2026-09-05.md`), then unblocked by the
extend-path MoE fix and built
(`benchmarks/results/nemotron35_lightning_5080_ngram_spec_impl_2026-09-05.md`). What follows is
the whole chain: why it was refused, what the blocker actually was, and what shipped.

**Acceptance is not the problem.** Under greedy decoding a prompt-lookup drafter is verified
against exactly the greedy continuation, so its acceptance is computable offline from ordinary
transcripts. On a copy-heavy agent tool-output prompt (a file pasted in, "output the complete
updated file") an **8-gram** drafter reaches **λ = 3.6 accepted tokens per step at 93–97 %
per-token acceptance**, while code and prose stay within ±0.5 % of neutral because the drafter
almost never fires on them. The standard n = 3 setting is *wrong here*: it fires on 12 % of code
steps and is right 23 % of the time, costing 12–14 %. **When verification is expensive, draft for
precision, not recall.**

**Mamba-2 state rollback is not the problem either.** The verify forward points
`fla.cache_indices` at the request's already-reserved ping-pong scratch slot, so the live state is
never advanced speculatively and there is nothing to roll back; each mixer caches its scan inputs
(`x`, `dt`, `B`, `C`, `conv_in` ≈ 24.7 KiB per layer per token, **~5 MiB at k = 8**) and a commit
pass runs one varlen SSD scan over the accepted j positions (~0.2–0.5 ms). No 47 MiB state copy.

**The blocker is that there is no cheap multi-token forward.** The only path that carries k > 1
query tokens for a running request is the prefill/extend path, and that path cost **290 ms of
host time per forward, flat from 1 to 32 tokens**, against 33.9 ms for a 1-token forward on the
decode path. Per-mixer attribution puts **267 ms of it in the 23 MoE layers — 11.6 ms per MoE
layer per forward, independent of token count**. That is 36–42× a 6.9–8.0 ms graphed decode step,
so break-even would need λ ≈ 40 against a ceiling of k + 1 ≤ 17.

**Why (measured 2026-09-05, ticket 1 closed):** `_prefill_routed` streams *every* expert of the
layer into its double buffer on *every* forward — nothing in that movement path reads `topk_ids`,
which is exactly why the cost is flat in M. That is 128 × 5.612 MB = **718 MB per layer** and
**16.5 GB per forward** (the whole 15.4 GiB expert bank set), and 16.5 GB / 267 ms = **61.9 GB/s**,
a saturated PCIe 5.0 x16 link. A 1-token extend routes 6 experts per layer and moves 128: a
**21.3× waste, in bytes, not host time.** It is invisible in normal serving because an 8 192-token
chunk does ~861 ms of GPU work and the host runs ahead of it — and it is why **prefill chunks below
~3K tokens go host-bound**, a second reason not to lower `--max-prefill-length`.

**The fix** is `--moe-extend-cache-tokens` (default **64**, 0 disables): an extend forward carrying
at most that many tokens takes the *decode* path — `ensure_experts` + `copy_missing` fetch the
experts those tokens route to and only the ones not already resident, and the NVFP4 decode GEMV is
m-general (grid `(m·top_k, cdiv(N, BLOCK_N))`). 64 is the crossover: the measured distinct-experts
curve `D(m) ≈ 6.2·m^0.75` reaches num_experts at m ≈ 57, above which the cached path degenerates
into the full-layer stream and also evicts the decode working set. Restricted to the NVFP4 bank
layouts, GPU decode target and pinned layers; everything else keeps the full-layer stream, as does
every M above the threshold.

**Measured** (2026-09-05, same binary both arms):

| m | forward before | forward after | MoE ms/layer before → after |
|---:|---:|---:|---|
| 1 | 282.7 ms | **27.7 ms** | 11.44 → **0.42** |
| 8 | 282.7 ms | **30.2 ms** | 11.42 → **0.47** |
| 32 | 282.5 ms | **30.9 ms** | 11.37 → **0.48** |

9.2–10.2× on the forward, 23.6–27.3× on the MoE, and the MoE is no longer the extend forward (at
m = 32 it is 11.1 ms against Mamba-2's 15.6). 131K prefill 5,059 → 5,105 tok/s (+0.9 %, noise),
needle recalled in both arms, and a long prompt whose last chunk is short is greedy token-identical.
Write-up: `benchmarks/results/nemotron35_lightning_5080_extend_moe_2026-09-05.md`.

**The threshold was re-measured on 2026-09-05 and stays 64**
(`benchmarks/bench_extend_moe_threshold.py` + `benchmarks/extend_moe/run_threshold.sh`,
one model load, 7 timed extends per cell on a
4,096-token base with a fresh tail per call). Wall time per extend forward, stream vs cached:
64 → 281.1 / **249.4** ms, 80 → **285.3** / 294.8, 96 → **284.0** / 330.5, 128 → **274.1** / 370.3.
**The crossover is between 64 and 80**, i.e. exactly at the shipped default. The mechanism is the
miss column: the cached path fetches only the routed experts, but by m=128 it is fetching 107.8 of
128 rows per layer anyway, at the scattered gather rate (52.9 GB/s) instead of the contiguous
stream rate (61.9 GB/s), and it pays a decode GEMV where the stream pays a grouped GEMM. The
following decode burst shows no eviction penalty worth the name (6.7–7.5 ms/step either way).
Read wall (or the CUDA-event GPU span, which agrees to ~2 %), never host time: the stream arm
blocks the host on its PCIe copies while the cached arm returns in ~60 ms and leaves the GPU
gathering for another ~250.

**There is also a hard servable width, and it is below 256 tokens.** `ensure_experts` reaches
flashlib's `lru_ensure`, whose `_seq`/`_insert` build a `[BLOCK_K, BLOCK_K]` dedup matrix at
`BLOCK_K = next_pow2(query.numel())`; Triton caps a tensor at 1,048,576 elements, so `BLOCK_K` can
never exceed 1024 and the query is `num_tokens * top_k` ids — a ceiling of **m ≤ 170 at top-6**.
`--moe-extend-cache-tokens 256` used to kill the engine mid-forward with
`ValueError: numel (4194304) exceeds triton maximum tensor numel (1048576)`; `use_cached_extend`
now refuses when `topk_ids.numel() > 1024` and the forward falls back to the full-layer stream,
which is faster at every width above the crossover anyway. Widths that *do* compile are still
pathological — the m=128 cell (`BLOCK_K = 1024`) cost **22 minutes of one-off Triton JIT** before
it ran, so do not raise the flag on a live server.
`benchmarks/results/nemotron35_lightning_5080_misc_tickets_2026-09-05.md` §3.

This also corrects the Phase 4 (MTP) write-up: its 1.63× verify cost came from a bs=2 *decode*
step, but a real verify step takes the *extend* path, so MTP's projection was ~25× optimistic
about its own verify step.

#### What shipped

`--speculative ngram` (off by default), with `--spec-ngram-n 8`, `--spec-draft-len 8` and
`--no-spec-adaptive`. **Greedy and single-stream in v1**: a request with `temperature > 0`, any
step with more than one running request, a multimodal prompt, a hidden-state probe, an SWA model
or `--cache-type naive` all silently take the ordinary decode path — the flag never refuses a
request, it just does not speculate on it.

A step drafts `k` tokens from the request's own prompt + output (most recent occurrence of the
trailing 8-gram), runs **one extend forward over `k + 1` positions** keeping every logits row,
accepts the longest agreeing prefix plus the bonus token, and commits. Mamba-2 state is verified
into a private scratch slot (never the live one) and the accepted prefix is committed with one
varlen SSD scan per layer; a self-check (`FREETOKEN_SPEC_CHECK_COMMIT=<n>`) shows that replay
reproduces the forward's own state to **0.000e+00** on both the recurrent block and the conv
window. Rejected KV pages are returned by `CacheManager.free_spec_tail` — without it every partial
rejection leaked `k - accepted` pages — and rejected tokens never reach the host token list, so the
prefix cache cannot see them.

#### The verify step, made cheap (2026-09-05, second pass)

`..._ngram_spec_fast_2026-09-05.md` closed the two cost tickets and corrected the record on a
third.

**A verify step went from 54.0 ms to 35.6 ms** (copy class, 100+ samples per arm, measured per
phase via `/v1/stats["scheduler"]["spec"]["cost_ms"]`):

| phase | before | after |
|---|---:|---:|
| batch preparation | 0.80 ms | **0.34 ms** |
| forward (host launch / GPU) | 50.6 / 54.7 ms | **30.6 / 36.4 ms** |
| state commit | 1.24 ms | **0.18 ms** |
| end-to-end step | **54.0 ms** | **35.6 ms** |

- **One SSD scan for all 23 layers.** Mamba-2 heads are independent and every Nemotron-H mixer
  has the same `(head_dim, state_size, heads_per_group)`, so the layer axis folds onto the head
  axis: 23 × 64 heads is one 1 472-head sequence, `A` and `dt_bias` concatenate, and `D` is
  dropped because it only feeds the scan output the commit discards. **~280 kernel launches →
  ~11, 7.12 → 0.45 ms of host time, bit-exact** (0.000e+00 at eight (m, n) shapes;
  `benchmarks/check_spec_fused_commit.py`, weightless).
- **The verify batch is built from its own fixed shape** instead of through
  `Scheduler._prepare_batch` — persistent device buffers for positions and rows, metadata cached
  by (extend width, state slot), and no `Sampler.prepare` (the verify forward is greedy and runs
  no sampler).
- **Engagement is decided post-drain**: the pre-drain peek now asks whether any indexed n-gram
  *starts with* the n−1 tokens already held, a strict superset that cannot miss a burst entry,
  and the exact test runs after the drain.

**Two corrections to the first write-up, both from measurement:**

1. **The burst-entry hysteresis was not a factor of 4.** Replaying the baseline transcript through
   both peeks (`benchmarks/spec_engage_replay.py`, CPU) puts the old one at draft rate 0.484 /
   λ 4.59 and the new one at 0.484 / 4.67 — **2 %**. The 0.079-against-0.353 that motivated the
   ticket was **stream variance**: speculation perturbs its own output, the copy prompt opens with
   a few hundred tokens of reasoning before the verbatim copy starts, and where that transition
   lands inside a 1 024-token window decides the draft rate. Single copy-class arms of the same
   binary span **1.04× to 1.67×**.
2. **A graph-captured verify forward is a no.** At m = 9 the eager forward is **30.6 ms of host
   launch against 36.4 ms of GPU**, and at 131K 31.0 against 91.8 — the Python already runs
   underneath the GPU. (Same shape as the 16-lane decode finding.)

**Measured, and projected on a fixed transcript with the measured per-step costs** — the second is
the load-bearing number, because it is not subject to the stream lottery:

| class | end-to-end arms (off / before / after) | fixed-transcript projection, before → after |
|---|---|---|
| code | 139.1 / 138.8 / 134.8 | 0.99× → 0.99× |
| prose | 146.5 / 146.4 / 143.2 | 0.99× → 0.99× |
| copy | 138.9 / 144.9 / **210.5** | **1.11× → 1.61×** at k = 8, **1.88×** at k = 16 |
| 131K needle | 131.9 / 107.2 / 110.8 | still a regression at every k (ratio ~12×) |

**`--spec-draft-len 16` is worth another ~1.17× on copy-heavy traffic** and is neutral (0.99×) on
code and prose: acceptance does not decay with the draft length on verbatim copy (15.7 of 17
tokens kept) while the verify step grows only ~1.8 ms per drafted token. **The default stays 8**
(confirmed 2026-09-05, below) — set 16 explicitly for copy-heavy agent traffic and nothing else.
**`--spec-draft-len 4` is a regression** and should not be used.

**Why 16 is not the default (measured 2026-09-05, NO-GO).** The criterion was "raise it only if
the 131K non-copy case stays within 2 % of spec-off". At 131K (123,612 prompt tokens, needle) k=16
measures **0.870x** of off and k=8 **0.898x**, against a control spread of 1 % — 13 % below, so it
fails outright. Two costs, and the second is the reason: a gate probe at 131K costs
`_GATE_MIN_SAMPLES = 2` verify steps (163 ms at k=8, 279 ms at k=16 against a 10.4 ms decode
step), and at k=16 the break-even gate **never closed** — `declined_uneconomic` 0 of 55 peeks, vs
16 at k=8 — because a longer draft raises `emit` (9.0 tokens/verify) about as fast as it raises
`verify_ms`, so the drafter keeps paying near-break-even steps for the whole generation. Raising k
weakens the one mechanism that is supposed to make long-context speculation free. The short-context
copy arms of that run are *not* a copy verdict (both landed at 0.98x of off with draft rates of
0.02–0.04 — the copy-class lottery of `..._ngram_spec_fast_2026-09-05.md` §1); what they do
contribute is the step cost, 35.8 ms at k=8 vs 49.8 ms at k=16. Pinned by
`tests/scheduler/test_spec_ngram.py::test_spec_draft_len_default_stays_8`.
`benchmarks/results/nemotron35_lightning_5080_misc_tickets_2026-09-05.md` §4.

**Two things to know before enabling it.**

1. **It is not token-identical to non-speculative greedy decoding, and cannot be.** The verify step
   argmaxes *extend*-path logits and commits state with the *SSD scan*, where a decode step uses
   the graphed decode kernels and the recurrent step — different reduction orders. The control arm
   reproduces the baseline exactly, so the engine is deterministic; the speculative arm diverges at
   token 40–71 of 1 023 on three of four prompts (identical on the fourth, and the 131K needle is
   recalled in both arms). Any multi-token verification scheme on this engine carries this.
2. **Long context regresses.** A verify step's extend attention reads the KV history once per query
   token, so at 131K it costs ~92 ms against a ~7.6 ms decode step — **~12×**, against ~5× at
   short context — and `k + 1 = 9` cannot beat that. The decoder therefore measures its own
   verify/decode ratio online and stops drafting when `accepted + 1` can no longer pay for it,
   re-probing every 16 gated steps. There is deliberately **no** context-length threshold: the
   ratio depends on KV dtype, attention backend and acceptance. The gate cannot refund the
   *measurement*, though — pricing itself costs two verify steps, which on a short generation at
   131K is the whole −11 %.

A 16-way passthrough soak passes with zero errors in both arms and flat p50 / request count; its
p95/p99 tail differs but one 10-minute pair cannot separate that from variance (write-up §11).

Still open, by upside: a cheaper long-context verify step — the extend attention reads the whole KV history once per
query token, and a fused multi-query kernel that reads it once for all `m` is the shape of the
fix, and the only thing that makes speculation pay above ~64K — batched (bs > 1) speculation, and
sampling support. The 46-launch commit, the batch-prep overhead, the burst-entry hysteresis, the
graph-captured verify forward and the draft-length default are all closed above.

### 1M single-session profile

Many long-lived agent sessions, each up to 1M tokens, few decoding at any instant.
There is no live host-KV tier for active sequences (every decode step reads the whole
KV; 3 GB/step over PCIe ≈ 100 ms/token — not viable). Instead:

- Growable KV (VMM segments, `--kv-grow-step-tokens`) funds KV from the expert cache
  as sessions grow.
- Session spill (`--session-spill-ram-gb`, `--session-spill-dir`) checkpoints idle
  sessions' KV + Mamba state to RAM then NVMe and restores exactly on the next turn.
- KV per 1M session ≈ 3 GB at fp8 + 47 MiB Mamba state → 2–3 concurrently decoding 1M
  sessions on the 5080; ~4 spilled sessions fit in the remaining host RAM
  (40 GB − 16.5 GB banks − process).

User decision (2026-09-04): the 1M profile runs **one** resident session
(`--max-running-requests 1`); all other sessions queue and are served in sequence via
spill/restore. The 16-way P2 profile remains for short-context Switchyard traffic.

Residency policy (task 3E, decided 2026-09-04):

- **No spill while the queue is empty.** The resident session's KV + Mamba state stays
  in VRAM until another session's request needs the slot; only then is it checkpointed
  (on demand, not on an idle timer). TTL-based release must not evict a resident
  session that nobody is waiting on. When an admission fails for lack of KV/state
  slots, the oldest idle reclaimable lease is reclaimed (re-run at the
  admission-failure point, not only at message receipt); the grace timer remains only
  as a very long safety (configurable, default off/∞).
- **Retention by capacity + age.** Checkpoint lifetime is decoupled from lease TTL:
  `--session-spill-limit-gb` (default 50) is the total across RAM+disk, and the
  oldest-by-last-use checkpoints are evicted first when the cap or the filesystem
  guard is hit instead of refusing. TTL closes the lease but keeps the checkpoint; a
  later request with the exact prefix restores it.
- **Survive restart.** Checkpoints are keyed on disk by session id + prompt-prefix
  hash + K/V layout fingerprint (manifest JSON next to chunks); startup scans the
  spill root, adopts valid records, and deletes stale/foreign ones; shutdown no longer
  rmtrees. Restore requires a fingerprint match and a matching token prefix.

Decisions from the 1M gate (tasks 3F/3G, 2026-09-04):

- **Look-ahead promotion.** NVMe restore measured ~1.3 GiB/s (0.98 s at 393K, ~2.5 s
  projected at 1M) against ~0.13 s from RAM, so while the resident session decodes, a
  queued session whose checkpoint is on disk has it read into the RAM tier on one
  background thread (one in flight, cancelled when the request goes away). It respects
  the RAM budget and the host reserve, and makes room by demoting other RAM checkpoints
  to disk (LRU) -- never the resident or restoring session's own. `--session-spill-ram-gb`
  now defaults to **4** so one 1M look-ahead fits beside the checkpoint being written.
- **Partial-prefix restore.** A checkpoint carries the recurrent state at several prefix
  boundaries, not only its end: the final state plus the earlier x`track_chunk_size`
  snapshots the radix tree still holds for that session (so the count follows
  `--linear-state-slots`), thinned to at most 8 entries spaced by
  `--session-spill-state-stride` (65536) at ~47 MiB each. A restore installs the longest
  matching prefix that ends on a stored boundary and re-prefills the tail, instead of
  discarding 1.24 GiB over one retokenized `</think>`. Records written before this
  (manifest v1) are deleted on adoption.
- **Prefill-time state capture** (`--session-spill-capture-states`, auto). At
  `--linear-state-slots 5` the chunk commit cannot reserve a replacement ping-pong slot,
  so it donates no snapshot at all and the radix-derived boundaries above are empty --
  the partial restore degrades to whole-or-nothing exactly where it matters. So while a
  turn prefills, the boundary snapshot the forward already wrote is copied to the host
  every `--session-spill-state-stride` tokens (async D2H on a side stream into pinned
  staging, synchronized before the spill; the current stream waits on the event so the
  next forward cannot overwrite the frozen slot mid-transfer). At most 8 are held --
  ~376 MiB per resident turn on Lightning -- merged across the session's turns, unioned
  with any radix-held boundaries at spill time, and freed at the spill or when the lease
  closes. Auto-on below 6 state slots per running request, i.e. only when the free
  snapshots are scarce.
- **A restore blocked by the resident session is retried, not discarded**, at the
  admission-failure point that spills its competitor.
- The `nemotron_v3` reasoning parser now drops a `<think>`/`</think>` that arrives
  outside a reasoning block instead of streaming it as content -- that stray token is
  what echoed back into the next turn's prompt and broke the prefix match.

Measured sizing (task 2B4, 2026-09-04 — see
`benchmarks/results/nemotron35_lightning_5080_cache_study_2026-09-04.md`):

```
FREETOKEN_PIN_BUDGET_GB=17 \
ft serve --model ~/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --max-running-requests 1 --max-seq-len-override 1048576 --num-tokens 1048576 \
  --kv-grow-step-tokens 131072 --kv-cache-dtype q8_0 --attention-backend triton \
  --moe-backend offload --moe-cache-auto --linear-state-slots 5 \
  --memory-ratio 0.85 --max-prefill-length 8192 --host-ram-reserve-gb 6 \
  --session-spill-ram-gb 12 --session-spill-dir <nvme>
```

- **`--linear-state-slots 5` is the accepted floor** at `--max-running-requests 1`
  (`4·mr + 1` for `hybrid_radix`); 3 and 4 are rejected at startup. The default is 9, so
  pinning 5 returns ~188 MiB (≈ 35 expert slots) to the MoE cache.
- **Prefer 6 when two conversations alternate.** Five slots are padding + live + 2 ping-pong
  + *exactly one* idle session lease, so a second session's first turn finds the pool full
  and the scheduler spills the idle lease on demand to get its snapshot slot (correct — it is
  the 3E residency policy — but it costs a checkpoint + restore per alternating turn, and at
  1M that is GiB of KV). One extra slot (47 MiB) lets an idle lease and a live request
  coexist. Before 2026-09-04 this shortage was fatal rather than slow: the chunk commit's
  unguarded `pool.alloc(1)` raised `LinearStatePool exhausted` and killed the scheduler.
- **Expert slots vs KV growth**: each committed 131 072-token KV step costs ~0.40 GiB and
  ~76 expert slots. Auto starts at 1 786 slots and steps 1 663 (262K) → 1 586 (393K) →
  1 510 (524K) → 1 434 (655K); a full 1M session extrapolates to ~1 180 slots (rate 0.40).
  **VRAM is not the blocker for one 1M session.**
- **Throughput** on a growing synthetic-needle prompt: 131K prefill 3 007 tok/s / decode
  72.6 tok/s; 262K 1 790 / 51.8; 524K 997 / 32.0. **Both halves of this curve are
  superseded** — see "Decode with context" and "Prefill with context" above. Prefill is
  still quadratic in context (it must be: every chunk attends over the whole prefix), but
  the constant is 2.46x smaller since 2026-09-05.
- ~~**Coherence caveat**: the needle passes at 131K but is missed at 262K and 524K.~~
  **Retracted 2026-09-04.** Those 2B4 runs predate both the chat-endpoint gate (`ec54e21`) and
  the Mamba-2 `dt`-floor fix (`3ac79ec`). Re-run through `/v1/chat/completions` at depth 0.50
  the needle **passes at 262,160 and at 524,304 tokens** (1,925 / 1,064 tok/s prefill, 56.3 /
  34.5 tok/s decode — both decode figures predate the launch-config fix below), and a
  1,040,080-token conversation recalls its needle as well —
  `benchmarks/results/nemotron35_lightning_5080_1m_sessions_2026-09-04.md`.
- **Recall at 262K / 524K / 1M, stated properly (cross-engine oracle, 2026-09-05).** A single
  needle at one depth is too coarse to describe this model. The twelve-needle suite
  (`benchmarks/oracle_cross_engine.py`, six needles each with a near-duplicate `register` twin,
  asked three ways) run against **both FreeToken and llama.cpp on byte-identical prompts** gives:

  | | direct `key→code` | combined | reverse `code→key` | control |
  |---|---|---|---|---|
  | FreeToken 262K | 5/6 | 2/6 | 6/6 | denied |
  | llama.cpp 262K | 3/6 | 2/6 | 6/6 | denied |
  | FreeToken 524K | 1/6 | 0/6 | 6/6 | denied |
  | llama.cpp 524K | 2/6 | 1/6 | 6/6 | denied |
  | FreeToken 1M | 1/6 | 0/6 | 5/6 | denied |

  **The state holds every needle at every length — `code → key` recovers 6/6 at 524K on both
  engines and 5/6 at 1M — but `key → code` collapses between 262K and 524K, on llama.cpp Q4_0
  exactly as on FreeToken NVFP4.** At 524K four of the six direct probes return the same key's
  near-duplicate twin **byte-for-byte in both engines**. So this is *addressing under
  interference*, a model property, not a FreeToken retention or prefix-cache defect: zero
  `retention` and zero `selection` classes on either engine at any length.
  `benchmarks/results/nemotron35_lightning_5080_oracle_2026-09-05.md` §§9, 11, 12.

  Practical reading for an agent workload: at ≥524K do not rely on "quote the value stored
  under key X" when a near-duplicate key exists in the context; a `value → which key` or
  a disambiguated composite question is far more reliable. Arithmetic over two retrieved
  values is the weakest axis on both engines at every length.
- **llama.cpp cannot serve this model at 1M on a 16 GiB card** (relevant only as the oracle's
  second engine): `-c 1052672` reserves the whole card at *every* `--n-cpu-moe`, and even with
  all 23 MoE blocks on host RAM the written KV outgrows residency at ~570K tokens and chunk
  cost then climbs 11.5 s per 4,096 tokens — ≈20 h of prefill. FreeToken serves the same 1M
  prompt at 573–576 tok/s. Same results file, §10.
- `--nvfp4-backend flashinfer` + `--kv-grow-step-tokens` used to die at init with
  `unsupported VMM tensor dtype: torch.int32` (growable KV allocates the slot cache as VMM
  tensors; the b12x banks include an int32 bank). **Fixed 2026-09-05** — `int16`/`int32`/
  `int64` added to `VMMTensor._DTYPE_NAMES` and `parse_dtype`, verified by an actual server
  start. `triton` is still the recommendation, on speed.

Gate closed 2026-09-04
(`benchmarks/results/nemotron35_lightning_5080_1m_sessions_2026-09-04.md`): three sessions
grown to 655K each and one to **1,039,989** tokens, needle recalled at every length; the 1M
session spilled on demand (3.53 GiB to NVMe in 2.980 s, 1.18 GiB/s) and, after the server was
restarted, restored by the new process (**2.681 s, 1.32 GiB/s**, whole prefix, byte-identical
answer). Capacity/age eviction confirmed at a 1.6 GiB cap (oldest-by-`last_used_at` first;
survivors restore; a record larger than the cap is refused, not thrashed).
Note `--session-spill-ram-gb 0` if you want the NVMe tier exercised: at the 4 GiB default a
3.5 GiB checkpoint stays in RAM and does not survive a restart, and a *resident* session is
never checkpointed at all (by design — spill is on demand), so a restart loses it.
