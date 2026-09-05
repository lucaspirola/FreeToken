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
  (was 2.3×). `auto` still resolves to b12x here (sm_120, ungated relu2,
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

**Measured** (1 023 output tokens, one model load, with a second non-speculative control arm):

| class | off | on | control off2 | draft rate | accepted/drafted | speedup |
|---|---:|---:|---:|---:|---:|---:|
| code | 135.8 | 140.4 | 136.8 | 0.005 | 0.83 | **1.03×** |
| prose | 138.3 | 141.9 | 141.9 | 0.007 | 0.25 | **1.02×** |
| copy (agent tool output) | 135.8 | 136.9 | 133.6 | 0.079 | 0.80 | **1.01×** |
| 131K needle | 87.3 | 77.3 | 86.9 | 0.036 | 0.63 | **0.89×** |

tok/s at bs=1; run-to-run spread on the non-speculative arms is 1.6–3.5 %, and an ungated control
run reached 1.11× on the copy class with the same drafter statistics, so **read the copy-class win
as ~1.05× with a spread of several points.** The n = 8 precision gate is what keeps code and prose
from regressing; n = 3 (the published prompt-lookup setting) costs 12–14 % on exactly those two.

The win is well short of the 1.63× the go/no-go projected, and the shortfall is measured, not
mysterious: the draft rate is **0.079** against the offline replay's 0.353 (engagement is decided
one token stale, so a copy burst is entered one step late), and an end-to-end verify step costs
**~52 ms** against the ~30 ms extend forward inside it (the commit issues 46 eager kernel launches).
Both are ordinary optimisation work; the write-up's §10 quantifies them.

**Two things to know before enabling it.**

1. **It is not token-identical to non-speculative greedy decoding, and cannot be.** The verify step
   argmaxes *extend*-path logits and commits state with the *SSD scan*, where a decode step uses
   the graphed decode kernels and the recurrent step — different reduction orders. The control arm
   reproduces the baseline exactly, so the engine is deterministic; the speculative arm diverges at
   token 40–71 of 1 023 on three of four prompts (identical on the fourth, and the 131K needle is
   recalled in both arms). Any multi-token verification scheme on this engine carries this.
2. **Long context regresses.** A verify step's extend attention reads the KV history once per query
   token, so at 131K it costs ~118 ms against an ~11.5 ms decode step — **~10×**, against ~7× at
   short context — and `k + 1 = 9` cannot beat that. The decoder therefore measures its own
   verify/decode ratio online and stops drafting when `accepted + 1` can no longer pay for it,
   re-probing every 16 gated steps. There is deliberately **no** context-length threshold: the
   ratio depends on KV dtype, attention backend and acceptance. The gate cannot refund the
   *measurement*, though — pricing itself costs two verify steps, which on a short generation at
   131K is the whole −11 %.

A 16-way passthrough soak passes with zero errors in both arms and flat p50 / request count; its
p95/p99 tail differs but one 10-minute pair cannot separate that from variance (write-up §11).

Still open, by upside: cutting the ~40 % of a verify step that is not the forward (batch the
46-launch commit), the burst-entry hysteresis, `--spec-draft-len 16` for long context, batched
speculation, a graph-captured fixed-width verify forward, and sampling support.

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
