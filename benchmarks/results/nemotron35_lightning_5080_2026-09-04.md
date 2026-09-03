# Nemotron-3.5-Lightning-30B-A3B-NVFP4 on RTX 5080 — Phase 1 bring-up

Run date 2026-09-04, commit `80f2838` + the working-tree changes listed under
"Code changes made during this run". Host: WSL2, 36.2 GiB RAM / 4 GiB swap,
RTX 5080 16 GB (sm_120), CUDA 13 / Torch 2.11 / Triton 3.6, GPU otherwise idle.

## Host preflight

`uv run python benchmarks/preflight_nemotron_host.py` → **READY**.
MemAvailable 33.4 GiB, SwapFree 3.8/4.0 GiB, expert banks 15.41 GiB, default CUDA
pin budget 14.49 GiB (0.4 x MemTotal, WSL) → 3 of 23 MoE layers pageable.
**`FREETOKEN_PIN_BUDGET_GB=17` pins all 23 layers**, so every run below drops
`--moe-pageable-gpu` and keeps decode CUDA graphs. No holder, no stale locks.

## Launch lines

P1 (bring-up, single stream) — 38 s to ready, CUDA graphs `[1]`:

```
CUDA_VISIBLE_DEVICES=0 FREETOKEN_PIN_BUDGET_GB=17 uv run ft serve \
  --model ~/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 --port 30001 \
  --max-running-requests 1 --moe-backend offload --moe-cache-auto \
  --num-tokens 65536 --memory-ratio 0.90 --max-prefill-length 4096 \
  --host-ram-reserve-gb 3
```

P2 (16 concurrent, elastic, prefix cache, quantized KV) — 50 s to ready,
CUDA graphs `[1, 2, 3, 4]` (elastic initial 4):

```
CUDA_VISIBLE_DEVICES=0 FREETOKEN_PIN_BUDGET_GB=17 uv run ft serve \
  --model ~/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 --port 30001 \
  --max-running-requests 16 --elastic-initial-requests 4 \
  --kv-grow-step-tokens 65536 --num-tokens 262144 --max-seq-len-override 131072 \
  --kv-cache-dtype fp8_e4m3 --attention-backend triton \
  --moe-backend offload --moe-cache-auto \
  --memory-ratio 0.90 --max-prefill-length 8192 --host-ram-reserve-gb 3 \
  --enable-cache-report
```

Both profiles ran **without** `--moe-pageable-gpu`. Auto-selected parsers:
`--tool-call-parser qwen3_coder`, `--reasoning-parser nemotron_v3`
(`qwen3` at the time of the P1 run; a sibling session landed `nemotron_v3`
mid-session — see "Confounds").

Needle runs use `benchmarks/bench_long_context.py`, which launches its own
server (`--max-running-requests 1`, `--cuda-graph-max-bs 1`, `--moe-cache-auto`):

```
CUDA_VISIBLE_DEVICES=0 FREETOKEN_PIN_BUDGET_GB=17 uv run python benchmarks/bench_long_context.py \
  --model ~/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --synthetic-needle --target-prompt-tokens 65536 --decode 128 \
  --max-context 131072 --kv-cache-dtype fp8_e4m3 --prefill-chunk 8192 \
  --mem-ratio 0.90 --kv-grow-step-tokens 65536 --host-ram-reserve-gb 3
```

## Memory and throughput

| Profile | VRAM used | Host RSS (all workers) | MoE slot cache | KV alloc |
|---|---:|---:|---:|---:|
| P1 (bs 1, 65 536 KV tokens) | 15 576 / 16 303 MiB | 19.5 GiB | 1 904 slots | 0.38 GiB |
| P2 (bs 16 elastic, 262 144 KV tokens) | 15 700 / 16 303 MiB | 19.6 GiB | 1 791 → 1 115 slots at 16 req | 0.80 GiB |
| needle server (bs 1, 262 144 KV tokens) | ~2.8 GiB reported by the bench | — | 1 931 slots | 0.80 GiB |

Free GPU memory after init: 1.21 GiB (P1) / 1.45 GiB (P2). Elastic transitions
observed: `4 -> 16 requests: GDN slots 25 -> 97, MoE slots 1791 -> 1115` and back.

| Measurement | Value |
|---|---:|
| P1 decode, bs 1, short ctx | 124–150 tok/s |
| P1 decode, bs 1, 6.5 K ctx | ~150 tok/s |
| P1 prefill, 4096-token chunk (cold / warm) | 1 746 / 302 tok/s instant |
| P2 prefill, 8192-token chunk (cold / warm) | 256 / 957 tok/s instant, 423 tok/s aggregate |
| P2 decode, 16 concurrent (aggregate) | 84–93 tok/s (≈ 5.6 tok/s per request) |
| Needle prefill, 65 536 tokens, chunk 8192 | 1 118 tok/s end-to-end |
| Needle prefill, 131 072 tokens, chunk 8192 | 1 098–1 202 tok/s end-to-end |
| Needle prefill, 131 072 tokens, chunk 4096 | **2 799 tok/s** end-to-end |
| Needle decode at 65 K / 131 K ctx | 61–67 / 53–57 tok/s |

Aggregate throughput at 16 concurrent requests is *lower* than at bs 1 — the
offload MoE path is PCIe-bound once 16 decode streams each touch top-6 of 128
experts. This is the Phase 2 / 2B4 workload.

## KV dtype decision: fp8_e4m3 vs q8_0

Identical server flags apart from `--kv-cache-dtype` (both with
`--attention-backend triton`); greedy everywhere.

| Check | `q8_0` | `fp8_e4m3` |
|---|---|---|
| KV allocation at 262 144 tokens | 0.80 GiB | 0.80 GiB |
| batch-invariance (8 solo vs 16 concurrent) | PASS | PASS |
| elastic-ramp 1→6→16→1 | PASS | PASS |
| tool-call round trip | PASS | PASS |
| prefix-cache equality (repeat runs) | **4 / 4 PASS** | **3 / 6 PASS** |
| needle 65 536, chunk 8192 | PASS (first answer correct) | PASS (first answer correct) |
| needle 131 072, chunk 8192 | **FAIL** — emitted `5663616` | PASS by substring only — first answer `5666363`, a later repetition is correct |
| needle prefill / decode at 65 K | 1 118 / 61.0 tok/s | 1 119 / 67.0 tok/s |
| needle prefill / decode at 131 K | 1 098 / 52.8 tok/s | 1 202 / 57.3 tok/s |
| reasoning suite (5 exact-answer problems, greedy, thinking on) | **5 / 5** | **5 / 5** |

`bf16` KV (`--kv-cache-dtype auto`, FlashInfer) at 131 072 also **fails** the
needle (`566363623`), which rules KV quantization *out* as the cause of the
128 K miss — see "Open issue" below.

**Recommendation: `q8_0`.** Both dtypes cost the same VRAM, score the same on
reasoning, and pass every short-context gate, but q8_0 reproduces cached-prefix
answers reliably (4/4) where fp8_e4m3 flips the first generated token in half of
its runs (3/6). Switchyard's prefix-reuse path depends on that equality, so the
determinism outweighs fp8_e4m3's ~8 % decode edge at long context. If fp8_e4m3
is required for other reasons, expect prefix-cache answer drift.

The AIME subset could not be run: `tests/e2e/test_aime.py` skips with
`set FREETOKEN_AIME25_JSONL to the aime25 jsonl file`, and no AIME jsonl exists
on this host. The five-problem reasoning suite above (divisor count, modular
exponentiation, arithmetic-series sum, ordered factor pairs, largest prime
factor; reference answers computed with sympy) stands in for it.

## Gate results

Parity (`benchmarks/parity_nemotron_h_layers.py --layers 0,1,5 --tokens 512`) and
the serving gates are appended below by their own scripts. Summary: parity PASS
(cosine ≥ 0.9998, routed-expert ids 100 %); serving gates PASS under q8_0.

## Open issue — 131 072-token needle

64 K retrieval is exact under both KV dtypes. At 131 072 the model finds the
needle sentence but corrupts its digits, in a way that is **not** explained by KV
quantization (bf16 fails too) and that **is** sensitive to the prefill chunk size:

| 131 072-token needle | prefill chunk | first answer | verdict |
|---|---:|---|---|
| `q8_0` | 8192 | `5663616` | wrong |
| `fp8_e4m3` | 8192 | `5666363` | wrong (later repetition correct) |
| `auto` (bf16) | 8192 | `566363623` | wrong |
| `auto` (bf16) | 4096 | needle sentence not reproduced at all; output degrades into repeated filler lines | much worse |

More chunk boundaries ⇒ worse output (8 chunks at 64 K: exact; 16 chunks at
128 K: digit corruption; 32 chunks at 128 K: collapse). That points at the
Mamba-2 state handoff across chunked-prefill boundaries rather than at attention
or at the KV cache, and it is the first thing Phase 2 (2A1/2A2, the vendored SSD
kernels and the `mamba2` state layout) should be validated against. Chunk 4096
is also 2.4x *faster* than chunk 8192 at the same prompt length, which is itself
worth explaining.

## Code changes made during this run

1. `python/freetoken/models/nemotron_h/chunk_scan.py` (**new**) and the two-line
   switch in `models/nemotron_h/model.py::NemotronHMamba2Mixer._scan`.
   transformers' pure-Torch `mamba2_chunk_scan` writes its four contractions as
   broadcast-multiply-then-`.sum()`, allocating rank-6 temporaries worth
   **4.19 MiB per token per mamba layer** — 17 GiB for one layer at a 4096-token
   prefill chunk. P1 died with `CUDA error: out of memory` inside
   `OffloadMoeCache._invalidate_prefill_buffer` on the first prompt longer than
   ~1 K tokens. The rewrite expresses the same contractions as `einsum`
   (batched GEMMs), keeping the algorithm, the fp32 accumulation, the dt clamp,
   the padding and the returned final state identical. Verified against the
   transformers original at T ∈ {1, 5, 127, 128, 129, 300, 512, 1024, 4097} with
   and without an initial state: max relative error 2.4e-7 on both the output and
   the final state. Peak transient 4256.6 → 224.6 KiB/token (19x). Layer parity
   numbers are unchanged to every printed digit.
2. `python/freetoken/kernel/csrc/vmm_tensor.cpp` + `python/freetoken/kernel/vmm.py`:
   `parse_dtype` / `VMMTensor._DTYPE_NAMES` gained `float8_e4m3fn` and
   `float8_e5m2`. With `--kv-grow-step-tokens` set, the MoE device bank cache is
   VMM-backed, and the NVFP4 expert banks carry an fp8 per-group scale tensor, so
   P2 aborted at startup with
   `ValueError: unsupported VMM tensor dtype: torch.float8_e4m3fn`
   (`moe/offload_cache.py:393` → `kernel/vmm.py:73`). `tests/kernels/test_vmm_tensor.py`
   still passes (4/4).
3. `benchmarks/parity_nemotron_h_layers.py`: `random_hidden` → `layer_input`, which
   samples real `backbone.embeddings` rows and applies the block's own input
   RMSNorm instead of feeding `torch.randn`. The mixer's FP8 projections carry
   *static* modelopt activation scales; synthetic normal inputs push the gated-norm
   output 9x past `out_proj`'s calibrated amax (11.5 vs 1.24), and the resulting
   outlier clipping showed up as a spurious mamba cosine of 0.9742 — reproduced
   exactly (0.97437) by emulating static-scale FP8 on the *reference's* activations,
   i.e. an artifact of the probe, not of either implementation. With in-distribution
   input, layer 0 cosine is 0.99981 and no activation is clipped.
4. `benchmarks/bench_long_context.py`: new `load_tokenizer()` falls back to
   `transformers.AutoTokenizer` when `--model` is a directory. The script hard-coded
   `load_gguf_tokenizer`, so an HF safetensors checkpoint died with
   `IsADirectoryError` before launching anything.

## Deviations from the plan and confounds

- `FREETOKEN_PIN_BUDGET_GB=17` set and `--moe-pageable-gpu` dropped in both
  profiles (the plan's preferred option); it works, and decode CUDA graphs are
  captured in both.
- P2's `--kv-cache-dtype` was q8_0 for the first full gate sweep and fp8_e4m3 for
  the second; both are reported above.
- The plan's P1 prefill chunk (4096) needs ~0.9 GiB of transient and P2's (8192)
  ~1.8 GiB *after* change 1; `--memory-ratio 0.90` leaves 1.21–1.46 GiB. 8192-token
  chunks did run, but the margin is thin — lower `--memory-ratio` to ~0.85 if
  anything else is resident.
- Confound: two sibling sessions were editing the working tree during this run
  (Phase 3 server work in `server/{api_models,generation,reasoning_parser}.py`,
  Phase 2 kernels under `kernel/triton/mamba2/`). The P2 servers therefore picked
  up `--reasoning-parser nemotron_v3`, `force_nonempty_content` and
  `context_preflight` from that work. Nothing was reverted or stashed.
- Nothing committed.


## Layer parity vs HuggingFace 2026-09-04T00:46:48

Model: `/home/lucas/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`  
Tokens: 512, seed 1234, device `cuda`

| Layer | Kind | Cosine | Expert ids | max abs err | scaled | rel L2 |
|---:|---|---:|---:|---:|---:|---:|
| 0 | mamba | 0.999806 | — | 4.688e-02 | 4.167e-02 | 1.969e-02 |
| 1 | moe | 0.999996 | 1.00000 | 1.562e-02 | 3.937e-03 | 2.972e-03 |
| 5 | attention | 0.999970 | — | 3.906e-03 | 4.545e-03 | 7.741e-03 |

Result: PASS
## Serving gates 2026-09-04T01:00:54

Endpoint: `http://127.0.0.1:30001`  
Model: `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`  
Streaming: on, thinking: off

| Gate | Result | Detail |
|---|---|---|
| batch-invariance | PASS | 8 solo answers reproduced by 16 concurrent copies |
| prefix-cache | PASS | prompt 5799 tokens, cached 0 cold -> 5760 warm |
| elastic-ramp | PASS | 1x0.3s -> 6x2.2s -> 16x2.5s -> 1x0.3s |
| tool-call | PASS | finish_reason=tool_calls calls=['get_current_weather'] |
