# 262K needle — cross-engine check on llama.cpp (2026-09-04)

Ticket: `tasks/nemotron35-handover.md` item 1, the decisive test left open by
`benchmarks/results/nemotron35_lightning_5080_262k_bisect_2026-09-04.md`.

**Verdict: llama.cpp recalls the needle at every depth at 262,144 tokens — including the
three points where FreeToken misses. The "model/quant retrieval limit" conclusion of the
bisect does not survive. Reopen: the miss is on the FreeToken side (engine and/or the
NVFP4 weight path), not a property of Nemotron 3.5 Lightning at 262K.** 8/8 llama.cpp runs
PASS with the byte-identical answer `5663623`; the FreeToken failure was re-reproduced on
the same host the same hour and still misses at depth 0.267/0.519/0.947.

## What was held fixed

The prompts are **byte-identical** to the bisect's — not "rebuilt the same way", the same
files: `benchmarks/bench_long_context.synthetic_needle_sample(depth)` + `trim_filler`,
tokenized with the NVFP4 checkpoint's own HF tokenizer, then SHA-1 compared against
`…/scratchpad/bisect262/prompt_*.txt`:

| prompt | SHA-1 | tokens (HF) | needle token | depth |
|---|---|---:|---:|---:|
| 262144 d0.05 | `4f3768ed68cbc374e0c86729829f09964a275179` | 262,144 | 14,873 | 0.0567 |
| 262144 d0.25 | `b53ec20a5768334cd4208f33362d956dfed4e5df` | 262,144 | 69,958 | 0.2669 |
| 262144 d0.50 | `cf8126cbf0a8c5637b01768891c1bf222e15ae4c` | 262,144 | 136,113 | 0.5192 |
| 262144 d0.75 | `f283980db1e5c7ca617d6e036d4310331697e5d7` | 262,144 | 199,443 | 0.7608 |
| 262144 d0.95 | `d209db461a18952f9c98149d859d5b31e0f6742c` | 262,144 | 248,194 | 0.9468 |
| 196608 d0.50 | `ef81a1a0dd14c857b0d893db14f2df11ff7487e0` | 196,608 | 101,871 | 0.5181 |
| 147456 d0.50 | `a54b6d5422fb60504292281b760b92004d4982d5` | 147,456 | 76,190 | 0.5167 |
| 131072 d0.50 | `dd3a25b79a13696116a83eb24c33bd20187d8dae` | 131,072 | 67,630 | 0.5160 |

Digit-free filler (the only digits in the haystack are the needle's seven). Grading goes
through `/v1/chat/completions` with `chat_template_kwargs {"enable_thinking": false}`,
`temperature 0`, streamed — the JSON `delta` fields are concatenated *before* anything is
searched; the raw SSE frames are never grepped. Pass = the literal `5663623` in the answer.

The two engines agree on tokenization end to end: llama.cpp reports `prompt_tokens =
262,160` for the 262,144-token prompts and `131,088` for the 131,072 one — exactly the
post-chat-template counts FreeToken reported in the bisect.

## Engines

| | FreeToken (bisect + today's re-check) | llama.cpp (this run) |
|---|---|---|
| build | `main` @ `ae334f6`+ (worktree clean) | `6b80c74f285390368b3c99c5e750f19e9b096e98`, version 9542, CUDA arch 1200 |
| weights | `~/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` (safetensors, NVFP4) | `ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF` → `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf` |
| GGUF facts | — | 18,898,091,584 B (17.6 GiB), 401 tensors, `general.architecture = nemotron_h_moe`, `block_count 52`, `expert_count 128`, `expert_used_count 6`, `context_length 1048576`, `file_type 2` (Q4_0) |
| KV | q8_0, Triton attention | q8_0 K and V, flash attention on |
| context | `--num-tokens 524288 --max-seq-len-override 1048576`, grow step 131072 | `-c 270336` |

**Deviation from the task statement, deliberate:** the named repo publishes only
`Q4_0` (18.9 GB), `Q8_0` (33.6 GB) and `BF16` (63.2 GB) — there is no `Q4_K_M` in
`ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF`. `Q4_0` was used: it is the official
4-bit conversion, the same bit-width class as the NVFP4 checkpoint under test, and the
*weakest* of the three, so a pass on it is the conservative direction for this comparison.
(`Q4_K_M` exists only in third-party repos — bartowski 25.5 GB, lmstudio-community 24.5 GB
— and would not have fit the host as comfortably.)

**Settings forced by the 16 GiB RTX 5080:** Q4_0 is 17.6 GiB of weights, so the model does
not fit. `--n-cpu-moe 14` keeps the routed-expert tensors of the first 14 of 52 blocks in
host RAM; everything else is on the GPU (`-ngl 999`). Steady-state VRAM was **15,428 MiB of
16,303 MiB** and prompt processing still ran at ~2,200–2,560 tok/s, so this cost throughput,
not capability. `-fa on -ctk q8_0 -ctv q8_0` (quantized KV + flash attention) as instructed;
`--no-context-shift`, `--cache-ram 0` (prompt cache off), `-b 4096 -ub 512 -t 16 --jinja
--no-warmup`, `-np 1`. Every run is a **fresh `llama-server` process** and the needle
request is the *first* request that process serves.

## Commands

```bash
# 1. GGUF (18.9 GB; 357 GB free on / beforehand)
hf download ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF \
  NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf \
  --local-dir ~/ai/models/nemotron35-gguf

# 2. build the prompts once, on the CPU, with the NVFP4 checkpoint's tokenizer
uv run benchmarks/bench_long_context_llamacpp.py --build-only \
  --target-prompt-tokens 262144 --needle-depth 0.25 --prompt-cache <scratch>/prompts

# 3. one fresh server per prompt, under the GPU lock
scripts/gpu_lock.sh <scratch>/xe/one.sh 262144 0.25 d25_262k 14 512 270336
#   which runs:
#   ~/ai/llama.cpp/build/bin/llama-server \
#     -m ~/ai/models/nemotron35-gguf/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf \
#     --host 127.0.0.1 --port 8080 -c 270336 -np 1 --no-context-shift --cache-ram 0 \
#     -ngl 999 --n-cpu-moe 14 -fa on -ctk q8_0 -ctv q8_0 -b 4096 -ub 512 -t 16 \
#     --jinja --no-warmup
#   then:
#   uv run benchmarks/bench_long_context_llamacpp.py --base-url http://127.0.0.1:8080 \
#     --target-prompt-tokens 262144 --needle-depth 0.25 --prompt-cache <scratch>/prompts

# 4. paired FreeToken re-check, same host, same hour (bisect variant `a`, unchanged)
VARIANT=a PORT=8130 DUMP="" PREFIX=prompt_    LENGTHS=262144 bash <bisect>/run_variant.sh
VARIANT=a PORT=8130 DUMP="" PREFIX=prompt_d05_ LENGTHS=262144 bash <bisect>/run_variant.sh
```

New file: `benchmarks/bench_long_context_llamacpp.py` — builds/caches the *same* prompt and
grades one chat completion against `--base-url`. It reuses `bench_long_context`'s
`synthetic_needle_sample` / `trim_filler` / `load_tokenizer` verbatim; no FreeToken engine
code was touched.

## Result

### 262,144 tokens, depth profile — the head-to-head

| needle depth | needle token | FreeToken NVFP4 (bisect) | FreeToken NVFP4 (re-run today) | **llama.cpp Q4_0** |
|---:|---:|---|---|---|
| 0.057 | 14,873 | PASS | PASS `The secret passcode is 5663623.` | **PASS** `5663623` |
| 0.267 | 69,958 | FAIL | — | **PASS** `5663623` |
| 0.519 | 136,113 | FAIL | FAIL `The orchard ledger says the copper marker is inactive. The orchard ledledger…` | **PASS** `5663623` |
| 0.761 | 199,443 | FAIL | — | **PASS** `5663623` |
| 0.947 | 248,194 | FAIL | — | **PASS** `5663623` |

### Length sweep at depth ≈0.52 — the non-monotonic rows

| prompt tokens | needle token | FreeToken NVFP4 (bisect) | **llama.cpp Q4_0** |
|---:|---:|---|---|
| 131,072 | 67,630 | PASS | **PASS** |
| 147,456 | 76,190 | FAIL | **PASS** |
| 196,608 | 101,871 | FAIL | **PASS** |
| 262,144 | 136,113 | FAIL | **PASS** |

All eight llama.cpp answers are the same string, `5663623`
(`output_sha1 = c19d39eeed4a`), emitted in 8 tokens and terminated by the model's own EOS
(no `ignore_eos`, no forced continuation).

### Timings (llama.cpp, RTX 5080, 14/52 MoE blocks on the CPU)

| run | prompt tokens | TTFT (s) | prefill (tok/s) | decode (tok/s) | needle |
|---|---:|---:|---:|---:|---|
| 262K d0.05 | 262,160 | 116.8 | 2,244 | 55.8 | PASS |
| 262K d0.25 | 262,160 | 117.5 | 2,232 | 30.2 | PASS |
| 262K d0.50 | 262,160 | 117.7 | 2,228 | 44.0 | PASS |
| 262K d0.75 | 262,160 | 118.5 | 2,212 | 52.6 | PASS |
| 262K d0.95 | 262,160 | 120.0 | 2,184 | 56.4 | PASS |
| 196K d0.50 | 196,624 | 84.9 | 2,315 | — | PASS |
| 147K d0.50 | 147,472 | 58.1 | 2,539 | 48.1 | PASS |
| 131K d0.50 | 131,088 | 51.2 | 2,561 | 71.9 | PASS |

Server load time was 4–5 s (page-cache warm). Decode rates are noisy because the answer is
only 8 tokens long; they are not a throughput measurement. FreeToken's paired re-check today
took TTFT 140.7 s (d0.05) and 141.0 s (d0.50) at the same 262,144 tokens — i.e. FreeToken
prefills 262K at ~1,860 tok/s fully on-GPU against llama.cpp's ~2,230 tok/s with a quarter of
the MoE blocks on the CPU. That is a separate, already-ticketed observation (no Nemotron
branch in `decode_launch_config`), not part of this verdict.

## Verdict

**llama.cpp recalls → reopen.** Every engine-side variable the bisect swept (KV dtype,
attention backend, kernel-vs-reference Mamba-2 scan, prefill chunk size, growable-vs-static
KV, dense dequantization) was swept *inside FreeToken*, so eight identical failures proved
only that whatever is wrong is common to all eight — and the depth-0.057 pass proved the
prompt reaches the model intact. The one variable the bisect could not move was the engine
itself, and moving it flips the outcome completely: on the byte-identical prompts, the same
architecture at the same 262,160 tokens with a *weaker* 4-bit quantization (Q4_0) and a
*less* exact KV path (q8_0 + flash attention, 14 blocks of experts on the CPU), llama.cpp
answers `5663623` at depths 0.06, 0.27, 0.52, 0.76 and 0.95, and at 147K/196K/262K, while
FreeToken on the NVFP4 checkpoint degenerates into haystack echo above depth ~0.1. There is
therefore no Nemotron-3.5-Lightning retrieval cliff at 262K to gate around: the bisect's
"model/quant limit" conclusion, its recommended "gate long needles at depth ≤0.1" acceptance
bar, and the 262K/524K needle rows attributed to it in the cache study and the 1M gate should
all be treated as open FreeToken defects again. The one thing this run does **not** isolate is
*which* FreeToken-side thing: engine and checkpoint moved together (NVFP4 safetensors vs Q4_0
GGUF), and FreeToken has no GGUF loader for `nemotron_h`, so a same-weights A/B is not
available on this host. The next step is the one the handover already names —
`FREETOKEN_MAMBA2_STATE_DUMP` against llama.cpp's per-layer state on the depth-0.52 262K
prompt, which now has a known-good reference to diff against for the first time; the
NVFP4-vs-BF16 weight hypothesis is the fallback if the states agree (dequantizing *all*
tensors, not just the shared experts and lm_head as `FREETOKEN_NEMOTRON_DENSE_DEQUANT=1`
does today).

## Reproduction

Scratch tree (prompts + metadata, per-run server logs, `results.jsonl`, driver scripts
`one.sh` / `batch.sh` / `ft_recheck.sh`):
`/tmp/claude-1000/-home-lucas-ai-FreeToken/f4e2e9e3-f4f5-40d0-9980-b3b09d1ef47d/scratchpad/{prompts,xe}`.
The bisect's original prompts, which these were SHA-1-matched against, are in
`/tmp/claude-1000/-home-lucas-ai-FreeToken/af23ede4-e8ad-4c8d-8b38-c8be515d8870/scratchpad/bisect262`.
Neither survives a WSL restart; the GGUF is at
`~/ai/models/nemotron35-gguf/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf`.
