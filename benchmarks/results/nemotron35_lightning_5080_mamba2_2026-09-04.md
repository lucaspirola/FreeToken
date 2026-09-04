# Nemotron-3.5-Lightning-30B-A3B-NVFP4 — Phase 2 tasks 2A1 + 2A4 (Triton Mamba-2 SSD on the model path)

Run date 2026-09-04, on top of `dc795ec` plus the working-tree changes listed below.
Host: WSL2, RTX 5080 16 GB (sm_120), CUDA 13 / Torch 2.11 / Triton 3.6, 33 GiB RAM.
GPU shared with two sibling NVFP4 agents (they hold ≤ 2.5 GiB); every serving/benchmark
run below was taken under `scripts/gpu_lock.sh`.

## What changed

The Nemotron-H Mamba-2 mixer now runs the vendored Triton SSD kernels instead of the
pure-PyTorch chunk scan:

| Path | Phase 1 | Now |
|---|---|---|
| prefill | Python loop over requests, `chunk_scan.mamba2_chunk_scan` per request, split again at the track boundary | one `mamba2_prefill` launch for the whole varlen batch; the track snapshot is a row of the per-chunk state block |
| decode | HF `mamba2_selective_state_update` with expanded `A`/`D`/`dt_bias` rebuilt every step | `mamba2_decode` (Triton) into a cached `out` buffer; `A = -exp(A_log)` cached on the module |
| gated norm | pure-Torch grouped RMSNorm | `mamba2_gated_rmsnorm` (fused Triton) |
| state layout | pool `[slots, H, N, P]` + a `transpose(-1,-2).contiguous()` on every read and write | pool `[slots, H, P, N]`, the SSD/flashinfer native block — no transposes anywhere |
| radix snapshot boundary | ×64 (the FLA constant) | ×128 (`LinearGatedDeltaGroupConfig.track_chunk_size`); Qwen3.5/Ornith GDN stay on 64 |

The Phase 1 scan is preserved verbatim in `models/nemotron_h/mamba2_reference.py` and
selected by `FREETOKEN_MAMBA2_REF=1`, for A/B.

Decode backend default is the **Triton** port (plan decision 2026-09-04);
`FREETOKEN_MAMBA2_DECODE=flashinfer` (or `auto`) opts back in.

## Bug found and fixed during integration

The first P2 run died in `gate_nemotron_h_serving all` (batch-invariance, right after the
elastic 4 -> 16 capacity raise) with

```
IndexKernel.cu:111 ... Assertion `-sizes[i] <= index && index < sizes[i]` failed
  scheduler.py:1893  self.token_pool[output_mapping] = forward_output.next_tokens_gpu
```

with exactly `#running-req` corrupted indices. Bisect, two ~3-minute runs:

| Run | Result |
|---|---|
| `dc795ec` (parent commit), identical flags, from a detached worktree | **PASS** -- so a regression, not pre-existing |
| this tree with `FREETOKEN_MAMBA2_REF=1` (2A1 wiring, Phase-1 scan) | **PASS** -- so 2A1 is clean, the kernels' *driver* is not |
| the kernels alone at the crashing shapes under `compute-sanitizer memcheck` | 0 errors -- the corruption needs the CUDA graph |

Cause: `NemotronHMamba2Mixer._decode_out` cached one grow-only `[bs, H, P]` buffer for
`mamba2_decode(out=...)`. Graph capture runs an eager warmup at every captured batch size,
so each graph bakes in the address it saw. The elastic raise to 16 running requests then
puts real decode batches *above* the largest captured size; those run eagerly, grow the
buffer, and free the block the smaller graphs still write to on every replay -- landing on
whatever the caching allocator handed out next, here the freshly staged write-mapping
index tensors.

Fix: one buffer **per batch size**, never replaced once handed out
(`_out_buffers[(bs, dtype, device)]`). Pinned by
`test_decode_out_buffer_is_stable_per_batch_size`.

## Launch lines

Both profiles: `FREETOKEN_PIN_BUDGET_GB=17`, q8_0 KV, `--attention-backend triton`,
`--memory-ratio 0.85 --max-prefill-length 8192`, no `--moe-pageable-gpu`.

```
P1  ft serve --model $LIGHTNING --max-running-requests 1 --moe-backend offload \
      --moe-cache-auto --num-tokens 65536 --kv-cache-dtype q8_0 \
      --attention-backend triton --memory-ratio 0.85 --max-prefill-length 8192 \
      --host-ram-reserve-gb 3

P2  ft serve --model $LIGHTNING --max-running-requests 16 --elastic-initial-requests 4 \
      --kv-grow-step-tokens 65536 --num-tokens 262144 --max-seq-len-override 131072 \
      --kv-cache-dtype q8_0 --attention-backend triton --moe-backend offload \
      --moe-cache-auto --memory-ratio 0.85 --max-prefill-length 8192 \
      --host-ram-reserve-gb 3 --enable-cache-report
```

Both log `Mamba-2 decode backend: triton` from the pre-capture `warm_mamba2_decode`.

## P1 smoke (greedy, thinking off)

| Prompt | Answer |
|---|---|
| 17 x 23 | `391` |
| capital of Australia | `Canberra` |
| 2^10 | `1024` |
| first five primes | `2, 3, 5, 7, 11` |

Ready in 64 s, CUDA graphs `[1]`, 13 110 MiB VRAM, 2.19 GiB free after init.
(A thinking-on prompt capped at 400 tokens spends all of them in the reasoning block
and returns empty content -- the model's own behaviour, unchanged from Phase 1.)

## P2 serving gates -- `benchmarks/gate_nemotron_h_serving.py all`

Ready in 34 s, CUDA graphs `[1, 2, 3, 4]`, 13 158 MiB VRAM, 2.19 GiB free after init.

| Gate | Result |
|---|---|
| batch-invariance | **PASS** -- 8 solo answers reproduced by 16 concurrent copies |
| prefix-cache | **PASS** -- 5 799-token prompt, cached 0 cold -> 5 760 warm |
| elastic-ramp | **PASS** -- 1x0.9 s -> 6x3.3 s -> 16x3.0 s -> 1x0.3 s |
| tool-call | **PASS** -- `finish_reason=tool_calls`, `get_current_weather` |
| overall | **PASS** |

Elastic transitions observed, all clean: `4 -> 16` (GDN slots 25 -> 97, MoE 1 652 -> 976),
`4 -> 6` (25 -> 37, 1 652 -> 1 500), and both returns to 4.

## Synthetic needle (`bench_long_context.py --synthetic-needle`, q8_0, mem-ratio 0.85)

| Prompt | Chunk | Prefill (end-to-end) | Prefill (engine avg) | Decode | Needle |
|---|---:|---:|---:|---:|---|
| 32 768 | 4 096 | **5 147 tok/s** | 4 174 tok/s | 72.5 tok/s | **PASS** |
| 131 072 | 8 192 | **2 968 tok/s** | 2 887 tok/s | 53.8 tok/s | raw probe **FAIL**, see below |
| 131 072 | 8 192 | **3 014 tok/s** | 2 953 tok/s | 73.1 tok/s | **PASS** on the chat-endpoint gate |

For scale, the Phase 1 pure-torch numbers on the same host were 1 098-1 202 tok/s at
131 072/8 192 and 2 799 tok/s at 131 072/4 096, so the kernels are ~2.5x on the 131 072
case. The raw-completion "FAIL" row was the probe, not the engine -- see the section
below; the second row is the same configuration re-measured after the gate moved to the
chat endpoint.

## Prefill/decode A/B at 32 768 tokens, 4 096-token chunks, decode 64

Same server flags, only `FREETOKEN_MAMBA2_REF` differing; both answered the needle
correctly.

| | Kernels | `FREETOKEN_MAMBA2_REF=1` | Ratio |
|---|---:|---:|---:|
| prefill, end-to-end | **5 250 tok/s** | 3 956 tok/s | **1.33x** |
| prefill, engine average | 4 264 tok/s | 3 520 tok/s | 1.21x |
| prefill, engine instant | 3 648 tok/s | 3 011 tok/s | 1.21x |
| decode | 53.3 tok/s | 54.9 tok/s | 0.97x |

**The >= 2x prefill target is not met end to end, and cannot be**: at 32 K on this host
prefill is dominated by streaming the offloaded NVFP4 experts over PCIe, not by the scan,
so a scan that got much faster moves the total by ~1.3x. Decode is PCIe-bound for the
same reason and comes out at parity (the run-to-run spread on decode is larger than the
difference -- the plain 32 K needle measured 72.5 tok/s on the same kernel build).
The scan-only speedup belongs to `benchmarks/bench_mamba2_ssd.py` (task 2A2), not to a
serving benchmark.

## Closed: the 131 072-token needle at 8 192-token chunks

`bench_long_context.py` missed the needle at 131 072/8 192 on the kernel path,
reproducibly (3/3, byte-identical), while the same build answered the same prompt at
4 096-token chunks, and through the chat endpoint at 8 192. The 2026-09-04 write-up
called that a marginal-continuation artifact but did not prove it. It is now proved,
by comparing the live state *inside* the server.

### The instrument

`FREETOKEN_MAMBA2_STATE_DUMP=<dir>` (`models/nemotron_h/state_dump.py`, called from
`NemotronHForCausalLM.forward`) saves, on the **last** prefill forward of each real
request, that request's live slot for every Mamba-2 layer -- recurrent `[23, 64, 64,
128]` and conv `[23, 6144, 3]`, fp32 -- plus the sampled position's logits. Warmup
batches (`uid=-1`) and `ChunkedReq` continuations are skipped. Unset, it costs one
module-level `if` per forward.

### The runs

One prompt (the synthetic needle trimmed to exactly 131 072 tokens, needle at token
67 630, depth 0.516), four servers, identical flags but for the two variables, each
under `scripts/gpu_lock.sh`: `--max-running-requests 1 --num-tokens 262144 --kv-cache-dtype
q8_0 --attention-backend triton --memory-ratio 0.85 --host-ram-reserve-gb 3`,
`FREETOKEN_PIN_BUDGET_GB=17`, probed with the *old* raw `/v1/completions` greedy
continuation (`ignore_eos`, 48 tokens).

| Run | scan | `--max-prefill-length` | raw-completion needle |
|---|---|---:|---|
| kernel-4k | Triton SSD | 4 096 | **found** |
| kernel-8k | Triton SSD | 8 192 | **absent** (the failure, reproduced) |
| ref-4k | `FREETOKEN_MAMBA2_REF=1` | 4 096 | **found** |
| ref-8k | `FREETOKEN_MAMBA2_REF=1` | 8 192 | **found** |

### State comparison

Per-layer relative RMS of the end-of-prefill state, `‖a-b‖ / ‖b‖`:

| layer | kernel-4k vs kernel-8k | ref-4k vs ref-8k | kernel-8k vs ref-8k |
|---:|---:|---:|---:|
| 0 (recurrent) | **0.000e+00** | 1.696e-09 | 5.128e-04 |
| 0 (conv) | **0.000e+00** | 0.000e+00 | 0.000e+00 |
| 2 (recurrent) | 7.226e-07 | 9.399e-08 | 1.822e-02 |
| 2 (conv) | 0.000e+00 | 0.000e+00 | 2.096e-02 |
| 4 (recurrent) | 1.389e-03 | 2.520e-04 | 1.014e-02 |
| 7 | 8.464e-03 | 7.099e-03 | 3.227e-02 |
| 14 | 2.540e-02 | 4.686e-02 | 1.009e-01 |
| 21 | 1.029e-01 | **1.543e-01** | **3.030e-01** |
| 28 | **1.126e-01** | 1.364e-01 | 2.654e-01 |
| 35 | 1.102e-01 | 8.472e-02 | 1.384e-01 |
| 48 | 1.234e-02 | 3.579e-02 | 6.514e-02 |
| worst recurrent | 1.126e-01 (L28) | 1.543e-01 (L21) | 3.030e-01 (L21) |
| worst conv | 2.130e-01 | 2.428e-01 | 3.642e-01 |
| final logits | 2.390e-01 | 2.806e-01 | 2.552e-01 |
| top-1 next token | `11` = `11` | `11` = `11` | `11` = `11` |

### Verdict: no chunk-boundary bug

Four facts, in order of weight:

1. **Layer 0 is bit-exact across the two chunkings on the kernel path** -- recurrent
   *and* conv, `0.000e+00` on both max-abs and relative RMS, after 131 072 tokens.
   Layer 0 is the only Mamba-2 layer whose inputs are identical by construction in both
   runs (it consumes the embeddings), so it is the one layer that isolates the
   integration: `build_fla_metadata` / `_build_track_metadata`, the conv-window write at
   `track_dst`, `has_initial_state` on chunk 2+, the ×128 snapshot row and the state-pool
   round trip all reproduce a 32-chunk prefill from a 16-chunk one *exactly*, at real
   geometry and full length. Every one of the plan's suspects is cleared by that zero.
2. **The known-good reference path diverges more across the same two chunkings than the
   kernels do** (worst recurrent 1.543e-01 vs 1.126e-01) -- and finds the needle at both.
   Divergence magnitude therefore carries no information about the needle outcome.
3. **The two implementations at the *same* chunking differ more than either differs
   across chunkings** (3.030e-01), yet ref-8k answers and kernel-8k does not. The needle
   outcome is not a function of state fidelity.
4. **All four runs agree on the top-1 next token** (id `11`, the `<|im_end|>` the model
   wants to emit after the 131 072-token prompt), with a comfortable 0.75-1.63 logit
   margin. Nothing is marginal at the first token: the raw probe sets `ignore_eos`, so
   generation is forced *past* the model's chosen end-of-text into an unanchored
   continuation of the haystack, and that is where the four runs part company.

Where the divergence does enter: layer 2's conv state (a function of the last three
tokens only) is still exactly zero while its recurrent state -- a decayed sum over all
131 072 positions -- is 7.2e-07, so layer 2's inputs differ at interior positions. Layer
1 is the first MoE layer, and NVFP4 expert GEMMs are not bit-invariant to the number of
rows in the batch, which *is* the prefill chunk size. (The NVFP4 *dense* linear is
invariant -- 4 096/4 096 rows bit-identical whether run as `M=4096` or as the head of
`M=8192` -- so the entry point is the routed fused-MoE path, not the dense GEMMs.) 52
layers and 131 072 tokens then amplify 1e-7 into 1e-1, on both scans equally.

### The gate

The 131 072-token needle probe now asks its question through
`/v1/chat/completions` with `enable_thinking: False` instead of continuing the haystack
through `/v1/completions` (`benchmarks/bench_long_context.py:stream_completion`,
pinned by `test_stream_completion_asks_the_question_through_the_chat_endpoint`).
`ignore_eos` is kept so the decode-rate sample stays a fixed number of steps.
Re-measured on the kernel path at the configuration that used to fail:

| | 131 072 tokens, 8 192-token chunks, kernel path |
|---|---|
| prompt | 131 088 tokens (chat template adds 16) |
| prefill | **3 013.9 tok/s** end-to-end, 2 952.9 tok/s engine average |
| decode | 73.1 tok/s |
| needle | **PASS** -- `'The secret passcode is 5663623.'` |
| VRAM | 13.42 GiB |

## Verification summary

- `uv run pytest tests/kvcache tests/scheduler tests/models tests/kernels/test_mamba2_*.py
  -q -m "not slow"`: **586 passed, 4 skipped, 6 errors**. The only failures are the six
  pre-existing `tests/models/test_laguna_modules.py` errors (`RuntimeError: TP info has been
  set`), a cross-directory ordering artifact reproduced unchanged at `dc795ec`.
- `tests/models/test_nemotron_h_chunked_prefill.py` (the chunked-vs-single-pass gate on the
  real mixer, plus the new `test_state_dump_hook_writes_only_the_last_chunk_of_a_real_request`):
  8 passed on the kernel path.
- `tests/benchmarks/test_bench_long_context.py`: 8 passed, including the new
  `test_stream_completion_asks_the_question_through_the_chat_endpoint`.
- ruff: no new findings in the touched files.

