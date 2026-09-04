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
| 131 072 | 8 192 | **2 968 tok/s** | 2 887 tok/s | 53.8 tok/s | **FAIL** (see below) |

For scale, the Phase 1 pure-torch numbers on the same host were 1 098-1 202 tok/s at
131 072/8 192 and 2 799 tok/s at 131 072/4 096, so the kernels are ~2.5x on the 131 072
case even where the answer is wrong.

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

## Open: the 131 072-token needle at 8 192-token chunks

`bench_long_context.py` misses the needle at 131 072/8 192 on the kernel path,
reproducibly (3/3 runs, byte-identical degenerate output). It is **not** state
corruption, and the standing gate (32 768 tokens, 4 096-token chunks) is unaffected:

| Probe | Result |
|---|---|
| kernel, 131 072, **4 096**-token chunks, raw completion | needle found |
| `FREETOKEN_MAMBA2_REF=1`, 131 072, 8 192, raw completion | needle found |
| kernel, 131 072, **8 192**, raw completion | needle absent (x3) |
| kernel, 131 072, **8 192**, `/v1/chat/completions`, same server flags | **needle found** -- `'The secret passcode is 5663623.'` |
| `mamba2_prefill` at H=64/P=64/N=128, 32 768 tokens fed as 2 048 / 4 096 / **8 192** / 16 384-token extends vs one pass | relative error **0.000e+00** on both the output and the carried state |
| same, first (autotuning) call vs warm calls at T=8 192 | 0.000e+00 -- not an autotune artifact |
| the kernels at the crashing shapes under `compute-sanitizer memcheck` | 0 errors |

So the scan is exactly chunk-invariant, and the same build answers the same 131 072-token
prompt at 8 192-token chunks through the chat endpoint. What the bench probes is a *raw*
`/v1/completions` greedy continuation with `ignore_eos`, whose first token differs between
any two chunkings (the conv and the scan tile differently, so 4 096 and 8 192 are
bit-different for *every* implementation, the Phase 1 reference included) -- and at 131 072
tokens of filler that continuation is a coin flip: the reference won it at 8 192 and lost
nothing by it, the kernels won it at 4 096 and at 8 192 through the chat template.

**Not closed.** Recommended follow-up before leaning on the 131 072 needle as a gate:
compare the live recurrent state inside one server across `--max-prefill-length` values
(a GPU test in the style of `tests/models/test_nemotron_h_chunked_prefill.py` but at real
geometry and 8 192-token extends), and switch the long-needle gate to the chat endpoint so
it stops depending on an unanchored raw continuation.

## Verification summary

- `uv run pytest tests/kvcache tests/scheduler tests/models tests/kernels/test_mamba2_*.py
  -q -m "not slow"`: **586 passed, 4 skipped, 6 errors**. The only failures are the six
  pre-existing `tests/models/test_laguna_modules.py` errors (`RuntimeError: TP info has been
  set`), a cross-directory ordering artifact reproduced unchanged at `dc795ec`.
- `tests/models/test_nemotron_h_chunked_prefill.py` (the chunked-vs-single-pass gate on the
  real mixer): passes on the kernel path.
- ruff: no new findings in the touched files.

