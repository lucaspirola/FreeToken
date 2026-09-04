# 262K needle bisect — Nemotron 3.5 Lightning 30B-A3B-NVFP4 on one RTX 5080 (2026-09-04)

Ticket: `tasks/todo.md` "262K recall bisect (blocks the 1M goal)". The 1M gate saw a fresh
262,144-token needle miss (hallucinated `1234`) with no spill/restore involved, while 131,072
passed. This run bisects that.

**Verdict: there is no FreeToken defect here.** Every engine variable was A/B'd on one fixed
prompt and none of them moves the outcome; the engine is proved to serve a *full* 262,144-token
context correctly by a positive control at the same length. The miss is a **model retrieval
limit** of this checkpoint, and it is a function of the needle's **absolute position in the
sequence**, not of context length, KV dtype, attention backend, prefill chunking, KV growth,
or the Mamba-2 kernels.

## Method

One prompt, built once on the CPU and reused byte-for-byte by every run
(`bench_long_context.synthetic_needle_sample()` + `trim_filler`, tokenized with the
checkpoint's own tokenizer, exactly 262,144 prompt tokens, 262,160 after the chat template).
The synthetic filler is already digit-free — the only digits anywhere in the haystack are the
seven of the needle `5663623` — so the 1M gate's digit-distractor artifact cannot apply.
Controls at 131,072 use the same builder at the same relative depth.

Requests go through `/v1/chat/completions` (greedy, `temperature 0`, `enable_thinking: false`,
48 decode steps, `ignore_eos`), never `/v1/completions` — the continuation-lottery artifact
closed on 2026-09-04. Pass = the literal `5663623` appears in the concatenated answer.

Server, identical for every row except the bisected variable:

```
FREETOKEN_PIN_BUDGET_GB=17 uv run ft serve \
  --model ~/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --memory-ratio 0.85 --max-running-requests 1 --max-seq-len-override 1048576 \
  --num-tokens 524288 --kv-grow-step-tokens 131072 --moe-cache-auto \
  --max-prefill-length 8192 --kv-cache-dtype q8_0 --attention-backend triton
```

Each variant is a **fresh server process** under `scripts/gpu_lock.sh`, and the 262,144-token
request is always the *first* request that process serves, so no spill, restore, session lease
or radix reuse is involved. `FREETOKEN_MAMBA2_STATE_DUMP` was on for every run: it saves the
end-of-prefill recurrent/conv state of all 23 Mamba-2 layers plus the sampled position's full
logit vector, i.e. the next-token distribution *at the question*.

## The matrix

Same prompt, needle at token 136,113 (depth 0.519), and the same prompt trimmed to 131,072
(needle at token 67,630, depth 0.516).

| | variant | KV dtype | attention | scan | chunk | KV | 262,144 | 131,072 |
|---|---|---|---|---|---|---|---|---|
| a | baseline (the reported failure) | q8_0 | triton | SSD kernel | 8192 | growable | **FAIL** | PASS |
| b | bf16 KV + FlashInfer | auto (bf16) | flashinfer | SSD kernel | 8192 | growable | **FAIL** | PASS |
| c | bf16 KV + triton | auto (bf16) | triton | SSD kernel | 8192 | growable | **FAIL** | PASS |
| d | reference Mamba-2 | q8_0 | triton | `FREETOKEN_MAMBA2_REF=1` | 8192 | growable | **FAIL** | PASS |
| e | half-size prefill chunks | q8_0 | triton | SSD kernel | 4096 | growable | **FAIL** | PASS |
| f | static KV (no growth) | q8_0 | triton | SSD kernel | 8192 | **static** | **FAIL** | PASS |
| g | fp8 KV (NVIDIA's recipe) | fp8_e4m3 | triton | SSD kernel | 8192 | growable | **FAIL** | PASS |
| h | bf16 dense (`FREETOKEN_NEMOTRON_DENSE_DEQUANT=1`) | q8_0 | triton | SSD kernel | 8192 | growable | **FAIL** | PASS |

Eight for eight. Prefill rates (prompt tokens / TTFT) for the record: (a) 1,847 → 3,100 tok/s,
(b) 4,391 → 5,767, (c) 3,964 → 5,412, (d) 1,683 → 2,652, (e) 1,763 → 2,952, (g) 2,107 → 3,487,
(h) 1,859 → 3,112 at 262K → 131K.

### What the wrong answers look like

The 262K answers are not "a wrong passcode". They are degenerate haystack echo with token-level
corruption, and they are near-identical across backends:

- (a)/(f): `The orchard ledger says the copper marker is inactive. The orchard ledledger says the copper marker is inactive. The orchard ledder says the copper ma…`
- (b) FlashInfer: `The orchard ledger says the copper marker is inactiveactive.\nThe orchard ledledger saysthe copper marker is inactiveactive.…`
- (d) reference scan: `The secret passcode is inactive.<|im_end|>\n</_________________________ 11111111…`
- (h): `The secret passcode is inactive.…`

At 131,072 the same servers answer `The secret passcode is 5663623.`

That two numerically independent attention implementations (Triton packed-q8_0 loader vs
FlashInfer bf16) converge on the *same* degenerate continuation is itself evidence against a
kernel bug: an indexing or precision fault would not agree across them.

## Exonerations, in order of strength

### 1. Growable KV is numerically a no-op — proved by exact equality

The strongest suspicion going in was the KV growth boundary: with `--kv-grow-step-tokens
131072` a 131K prompt never crosses one and a 262K prompt crosses it mid-prefill, tearing down
CUDA graphs and rebuilding the MoE expert cache (`engine/engine.py:1299-1389`).

Variant (f) commits the whole pool up front. Its end-of-prefill state and its logits are
**bit-identical** to (a) at both lengths:

| a vs f | worst recurrent | worst conv | logits |
|---|---:|---:|---:|
| @262,160 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| @131,088 | 0.000e+00 | 0.000e+00 | 0.000e+00 |

and the 262K answers are byte-identical strings. The growth path — `grow_runtime_kv`, the MoE
rebuild, the graph recapture, `add_committed_pages` — changes nothing about the numbers.

### 2. State-divergence magnitude carries no signal (the A/B of the A/B)

Per-layer relative RMS `‖a−b‖/‖b‖` of the end-of-prefill Mamba-2 state:

| pair | length | worst recurrent | worst conv | logits | outcome |
|---|---:|---:|---:|---:|---|
| a vs c (q8_0 vs bf16 KV) | 262,160 | 5.71e-02 | 2.06e-01 | 1.17e-01 | both FAIL |
| a vs c | 131,088 | 1.46e-01 | 2.44e-01 | 2.34e-01 | both PASS |
| a vs d (kernel vs reference scan) | 262,160 | 2.90e-01 | 3.28e-01 | 4.09e-01 | both FAIL |
| a vs d | 131,088 | 2.86e-01 | 5.63e-01 | 3.31e-01 | both PASS |

The divergence at the failing length is **smaller** than at the passing length for a-vs-c, and
identical for a-vs-d. Consistent with the 2026-09-04 lesson: on a 52-layer hybrid at these
lengths, per-layer state RMS is amplification, not correctness.

### 3. Top-5 next-token logits at the question

From the state dump, at the last prompt token:

| variant | @131,072 | @262,144 |
|---|---|---|
| a / f | `The` 17.00, **`5` 16.88**, `1` 15.75, `cop` 15.38, `3` 14.88 | `The` 17.75, `1` 15.75, `0` 15.44, `8` 15.31, `6` 15.00 |
| b | `The` 16.75, `1` 16.38, **`5` 15.38**, `0` 15.38, `9` 15.06 | `The` 17.50, `1` 15.12, `0` 15.00, `8` 14.81, `6` 14.56 |
| c | **`5` 16.88**, `The` 16.62, `1` 16.00, `8` 14.94, `4` 14.62 | `The` 18.12, `1` 15.31, `0` 15.19, `8` 15.19, `c` 15.00 |
| d | **`5` 17.88**, `The` 17.62, `1` 15.50, `Pass` 14.69, `Answer` 14.31 | `The` 15.81, `1` 14.88, `0` 14.19, `8` 14.00, `9` 13.81 |
| e | `The` 17.12, `1` 16.75, **`5` 16.25**, `8` 15.50, `9` 15.25 | `The` 17.88, `c` 14.44, `8` 14.31, `1` 14.25, `0` 14.19 |

At 131K the needle's leading digit `5` is in the top-5 in every variant that has a digit there
at all; at 262K it is gone from all of them, displaced by `1`/`0`/`8`/`6` — precisely the
distractor set that produced the reported `1234`. Caveat: this is a weak instrument on its own,
because when top-1 is `The` the digits are chosen several steps later (the 262K depth-0.06
*pass* below also has no `5` in its top-5). The decoded answer remains the real signal.

### 4. Not the haystack's repetitiveness

The built-in synthetic filler is two sentences repeated 50,000× each. A control with a
seeded, non-repetitive, still digit-free haystack (14 sites × 15 adjectives × 14 nouns × 8
locations × 9 predicates, 180,000 distinct-ish records) behaves the same way: 131,072 PASS,
262,144 FAIL. The 262K answer there collapses to `012345678901234567890123456789…`.

## What actually predicts the outcome: absolute needle position

Once every engine variable was exhausted, the prompt itself was swept. All rows below are the
*same* server configuration (a).

**Length sweep at fixed depth ≈0.52 — non-monotonic, so there is no cliff:**

| prompt tokens | needle token | needle |
|---:|---:|---|
| 131,072 | 67,630 | PASS |
| 147,456 | 76,190 | **FAIL** |
| 163,840 | 84,751 | PASS |
| 180,224 | 93,311 | PASS |
| 196,608 | 101,871 | **FAIL** |
| 262,144 | 136,113 | **FAIL** |

A code path keyed on a power of two (2^18 = 262,144, or the checkpoint's tokenizer
`model_max_length` of 262,144) cannot produce a failure at 147,456 and a pass at 180,224.

**Depth profile at a fixed 262,144 tokens — the decisive control:**

| needle depth | needle token | tokens after the needle | needle |
|---:|---:|---:|---|
| 0.057 | 14,873 | 247,271 | **PASS** — `The secret passcode is 5663623.` |
| 0.267 | 69,958 | 192,186 | FAIL |
| 0.519 | 136,113 | 126,031 | FAIL |
| 0.761 | 199,443 | 62,701 | FAIL |
| 0.947 | 248,194 | 13,950 | FAIL |

and at 196,608 tokens: depth 0.058 PASS, depth 0.518 FAIL, depth 0.944 FAIL.
At 131,072 tokens: depth 0.267 PASS, depth 0.516 PASS.

**This is the whole result.** On a *fresh* 262,144-token prompt the engine prefills all 262,160
tokens, holds them in the KV/state, and reproduces a seven-digit fact planted at token 14,873
exactly. The identical machinery, identical length, identical everything, fails on a fact
planted at token 69,958. No paging, indexing, quantization, position-width or metadata defect
can be selective in the *content's* sequence position while being correct for the same length,
the same page count, the same number of prefill chunks and the same decode.

Read the other way: needle token 67,630 is recalled in a 131,072-token prompt and needle token
69,958 is not recalled in a 262,144-token prompt. Nearly the same absolute position; what
changed is the mass of homogeneous context that follows it. The surviving band shrinks as total
length grows — ~<15K tokens of it at 262,144.

Architecturally this is where a Nemotron-H hybrid would be expected to give: 52 layers, of
which only **6 are full attention** (`layers_block_type` indices 5, 12, 19, 26, 33, 42) — the
other 23 mixers are fixed-size Mamba-2 recurrent state that cannot grow with context. NVIDIA's
1M claim is for the BF16 checkpoint; this is the NVFP4 (4-bit weight) release, and variant (h)
only dequantizes the shared experts and lm_head, so the routed experts, attention and mamba
projections stay NVFP4 in every row here.

## Conclusion

- **No FreeToken bug was found.** q8_0 / fp8_e4m3 / bf16 KV, Triton / FlashInfer attention,
  SSD-kernel / reference Mamba-2, 4096 / 8192 prefill chunks, growable / static KV, and
  NVFP4 / bf16 dense all give the identical pass/fail pattern, and growable-vs-static is
  bit-identical.
- **The engine is demonstrably correct at 262,144 tokens** (depth-0.057 exact recall on a fresh
  262,144-token prompt).
- **The limit is the checkpoint's.** Reliable mid-depth needle recall holds to ~131K, is
  marginal and non-monotonic through ~180K, and is gone by 196,608 for anything past roughly
  the first 6% of the sequence.
- **The 262K/524K needle rows in the 2026-09-04 cache study, and the 1M gate's 262K miss, are
  not regressions.** They are this limit. Advertising 1M for *retrieval* is not supportable on
  this NVFP4 checkpoint; 1M for *capacity, throughput and coherence* still is, and the sizing
  work stands.

## Recommended acceptance bar for the 1M gate

Gate long-context serving on what the model can actually do, and say so explicitly:

- keep the mid-depth needle as a **131,072-token** gate (it is stable and it catches real
  engine regressions);
- at ≥196,608 tokens gate on a **depth ≤0.1** needle plus a capacity/coherence check, not on
  mid-depth retrieval;
- do not treat a mid-depth miss above ~180K as a FreeToken defect without first re-running the
  depth-0.05 control at the same length — that control is the engine's alibi and costs one
  request.

`benchmarks/bench_long_context.py` gained `--needle-depth` for this; the synthetic needle
defaults to 0.5 as before.

## Adjacent findings (real, not this bug, not fixed here)

1. `kernel/triton/attention.py:decode_launch_config` has **no context-length key**; every tuned
   branch requires `head_dim==256, 16 q heads, 2 kv heads` (the Ornith shape), so Nemotron
   (`head_dim 128`, 32 q heads, 2 kv heads) always takes the `(kv_splits=8, block_n=32,
   warps=4)` fallback. At 262K KV that is a 16-CTA grid on an 84-SM GPU, each CTA walking
   32,768 tokens. Split math is correct (double-ceil, no truncation, scratch assertions hold) —
   this is a **decode throughput** bug, not a correctness one, and it is the likely cause of the
   72.6 → 51.8 → 32.0 tok/s decode curve in the cache study. `num_kv_splits_ptr` is passed to
   both decode kernels and dereferenced in neither.
2. Triton KV loaders widen slot ids to int64 on **store** (`kv_quant.py:52,161`) but not on
   **load** (`attention.py:621,1119,1271` — `slots[None,:] * stride_ks`). For Nemotron's
   `stride_ks = 256` the int32 ceiling is ~8.4M slots, so it is safe here, but at head_dim 256
   / 8 kv heads it drops to ~1.05M — right on top of the 1M profile. Worth widening before the
   1M gate ships.
3. `scheduler/cache.py:compact_active_pages` rewrites the page-table row but not the radix
   nodes' `kv_indices`, and returns the vacated pages to the free list — a multi-turn/lease
   corruption that can only trigger once KV is above its initial growth step. Owned by the
   sibling scheduler task (`tasks/todo.md` follow-up ticket on `_maybe_shrink_growable_kv`);
   untouched here.
4. `--max-seq-len-override` bypasses the tokenizer's `model_max_length` silently
   (`engine/config.py:264-273`); the only trace is a transformers warning on the client side.
   Harmless for this model (`max_position_embeddings` is 1,048,576 and it is NoPE — no rotary
   table is ever built) but it is why "262,144 is exactly `model_max_length`" looked like a lead
   and is not one.

## Reproduction

Scratch tree (prompts, per-variant server logs, state dumps, `results.jsonl` with all 29 rows):
`/tmp/claude-1000/-home-lucas-ai-FreeToken/af23ede4-e8ad-4c8d-8b38-c8be515d8870/scratchpad/bisect262`.
The one-command form of the headline control:

```
uv run benchmarks/bench_long_context.py --synthetic-needle \
  --model ~/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --kv-cache-dtype q8_0 --attn triton --target-prompt-tokens 262144 \
  --max-context 1048576 --needle-depth 0.05     # PASS
#                       --needle-depth 0.5      # FAIL
```
