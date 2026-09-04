# 262K needle — root cause on the FreeToken side (2026-09-04)

Ticket: `tasks/nemotron35-handover.md` item 1, reopened by
`benchmarks/results/nemotron35_lightning_5080_262k_crossengine_2026-09-04.md` (llama.cpp
Q4_0 recalls the byte-identical prompts 8/8 where FreeToken misses above depth ~0.1).

**Verdict: FreeToken floors the Mamba-2 discretized timestep at `dt >= time_step_min =
1e-3` during prefill. `time_step_min` is HF's *initializer* range for `dt_bias`, not a
runtime bound; the floor caps every Mamba head's memory horizon at `1/(|A|*1e-3)` tokens
regardless of what the network computes, which is invisible below ~131K and destroys
mid-depth needle recall above it. Removing the floor turns every previously failing point
PASS with no other change — 11/11 across the bisect's length sweep and the 262K depth
profile, answer `5663623` every time, which is exactly llama.cpp's result on the same
prompts.**

Fix: `python/freetoken/models/nemotron_h/config.py` — the dt floor is now `0.0`
(`_dt_floor()`), with `FREETOKEN_NEMOTRON_DT_MIN=<float>` as the A/B escape hatch.
One number. vLLM passes `dt_limit=(0.0, inf)`, llama.cpp does not clamp at all, and
FreeToken's own *decode* kernel never clamped — so zero also removes a prefill/decode
inconsistency that had been there since Phase 2.

## The A/B

Same host, same hour, same server flags, same prompt files (the bisect's, SHA-1 matched
in the cross-engine run), every request the **first** request its server process serves
(`#cached-token: 0` in every prefill log line — no radix reuse, no session lease).

```
FREETOKEN_PIN_BUDGET_GB=17 uv run ft serve \
  --model ~/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --memory-ratio 0.85 --max-running-requests 1 --max-seq-len-override 1048576 \
  --num-tokens 524288 --kv-grow-step-tokens 131072 --moe-cache-auto \
  --max-prefill-length 8192 --kv-cache-dtype q8_0 --attention-backend triton
```

| run | dt floor | prompt tokens | needle token | depth | needle | answer |
|---|---|---:|---:|---:|---|---|
| **r0** (baseline) | 1e-3 | 131,072 | 67,630 | 0.516 | **PASS** | `The secret passcode is 5663623.` |
| **r0** | 1e-3 | 147,456 | 76,190 | 0.517 | **FAIL** | `1234567890…abc1234567890…` |
| **r0** | 1e-3 | 262,144 | 136,113 | 0.519 | **FAIL** | `The orchard ledger says the copper marker is inactive. The orchard ledledger…` |
| **r3** (fix) | 0 | 147,456 | 76,190 | 0.517 | **PASS** | `5663623` |
| **r3** (fix) | 0 | 262,144 | 136,113 | 0.519 | **PASS** | `5663623` |

### The whole failing matrix, re-run with the fix (r6)

One server, `FREETOKEN_NEMOTRON_DT_MIN=0`, ten requests (the 147,456 row is r3's, run on
its own server), `#cached-token: 0` on all 48 prefill batches — no prefix reuse, no
session lease, every request a full cold prefill. These are the exact rows the bisect and
the cross-engine check reported, so the columns are directly comparable.

| prompt tokens | depth | needle token | bisect (floor) | **fixed** | llama.cpp Q4_0 |
|---:|---:|---:|---|---|---|
| 131,072 | 0.516 | 67,630 | PASS | **PASS** | PASS |
| 147,456 | 0.517 | 76,190 | FAIL | **PASS** (r3) | PASS |
| 163,840 | 0.517 | 84,751 | PASS | **PASS** | — |
| 180,224 | 0.518 | 93,311 | PASS | **PASS** | — |
| 196,608 | 0.518 | 101,871 | FAIL | **PASS** | PASS |
| 196,608 | 0.944 | 185,585 | FAIL | **PASS** | — |
| 262,144 | 0.057 | 14,873 | PASS | **PASS** | PASS |
| 262,144 | 0.267 | 69,958 | FAIL | **PASS** | PASS |
| 262,144 | 0.519 | 136,113 | FAIL | **PASS** | PASS |
| 262,144 | 0.761 | 199,443 | FAIL | **PASS** | PASS |
| 262,144 | 0.947 | 248,194 | FAIL | **PASS** | PASS |

11/11, answer `5663623` every time — FreeToken now matches llama.cpp on every point the
cross-engine check measured, and the non-monotonic band is gone. TTFT: 41.6 s @131K,
60.8 @163K, 71.7 @180K, 83.4 @196K, 139.0-140.0 @262K.

r0's 262K answer is byte-identical to the failure recorded in the bisect and re-recorded
in the cross-engine run, so this is the same defect and not a new one.

TTFT is unchanged by the fix (147K: 51.99 s baseline vs 52.25 s fixed; 262K: 141.91 vs
141.85). `tl.clamp(dt, 0.0, inf)` on a softplus output is a no-op, so the floor's removal
costs nothing.

## Why the floor is wrong

`config.json` carries `time_step_min: 0.001`, `time_step_max: 0.1`,
`time_step_floor: 0.0001`. In HF's `NemotronHMamba2Mixer.__init__` these are the
*initialization* range for `dt_bias` (`dt_bias = inv_softplus(U[log tmin, log tmax])`,
`.clamp(min=time_step_floor)`). The forward then reuses one of them as a runtime bound:

```python
# transformers/models/nemotron_h/modeling_nemotron_h.py:381
self.time_step_limit = (config.time_step_min, float("inf"))   # "# No upper limit"
```

and FreeToken copied that into the mixer (`model.py`, `dt_limit=(args.time_step_min, inf)`)
where the Triton chunk-state kernel applies it after softplus
(`kernel/triton/mamba2/ssd_chunk_state.py:140`, `dt = tl.clamp(dt, dt_min, dt_max)`).

The other two implementations of the same architecture do not:

| stack | dt handling |
|---|---|
| HF `nemotron_h` (pure-Torch fallback) | `clamp(softplus(dt+bias), 1e-3, inf)` |
| **vLLM** `mamba_mixer2.py` | `dt_limit=(0.0, float("inf"))` |
| **llama.cpp** `ggml_compute_forward_ssm_scan` (`ggml/src/ggml-cpu/ops.cpp:9468`) | `softplus`, **no clamp** |
| **FreeToken decode** (`selective_state_update.py:184-190`) | `softplus`, **no clamp** |

So FreeToken was reproducing an HF artifact in prefill while its own decode kernel did
the right thing.

### The mechanism, in the checkpoint's own numbers

Mamba-2 retains an impulse for `exp(-|A| * dt * T)`. Flooring `dt` at `1e-3` therefore
imposes a hard per-head horizon of `1/(|A| * 1e-3)` tokens *no matter what the dt
projection computes for the token*. Read from the 23 `A_log` / `dt_bias` tensors
(1,472 heads):

```
softplus(dt_bias):  min 1.14e-06   p1 6.10e-05   p5 3.92e-04   p50 5.49e-02   max 2.13e+01
exp(A_log) = |A|:   min 1.16e-04   p50 3.09e-01   max 2.20e+04
heads whose resting dt is already below the 1e-3 floor: 133 / 1472  (9.0%)
largest gain the floor applies to such a head: 880x
```

The model deliberately configures heads two to three orders of magnitude below the floor
— that is what a "remember this for a long time" head looks like — and the input-dependent
`dt` projection can push any head further down on tokens it wants to skip. The floor
deletes exactly that regime, and only that regime.

`tests/models/test_nemotron_h.py::test_dt_floor_would_erase_a_long_memory_head` pins the
consequence on the scan itself: one head, `|A| = 0.3`, `dt = 1e-5`, unit impulse at token
0, read out 32,768 tokens later. Unfloored, 90.6% of the impulse survives; floored,
0.005% does — a 1.7e4x difference from changing nothing but `dt_limit`.

### Why it looked non-monotonic

The bisect's length sweep at fixed depth (PASS 131,072 / FAIL 147,456 / PASS 163,840 /
PASS 180,224 / FAIL 196,608 / FAIL 262,144) reads like noise because it is: the floor does
not produce a cliff, it produces a *steadily shrinking* set of heads that can still carry
information across the prompt, and whether one particular needle survives the erosion is
a coin flip in the band where the model is marginal. That is also why every engine-side
variant in the bisect failed identically — all eight shared the floor — and why the
depth-0.05 control passed: a needle 14,873 tokens in is still inside the horizon the
floor leaves intact.

## What was exonerated on the way

### 1. The FP8 W8A8 Mamba projections are not the cause

The checkpoint is `MIXED_PRECISION`, not uniform NVFP4:

| family | scheme |
|---|---|
| `layers.{N}.mixer.{in,out}_proj` (23 mamba layers) | **FP8 W8A8**, weights *and* a static per-tensor `input_scale` |
| routed experts, shared experts, `lm_head` | NVFP4 **W4A16**, group 16 (`input_activations: null`) |
| attention `q/k/v/o_proj`, `conv1d`, router `gate`, embeddings, norms | excluded — plain BF16/FP32 |

That FP8 pair is the **only** activation quantization anywhere in FreeToken's Nemotron
path (the NVFP4 dense and fused-MoE kernels are W4A16 end to end: activations are loaded
bf16 or widened to fp32, never scaled or clamped; `models/loader.py:144` says so
explicitly). `fp8_pertensor_linear` quantizes the activation as
`clamp(x / input_scale, -448, 448)`.

`FREETOKEN_DEBUG_FP8_ACT_STATS=<file>` (new,
`python/freetoken/models/nemotron_h/fp8_act_stats.py`, the same shape as `state_dump.py`)
records per-module `amax`, the `448 * input_scale` ceiling and the clipped-element count
during prefill. Measured on the r0 server, per request:

| prompt | needle | clipped fraction (all 46 matrices) | worst `amax / limit` |
|---:|---|---:|---:|
| 131,072 | PASS | 1.8052e-05 | 6.46 (`layers.7.mixer.out_proj`) |
| 147,456 | FAIL | 1.8066e-05 | 6.58 (`layers.7.mixer.out_proj`) |

The activations *do* saturate the calibrated scale on 11 of 46 matrices — but by the same
amount at the length that passes and the length that fails. It is a constant quality tax,
not the length-dependent term, so it cannot be the cause. (It remains a real difference
against llama.cpp, which quantizes no activations; worth a separate look if quality ever
needs another point.)

### 2. Everything the bisect already retired stays retired

KV dtype (q8_0 / fp8_e4m3 / bf16), attention backend (Triton / FlashInfer), prefill chunk
size, growable-vs-static KV (bit-identical), dense NVFP4 dequantization, kernel-vs-
reference Mamba-2 scan. All eight of those variants carried the dt floor, which is why
they all failed identically — a matrix run inside one engine cannot exonerate a term
common to every cell of it.

## The change

Behaviour change: mid-depth long-context retrieval is restored, so the bisect's
"gate mid-depth needles at depth <=0.1 above 196,608" acceptance bar is withdrawn and the
262K/524K needle rows in the cache study and the 1M gate should be re-measured.

```
python/freetoken/models/nemotron_h/config.py   _dt_floor() -> 0.0 (was hf.time_step_min)
                                               + FREETOKEN_NEMOTRON_DT_MIN escape hatch
python/freetoken/models/nemotron_h/model.py    comment; + the FP8 act-stats debug hook
python/freetoken/models/nemotron_h/fp8_act_stats.py   new, env-gated debug instrument
tests/models/test_nemotron_h.py                3 tests (see below)
```

Focused suite, GPU included, after the change:
`scripts/gpu_lock.sh uv run pytest tests/models/test_nemotron_h*.py
tests/kernels/test_mamba2_ssd.py -q` -> **81 passed, 2 skipped**. `ruff check` clean on
every touched file (the one remaining `E741` is pre-existing in `weight.py`, untouched).

Tests:
- `test_mamba_mixer_does_not_floor_dt` — `mixer.dt_limit == (0.0, inf)`, and `_decode_scan`
  still never mentions `dt_limit` (prefill and decode agree).
- `test_dt_min_env_restores_the_floor` — the A/B hatch works.
- `test_dt_floor_would_erase_a_long_memory_head` — the mechanism, on
  `models/nemotron_h/chunk_scan.py`: retention over 32,768 tokens is 0.906 unfloored and
  5.4e-05 floored.
- `test_real_lightning_config_parses_and_builds_on_meta` updated (`time_step_min == 0.0`).

## Reproduction

Scratch tree (driver scripts `serve.sh` / `run.sh` / `run_multi.sh` / `drive.py`, per-run
server logs, `results.jsonl`, FP8 activation-statistics snapshots):
`/tmp/claude-1000/-home-lucas-ai-FreeToken/f4e2e9e3-f4f5-40d0-9980-b3b09d1ef47d/scratchpad/rootcause`.
Prompt files are symlinks into the bisect's tree
(`…/af23ede4-…/scratchpad/bisect262/prompt_*.txt`); neither survives a WSL restart.

```bash
# baseline (floor restored) vs fixed, one variant per gpu_lock hold
VARIANT=r0 PORT=8140 LENGTHS=131072,147456,262144 scripts/gpu_lock.sh <scratch>/run.sh
VARIANT=r3 PORT=8141 LENGTHS=147456,262144        scripts/gpu_lock.sh <scratch>/run.sh
VARIANT=r6 PORT=8142 SPECS="prompt_:131072 prompt_:163840 prompt_:180224 \
  prompt_:196608 prompt_d05_:262144 prompt_d25_:262144 prompt_:262144 \
  prompt_d75_:262144 prompt_d95_:262144 prompt_d95_:196608" \
  scripts/gpu_lock.sh <scratch>/run_multi.sh
#   r0 = stock flags + FREETOKEN_DEBUG_FP8_ACT_STATS
#   r3 = the same + FREETOKEN_NEMOTRON_DT_MIN=0   (now the default; use
#        FREETOKEN_NEMOTRON_DT_MIN=0.001 to reproduce the failure on the fixed tree)

# one-command form of the headline result
uv run benchmarks/bench_long_context.py --synthetic-needle \
  --model ~/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --kv-cache-dtype q8_0 --attn triton --target-prompt-tokens 147456 \
  --max-context 1048576 --needle-depth 0.5
```
