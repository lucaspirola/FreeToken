# Extend-path MoE — Nemotron-3.5-Lightning-30B-A3B-NVFP4, RTX 5080 (2026-09-05)

Ticket 1 of `benchmarks/results/nemotron35_lightning_5080_ngram_spec_2026-09-05.md` §6:
*"extend-path MoE: reuse the decode expert cache and stop paying a fixed per-layer gather
cost."* Base commit `193da80`. Host: RTX 5080 16 GB, WSL,
`FREETOKEN_PIN_BUDGET_GB=17`, `--moe-backend offload --moe-cache-auto --nvfp4-backend triton
--kv-cache-dtype q8_0`.

> **The 11.6 ms per MoE layer per forward is not a planning cost. It is the whole expert
> layer crossing PCIe.**
>
> `OffloadMoELayer._prefill_routed` streams **every** expert of the layer into its double
> buffer on **every** forward. Nothing in that movement path reads `topk_ids` — which is
> exactly why the cost is flat from 1 to 32 tokens. On this checkpoint that is 128 experts
> x 5.612 MB = **718 MB per layer** and **16.5 GB per forward**, and 16.5 GB / 267 ms is
> **61.9 GB/s**: a saturated PCIe 5.0 x16 link. A 1-token extend routes 6 experts per layer
> and moves 128. The waste factor is **21.3x**, and it is bytes, not host time.
>
> The fix is the one the ticket named, and it is one line of routing: an extend forward of
> m <= 64 tokens takes the **decode** path (`ensure_experts` + `copy_missing` fetch the experts
> these tokens actually route to, and only the ones not already resident; the NVFP4 decode GEMV
> is m-general).
>
> **Measured: 282.7 -> 27.7 ms at m = 1, 282.7 -> 30.2 at m = 8, 282.5 -> 30.9 at m = 32 --
> 9.2-10.2x on the forward and 23.6-27.3x on the MoE (11.4 -> 0.42-0.48 ms per layer).** The MoE
> is no longer the extend forward: at m = 32 it is 11.1 ms against Mamba-2's 15.6. 131K prefill
> is unchanged (5,059 -> 5,105 tok/s, +0.9 %), the needle is recalled in both arms, and the
> realistic greedy shape -- a long prompt whose last chunk is short -- is token-identical.
> A verify step now costs **4.4x** a graphed decode step instead of 42x, which projects
> **1.63x** on the copy class: n-gram speculation goes from impossible to worth building.

---

## 1. Where the 11.6 ms goes — read off the code, confirmed by arithmetic

`NemotronHMoE.forward` (`models/nemotron_h/model.py:358`) has one MoE call site,
`self.experts.routed_forward(...)`, which branches on `ctx.batch.is_prefill`
(`layers/moe.py:247`). An extend forward for a running request is a prefill batch, so it
takes `_prefill_routed` (`layers/moe.py:441`), whose movement half is:

```python
if cache.prefill_overlap:
    views = self._wait_prefill_overlap(cache)      # prefetch layer, prefetch layer+1, wait
    ...
cache.materialize_layer(self.layer_id)             # non-overlap: the same, synchronously
```

`prefetch_prefill_layer` (`moe/offload_cache.py:1116`) copies **the whole layer**:

```python
def copy() -> None:
    self._invalidate_prefill_buffer(buffer_id)
    for name, buffer in zip(self.bank_schema, self.prefill_bank_buffers):
        src = self.bank_sources[name][layer_id]     # [num_experts, ...]
        dst = buffer[buffer_id]                     # [num_experts, ...]
        dst.copy_(src, non_blocking=True)
```

`topk_ids` appears nowhere in that path. **That is the finding.** The routing decides which
rows the GEMM *reads*; it has never decided which rows are *fetched*.

### 1.1 The bytes

Nemotron 3.5 Lightning: hidden 2688, 128 routed experts, ungated ReLU² (up + down only),
I = 1856, 23 MoE layers. Native NVFP4 (`quant_format="nvfp4"`) banks per expert:

| bank | shape | bytes |
|---|---|---:|
| `gate_up_packed` | [1856, 2688/2] u8 | 2,494,464 |
| `gate_up_scale` | [1856, 2688/16] e4m3 | 311,808 |
| `gate_up_global` | [1] f16 | 2 |
| `down_packed` | [2688, 1856/2] u8 | 2,494,464 |
| `down_scale` | [2688, 1856/16] e4m3 | 311,808 |
| `down_global` | [1] f16 | 2 |
| **per expert** | | **5,612,548 (5.352 MiB)** |
| **per layer (x128)** | | **718.4 MB (685.1 MiB)** |
| **per forward (x23)** | | **16.52 GB (15.39 GiB)** |

15.39 GiB is exactly `docs/nemotron.md`'s "the NVFP4 routed-expert banks are 15.4 GiB" — the
extend path moves **the entire expert bank set, once per forward**.

### 1.2 The rate check

| quantity | measured (§4 of the n-gram write-up) | bytes | implied rate |
|---|---:|---:|---:|
| one MoE layer | 11.6 ms | 718.4 MB | **61.9 GB/s** |
| 23 MoE layers | 267 ms | 16.52 GB | **61.9 GB/s** |

PCIe 5.0 x16 is 63.0 GB/s unidirectional. **The extend MoE runs the link flat out**, and the
"host-side" `perf_counter` reading is the compute stream (and behind it the host) waiting on
the copy stream — which is also why host (303.6 ms) and CUDA-event (310.6 ms) agreed.

The n-gram write-up rejected bandwidth by pricing the **routed** experts: "138 x 5.35 MiB =
738 MiB, which at 52.9 GB/s would be 14 ms". That arithmetic is right, and it is the number
this ticket is trying to reach. The path moves 2,944 experts, not 138. `52.9 GB/s` is the
*gather* rate (scattered per-expert rows); the prefill copy is six contiguous whole-bank
`copy_()` calls per layer and gets the full link rate, which is why 61.9 > 52.9 is
self-consistent rather than a contradiction.

### 1.3 What is *not* costing anything

Everything the ticket suspected as host-side re-planning, checked and cleared on the default
configuration:

- **No Python loop over experts.** Six `Tensor.copy_` calls per layer, one per registered bank.
- **No host sync per layer.** `wait_prefill_layer` is `current_stream().wait_event(...)`, a
  stream wait. `begin_prefill` does one `copy_stream.synchronize()` per *forward*, and only
  when `--moe-prefill-hit-d2d` is on — which is **off by default** (`EngineConfig.
  moe_prefill_hit_d2d = False`) and was off in the run that measured 11.6 ms.
- **No per-forward plan build.** `_copy_src_ptrs_host`, `_copy_feat_bytes_by_layer_host` and
  the rest of the copy plan are built once, in `_build_copy_plan`.
- The one path with real per-layer host work — `_prefetch_split`'s numpy miss classification
  plus three `torch.tensor(python_list)` builds per layer — belongs to the **opt-in**
  hit-D2D split and does not run by default.

**So there is no per-forward re-planning to hoist above the threshold, and this change
deliberately does not touch the large-M path.** Its cost there is bytes that a full chunk
needs anyway (all 128 experts are routed to at M = 8192), hidden behind ~861 ms of GPU work.

## 2. The change

`OffloadMoeCache.use_cached_extend(layer_id, num_tokens)` gates one branch at the top of
`_prefill_routed`, and the branch is the decode path verbatim:

```python
if cache.use_cached_extend(self.layer_id, hidden_states.shape[0]):
    return self._cached_extend_routed(cache, hidden_states, topk_weights, topk_ids)
...
def _cached_extend_routed(self, cache, hidden_states, topk_weights, topk_ids):
    assert cache.quant_format.startswith("nvfp4"), cache.quant_format
    return self._decode_routed(hidden_states, topk_weights, topk_ids)
```

`_decode_routed` fetches exactly the experts these tokens route to, and only the ones not
already resident (`ensure_experts` rewrites `topk_ids` into slot ids and stages the misses;
`copy_missing` moves them device-side, no host sync), then runs the NVFP4 decode GEMV -- which
is m-general: its grid is `(m * top_k, cdiv(N, BLOCK_SIZE_N))`. Nothing else moves. The extend
path is never CUDA-graph captured, so there is no capture constraint.

**Gate** (`use_cached_extend`), each condition a fallback to the legacy full-layer stream:

| condition | why |
|---|---|
| `0 < num_tokens <= extend_cache_tokens` | the crossover, §2.1 |
| `quant_format in {nvfp4, nvfp4_marlin, nvfp4_b12x}` | the layouts that decode through the slot cache; every other format assumes position == expert id |
| `decode_target == "gpu"` | cpu/hybrid layers have their own decode routing |
| `not _size_class_enabled` | mixed-GGUF class-local row ids |
| `not is_cpu_layer` / `not is_unpinned_layer` | `copy_missing` cannot honour a slot remap without a device alias for the host bank |

**Flag:** `--moe-extend-cache-tokens` (`EngineConfig.moe_extend_cache_tokens`), **default 64**,
0 disables. `FREETOKEN_MOE_EXTEND_CACHE_TOKENS` overrides it at cache construction so an A/B
is two invocations of one binary, as `FREETOKEN_EXTEND_BLOCK_M` was for the prefill profile.

**The kernel this does NOT use.** The grouped prefill GEMM reads each distinct expert once per
token routed to it, where the GEMV re-reads it per route, so it should win as m grows. Pointing
it at the slot cache needs the token/expert sort to cover `cache_size` rows, and that faults
(§4.4) -- which the tree already knew: `_expert_gemm`'s `ds_fp4` branch uses the GEMV for its
small-chunk slot path because "sorting over the full slot cache would drown in padding". §7
keeps it as a ticket; at the measured widths the GEMV is already 23-27x.

### 2.1 Why 64 — a crossover, not a round number

The threshold should be the M at which the cached path stops being cheaper, i.e. where the
distinct experts an M-token forward routes to reaches `num_experts`. The n-gram write-up's §3
measured that curve directly (copy class, mean over 23 layers): D(1) = 6.00, D(2) = 10.56,
D(4) = 18.24, D(9) = 33.23, D(17) = 50.02. A power fit through the endpoints is
`D(m) ≈ 6.2 · m^0.75`, which reaches 128 at **m ≈ 57**. 64 is that crossover rounded to a
power of two.

Above it the cached path degenerates: it fetches at most 128 rows per layer, which is what the
legacy stream fetches unconditionally, and it additionally evicts the decode working set.
Below it, it fetches `D(m) x (miss fraction)` rows.

## 3. Cost model, and the prediction it made

Per layer, cached path: `PCIe = D(m) · f · 5.612 MB / 52.9 GB/s` (the measured gather rate --
scattered rows, so the lower of the two) and `HBM = m · top_k · 5.612 MB / 960 GB/s`, with `f`
the fraction of routed experts not already resident. `docs/nemotron.md` records a 51 % decode
miss rate under LFU at 16 lanes; a single warm stream re-routing to its own recent experts
should sit well below that, so `f ∈ [0.2, 0.5]` was taken as the bracket.

| m | D(m) | forward ms before | predicted after (f = 0.2 → 0.5) | **measured after** |
|---:|---:|---:|---:|---:|
| 1 | 6.0 | 290 | 26 → 31 | **27.7** |
| 8 | 29.5 | 308 | 41 → 63 | **30.2** |
| 32 | 83.4 | 314 | 75 → 136 | **30.9** |

The model is right at m = 1 and increasingly pessimistic above it: a warm single-stream cache
misses far less often than the 16-lane aggregate figure, so almost nothing crosses PCIe and the
residual is HBM plus the fixed Mamba-2 and attention cost. **Prefill at 8K chunks was predicted
unchanged** (M = 8192 is 128x the threshold, so the code path is untouched) and measured at
+0.9 %.

**What this does not fix: sub-3K prefill chunks stay host-bound.** The 267 ms is PCIe bytes,
not host work, so there is nothing to plan away, and a 512-token chunk already routes to ~128
experts per layer, where the cached path would save only the miss fraction and would pay for
evicting the decode working set every chunk. Raising the threshold into that range is a
separate, measurable question (§7).

## 4. Measurements

One GPU session, `193da80`, both arms the same binary with `extend_cache_tokens` forced from
a counting wrapper around `use_cached_extend`, so every row carries its own path proof
(`gate_hits/gate_calls`) and the recorded forward bucket (`"<m>/extend"`) proves the forward
really carried m tokens rather than being served from the radix tree. Every timed call gets a
fresh tail for the same reason. 90 unit tests pass (`tests/moe`, CUDA).

### 4.1 Per-forward cost, m = 1 / 8 / 32 (`probe_ngram_spec.py --layer-profile` shape)

| m | arm | forward host ms | mamba | attention | **MoE** | **MoE ms/layer** | gate |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | before | 282.7 | 15.37 | 1.54 | 263.09 | **11.439** | 0/23 |
| 1 | **after** | **27.7** | 14.37 | 1.29 | **9.63** | **0.419** | 23/23 |
| 8 | before | 282.7 | 15.96 | 1.50 | 262.58 | **11.416** | 0/23 |
| 8 | **after** | **30.2** | 15.60 | 1.28 | **10.78** | **0.469** | 23/23 |
| 32 | before | 282.5 | 16.57 | 1.63 | 261.59 | **11.374** | 0/23 |
| 32 | **after** | **30.9** | 15.64 | 1.47 | **11.07** | **0.481** | 23/23 |

The before arm reproduces the n-gram write-up exactly (282.5-282.7 ms, 11.37-11.44 ms per MoE
layer, flat from 1 to 32 tokens). After: **9.2-10.2x on the forward, 23.6-27.3x on the MoE**,
and the MoE is no longer the forward -- at m = 32 it is 11.1 ms against Mamba-2's 15.6 ms.
Measured 27.7-30.9 ms against a predicted 26-136 ms: the prediction's pessimistic end assumed
a 50 % miss rate, and a warm single-stream cache is far better than that.

**This lands below the eager decode forward (33.9 ms), which ticket 2 only hoped to reach.**
Re-running the copy class's projection (draft_rate 0.353, lambda 3.615) against a 30.2 ms
verify step and a 6.88 ms graphed decode step -- 4.4x, not 42x -- gives
`3.615 / (0.353 * 4.4 + 0.647)` = **1.63x**, above the 1.52x the ticket projected for step 1
alone and clear of the 1.25x bar MTP failed. A graph-captured verify forward (ticket 2) is now
an improvement, not a precondition.

### 4.2 131K prefill, needle, decode

Same 131,088-token needle prompt per arm, separate processes (a warm radix tree would serve
the second arm and measure nothing). 16 chunks of 8,192 plus a 16-token remainder, so the last
chunk takes the changed path inside an otherwise ordinary chunked prefill.

| | before | after |
|---|---:|---:|
| prefill | 5,058.9 tok/s | **5,104.5 tok/s** (+0.9 %) |
| needle recalled | yes | yes |
| decode | 9.28 tok/s | 9.39 tok/s |
| gate hits / calls | 0 / 391 | **23 / 391** |

23 of 391 is exactly one chunk's 23 MoE layers: the path fired where it should and nowhere
else. **No prefill regression** -- the +0.9 % is inside run-to-run noise, as it must be, since
at M = 8192 the code path is unchanged.

### 4.3 Greedy equivalence

This is a numerics change: a sub-threshold chunk now runs the decode GEMV where it ran the
grouped prefill GEMM, and the two reduce K in a different order. The gate is agreement, not
bitwise equality -- the same standard the 2026-09-04 extend-tile change was held to.

| prompt | shape | gate hits | result |
|---|---|---:|---|
| 8,192 tokens | no sub-threshold chunk (control) | 0 -> 0 | **identical**, 27/27 tokens |
| 8,232 tokens | one 8,192 chunk + a 40-token tail | 0 -> 23 | **identical**, 27/27 tokens |
| 30 tokens | the WHOLE prefill takes the new path | 0 -> 23 | identical for 60 tokens, then diverges |

The realistic shape -- a long prompt whose last chunk is short -- is token-identical. Only when
*every* MoE layer of the entire prefill switches kernel does the greedy stream eventually flip
on a near-tie (token 60 of 255, a word choice; both continuations answer the prompt). The 131K
needle run diverges earlier, but it was generated with `ignore_eos=True`, which forces both
arms past `<|im_end|>` into a degenerate repeat whose phase is not a meaningful signal; both
arms recall the passcode.

### 4.4 What the first attempt got wrong

The change first pointed the *grouped prefill GEMM* at the slot cache (a `num_slots` argument
widening `moe_align_block_size` to `cache_size` rows), because the grouped GEMM reads each
distinct expert once where the GEMV re-reads it per route. That faults: sgl's
`moe_align_block_size` over ~1,800 experts gives an illegal memory access on the real
geometry, and the repo had already written the reason down -- `_expert_gemm`'s `ds_fp4` branch
notes that its small-chunk slot path uses the GEMV because "sorting over the full slot cache
would drown in padding". The `num_slots` plumbing was backed out; §7 keeps it as a ticket.

## 5. Files

- `python/freetoken/layers/moe.py` — `_cached_extend_routed` and the `_prefill_routed` branch.
- `python/freetoken/moe/offload_cache.py` — `extend_cache_tokens`, `use_cached_extend`,
  `_CACHED_EXTEND_FORMATS`, the `FREETOKEN_MOE_EXTEND_CACHE_TOKENS` override.
- `python/freetoken/engine/config.py`, `python/freetoken/engine/engine.py`,
  `python/freetoken/server/args.py` — `--moe-extend-cache-tokens`.
- `tests/moe/test_extend_cache.py` — the gate table (16 CPU tests) plus 3 CUDA tests: the
  cached path agreeing with the full-layer path at m = 1/3/8, only the routed experts fetched,
  and above-threshold still streaming the full layer. 90 pass across `tests/moe` on the 5080.
- `docs/nemotron.md`, `tasks/todo.md`, `tasks/lessons.md`.

## 6. Reproduction

```
# per-forward and per-mixer cost at m = 1 / 8 / 32, both arms in one model load
FREETOKEN_PIN_BUDGET_GB=17 PYTHONPATH=python scripts/gpu_lock.sh \
  scratchpad/extend_moe/run.sh > scratchpad/extend_moe/session.log 2>&1
# greedy equivalence (one process per arm, cold prefix cache)
... scripts/gpu_lock.sh scratchpad/extend_moe/run3.sh > .../session3.log 2>&1
```

`driver1.py` forces the arm from a wrapper around `use_cached_extend` that also counts the
calls taking the cached path, and keys every row on the `"<m>/extend"` forward bucket with a
fresh tail per timed call — without that, the radix tree serves the prompt and an m-token
extend is silently measured as a 1-token one.

Do not pipe `scripts/gpu_lock.sh` into anything — its exit trap runs `pkill -9 -g $$` and
kills the reader; redirect to a file and grep the file. Every run here ended `Killed` after
writing complete output.

## 7. Still open — tickets

1. **The grouped prefill GEMM over the slot cache.** The GEMV re-reads an expert per route;
   the grouped GEMM reads it once per distinct expert, which is ~2.3x less HBM at m = 32 and
   grows with m. It needs a token/expert sort that does not span `cache_size` rows — either
   the Triton `moe_align_block_size` (whose small-numel path is a single CTA over
   `next_pow2(cache_size)` lanes) or a compaction of the routed slot ids to a dense range with
   a gather back into `expert_ids`. Worth roughly 10 ms of a 31 ms forward at m = 32; not
   worth blocking this on.
2. **Raise the threshold for small prefill chunks, or do not.** The scheduler's interleave
   share produces 512-token chunks (§R7 ticket 1 of the soak write-up), and those pay the full
   16.5 GB stream for ~128 routed experts per layer. The cached path would save the miss
   fraction (~2x) and cost a full eviction of the decode working set per chunk. Decidable in
   one run: per-chunk time and the following decode's miss rate at
   `--moe-extend-cache-tokens` 64 / 512 / 2048 on a 131K prompt.
3. **The slot-cache miss counters now include extend forwards.** `ensure_experts` bumps
   `lru_stats` and `decode_freq`, so `--moe-collect-stats`' "MoE decode miss stats" and the
   pageable-layer profile both see extend routings below the threshold. Arguably correct (they
   are slot-cache events), but the metric's name no longer says what it counts.
4. **Ticket 2 of the n-gram write-up is now an improvement, not a precondition.** A verify
   step costs 4.4x a graphed decode step (§4.1), which already projects **1.63x** on the copy
   class. A graph-captured fixed-width verify forward would remove most of the residual 30 ms
   — half of which is now Mamba-2, not MoE.
