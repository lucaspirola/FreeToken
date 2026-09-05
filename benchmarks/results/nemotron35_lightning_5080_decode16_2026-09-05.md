# 16-way batched decode on a 5080 — where the step goes, and the graph that was not there

Tree: `78f29d3` + this change. Model NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4
(52 blocks = 23 Mamba-2 + **23 MoE** + **6 full attention**; 128 routed experts, top-6,
H=2688, I=1856; 5.612 MB per NVFP4 expert, 16.5 GB of banks). RTX 5080, 84 SMs, 16.3 GB,
PCIe 5.0 x16. Host 33 GiB. Every timed run under `scripts/gpu_lock.sh`, never piped.

Reference points this study starts from: single-stream decode **145.3 tok/s at 131K**
(`acc91e9`, `..._decode_launch_2026-09-04.md`); 16-way soak aggregate **99.9 tok/s stage /
190.6 passthrough** (`..._switchyard_soak_2026-09-04.md` §V3); 16-way `bench_decode_moe`
aggregate **168.2 tok/s** with LFU (`..._cache_study_2026-09-04.md` §2b).

## 0. Summary

**Where the 16-lane step goes** (measured, §2): **74 % PCIe expert misses**, 12 % expert
GEMV, 8 % attention at 131K, 1 % Mamba-2, 4 % launch. The gather runs at **51-52 GB/s at
every batch size and miss count** — the measured PCIe roofline — so the movement term has no
headroom and is a *bytes* problem, not a kernel one. Hypotheses (b), (c) and (d) from the
brief are all clean at batch 16: the `acc91e9` split rule is still optimal (64 splits, 80 %
of the card's bandwidth), the m-general decode GEMV needs no switch to the grouped GEMM
(11k CTAs, 71.5 % of roofline), and Mamba-2 is one batched launch worth 0.9 ms.

**What was broken** (§2e): `_elastic_graph_batch_sizes` returned `(1,2,3,4,8)` for every
elastic capacity tier while `can_use_cuda_graph` gates on `max(set)`, so on the P2 profile
**every decode batch of 9-16 lanes ran eager** — 314 of the 427 decode batches (73.5 %) of
the passing `13af13d` soak, 421 of which were taken at elastic capacity 16.

**The fix** (§3): the capture set is dense to 16, then a 1.33-1.5x ladder, with the tier's
own capacity always appended. The obvious sparse version (`[1,2,3,4,8,16]`) was measured and
would have been a **net loss** (§4d): a padded row on an offload-MoE model routes its own
experts, so padding 12 up to a bs-16 graph costs **6.7 % more than running eager**, and
164 of the soak's batches sit in the 9-15 band that would have padded.

**Result** (§4a): at 12 lanes in a 16-request pool, eager -> exact graph is
**143.21 -> 153.84 tok/s, 1.074x**, step 83.8 -> 78.0 ms, P(after > before) = 0.85 over
11/10 intervals. At 16 lanes, 1.039x (within ~3 % run-to-run spread at that width).
Weighted over the soak's own batch histogram, ~5 % on its decode half. The dense set costs
**80 MiB** of VRAM and under a second per elastic resize, and does not shrink the expert
cache. Single-stream and the 131K needle are untouched by construction and were measured
anyway: **135.84 tok/s at bs=1**; **132.32 tok/s decoding a 130,016-token context, needle
recalled exactly**.

**The honest framing** (§4c): the CUDA graph is worth ~27 ms at bs=1 and only 4-6 ms at
12-16 lanes, because a wide step is 78-107 ms of GPU work and the eager launch path's ~30 ms
of Python runs ahead of it. At bs=1 this model is host-bound; at 16 lanes it is PCIe-bound,
and **no launch-path change makes 16-way decode fast while 74 % of the step is PCIe at the
link roofline** (§5 has the arithmetic a future attempt needs).

## 1. Exact commands

```bash
# Phase A -- weightless kernel sweep, 5 minutes, no checkpoint loaded
scripts/gpu_lock.sh benchmarks/decode16/phaseA.sh benchmarks/decode16/runs/phaseA
#   -> bench_mamba2_decode --batch 1 2 4 8 16
#   -> bench_nvfp4_moe_kernels --backend triton b12x --decode-m 1 2 4 8 16 32
#   -> bench_offload_cache_copy --models nemotron35-lightning --batch-sizes 1 4 16
#   -> bench_decode_launch --q-heads 32 --kv-heads 2 --head-dim 128 --quant q8_0
#        --ctx-lens 4096 32768 131072 --batch-sizes 1 8 16 --splits 1 2 4 8 16 32 64

# Phase B/C/D/E -- the graph A/B on the real serving path, P2 profile, one server per arm
scripts/gpu_lock.sh benchmarks/decode16/phaseB.sh benchmarks/decode16/runs/phaseB  # mixed 16
scripts/gpu_lock.sh benchmarks/decode16/phaseC.sh benchmarks/decode16/runs/phaseC  # uniform 16, bs=1, needle
scripts/gpu_lock.sh benchmarks/decode16/phaseD.sh benchmarks/decode16/runs/phaseD  # 12 lanes, exact graph
scripts/gpu_lock.sh benchmarks/decode16/phaseE.sh benchmarks/decode16/runs/phaseE  # 12 lanes padded to a bs-16 graph
scripts/gpu_lock.sh benchmarks/decode16/phaseF.sh benchmarks/decode16/runs/phaseF  # 12 lanes, dense set: the headline A/B
```

`FREETOKEN_ELASTIC_GRAPH_MAX_BS=8` reproduces the pre-fix capture set exactly, so every
before/after pair below is **two runs of the same binary**, not a rebuild.

## 2. Where a 16-lane decode step goes

Measured per-layer costs at the Lightning geometry, batch 1 vs batch 16, multiplied by
the layer counts of one forward (6 attention, 23 MoE, 23 Mamba-2). Movement numbers come
from the 2026-09-04 cache study's measured miss rates (bs=1: 0.718 misses/layer, 12.0 %;
bs=16 LFU: 31.45 misses/layer, 51.1 %) against this run's measured gather bandwidth.

| term | source | bs=1 per layer | bs=1 per step | bs=16 per layer | bs=16 per step | share @16 |
|---|---|---|---|---|---|---|
| **MoE expert misses over PCIe** | 51.9 GB/s gather (measured) | 0.075 ms | **1.73 ms** | 3.40 ms | **78.1 ms** | **74 %** |
| MoE expert GEMV (HBM) | `bench_nvfp4_moe_kernels` triton | 0.058 ms | 1.33 ms | 0.559 ms | 12.9 ms | 12 % |
| decode attention @131K | `bench_decode_launch` 64 splits | 0.146 ms | 0.88 ms | 1.476 ms | 8.9 ms | 8 % |
| Mamba-2 selective-state update | `bench_mamba2_decode` graphed | 0.008 ms | 0.18 ms | 0.038 ms | 0.9 ms | 1 % |
| **Python / launch, eager (off-graph)** | measured A/B, §4a | — | 27 ms | — | **3.9 ms** | **4 %** |
| modelled total | | | 4.1 ms (+27 eager) | | ~101 ms (+4 eager) | |

At 16 lanes the step is **movement-bound and nothing else is close**: 74 % of it is the
PCIe gather of the routed experts the GPU slot cache does not hold.

### (a) Expert misses over PCIe — at the link roofline, so the bytes are the only lever

`bench_offload_cache_copy` at the Lightning bank layout (6 NVFP4 banks, 8.04 MiB of
allocated slot per expert-layer; 5.612 MB of it is the ungated ReLU² weight the GEMV
reads):

| bs | active | miss rate | misses | time ms | copied MiB | **GB/s** | tok_ms (23 layers) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6 | 0.25 | 2 | 0.350 | 16.1 | 48.1 | 8.06 |
| 1 | 6 | 1.00 | 6 | 0.995 | 48.2 | 50.9 | 22.88 |
| 4 | 24 | 0.50 | 12 | 1.965 | 96.5 | 51.5 | 45.20 |
| 16 | 96 | 0.25 | 24 | 3.898 | 193.0 | 51.9 | 89.64 |
| 16 | 96 | 0.50 | 48 | 7.777 | 386.0 | **52.0** | 178.86 |
| 16 | 96 | 1.00 | 96 | 15.541 | 771.9 | 52.1 | 357.43 |

The gather runs at **51-52 GB/s at every batch size and every miss count**, against the
52.9 GB/s `ft bench bw` scattered-gather ceiling and PCIe 5.0 x16's 63 GB/s. **The copy
kernel has no headroom left** — there is no launch-config or fusion win hiding here, and
the LFU is not thrashing in the sense of a policy defect either: at 16 lanes the step's
working set is ~61.6 distinct experts/layer × 23 = 1,417 against the ~976 slots the pool
leaves at elastic capacity 16, so the cache is **capacity-bound with near-uniform per-step
demand**. LFU already beats the uniform-random model (48.9 % hit vs 976/2944 = 33 %); LRU
collapses to 99.6 % miss. Reducing the 78 ms means moving fewer bytes, which means either
more slots or fewer distinct experts — neither is a bounded change, so this study does
not attempt it. See §6 for what a future one would have to do.

Cross-check: an expert routed to at 16 lanes serves 16·6/61.6 = 1.56 tokens, so nothing
about the fetch amortizes with batch — the D(m) ≈ 6.2·m^0.75 law says the distinct-expert
count grows as m^0.75 while the tokens grow as m, and that sublinearity is the *only*
thing keeping 16-way aggregate above single-stream at all.

### (b) MoE GEMV at m=16 — not the m-general kernel's fault, but the roofline dips

`bench_nvfp4_moe_kernels`, one MoE layer, triton Marlin-style decode GEMV:

| M | distinct routed | us/layer | GB/s | % of 960 |
|---:|---:|---:|---:|---:|
| 1 | 6.0 | 57.7 | 584.6 | 60.9 % |
| 4 | 22.1 | 168.3 | 739.0 | 77.0 % |
| 8 | 39.5 | 299.2 | 742.2 | **77.3 %** |
| 16 | 68.2 | 559.3 | 686.0 | 71.5 % |
| 32 | 98.9 | 1064.3 | 522.3 | 54.4 % |
| 64 (grouped GEMM) | 121.9 | 1011.7 | 677.2 | 70.5 % |

The decode grid is `(m·top_k, cdiv(N, BLOCK_N))` = 11,136 CTAs at m=16, so it is not
under-occupied. Efficiency peaks at m=8 and falls to 71.5 % at 16 and 54.4 % at 32; the
grouped prefill GEMM does not overtake it until well past 32 routed rows. **No kernel
switch at m>=8 is indicated** — the crossover the brief asked about does not exist at 16.
The 5.8 percentage points lost between m=8 and m=16 are worth 1.0 ms of a ~100 ms step.

### (c) Decode attention at 16 lanes — the acc91e9 split rule still wins, batch and all

`decode_launch_config` picks `kv_splits` as if batch were 1 (`_grid_filling_splits` has no
batch parameter), and the stage-1 grid is `batch · head_blocks · splits`, so at 16 lanes it
launches **2,048 CTAs on 84 SMs**. That looked like over-splitting. It is not: the kernel is
bandwidth-bound, and the extra CTAs buy memory parallelism, not redundant work.

| ctx | batch | best splits | ms/layer | vs 64 splits | 64-split GB/s |
|---:|---:|---:|---:|---:|---:|
| 4,096 | 1 | 32 | 0.017 | 1.24x | 106.6 |
| 4,096 | 16 | 16 | 0.041 | **1.27x** | 685.6 |
| 32,768 | 16 | 32 | 0.394 | 1.01x | 721.1 |
| 131,072 | 8 | 64 | 0.792 | 1.00x | 720.3 |
| 131,072 | 16 | **64** | **1.476** | 1.00x | **773.0** (80 % of roofline) |

At the long contexts that matter, 64 splits is still the optimum at batch 16 and reaches
80 % of the card's bandwidth. The only loss is short-context × wide-batch (4K × 16 lanes,
where 16 splits would be 1.27x faster) and that is 0.011 ms/layer × 6 layers = **0.07 ms
of the step**. Not worth a batch-aware branch and its graph-capture consequences.
Agreement gate: max |Δ| vs the 8-split baseline 4.88e-04 at 4K, 1.22e-04 at 131K — bf16
scale, as expected for a reordered log-sum-exp reduction.

### (d) Mamba-2 at batch 16 — batched in one launch, 1 % of the step

`_selective_state_update_kernel` launches `(cdiv(dim,64), bs, nheads)` — one kernel for the
whole batch, with `do_not_specialize` on every runtime integer so a new batch size never
recompiles. Graph-replay cost per layer:

| backend | bs=1 | bs=2 | bs=4 | bs=8 | bs=16 | state GB/s @16 |
|---|---:|---:|---:|---:|---:|---:|
| triton (default) | 7.95 us | 9.55 | 14.66 | 23.23 | **38.43 us** | 1,734.6 |
| flashinfer | 7.34 us | 8.22 | 12.54 | 17.92 | **26.75 us** | 2,361.7 |

23 layers × 38.4 us = **0.88 ms/step** at 16 lanes. `FREETOKEN_MAMBA2_DECODE=flashinfer`
would save 0.27 ms/step (0.3 %) — noted, not pursued; the triton default is the one the
graph-capture and rollback paths are tested against.

### (e) Python / launch overhead — **the finding: batch 9-16 was never CUDA-graphed**

`engine.py::_elastic_graph_batch_sizes` returned

```python
return [bs for bs in (1, 2, 3, 4, 8) if bs <= capacity]
```

and `GraphRunner.can_use_cuda_graph` is `batch.is_decode and batch.size <= self.max_graph_bs`
with `max_graph_bs = max(cuda_graph_bs)`. On the P2 Switchyard profile
(`--max-running-requests 16 --elastic-initial-requests 4`) `_adjust_config` installs that
list, and `resize_elastic_capacity` recaptures with `_elastic_graph_batch_sizes(target)` on
every capacity change — so the set is `[1, 2, 3, 4, 8]` at capacity 4 **and at capacity 16**.
Every decode batch of 9 to 16 lanes fell through to `self.model.forward()` eagerly.

The docstring's stated intent was "larger optional bursts keep the sparse power-of-two set";
the tuple simply stopped at 8. The tier's own capacity was never in it.

**This is not a hypothetical.** The passing `13af13d` soak's own server log
(`benchmarks/switchyard_soak/runs/13af13d/server.log`, already in the repo) shows it:

```
INFO  Start capturing CUDA graphs with sizes: [1, 2, 3, 4, 8]
INFO  Elastic capacity 4 -> 16 requests: GDN slots 25 -> 97, MoE slots 1652 -> 976
```

— repeated at every recapture, never once including 16. And its decode batch lines:

| `#running-req` | 1 | 2-4 | 5-8 | **9** | 10 | 11 | 12 | 13 | 14 | 15 | **16** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| decode batches | 48 | 34 | 31 | 13 | 16 | 15 | 26 | 24 | 17 | 54 | **149** |

**314 of 427 decode batches — 73.5 % — ran off the graph**, and 149 of them were the
full-width 16-lane batch the profile exists for. The soak's headline "decode agg tok/s @
`#running-req == 16`" was measured entirely in the eager regime.

## 3. The fix

`python/freetoken/engine/engine.py::_elastic_graph_batch_sizes`. The set is now **dense to
`_DENSE_GRAPH_BS = 16`**, then a 1.33-1.5x ladder, with the tier's own capacity always
appended:

```python
sizes = list(range(1, min(cap, _DENSE_GRAPH_BS) + 1))
sizes += [bs for bs in _SPARSE_GRAPH_BS if bs <= cap]   # 24, 32, 48, 64, 96, 128, 192, 256
if cap not in sizes:
    sizes.append(cap)
```

Two properties this buys, both of which the old `(1, 2, 3, 4, 8)` violated and a sparse
`[…, 8, 16]` would only half-fix:

1. **`max(sizes) == capacity` for every tier.** `can_use_cuda_graph` gates on the largest
   captured size, so anything the ladder does not reach decodes eagerly.
2. **No batch in the range this stack serves ever pads.** §4d measures why that is not a
   nicety: on an offload-MoE model a padded row routes its own experts, and padding 12 up
   to a bs-16 graph costs 5.8 ms/step — more than twice what the graph saves.

Capacity 4 is unchanged (`[1,2,3,4]`). Capacity 16 becomes `[1..16]`; capacity 100 becomes
22 graphs, not 100. `FREETOKEN_ELASTIC_GRAPH_MAX_BS` caps the set, so before/after is two
runs of the same binary (a graph-captured constant gets its env override *before* it is
needed for an A/B — 2026-09-05 lesson); `=8` reproduces the old ceiling.

Tests — `tests/engine/test_elastic_graph_sizes.py`, seven cases, written as **invariants**
rather than literal lists (the pre-existing `== [1,2,3,4,8]` test passed happily through the
bug): `test_every_tier_capacity_has_its_own_graph` (tiers 1..128),
`test_no_batch_in_the_common_range_ever_pads` (every batch's `next(bs >= batch)` is itself),
`test_graph_set_stays_sparse_above_the_dense_range` (every gap above 16 within 1.5x, so the
original memory intent is not regressed), plus the full-width case and two for the env knob.

Benchmark support — `benchmarks/bench_decode_moe.py --pad-lanes N --pad-tokens T` prepends
digit-free filler to the first N concurrent streams, so a 16-lane batch can be measured with
**mixed** context lengths instead of 16 copies of one short prompt.

## 4. Before / after on the real serving path

Both arms: the P2 profile through `bench_decode_moe.py`, one server each, same binary,
same 16 AIME-25 problems, `--decode 256`, `--moe-cache-policy lfu`, `--nvfp4-backend
triton`, `--kv-cache-dtype q8_0`, `--num-tokens 262144`, `--elastic-initial-requests 4`.
The only difference is `FREETOKEN_ELASTIC_GRAPH_MAX_BS`.

**Path proof** — from each arm's own server log, the capture line at every elastic tier:

| arm | capacity 4 | capacity 8 | capacity 16 | MoE slots @16 |
|---|---|---|---|---|
| before | `[1, 2, 3, 4]` | `[1, 2, 3, 4, 8]` | **`[1, 2, 3, 4, 8]`** | 976 |
| after | `[1, 2, 3, 4]` | `[1, 2, 3, 4, 8]` | **`[1, 2, 3, 4, 8, 16]`** | 976 |

Identical MoE slot count at capacity 16, so the expert-cache regime is not a confound.

### 4a. How the step is measured

Each arm pair is two servers, same binary, `--decode 256`, with
`FREETOKEN_ELASTIC_GRAPH_MAX_BS` the only difference. The headline is the engine's own
`gen throughput (token/s)` on its `Decode batch` lines at the batch size under test, median
over the steady-state intervals (the cold first-touch interval is dropped). The client-side
aggregate is reported next to it; where the two disagree the engine row is the one measuring
the step, because `bench_decode_moe`'s window runs from the first token of any stream to the
last token of any stream and so charges lane stagger to the step.

| arm | pool | batch | before (eager) | after (exact graph) | ratio | step before | step after | P(after>before) | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **F1/F2 uniform** | 15-16 | **12** | 143.21 | **153.84** | **1.074x** | 83.8 ms | **78.0 ms** | **0.85** | 11/10 |
| C1/C2 uniform | 16 | 16 | 149.76 | 155.55 | 1.039x | 106.8 ms | 102.9 ms | 0.69 | 10/10 |
| D1/D2 uniform | 12 | 12 | 159.67 | 164.55 | 1.031x | 75.2 ms | 72.9 ms | 0.72 | 10/10 |
| B1/B2 mixed ctx | 16 | 16 | 134.82 | 135.85 | 1.008x | 118.7 ms | 117.8 ms | 0.78 | 3/3 |

**The headline is F1/F2: 1.074x, −5.8 ms of an 83.8 ms step, at 12 lanes in a 16-request
pool** — the shape the soak actually runs (§4d: 421 of its 427 decode batches were taken at
elastic capacity 16, 164 of them at 9-15 lanes). Its client-side numbers agree for once,
because all 12 lanes start together: aggregate 146.70 → 154.04 (1.050x), sum of per-stream
rates 149.41 → 157.96 (1.057x).

At **16** lanes the gain is smaller (1.039x) and closer to the run-to-run spread: a third,
independent 16-lane arm (F3, dense set) medians 150.98 against C1's 149.76 and C2's 155.55,
so ±3 % is about what this measurement resolves at that width. The 12-lane result is the
one carried by its own statistics.

**Graph memory, measured from the capture log at capacity 16:** free VRAM 2.36 → 2.28 GiB
for the full `[1..16]` set — **80 MiB, under 1 s**, and the MoE slot cache is unchanged at
976 slots either way. A second capture into the warm mempool costs 20 MiB. The dense set is
not paid for out of the expert cache.

### 4b. The client-side numbers, and why they disagree with each other

| arm | batch | client aggregate before | after | | per-stream median | sum of streams |
|---|---:|---:|---:|---:|---|---|
| F1/F2 uniform | 12 (pool 16) | 146.70 | 154.04 | 1.050x | 12.387 -> 13.174 | 149.41 -> 157.96 |
| C1/C2 uniform | 16 | 144.56 | 146.69 | 1.015x | 9.200 -> 9.868 | 150.10 -> 154.53 |
| D1/D2 uniform | 12 (pool 12) | 155.38 | 152.81 | **0.983x** | 13.694 -> 13.243 | 162.43 -> 161.02 |
| B1/B2 mixed ctx | 16 | 66.59 | 80.17 | **1.204x** | 7.052 -> 8.557 | 121.43 -> 136.84 |

Four arms, four different answers (+5.0 %, +1.5 %, −1.7 %, +20.4 %) for an effect the engine
measures at +1 to +7 % in all four. **The end-to-end harness resolves the 12-lane effect and
does not resolve the 16-lane one**, and the mixed arm is the worst of the four because its 8
padded lanes prefill first and finish first, so most of its 61 s window is not a 16-lane
batch at all. Recorded here so the 1.204x is never quoted as the result.

### 4c. Why the graph is worth 27 ms at bs=1 and 4-6 ms at bs=12-16

The eager-vs-graph gap on this checkpoint at bs=1 is ~27 ms (33.9 ms eager forward vs a
6.88 ms graphed step). The projection going in was that a fixed per-step host cost carries
over to batch 12-16 unchanged — worth 1.3x. It does not, and the reason is §2: **a wide
decode step is 78-107 ms of GPU work, and the eager launch path's ~30 ms of Python runs
*ahead* of it.** The CPU enqueues layer L+1 while the GPU is still pulling layer L's experts
over PCIe, so most of the launch cost overlaps. The 4-6 ms that does not is what the graph
recovers — 7.4 % of the step at 12 lanes, 3.9 % at 16.

That inverts the usual framing and is the more useful half of this study: **at bs=1 this
model is host-bound and the decode CUDA graph is the single largest win available; at 12-16
lanes it is PCIe-bound and the graph is a 4-7 % cleanup.** The fix is still right — a
profile silently off the graph for 73.5 % of its decode batches is a defect at any price,
and the term comes straight back the moment the movement term shrinks (a larger expert
cache, a card with more VRAM, a shorter-context profile) — but no launch-path change makes
16-way decode fast while 74 % of the step is PCIe at the link roofline.

Weighted over the `13af13d` soak's own decode-batch histogram at capacity 16 (§4d), the fix
moves 313 of 421 batches from eager to an exact graph (164 at 9-15 lanes, 149 at 16) and 19
more from a padded bs-8 graph to an exact one, for an expected **~5 %** on the decode half of
that workload. Prefill is untouched.

### 4d. The trap: a sparse graph set is worse than no graph for any batch that pads

The first version of this fix appended only the tier's capacity, giving `[1,2,3,4,8,16]` at
capacity 16 — the sparse ladder the docstring described. That would have been a **net loss**,
and the reason is a property of this model, not of graphs in general: **`pad_batch` fills the
graph with dummy rows, and a dummy row is not free on an offload-MoE model.** It carries a
hidden state, so it routes its own top-6 experts and adds `top_k` rows to every expert GEMV.

Measured — both arms non-elastic at `--max-running-requests 16` (the pool the soak actually
runs), driving 12 clients, `--cuda-graph-max-bs` picking the arm:

| arm | graph set | what a 12-lane batch does | median tok/s @bs=12 | step |
|---|---|---|---:|---:|
| E1 | `[1,2,4,8]` | runs **eager** at 12 | 146.07 | **82.2 ms** |
| E2 | `[1,2,4,8,16]` | replays the **bs-16** graph, 4 dummy rows | 136.31 | **88.0 ms** |

**0.933x — padding up to the next captured size costs 5.8 ms/step, more than twice what the
graph saves.** n = 10/10 intervals, P(padded > eager) = 0.16; the client aggregate agrees
(148.75 → 139.16, 0.936x) and here it agrees because both arms start all 12 lanes together.

And the exposure is not marginal. Correlating the `13af13d` soak's `Elastic capacity` lines
against its decode batches: **421 of its 427 decode batches were taken at capacity 16**, with
batch sizes spread right across the band — 164 of them in 9-15. Elastic capacity grows on
demand and does not come back down, so "capacity 16 running a narrower batch" is the soak's
normal state, not a transient. A `[1,2,3,4,8,16]` set would have helped 149 batches by ~3 %
and hurt 164 by ~7 %.

**So the set is dense to 16** (`_DENSE_GRAPH_BS`), then a 1.33-1.5x ladder
(24, 32, 48, 64, 96, 128, 192, 256) with the capacity always appended. No batch in the range
this stack serves ever pads or falls off the graph. Cost, measured from the capture log:
~5 MiB and ~50 ms per graph, so ~80 MiB and ~0.8 s per elastic resize for a dense set to 16
— against a 976-slot expert cache that is 5.5 GiB, i.e. ~14 slots.

| batch at capacity 16 | before the fix | sparse `[…,8,16]` | dense `[1..16]` |
|---|---|---|---|
| 1-4 | exact graph | exact graph | exact graph |
| 5-7 | pad to 8 | pad to 8 | **exact graph** |
| 8 | exact graph | exact graph | exact graph |
| 9-15 | **eager** | **pad to 16 (−6.7 %)** | **exact graph** |
| 16 | **eager** | exact graph | exact graph |

### 4e. Single stream and the 131K needle — untouched, by construction

`_elastic_graph_batch_sizes` is only consulted when `--elastic-initial-requests` is set,
and both of these arms run without it (`--concurrency 1` forces `max_running_req = 1`,
which the elastic flag may not equal or exceed). Their capture sets are `[1]` and
`[1,2,3,4]`, identical in both arms of the study. Measured after the fix:

| arm | result |
|---|---|
| bs=1 decode, short prompt | **135.84 tok/s** (7.362 ms/token, event p50 7.255 / p99 9.317 ms) |
| 131K synthetic needle, depth 0.50 | prompt **130,016 tokens**, prefill **6,059.8 tok/s** end-to-end (3,157 instant / 5,648 average engine), decode **132.32 tok/s**, **needle recalled: `5663623`, `expected_found: true`, `accepted: true`** |

The 132.3 tok/s at 130K is below the 145.3 recorded for `acc91e9` because this profile runs
`--memory-ratio 0.85` and `--max-context 131072` (the P2 shape) rather than the launch
study's; the expert cache is correspondingly smaller. It is not an A/B against the fix —
there is nothing for the fix to change here — it is the absolute long-context datum for
this serving profile, and the needle is exact.

## 5. What this does NOT fix, and why the 16-lane regime stays hard

The graph fix removes a fixed ~25 ms host cost from every wide decode step. It does not
touch the 78 ms of PCIe expert movement underneath it, which is why the after number is
1.2x and not 2x. The arithmetic of that term, spelled out so the next attempt does not
have to rediscover it:

* 23 MoE layers × 61.6 distinct routed experts per layer at 16 lanes = **1,417 expert-layer
  slots wanted per step**, against **976 slots** the pool leaves at elastic capacity 16.
* Slots are 5.612 MB each, so buying the missing 441 slots costs **2.4 GiB of VRAM** on a
  16.3 GiB card whose budget already reads: 2.3 GiB weights + 4.45 GiB Mamba-2 state
  (97 slots × 47 MiB at capacity 16) + ~0.9 GiB q8_0 KV + 5.5 GiB expert cache.
* The only pool big enough to fund it is the recurrent-state pool, 64 of whose 97 slots are
  the radix snapshot cache that buys the 82-89 % prefix reuse the soak measures. That is a
  scheduler trade, not a decode-path change, and it should be measured as one.
* Prefetching the next step's experts is **not** available: consecutive tokens of one
  sequence route almost disjointly (a 2-token step touches 11.61 of 12 possible experts,
  cache study §4), so there is nothing to predict. D(m) ≈ 6.2·m^0.75 is the whole reason
  16-way aggregate beats single-stream at all, and it is already being collected.
* Overlapping the fetch with compute buys at most the compute: 12.9 ms of MoE GEMV under
  78 ms of PCIe, and the two are serially dependent within a layer, with only a 38 us
  Mamba-2 layer between consecutive MoE layers to hide behind.
* `--moe-backend hybrid` is the one untested lever with real headroom — the profile's
  *overlapped* pair measures CPU MoE 45.2 + PCIe gather 46.2 = **91.4 GB/s vs 52.9 alone**,
  and `load_hybrid_fetch_fraction` already computes the bandwidth-matched split from it.
  But `load_backend_recommendation` gates hybrid on the **standalone** ratio
  (CPU 66.9 / PCIe 52.9 = 1.26 < the 2.0 threshold), which is the criterion for *replacing*
  PCIe, not for *splitting* across both. Hybrid was only ever measured at bs=1 (3.6x slower,
  where 0.72 misses/layer cannot pay for the per-layer handshake); at 16 lanes it is
  31.45 misses/layer. **Untested, and the threshold looks like it is asking the wrong
  question.** See §7 ticket 1.

## 6. Artifacts

`benchmarks/decode16/` (tracked): `phaseA.sh` (the weightless sweep), `phaseB.sh` (mixed 16),
`phaseC.sh` (uniform 16 + bs=1 + needle), `phaseD.sh` (12 lanes, exact graph), `phaseE.sh`
(12 lanes padded to a bs-16 graph — the risk check), `phaseF.sh` (12 lanes at a 16-request
pool, the headline A/B). `runs/` is gitignored; this study is
`runs/phase{A,B,C,D,E,F}/` with each arm's `*.stdout`, `*.server.log` (kept out of the
bench's tempdir precisely so the capture-set and elastic-capacity lines survive), and `*.json`.

The Phase A sweep loads no checkpoint and finishes in 5 minutes; run it first on any new
geometry before booking a server.

## 7. Still open after this run

1. **The NON-elastic capture set has the same padding defect and was not changed.**
   `graph.py::_determine_cuda_graph_bs` returns `[1, 2, 4] + range(8, max_bs + 1, 8)`, so a
   server without `--elastic-initial-requests` pads 3->4, 5-8->8 and 9-16->16 — and §4d
   measures the 12->16 case at **−6.7 %** on this checkpoint. The evidence is model-specific
   (a padded row is only expensive because it routes experts; on a dense model it is nearly
   free), and the blast radius is every model and profile, so it is filed rather than
   changed. The experiment is already written: Phase E, two servers, `--cuda-graph-max-bs 8`
   vs `16` at `--concurrency 12`. Do it for one dense model before touching the default.
2. **`--moe-backend hybrid` at 16 lanes has never been measured, and the auto-threshold
   asks the wrong question.** `bench_profile.load_backend_recommendation` recommends hybrid
   only when standalone CPU MoE BW > 2.0 × standalone PCIe gather BW (1.26 on this host, so
   never). Hybrid does not replace the gather, it splits the step's misses across both
   engines, and `load_hybrid_fetch_fraction` already prefers the *overlapped* pair for the
   split. The criterion for a split should be `(cpu_ov + pcie_ov) / pcie_alone > 1 + margin`
   = 91.4/52.9 = 1.73 here. Run `bench_decode_moe --backend offload,hybrid --concurrency 16`
   before changing the threshold — the per-layer CPU handshake (~0.9 ms/layer inferred from
   the bs=1 arm) may still eat the win, and the host is also running 16 lanes of scheduler.
3. **The 976-vs-1,417 slot deficit.** Quantify what the 64-slot Mamba-2 snapshot cache is
   worth (prefix reuse, and therefore prefill) against the ~2.4 GiB of expert slots it
   holds. This is a scheduler experiment, not a decode one.
4. **`decode_launch_config` ignores batch and is right to at long contexts, but loses 1.27x
   at 4K × 16 lanes.** Worth 0.07 ms/step here; revisit only if a short-context 16-lane
   profile becomes a target (the split count is baked into the graph at capture, so a
   batch-aware rule means a per-bs scratch allocation).
5. **`FREETOKEN_MAMBA2_DECODE=flashinfer` is 1.44x the triton default at bs=16**
   (26.75 vs 38.43 us/layer, 0.27 ms/step). Not taken: triton is the backend the
   graph-capture and state-rollback paths are tested against.
6. **`--moe-collect-stats` did not produce a stats delta in the mixed arm** — the scheduler
   only dumps at an idle boundary and `wait_for_moe_stats` timed out at 60 s on the warm
   snapshot. The hit-rate numbers in §2 are therefore the 2026-09-04 cache study's, not
   this run's. A bench that needs the counters should force an idle boundary rather than
   race one.
7. The `#token` field on a decode batch line counts **unique** KV pages, so lanes padded
   with identical filler report far less than the sum of their context lengths (24,369 for
   8 × 16.5K on the second pass). Per-request `device_len` — what attention costs — is
   unaffected. Do not read `#token` as "the contexts did not materialize".
