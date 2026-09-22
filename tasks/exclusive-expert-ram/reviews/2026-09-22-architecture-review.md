# Independent architecture review — 2026-09-22, at HEAD `826bc04`

Requested by the owner: *"see if we are constructing a good software that will
allow us expand to different models keeping our smart kv scheduler and smart ram
saver. compare with upstream freetoken, give an honest opinion on what we build
so far. it seems we are dumping code and glueing things over... do we really need
a refactoring, and would refactoring in Rust bring benefits?"*

Reviewer: Fable (campaign-specialist), read-only, no edits, no GPU. Everything it
marks "verified" it read in this worktree; "inferred" is its judgement. Verbatim
report follows; nothing in it has been acted on.

---

## Verdict

The owner is right that glue is accreting, but wrong about where. The two new modules that carry the design (`MirrorExpertPool`, `VMMTensor` + `vmm_tensor.cpp`) are clean, bounded and model-agnostic; the accretion is inside three upstream files that tripled in size (`offload_cache.py` 1042→3293, `engine.py` 1505→3289, `scheduler.py` 905→3154) and in helpers copied three times because task-ownership boundaries were treated as module boundaries. The smells listed are not six isolated bugs: four of them share one structural cause (residency state has no single owner, so schemas and boundary contracts live in prose across four files). A bounded extraction, not a rewrite, fixes that; Rust would not.

Everything below marked "verified" I read in the worktree at HEAD `826bc04`; "inferred" is judgement.

---

## 1. Generalisation across models

**Genuinely model-agnostic (verified):**
- Expert location goes through `Nvfp4ExpertSourceSpec` (`models/nvfp4_banks.py:20-30`), which is upstream's; the fork added `gated` and `hidden_size_attr` to it (upstream at merge-base `bd372b6` had only `key_pattern/proj_to_role/layer_to_bank/desc`). Both models export it: `models/nemotron_h/weight.py:36-42` (ungated, `moe_layer_ids` index), `models/qwen3_5_moe/weight.py:44-49` (gated, identity). `MirrorExpertPool._scan_checkpoint` (`mirror_pool.py:334-437`) has no key format of its own and refuses a model with no spec. Good boundary.
- All residency maps are flat `layer*E + expert` ids; kernel loops are bounded by `E` or `top_k*batch` (`mirror_kernels.py` `_resolve_swaps_kernel`, `_writeback_buffer_kernel`), `BLOCK=next_power_of_2`. E=256 changes nothing.
- `ModelConfig.expert_gated` / `expert_hidden_size` / `num_moe_layers` (`models/config.py:336-356`) are the right place for geometry, and engine checks spec vs config agreement (`engine.py:812-817`).
- VMM arena is dtype-tabled (`vmm.py:54-68`), no geometry.

**Silent or hand-edit assumptions (verified):**
- `nvfp4_bank_shapes` (`mirror_pool.py:63-84`) and `_row_layout` (`mirror_pool.py:311-332`) re-state the NVFP4 row byte layout that `nvfp4_banks._alloc_nvfp4_host_banks` and the loader already define; the docstring admits it "mirrors ... exactly". Upstream has since added `kind_map`/`global_reciprocal` to the spec for compressed-tensors NVFP4 (`6eca2d7`); the mirror's `_KIND_DTYPE`/`_KIND_BANK_INDEX` (`mirror_pool.py:59-60`) know nothing of that. **A third model in llm-compressor NVFP4 format (e.g. GLM-5.3-Flash-NVFP4) is refused by `_scan_checkpoint` with "unknown tensor kind"** — loud, but it is the first concrete "hand-edit needed" site.
- `--pin-prefix-*` (fork commit `c819a81`) is gated on `is_hybrid = type == "hybrid_radix"` (`scheduler/cache.py:93`), and `_resolve_cache_type` only picks `hybrid_radix` when `has_linear_attention` (`engine.py:2520-2526`), which is true only for `LinearGatedDeltaGroupConfig` (`models/config.py:372-376`) — Qwen3.5/Ornith GDN. Nemotron-H's Mamba layers are a different group, so the fork's own flagship model gets `radix` and the flags are inert with no refusal. This is a fork feature that works on one of the fork's two models.
  **Correction (S4, 2026-09-22, kept in place rather than deleted — this file is the evidence trail):** the premise above is false. Nemotron-H's Mamba layers ARE `LinearGatedDeltaGroupConfig` (`models/nemotron_h/config.py:204`), so `has_linear_attention` is true and `_resolve_cache_type` resolves Nemotron to `hybrid_radix`, not `radix` — confirmed by the `Resolved config: ... cache_type='hybrid_radix'` line in every 1M journal (e.g. `tasks/exclusive-expert-ram/results/nemotron-reserve-2e-1m-journal.txt:10`; the `ServerArgs(...cache_type='radix'...)` line the same journal prints on line 3 is the *requested*, pre-resolution value, not the resolved one). `--pin-prefix-*` is honoured on Nemotron. What actually makes it do nothing for a single agent re-reading its own haystack is the cross-session-only pin policy at `scheduler/cache.py:610-663`, not this gate. See `tasks/exclusive-expert-ram/reviews/2026-09-22-refactor-plan-final.md` section 3.3 and `STATUS.md`'s corrected Lever-1 paragraph.
- `_mirror_final_gpu_slots` (`engine.py:1553-1620`) re-derives the KV/arena budget "from config alone" because the pool must exist before the KV pool, with a `- 4 * step` fudge and a comment saying the previous estimate killed a 600K request. That arithmetic is a second copy of `_plan_growable_kv`; any model whose KV cost model differs (Mamba state, GDN state, SWA) has to be right in two places.
- Mirror is NVFP4-safetensors + triton backend + single rank + one shard per expert only (`engine.py:795-804`, `offload_cache.py:1648-1651`, `mirror_pool.py:412-413`). All refusals are loud. The GGUF Ornith path (`moe/fused_gguf.py`, vendored `gguf_mmq/`) cannot use the mirror at all.
- `benchmarks/bench_moe_prefill_gemm.py:60` hardcodes `H, I, E, TOP_K = 2688, 1856, 128, 6`. Verified, trivial, benchmark-only.

**Is the boundary in the right place?** Yes for the spec. Wrong for the row layout: byte layout of one expert row is a property of the quant format, and it currently lives in `nvfp4_banks.py` (loader), `mirror_pool.py` (pool) and implicitly in `expert_bytes_per_slot` (engine sizing). One function, three consumers, is the fix.

## 2. Upstream comparison

Remote `origin` = FlashML-org/FreeToken, last fetched 2026-09-17. Merge-base `bd372b6` (2026-08-24). Fork is **328 ahead / 19 behind**, 782 files, +135,389/−1,290. 326 of 328 commits are authored `probe` (agent) over 30 days.

**Upstream already provided (verified at `bd372b6`):** `OffloadMoeCache` with LRU/LFU slot maps, prefill double buffer and `_invalidate_prefill_buffer`, hit-D2D split, hybrid CPU routing, host bank residency classes (`host_banks.py` PINNED/LOCKED/PAGEABLE), `Nvfp4ExpertSourceSpec`, `pinned_tensor.cpp`, `fast_index_copy`, `hybrid_radix` for GDN models, `cache_budget.py`. **No** VMM, growable KV, arena, mirror, session spill/single lane, prefix pinning, Nemotron-H or Mamba kernels.

**Working with or around upstream?** Mostly *with*: the mirror reuses `fast_index_copy_multi_jit` for all three copy directions, `host_register`, the spec, and the existing slot maps. Around, in three places: (a) `_ensure_experts_sized_kernel_v2`, `_materialize_layer_sized_kernel_v2`, `_ensure_experts_hybrid_kernel_v2` are near-verbatim twins of upstream kernels differing only in `tl.load`-ed bounds (`offload_kernels.py:618-660` docstring: "Identical logic, except..."); (b) `MirrorExpertPool._read_row` is a second safetensors reader beside `nvfp4_banks.load_nvfp4_expert_source_banks`; (c) `_arena_chunk_boundaries` exists three times (`cache_budget.py:84`, `offload_cache.py:586`, `engine.py:134`), each with a comment saying it is duplicated because of an "ownership boundary" / "this task must not edit".

**Upstreamable as-is:** `VMMTensor` (`kernel/vmm.py` + `csrc/vmm_tensor.cpp`, self-contained, RAII, tested), the `gated`/`hidden_size_attr` spec fields, `expert_gated`/`expert_hidden_size` on `ModelConfig`. **Not upstreamable without work:** the mirror and arena (env-var gated `FREETOKEN_EXPERT_ARENA`, `FREETOKEN_MIRROR_*`, single-lane assumptions in `_mirror_writeback_buffer`'s docstring).

**Rebase pain:** upstream's 19 commits touch the fork's hot files lightly (253 lines across `engine.py`, `offload_cache.py`, `scheduler.py`, `nvfp4_banks.py`, `config.py`, `args.py`, `fused_nvfp4.py`); 41 files overlap. Mechanically feasible today; the `nvfp4_banks.py` `kind_map` change conflicts semantically with the mirror's duplicate reader. Every month of upstream activity on those three 3K-line files makes it worse. No upstream branch is building the same thing (`feat/split-residency` has zero commits over main; `feat/ple-disk` is an unrelated disk table).

## 3. Verdict on the code and the listed smells

| Smell | Verified? | Isolated or structural |
|---|---|---|
| stats `[6]`→`[7]`, two unpack sites | Yes: positional unpacks at `offload_cache.py:1841-1842` and `1980-1981`; layout documented only in kernel comments `mirror_kernels.py` (both kernels) | **Structural**: a shared counter vector with no owner and no named schema, written by two kernels in one file and read in another |
| `free_eviction_rate` > 1.0 | Yes, fixed in `d9b0070` (`offload_cache.py:1847-1855`, `mirror_kernels.py:439-443`) | Same cause as above; the metric was defined at a third site with no knowledge of who bumps which slot |
| forced retention across 4 files | Yes: `_resolve_swaps_kernel` `buffer_slots` branch, `_mirror_prefill_base` (`offload_cache.py:1701`), `_writeback_buffer_kernel`, snapshot patching in `_mirror_writeback_buffer` (`2321-2400`), and `_mirror_final_gpu_slots` still pricing `prefill_buffer_slots` (`engine.py:1608-1617`). `plan.md` "Slot regions" still says `[0,2E)` is "prefill's alone" — stale | **Structural**: the residency state machine (device maps + free stack + host numpy snapshots + sizing) has no single owner |
| `--pin-prefix-*` inert | **No — see the S4 correction note under Q1 above.** Nemotron resolves to `hybrid_radix`; pins ARE honoured. The real miss is the cross-session-only pin policy at `scheduler/cache.py:610-663`. | Not a gate bug; a policy scope, tracked as plan step S13 |
| shrink sync / grow no sync | Yes, fixed `67ec22a` (`offload_cache.py:1311-1313`) | Isolated. The structural part is that "must run at a no-forward-in-flight boundary" is a docstring contract (`set_usable_slots` `1226-1232`, `_grow_runtime_kv_arena` `engine.py:1775-1780`), never asserted |
| benchmark hardcode | Yes | Isolated, low value |

**Further evidence of accretion I found (verified):**
- `_mirror_restore_coverage` (`offload_cache.py:1793-1835`) has **no callers** anywhere in `python/`, `tests/`, `tasks/`. Dead.
- `mirror_warm_start` is called from four sites (`engine.py:1097`, `offload_cache.py:2781`, `:2850`, `scheduler.py:3093`). Two of them implement the same "restore at prefill→decode" with the rationale "a prefill sweep empties the mirror" — which STATUS.md says stopped being true at `f007a7c`. The trigger `_mirror_needs_coverage = True` is set only in `materialize_layer` (`:2828`), and under the mirror prefill overlap is mandatory (`engine.py:820-835`), so `prefetch_prefill_layer` takes `_prefetch_split_mirror` (`:2461-2462`) and never materializes. Inferred: the batch-boundary restore is unreachable in the served configuration and its comment is stale.
- Four coverage-repair paths coexist (`mirror_warm_start`, `_mirror_refill_uncovered`, `_mirror_stage_layer`, `_mirror_restore_coverage`), each with its own host/device map reconciliation.
- Env-var knobs 52 → 108 across `python/freetoken`. Fifteen `getattr(self, "_mirror", None)` branches in `OffloadMoeCache`.
- 41 comments in library code reference measurement arms, dates, or "lever N"; `engine.py:122-124` literally says a helper is duplicated because "this task must not edit" another file.
- `tests/moe/test_mirror_device.py` shells out to `tasks/exclusive-expert-ram/swap_smoke.py` and `graph_race_repro.py` (tracked, but "dev-only assets", skipped if absent). The regression tests that "found every serious bug" live outside `tests/`.

**What is good and should not be mistaken for glue:** `mirror_pool.py` (590 lines, one class, one invariant, no engine imports), `mirror_kernels.py` (three hazards documented and encoded), `vmm_tensor.cpp` (RAII, mutex, overlap validation, exact-match uncommit), and the test count (7 mirror test files, 9 growable/arena test files). The correctness story (14/14 byte-identical battery, 0 faults at 1M) is real.

## 4. Refactor: yes, bounded, in this order

Do the near-zero-risk items first; they need no GPU.

1. **Named counter schema** (S, half a day). One `IntEnum`/constexpr table for the 7 stats slots, imported by both kernels and both unpack sites. Kills the class behind smells 1 and 2.
2. **Delete dead and stale restore paths** (S, half a day). Remove `_mirror_restore_coverage`; make `materialize_layer` raise under the mirror (it is not a supported path) and delete the `_mirror_needs_coverage` machinery in `ensure_experts` and `scheduler._forward`, or keep exactly one with an assertion that it never fires. Fix `plan.md` slot regions.
3. **One `_arena_chunk_boundaries`** (S, one hour). Import from `cache_budget`; bodies are identical.
4. **Pin-prefix refusal** (S). Raise at startup when the flags are set and the resolved cache type is not `hybrid_radix`.
5. **Single row-layout source** (M, 2-3 days). Move `nvfp4_bank_shapes` and `_row_layout` beside `_alloc_nvfp4_host_banks` in `nvfp4_banks.py`, make them `kind_map`-aware, have the pool import them. This is the change that makes model #3 a one-file edit. Guard: `tests/moe/_mirror_checkpoint.py` synthetic checkpoint plus the existing byte-identical battery.
6. **Extract `MirrorResidency` from `OffloadMoeCache`** (M-L, 1-2 weeks, needs the GPU for device tests). Pure *move*: the `_mirror`/`_mirror_writeback` dicts, snapshots, `attach/warm_start/refill_uncovered/writeback_buffer/prefetch_split_mirror/fault_check`, and the kernel launches go into one object; `OffloadMoeCache` keeps slot maps and calls `residency.before_shrink(n, current)`, `residency.before_buffer_fill(buffer_id)`. Kernel bodies stay byte-identical. This is the one step that can reintroduce the stale-publish class; mitigate by moving code without editing it, then running `swap_smoke.py`, `graph_race_repro.py` and the recall battery before and after and diffing counters.
7. **Make `_mirror_final_gpu_slots` call the real planner** (M). Split `_plan_growable_kv` into a pure function of config so the pool sizing and the runtime handover share arithmetic; remove `4 * step`.

**Do not:** rewrite or "simplify" the three Triton kernels or the free-stack/`publish_freed_rows` protocol; touch the scheduler's session spill; change the coverage invariant or reserve arithmetic; attempt non-NVFP4 mirror support; rebase onto upstream *during* steps 5-6 (do it before step 5 or after step 7); collapse the `_v2` kernel twins (the gating is the whole point of graph-safe resize; leave them).

## 5. Would Rust help?

No, and not for a narrow component either. Where the risk actually lives:
- The hot paths are Triton kernels and `fast_index_copy` CUDA. Rust cannot express those.
- Every serious bug in this branch's history was a stream/ordering or state-ownership error: hazards 1-3 in `mirror_kernels.py`, the stale free-row publish, the missing `synchronize` before `commit_ranges`, the numpy snapshot going stale mid-chunk. Rust's ownership model does not model CUDA streams, graph replay, or "this tensor is read by an in-flight kernel". `TORCH_CHECK` plus device-side counters already give the runtime checks Rust would.
- `vmm_tensor.cpp` is 329 lines, already RAII (`VMMState` destructor unmaps and frees; `from_blob` keepalive closure), mutexed, and validates overlap and exact-match uncommit. The 1M crash was a Python-side ordering bug (`67ec22a`), not a C++ lifetime bug. A PyO3/cust rewrite would add a toolchain and a torch-tensor interop seam for zero additional guarantees.
- Maintenance: one agent author and one owner; a second language halves review capacity.

What *would* buy safety in the places Rust cannot: a host-side reference model of the swap protocol with property tests (random miss sequences against a Python dict model, asserting coverage after every publish), and turning the two prose boundary contracts (`set_usable_slots`, `_grow_runtime_kv_arena`) into a runtime assertion on an "in-flight forward" flag.

## Uncertainties

- I ran nothing (read-only constraint); reachability of the batch-boundary restore is inferred from call paths, not execution.
- Ornith geometry (40x256, gated) is from `plan.md`; I did not read the checkpoint.
- Upstream view is from a 2026-09-17 fetch.
- I did not audit `session_spill.py` or the scheduler's growable-KV shrink-on-completion behaviour beyond what STATUS.md reports (`scheduler.py:2271`, `:2344`, `:2473-2489`); that is a separate review.

Key paths: `/home/lucas/ai/FreeToken-wt/exclusive-expert-ram/python/freetoken/moe/{mirror_pool.py,mirror_kernels.py,offload_cache.py,offload_kernels.py}`, `.../engine/{engine.py,cache_budget.py}`, `.../kernel/{vmm.py,csrc/vmm_tensor.cpp}`, `.../models/{nvfp4_banks.py,config.py,nemotron_h/weight.py,qwen3_5_moe/weight.py}`, `.../scheduler/{scheduler.py,cache.py}`, `tasks/exclusive-expert-ram/{STATUS.md,plan.md}`.