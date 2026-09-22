# Refactor plan — 2026-09-22, written at HEAD `bc6f815`

Requested by the owner after reading `2026-09-22-architecture-review.md`. His
four goals, verbatim:

* "bring from upstream anything that can help, actually, we should be able to
  sync with upstream at any time"
* "i want our smart, growing kv, that with prefil cache, and spilling to disk
  (yes, our smart kv manager)"
* "I want the smart ram saver to be a flag option, or something like that"
* "I want it to be easy to add new models, without fucking the code"

Author: Fable (campaign-specialist), read-only, no edits, no GPU, resumed with
its review context intact. Verbatim plan follows.

**Not yet accepted.** It ends with eight questions for the owner, several of
which change the plan's shape. Nothing here has been started.

---

I have what I need. No further reads; the plan follows.

---

# Refactor plan for `exp/exclusive-expert-ram` (read at HEAD `bc6f815`)

## Framing the four goals against what I verified

**Goal 1 (sync with upstream at any time).** A real dry-run merge (`git merge-tree --write-tree origin/main HEAD`, touches nothing) conflicts in **24 files, ~40 hunks**: `engine.py` 10, `args.py` 4, `attention/linear.py` 4, `cpu_moe_ext.cpp` 5, `config.py` 2, `kvcache/{__init__,base}.py` 2+2, `register.py` 2, `fused_nvfp4.py` 2, and one each in `offload_cache.py` (the `_BANK_BYTES_PER_EXPERT` table, line ~130), `nvfp4_banks.py` (upstream's new `kind_map`/`global_reciprocal` spec fields), `fused.py`, `cpu_executor.py`, `sampling.py`, `aot_models.py`, `qwen3_5_moe/moe.py`. The engine hunks are all small upstream changes (gpu_select, attention-backend list, `_pin_budget_bytes` signature, `forward_host_ctx`, activation list, WSL pin cap, BSA/QSA dtype) landing in regions the fork rewrote. That is 1-2 days of merging today, and it is the cost of "sync at any time" right now; the plan's job is to keep it at that size.

Two things follow that the owner should hear plainly:
- **Merge, not rebase.** 326 of 328 fork commits are agent-authored phase commits; replaying them through recurring conflict hunks is days of pain with no benefit. A merge resolves each hunk once. Upstream policy does not care about fork history.
- **Upstreaming is a distraction for goal 1.** Upstream's `AGENTS.md`/`CONTRIBUTING.md` (origin/main) refuse agent PRs: "a human must understand what the agent did, have run the code on real hardware, and be able to explain the change to a reviewer without AI help." So every PR is owner time. And the volume it would remove is small: `VMMTensor` is ~490 lines of a 135K-line delta, the spec fields ~10 lines. What makes syncs cheap is the *shape* of the fork's changes (new modules plus small hooks in upstream files, no import-time env constants, no tests that parse upstream class bodies), not shrinking the diff by upstreaming. The one exception worth a single owner-owned PR: the `gated`/`hidden_size_attr` spec fields and the ungated `_BANK_BYTES_PER_EXPERT["nvfp4"]` fix, because those sit exactly where upstream's `kind_map` change conflicts and will conflict again.

**Goal 2 (smart KV as one kept thing)** is growable KV (`engine.py:1515-2200`, ~700 lines across `_plan_growable_kv`, `_grow/_shrink_runtime_kv_arena`, `grow/shrink_runtime_kv`, rollback), the arena (`offload_cache.py:1212-1338`), radix prefix cache + pinning (`scheduler/cache.py`), and session spill (`scheduler/session_spill.py`, 1366 lines, wired at `scheduler.py:311-325, 875, 1354-1420`). Today it has two modes: the arena path (production, `FREETOKEN_EXPERT_ARENA=1` in `scripts/serve-default.sh`) and a legacy rebuild-and-recapture path (`engine.py:2050-2140`) that is still live for formats the arena refuses (`offload_cache.py:260-266` marlin/b12x, `:721` GGUF size classes). The pool hangs off this in three places only: `_arena_shrink` calls `_mirror_refill_uncovered` (`:1272-1274`), `_grow_runtime_kv_arena` reads `_mirror_pool_ref.min_gpu_slots` (`engine.py:1800-1811`), and `scheduler._forward` runs the mirror restore/fault check (`scheduler.py:3080-3101`). Those three are the seam.

**Goal 3 (RAM saver as "a flag option").** It *is* a flag today (`--moe-mirror-host-rows` / `FREETOKEN_MIRROR_EXPERT_RAM=1`, `engine.py:785-789`), and production does not set it. So the owner already has the cheap thing, and it is not what he needs: with the flag off, the pool's state still threads through `OffloadMoeCache` (15 `getattr(self, "_mirror", None)` branches, the prefill path `:2299-2585`, `_init_prefill_overlap_buffers` `:2189-2299`), `engine.py` (9 `if mirror` branches inside `_init_offload_moe_cache`, plus `_mirror_final_gpu_slots`), and the scheduler. Every upstream change to those files has to be reasoned about with the pool in mind, and every model added has to be checked against both paths. **The design call: a residency seam, selected by a config value, with two implementations.** `ExpertResidency` protocol; `WholeModelResidency` (today's default: banks pinned, misses served from bank sources, no-op hooks); `MirrorResidency` (owns the `_mirror`/`_mirror_writeback` dicts, the numpy snapshots, the four kernel launches, warm start, refill, buffer writeback, prefill assembly, fault check). `OffloadMoeCache` keeps the slot maps and calls `self.residency.before_shrink(n, current)`, `.before_buffer_fill(buffer_id)`, `.prefetch_layer(layer_id, buffer_id)`, `.before_ensure(layer_id)`, `.fault_check()`. Cost: 1-2 weeks and one GPU window, and it is the single riskiest step (the stale-publish class lives here). Benefit: it serves goals 1, 3 and 4 at once, and it is what makes "flag off" mean "code absent from the path". Build-time absence (separate package/plugin) is not worth it: the pool needs `OffloadMoeCache`'s maps and kernels' `victim_ids`/`prior_ids`; a plugin boundary there would be fiction.

**Goal 4 (easy to add a model).** Verified today: the NVFP4 row byte layout is stated in **four** places: `nvfp4_banks._alloc_nvfp4_host_banks` (`:79-94`, gated-aware), `mirror_pool.nvfp4_bank_shapes` + `_row_layout` + `_KIND_DTYPE` (`:63-84, :311-332, :59-60`, gated-aware, `kind_map`-unaware), `offload_cache._BANK_BYTES_PER_EXPERT["nvfp4"]` (`:126`, **assumes gated `2*I` regardless of `expert_gated`**, consumed by `expert_banks.bank_bytes_estimate` `:453-467` for pin-budget sizing; docstring calls it "a slight over-estimate", for Nemotron it is ~1.6x; whether that changes any decision on this host I did not verify), and `cache_budget.expert_bytes_per_slot` (`:17-31`, derives from tensors, fine). A compressed-tensors NVFP4 checkpoint (upstream `6eca2d7`, GLM-5.3-Flash-NVFP4) is refused by the mirror with "unknown tensor kind" (`mirror_pool.py:365-367`). And `--pin-prefix-*` is inert for any non-GDN model (`scheduler/cache.py:93`, `engine.py:2520-2526`, `models/config.py:372-376`) while `serve-default.sh` passes `--pin-prefix-min-tokens 1024` to production. **Correction (S4, 2026-09-22, kept in place — this draft is superseded, not deleted):** false. Nemotron-H's Mamba layers are `LinearGatedDeltaGroupConfig` (`models/nemotron_h/config.py:204`) too, so `has_linear_attention` is true and Nemotron resolves to `hybrid_radix`, not `radix` (see the `Resolved config: ... cache_type='hybrid_radix'` line in any 1M journal, e.g. `tasks/exclusive-expert-ram/results/nemotron-reserve-2e-1m-journal.txt:10`). `--pin-prefix-*` is honoured on Nemotron; what makes it a no-op for one agent re-reading its own haystack is the cross-session-only pin policy at `scheduler/cache.py:610-663`, tracked separately as plan step S13. "Easy to add a model" becomes checkable as: one row-layout function, one conformance check that reads a checkpoint's headers and either accepts or refuses with the exact reason, run against both checkpoints on disk.

---

## Steps

Size key: S ≤ 1 day, M 2-5 days, L 1-2 weeks. "Acceptance bundle" is defined once at the end. All test commands use the server venv, per the handover: `PYTHONPATH=$PWD/python /home/lucas/ai/FreeToken/.venv/bin/python -m pytest -q ...`, model unloaded.

### Phase A — no GPU, three parallel lanes (1 day)

**S1. Named counter schema** (goal 3; lane 1)
- What: one `MirrorStat` IntEnum (or a `tl.constexpr` offset table) in a new `moe/mirror_stats.py`; `_resolve_swaps_kernel` and `_writeback_buffer_kernel` take the offsets as constexpr; `mirror_stats()` and `mirror_fault_check()` unpack by name.
- Why: positional unpacks at `offload_cache.py:1841-1842` and `1980-1981`; layout lives only in kernel comments (`mirror_kernels.py:117-119, 365-369`); this is the cause of both the `[6]→[7]` miss and the >1.0 rate (`d9b0070`).
- Breaks: nothing at runtime if offsets are unchanged; the counters' meaning is preserved by construction.
- Verify: `grep -n 'stats"\].tolist()\|stats_host"\].tolist()' python/freetoken/moe/offload_cache.py` returns 0 positional unpacks; new CPU test feeds a fake 7-vector to `mirror_stats()` and asserts `free_eviction_rate == free_evictions/swaps` and `buffer_free_evictions` reported separately; `pytest tests/moe -k mirror` green (CPU parts).

**S2. Delete dead and stale coverage-restore paths** (goal 3; lane 1)
- What: remove `_mirror_restore_coverage` (`offload_cache.py:1793-1835`, no callers in `python/`, `tests/`, `tasks/`); make `materialize_layer` raise under the mirror (it is not the prefill path: overlap is mandatory, `engine.py:820-835`, and `prefetch_prefill_layer` dispatches to `_prefetch_split_mirror` `:2461-2462`); delete `_mirror_needs_coverage` and the two duplicate restore sites (`offload_cache.py:2770-2782`, `scheduler.py:3080-3094`) leaving `mirror_fault_check` at the batch boundary; fix `plan.md` "Slot regions" (stale: says `[0,2E)` is prefill's alone).
- Breaks: if my reachability inference is wrong and `materialize_layer` is hit under the mirror, the new raise fires on the first prefill. That is loud and the device test catches it.
- Verify: `grep -rn "_mirror_restore_coverage\|_mirror_needs_coverage" python/ tests/ tasks/` → 0; `pytest tests/scheduler tests/moe` green. **Not "done" until GPU checkpoint 1 runs `tests/moe/test_mirror_device.py` (`graph_race_repro.py` is the test for exactly this boundary).**

**S3. One `_arena_chunk_boundaries`** (goal 1 hygiene; lanes 1 and 2, one file each)
- What: delete the copies at `offload_cache.py:586-597` (lane 1) and `engine.py:134-142` (lane 2); import `cache_budget._arena_chunk_boundaries` (`:84-96`; identical bodies, tuple vs list return, callers use `.index()`/iteration so either works, but normalise to tuple).
- Verify: `grep -rn "def _arena_chunk_boundaries" python/` → exactly one; `pytest tests/engine/test_cache_budget.py tests/engine/test_growable_kv_arena_engine.py tests/moe/test_expert_arena_vmm.py` green (CPU parts).

**S4. Pin-prefix refusal** (goal 2; lane 2)
- What: in engine config resolution (next to `engine.py:2895-2910`, which already validates `--elastic-initial-requests` the same way), raise if any `pin_prefix_*` is set and `_resolve_cache_type(has_linear_attention, cache_type) != "hybrid_radix"`. **Remove `--pin-prefix-min-tokens 1024` from `scripts/serve-default.sh`** or production will refuse to start after this step; keep the comment truthful per CLAUDE.md.
- Why: `scheduler/cache.py:93`, `models/config.py:372-376`; fork commit `c819a81`.
- Breaks: any serve.env/measure.sh arm passing the flag on Nemotron. `grep -rn "pin-prefix" scripts/ tasks/ ~/.config/freetoken/serve.env`.
- Verify: CPU test with the Nemotron-H config fixture from `tests/models/test_nemotron_h.py` asserts the `ValueError`; with a Qwen3.5 config it does not; `grep pin-prefix scripts/serve-default.sh` → 0.

**S5a. Row-layout single source (part 1: the function)** (goal 4; lane 3)
- What: add `nvfp4_expert_row_layout(H, I, *, gated, kind_map=None)` to `models/nvfp4_banks.py` returning bank shapes, per-tensor `(bank, byte_offset, expected_shape, dtype)`, and `row_bytes`; make `_alloc_nvfp4_host_banks` and `MirrorExpertPool` (`nvfp4_bank_shapes`, `_row_layout`, `_KIND_DTYPE`, `_KIND_BANK_INDEX`) consume it; `_scan_checkpoint` canonicalises kinds through `spec.kind_map` and honours `global_reciprocal` exactly as upstream's loader does. **Depends on S0** for the upstream fields to exist in `nvfp4_banks.py`; until S0 lands, lane 3 can write the function and the tests against the fork's current spec and rebase the small diff after the sync.
- Verify: `grep -n "H // 16\|I // 16\|H // 2\b" python/freetoken/moe/mirror_pool.py` → 0; new CPU test extends `tests/moe/_mirror_checkpoint.py` to write three synthetic checkpoints (ungated modelopt, gated modelopt, gated compressed-tensors with `weight_packed`/`weight_global_scale` names) and asserts **pool rows are byte-equal to `load_nvfp4_expert_source_banks` rows** for the same checkpoint. That equivalence is the invariant the mirror's docstring claims (`mirror_pool.py:70-71`) and nothing tests today.

### Phase B — sync (serial, one worker owns every file; 1-2 days) then GPU checkpoint 1

**S0. Merge `origin/main` into `exp/exclusive-expert-ram`** (goal 1)
- What: `git fetch origin`, `git merge origin/main` (no rebase), resolve the 24 files. Policy for hunks: upstream wins on anything the fork did not deliberately change; fork wins in fork-owned regions; `nvfp4_banks.py` takes upstream's `kind_map`/`global_reciprocal` *and* keeps `gated`/`hidden_size_attr`; `offload_cache.py:126` takes upstream's table then applies S5b below. Also merge the trivial `AGENTS.md`/`CLAUDE.md` add/add.
- Breaks: behaviour can shift (`03c28d2` exact top-k/top-p sampling, `e05cff8` fused_topk via triton router, `2757bb5` `--gpu` binding via NVML, `58f4b9e` sm_89 `_scaled_mm` change for the Ada box). The needle/recall reference outputs may legitimately change, so **the whole-model reference must be re-recorded once after this step**, on the same commit as the pool arm.
- Verify: `git merge-tree --write-tree origin/main HEAD | grep -c '^CONFLICT'` → 0; both CPU suites at or above the handover counts (809/5, 632/11); then **GPU checkpoint 1** (bundle below), including re-recording the whole-model needle outputs.

**S5b. `_BANK_BYTES_PER_EXPERT["nvfp4"]` takes `gated`** (goal 4; folded into S0's resolution of that hunk, since it is the conflicting line)
- Verify: CPU test `bank_bytes_estimate(nemotron_config) == num_moe_layers * E * row_bytes(ungated)` and the gated case unchanged.

### Phase C — the two extractions (parallel lanes with disjoint files) then GPU checkpoint 2

**S6. `ExpertResidency` seam** (goal 3, serves 1 and 4; L; lane 1 owns `moe/offload_cache.py`, `moe/mirror_kernels.py`, `moe/mirror_pool.py`, new `moe/residency.py`, plus the three-line hooks in `engine.py:1093-1097` and `scheduler.py:3095-3101` by agreement with lane 2)
- What: pure move. `MirrorResidency` receives everything `_build_mirror_plan`, `_init_prefill_overlap_buffers`'s mirror branch, `_mirror_prefill_base`, `mirror_warm_start`, `_mirror_refill_uncovered`, `_mirror_stage_layer`, `_mirror_writeback_buffer`, `_prefetch_split_mirror`, `copy_missing_mirror`, `mirror_stats`, `mirror_fault_check`, `_mirror_publish_free_rows` do today; kernel bodies untouched; `OffloadMoeCache` gains one `residency` attribute and five hook calls; `WholeModelResidency` implements them as no-ops. Selection: `EngineConfig.expert_residency: Literal["whole","mirror"]` from `--expert-residency` (default `whole`); `--moe-mirror-host-rows` stays as the mirror's sizing knob; the `FREETOKEN_MIRROR_*` env names are resolved **only** in `server/args.py` as aliases so `measure.sh`'s write/strip logic keeps working. The engine's nine `if mirror` branches in `_init_offload_moe_cache` (`:785-990`) collapse to `residency = build_residency(config, mc, self)` plus the bank-source choice. `_mirror_final_gpu_slots` moves with it (fixing the `4 * step` fudge is S8, not here: move first, change later).
- Breaks: the stale-publish class. Mitigation is procedural: no line inside a moved function changes in this step; `swap_smoke.py` and `graph_race_repro.py` before and after; the acceptance bundle's counters and needle bytes compared to the record.
- Verify: `grep -c "_mirror" python/freetoken/moe/offload_cache.py` from ~110 to ≤ 15; `grep -c 'getattr(self, "_mirror"' ...` → 0; `python -c "import sys, freetoken.moe.offload_cache; assert 'freetoken.moe.mirror_pool' not in sys.modules"` (whole-model path never imports the pool); CPU suites; **GPU checkpoint 2**, which for this step also includes a whole-model arm at 8K/32K/80K (expect ~194/190/179 tok/s within the 9% noise floor) so the default profile is shown untouched.

**S7. `GrowableKvController`** (goal 2; M; lane 2 owns `engine/engine.py`, new `engine/growable_kv.py`, `engine/cache_budget.py`)
- What: move `_growable_moe_bytes`, `_plan_growable_kv`, `_rollback_growable_kv_transition`, `_refuse_if_growable_transition_failed`, `_grow_runtime_kv_arena`, `_shrink_runtime_kv_arena`, `grow_runtime_kv`, `shrink_runtime_kv` (`engine.py:1515-2200`) into a controller object holding `kv_cache`, `moe`, `graph_runner`, `attn_backend`; `Engine.grow_runtime_kv` becomes a one-line delegation so the scheduler's call sites (`scheduler.py:640, 751`) do not change. The mirror floor read (`engine.py:1800-1811`) becomes `moe.residency.min_gpu_slots()` (0 for whole-model), which is the only contact between S6 and S7; agree that name on day one. Replace the import-time constants `FREETOKEN_EXPERT_ARENA` (`offload_cache.py:151`, `offload_kernels.py:32`) with a config value derived at engine init (`kv_grow_step_tokens > 0` and format supports arena), keeping the env var as an `args.py` alias.
- Breaks: `tests/engine/test_growable_kv_transaction_source.py` and `tests/scheduler/test_growable_handoff_policy_source.py` ast-parse the `Engine`/`Scheduler` class bodies and `exec` selected methods — they will fail and must be rewritten to import the controller (which makes them real unit tests). `tests/moe/test_offload_kernels_usable_slots_source.py:50` asserts the literal string `FREETOKEN_EXPERT_ARENA` in the kernels' source; update it.
- Verify: `engine.py` shrinks by ≥ 600 lines; the three source-parsing tests rewritten and green; CPU growable suites (`tests/scheduler/test_growable_*`, `tests/engine/test_growable_*`, no GPU needed, verified 0 skip markers) green; at GPU checkpoint 2 the 1M arm's journal must produce the **same sequence** of `Committed growable KV through N tokens ... MoE slots a -> b` transitions as `results/nemotron-reserve-2e-1m-journal.txt` (`grep -o "MoE slots [0-9]* -> [0-9]*" | diff`), which is deterministic for a fixed config.

### Phase D — make goals 1 and 4 checkable (no GPU, parallel)

**S8. Model conformance check** (goal 4; M; lane 3 owns `models/nvfp4_banks.py`, `models/*/weight.py`, new `models/check_experts.py`, `docs/models.md`)
- What: `python -m freetoken.models.check_experts <checkpoint_dir>` reads `config.json` and safetensors headers only and prints/refuses: spec found; `gated` agrees with `expert_gated`; every expert has exactly the expected tensors after `kind_map`; one shard per expert; row bytes and total bank bytes from the single layout function; cache type the engine would resolve and whether `--pin-prefix-*` would be honoured. Add a CPU test parametrised over every module exporting `NVFP4_EXPERT_SOURCE_SPEC` (today: nemotron_h, qwen3_5_moe, minimax_m2/m3, glm4_moe, gemma4) that builds a synthetic checkpoint from a new `key_template` field on the spec (the regex cannot be rendered; the template can, and a test asserts template output matches `key_pattern`). Write the "adding a model" checklist in `docs/models.md` as the sequence of checks this command performs. Also fold in the `_mirror_final_gpu_slots` cleanup: make it call `_plan_growable_kv`'s arithmetic through a pure function of config and drop the `4 * step` term (`engine.py:1608-1617`), with a CPU test asserting planner and estimator agree on the Nemotron config.
- Verify: the command accepts both checkpoints on disk (`~/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`, and Ornith's path) and refuses a synthetic checkpoint with a renamed kind with a message naming the kind; the parametrised test is green for all six specs.

**S9. Conflict budget and shape rules** (goal 1; S; lane 2)
- What: `scripts/upstream-sync-check.sh`: `git fetch origin`, `git merge-tree --write-tree origin/main HEAD`, print conflicting files and hunk counts, exit non-zero above a budget (set it to today's number after S0, i.e. 0, and let it grow only with explicit acknowledgement). A short "fork shape rules" section proposed for the repo's `CLAUDE.md` (owner adopts; I do not edit it): new behaviour in new modules; ≤ ~20 lines of hooks per upstream file per feature; no import-time env constants; env vars resolved in `args.py` only; no tests that parse upstream class bodies; no measurement arm names, dates or "lever N" in library code (41 such comments today; they belong in `tasks/`); sync on a fixed cadence.
- Verify: the script runs and reports 0 after S0; re-run at the end of each phase.

**S10 (conditional on Q2). Drop the legacy growable path** (goal 2; S)
- Only if the owner confirms the arena is the sole supported growable mode. Otherwise keep it: it is live for GGUF size classes and marlin/b12x (`offload_cache.py:260-266, 721`), which is the GGUF Ornith path.

---

## Ordering, dependencies, lanes

```
Phase A (parallel, no GPU)      lane 1: S1 → S2 → S3(offload_cache side)
                                lane 2: S4 → S3(engine side)
                                lane 3: S5a (function + tests; small rebase after S0)
Phase B (serial)                one worker: S0 (+S5b)      → GPU checkpoint 1
Phase C (parallel, no GPU)      lane 1: S6                  lane 2: S7        lane 3: S8 (tests + docs, no engine edits)
                                                            → GPU checkpoint 2
Phase D (parallel, no GPU)      lane 2: S9                  lane 3: S8 remainder (engine estimator edit lands here, after S7)
```

File ownership that must hold: `offload_cache.py`, `mirror_*.py` → lane 1 only; `engine.py`, `cache_budget.py`, `args.py`, `engine/config.py`, `scripts/serve-default.sh` → lane 2 only; `models/**`, `expert_banks.py`, `tests/moe/_mirror_checkpoint.py`, `docs/models.md` → lane 3 only; `scheduler.py` is touched by S2 (lane 1, the `_forward` hook) and S6 (same hook) and by nothing in lane 2. S6 and S7 meet at one name (`residency.min_gpu_slots()`), agreed before either starts. S0 is exclusive: no other worker edits `python/` while the merge is open. S5a may be written before S0 but its `kind_map` part lands after.

Phase A can equally run after S0 if the owner wants the sync proven first; nothing in A changes the conflict count.

## Acceptance bundle (run at each GPU checkpoint; the owner runs the server, never an agent shell)

1. CPU: `tests/moe tests/engine tests/scheduler` ≥ 809 passed and `tests/kernels tests/models` ≥ 632 passed, server venv, model unloaded.
2. Device (model unloaded, GPU empty): `pytest tests/moe/test_mirror_device.py tests/moe/test_expert_arena_vmm.py tests/kernels/test_vmm_tensor.py`.
3. Measurement, one arm, the record config (`results/nemotron-reserve-2e-1m.env`: empty GPU, ratio 1.00, auto pool, reserve 2E, tie-break on, q8_0/q8_0, thinking off, port 1920, two passes): pass-2 decode within the 9% noise floor of 76.9 tok/s at 1M and of the 8K/32K/80K record; host RAM in 12.26 ± 0.6 GiB; `stats.json` `coverage_faults == 0 and starved_writebacks == 0`; journal has exactly one graph capture and no `Traceback`; `acceptance.sh R3` and `R6` pass.
4. Correctness: `needles.py` at 21K/120K with the raised thinking budget (`d4f04a6`) and `recall.py` 3/3 at 21K/120K/240K; answers byte-identical to the whole-model reference **recorded on the same commit** (re-recorded once at checkpoint 1 after the sync, then reused).
5. For S7 only: the `MoE slots a -> b` transition sequence in the 1M journal diffs clean against the record.

A step whose GPU checkpoint fails any of 3-5 is reverted on the branch, not patched forward.

## Do not do this

Carried over: no kernel body edits; no change to the coverage invariant, free stack or `publish_freed_rows`; no rewrite of session spill; no touching `main`, no push, no merge into `main`; no `uv sync`; no server from an agent shell; embedder stays down; no numbers on a non-empty GPU.
Added: no `git rebase` onto upstream (merge only); no upstream PR without the owner running it on hardware himself; no collapsing the `_v2` kernel twins; do not change `measure.sh` knob names (`FT_RESERVE`, `FT_TIEBREAK`) without both a write and a strip entry; do not move `tasks/exclusive-expert-ram/*.py` while `test_mirror_device.py` shells out to them (S6 may move them into `tests/moe/device/` afterwards, as one commit); do not edit a moved function's body in the same commit as the move; do not "fix" `_mirror_final_gpu_slots`'s `4 * step` before S8's planner/estimator test exists; do not touch `~/.config/freetoken/serve.env` except the already-authorised ratio.

## Questions for the owner before committing

1. **Sync mechanics.** Merge `origin/main` into `exp/exclusive-expert-ram` directly (recommended, since `main` is off-limits), or do you want fork `main` to track upstream and the exp branch to merge from it? And what cadence counts as "at any time": weekly, or before every measurement phase?
2. **Legacy growable path.** Is the arena the only supported growable mode from now on, or must GGUF Ornith (size classes, arena-refused) keep the rebuild-and-recapture path? That decides S10 and how much of `grow_runtime_kv` S7 carries.
3. **Pin-prefix on Nemotron.** Refuse loudly (S4, cheap) and drop the flag from `serve-default.sh`, or is prefix pinning on the `radix` + Mamba path a feature you want built? That is a separate design, not a refactor.
4. **Upstream PRs.** Are you willing to own one small PR (spec fields + ungated byte table) on upstream's terms? If not, I drop upstreaming entirely; goal 1 does not depend on it.
5. **GPU windows.** Two checkpoints of roughly half a day each (probes + one 1M arm + needles), plus re-recording the whole-model needle reference once after the sync. Acceptable, and who runs the server?
6. **Flag naming.** `--expert-residency {whole,mirror}` with `--moe-mirror-host-rows` as the sizing knob, or keep the single `--moe-mirror-host-rows` (0 off / -1 auto / N) as the switch? The seam is the same either way.
7. **Model #3 for the conformance check.** Fine to use a compressed-tensors NVFP4 checkpoint (upstream's GLM-5.3-Flash-NVFP4 layout) only as a headers-on-disk or synthetic target, never served?
8. **The ungated byte table.** `bank_bytes_estimate` overestimates Nemotron's banks by ~1.6x. Do you know whether that number feeds any live decision on this host (pin budget is 17 GiB against 15.41 GiB of banks, and R6 passes), or should S5b's test be the first time anyone checks?

Paths cited: `/home/lucas/ai/FreeToken-wt/exclusive-expert-ram/python/freetoken/{moe/offload_cache.py,moe/mirror_pool.py,moe/mirror_kernels.py,moe/offload_kernels.py,moe/expert_banks.py,engine/engine.py,engine/cache_budget.py,engine/config.py,scheduler/scheduler.py,scheduler/cache.py,scheduler/session_spill.py,models/nvfp4_banks.py,models/config.py,server/args.py}`, `scripts/serve-default.sh`, `tasks/exclusive-expert-ram/{plan.md,measure.sh,results/,reviews/2026-09-22-handover.md}`, `tests/{moe,engine,scheduler}/`.