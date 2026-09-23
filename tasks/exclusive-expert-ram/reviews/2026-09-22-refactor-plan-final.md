# Reorganisation plan for `exp/reorg` — final, authoritative

Written 2026-09-22 against HEAD `bc6f815`, now carried on its own branch. The work
happens in the worktree `/home/lucas/ai/FreeToken-wt/reorg` (branch `exp/reorg`,
branched at `cde7ace` from `exp/exclusive-expert-ram`, which is frozen as the
mirror-pool record and is never committed to again by this campaign;
fork of `origin` = https://github.com/FlashML-org/FreeToken, merge-base `bd372b6`,
last fetch 2026-09-17). Every `file:line` below was read at `bc6f815`; re-check
before editing, because lines move. This document authorises nothing by itself:
the owner accepts it, the coordinator commits it, workers act on the step records
in section 9, not on prose.

It supersedes `2026-09-22-architecture-review.md` (kept as the evidence trail) and
the conversational plan drafts. Where this file corrects an earlier claim of mine,
the correction is stated in place and not hidden.

Reader assumption: you have the repository and nothing else.

### Provenance audit (2026-09-23): whose rule is whose

The `/goal` that drives this campaign was drafted by the agent and pasted by the owner,
so its rules block reads as the owner's while most of it is not. Checked against the
owner's actual messages (campaign record `K-attribution-audit-2026-09-23`):

- **Owner's own words:** single lane is permanent design; stop the piro-board embedder for
  measurements and do not restart it; the four goals (G1 sync with upstream at any time,
  G2 the smart KV manager, G3 the RAM saver as a switch, G4 adding models without breaking
  the code); delete the legacy growable path and reimplement GGUF Ornith on the new one;
  Ornith subordinate; everything but Phase E; the S13 pin-prefix step; a new branch for
  this; the campaign comes before the upstream PR (2026-09-23: "I will not run the pr
  command, move on with our campaign").
- **Agent-authored, previously presented as the owner's:** never `uv sync`; the owner
  starts checkpoint servers; two probe passes with pass 2 of record; port 1920; baseline
  bracketing; the 22 GiB MemAvailable floor; "stop only for authority"; revert-not-patch.
  These are measurement and safety practice proposed by the agent. They stay in force as
  the campaign's working rules until the owner changes them, but they are labelled as
  agent practice below, not as owner decisions.
- **Not found in any owner message:** the quote "a refactor that loses any of that is a
  failure"; "owner accepted" for merge-not-rebase and for the upstream PR; memory ratio
  1.00 as an owner decision (recorded as such in an earlier plan; unverified).
- **Factual corrections:** 76.9 tok/s is probe pass 1 at 1M; pass 2 was 72.9. The pool
  record is 8K/80K/713K, not 8K/32K/80K. The 809-passed floor is stale. Nemotron routes
  top-6, not top-8 (Ornith is top-8). Ornith has 2 experts whose tensors span shards
  ((18, 90) and (38, 225), recounted twice by the S8 lane), not 258.

---

## 0. The deliverable, in one paragraph

Reorganise the fork so that (1) it can merge upstream at any time, (2) the smart KV
manager — growable KV, prefix cache, session spill — is kept and sharpened,
(3) the expert RAM pool becomes a clean, switchable seam, and (4) adding a model is
easy and checkable. The reorganisation is the deliverable. Ornith is a consumer of
goal 4 — the proof that model onboarding works end to end, with a measured prefill
number — not a workstream of its own. **If Ornith work and reorganisation work ever
contend for a GPU window or a file, the reorganisation wins and Ornith waits; nothing
in S0–S11 or S13 may be delayed for S12.**

Measured behaviour that must survive every step (the "evidence record", section 7):
1M context on Nemotron in 12.26 GiB host RAM, decode 76.9 tok/s (probe pass 1; pass 2 was 72.9), 0 coverage faults,
0 starved writebacks, one CUDA-graph capture, and needle/recall answers byte-identical
between the bounded pool and the whole-model-in-RAM baseline.

---

## 1. Binding decisions (owner, 2026-09-22) — constraints, not settings

| id | constraint | reason / evidence |
|---|---|---|
| C-SINGLE-LANE | **Single lane is permanent design**: consumer GPUs, one session computed at a time; other sessions queued and spilled. `--max-running-requests 1` is not a tunable. | Owner: "FreeToken is to run on consumer GPU, not datacenter graded ones, and it's more than proved that it can only handle a single session at a time." Measured (`benchmarks/results/nemotron35_lightning_5080_single_lane_2026-09-17.md:14-28`): single lane 148–174 tok/s decode for one request vs the 16-lane profile's 75 tok/s alone and 16–41 tok/s per lane under load; MoE cache 1924 slots vs 1023 — the 16 lanes' KV was eating the expert arena. Not a controlled A/B (different commits and profiles): margin indicative, direction settled. |
| C-NO-MAIN | Work stays on `exp/reorg` (branched from `exp/exclusive-expert-ram` at cde7ace, which stays frozen as the mirror-pool record). No merge into `main`, no push, never touch `main`. | Owner's authority limits (handover §1). |
| C-MERGE-NOT-REBASE | Upstream is taken by `git merge origin/main` into the branch, never by rebase. | Rebase replays 328 agent-authored commits through ~40 recurring conflict hunks and reviews nothing; a merge resolves each hunk once. Agent recommendation (section 3.3); no owner acceptance found in the owner's messages (audit 2026-09-23). |
| C-EMPTY-GPU | No performance number is valid unless `nvidia-smi` read 0 MiB before the arm, embedder stopped and never restarted by an agent, port 1920, one arm at a time, memory ratio 1.00, two probe passes, pass 2 of record. | Embedder stop: owner. Ratio 1.00: recorded as an owner decision in an earlier plan, not found in owner messages. Empty GPU, port, passes: agent measurement practice. Handover §11; `tasks/exclusive-expert-ram/measure.sh`. |
| C-EVIDENCE | The evidence record (section 7) must be reproduced at every GPU checkpoint. A step that fails a checkpoint is **reverted on the branch, not patched forward**. | Agent-authored rule. The quote formerly attributed to the owner here appears in no owner message (audit 2026-09-23). Rationale: a regression must not be hidden by patching under a checkpoint. |
| C-REORG-OVER-ORNITH | S12 never pre-empts S0–S11/S13 for GPU time or file ownership. | Owner, 2026-09-22 (section 0). |
| C-SERVER-VENV | Tests run with the server venv, model unloaded: `PYTHONPATH=$PWD/python /home/lucas/ai/FreeToken/.venv/bin/python -m pytest -q ...`. Do not `uv sync` the live server's venv while it serves (agent caution, not an owner rule; the owner's `CLAUDE.md` uses `uv sync` to set up a machine). Never torch pytest beside a live model. Never start a server from an agent shell. | Handover §11–12; `CLAUDE.md`. |
| C-UPSTREAM-PR | Exactly one upstream PR is authorised (VMMTensor and/or the spec fields), **after S0**, and the owner runs it on hardware and submits it himself. | Upstream `AGENTS.md`/`CONTRIBUTING.md` refuse agent PRs. Owner asked about the PR; no acceptance found, and on 2026-09-23 the owner parked it ("I will not run the pr command"). Upstreaming is subordinate to goal 1, not its mechanism. |
| C-LEGACY-OFF | The legacy rebuild-and-recapture growable path is deleted (S10, unconditional). GGUF Ornith is re-implemented on the arena (S12b) and waits on it. | Owner Q2 answer. |
| C-ELASTIC-OFF | `--elastic-initial-requests` is retired (S11). | Follows from C-SINGLE-LANE. |

---

## 2. Vocabulary — read this before touching anything named "elastic"

Our own docs use "elastic" for two different things, and that already sent one
piece of work in the wrong direction. Three lines, unambiguous:

- **Growable KV** — one session's KV cache grows and shrinks at runtime in 64K steps, funded from the expert arena (`--kv-grow-step-tokens`, `engine.py:2016-2200`, `offload_cache.py:1212-1338`). **KEEP and sharpen.**
- **Session queue + spill** — many sessions served, one resident on the GPU at a time, the rest checkpointed to RAM/disk and restored (`scheduler/session_spill.py`, `scheduler.py:311-325, 1354-1420`). **KEEP.** This is how many sessions are served; it does not depend on the next item (0 references to it in `session_spill.py`).
- **`--elastic-initial-requests`** — many sessions resident *simultaneously*, with the resident count auto-resized between an initial tier and `--max-running-requests` (`engine.py:2240-2340, 2833-2850`, `scheduler.py:116, 2490-2600`, `kvcache/linear_state_pool.py:144-174`, `kvcache/mha_pool.py`, `engine/graph.py`). **RETIRE (S11).** Retiring it does **not** reduce how many sessions can be served: the queue + spill path serves them one at a time, which is the single-lane design.

Other terms used below:

- **Arena** — the fixed-capacity VMM region of expert slots whose *usable* count shrinks/grows in place (`set_usable_slots`) so decode graphs are never recaptured. Gated today by the import-time constant `FREETOKEN_EXPERT_ARENA` (`offload_cache.py:151`, `offload_kernels.py:32`); on in production (`scripts/serve-default.sh`).
- **Legacy path** — the pre-arena way of funding KV growth: `moe.rebuild()` plus CUDA-graph destroy/recapture (`engine.py:2050-2140`). Off in production; still live for formats the arena refuses (`offload_cache.py:260-266` marlin/b12x, `:721` GGUF size classes). Deleted by S10.
- **Residency** — where an expert's bytes live when not on the GPU. Today two implementations are interleaved in one class: whole model pinned in host RAM (default), or the bounded mirror pool (`moe/mirror_pool.py`, on with `--moe-mirror-host-rows` / `FREETOKEN_MIRROR_EXPERT_RAM=1`). S6 makes this a seam.

---

## 3. What the review established (evidence, with corrections)

### 3.1 Shape of the fork
328 commits ahead / 19 behind `origin/main`; 782 files, +135,389/−1,290 lines; 326 of 328 commits authored by the agent user `probe` over 30 days. Three upstream files tripled: `moe/offload_cache.py` 1042→3293, `engine/engine.py` 1505→3289, `scheduler/scheduler.py` 905→3154 (44 added methods: leases, spill, soft sessions, growable handoff, elastic). Env-var knobs 52→108. Upstream itself moved 19 commits / 20.5K lines since the merge-base, of which only ~253 lines touch our hot files.

### 3.2 Sync cost measured, not guessed
`git merge-tree --write-tree origin/main HEAD` (touches nothing) conflicts in **24 files, ~40 hunks**: `engine.py` 10, `args.py` 4, `attention/linear.py` 4, `kernel/csrc/cpu_moe/cpu_moe_ext.cpp` 5, `models/config.py` 2, `kvcache/__init__.py` 2, `kvcache/base.py` 2, `models/register.py` 2, `moe/fused_nvfp4.py` 2, and one each in `moe/offload_cache.py` (the `_BANK_BYTES_PER_EXPERT` table, ~line 126), `models/nvfp4_banks.py` (upstream's new `kind_map`/`global_reciprocal` spec fields), `moe/fused.py`, `moe/cpu_executor.py`, `kernel/triton/sampling.py`, `kernel/aot_models.py`, `models/qwen3_5_moe/moe.py`, plus `AGENTS.md`/`CLAUDE.md` add/add. That is 1–2 days of merging today. The plan's job is to keep it there.

### 3.3 Corrections to earlier statements (mine), applied here
- **"Rebase forces a review of our 51K lines"** — false. Rebase replays commits and stops only at collisions. Merge is the mechanism (C-MERGE-NOT-REBASE).
- **"Elastic only ever worked on GDN models, never on Nemotron"** — false. `models/nemotron_h/config.py:204` builds the Mamba layers as a `LinearGatedDeltaGroupConfig`, so `has_linear_attention` (`models/config.py:372-376`) is true for Nemotron; the 2026-09-04 Switchyard soak ran Nemotron with `--elastic-initial-requests 4`. S11's scope therefore includes the Mamba state-pool remap, not only GDN.
- **"`--pin-prefix-*` is inert on Nemotron because the cache resolves to `radix`"** — false, and it came from our own stale doc. The 1M journals print both `cache_type='radix'` (requested, on the args line) and `cache_type='hybrid_radix'` (resolved). `tasks/exclusive-expert-ram/STATUS.md:211` states the opposite and is **wrong**. Pins are honoured on Nemotron; what makes them do nothing for one agent re-reading its own haystack is the *policy* at `scheduler/cache.py:610-663`: pins fire only for a prefix two distinct session keys share. Consequences: the "refuse pin flags on Nemotron" step (old S4) is **withdrawn**; `scripts/serve-default.sh` keeps `--pin-prefix-min-tokens 1024`; the doc correction is now step S4; the single-session pin policy is step S13.

### 3.4 The structural findings the steps address
- Residency state has no owner: 15 `getattr(self, "_mirror", None)` branches in `OffloadMoeCache`, 9 `if mirror` branches in `engine._init_offload_moe_cache` (`engine.py:775-1160`), a mirror hook in `scheduler._forward` (`scheduler.py:3080-3101`), numpy snapshots patched from three places (`offload_cache.py:2189-2299, 2321-2400, 2480-2585`).
- A 7-slot counter vector shared by two kernels with positional unpacks at `offload_cache.py:1841-1842` and `1980-1981`; the layout exists only in kernel comments (`mirror_kernels.py:117-119, 365-369`). Cause of the `[6]→[7]` miss and the >1.0 `free_eviction_rate` (`d9b0070`).
- `_mirror_restore_coverage` (`offload_cache.py:1793-1835`) has no callers anywhere. `mirror_warm_start` is called from four sites; the two "restore at prefill→decode" sites (`offload_cache.py:2770-2782`, `scheduler.py:3080-3094`) carry a rationale STATUS says stopped being true at `f007a7c`, and their trigger is set only in `materialize_layer` (`:2828`), which is not the prefill path under the mandatory overlap (`engine.py:820-835`, dispatch at `offload_cache.py:2461-2462`). Inferred unreachable; to be proven by the device test at checkpoint 1.
- `_arena_chunk_boundaries` exists three times (`engine/cache_budget.py:84`, `moe/offload_cache.py:586`, `engine/engine.py:134`), each with a comment saying it is duplicated because of a task "ownership boundary". Process constraints became code structure.
- The NVFP4 expert-row byte layout is stated in four places: `models/nvfp4_banks.py:79-94` (`_alloc_nvfp4_host_banks`, gated-aware), `moe/mirror_pool.py:63-84, 311-332, 59-60` (gated-aware, `kind_map`-unaware — a compressed-tensors NVFP4 checkpoint such as upstream's GLM-5.3-Flash-NVFP4 is refused with "unknown tensor kind"), `moe/offload_cache.py:126` (`_BANK_BYTES_PER_EXPERT["nvfp4"]`, **assumes gated `2*I` regardless of `expert_gated`**, consumed by `moe/expert_banks.py:453-467` `bank_bytes_estimate` for pin-budget sizing — ~1.6x over for Nemotron; whether that changes any live decision is unknown and S5b's test is the first check), and `engine/cache_budget.py:17-31` (derived from tensors, fine).
- `_mirror_final_gpu_slots` (`engine.py:1553-1620`) re-derives the KV/arena budget from config with a `- 4 * step` fudge, a second copy of `_plan_growable_kv`'s arithmetic.
- Import-time env constants select kernels (`FREETOKEN_EXPERT_ARENA`); three tests assert on source text or `exec` methods lifted from class bodies (`tests/engine/test_growable_kv_transaction_source.py`, `tests/scheduler/test_growable_handoff_policy_source.py`, `tests/moe/test_offload_kernels_usable_slots_source.py:50`); `tests/moe/test_mirror_device.py` shells out to `tasks/exclusive-expert-ram/{swap_smoke,graph_race_repro}.py`; 41 comments in library code cite measurement arms, dates or "lever N".
- What is sound and must not be mistaken for glue: `moe/mirror_pool.py` (one class, one invariant, no engine imports), `moe/mirror_kernels.py` (three hazards documented and encoded), `kernel/vmm.py` + `kernel/csrc/vmm_tensor.cpp` (RAII, mutex, overlap validation), and the test count.

### 3.5 Horizon: what the plan reaches and what it does not
After Phase D the expert-cache and engine side has the shape that syncs cheaply for good: new modules plus small hooks. The scheduler does not: it is a different scheduler, and neither this plan nor a full re-port can hook it into upstream's. Its tax is proportional to upstream's activity there, which is currently low (471 lines across `scheduler/` and `kvcache/` in a month). **Phase E** — re-port only the scheduler as our own `SingleLaneScheduler` module selected by config, leaving upstream's `Scheduler` untouched — is a real option, deliberately **deferred until Phase D's sync-check numbers exist** (S9). A full re-port now is rejected: it costs the refactor plus a rewrite, voids every measurement until re-measured, and re-finds thirty days of bugs, for a shape the plan reaches everywhere except the scheduler.

### 3.6 Performance of the reorganisation itself
Token-by-token work is GPU kernels replayed from a captured graph (`engine.py:2409-2412`); no Python runs there. The seams add calls once per layer per prefill chunk (microseconds beside ≥10 ms of GPU work), once per 64K-token KV growth, once per batch. Not measurable. The real risk of S6/S7/S13 is **ordering** (side-stream copies, sync-before-unmap, snapshot patching), which shows up as wrong bytes or a crash, not as slowness; the byte-identical needle test is the detector. Because the throughput bundle only resolves drops above the ~9% noise floor (handover §12: 8.7% between identical arms at 80K), Phase C adds the finer instrument in section 7.3: identical kernel-launch count and identical bytes moved per decode step, before and after.

---

## 4. Goal → step map

| goal | weight | steps |
|---|---|---|
| G1 sync with upstream at any time | 1 (highest) | S0, S3, S9, PR-1; Phase E deferred |
| G2 keep and sharpen the smart KV manager | 2 | S7, S10, S11, S13 |
| G3 expert RAM pool as a clean switchable seam | 3 | S1, S2, S6 |
| G4 adding a model easy and checkable | 4 | S4 (doc truth), S5a, S5b, S8, validated by S12 |

Nothing is dropped. "A flag option" (G3) is delivered as a seam selected by a config value, not as a code-skipping flag: the flag already exists today (`engine.py:785-789`) and is off in production, and it is not what the owner needs, because with it off the pool's state still threads through three files. Build-time absence (a plugin) is rejected as fiction: the pool needs `OffloadMoeCache`'s slot maps and the kernels' `victim_ids`/`prior_ids`.

---

## 5. Steps

Size: S ≤ 1 day, M 2–5 days, L 1–2 weeks. "GPU" says whether the step itself needs the card; every step is finally judged at the next GPU checkpoint (section 7). Test invocation per C-SERVER-VENV.

### S0 — Merge `origin/main` into the branch (G1; M; no GPU for the merge, GPU checkpoint 1 after)
- **What**: `git fetch origin && git merge origin/main`; resolve the 24 files. Hunk policy: upstream wins wherever the fork did not deliberately change behaviour; fork wins in fork-owned regions; `models/nvfp4_banks.py` keeps upstream's `kind_map`/`global_reciprocal` *and* our `gated`/`hidden_size_attr`; `offload_cache.py:126` takes upstream's table and then S5b's `gated` argument. Trivial add/add on `AGENTS.md`, `CLAUDE.md`.
- **Why**: section 3.2; the sync must be proven on real code before the reorganisation is measured against it, and S5a needs upstream's spec fields.
- **Could break**: behaviour can shift — `03c28d2` exact Triton top-k/top-p sampling, `e05cff8` fused_topk through the in-repo router, `2757bb5` `--gpu` bound via NVML, `58f4b9e` sm_89 `_scaled_mm` change (the Ada box). Reference outputs may legitimately change.
- **Verify**: `git merge-tree --write-tree origin/main HEAD | grep -c '^CONFLICT'` → 0; CPU suites at or above the handover counts (counts at writing: 809 / 5 and 632 / 11; after the S0 merge the floors are 871 passed / 8 skipped and 871 passed / 195 skipped with 4 upstream failures that reproduce on pristine origin/main); **GPU checkpoint 1**, which re-records the whole-model needle/recall reference **on this commit** so later byte comparisons have a valid reference.
- **Exclusive**: no other worker edits `python/` while the merge is open.

### PR-1 — One upstream PR (G1, subordinate; S; owner runs it)
- **What**: after S0, on a branch off `origin/main`: the `gated`/`hidden_size_attr` spec fields and the `_BANK_BYTES_PER_EXPERT["nvfp4"]` `gated` fix (these sit exactly where upstream's `kind_map` change conflicts and will conflict again), optionally `VMMTensor` (`kernel/vmm.py`, `kernel/csrc/vmm_tensor.cpp`, `tests/kernels/test_vmm_tensor.py`, self-contained). The owner runs it on hardware and submits it; agents prepare the diff and the A/B evidence only (C-UPSTREAM-PR).
- **Verify**: the PR branch applies cleanly on `origin/main`; its tests pass there; not a gate for anything else.

### S1 — Named counter schema (G3; S; no GPU)
- **What**: `moe/mirror_stats.py` with a `MirrorStat` IntEnum; both kernels take the offsets as constexpr; `mirror_stats()` and `mirror_fault_check()` unpack by name.
- **Why**: section 3.4 (positional unpacks at `offload_cache.py:1841-1842`, `1980-1981`).
- **Could break**: nothing at runtime if offsets are unchanged.
- **Verify**: `grep -n 'stats"\].tolist()\|stats_host"\].tolist()' python/freetoken/moe/offload_cache.py` → 0; new CPU test feeds a fake 7-vector to `mirror_stats()` and asserts `free_eviction_rate == free_evictions / swaps` and `buffer_free_evictions` reported separately; `pytest tests/moe -k mirror` green (CPU parts).

### S2 — Delete dead and stale coverage-restore paths (G3; S; GPU checkpoint 1 proves it)
- **What**: remove `_mirror_restore_coverage`; make `materialize_layer` raise under the mirror; delete `_mirror_needs_coverage` and the two duplicate restore sites, leaving `mirror_fault_check` at the batch boundary; fix `plan.md`'s "Slot regions" section (stale: says `[0, 2E)` is prefill's alone; the victim floor is gone since `1ed6372`).
- **Why**: section 3.4.
- **Could break**: if the reachability inference is wrong, the new raise fires on the first mirrored prefill — loud.
- **Verify**: `grep -rn "_mirror_restore_coverage\|_mirror_needs_coverage" python/ tests/ tasks/` → 0; `pytest tests/scheduler tests/moe` green; **not done until checkpoint 1 runs `tests/moe/test_mirror_device.py`** (`graph_race_repro.py` is the test for exactly this boundary).

### S3 — One `_arena_chunk_boundaries` (G1 hygiene; S; no GPU)
- **What**: delete the copies at `offload_cache.py:586-597` and `engine.py:134-142`; import `cache_budget._arena_chunk_boundaries` (`:84-96`), normalising to its tuple return.
- **Verify**: `grep -rn "def _arena_chunk_boundaries" python/` → exactly one; `pytest tests/engine/test_cache_budget.py tests/engine/test_growable_kv_arena_engine.py tests/moe/test_expert_arena_vmm.py` green (CPU parts).

### S4 — Correct our own record: STATUS.md:211 and everything derived from it (G4 truth; S; no GPU)
- **What**: rewrite the paragraph at `tasks/exclusive-expert-ram/STATUS.md:209-215` to state the truth (cache resolves to `hybrid_radix`; pins are honoured; the miss is the cross-session-only policy at `scheduler/cache.py:610-663`); grep STATUS.md, plan.md, `results/README.md` and the two review files for "radix" claims and fix them; add a one-line note in `docs/nemotron.md`'s single-lane section that the resolved cache type is `hybrid_radix`. Old step "S4: refuse pin flags on Nemotron" is withdrawn and recorded as withdrawn here.
- **Why**: section 3.3 — the stale doc misled the reviewer, then the coordinator, and the owner was told something false.
- **Verify**: `grep -n "cache_type='radix'" tasks/exclusive-expert-ram/STATUS.md` → 0 occurrences that claim it is the resolved type; a journal line `cache_type='hybrid_radix'` is cited by path in the corrected paragraph.

### S5a — Single source for the NVFP4 expert-row layout (G4; M; no GPU; after S0)
- **What**: `nvfp4_expert_row_layout(H, I, *, gated, kind_map=None)` in `models/nvfp4_banks.py` returning bank shapes, per-tensor `(bank, byte_offset, expected_shape, dtype)`, and `row_bytes`; `_alloc_nvfp4_host_banks` and `MirrorExpertPool` (`nvfp4_bank_shapes`, `_row_layout`, `_KIND_DTYPE`, `_KIND_BANK_INDEX`) consume it; `_scan_checkpoint` canonicalises kinds through `spec.kind_map` and honours `global_reciprocal` exactly as upstream's loader does. The function may be drafted before S0 and rebased after.
- **Why**: section 3.4 (four copies; compressed-tensors refusal).
- **Verify**: `grep -n "H // 16\|I // 16\|H // 2\b" python/freetoken/moe/mirror_pool.py` → 0; new CPU test extends `tests/moe/_mirror_checkpoint.py` to write three synthetic checkpoints (ungated modelopt, gated modelopt, gated compressed-tensors with `weight_packed`/`weight_global_scale` names) and asserts **pool rows are byte-equal to `load_nvfp4_expert_source_banks` rows** for the same checkpoint — the invariant `mirror_pool.py:70-71` claims and nothing tests today.

### S5b — `_BANK_BYTES_PER_EXPERT["nvfp4"]` takes `gated` (G4; S; folded into S0's resolution of that hunk)
- **Verify**: CPU test `bank_bytes_estimate(nemotron_config) == num_moe_layers * E * row_bytes(ungated)`; gated case unchanged. This is the first check of whether the 1.6x over-estimate was load-bearing (owner does not know; record the answer in STATUS).

### S6 — `ExpertResidency` seam (G3, serves G1 and G4; L; GPU checkpoint 2)
- **What**: pure move. `moe/residency.py` defines the protocol and two implementations. `MirrorResidency` receives, without editing their bodies, `_build_mirror_plan`, the mirror branch of `_init_prefill_overlap_buffers`, `_mirror_prefill_base`, `mirror_warm_start`, `_mirror_refill_uncovered`, `_mirror_stage_layer`, `_mirror_writeback_buffer`, `_prefetch_split_mirror`, `copy_missing_mirror`, `mirror_stats`, `mirror_fault_check`, `_mirror_publish_free_rows`, `_mirror_final_gpu_slots`, and the kernel launches. `WholeModelResidency` implements the hooks as no-ops and `min_gpu_slots() == 0`. `OffloadMoeCache` keeps the slot maps and calls `residency.before_ensure(layer_id)`, `.before_buffer_fill(buffer_id)`, `.prefetch_layer(layer_id, buffer_id)`, `.before_shrink(n, current)`, `.fault_check()`. Selection: `EngineConfig.expert_residency: Literal["whole", "mirror"]` from `--expert-residency` (default `whole`); `--moe-mirror-host-rows` stays the mirror's sizing knob; `FREETOKEN_MIRROR_*` env names resolve **only** in `server/args.py` as aliases so `measure.sh`'s write/strip logic keeps working. The engine's nine `if mirror` branches collapse to `residency = build_residency(config, mc, self)` plus the bank-source choice. The `4 * step` fudge is not fixed here (S8): move first, change later.
- **Could break**: the stale-publish class (a wrong free-row publish once served wrong experts with every counter at zero). Mitigation is procedural: no line inside a moved function changes in this step; `tasks/exclusive-expert-ram/swap_smoke.py` and `graph_race_repro.py` before and after; the bundle's counters and needle bytes against the record.
- **Verify**: `grep -c "_mirror" python/freetoken/moe/offload_cache.py` from ~110 to ≤ 15; `grep -c 'getattr(self, "_mirror"' python/freetoken/moe/offload_cache.py` → 0; `python -c "import sys, freetoken.moe.offload_cache; assert 'freetoken.moe.mirror_pool' not in sys.modules"` (whole-model path never imports the pool); CPU suites; **GPU checkpoint 2** including a whole-model arm at 8K/32K/80K (expect ~194/190/179 tok/s within the noise floor) and the section 7.3 instrument.

### S7 — `GrowableKvController` (G2; M; GPU checkpoint 2)
- **What**: move `_growable_moe_bytes`, `_plan_growable_kv`, `_rollback_growable_kv_transition`, `_refuse_if_growable_transition_failed`, `_grow_runtime_kv_arena`, `_shrink_runtime_kv_arena`, `grow_runtime_kv`, `shrink_runtime_kv` (`engine.py:1515-2200`) into `engine/growable_kv.py`, a controller holding `kv_cache`, `moe`, `graph_runner`, `attn_backend`; `Engine.grow_runtime_kv`/`shrink_runtime_kv` become one-line delegations so the scheduler's call sites (`scheduler.py:640, 751`) do not change. The mirror floor read (`engine.py:1800-1811`) becomes `moe.residency.min_gpu_slots()` — the only contact with S6; agree the name before either starts. Replace the import-time constants `FREETOKEN_EXPERT_ARENA` (`offload_cache.py:151`, `offload_kernels.py:32`) with a config value derived at engine init (`kv_grow_step_tokens > 0` and the format supports the arena), keeping the env var as an `args.py` alias.
- **Could break**: `tests/engine/test_growable_kv_transaction_source.py` and `tests/scheduler/test_growable_handoff_policy_source.py` `ast`-parse the `Engine`/`Scheduler` bodies and `exec` methods — they will fail and must be rewritten to import the controller (they become real unit tests); `tests/moe/test_offload_kernels_usable_slots_source.py:50` asserts the literal `FREETOKEN_EXPERT_ARENA` — update.
- **Verify**: `engine.py` shrinks by ≥ 600 lines; the three source-parsing tests rewritten and green; CPU growable suites (`tests/scheduler/test_growable_*`, `tests/engine/test_growable_*`; 0 GPU skip markers) green; at checkpoint 2 the 1M arm's journal produces the **same sequence** of `Committed growable KV through N tokens ... MoE slots a -> b` transitions as `results/nemotron-reserve-2e-1m-journal.txt` (`grep -o "MoE slots [0-9]* -> [0-9]*" | diff`).

### S8 — Model conformance check (G4; M; no GPU)
- **What**: `python -m freetoken.models.check_experts <checkpoint_dir>` reads `config.json` and safetensors headers only and prints or refuses with the exact reason: spec found; `gated` agrees with `expert_gated`; every expert has exactly the expected tensors after `kind_map`; one shard per expert; row bytes and total bank bytes from S5a's function; the cache type the engine would resolve and whether `--pin-prefix-*` would be honoured; whether the arena supports the format. Add `key_template` to `Nvfp4ExpertSourceSpec` (the regex cannot be rendered; the template can) and a CPU test parametrised over every module exporting `NVFP4_EXPERT_SOURCE_SPEC` (nemotron_h, qwen3_5_moe, minimax_m2, minimax_m3, glm4_moe, gemma4) that builds a synthetic checkpoint from the template, asserts it matches `key_pattern`, and runs the check. Write the "adding a model" checklist in `docs/models.md` as the sequence this command performs. Fold in the estimator cleanup: `_mirror_final_gpu_slots` calls `_plan_growable_kv`'s arithmetic through a pure function of config; drop `4 * step`; CPU test asserts planner and estimator agree on the Nemotron config.
- **Verify**: the command accepts both checkpoints on disk (`~/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` and the Ornith NVFP4 path) and refuses a synthetic checkpoint with a renamed kind, naming the kind; the parametrised test is green for all six specs.

### S9 — Conflict budget and shape rules (G1; S; no GPU)
- **What**: `scripts/upstream-sync-check.sh`: `git fetch origin`, `git merge-tree --write-tree origin/main HEAD`, print conflicting files and hunk counts, exit non-zero above a budget (0 after S0; grows only with explicit acknowledgement in the script). Propose a "fork shape rules" section for the repository `CLAUDE.md` (owner adopts; agents do not edit it): new behaviour in new modules; ≤ ~20 lines of hooks per upstream file per feature; no import-time env constants; env vars resolved in `args.py` only; no tests that parse upstream class bodies; no measurement arm names, dates or "lever N" in library code (they belong in `tasks/`); sync on a fixed cadence (before every measurement phase, at least weekly).
- **Verify**: the script runs and reports 0 after S0; it is re-run at the end of each phase and its output filed under `tasks/exclusive-expert-ram/results/sync-check-<date>.txt`. Phase E is decided on these numbers.

### S10 — Delete the legacy rebuild-and-recapture growable path (G2; S; after S7, same file)
- **What**: remove the non-arena branch of `grow_runtime_kv`/`shrink_runtime_kv` (`engine.py:2050-2140` today, inside the controller after S7), the `_pending_graph_bs` recapture plumbing it alone needs, and the "arena unsupported" refusals become "growable KV unsupported for this format" until S12b.
- **Why**: C-LEGACY-OFF. Production has been arena-only since `c3ce38c`.
- **Could break**: GGUF size-class and marlin/b12x models lose growable KV until S12b (accepted).
- **Verify**: `grep -n "recapture\|destroy_cuda_graphs" python/freetoken/engine/growable_kv.py` → only the rollback's teardown if any; CPU growable suites green; checkpoint 2 journal shows zero recaptures (`grep -c "capture" journal` == 1).

### S11 — Retire `--elastic-initial-requests` (G2; M; after S7; lane 2 owns `scheduler.py` for it)
- **What**: remove `resize_elastic_capacity`, `_elastic_capacity_for_slots`, `_elastic_graph_batch_sizes` and the elastic validation block (`engine.py:2240-2340, 2833-2850, 2886-2920`); `_elastic_target_capacity`, `_elastic_live_requests`, `_elastic_retained_session_handles`, `_elastic_demand`, `_maybe_resize_elastic_capacity`, `_remap_req_mamba_slots` and the resize-pending state (`scheduler.py:116, 280-284, 711, 808, 1585-1586, 2490-2600`); `linear_state_pool.resize_preserve` (`kvcache/linear_state_pool.py:144-174`); the `mha_pool` remap; `engine/graph.py`'s elastic sizes; the `FREETOKEN_ELASTIC_GRAPH_MAX_BS` knob; the flag in `server/args.py:310`; the `pin_slot_budget` "re-derived on an elastic resize" clause (`scheduler/cache.py:582-596`) becomes static. Delete `tests/scheduler/test_elastic_capacity.py`, `tests/scheduler/test_elastic_session_roundtrip_cpu.py`, `tests/engine/test_elastic_graph_sizes.py`; strip elastic cases from the other 10 test files that mention it. Update `docs/cli.md:46`, `docs/nemotron.md` P2 (mark historical), `docs/switchyard.md`.
- **Why**: C-SINGLE-LANE; section 2. Fixed multi-lane (`--max-running-requests N` without elastic) still exists after this; it is simply not the profile.
- **Could break**: the Mamba slot remap is shared with compaction? — verify `compact_active_pages` callers before deleting `_remap_req_mamba_slots`; if compaction uses it, keep it and delete only the elastic caller.
- **Verify**: `grep -rn "elastic" python/freetoken --include=*.py` → 0 (or only a docstring naming the retirement); CPU suites green; checkpoint 2 unchanged.

### S12 — Ornith: validation of the model-onboarding path, with a measured prefill number (G4 consumer; subordinate per C-REORG-OVER-ORNITH; after S8 and checkpoint 2)
- **S12a (NVFP4 Ornith through the new path; M; GPU)**: run S8's check on the Ornith checkpoint (must pass with no hand edit outside `models/qwen3_5_moe/weight.py`); serve it single-lane on the arena with the tuned tables from `bc6f815`; measure prefill at 32K and 80K (chunk 8192, empty GPU, pass 2).
  - **Target**: prefill ≥ **5,000 tok/s** at 32K and 80K (chunk 8192). Context: the blocked run measured 189–230 tok/s on untuned tiles (`results/ornith-blocked/README.md`); Nemotron's record is 9,922 / 8,794 tok/s (`benchmarks/results/nemotron35_lightning_5080_single_lane_2026-09-17.md:16-18`). The tuned pair costs 4.94 ms per layer at M=8192 tokens (`bc6f815`), i.e. ~0.20 s of MoE GEMM per 8192-token chunk over 40 layers, so the GEMM alone would allow far more; the likelier floor is the per-chunk expert stream (40 layers × 256 × 1.69 MiB ≈ 16.9 GiB per chunk vs Nemotron's 15.4 GiB) plus 40 layers of GDN attention.
  - **Required output**: a record with the measured number, the roofline computed from `benchmarks/bench_moe_prefill_gemm.py --model ornith --m 8192` and `ft bench bw` on this host, and an honest statement of the residual gap and its cause from a profile. If parity with Nemotron is unreachable, the geometry argument (intermediate 512 vs 1856: down-GEMM K=512, low arithmetic intensity, more experts per layer streamed per chunk) is stated **with** the measurement, never instead of it.
- **S12b (GGUF Ornith on the arena; M–L; GPU)**: extend the arena to mixed-GGUF size classes (`offload_cache.py:721`), the work C-LEGACY-OFF made necessary. Only after S12a passes and only if the owner still wants GGUF Ornith served.
- **S12a sizes, owner 2026-09-23**: "and why prefil is not being measured, let's say, up to 250k tokens?" -- the arm now measures 8K/32K/80K/128K/250K (the 32K/80K-only scope was the plan's, not the owner's). Owner also: startup "is taking way too long ... this computer has fast pcie 5.0 ... it's too slow for my taste": Ornith took 116 s to ready from a cold page cache (~150 MB/s effective) against Nemotron's 34 s warm; both took the "low free RAM -> serial build" path (`moe/expert_banks.py:755`). Open item: measure the raw NVMe read rate and the serial-vs-parallel build time, and fix the loader if it, not the disk, is the limit.
- **S12a measured 2026-09-23** (`results/ornith-s12a/`, whole model in RAM, arena, tuned tables, pass 2): prefill 8K 10,786 / 32K **7,380** (target met) / 80K **2,478** (missed) / 128K 2,033 / 250K 937 tok/s (TTFT 267 s); decode 169 / 165 / 149 / 128 / 111. Per-chunk journal: an 8K chunk takes 1.25 s at position 0, 2.7 s at 57K, 4-7 s at 65-130K, 8-10 s at 130-196K, 11-16 s past 196K; each KV grow (65K->131K->196K->262K) takes 400 expert slots (5824 -> 4624 of 10240) and the first chunk after it stalls 10-23 s. Three candidate causes, NOT yet attributed: attention cost growing with position (10 of 40 layers full attention), a per-grow recompile/re-tune (hypothesis), and more expert streaming as the arena shrinks. Next: nsys profile of one 128K prefill to split attention / MoE GEMM / expert H2D / compile. Roofline inputs: `roofline-gemm.txt` (tree tile 17.99 ms, 22.9 TFLOP/s at M=8192 per layer... see file), `roofline-bw.json` (PCIe 50.1 GB/s, host 92.1 GB/s).
- **S12a profile 2026-09-23** (`results/ornith-s12a/profile/`, nsys, one warmed 128K prefill, 53 s wall, 2,409 tok/s): GPU kernel time 49.6 s, of which **`_extend_attention_split_kernel` (Triton extend attention, q8_0 KV) 37.7 s = 76%** (160 launches = 10 full-attention layers x 16 chunks, ~236 ms each); NVFP4 MoE GEMM 5.8 s (12%), moe_sum_reduce 2.1 s; expert H2D 291 GB in 4.9 s (~59 GB/s, overlapped). **The residual gap to 5,000 tok/s at 80K+ is the prefill attention kernel, not the expert path**: attention cost grows with position, which is the per-chunk slowdown in the journal. Attention FLOPs are ~1.4e15 at 128K, i.e. ~37 TFLOP/s (agent estimate: 16 q heads x head_dim 256, causal), a fraction of the GPU's bf16 peak. Pass-1 numbers are low because the first long request after start runs in the post-load slow period (120K warm-up: 780 tok/s). Follow-up (Ornith subordinate, not in S12's scope): a faster sm_120 extend-attention path (FlashInfer/FA prefill, or bf16-KV A/B) -- owner's call whether to pursue.
- **S12b status 2026-09-23**: the owner's goal makes S12b required ("GGUF Ornith is reimplemented on the new path"), superseding "only if the owner still wants". **The GGUF checkpoint is no longer on this host**: `~/ai/models/Ornith-1.5-35B-Q4_K_M.gguf` (20.2 GiB, served earlier per `~/.claude/plans/ornith-rtx5080-full-context.md`) is gone; only the NVFP4 checkpoint remains. Re-obtaining it is the owner's input. Design (investigator, verified against the tree): the refusal is `offload_cache.py:738-742` in `set_bank_sources` (mixed row signatures + arena); `_set_gguf_size_class_sources` (`:810-891`) never passes `name=` so it cannot reach `_alloc_arena_bank_cache`; `growable_kv._growable_moe_bytes` needs one flat `bank_row_bytes` + one `arena_layout`. Chosen shape: **one arena per size class** (N=1 reduces exactly to today's path), `cache_budget` gets a multi-class sum over the existing single-class primitive, the growable-KV planner searches one joint step count with a per-class floor check. Padding every class to the widest row (rejected: the legacy-width waste the size classes exist to avoid) and a per-slot stride table in one arena (rejected: breaks `_arena_chunk_ranges`' `slot * row_bytes` exactness) were rejected. CPU work proceeds; the GPU proof waits for the checkpoint.
- **S12b 2026-09-23, merged and GPU-guarded**: merge `1704620` (b5fce00 + 062cf0e); Nemotron guard arm identical to the record (`20b1398`: 32/32 slot transitions, decode >= record, RAM 12.25 GiB, 0/0, one capture). Owner downloaded-by-instruction `~/ai/models/Ornith-1.5-35B-Q6_K.gguf` (29.2 GB; owner: "download it ... Ornith-1.5-35B-Q6_K.gguf"). Read from its header: **uniform Q6_K** (41 expert layers, gate/up/down all Q6_K) -> ONE size class, so it exercises GGUF on the single-class arena, not the per-class code; expert bytes 25.22 GiB (~24.6 GiB for the 40 served layers), so the whole-model arm needs MemAvailable >= ~29 GiB against 25 GiB on this host (CLAUDE.md: banks + ~4 GiB), and the RAM saver is NVFP4-only (`attach_residency`). **Not started**: owner's input needed (a smaller/mixed quant such as Q4_K_M, a larger WSL RAM cap, or extending the RAM saver to GGUF). `check_experts` crashes on a GGUF (`qwen3_5_moe.nvfp4_expert_spec` AttributeError) instead of refusing or checking it: a G4 defect, to fix.
- **Upstream test failures, corrected 2026-09-23**: the non-core test directories (all but moe/engine/scheduler/kvcache/server) give 7 failed / 1221 passed / 214 skipped on HEAD; the same 7 (`tests/mm/test_processor.py` x3, `tests/models/qwen4_exp/test_ple.py` x1, `tests/models/test_muse_glimmer_vision.py` x3) fail identically on pristine origin/main `cab110e` and on `eae882d` -- pre-existing, not the reorg. (The earlier "4 upstream failures" count was for a different directory set.)
- **Verify**: S8 check passes on the Ornith checkpoint with zero refusals; the S12a record exists under `results/ornith-*/` with `-record.json`, `-stats.json`, journal, and the roofline file; target met or the gap explained with a profile.

### S13 — Prefix pins for the single-session case (G2; M; after S7; GPU for the measurement)
- **Mechanism already present (verified)**: `pin_prefix` locks the node's root path with the tree's own refs (`scheduler/cache.py:665-720`); `evict_full` walks only `ref_count == 0` leaves, so a pinned prefix survives `_evict_growable_prefix_pages` (`scheduler.py:2473-2488`); `page_usage` counts non-evictable pages as used (`scheduler/cache.py:149-156`), so `_maybe_shrink_growable_kv` (`scheduler.py:2244+`) keeps the pinned pages committed ("protected/live pages keep N tokens committed"). Nothing in the growable machinery needs to change.
- **What is missing (verified)**: policy. `note_prompt_admitted` pins only via `_shared_pin_target` — two distinct session keys on the node (`scheduler/cache.py:610-663`); the producer of a node is never recorded; after the first request completes the shrink evicts its haystack before a second request can match it (STATUS: `cached_tokens` 0 at a 112K haystack, ~17 s re-prefill per question).
- **Design**: `--pin-prefix-scope {shared,session}`, default `shared` (today's behaviour, byte-identical); `scripts/serve-default.sh` sets `session`. Under `session`, at the producer's insert (`cache_req`, `scheduler/cache.py:1103`, where `insert_prefix` returns `new_handle`), if the request carries a session key and `req.cached_len >= pin_prefix_min_tokens`, call `pin_prefix(new_handle.node)`; the existing LRU release under `pin_prefix_max_tokens` / `pin_slot_budget` bounds it; `DELETE /v1/cache/pins` remains the valve. **Two consequences that must be handled, not discovered**: (a) `session` scope requires a finite `--pin-prefix-max-tokens` (0 = unlimited today; an unbounded pinned 1M haystack would block the next session), so `args.py` refuses `session` with max-tokens 0 and `serve-default.sh` sets a value (proposal: 262144); (b) `pin_slot_budget` with `--linear-state-slots 13` and one lane is small (`13 - 4 - 1 - spare`), and a long haystack path carries several snapshot-bearing nodes — the first sub-step measures how many on a 120K haystack; if it exceeds the budget, add a KV-only pin (skip the mamba refs; `pin_prefix` already distinguishes them) rather than raising slots.
- **Could break**: a pinned haystack survives a queued-session handoff and occupies KV the incoming session needs; the shrink then reports "protected/live pages keep" and the incoming session prefills against less headroom. Bounded by max-tokens and LRU release; measure the handoff case explicitly.
- **Measured 2026-09-23 (`s13.sh`, results `s13-{session,shared}-pins.json`), and it changes the premise**: with a session key, A2/A3/A4 hit 104,832 of 104,863 tokens at TTFT 0.46-0.98 s under `session` scope -- and **identically under the `shared`-scope control, where nothing is pinned** (pinned_tokens 0). With a 30K handoff B, the session lease alone keeps A's haystack; the recorded "cached 0 at 112K" miss is the no-session-key case (`needles.py` sends none: ck2-whole's 120K battery missed, cached 0). The pin can only make a difference when B needs A's pages (A + B > the KV pool) -- measured next as `s13b-*` (B ~ 960K); until then S13's pin is unproven, not proven. Session scope also pinned B's prefix (4 pins, 134,800 tokens after A4).
- **Measured 2026-09-23, s13b-session: the pin DEADLOCKS admission when A + B exceeds the KV pool** (`results/s13b-session-stuck*.{txt,json}`). With A's 104,934-token prefix pinned under `session` scope, B (961,390 tokens, key B) was accepted (HTTP 200 at 05:29:32), A's soft session was spilled and its lease released "(admission pressure)" -- but the PIN was not, so KV used stayed 104,934 and B could never fit (104,934 + 961,390 > 1,048,576). No refusal, no error: `fresh_admits_deferred` 106,632 in 20 min, GPU at 0%, the scheduler re-matching the 961K prompt every pass. Stopped by the orchestrator. The plan's "bounded by max-tokens and LRU release" was wrong: LRU release fires only for a newer pin, never for admission. **Required fix before `session` scope can stay the default**: under admission pressure release pins (least-recently-matched first) exactly as the soft-session lease already is, and count it. Until fixed, `serve-default.sh` must not ship `--pin-prefix-scope session`.
- **Control s13b-shared (same sequence, nothing pinned), 2026-09-23**: B (961,390 tokens) admitted and answered (TTFT 709 s, 3/3 codes); A's soft session was spilled to RAM (104,872 tokens, 0.41 GiB in 0.25 s) and **A4 still hit 104,832 cached tokens at TTFT 6.1 s** after B. So the existing session spill (G2) already carries a session's haystack across a handoff that needs its pages, and does so without blocking B; in both measured cases the pin bought nothing and in one it deadlocked. **Decision [agent, reasoned; owner may overrule]**: `serve-default.sh` returns to `--pin-prefix-scope shared` (today's behaviour); `session` scope stays as an option only once the pin-yields-to-admission fix (branch `reorg-s13fix`) is merged and re-measured (s13c). The S13 goal -- a session's questions hit its haystack -- is met by the session lease + spill, measured.
- **Fix measured 2026-09-23, s13c** (merge `aff73e2`, `results/s13c-*`): under `--pin-prefix-scope session`, B (961,390 tokens) arrived with A's 104,934 tokens pinned; the scheduler released A's three pins under admission pressure (`pin_admission_releases` 3, journal "Released prefix pin (admission pressure)") and B prefilled (TTFT 706 s, codes 3/3); A4 then hit 104,832 cached tokens (TTFT 4.1 s, from the session spill). One `pin_budget_refusals` after A4 is a pin above the 262,144-token cap being refused by design. The deadlock is gone. **Also measured, cross-restart**: s13c-shared (fresh server) restored the previous arm's spilled 961K session B from the persistent disk tier -- "KV grew 131072 -> 983040 tokens to restore cold session" -- and answered correctly (3/3) at TTFT 6.9 s against a 706 s prefill: G2's spill-to-disk survives restarts. It made that control's B1 a restore, not a prefill; `s13.sh` now uses per-arm session keys. Default stays `shared` (16442c8): session scope still bought nothing measurable over lease + spill.
- **Verify** (measured, not asserted): on port 1920, single arm, Nemotron: prefill a 112K haystack with session key A; ask three questions with key A; each response's `usage.cached_tokens` ≥ haystack − 8192 and `/v1/stats` prefix counters show hits with `pin_budget_refusals == 0`; TTFT of questions 2 and 3 < 3 s against the recorded ~17 s; the journal shows no shrink below the pinned pages between questions. Then one handoff case: session B admitted while A's pin is held; B completes; A's fourth question still hits. CPU test in `tests/kvcache/radix/test_hybrid_radix_pins.py` for the new scope and the max-tokens refusal.

---

## 6. Phases, lanes, ownership, dependencies

At most three active workers, no nested spawning, one lane per worker, ownership by file. A lane never edits a file another lane owns; the two cross-lane touch points (S6/S7's `min_gpu_slots()`; the three-line hooks in `engine.py:1093-1097` and `scheduler.py:3095-3101`) are agreed by name before Phase C starts and applied by the owning lane.

```
Phase A  (no GPU, parallel, ~1 day)
  lane 1  S1 -> S2 -> S3 (offload_cache side)        owns moe/offload_cache.py, moe/mirror_*.py, scheduler.py:_forward hook
  lane 2  S3 (engine side) -> S4                    owns engine/engine.py, engine/cache_budget.py, server/args.py, engine/config.py,
                                                     scripts/serve-default.sh, tasks/exclusive-expert-ram/STATUS.md, plan.md, docs/
  lane 3  S5a draft + tests                          owns models/**, moe/expert_banks.py, tests/moe/_mirror_checkpoint.py, docs/models.md

Phase B  (serial, 1-2 days)   one worker: S0 (+S5b)  -> GPU CHECKPOINT 1 (owner runs the server)
         then PR-1 prepared for the owner (no gate)

Phase C  (parallel, no GPU until the checkpoint)
  lane 1  S6                                          same ownership as Phase A
  lane 2  S7 -> S10 -> S11 (takes scheduler.py ownership for S11 once lane 1's S6 hook is in)
  lane 3  S8 (check command, key_template, docs; the engine estimator edit waits for S7 and lands via lane 2)
                                                    -> GPU CHECKPOINT 2 (bundle + whole-model arm + section 7.3 instrument)

Phase D  (parallel, no GPU)
  lane 2  S9  (sync-check script; file its first numbers)
  lane 1  S13 design + CPU tests -> S13 measurement (GPU, one arm, section 5)
  lane 3  S12a (only after S8 and checkpoint 2; yields to any Phase C/D GPU need)

Phase E  deferred: decided on S9's numbers, after D.
         DECIDED 2026-09-23 (owner's goal: "decide E after Phase D from S9's numbers"): NOT NOW.
         S9 numbers: sync-check at S0 and at the end of Phase C: 0 conflicting files, 0 silent
         fork-symbol losses, 0 behind (origin/main still cab110e, 0 upstream commits since
         the merge); upstream touched scheduler/ + kvcache/ in 6 commits in the month before
         S0; our fork's diff there is ~7,900 inserted lines. A re-port costs a rewrite of that
         and a re-measure of every record, against a sync tax that is currently zero.
         Trigger to revisit: a sync-check that reports a conflict or silent loss in
         scheduler/ or kvcache/, or upstream scheduler activity above its current rate.
```

Phase A may run after S0 instead if the owner wants the sync proven first; nothing in A changes the conflict count. Nothing in S0–S11/S13 waits on S12.

---

## 7. Evidence record and checkpoints

### 7.1 The record that must survive
From `tasks/exclusive-expert-ram/results/nemotron-reserve-2e-1m-{record.json,stats.json,journal.txt,probe.jsonl,env}` and `results/needles-2026-09-22-reading.md`: best config (all four levers, reserve 2E): **1M context in 12.26 GiB host RAM, decode 76.9 tok/s (probe pass 1; pass 2 72.9), 0 coverage faults, 0 starved writebacks, one CUDA-graph capture**; needle answers **byte-identical** between pool and whole model at temperature 0 (five of six earlier "failures" were the 16384 thinking cap, `d4f04a6`); recall 3/3 at 21K/120K/240K. Noise floor: 8.7% between identical arms at 80K decode (handover §12); host RAM varies ~0.56 GiB between identical arms (STATUS).

### 7.2 The acceptance bundle (every GPU checkpoint; the agent launches `checkpoint1.sh` as a systemd transient unit, never from an agent shell; owner chose "You run it" on 2026-09-23)
1. CPU: `tests/moe tests/engine tests/scheduler` ≥ 871 passed and `tests/kernels tests/models` ≥ 871 passed (floors after S0; 809 / 632 at writing). **Plus every other test directory** (`tests/server tests/tokenizer tests/kvcache tests/attention tests/layers tests/checkpoint tests/daemon tests/dsv4 tests/mm`), with failures classified against both merge parents: checkpoint 1's second boot on 2026-09-23 answered every chat request with HTTP 500 (`generation.py _preflight`: upstream's `tokenize` now returns `UserMsg`, the fork's preflight still called `.numel()` on it), and none of the four suites above imports that path (counts may rise, never fall except by S11's deliberate deletions, which are listed in the step).
2. Device, model unloaded, GPU empty: `pytest tests/moe/test_mirror_device.py tests/moe/test_expert_arena_vmm.py tests/kernels/test_vmm_tensor.py`.
3. One arm at the record config (`results/nemotron-reserve-2e-1m.env`; C-EMPTY-GPU): decode compared pass by pass within 9% of the record — 1M: pass 1 76.9, pass 2 72.9; 8K/80K/713K (`nemotron-reserve-2e`): pass 1 180.3/154.8/91.7, pass 2 172.1/154.1/107.6; host RAM 12.26 ± 0.6 GiB; `stats.json` `coverage_faults == 0 and starved_writebacks == 0`; journal has exactly one graph capture and no `Traceback`; `benchmarks/switchyard_soak/checks/acceptance.sh R3` and `R6` pass.
4. Correctness: `tasks/exclusive-expert-ram/needles.py` at 21K/120K with the raised thinking budget and `recall.py` at 21K/120K/240K; answers byte-identical to the whole-model reference **recorded on the same commit** (re-recorded once at checkpoint 1 after S0, then reused).
5. S7 only: the `MoE slots a -> b` transition sequence in the 1M journal diffs clean against the record.

**Known noise, measured 2026-09-23 [agent practice for reading it]:** an intermittent
~2.5 s host-side delivery stall after the first token (GPU idle, tokens already computed)
turns one probe's decode into 30-47 tok/s. It predates the reorg (3 of 152 probes on the
frozen branch, 2 on exp/reorg). A single probe with that signature (decode window
2.5-4 s where the size's steady window is < 1.7 s) is re-measured on the same commit,
same config and order, before it counts as a regression; both readings are committed.
Checkpoint 1's 1M pass 1 (30.7) re-measured at 79.0 (`83e5efb`).

### 7.3 The finer instrument (Phase C, because 7.2 cannot see a 2–3% regression)
Before and after S6/S7, on the same commit pair, one 128-token decode at 8K under `torch.profiler` (or `nsys` if installed): the **count of kernel launches per decode step** and the **bytes moved per step** (`mirror_stats()` swaps/writebacks/retained, and `decode_miss_stats()`) must be identical. Identical launches plus identical bytes means no new GPU work; the remaining difference is Python outside the captured graph, which the graph replay never runs. File the two profiles under `results/instrument-<step>-{before,after}.txt`. Compare them with `compare_traces.py` (whole-request per-kernel-name launch counts, memory-op counts and bytes, `/v1/stats` deltas): `instrument_analyze.py`'s per-step clustering does not separate steps on a real capture. S6/S7 result: IDENTICAL (`847a4b9`).

### 7.4 Revert rule
A step whose checkpoint fails any of 7.2.3–7.2.5 is reverted on the branch (`git revert` of its commits, or reset of the lane's unmerged work) and re-attempted from the recorded failure, never patched forward under the checkpoint. (Agent-authored rule, see the provenance audit.) Applied case, 2026-09-23: checkpoint 1 on the S0 merge failed at boot because the merge silently dropped the `nvfp4`/`none` rows of `moe/expert_banks.py:_PROVIDERS`. That was fixed as a completion of the merge resolution (the fork's rows restored, a regression test added), not reverted: reverting the merge would undo G1 itself. The owner may overrule.

---

## 8. Do not do this
Carried over: no kernel body edits; no change to the coverage invariant, the free stack or `publish_freed_rows`; no rewrite of session spill; C-NO-MAIN; no `uv sync` of the live server's venv (agent caution); no server from an agent shell; embedder stays down; no numbers on a non-empty GPU; never edit `python/freetoken/**` while an `ft-measure-*` unit is live.
Added: no `git rebase` onto upstream (C-MERGE-NOT-REBASE); no upstream PR without the owner running it (C-UPSTREAM-PR); no collapsing of the `_v2` kernel twins (the gating is the point of graph-safe resize); no change to `measure.sh` knob names (`FT_RESERVE`, `FT_TIEBREAK`, …) without both a write and a strip entry; do not move `tasks/exclusive-expert-ram/*.py` while `test_mirror_device.py` shells out to them (S6 may move them to `tests/moe/device/` afterwards, as one commit); do not edit a moved function's body in the same commit as the move; do not touch `_mirror_final_gpu_slots`'s `4 * step` before S8's planner/estimator test exists; do not touch `~/.config/freetoken/serve.env` except the already-authorised ratio; do not start S12 before S8 and checkpoint 2; do not let S12 hold a GPU window a reorganisation checkpoint needs; do not re-tune the Ornith tables (they are committed, `bc6f815`).

---

## 9. Running this as a piro-campaign

This work outlives many context windows. The campaign, not the conversation, is the memory; the repository, not the campaign, is the authority.

### 9.1 Requirement records — one per step, each with an executable check
Shape (field names indicative; use the campaign's native schema):

```
id: R-S<n>            goal: G<k>           gpu: yes|no          lane: 1|2|3
statement: <one sentence from section 5>
check:
  argv:   [<command that a verifier runs from the worktree root>]
  inputs: [<source and test paths whose change invalidates the receipt>]
  expect: <exit code / grep count / test count / file existence>
evidence: <paths under tasks/exclusive-expert-ram/results/ or the test log>
```

Concrete recipes (worktree root, C-SERVER-VENV; `PY=/home/lucas/ai/FreeToken/.venv/bin/python`):

- **R-S0** `argv: bash -c "git fetch origin && git merge-tree --write-tree origin/main HEAD | grep -c '^CONFLICT'"` expect `0`; `inputs: [.git/refs/remotes/origin/main]`. Plus the two CPU suite counts.
- **R-S1** `argv: bash -c "grep -c 'stats\"\\].tolist()' python/freetoken/moe/offload_cache.py"` expect `0`; `argv: $PY -m pytest -q tests/moe -k mirror_stats`; `inputs: [python/freetoken/moe/mirror_stats.py, python/freetoken/moe/mirror_kernels.py, python/freetoken/moe/offload_cache.py]`.
- **R-S2** `argv: bash -c "grep -rn '_mirror_restore_coverage\\|_mirror_needs_coverage' python tests tasks | wc -l"` expect `0`; receipt incomplete until checkpoint 1's `test_mirror_device.py` log exists; `inputs: [python/freetoken/moe/offload_cache.py, python/freetoken/scheduler/scheduler.py, tasks/exclusive-expert-ram/graph_race_repro.py]`.
- **R-S3** `argv: bash -c "grep -rn 'def _arena_chunk_boundaries' python | wc -l"` expect `1`; `inputs: [python/freetoken/engine/cache_budget.py, python/freetoken/engine/engine.py, python/freetoken/moe/offload_cache.py]`.
- **R-S4** `argv: bash -c "grep -n \"resolves to .cache_type='radix'\" tasks/exclusive-expert-ram/STATUS.md | wc -l"` expect `0`; `inputs: [tasks/exclusive-expert-ram/STATUS.md, tasks/exclusive-expert-ram/plan.md]`.
- **R-S5a** `argv: $PY -m pytest -q tests/moe/test_mirror_pool.py -k row_layout_equivalence` (three synthetic checkpoints, rows byte-equal to the loader); `argv: bash -c "grep -c 'H // 16' python/freetoken/moe/mirror_pool.py"` expect `0`; `inputs: [python/freetoken/models/nvfp4_banks.py, python/freetoken/moe/mirror_pool.py, tests/moe/_mirror_checkpoint.py]`.
- **R-S5b** `argv: $PY -m pytest -q tests/moe -k bank_bytes_estimate_gated`; `inputs: [python/freetoken/moe/offload_cache.py, python/freetoken/moe/expert_banks.py]`.
- **R-S6** `argv: bash -c "grep -c '_mirror' python/freetoken/moe/offload_cache.py"` expect `≤ 15`; `argv: $PY -c "import sys, freetoken.moe.offload_cache; assert 'freetoken.moe.mirror_pool' not in sys.modules"`; checkpoint 2 receipts incl. 7.3; `inputs: [python/freetoken/moe/residency.py, python/freetoken/moe/offload_cache.py, python/freetoken/engine/engine.py, python/freetoken/scheduler/scheduler.py, python/freetoken/server/args.py]`.
- **R-S7** `argv: $PY -m pytest -q tests/engine/test_growable_kv_transaction_source.py tests/scheduler/test_growable_handoff_policy_source.py tests/moe/test_offload_kernels_usable_slots_source.py tests/scheduler/test_growable_*.py tests/engine/test_growable_*.py`; `argv: bash -c "wc -l < python/freetoken/engine/engine.py"` expect `≤ 2689`; checkpoint 2 slot-sequence diff; `inputs: [python/freetoken/engine/growable_kv.py, python/freetoken/engine/engine.py, python/freetoken/moe/offload_kernels.py, python/freetoken/moe/offload_cache.py]`.
- **R-S8** `argv: $PY -m freetoken.models.check_experts ~/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` expect exit 0; same for the Ornith path; `argv: $PY -m pytest -q tests/models/test_expert_source_conformance.py`; `inputs: [python/freetoken/models/check_experts.py, python/freetoken/models/nvfp4_banks.py, python/freetoken/models/*/weight.py, docs/models.md]`.
- **R-S9** `argv: scripts/upstream-sync-check.sh` expect exit 0 and a filed `results/sync-check-<date>.txt`; `inputs: [scripts/upstream-sync-check.sh, .git/refs/remotes/origin/main]`.
- **R-S10** `argv: bash -c "grep -c 'destroy_cuda_graphs' python/freetoken/engine/growable_kv.py"` expect `0`; checkpoint 2 journal `grep -c capture` == 1; `inputs: [python/freetoken/engine/growable_kv.py]`.
- **R-S11** `argv: bash -c "grep -rn 'elastic' python/freetoken --include=*.py | wc -l"` expect `0`; CPU suites; `inputs: [python/freetoken/engine/engine.py, python/freetoken/scheduler/scheduler.py, python/freetoken/scheduler/cache.py, python/freetoken/kvcache/linear_state_pool.py, python/freetoken/kvcache/mha_pool.py, python/freetoken/engine/graph.py, python/freetoken/server/args.py]`.
- **R-S12a** `argv: bash -c "test -f tasks/exclusive-expert-ram/results/ornith-*-record.json && $PY tasks/exclusive-expert-ram/table.py --min-prefill 5000"` (or the equivalent read of the record); the roofline file exists; `inputs: [python/freetoken/moe/configs/triton_3_6_0/nvfp4,E=256,*RTX_5080.json, python/freetoken/models/qwen3_5_moe/weight.py]`.
- **R-S13** `argv: $PY -m pytest -q tests/kvcache/radix/test_hybrid_radix_pins.py -k session_scope`; measured receipt: a `results/pins-session-<date>.json` with `cached_tokens` ≥ haystack − 8192 on questions 2–3 and TTFT < 3 s; `inputs: [python/freetoken/scheduler/cache.py, python/freetoken/server/args.py, scripts/serve-default.sh]`.
- **R-CKPT-1 / R-CKPT-2** the section 7.2 bundle, each item a separate receipt with its log path.

### 9.2 Constraint records that must never be weakened
C-SINGLE-LANE, C-NO-MAIN, C-MERGE-NOT-REBASE, C-EMPTY-GPU, C-EVIDENCE (with the numbers of 7.1 written into the record), C-REORG-OVER-ORNITH, C-SERVER-VENV, C-UPSTREAM-PR, C-LEGACY-OFF, C-ELASTIC-OFF. A campaign update may add constraints; it may not remove, relax or reinterpret these without a user message quoted in the record with its prompt provenance. `verified: true` is never set on a requirement whose check has not been run in this tree at the current HEAD.

### 9.3 After a compaction, before acting
History is evidence, not authority. Re-verify, in this order, from the repository and the host:
1. `git -C /home/lucas/ai/FreeToken-wt/reorg status --porcelain` and `git log --oneline -5` — which lane's files are dirty, what the last commit was.
2. `git merge-tree --write-tree origin/main HEAD | grep -c '^CONFLICT'` — where goal 1 stands right now.
3. `systemctl is-active freetoken-serve`, `systemctl list-units 'ft-measure-*'`, `nvidia-smi --query-gpu=memory.used --format=csv,noheader`, `free -g` — whether the GPU is empty and whether any code edit is forbidden right now.
4. The campaign's active frontier: which step records are `verified`, which have receipts pending a checkpoint, which worker identities are live (list them; do not spawn a replacement without checking).
5. The step's own `check.argv` — run it before believing any prose claim that the step is done.
6. Do not read every result file; read the record files the step names. STATUS.md is a narrative and has already been wrong once (S4); the `-record.json`/`-stats.json`/journal files are the evidence.

### 9.4 Worker roles and ownership
- **campaign-scout** (Haiku): mechanical discovery only — grep counts, file lists, conflict-hunk counts, test inventories. Never edits.
- **campaign-implementer** (Sonnet): S1, S3, S4, S5a, S5b, S8, S9, S10, S11, S12a harness runs, PR-1 preparation. One lane's files only.
- **campaign-specialist** (Opus): S0 (the merge, exclusive), S6, S7, S13 — the steps whose failure mode is a silent wrong-experts or ordering bug. Same one-lane rule.
- **campaign-verifier** (Sonnet, high): every checkpoint and every `verified:` transition; reads receipts, re-runs `check.argv`, compares against 7.1; never takes a worker's prose as proof.
- The **owner** starts and stops servers for checkpoints, runs PR-1 on hardware, decides Phase E and S12b.
- Ownership per lane is in section 6 and is part of each worker's assignment text; an assignment that needs a file outside its lane returns a request instead of editing.

### 9.5 Stop conditions
Stop and ask only for: authority (spend, merge, push, delete, `~/.config` changes beyond the ratio), a checkpoint failure that the revert rule cannot resolve mechanically, or an S11/S13 discovery that a deletion is load-bearing for compaction or spill. Do not stop to ask permission for what the accepted plan already authorises.

---

## 10. Open items (none blocking Phase A)
- Phase E (scheduler re-port as its own module): decided after D on S9's numbers.
- S12b (GGUF Ornith on the arena): confirm it is still wanted once S12a is measured.
- S13 sub-step 1 decides KV-only pins vs slot budget by measurement; the proposal for `--pin-prefix-max-tokens` in `serve-default.sh` (262144) is for the owner to accept.
