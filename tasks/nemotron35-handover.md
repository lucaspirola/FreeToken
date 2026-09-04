# Nemotron 3.5 Lightning on FreeToken — handover

State as of 2026-09-04 ~14:40, HEAD 1f2de67+ on `main` (not pushed). Read this, then
`tasks/nemotron35-plan.md` (spec + decisions), `tasks/todo.md` (checklist), `tasks/lessons.md`
(rules), `docs/nemotron.md` (profiles), `docs/switchyard.md` (router contract + e2e harness).
Memory notes: host embedder service, host-OOM rules, plan pointer.

## One-line status
Model serves correctly on the RTX 5080 with the Triton Mamba-2 SSD kernels and NVFP4 experts;
Switchyard's text-based escalation contract is met (contract 12/12); session residency
(spill on demand, capacity/age retention, restart persistence, RAM prefetch, partial-prefix
restore) is implemented and unit-tested. Three things are NOT finished (below).

## Done (commits 80f2838..e50bc22, ~26 commits)
- Phase 1 bring-up, Phase 2 kernels (SSD prefill/decode, b12x relu2, Triton MoE + dense tuning,
  cache study → Triton default, LFU for 16-way), Phase 3A–3G (wire contract, JSON mode,
  sessions/parsers, soak harness, residency policy, prefetch, partial restore + prefill-time
  state capture), slot-reclaim crash fix (+ /health 503, bounded shutdown), MTP NO-GO.
- Results: benchmarks/results/nemotron35_lightning_5080_{,mamba2_,cache_study_,switchyard_}2026-09-04.md
- Numbers: 131K prefill ~3,000 tok/s, decode 63–73 tok/s single stream, 16-way aggregate
  ~168 tok/s with LFU; 131K needle passes (chat endpoint); spill 2.7–3 GiB/s, RAM restore
  5–8 GiB/s, NVMe restore ~1.3 GiB/s.

## Not finished — do these in order, ONE GPU job at a time under scripts/gpu_lock.sh
1. **262K recall — CLOSED 2026-09-04, root cause fixed.** The Mamba-2 prefill scan floored
   the discretized timestep at `dt >= config.time_step_min` (1e-3). `time_step_min` is HF's
   *initializer* range for `dt_bias`, not a runtime bound; the floor caps every head's memory
   horizon at `1/(|A|*1e-3)` tokens. `dt_limit=(0.0, inf)` — vLLM's value, and what llama.cpp
   and FreeToken's own decode kernel always did — turns 147,456 and 262,144 @ depth 0.52 from
   FAIL to PASS at identical TTFT. Fix: `models/nemotron_h/config.py::_dt_floor`
   (`FREETOKEN_NEMOTRON_DT_MIN=<float>` restores a floor for A/B) + 3 tests in
   `tests/models/test_nemotron_h.py`. Write-ups:
   `benchmarks/results/nemotron35_lightning_5080_262k_{rootcause,crossengine}_2026-09-04.md`.
   The bisect's "model/quant limit" verdict and its "gate mid-depth needles at depth <=0.1"
   acceptance bar are **retracted**; retest the 262K/524K rows in the cache study and the 1M
   gate against the fix. Perf tickets from the bisect still open: `decode_launch_config` has no
   Nemotron head-shape branch (kv_splits=8 fallback: 16 CTAs on 84 SMs at 262K); Triton KV
   loaders widen slot ids to int64 on store but not load (safe here, a ceiling at head_dim 256).
   Exonerated with evidence along the way: the FP8 W8A8 Mamba in/out projections (11 of 46
   saturate their calibrated `input_scale`, but by the same 1.8e-5 clipped fraction at the
   passing 131K and the failing 147K — see `FREETOKEN_DEBUG_FP8_ACT_STATS`) and the whole
   NVFP4 path (W4A16 end to end, no activation quantization anywhere).
2. **16-way Switchyard soak — RERUN 2026-09-04 against `fad1fc4`. The KV fatal is CLOSED;
   the soak's own gate needs one more fix (uncommitted, in the tree).** Write-up:
   `benchmarks/results/nemotron35_lightning_5080_switchyard_soak_2026-09-04.md` §"Rerun
   against fad1fc4". 50 min of load at `befcde6`, 1,941 successful requests, **0**
   `committed_pages_required`, 0 `LinearStatePool exhausted`, 0 tracebacks, `/health` ok on
   every sample, clean 4 s shutdown, GPU 0 MiB. Soak A/`switchyard/passthrough` 20 m PASS
   (1,219 req / 0 err; 161 tok/s aggregate at 16-way, 10.1 per stream) and soak B/resilience
   10 m PASS (448 / 0) — both routes the previous attempt never reached. Soak
   A/`switchyard/stage` 20 m **FAILED**: 4 long-context `timeout`s (600 s client limit) and
   4 STALLED intervals, no crash — **prefill starvation**, one 512-token lane per pass with
   half the KV pool free. Cause: `fad1fc4`'s chunk cap subtracts `reserved_size` (each
   admitted req's WHOLE remaining prompt) although the batch only allocates its *chunk*, so
   one long continuation reserves the pool away from its peers (CPU-only A/B, no GPU:
   6/6 lanes at `c4486b6`, **2/6** at `befcde6`, 6/6 with the fix —
   `scratchpad/soak2/lane_ab.py`). **Fix is in the working tree, NOT committed**:
   `PrefillAdder.reserved_pages` (decode in-flight + one page span per admitted chunk) is
   what the cap charges against — strictly tighter than `committed_pages_required`, so the
   fatal stays closed — plus a sixth case in
   `tests/scheduler/test_chunked_prefill_kv_backpressure.py` (fails on the old cap).
   Verified end to end: the stage route re-run with it **PASSES** — 471 req / 0 err vs
   278 / 4, 1 STALLED vs 4, p95 201 s vs 393 s, mean 2.37 lanes/prefill batch vs 1.56
   (`scratchpad/soak3/`). Still open (tickets in §R7): `chunk_limit = token_budget //
   waiting` divides the 8 K budget by QUEUE DEPTH not by admissible lanes (pre-existing; it
   is what leaves stage p95 at 200 s); no `/v1/stats` counter for deferred/capped chunks, so
   engagement can only be shown circumstantially; `_try_allocate_one` reserves a fresh
   admit's whole prompt, capping long-context concurrency at 1-2 in a 262 K pool. Also
   measured: a blocked prefill does **not** spin — the busiest process holds median 106 %
   CPU in every phase, including 60 s intervals that completed 0 requests. Drivers:
   scratchpad/soak2/{run.sh,serve.sh,sample.sh,analyze.py,lane_ab.py} and scratchpad/soak3/
   under /tmp/claude-1000/-home-lucas-ai-FreeToken/f4e2e9e3-.../scratchpad/.
3. **1M gate — CLOSED 2026-09-04, all four criteria PASS.** Write-up:
   `benchmarks/results/nemotron35_lightning_5080_1m_sessions_2026-09-04.md`. One session grown
   to **1,039,989 tokens** (8 turns × 130K, needle recalled at every length, twice); demand
   spill of the resident 1M session **3.53 GiB to NVMe in 2.980 s (1.18 GiB/s)**; a **new**
   `ft serve` process adopted the checkpoint (`adopted 1 checkpoint(s)`) and its next turn
   **restored 1,040,020/1,040,020 tokens from disk in 2.681 s (1.32 GiB/s)** — 9.8 s wall
   against the 1,861 s of prefill that built the prefix, with a byte-identical (correct)
   answer across the restart. Capacity/age eviction verified at a 1.6 GiB cap: the third spill
   evicted the older of two candidates by `last_used_at`, survivors still restored (0.255 s),
   and a record larger than the whole cap is refused rather than evicting the world.
   262K/524K needles re-run through `/v1/chat/completions` at depth 0.50 after the `dt` fix:
   **262,160 PASS** (1,925 tok/s prefill, 56.3 decode) and **524,304 PASS** (1,064, 34.5) —
   the cache study's "~131K–256K coherent ceiling" caveat is retracted.
   Notes/tickets from the run (§6 of the write-up): `_restore_cold_session` uses
   `session.spill` without checking `.valid`, so a capacity eviction is reported as
   "client tokens diverge" (one-line fix, not applied — scheduler.py is another agent's file);
   a *resident* session is never checkpointed, so a restart loses it (spill-on-shutdown flag
   would fix); `_evict_one_lru` can evict the record the pending admission is about to restore;
   `--session-spill-ram-gb 0` is what forces the NVMe tier for this test (the 4 GiB default
   keeps a 3.5 GiB checkpoint in RAM, where a restart destroys it). Drivers:
   scratchpad/1m2/{serve.sh,drive.py,trigger.py,hold1b.sh,hold4_lru.sh,hold3_evict.sh,
   hold2_needles.sh} under /tmp/claude-1000/-home-lucas-ai-FreeToken/f4e2e9e3-.../scratchpad/.
4. **Phase 3H hidden-state export — CLOSED 2026-09-04, parity PASS.** All 52 exported
   layers match transformers' own `NemotronHBlock` stack at cosine **>= 0.998840** on the
   mean-pooled residual (gate 0.99); median 0.999760. Write-up:
   `benchmarks/results/nemotron35_lightning_5080_hidden_states_parity_2026-09-04.md`.
   Two things had to be fixed in `benchmarks/probe_hidden_states_parity.py` first (only
   file changed, uncommitted): (a) its `AutoModelForCausalLM.from_pretrained` reference is
   impossible on this release — modelopt MIXED_PRECISION, which transformers 5.15 has no
   quantizer for, `backbone.*` vs `model.*` names, per-expert NVFP4 tensors vs a fused 3-D
   parameter (400 missing / 18 486 unexpected keys), and 58.8 GiB dense bf16 against a
   34 GiB host. It now builds the model on `meta` and streams one block at a time
   (dequant on the sibling scales, ~3.5 GiB VRAM, ~10-22 s for the whole forward), with
   the per-block forward hook recording `residual + mixer` directly. (b) `--capture-only`
   / `--artifact <path>` split the run into two phases (server up, then server stopped),
   since the served model and the reference cannot be resident together.
   **The reference needs `--reference-dt-min 0.0` (the default).** transformers hard-codes
   the same 1e-3 `dt` floor that item 1 identified as a bug; leaving it in fails 12
   shallow layers (worst 0.9406 at layer 3) — an independent confirmation of item 1.
5. Ticket: `--kv-grow-step-tokens` + `--nvfp4-backend flashinfer` crashes (VMM int32 bank).
6. Ticket: `_maybe_shrink_growable_kv` evicts all unlocked prefixes before checking whether a
   shrink is possible (wipes the prefix cache at idle above the initial KV step).
7. Ticket: tests/moe/test_prefill_hit_d2d.py order-dependent flake.

Scratchpad root (survives Claude restarts, not WSL restarts):
/tmp/claude-1000/-home-lucas-ai-FreeToken/af23ede4-e8ad-4c8d-8b38-c8be515d8870/scratchpad/

## Host rules (all learned the hard way)
- `systemctl --user stop piro-board-embedder.service` before GPU work (Restart=always; holds 4–10 GB).
- Host RAM is the constraint (34 GB WSL): ANY job that loads the checkpoint runs under
  scripts/gpu_lock.sh (refuses < 22 GiB available, 4 h cap, oom_score_adj 1000, reaps workers).
  Claude/tmux are protected by the root timer protect-terminal-oom.timer (oom_score_adj −900).
- Max two subagents at once; never one that ends its turn while its GPU job continues.
- Never `git stash`/`commit -a`; implementers use worktrees; kill workers by venv path.
- FREETOKEN_PIN_BUDGET_GB=17 pins all expert banks (no --moe-pageable-gpu); ratio 0.85;
  8K chunks; q8_0 KV + Triton attention; `--nvfp4-backend` auto→triton; LFU for 16-way.
- Needle gate goes through /v1/chat/completions, digit-free filler; never grade raw SSE.

## Two stale worktrees to ignore/remove
.claude/worktrees/agent-a45f827ae98e76526 and agent-a4ce5e26d2ccafdb6 predate this effort.
