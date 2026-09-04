# Nemotron 3.5 Lightning on FreeToken — handover

State as of 2026-09-04 ~22:30, HEAD 508ea32 on `main` (not pushed). Read this, then
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
2. **16-way Switchyard soak — STILL OPEN. `81ab30e` FAILS both routes; the last tree that
   passed is `befcde6` + the §R6 `reserved_pages` fix.** Write-ups in
   `benchmarks/results/nemotron35_lightning_5080_switchyard_soak_2026-09-04.md`:
   §"Rerun against fad1fc4" (the KV fatal, closed) and §"Run against 81ab30e" (this verdict).
   - History: `fad1fc4` closed the `committed_pages_required` fatal but starved the stage
     route by charging its chunk cap in `reserved_size` (whole remaining prompts) instead of
     pages; the §R6 fix (`PrefillAdder.reserved_pages`) restored it — stage 471 req / 0 err /
     1 STALLED, mean 2.37 lanes per prefill batch.
   - **`81ab30e` (fresh admits gated on finishability + this chunk) regresses it: stage
     268 req / 15 timeouts / 7 STALLED, passthrough 720 req / 16 timeouts / 10 STALLED**, all
     failures `long-context`, p95 600 s on stage. The fatal stays closed (0
     `committed_pages_required`, 0 `LinearStatePool exhausted`, 0 tracebacks, 0 oversize
     warnings, `/health` ok on every sample, 3 s shutdown, GPU 0 MiB).
   - Root cause, from `py-spy dump --locals` on the wedged core process (§S5): the scheduler
     spends **52–53 % of each phase emitting no batch at all** (gaps of 492 s, 515 s, 624 s),
     looping `schedule_next_batch` → a full 118 K-token radix `match_prefix` per pending fresh
     candidate → refuse all → `None`. Every gap opens at `token usage: 1.00` with
     `#running-req` 0–2 and `#queue-req` 10–16. The finishability budget is
     `cache_manager.max_size - inflight_prefill_size`, i.e. **the whole pool**: it subtracts
     neither the KV held by requests already decoding nor the retained/locked session
     prefixes, so admissions keep arriving until the pool is full and then no lane can buy its
     next chunk and nothing can complete to free one. `_reclaim_for_blocked_prefill` fires on
     every `None` and returns False (`Cold restore ... failed (AssertionError('Eviction did
     not free enough space.'))`); the only escape is the session idle timeout minutes later.
   - While it *is* scheduling `81ab30e` is genuinely better: 4.71 lanes/prefill batch vs 2.37
     and 2,938 vs 1,877 prefill tok/s on stage. The fix direction is to bound the
     finishability budget by what the pool can actually give back, not to revert the lane win.
   - Do the next attempt as a CPU-only A/B first (`scratchpad/soak2/lane_ab.py` shape), then
     one 20 m stage soak. Drivers: `scratchpad/soak5/` (this run), `scratchpad/soak3/`,
     `scratchpad/soak2/` under
     /tmp/claude-1000/-home-lucas-ai-FreeToken/f4e2e9e3-.../scratchpad/.
   - Open tickets from this run are 8-11 below.
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
4b. **1M multi-needle recall — 2026-09-04, 5/8.** One 1,039,994-token prefill (TTFT
   **1,815 s**, 573 tok/s whole-prompt), then eight questions on the same chat prefix, each
   hitting the prefix cache for 99.9954 %+ of its prompt (TTFT 4.7–7.0 s, decode 19.2–20.3
   tok/s). Write-up:
   `benchmarks/results/nemotron35_lightning_5080_1m_multineedle_2026-09-04.md`; harness:
   **`benchmarks/bench_multi_needle.py`** (new, untracked). Depths 0.05 / 0.75 / 0.95 recall,
   0.25 / 0.50 / 0.60 do not; the control (a key absent from the text) is correctly denied
   with no fabrication. **The headline is question 8**: asked which of the depth-0.05 and
   depth-0.25 codes is larger and for their sum, the model returned
   9,854,500 = 5,663,623 + 4,190,877 — so the depth-0.25 needle it had just "missed" when
   asked directly *is* in the state. Grade long-context recall with more than one question
   shape per needle before blaming retention.

5. Ticket: `--kv-grow-step-tokens` + `--nvfp4-backend flashinfer` crashes (VMM int32 bank).
6. Ticket: `_maybe_shrink_growable_kv` evicts all unlocked prefixes before checking whether a
   shrink is possible (wipes the prefix cache at idle above the initial KV step).
7. Ticket: tests/moe/test_prefill_hit_d2d.py order-dependent flake.
8. Ticket: **an over-pool prompt has no client rejection path.** `PrefillManager.
   schedule_next_batch` skips a fresh request whose `input_len + output_len >
   cache_manager.max_size`, logs one `... can never be admitted and is being skipped` warning
   and `continue`s — but never removes it from `pending_list` and never fails the request, so
   the client hangs until its own timeout with no error. It also keeps inflating `waiting =
   len(pending_list) - index`, shrinking every other lane's interleave chunk. Worse,
   `_seatable_lanes` does **not** have the same skip: it sets `blocked_fresh = True` on the
   first request whose cost exceeds the budget, so one permanently unadmittable prompt pins
   the seatable-lane estimate at the number of continuations for as long as it sits in the
   queue. Fix: fail it with a 400/413 at admission (or at `add_one_req`), and mirror the skip
   in `_seatable_lanes`.
9. Ticket: **`stopped_for_lane_cap` rotation is dead code on this model.**
   `stopped_for_lane_cap` is only assigned inside `if lane_cap and len(reqs) >= lane_cap`, and
   `lane_cap = max_batch_seqs = _resolve_max_prefill_seqs(config)` is **0** for Nemotron
   (it returns 1 only when the checkpoint has `gguf_expert_types`) — confirmed live by
   `py-spy dump --locals` during the `81ab30e` soak (`lane_cap: 0`). So the interleaved branch
   `self.pending_list = remaining + chunked_list` is unreachable exactly on the profile that
   turns interleaving on. Its own comment describes a different trigger ("admission stopped on
   a resource-constrained request"), which is `blocked_fresh` / the `refusals` break, not the
   lane cap. Decide which one it means and set the flag there, or delete the branch.
10. Ticket: a refused prefill pass costs `O(queue x prompt)` radix walks — 16 pending
   118 K-token prompts are re-`match_prefix`ed from scratch on every pass that returns `None`
   (4 of 5 py-spy samples during the stall were inside `fast_compare_key`). Cache the match
   per pending request until its prompt or the tree changes, or skip the walk when the pass
   has already refused a fresh admit.
11. Ticket: `benchmarks/scheduler_replay.py` (the CPU replay gate added in `508ea32`) scored
   `81ab30e` at "2.49x tokens / 2.14x completions" — the commit that then failed the live soak
   on both routes. It models neither retained session leases, nor decode residency, nor the
   idle timeout, which is where the stall comes from. Either extend it or stop treating it as
   an acceptance gate for scheduler policy.

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
