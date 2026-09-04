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
2. **16-way Switchyard soak against the slot-reclaim fix** (dcb617a): docs/switchyard.md
   commands; previous run stalled on the now-fixed crash. Pass = 0 errors, no STALLED.
   Driver scripts: scratchpad/fix/{run.sh,serve.sh}.
3. **1M gate remainder**: restart persistence, capacity/age eviction, a ≥1M-size NVMe restore
   timing, results file benchmarks/results/nemotron35_lightning_5080_1m_sessions_<date>.md.
   Growth to 524K×3 and spill/restore-on-demand were verified; driver:
   scratchpad/1m/{drive.py,serve.sh,summarize.py}. Retest 262K/524K needles via chat only
   AFTER item 1 resolves.
4. **Phase 3H hidden-state export** is merged (1f2de67; docs/switchyard.md §6). Only the GPU
   parity check remains: start a P1 server with `--hidden-states-dir /tmp/ft-hidden-states`
   (mkdir first) then `scripts/gpu_lock.sh uv run benchmarks/probe_hidden_states_parity.py
   --model ~/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 --base-url
   http://127.0.0.1:1919 --hidden-states-dir /tmp/ft-hidden-states --prompt-tokens 300`
   (per-layer cosine > 0.99 vs HF output_hidden_states on CPU).
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
