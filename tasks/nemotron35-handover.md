# Nemotron 3.5 Lightning on FreeToken — handover

State as of 2026-09-05, HEAD `d960467` on `main`. **The four tickets closed below (graph ladder,
A-operand deinterleave, extend-cache guard, spec draft length) are in the working tree,
uncommitted** — commit them before branching. `fork/nemotron35` is at HEAD (pushed); `fork/main`
(`62f5a66`) is 82 commits behind and 0 ahead — a fast-forward is available and is the user's
call. Read this, then `tasks/nemotron35-plan.md` (spec),
`tasks/todo.md` (open checklist), `tasks/lessons.md` (rules — read before touching the GPU),
`docs/nemotron.md` (profiles + numbers), `docs/switchyard.md`, `docs/oracle.md`,
`docs/cpu-checks.md`.

## Status
Serving is correct and fast on the RTX 5080; the 262K/1M recall, scheduler-stall and
16-way-soak blockers are all closed, and what remains is a ranked ticket list plus one
merge decision.

## Performance (start of effort → now)

| metric | start (2026-09-04, ~`508ea32`) | now (`d960467` + the 2026-09-05 tickets) | evidence |
|---|---|---|---|
| decode 131K | 82.8 tok/s | **145.3** (1.75x) | decode_launch_2026-09-04 |
| decode 262K | 58.7 | **132.4** (2.26x) | decode_launch_2026-09-04 |
| decode 524K | 35.4 | **113.6** (3.21x) | decode_launch_2026-09-04 |
| decode 1M | ~20 | **95.8** (single sample; 80.7 on the oracle's 1M leg) | decode_launch, oracle |
| prefill 131K | 3,230 tok/s | **6,577.8** (2.04x; TTFT 21.6 → 19.8 s) | + misc_tickets (A-operand) |
| prefill 262K | 1,965 | **3,936** (2.00x) | prefill_profile + moe_prefill_gemm |
| prefill 524K | 1,064 | **2,297** (2.16x, measured live in the oracle run) | oracle_2026-09-05 |
| prefill 1M | 573–576 | **1,307** (2.28x; MoE-GEMM gain not re-measured at 1M) | prefill_profile |
| 1M TTFT | 1,810–1,824 s | **795.8 s** | prefill_profile |
| 16-way decode aggregate (soak) | 81.6 stage / 161.4 passthrough tok/s | **96.8 / 177.5** | soak §U |
| 16-way decode (engine, 12 lanes) | 143.21 eager | **153.84** (1.074x; 1.039x at 16) | decode16 |
| soak stage | FAIL (crash, then stalls/deadlock at `81ab30e`/`ea7ed7c`) | **PASS** 492 req / 0 err / 0 STALLED, p95 109.4 s (−25%) | soak §V |
| soak passthrough | FAIL | **PASS** 1,904 req / 0 err / 0 STALLED, p95 24.6 s (−25%) | soak §V |
| prefill starvation signature | 61% stage / 19% passthrough of passes | **0 / 1,202 passes** | soak §V |

Oracle recall by question shape (needles all present in state; **0 `retention`, 0 `selection`
at every length on both engines**):

| length | direct (key→code) | combined | reverse (code→key) | control | FreeToken total | llama.cpp |
|---|---|---|---|---|---|---|
| 262K | 5/6 (llama.cpp 3/6) | 2/6 (2/6) | 6/6 (6/6) | pass | **19/24** | 17/24 |
| 524K | 1/6 (2/6) | 0/6 (1/6) | 6/6 (6/6) | pass | 8/19 | 10/19 |
| 1M | 1/6 | 0/6 | 5/6 | pass | 7/19 | not runnable on this card |

The `key → code` collapse between 262K and 524K returns the **same wrong near-duplicate code
byte-for-byte in both engines** — a model property, not an engine defect. 131K has not been
run on either engine (cheap open rung).

## Closed
- **262K recall** — Mamba-2 prefill `dt` floor (`time_step_min` is an *initializer* range, not
  a runtime bound). `3ac79ec`; `262k_rootcause_2026-09-04.md`, `262k_crossengine_2026-09-04.md`.
- **Decode launch config** — `_grid_filling_splits` sizes the split count to the SM count for
  untuned head shapes; int64 slot ids on KV load. `acc91e9`; `decode_launch_2026-09-04.md`.
- **Prefill superlinearity** — extend-attention `BLOCK_M` capped by the fp32 accumulator's
  register budget (396 spill slots → 14). `4a99e34`; `prefill_profile_2026-09-05.md`.
- **Native-Q8 extend QK** — closed NEGATIVE, kernel unchanged (the 225 TFLOP/s premise is the
  spec sheet; the kernel is at 57–60% of the achievable 123). `prefill_q8_2026-09-05.md`.
- **MoE prefill GEMM 1.74x** — one e4m3 block scale per 8 bytes + hardware `cvt.rn.f16x2.e2m1x2`;
  29.47 → 16.95 ms/layer at M=8192, bit-identical. `2a139ad`; `moe_prefill_gemm_2026-09-05.md`.
- **Extend-path MoE 9–10x** — `--moe-extend-cache-tokens` (default 64): small extends take the
  decode movement path instead of streaming all 128 experts (16.5 GB/forward at the PCIe
  roofline). `89b632b`; `extend_moe_2026-09-05.md`.
- **Scheduler admission** — standing reservation + finishability invariant (`b030c7f`), then
  the seatable-lanes chunk divisor (`812bc57`). Two failed attempts reverted first
  (`81ab30e`→`5bf0bcc`, `ea7ed7c` deadlock). `f6ed0b5` soak PASS; soak §U/§V.
- **Client-disconnect abort in prefill** — `ff470e7`; `server/disconnect.py`, 12 tests.
- **Elastic CUDA graphs** — dense batch sizes to 16; 73.5% of soak decode batches ran eager.
  `14c1bd8`; `decode16_2026-09-05.md`.
- **Non-elastic graph ladder** — `_determine_cuda_graph_bs` goes dense to 16 for **offload-MoE
  models only** (dense models keep the historical sparse list, pinned by a test); 140.43 → 150.90
  tok/s at 12 lanes (**1.074x**, three alternating repeats per arm out of one binary, perfect
  separation). Hatch `FREETOKEN_GRAPH_DENSE_BS=0|1`. `misc_tickets_2026-09-05.md` §1.
- **MoE prefill A-operand deinterleave** — both A gathers were stride-2; an even-k/odd-k plane
  rewrite makes them unit-stride at an unchanged reduction order. **1.215x at M=8192, bit-exact**
  (gap to b12x 1.34x → 1.10x); 131K prefill 6,124.7 → 6,577.8 tok/s, TTFT 21.6 → 19.8 s, decode
  unchanged, needle PASS. **On by default**, `FREETOKEN_NVFP4_PREFILL_DEINTERLEAVE_A=0` disables.
  `misc_tickets_2026-09-05.md` §2.
- **Extend-cache threshold** — `--moe-extend-cache-tokens` stays **64**: the crossover is between
  64 and 80 on wall time, and above it the cached path gathers nearly every expert anyway. Plus a
  live crash guard — flashlib's `lru_ensure` cannot compile past `BLOCK_K = 1024`, so `256` used
  to kill the engine mid-forward; `use_cached_extend` now refuses above 1,024 routed ids
  (m ≤ 170 at top-6). New harness `benchmarks/bench_extend_moe_threshold.py`.
  `misc_tickets_2026-09-05.md` §3.
- **`--spec-draft-len` default** — stays **8**: k=16 is 0.870x of spec-off at 131K (k=8 0.898x)
  against a ±2% criterion, and at k=16 the break-even gate never closes (`declined_uneconomic`
  0 of 55 peeks). 16 stays a per-request setting for copy-heavy traffic.
  `misc_tickets_2026-09-05.md` §4.
- **Observability** — `/v1/stats.scheduler` + `requests.aborts`, invariant counted every pass.
  `78f29d3`.
- **1M gate** (restart persistence, eviction, NVMe restore) `31d606d`; **hidden-state parity**
  (52 layers, cosine ≥ 0.998840) `befcde6`; **1M direct-addressing** closed model-limited
  `be85ffa`; **MTP** NO-GO; **n-gram speculation** shipped behind `--speculative ngram` but
  measures 1.01–1.03x (`e4070da`, `ngram_spec_impl_2026-09-05.md`).
- **CI** — `.github/workflows/cpu-checks.yml` (ruff + CPU unit tests + scheduler replay gate),
  `508ea32`; `docs/cpu-checks.md`.
- **Soak drivers in-repo** — `benchmarks/switchyard_soak/` (`f6ed0b5`).

## Open, ranked by value
1. **n-gram verify overhead** — ~40% of a verify step is not the forward (~52 ms vs a ~30 ms
   extend forward): 46 eager kernel launches in the commit, `_prepare_batch` rebuilding pinned
   staging for a one-request batch. Taking the step from 7x to 4x a decode step moves the copy
   class 1.03x → ~1.12x. Also: burst-entry hysteresis costs ~4x in draft rate (0.079 measured
   vs 0.353 offline). The **131K regression is this ticket** and nothing else: at the shipped k=8
   the needle case is 0.898x of spec-off and the gate cannot refund its own two probe steps
   (163 ms at k=8, 279 at k=16, against a 10.4 ms decode step) — draft length is settled, the
   verify step is not. `ngram_spec_impl_2026-09-05.md` §6; `misc_tickets_2026-09-05.md` §4.
2. **MoE prefill leftovers.** (a) gemm2's A *is* gemm1's output, so its deinterleave prepass is
   removable by having gemm1's store emit the two k-planes — ~0.3 ms of the 0.551 at M=8192.
   (b) The M=256 GEMM bucket runs at 20% of ceiling with +53% padding waste at `BLOCK_M=16`.
   (c) The 1,024-routed-id extend guard is conservative under LFU (`cache_policy_id == 1` takes
   the in-repo sized kernel, which has no width limit), and the width just under the ceiling
   costs 22 minutes of Triton JIT — a warmup job, never a live request, if the threshold is ever
   raised. `misc_tickets_2026-09-05.md` §2/§3, `moe_prefill_gemm_2026-09-05.md` §10e.
3. **16-way decode is at the hardware ceiling — do not re-litigate.** 74% of the step is the
   PCIe expert gather at 51–52 GB/s against a measured 52.9 GB/s link, working set ~1,417
   expert-layer slots against 976 in the pool. Attention, the MoE GEMV and Mamba-2 were all
   measured fine at batch 16. Only two levers left: `--moe-backend hybrid` at 16 lanes (never
   measured; the auto-threshold asks the wrong question) and the 976-vs-1,417 slot deficit.
   `decode16_2026-09-05.md` §0/§2/§7.
4. **fork/main fast-forward** — user decision. `fork/main` (62f5a66) is a strict ancestor of
   HEAD; `fork/nemotron35` already carries the merge.
5. **`_gguf` extension rebuild before deploying on Ada** — the fork/main merge (`32cc504`)
   changed multiwarp bool → warps int64; a stale `.so` silently picks the 4-warp path.
6. **Oracle 131K rung** on both engines (~10 min total), and the 524K `direct:harbour` lead
   (the one leak-free direct probe llama.cpp holds and FreeToken loses — also the one turn with
   a 50 s TTFT, i.e. a partial-prefix re-prefill; re-run with `--filler-cursor 65`).
7. **Scheduler tickets** (all in `tasks/todo.md` with evidence): an over-pool prompt has no
   client rejection path and pins `_seatable_lanes`; `stopped_for_lane_cap` rotation is dead
   code on this model; a refused pass costs O(queue × prompt) radix walks;
   `_maybe_shrink_growable_kv` wipes the prefix cache at idle; `scheduler_replay.py` scored the
   commit that then failed the live soak, so it is not an acceptance gate for policy.
8. **Watch mean lanes per prefill batch** every soak: 3.43 stage / 4.92 passthrough at
    `13af13d`. Stage >~5 **together with** rising errors or p95 is the §R6/§R7 mode returning.
9. **CI follow-ups** — re-run the full unit-test step with no GPU job live and record the wall
    time; fold `tests/moe`+`tests/kernels` in once `test_offload.py::test_adjust_config_*` skips
    without flashinfer; pay down the `[tool.ruff.lint] ignore` list.
10. **Pre-existing test issues, not this effort's** — `tests/server/test_muse_glimmer_parsers.py`
    is reported to need the repo root on `PYTHONPATH` to collect; `tests/models/
    test_laguna_modules.py` errors only under GPU contention (6 RuntimeErrors while a sibling
    agent's server was loading, passes alone) — a suite-ordering artifact, re-run alone before
    blaming a diff. Neither was re-verified this session (no pytest while a model is loaded).

## How to run things
- **Soak**: `benchmarks/switchyard_soak/run.sh [tag] [duration]` — stage 20 m then passthrough
  20 m, c=16, server under `scripts/gpu_lock.sh` with `FREETOKEN_SCHEDULER_INVARIANT=warn`.
  It refuses to start below 26 GiB `MemAvailable`; everything lands in `runs/<tag>/` (gitignored).
  Grade with `analyze.py` (per-route stats, lanes/batch, starvation-signature fraction,
  `stats_*.json` deltas) and `gaps.py` (leading/trailing silence). Contract/e2e checks:
  `scripts/switchyard_e2e.sh contract|soak|agents` (`docs/switchyard.md` §8).
- **Oracle**: three phases in `docs/oracle.md` §"The three commands" — A FreeToken (you start
  the server; include `--enable-cache-report`), B llama.cpp (starts/stops its own server;
  `--n-cpu-moe 14` at 262K, **23** at 524K), C compare (CPU only). Sweep dimension is *length*,
  never depth. `--target-prompt-tokens 1044480` at the 1M rung (the suite is a conversation and
  grows). Verify prompt identity on CPU with `record --build-only` before taking the lock.
- **The 2026-09-05 ticket harnesses**: `scripts/gpu_lock.sh benchmarks/decode16/phaseE2.sh
  <outdir>` (graph ladder, three alternating repeats per arm out of one binary, ~22 min);
  `benchmarks/bench_moe_prefill_gemm.py --variant tree deint prepass --grid shipped --verify`
  (~1 min); `FREETOKEN_GPU_LOCK_WAIT=7200 scripts/gpu_lock.sh
  benchmarks/extend_moe/run_threshold.sh` (~5 min to m=96 — but the first m whose `m*top_k`
  crosses 1024 costs **~22 min of Triton JIT**, and m ≥ 256 is refused by the guard);
  `benchmarks/probe_spec_ngram_impl.py --sweep-k ...` (~2 min). Full commands in the
  results file's Reproduction section.
- **Replay gate / CI**: `uv run benchmarks/scheduler_replay.py --gate` (CPU, ~4 s, 428 MB RSS);
  the same thing CI runs alongside `ruff check .` and the CPU test directories
  (`docs/cpu-checks.md`).
- **A/B hatches**: `FREETOKEN_DECODE_{KV_SPLITS,BLOCK_N,NUM_WARPS}`,
  `FREETOKEN_EXTEND_{BLOCK_M,BLOCK_N,NUM_WARPS,NUM_STAGES}`,
  `FREETOKEN_NVFP4_PREFILL_{BLOCK_M,BLOCK_N,BLOCK_KB,GROUP_M,NUM_WARPS,NUM_STAGES}`,
  `FREETOKEN_ELASTIC_GRAPH_MAX_BS`, `FREETOKEN_GRAPH_DENSE_BS` (non-elastic ladder, `0|1`),
  `FREETOKEN_NVFP4_PREFILL_DEINTERLEAVE_A` (`=0` restores the interleaved A gathers),
  `FREETOKEN_NVFP4_NO_NATIVE_CVT`, `FREETOKEN_NEMOTRON_DT_MIN`, `FREETOKEN_MAMBA2_REF`.
  Both launches are logged once at startup (`Triton decode launch:`, `Triton extend launch:`).

## Host rules (from tasks/lessons.md — all learned the hard way)
- `systemctl --user stop piro-board-embedder.service` before GPU work (Restart=always, 4–10 GB).
- Host RAM (34 GiB WSL, 4 GiB swap) is the constraint, not VRAM. **Any** job that loads the
  checkpoint runs under `scripts/gpu_lock.sh`: refuses below 22 GiB `MemAvailable`, 4 h cap,
  `oom_score_adj=1000`, reaps the worker tree on exit.
- **Never run pytest, or import torch at all, while a model is loaded.** A CPU test sweep
  overlapping an expert-bank build OOM-restarted WSL twice. Check `pgrep -f "ft serve"` and
  `nvidia-smi` first; if anything is loaded, stay under ~1 GB RSS and do desk work.
- **Never pipe `scripts/gpu_lock.sh`** — its exit trap runs `pkill -9 -g $$` and kills the
  reader. Redirect to a file, then read the file; expect `Killed`/137 after a *successful* run.
  A wrapped script must `exec >` its own log and use `python -u`, or the log is 0 bytes.
- **No detached GPU job**: never end a turn while your GPU job continues. A background `until`
  loop does not make the agent wait; a foreground `until ! kill -0 $PID; do sleep 20; done` does.
  `pgrep -f <script>` matches the `bash -c` wrapper — check `ps -o etime -p <pid>` or an
  artifact's mtime instead.
- The scratchpad does **not** survive a WSL restart. Anything that produced a number in a
  results file belongs under `benchmarks/` or `scripts/`.
- Never `git stash` / `git commit -a` / `git add -A`; implementers work in worktrees.
  Kill leftovers by venv path: `pkill -9 -f "FreeToken/.venv/bin/python3"`, then `free -g`.
  Clear a stale JIT lock on sight: `rm ~/.cache/torch_extensions/py312_cu130/*/lock`.
- Serving profile: `FREETOKEN_PIN_BUDGET_GB=17`, `--memory-ratio 0.85`, 8K chunks, q8_0 KV +
  Triton attention, `--nvfp4-backend` auto→triton, LFU for 16-way. Needle gates go through
  `/v1/chat/completions` with digit-free filler; never grade raw SSE.

## Orchestration model
Lead session steers and does not read files it can delegate. Implementation goes to Opus
subagents in `git worktree`s, **at most two at a time**. **One GPU job at a time**, under the
lock, and the agent that started it stays attending it until it exits. CPU-side agents run no
pytest and no torch import while a model is loaded, and stay under ~1 GB RSS. Two stale
worktrees predate this effort and can be ignored or removed:
`.claude/worktrees/agent-a45f827ae98e76526`, `.claude/worktrees/agent-a4ce5e26d2ccafdb6`.

Results files referenced above live in `benchmarks/results/` with the prefix
`nemotron35_lightning_5080_`.
