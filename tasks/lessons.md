
## 2026-08-26 (Ornith phase 2 live A/B)
- Never grade model output by grepping a raw SSE stream: tokens split codes
  across `data:` events. Concatenate the JSON `text` fields first. (Caused 3
  false NEEDLE_MISS on a run that was actually 3/3.)
- A "server startup timeout" on this stack is usually one of: (a) stale
  torch_extensions `lock` from a SIGKILLed build -- the loader sleep-polls it
  forever; (b) swap-cold expert-bank build. Diagnose via /proc/<pid>/wchan
  (hrtimer_nanosleep) + lock mtime BEFORE extending timeouts or killing.
- `pkill -f "ft serve"` does NOT kill the workers: they are
  `multiprocessing.spawn` stubs holding ~20 GiB shmem banks. Kill by venv path
  and verify with `free -g`, or the next run OOMs the box (took the terminal
  down 3x).
- `--num-tokens` is GPU KV capacity, not host RAM. Don't guess flag semantics
  in explanations to the user -- read the argparse help first.
- Structure live perf claims as env-gated A/B on the same box minutes apart
  (FREETOKEN_GGUF_DISABLE_MMA=1), with a path-proof mechanism decided BEFORE
  the run; in-server INFO logging is swallowed by the log handler, so proof
  must be external (JIT cache touch, wall-time signature).

## 2026-09-03 (Nemotron 3.5 Lightning, parallel subagents)
- Subagents must never `git stash` / checkout while siblings edit the tree; get a
  clean-main baseline from a `git worktree` instead. Say so explicitly in every
  parallel-agent prompt.
- Host facts change between sessions (RAM, swap, GPU holders); re-run
  benchmarks/preflight_nemotron_host.py before trusting plan numbers.

## 2026-09-04 (Nemotron 3.5 Lightning, Phase 1 GPU bring-up)
- `pkill -f /path/to/.venv/bin/python` in the SAME command line that also
  launches the server kills its own shell: `pkill -f` matches the bash `-c`
  string, which contains the pattern. Kill in a separate tool call.
- Layer-parity probes against a statically-quantized checkpoint (modelopt FP8
  `input_scale`) must use in-distribution activations. `torch.randn` hidden
  states drove Nemotron's mamba `out_proj` 9x past its calibrated amax and
  produced a fake 0.974 cosine. Sample real embedding rows through the block's
  own input norm instead.
- HF's pure-Torch `mamba2_chunk_scan` costs 4.2 MiB/token/layer in rank-6
  broadcast temporaries (17 GiB at a 4096-token prefill chunk). Measure a
  fallback path's peak transient BEFORE sizing `--memory-ratio` around it.
- Needle quality at 128K degrades with the NUMBER of prefill chunk boundaries
  (64K/8 chunks exact, 128K/16 corrupted digits, 128K/32 collapse) and is
  identical under q8_0, fp8_e4m3 and bf16 KV -- so chunk-size sweeps, not KV
  dtype sweeps, are the discriminator for long-context regressions on hybrids.
- GPU measurements are only valid with the GPU exclusive: wrap every server /
  benchmark / timing run in `scripts/gpu_lock.sh`; overlap only CPU work.
- Implementer subagents run with `isolation: worktree` and hand back a diff;
  never on the shared tree.
- Never `git commit -a` / `git add -A` while subagents are editing: it sweeps their
  in-progress work into an unrelated commit (happened at 78ba6d7, rewritten). Always
  `git add` explicit paths owned by the finished task.

## 2026-09-04 (Nemotron 3.5 Lightning, 128K needle root-cause)
- **Retraction.** The entry above ("needle quality at 128K degrades with the NUMBER
  of prefill chunk boundaries") is wrong. Controlled re-runs: the 65,536-token
  needle passes with 8 chunks (`--prefill-chunk 8192`) AND with 32 chunks (2048);
  131,072 fails with 16 chunks and with 32. Chunk count is not the variable.
- Before blaming a code path, prove the benchmark measures what its name claims.
  `bench_long_context.trim_filler` drained the largest filler gap first, which
  removed *all* the text before the needle: at every `--target-prompt-tokens` the
  needle landed at token ~1024 (depth 0.008-0.016). The "128K needle" gate was a
  retention test at an ever-growing retrieval distance, not needle-in-the-middle,
  and the 64K -> 128K delta was a doubling of that distance.
- Exonerate numerics cheaply and first, with an fp64 oracle, before touching the
  server: `chunk_scan.py` vs a sequential fp64 recurrence is 2.5e-7 relative at
  T=131,072 and *identical to the last printed digit* for chunk in {T, 8192, 4096};
  the full mixer path (conv carry + x64 track split + state-pool round trip)
  reproduces a single-pass 16K prefill to 1.6e-5 (the bf16 output floor).
- A/B a suspect subsystem by *disabling* it, not by reasoning about it. Growable KV
  looked guilty (the 128K runs grow mid-prefill, the 64K ones do not) until a run
  with the whole pool committed up front produced the byte-identical wrong answer.
- `--memory-ratio` is correctness-neutral but throughput-critical for the pure-torch
  Mamba scan: at 0.90 an 8192-token chunk's ~1.8 GiB of fp32 transients exceed the
  ~1.2 GiB of free VRAM and the allocator thrashes (1,100 tok/s); at 0.80 the same
  chunk runs 2,400-2,700 tok/s. That -- not scheduling -- is why 4096-token chunks
  looked "2.4x faster than 8192" in the Phase 1 results.
- `GET /v1/models` answers 200 while the backend is still loading weights, so it is
  not a readiness probe: poll the server log for `Scheduler is idle` (or use the
  bench's own `wait_ready`) or the first real request returns 503.
- Host RAM, not just VRAM, is the contended resource with sibling agents: the expert
  banks need ~15.4 GiB resident and startup refuses below that. Gate long runs on
  `MemAvailable` as well as on `scripts/gpu_lock.sh`.
- Closed 2026-09-04: same-session A/B at 131 072 tokens / 16 chunks / q8_0 / triton,
  identical server flags, only the prompt's needle depth differing — pre-fix trim
  answers `5663616` (byte-identical to the recorded failure), fixed trim answers
  `5663623`. 4096-token chunks (32 boundaries) are exact too. The engine was never
  wrong; the gate was. `tests/models/test_nemotron_h_chunked_prefill.py` now pins
  chunked == single-pass on the real mixer so the exoneration cannot silently rot.
- The `--memory-ratio` effect above is measurable end to end: 131 072 tokens with
  8192-token chunks runs 961 tok/s at 0.90 and 1 769 tok/s at 0.80; 4096-token
  chunks at 0.90 run 1 910 tok/s. Chunk size was a proxy for free VRAM.
- A per-module output buffer that GROWS is a use-after-free under CUDA graphs (found
  2026-09-04, task 2A4). Graph capture runs an eager warmup at every captured batch
  size, so each graph bakes in the address it saw; an elastic capacity raise then puts
  real decode batches ABOVE the largest captured size, those run eagerly, the buffer is
  reallocated, and every later replay of a smaller graph writes into the freed block.
  It surfaced three kernels away as `IndexKernel.cu:111 index out of bounds` on
  `token_pool[output_mapping]` (exactly `#running-req` corrupted int64 indices). Cache
  one buffer PER batch size and never replace an entry. Compute-sanitizer on the kernel
  in isolation is clean for this class of bug -- the corruption needs the graph.
- Bisecting a serving crash: (1) rerun the same profile from a detached worktree at the
  parent commit (`git worktree add --detach`, `PYTHONPATH=<wt>/python`, and copy the
  built `kernel/*.so` in -- the editable install's extensions do not come along) to tell
  a regression from a pre-existing bug; (2) rerun on the same tree with the reference
  path (`FREETOKEN_MAMBA2_REF=1`) to tell wiring from kernels. Two ~3-minute runs beat
  an hour of reading.
- `/health` returns 200 with `{"status": "loading"}` while weights load; a readiness
  poll must require `"status": "ok"`, otherwise every gate request 503s.

## 2026-09-04 (Nemotron 3.5 Lightning, 131K needle at 8192 chunks — closed)
- **Prove chunk invariance on the ONE layer whose inputs are identical by construction.**
  On a hybrid, layer 0 (Mamba) consumes the embeddings, so its end-of-prefill state must
  be bit-identical between two `--max-prefill-length` values. It was (0.000e+00 after
  131 072 tokens), which clears the whole Mamba integration in one number. Every deeper
  layer's divergence is about its *inputs*, not its scan — comparing those is comparing
  amplification, not correctness.
- A per-layer state diff of 1e-1 at depth is NOT evidence of a bug on a 52-layer hybrid at
  131K tokens. Controls first: the known-good reference scan diverged *more* between the
  same two chunkings (1.5e-1 vs 1.1e-1) and answered correctly at both; the two scans at
  the *same* chunking differed most of all (3.0e-1). Magnitude of state divergence carried
  zero signal about the needle outcome. Always take the A/B of the A/B.
- Non-Mamba layers are not bit-invariant to the prefill chunk size: chunk size *is* the
  GEMM's M. The NVFP4 dense linear is invariant (bit-identical at M=4096 vs the head of
  M=8192), the routed fused-MoE path is not — that is where the 1e-7 seed enters, and 52
  layers turn it into 1e-1. Expect this on every chunked-prefill A/B; it is not a bug.
- **Never gate long-context retrieval on a raw `/v1/completions` continuation with
  `ignore_eos`.** All four runs agreed on the top-1 next token (`<|im_end|>`, margin
  0.75-1.63); `ignore_eos` then forces generation *past* the model's chosen end-of-text
  into an unanchored continuation of the haystack, and whether the needle reappears there
  is a coin flip. Ask the question through the chat template instead. (`bench_long_context`
  now does; the 131 072/8 192 case that "failed" 3/3 passes.)
- An env-gated in-server state dump (`FREETOKEN_MAMBA2_STATE_DUMP`) is cheap to build and
  settles in one afternoon what a week of reasoning about metadata cannot. Dump at the last
  prefill forward (`ChunkedReq.can_decode` is False on continuations) and skip the engine's
  warmup batches (`uid == -1`) or the record is the wrong forward.

## 2026-09-04 — never `pkill -f <pattern>` that matches your own shell
`pkill -f "drive.py --port 8123"` (and later `pkill -f "freetoken.cli serve --model ..."`)
matched the `bash -c ... eval '<the same text>'` wrapper the Bash tool runs, so pkill killed
the command that issued it: exit 144, and every step chained after the pkill (the `rm`, the
`mv`, the relaunch) silently never ran. Twice in one session the follow-up work was lost.
Rule: kill by PID (`pgrep -f` first, inspect, then `kill <pid>`), or by venv path with a
pattern that cannot appear in your own command line, and NEVER put cleanup or relaunch steps
in the same command as a `pkill`.

## 2026-09-04 — a shared repo can move under a long GPU run
Two other agents were editing `/home/lucas/ai/FreeToken` during the 1M-session gate: HEAD moved
ec54e21 -> da02c16 mid-run and `scheduler/cache.py` was rewritten 2 minutes after my server
started. The server kept running the code it imported at launch, so its traceback line numbers
no longer matched the file on disk -- which made the crash look like it came from a line that
contains no such call. Before debugging a serving crash, check `git log -1` and the source
file's mtime against the server's start time; if the tree moved, re-read the diff before
writing a fix, because the bug may already be fixed in the working tree.

## 2026-09-04 — a numeric needle needs a digit-free haystack
The first 1M-gate filler numbered every record ("Record 0000123 ... bay 45 ... day 12"). At
131 K tokens Nemotron 3.5 Lightning answered `1563630` -- a 7-digit string assembled from the
distractors -- instead of the planted `5663623`. The same prompt with digit-free filler (the
shape `bench_long_context.py`'s synthetic needle uses) recalled correctly. When planting a
numeric needle, keep every other digit out of the haystack, or the gate measures the
distractor set rather than retrieval.

## 2026-09-04 (Nemotron 3.5 Lightning, Switchyard soak scheduler crash — task 3F)
- **A cache-management step must never be able to kill the scheduler.** `_cache_req_hybrid`
  donated a frozen Mamba snapshot to the radix tree and *then* `pool.alloc(1)`'d a replacement
  ping-pong slot. Donating is an optimization; once the slot is in the tree there is no way
  back, so the shortage was unrecoverable and raised. Reserve the replacement BEFORE the
  irreversible step, and skip the whole commit when it cannot be reserved.
- **Two-currency pools need the *demand* signal wired to every allocation site, not just to
  admission.** `ensure_mamba_slots` can only reach UNLOCKED radix snapshots; an idle session
  lease (`retain_prefix`) locks its node, so a pool whose entire snapshot cache has become
  leases reports `mamba usage 1.00` with zero evictable and every eviction attempt frees
  nothing. `_reclaim_for_blocked_prefill` only runs for a *queued* request — nothing covered
  the mid-flight chunk commit or a cold restore. Route them through one escalating helper
  (free-list -> LRU evict -> spill the LRU idle lease).
- Slot accounting is the root cause, and it is arithmetic, not a leak: at
  `--max-running-requests 16` the pool is `4·16 + 2·16 + 1 = 97` slots (96 reportable), and the
  32-slot snapshot *cache* is exactly what session leases convert into non-evictable state.
  The same shape bites at R=1: `--linear-state-slots 5` seats padding + live + 2 ping-pong +
  ONE lease, so a second conversation's first turn was fatal.
- `mamba usage 1.00` in the prefill log is not "the pool is busy": `_mamba_slot_usage` excludes
  free slots AND evictable snapshots, so 96/96 literally means `free == 0 and evictable == 0`.
  It is a precise pre-crash signature — read the gauge's definition before interpreting it.
- **`/health` that reads a latched flag is not a liveness probe.** `fatal_error` is set by the
  supervisor thread one poll after a death and that thread then *returns*; any later death
  leaves `/health` answering `{"status":"ok"}` forever (nine minutes of soak probes passed
  against a server answering nothing). The `multiprocessing.Process` handles are already on
  the frontend state — ask `is_alive()` in the handler, and answer 503.
- **uvicorn's `timeout_graceful_shutdown` defaults to `None` = wait forever.** With a dead
  backend the in-flight ASGI tasks never finish, so the stop wedges in "Waiting for background
  tasks to complete" (38 min observed) holding the GPU and ~20 GB of pinned expert banks.
  Bound it, reap (join + SIGKILL) the workers on every stop path rather than only `terminate()`,
  and arm a plain-thread hard-exit backstop for the case where the event loop itself is wedged.
- A GPU repro is not the only "before/after" available: running the new regression test from a
  detached `git worktree` at the parent commit reproduced the byte-identical
  `RuntimeError: LinearStatePool exhausted: need 1, have 0` in 3 seconds, with no GPU lock.
- Unrelated tests error under GPU contention (`tests/models/test_laguna_modules.py` raised 6
  RuntimeErrors while a sibling agent's server was loading, passed alone). Re-run a suspicious
  failure alone before attributing it to your diff.
- A server whose scheduler died keeps the API process (and any wrapper lock)
  alive indefinitely; `scripts/gpu_lock.sh` then blocks every other GPU job.
  Check `fuser <lock>` + `nvidia-smi` (holder with 0 MiB = dead server) before
  assuming a queue is merely slow. The /health-503 + bounded-shutdown fix is
  the real cure.

## 2026-09-04 — never kill a session driver mid-turn: the restore already consumed the checkpoint
Pausing the 1M gate between rounds, I killed the driver one second after the last turn's
notification -- but the driver had already issued the next turn, and the scheduler had already
restored that session's cold checkpoint (`_restore_cold_session` calls `_discard_session_spill`
on success: the record is *consumed*, its bytes now live in the KV pool). Aborting the request
then freed that KV, so the session lost its whole 262 K prefix and the next turn re-prefilled
393 K tokens from scratch (305 s, `cached_tokens` 0) -- which looks exactly like a spill/restore
bug in the logs. Before stopping a driver, wait for it to be idle (no in-flight request), or
give it an explicit stop-after-this-turn flag. A checkpoint survives a *disconnect*, but not a
disconnect that lands after the restore.

## 2026-09-04 — cold-restore is all-or-nothing, so any client retokenization drift costs the whole checkpoint
Three separate harness details each silently destroyed a multi-GiB session checkpoint in the 1M
gate, all through the same mechanism: `_restore_cold_session` requires
`record.token_ids == input_ids[:record.num_pages]` exactly, so ONE divergent token in the
assistant turn discards the entire record and forces a full re-prefill (up to 521 s at 524 K).
The three triggers were (1) `ignore_eos=True`, whose forced tail no resend can reproduce;
(2) a stray `</think>` the model emits at long context, which FreeToken streams as ordinary
content and which retokenizes differently when echoed back; (3) killing the driver after a
restore had already consumed the record. When driving a multi-turn session gate, sanitize every
chat-control marker out of the assistant text before resending it, never force the reply length,
and expect `cached_tokens == 0` to mean "prefix diverged", not "spill/restore is broken" --
check for `Discarded cold session ...: client token prefix changed` before blaming the store.
- 2026-09-04 12:18 host OOM: kernel killed a FreeToken worker (18 GB shmem banks + 5 GB anon)
  and the sweep took the tmux scope incl. Claude Code. Rules: (1) ANY job that loads the
  checkpoint (server, parity, kernel test on real layers, ft bench bw) takes
  scripts/gpu_lock.sh — "small GPU footprint" is not the criterion, host RAM is; (2) the lock
  wrapper refuses to start below 22 GiB MemAvailable, caps a hold at 4 h, sets
  oom_score_adj=1000 on the job, and kills its worker tree on exit; (3) at most two
  subagents at a time, never one that ends its turn while its GPU job continues.

## 2026-09-04 (Nemotron 3.5 Lightning, 262K needle bisect — closed, no engine bug)
- **Sweep the PROMPT before you sweep the engine.** Eight engine variants (three KV dtypes, two
  attention backends, kernel-vs-reference Mamba, two chunk sizes, growable-vs-static KV,
  NVFP4-vs-bf16 dense) all failed at 262,144 and all passed at 131,072 — 8 servers, ~50 GPU
  minutes, zero signal. The variable that *did* move the outcome was the needle's depth, found in
  one extra server: at a fixed 262,144 tokens, depth 0.057 recalls exactly and 0.267 / 0.519 /
  0.761 / 0.947 all miss. A depth-0.05 control at the failing length is the engine's alibi and
  costs one request — run it FIRST, before any variant matrix.
- **Non-monotonic ⇒ not a code path.** The length sweep at fixed depth went PASS(131,072),
  FAIL(147,456), PASS(163,840), PASS(180,224), FAIL(196,608), FAIL(262,144). Nothing keyed on a
  power of two — 2^18, a page-table width, a tokenizer `model_max_length` of 262,144 — can pass at
  180,224 and fail at 147,456. Check monotonicity before writing the boundary-bug hypothesis down.
- **Byte-identical wrong answers across two independent kernels exonerate both.** The Triton
  packed-q8_0 loader and FlashInfer bf16 produced the same degenerate continuation
  ("…ledledger…ledder…") at 262K. An indexing or precision fault does not agree across
  implementations; a model behaviour does.
- Prove a suspect subsystem is a no-op by *exact equality*, not by reasoning: growable-vs-static
  KV was 0.000e+00 on state, conv and logits at both lengths, which retires the whole
  `grow_runtime_kv` / MoE-rebuild / graph-recapture path in one number.
- Re-confirmed: state-divergence magnitude carries zero signal. q8_0-vs-bf16 KV diverged *more*
  at the passing 131K (1.46e-01) than at the failing 262K (5.71e-02).
- Top-5 next-token logits at the question are a weak instrument when top-1 is a word: the 262K
  depth-0.05 run *passes* with no `5` in its top-5, because the digits are sampled several steps
  after "The secret passcode is". Grade the decoded answer; use logits only as corroboration.
- **Never edit a shell script while it is executing.** bash re-reads the file by byte offset, so
  rewriting `run_variant.sh` mid-run produced `line 17: t: command not found` and a bogus rc=127
  on a variant whose data was actually fine. Write a new file, or edit between runs.
- 6 of Nemotron-H's 52 layers are full attention (`layers_block_type` 5, 12, 19, 26, 33, 42); the
  other 23 mixers carry fixed-size Mamba-2 state. Expect depth-dependent, non-monotonic retrieval
  well below any architectural context limit, and expect NVIDIA's 1M claim (BF16 checkpoint) not
  to transfer to the NVFP4 release for *retrieval* — capacity and coherence are separate claims.

## 2026-09-04 (Nemotron 3.5 Lightning, 262K needle — cross-engine, the bisect's verdict RETRACTED)
- **Retraction.** The entry above ("262K needle bisect — closed, no engine bug") concluded a
  model/quant retrieval limit. llama.cpp (commit 6b80c74, official `ggml-org` **Q4_0** GGUF —
  that repo has no Q4_K_M) recalls the *byte-identical* prompts at 262,160 tokens at depths
  0.06 / 0.27 / 0.52 / 0.76 / 0.95 and at 147K / 196K / 262K — 8/8 PASS, answer `5663623` —
  while FreeToken on the NVFP4 checkpoint, re-run the same hour, still passes only at 0.06.
  Details: `benchmarks/results/nemotron35_lightning_5080_262k_crossengine_2026-09-04.md`.
- **A variant matrix run entirely inside one engine cannot exonerate that engine.** Eight
  FreeToken variants failing identically means the fault is common to all eight, not absent.
  The cheapest decisive control was another *implementation* of the same architecture, and it
  cost ~25 GPU minutes end to end — less than the matrix it overturned. Run the cross-engine
  control BEFORE writing "no defect found", not after.
- Make the prompt identity provable, not assumed: rebuild through the same helpers with the
  same HF tokenizer and `sha1sum` the result against the earlier run's prompt files. All eight
  matched, which is what makes the contradiction airtight (and llama.cpp then reported the
  same 262,160 post-template token count, so the chat templates agree too).
- A 16 GiB card is not a reason to skip a llama.cpp cross-check of a 30B-A3B model: `-ngl 999
  --n-cpu-moe 14` (routed experts of 14 of 52 blocks in host RAM) held 15.4/15.9 GiB VRAM and
  still prefilled 262K tokens at ~2,230 tok/s, ~2 minutes per data point.
- `scripts/gpu_lock.sh`'s exit trap `pkill -9 -g $$` kills the whole process group, i.e. the
  pipeline that invoked it: every `gpu_lock.sh ... | tail` returns 137 with the output lost.
  Have the wrapped script `exec >` its own log file and read that afterwards.

## 2026-09-04 (Nemotron 3.5 Lightning, 262K needle — ROOT CAUSE: the Mamba-2 dt floor)
- **The bug was one number**: `dt_limit = (config.time_step_min, inf)` in the prefill scan.
  `time_step_min`/`time_step_max`/`time_step_floor` are HF's *initializer* range for
  `dt_bias`, not runtime bounds; HF's `NemotronHMamba2Mixer.forward` reuses one of them as a
  clamp and FreeToken copied it. A 1e-3 floor caps every head's memory horizon at
  `1/(|A|*1e-3)` tokens whatever the network computes. 147,456 @ depth 0.52 and 262,144 @
  depth 0.52 go FAIL -> PASS on `dt_limit=(0.0, inf)` alone, at identical TTFT.
- **Read a hyperparameter's *use site* in the reference implementation before copying it.**
  Names ending in `_min`/`_max`/`_floor` on a Mamba config are almost always init ranges.
  vLLM passes `dt_limit=(0.0, inf)`; llama.cpp does not clamp; FreeToken's own *decode*
  kernel never clamped either — the prefill/decode disagreement was the tell, and it sat in
  a code comment ("Prefill only: ... keeps parity with both") that rationalized it instead of
  questioning it.
- **Non-monotonic ⇒ marginal, not "not a code path".** The bisect's rule ("nothing keyed on a
  power of two can pass at 180K and fail at 147K") was right but the *conclusion* drawn from it
  was wrong. A defect that erodes a resource continuously — here, the set of heads that can
  still carry information across the prompt — produces exactly a ragged pass/fail band. Rule
  out *thresholds*, not *defects*.
- **A per-length A/B of one suspected term beats any amount of state diffing.** Two servers,
  ~20 GPU minutes, one env var. The three previous rounds (state dumps, 8-variant matrix,
  cross-engine) cost hours and their value was entirely in narrowing *which* term to A/B —
  the llama.cpp source read that found it took one subagent and no GPU at all.
- Exonerated with evidence, not argument: the FP8 W8A8 Mamba `in_proj`/`out_proj` (a new
  env-gated hook, `FREETOKEN_DEBUG_FP8_ACT_STATS`, shows 11 of 46 matrices saturate their
  calibrated `input_scale`, but by the *same* 1.8e-5 clipped fraction at the passing 131K and
  the failing 147K — a constant tax, not a length term); and the NVFP4 path, which is W4A16
  end to end and quantizes no activation anywhere.
- `torch.save`/`.tolist()` in a debug hook must be gated on `batch.is_prefill`: the engine
  captures CUDA graphs on decode batches and any device->host copy inside capture raises
  "Cannot copy between CPU and CUDA tensors during CUDA graph capture". Also skip `uid < 0`
  (warmup) batches or the recorded amax is dummy-token magnitude.
- `set -e` + `[ -n "$X" ] && export ...` as a statement kills the script when `$X` is empty
  (the AND-list returns 1). Use `if ... then ... fi` in any `serve.sh` an unset variable can reach.

## 2026-09-04 (Switchyard 16-way soak vs. the slot-reclaim fix — a second currency, same anti-pattern)
- A fix that closes one pool's fatal exposes the next pool's. `dcb617a` made the GDN slot
  shortage recoverable (41 batches at 96/96, zero `LinearStatePool exhausted`); the soak then
  ran 602 s and died on the KV *page* pool instead, in `committed_pages_required`. When
  re-testing a resource-exhaustion fix, expect the queue to move to the next scarce currency
  and grep the new traceback before concluding "the fix did not work".
- **Gate every admission path, not the one you were looking at.** `PrefillAdder.try_add_one`
  has two branches; only the fresh-admit branch checks `available_size`. A chunked-prefill
  *continuation* is admitted unconditionally on the premise that "a continuation already owns
  its resources" — it owns its table slot and state slots, but not the pages for its **next**
  chunk. Continuations are also scheduled first, so the ungated path runs first.
- **A per-pass reservation is not a reservation.** `PrefillAdder` is rebuilt every scheduling
  pass, so `reserved_size` protects a prompt's remaining pages only within the pass that
  admitted it. Between chunks those pages are invisible to `available_size` and other traffic
  spends them. The SWA currency already solved this shape (`reserved_swa` + `max_end` cap);
  the KV currency did not.
- A soak whose failures come back in 755 ms instead of at the 600 s client timeout is a *good*
  sign, not a worse one: it means the bounded-shutdown fix closed the port instead of leaving
  requests hanging. Read `error_kinds` (`http_502` = connection refused) before reading the
  error *rate*.
- The soak's `health` column is the ROUTER's `/health`, not FreeToken's — it read `ok` through
  the entire outage. Poll the upstream's own `/health` separately (this run: 62 non-ok samples,
  first 503 eleven seconds after the death).

## 2026-09-04 (Phase 3H hidden-state export — the parity check's own reference was the bug)
- **A parity harness whose reference is `AutoModelForCausalLM.from_pretrained` must be run
  once before it is merged.** `probe_hidden_states_parity.py` shipped with a reference that
  cannot load this checkpoint on any host: modelopt `MIXED_PRECISION` (no `modelopt`
  quantizer in transformers 5.15), `backbone.*` vs HF's `model.*`, per-expert NVFP4 2-D
  tensors vs HF's fused 3-D parameter — a meta-device skeleton diffs 400 missing / 18 486
  unexpected keys — and 58.8 GiB of dense bf16 against 34 GiB of host RAM. Diff the
  checkpoint index against `AutoModel.from_config(...)` on `meta` (seconds, no GPU, no
  RAM) before trusting any "HF reference" path.
- **A meta-device model plus per-block load/free hooks is the cheap way to run a
  too-large reference**: pre-hook `load_state_dict(..., assign=True)` from the shards,
  post-hook record the output and assign the meta tensors back. transformers keeps
  ownership of masks, position ids, the Mamba-2 scan and the residual adds; peak was
  3.5 GiB VRAM and 10-22 s for a 52-block 316-token forward of a 30B model. The
  post-block forward hook also *is* the definition FreeToken exports, so the check stops
  depending on HF's `output_hidden_states` indexing.
- **When a per-layer parity curve is worst at the SHALLOW end and improves monotonically
  with depth, suspect a fixed absolute perturbation early, not accumulating quantization
  error.** Here it was transformers' own `time_step_limit = (config.time_step_min, inf)`
  (`modeling_nemotron_h.py:381`) — the exact 1e-3 `dt` floor whose removal fixed 262K
  recall. The reference was wrong, not the engine: dropping the floor moved the worst
  layer 0.9406 → 0.9988. Any HF-referenced Mamba-2 parity check on this stack must set
  `dt_limit=(0.0, inf)` on the reference, and a reference's own hyperparameters deserve
  the same "read the use site" scrutiny as the engine's.

## 2026-09-04 (Switchyard 16-way soak rerun vs. the KV back-pressure fix)
- **Charge a resource cap in the currency of the check it protects.** `fad1fc4`'s chunk cap
  subtracted `reserved_size` — the sum of admitted requests' WHOLE remaining prompts, which is
  the right figure for `_try_allocate_one`'s admission *policy* — while the thing it exists to
  keep satisfiable (`committed_pages_required`) demands only the batch's per-chunk page deltas.
  One 118 K-token continuation therefore reserved the entire pool and starved every peer in the
  pass: 6/6 lanes before the fix, 2/6 after, with 700 free pages and a 600-page batch. Fatal
  closed, throughput halved. Read what the guarded check actually sums before picking the budget.
- **A fix that closes a fatal can open a starvation.** The retest passed on "no crash" and still
  failed the soak's gate, on a route the previous run never reached. Grade a resource fix on
  *throughput under the same pressure*, not only on the absence of the traceback.
- **Run every route.** `switchyard_e2e.py soak` defaults to passthrough THEN stage and returns
  early on the first failing route, so a crash in route 1 silently skips route 2 — the stage
  route and the 10 m resilience set went untested for a whole run. The stage route is the harder
  test: its classifier doubles prefill demand (7,876 new prompt tokens/request vs 1,637) and
  drops prefix reuse from 88 % to 74 %, which is what exposed the lane starvation.
- **A CPU-only A/B against the parent commit settles a scheduler-policy regression in seconds.**
  `git worktree add --detach <parent>` + `PYTHONPATH=<wt>/python` and a 90-line script that
  builds a CacheManager/PrefillManager and counts admitted lanes: three data points (parent,
  HEAD, HEAD+fix), no GPU, no model. Do this before spending 25 GPU minutes on a re-soak.
- **"Did the back-pressure path engage?" needs a counter, not a grep.** The deferral returns
  `None` silently and `/v1/stats` has no scheduler counters, so engagement could only be argued
  from `Released soft session … (admission pressure)` rates (0.25/s vs 0.086/s pre-fix) and from
  batches surviving at `#mamba-slot: 96/96`. Add the counter when you add the back-pressure.
- A deferred prefill does **not** spin the loop: sampling every FreeToken venv process every 5 s
  showed the busiest at median 106 % CPU in all phases, including 60 s intervals that completed
  zero requests. Sample per-process CPU during a soak; it costs nothing and it retires that
  whole class of worry with a number.
- `#new-seq: 1, #new-token: 512` with `token usage: 0.49` is a *scheduling* symptom, not a
  memory one: `chunk_limit = token_budget // waiting` (interleaved mode) divides the 8 K budget
  by the QUEUE DEPTH, so 16 queued requests cost a 512-token chunk even when the pass seats one
  lane. Compare the `#new-seq`/`#new-token` histogram between two runs before blaming a pool.

## 2026-09-04 (Nemotron 3.5 Lightning, 1M session gate — residency criteria)
- **A demand-driven mechanism needs a demand that actually fails admission.** The spill
  trigger was a 6-token request from a foreign session; the resident 1.04M session leaves
  ~8.5K tokens of the 1,048,576-token pool free, so it fit, nothing was reclaimed, no
  checkpoint was written — and the next phase silently degraded from a 2.7 s restore into a
  ~1 h cold re-prefill. Size the competing request from the *free* pool (60K worked), and
  make the script assert the expected artifact (`ls <spill root>/*/manifest.json`) before it
  spends the next hour.
- **Rebuilding a long context incrementally beats re-prefilling it.** Eight 130K turns grow a
  session to 1.04M in 31 min of prefill because each turn only prefills its own tail; one cold
  1.04M prompt is ~1 h. When a long-context run has to be redone, redo it as growth.
- **Check the on-disk manifest version before planning a run around an old checkpoint.**
  `MANIFEST_VERSION` went 1 → 2 in `b7242d2` (boundary states), so the 2.1 GiB record left by
  the previous session was deleted at startup (`adopted 0 checkpoint(s), removed 1 stale
  entr(ies)`) — the whole plan to restore it was dead before the server came up.
- **A cap that fits exactly one record cannot demonstrate *which* victim LRU picks.** Size the
  budget to hold N−1 of N candidates (1.6 GiB for three 0.54 GiB checkpoints) or the eviction
  test proves only that the cap is a cap.
- `--session-spill-ram-gb 0` is how you force the NVMe tier: at the 4 GiB default a 3.5 GiB
  checkpoint stays in RAM, where a restart destroys it and the "survives restart" criterion
  cannot be tested at all.
- Measure a session checkpoint at 3.65 KiB/token (q8_0 KV + 8 × 47 MiB boundary states), not
  at the 3.1 KiB/token the KV alone suggests: the v2 state snapshots are ~20 % of a 131K
  record and the byte cap is charged on the total.

## 2026-09-04 (experiment design, from user correction)
- Before any run longer than ~10 min, write down the question it answers and check whether ONE
  run can answer several (e.g. multiple needles at different depths in a single prefill, then
  cached follow-up turns). Executing the checklist wording literally cost ~1 h of GPU today
  (five 262K prefills for five depths). Design the experiment, then run it.
- Second question, before AND after each task: "Knowing the user's actual goal, what would I
  improve if I ignored the handover, the requirements, and the written rules — all of which may
  be wrong?" Handovers and plans were written by an earlier session with less evidence than I
  have now (today the bisect's "model/quant limit" verdict and "gate needles at depth <=0.1"
  acceptance bar were both wrong). Say the answer out loud to the user even if it means
  disagreeing with the plan; don't just execute the checklist.

## 2026-09-04 (final Switchyard soak vs `81ab30e`, and the 1M multi-needle run)
- **A scheduler policy is graded on the wall clock it spends NOT scheduling.** `81ab30e` seats
  2.0x the lanes and prefills 1.6x faster than the tree that passed — and fails the soak,
  because 52–53 % of each phase produces no batch line at all (gaps of 492 s, 515 s, 624 s).
  Error rate and p95 say "15 timeouts"; the diagnosis only appears when you measure the
  wall-clock **gaps between consecutive batch-log lines**. Add that to every soak analyzer.
- **`py-spy dump --locals` settles "wedged or merely slow" in one call.** No instrumentation,
  no restart, no code change: it named the frame (`schedule_next_batch`), the hot leaf
  (`fast_compare_key` inside a 118 K-token radix walk) and the exact locals that decide the
  branch (`lane_cap: 0`, `seatable_lanes: 2`, `reqs: []`, `refusals: 2`,
  `blocked_fresh: False`). Under WSL `ptrace_scope=1` it needs `sudo <path to py-spy>` —
  `sudo env PATH=$PATH py-spy` does not resolve a `uvx`-installed binary; find the archive
  path first. CPU% cannot distinguish a livelock here: the loop reads ~106–109 % in every
  state, healthy or stalled.
- **A CPU replay gate is not an acceptance test for a scheduler policy.**
  `benchmarks/scheduler_replay.py` scored the failing commit at 2.49x tokens / 2.14x
  completions. It models no retained session leases, no decode residency and no idle timeout —
  which is precisely where the stall lives. Keep the live soak as the gate.
- **A resource budget must subtract what the pool has already given away.** `81ab30e` gates a
  fresh admit's finishability against `cache_manager.max_size - inflight_prefill_size`, i.e.
  the WHOLE pool minus prompts mid-prefill — not minus the KV held by requests already
  decoding, nor the retained/locked session prefixes that reclaim cannot evict. Admissions
  keep arriving until `token usage: 1.00`, then no lane can buy its next chunk and nothing can
  complete to free one. Same family as the two earlier cap bugs (§R6, `fad1fc4`): every time,
  the budget was expressed in a currency the guarded check does not spend.
- **One prefill, many questions: design long-context recall runs as a conversation.** A 1.04M
  prompt costs 1,815 s to prefill and ~5 s per follow-up turn once the prefix cache holds it
  (`cached_tokens` = prompt − 40). Six needles + a control + a combined question cost 34 min of
  GPU instead of ~4 h as eight separate prompts — and the 4 h `gpu_lock` cap makes the naive
  version impossible, not merely slow.
- **Ask each needle in more than one question shape before concluding it was not retained.**
  At 1.04M the model answered the depth-0.25 needle with the depth-0.05 code when asked
  directly, then produced `9,854,500 = 5,663,623 + 4,190,877` when asked to compare and add the
  two — so the "missing" needle was in the state all along. A single-question gate would have
  filed a retention bug against the kernels. (Same shape as the 262K bisect's wrong "model/quant
  limit" verdict.) Also grade a control key that is absent: it separates "cannot retrieve" from
  "fabricates", and this checkpoint passed it (`No belfry ledger code found.`).

## 2026-09-05 (Nemotron 3.5 decode launch config — a heuristic keyed on the wrong thing)
- **A launch heuristic that matches on a head shape silently excludes every other model.**
  `decode_launch_config` had four carefully measured branches, all requiring
  `head_dim==256, 16 q heads, 2 kv heads`; the *fallback* they all shared was a flat
  `kv_splits=8`, which on Nemotron's 32Q/2KV/D128 is a 16-CTA grid on 84 SMs — constant in
  context, so decode slowed linearly with prompt length (8.3-9.7x off the achievable per-layer
  time at 131K-1M). Write the fallback against the *machine* (`_grid_filling_splits`: CTAs per
  SM given the kernel's own grid formula), not against a constant; keep the measured branches
  as overrides. The general rule then reproduced the tuned Ornith split count independently,
  which is the check that it is a rule and not a second curve fit.
- **Derive the heuristic from the kernel's grid expression, not from intuition.** Stage 1
  launches `batch * cdiv(num_q_heads, min(16, group)) * kv_splits`; once that is written down
  it is obvious that splits is the only term that can scale with the GPU, and that two
  different-looking geometries (16Q/2KV/D256 and 32Q/2KV/D128) have the *same* 2 head blocks
  and therefore want the same split count.
- **"Bit-exact before/after" is the wrong gate for a split-K kernel.** Changing the split
  count changes the ORDER of the flash-decoding log-sum-exp reduction, so the outputs cannot
  be bitwise equal and a bitwise gate would have rejected a correct 9x speedup. Gate on
  agreement instead: max |Δ| vs the old config (1.22e-04 here, one sixtieth of a bf16 ulp)
  *and* vs the dequantized-pool oracle — the new launch was no further from the oracle than
  the old one, and at 131K closer.
- **A tile that is optimal at head_dim 128 can be 43 % slower at 256.** `BLOCK_N=64` + 8 warps
  won every Nemotron length and lost badly on the quantized D256 pool (0.500 vs 0.348 ms), so
  the tile is keyed on `head_dim` while the split count is keyed on the grid. Sweep the second
  geometry before generalizing from the first.
- **Give a graph-captured constant an env override before you need to A/B it.** The split
  count is baked into the CUDA-graph grid and the fp32 scratch at capture time, so it cannot
  be varied inside a live process: `FREETOKEN_DECODE_KV_SPLITS/_BLOCK_N/_NUM_WARPS` made the
  end-to-end before/after two runs of the SAME binary instead of a rebuild, and a one-line
  startup log line (`Triton decode launch: ...`) is what proves which arm actually ran.
- **A `/health` 200 does not mean the model is loaded.** FreeToken answers `/health` while the
  weights are still streaming and then 503s the first real request (`model is still loading`).
  Poll with an actual 1-token completion before starting a timed run.
- Kernel microbenchmarks are cheap enough to sweep exhaustively: 204 configurations across
  four contexts, three geometries and three batch sizes cost ~25 min of GPU with no weights
  loaded, and they predicted the end-to-end decode rate to within a few percent
  (`non-attention ms/token` inferred at one length extrapolated correctly to the others).
  Do the kernel sweep first; the server run then only has to confirm it.

## 2026-09-05 (soak vs `ea7ed7c`, and the first cross-engine oracle sweep)
- **A deadlock produces ZERO "gaps between batch lines".** The `gaps >= 30 s` analyzer written
  for the `81ab30e` stalls reported **0 gaps** inside the failing stage phase of `ea7ed7c` —
  because the silence starts after the *last* batch line and never ends, so there is no second
  line to measure against. Always report the **trailing silence** (last batch line → end of
  phase) and the *fraction of the phase after the last batch*, not only inter-line gaps.
  Here: last batch 5 m 35 s into a 50-minute run, 2,616 s of unbroken silence.
- **`/v1/stats` freezes when the scheduler does.** Two calls six minutes apart returned
  byte-identical `used_pages` and `completed` during the deadlock: the report is refreshed by
  the batch loop, so a wedged engine serves a stale snapshot that looks like a healthy one.
  Trust `py-spy --locals` over `/v1/stats` for "what is the pool doing right now"; use the
  stats only for the last-known-good value, and say so.
- **`py-spy --locals` twice, 40 minutes apart, is the cheapest proof of "deadlock" vs "slow".**
  `inflight_prefill: 222538` byte-identical in both dumps settles it in one line: not one token
  was forwarded in 40 minutes. Sample the same frame twice, far apart, before writing "stall".
- **Same bug family, third time: a budget checked only at admission is not a budget.**
  `fad1fc4` charged the wrong currency; `81ab30e` charged against the whole pool; `ea7ed7c`
  charges the right quantity but only *at the moment of admission*, so the same idle-lease
  tokens are counted as admissible for prompt A, then again for prompt B on the next pass. The
  invariant that fails is always about the **set already admitted**, never about the arrival.
  Ask "what re-validates this after the pass ends?" of every admission gate.
- **More lanes is not the metric.** Mean lanes per prefill batch went 2.37 → 4.71 → 6.57 across
  the three trees, and errors went 0 → 15 → 32. Report lanes next to error rate or it reads as
  progress.
- **A soak result can be un-measurable for the other change in the tree.** `acc91e9`'s decode
  win could not be read from this run at all (5 decode batches at 16 lanes, engine starved 92 %
  of the wall clock). Say "not measured", not "regressed" — and get the number somewhere the
  engine actually runs (the 262K oracle turn loop gave it: 105.5 tok/s median).
- **The cross-engine oracle paid for itself on its first run.** Both engines, independently,
  answered the depth-0.500 direct question with the planted near-duplicate twin. A
  single-engine run logs that as "FreeToken loses mid-depth needles at 262K" — the exact wrong
  verdict the 2026-09-04 bisect drew. Two engines turned it into `both-miss` /
  `interference-near` in one command.
- **Grade the failure, not the boolean.** Both `freetoken-only-miss` rows at 262K were
  composition failures on turns whose retrieval was correct — one added two correctly
  retrieved codes **off by one** (9,854,499 vs 9,854,500), the other produced the right sum and
  named the wrong key as larger. Read the answer text before filing anything.
- **Check the doc's paths against the host before scripting them.** `docs/oracle.md`'s Phase-A
  serve line names `--session-spill-dir /mnt/nvme/ft-spill`; there is no `/mnt/nvme` on this
  machine (one `/` on `/dev/sdd`). One `ls` before the run, not one failed 30-minute prefill.
- **At 1M, ask `code -> key` before concluding anything.** Direct `key -> code` questions pass
  1 of 6; leak-free **reverse** probes recover 5 of 6 of the same codes exactly. The needles are
  in state and the *addressing* is what fails. Six retention bugs would have been filed off the
  direct column alone. (Same shape as the 2026-09-04 combined-question recovery, now general.)
- **A long-context suite that is a conversation must be sized below the context ceiling.**
  `--target-prompt-tokens 1048576` against `--num-tokens 1048576` passes turn 1 and then fails
  turn 2 with `context_length_exceeded` — after 1,818 s of prefill. Every graded turn appends
  its question *and its reply*, and the server also reserves the decode budget. Subtract the
  whole conversation's growth from the target before starting a 30-minute prefill.

## 2026-09-05 (third scheduler-gate attempt: restore d685e99, then fix the invariant)
- **A CPU replay that lets a starved request die is not modelling the server.** The soak's
  600 s client timeout does NOT hand the engine's KV back: FastAPI's disconnect check runs
  only after the response generator yields a chunk (`stream_with_cancellation`,
  api_server.py:419) and a non-streaming handler is not cancelled on disconnect at all, so a
  request stuck mid-prefill is never aborted and keeps its pending entry, its table slot and
  every page it has forwarded for the life of the server. The replay modelled the timeout as
  a free abort, which gave every deadlock an escape hatch the real server does not have --
  and that is why it passed BOTH trees that then deadlocked live. Before trusting a
  simulator, ask what it lets go of that the real thing never does.
- **The invariant, not the throughput number, is the gate.** Every failed tree beat the
  replay's throughput floors: 81ab30e 7.05 M tokens, ea7ed7c 6.19 M, against d685e99's
  2.81 M -- and 81ab30e/ea7ed7c both failed the live soak. What separates them is a
  property, checked every pass: `owed(admitted set) <= obtainable`. ea7ed7c violates it on
  566 switchyard-stage passes; d685e99 and the fix never do. Add the property check before
  adding another floor.
- **"Admitted" is a set, and a per-arrival check is not a set check.** The fix is one line
  of accounting -- seed the adder's `reserved_size` with the standing reservation of every
  prompt already mid-prefill, so a request keeps costing admission until it finishes -- plus
  a cap on concurrent chunked prefills as the bound that survives an arithmetic slip. Charge
  it only to FRESH admits, and keep it out of `reserved_pages` (that is the per-chunk cap's
  budget; mixing them is the 6-lanes-to-2 regression of R6).
- **Throughput cleverness is what broke the last two attempts.** ea7ed7c's continue-past-
  refusals, admission aging, match memo, oversize skip and `admissible_size` were all
  defensible individually and together they re-sold reclaimable capacity once per pass. The
  restored tree keeps none of them. Third attempt: ship the smallest correct invariant.

## 2026-09-05 (Nemotron 3.5 prefill profile — a tile constant measured on another geometry)
- **Same bug, second kernel: a launch constant swept for one head shape becomes the silent
  default for every other.** The decode fix (2026-09-04) was `kv_splits=8` from a fallback;
  this one is `num_warps=4`, swept for the D=256 consumer `(64,32)` extend tile and applied
  to *all* geometries, while `_select_extend_tile`'s `head_dim<=128` arm independently
  hard-coded `BLOCK_M=128`. Neither half was wrong on its own; their product was a spilling
  kernel. **When two launch parameters are chosen in different places, check the pair.**
- **`n_regs` / `n_spills` off the compiled Triton kernel settle "why is this slow" in one
  call.** Walking `kernel.device_caches` after one launch gave 396 spill slots per thread at
  `BLOCK_M=128`/4 warps against 14 at 64 — the whole 2.46x, before any profiler. Do this
  before hypothesising about bandwidth: the microbench said 7.4 GB/s of a 960 GB/s part,
  which reads as "memory bound" and is in fact "spilling".
- **Fewest spills is not fastest.** `32/4` and `64/8` spill zero and are 50–70 % slower than
  `64/4` with its 14. Spills explain the *collapse*; the winner still has to be measured.
- **Derive the cap from the kernel's own accumulator shape, not from the winning number.**
  `BLOCK_M x BLOCK_DV / (32 * num_warps) <= 64` reproduces the measured winner *and* leaves
  every 8-warp device and every measured `head_dim>=256` branch untouched — a rule, not a
  second curve fit. (Same discipline as `_grid_filling_splits`.)
- **Two engine measurements plus the kernel's own ratio bound the engine's share without
  instrumenting the engine.** `slope = s_att + s_other` with a before/after A/B and the
  microbench's 2.46x solves to `s_other = 0 ± 0.2e-3 ms/token` — i.e. ±13 s of a 1M prefill
  for KV grow, page allocation and the `O(prefix)` page-index build combined. No torch
  profiler, no model.py hook, no per-layer CUDA events. Fit the curve you can already
  measure before adding instrumentation.
- **`input throughput (token/s): X instant` on a `Prefill batch` line IS a per-chunk timer.**
  `#new-token / X` is that chunk's wall time (the reporter runs after the drain barrier) and
  the running sum of `#new-token` is its prefix, so a per-chunk cost-vs-position regression
  comes out of an ordinary server log with no code change. r2 0.999 over 127 chunks.
  Exclude chunk 0: Triton autotune + MoE cache first-touch make it a 4.8 s outlier.
- **Do not pipe `scripts/gpu_lock.sh` into anything.** Its exit trap runs `pkill -9 -g $$`,
  which kills the reader too: the job completes, the file is written, and the caller sees
  `Killed` / exit 137 and concludes the run died. Redirect to a file, then read the file.
- **A background `until` loop does not make the agent wait.** Backgrounded waits return
  immediately, so polling with them burns turns without advancing the clock; a *foreground*
  `until ! kill -0 $PID; do sleep 20; done` is what actually blocks on a 13-minute GPU job.
- **Sweep the cheap length, confirm the expensive one.** 66 configurations at a 131K prefix
  cost ~8 min; the 6 survivors at 262K/524K/1M cost ~6 min and the ratio was flat (2.41 ->
  2.46x) — the full grid at 1M would have been ~30 min of Triton compiles and timing for the
  same answer.

## 2026-09-05 (the soak that finally passed, `4a99e34`)
- **The invariant is cheap enough to leave on in a live soak.** `FREETOKEN_SCHEDULER_INVARIANT
  =warn` evaluates `owed(admitted set) <= obtainable` before every scheduling pass and cost
  nothing measurable (busiest process at 109.9 % CPU, the same ~106 % every healthy run has
  recorded). Zero warnings across ~3,141 passes is a stronger statement than any throughput
  number: it says the property held, not that the run happened to survive.
- **Report trailing silence and the fix looks different.** With the §T lesson wired into
  `gaps.py`, the passing tree reads as "trailing silence 1 s, scheduling wall clock 97.2 % of
  the phase" — a positive measurement, not merely the absence of the failure. Instrument the
  failure mode you just diagnosed *before* the run that is supposed to close it.
- **A silent `continue` cannot be verified; infer it or instrument it.** `max_chunked_prefills`
  logs nothing, so "it never bound" had to come from a log identity: a prefill pass with
  `#cached-token > 0` necessarily admitted a FRESH request (continuations book 0 cached), which
  proves `chunked_inflight < 8` at that pass — 282/2,091 such passes, median 2 s apart. That
  bounds the answer honestly; it does not prove it. Ship the counter with the knob.
- **Fewer lanes, zero errors, best latency.** Mean lanes per prefill batch 2.37 -> 4.71 -> 6.57
  -> **1.83** with errors 0 -> 15 -> 32 -> **0** and stage p95 200.7 s -> 145.8 s. The
  reservation seats fewer prompts on purpose; that IS the fix. Never grade a scheduler on lanes.
- **A 54 s hole in the batch log is not automatically a stall.** The one gap of the run was ten
  session spills (0.45 GiB each) and the restore of 589,680 cached tokens across 6 lanes. Read
  the non-batch lines inside a gap before naming it.
- **Prove a leak fix at the edge you can observe.** The disconnect-abort (`ff470e7`) has no
  server-side counter, so it was verified by dropping a raw socket mid-prefill and watching
  `/v1/stats.requests.active` go 1 -> 0 in 7 s, plus "0 client failures on 2,070 requests" as
  the no-spurious-aborts half. Two cheap observations beat one missing metric.

## 2026-09-05 (native-Q8 extend attention — a ticket closed negative in 35 GPU minutes)
- **Measure the denominator before filing a "% of peak" ticket.** The prefill profile filed
  "the extend kernel is at 31 % of peak (70.4 of ~225 TFLOP/s)" off the RTX 5080's spec sheet.
  The part actually does **123.0 TFLOP/s** bf16 through cuBLAS and **118.4** through Triton's
  own `tl.dot` — the kernel was at 57-60 % of achievable, and the bf16-KV variant at 72 %,
  which is where a good flash kernel sits. One 30-line `torch.matmul` benchmark would have
  stopped a projected "1.5-2x" from being written down. Vendor tensor-TFLOPS figures are
  fp8/fp4 and/or sparse; never use one as a kernel's denominator.
- **Never assume int8 tensor cores are 2x bf16.** `torch._int_mm` on sm_120 (consumer
  Blackwell): 128.0 TOP/s against 123.0 TFLOP/s bf16 = **1.04x**. The 2:1 int8:bf16 ratio is a
  datacenter-part property. A "native int8 dot" plan whose payoff is the dot is dead on
  consumer silicon before any code is written.
- **The cheapest upper bound on a dequant optimisation is the same kernel over an
  unquantized pool.** q8_0 vs bf16 KV, same launch, same shapes: **1.206x, flat at
  131K/262K/524K/1M**. That bounds *every* possible native-Q8/dequant rewrite, perfect ones
  included, in two microbenchmark runs and with no new kernel. Do this before designing one.
- **Fold a per-block quant scale after the dot only when the tile's row count is below the
  quant block size.** Dequantizing K in place costs `BLOCK_D*BLOCK_N` multiplies per KV tile;
  folding the scale onto `D/QBLOCK` int32 partials costs `BLOCK_M*BLOCK_N*BLOCK_D/QBLOCK`.
  Folding wins iff `BLOCK_M < QBLOCK`. That is exactly why `_Q8_NATIVE_QK` pays in decode
  (`BLOCK_M` = 16 query heads) and loses in extend (64 tokens) — the *absence* of the fast path
  in the second kernel was arithmetic, not an oversight. Always check the direction of the
  inequality before porting an optimisation between two kernels.
- **A scale on the reduction axis cannot be folded at all.** q8_0's V scale varies along `n`,
  which is PV's reduction dimension, so half of any "native Q8 attention" is unreachable by
  construction. Write the index expression out before estimating a gain.
- **`num_stages>1` never helping is not proof the kernel is ALU-bound** — loop-variant masks on
  the loads inhibit Triton's pipeliner. Re-sweep stages *after* removing the masks; here it
  still lost at all 18 tiles, which is what makes "not latency-bound" an observation instead of
  an assumption.
- **Grade a micro-optimisation against the fp32 oracle, not against the old kernel.** The 1.14x
  combination looked fine at 1.8e-4 from the production output and was **2.0x worse than
  production against the oracle** (7.6e-4 vs 3.7e-4). The culprit: dequantizing into bf16
  multiplies by a scale that is stored fp16 (10 mantissa bits) in a format with 7, so every
  32-element block picks up a systematic relative error. Agreement with the thing you are
  replacing is not accuracy.
- **A negative result is a deliverable, and it is cheap if you sequence it right.** Peak +
  bf16-KV control + variant sweep + oracle = ~35 GPU minutes and no model load; the planned
  end-to-end A/B (two servers, a 1M prefill, a needle) was ~2 GPU hours and had nothing left to
  measure once the kernel was not going to change. Ask "what would the e2e run answer that the
  microbench has not?" before booking the lock.
- `ncu` needs admin-enabled performance counters (`ERR_NVGPUCTRPERM`) and there is no root on
  this host. Do not plan a profile-counter step here; a wall-clock A/B against a variant that
  removes the suspected term answers the same question and is not permission-gated.

## 2026-09-05 (prompt-lookup speculative decoding — NO-GO in ~40 GPU minutes)
- **Under greedy decoding, a drafter's acceptance rate is computable offline from an ordinary
  transcript.** Speculative decoding is verified against exactly the greedy continuation, so
  λ(n, k) is a deterministic function of a plain generation — no verify step, no engine change,
  no second GPU run. One 3-prompt generation plus a CPU replay answered half the go/no-go.
  Ask "is this quantity a function of something I can already record?" before building the thing.
- **Record token *ids*, not text.** The offline `LLM` returns `token_ids`; the HTTP API has no
  logprobs, so an API-based transcript would have forced a retokenization approximation into the
  one number the decision rests on.
- **When verification is expensive, draft for precision, not recall.** The published
  prompt-lookup setting (n=3, maximize hit rate) assumes a nearly-free verify step. Here n=3
  fires on 12% of code steps and is right 23% of the time, costing 12–14%; **n=8** drops the
  code/prose draft rate 30x, keeps 93% of the copy-class λ, and turns a 12% regression into
  0.0%. A hyperparameter default imported from a paper carries the paper's cost model with it.
- **A monkeypatched Python hook records nothing on a CUDA-graph path.** The router probe on
  `NemotronHMoE.forward` captured zero decode tokens: graph replay never enters Python. Routing
  is a pure function of the residual stream, so re-prefilling the transcript recorded the same
  thing. Check whether the path you are hooking is captured before believing an empty result.
- **CUDA events measured 304 ms for a forward inside a 279 ms call.** A subinterval cannot
  exceed its container — that contradiction, not any profiler, is what proved the events were
  straddling host idle rather than timing kernels. Sanity-check a timing against a wall clock
  that must bound it.
- **`perf_counter` inside `model.forward` split the answer in one run.** Host 303.6 ms vs
  GPU-event 310.6 ms for the same 1-token extend: the stream is *waiting on the host*, so it is
  a CPU-side fixed cost, not bandwidth. Then per-mixer `perf_counter` put 267 of 290 ms in the
  23 MoE layers — 11.6 ms per MoE layer per forward, **flat from 1 to 32 tokens**. Two hooks,
  no `ncu` (which is permission-gated on this host anyway).
- **Flat in the variable you are sweeping is the whole finding.** Once the extend forward cost
  the same at m=1 and m=32, it was a fixed per-forward cost, and no acceptance rate could pay
  it. Sweep the width first; the shape of the curve decides faster than any single point.
- **A cost proxy must be measured on the path the feature would take.** Phase 4 (MTP) priced its
  verify step with a bs=2 *decode* step: 1.63x. A real verify step takes the *extend* path:
  42x. Same model, same host, 25x apart — because the decode path reuses the expert cache and
  the extend path re-gathers per layer per forward. Before trusting a proxy, name the code path
  it exercises and check it is the one the feature will run on.
- **A fixed cost hidden by latency-hiding reappears the moment the batch shrinks.** 290 ms of
  host work per prefill forward is invisible behind an 8 192-token chunk's ~861 ms of GPU work
  and is the *entire* cost of a 9-token verify step. The same measurement says prefill chunks
  below ~3K tokens go host-bound — a second result the feature investigation was not looking for.
- **The part flagged as hard was the easy part.** Mamba-2 state rollback was the item's stated
  risk; the answer is to never advance the live state at all (verify into the already-reserved
  ping-pong slot, cache ~5 MiB of scan inputs, commit with one varlen SSD scan over the accepted
  j for ~0.3 ms). Write the design down even when the verdict is NO-GO: it is what makes the
  follow-up ticket executable instead of a re-investigation.
- **Do not pipe `scripts/gpu_lock.sh`, and read the exit code with that in mind.** Every run here
  ended `Killed` / 137 from its own cleanup trap *after* writing complete output. Grep the
  redirected file; do not conclude the job died.

## 2026-09-05 (WSL OOM restart, 08:59)
- Kernel OOM killed the gpu_lock worker (16 GB shmem + 2 GB anon) at 32.5 GB user peak,
  then WSL tore the session down. Cause: a CPU-side agent running pytest/torch (2–7 GB)
  while the model was loading. gpu_lock only protects the GPU job from being the last
  straw; it does not stop a sibling from pushing the total over.
- RULE: while any model is loaded, sibling agents run NO pytest/torch imports and stay
  under ~1 GB RSS. Test suites for CPU work run before the GPU job starts or after it ends.
- RULE: the scratchpad does not survive WSL restarts; soak drivers/scripts worth keeping
  go under benchmarks/ or scripts/, not scratchpad.

## 2026-09-05 (the soak that closed §R7 ticket 1, `13af13d`)
- **A benchmark harness that lives only in a session scratchpad is one host event away from
  never having existed.** A WSL OOM restart at 08:59 took `scratchpad/soak7/` — drivers,
  analyzers, and a running soak's logs — with it. The drivers now live at
  `benchmarks/switchyard_soak/` with `runs/` gitignored. If a script produced a number you
  wrote into a results file, that script belongs in the repo.
- **Instrument the failure that killed you, in the driver.** `run.sh` now refuses to start below
  26 GiB `MemAvailable` and `sample.sh` logs it every 5 s. The floor during the passing run was
  5.1 GiB with the model resident — so the *check* has to happen before the load, not during it.
- **Grade the fix on its own signature, not on the headline.** §R7 ticket 1 was "61 % of stage
  passes carry `#new-seq: 1`, `#new-token ≤ 512`, `#queue-req ≥ 8`". `analyze.py` counts exactly
  that predicate, so `812bc57` closes as **0/1,202 passes** — a statement about the mechanism.
  p95 −25 % is the consequence, and on its own it would have been consistent with luck.
- **When a metric lands in the band that marked previous failures, check WHICH ROUTE the band
  was measured on before calling it a regression.** Passthrough lanes 4.92 sits inside the
  4.71–6.57 band of two failing trees — but that band was a *stage-route* measurement, and this
  run's stage is 3.43 with 0 errors and the best p95 on record. Same number, different
  population.
- **A per-chunk rate can fall while goodput rises.** Passthrough `input throughput … instant`
  median went 1,838 → 1,547 while requests went 1,600 → 1,904 and p95 fell 25 %: more lanes per
  pass, smaller median `#new-token`, 89.6 % prefix reuse. Never read the per-chunk median as
  engine throughput; divide total new tokens by the phase window instead.
- **Fixing an admission cap makes the thing it capped a free variable.** The chunk divisor was
  also, accidentally, a lane brake. With it gone, nothing bounds lanes but the pools — so the
  soak checklist gains "record mean lanes" permanently, not because lanes are bad but because
  the quantity is now unpinned.
- **Check the serve line actually exercises the code you fixed.** The batch_memcpy probe fix was
  supposed to be confirmed by grepping the soak's server log; `moe_prefill_hit_d2d=False` in the
  P2 profile means that path never ran. A unit test that reproduces the hostile precondition
  (queued ambient-stream work before `load_batch_memcpy()`) is the evidence — and it is cheaper
  and repeatable. Read the ServerArgs dump before planning to grep for a feature.
- **A 31 s hole with 291 non-batch lines in it is a spill burst, not a stall.** Second run in a
  row where the only gap ≥ 30 s was session spill/release under admission pressure
  (`#queue-req 11`, `#running-req 3`). Read inside the gap before naming it; the §U lesson held.

## 2026-09-05 (extend-path MoE ticket — a host OOM and a malformed test bank)
- **Never run pytest, or import torch at all, while a model is loaded on this host.** A CPU
  test sweep overlapping the soak's expert-bank build OOM-restarted WSL and took the whole
  session's background jobs with it. Check `pgrep -f "ft serve"` / `nvidia-smi` FIRST; if
  anything is loaded, stay under ~1 GB RSS and do desk work. CPU test runs are a GPU-window
  activity here, not a free one.
- **The 11.6 ms/MoE-layer extend cost was bytes, not planning.** `_prefill_routed` streams
  every expert of the layer into its double buffer on every forward and nothing in that
  movement path reads `topk_ids` -- which is exactly why it was flat from 1 to 32 tokens.
  128 x 5.612 MB = 718 MB per layer, 16.5 GB per forward (the whole 15.4 GiB bank set), and
  16.5 GB / 267 ms = 61.9 GB/s: a saturated PCIe 5.0 x16 link. The n-gram write-up rejected
  bandwidth by pricing the *routed* experts (138 fetches) instead of the *streamed* ones
  (2,944). **When a cost is flat in the variable you swept, price the thing that is NOT a
  function of it before concluding "host overhead".**
- **Check a candidate fix's arithmetic against the model's own documented totals.** "15.39 GiB
  per forward" landing exactly on `docs/nemotron.md`'s "the NVFP4 routed-expert banks are
  15.4 GiB" is what turned a hypothesis into a diagnosis, before any GPU time.
- **A reference output full of NaN makes an equality test vacuous in both directions.** Four
  CUDA tests "failed" on synthetic NVFP4 banks whose `*_global` was `[E, 1]`; the real layout
  is per OUTPUT ROW (`[E, I]` / `[E, H]`, see `tests/moe/test_nvfp4_backends.py::
  _make_ungated_sources`), so `stride(0) = 1` made the kernel read overlapping rows and every
  output was NaN or 1e15. **Assert `isfinite().all()` and `abs().max() > 0` on the reference
  before comparing to it** -- and build synthetic quantized banks by copying the repo's own
  builder, never by guessing the shapes from the bank schema names.
- **`moe_align_block_size` is the first thing to exonerate, not to suspect.** Printing its
  `sorted_ids`/`expert_ids`/`ntpp` for the compact and the wide call took one 10-line script
  and proved they were identical (same sort, remapped expert ids), which moved the search to
  the data. Dump the intermediate before theorising about the kernel.
- **A `git worktree` has no compiled CUDA extensions.** `freetoken.kernel._pinned_tensor`,
  `_cpu_moe` and `_pageable_stage` live only in the main checkout's `python/freetoken/kernel/`;
  with `PYTHONPATH=<worktree>/python` every `set_bank_sources` test dies on an ImportError that
  reads like a broken install. Symlink the three `.so` files in (they are gitignored).
- **`str.replace(old, new, 1)` lands on the FIRST match, and near-duplicate method bodies make
  that the wrong one.** A three-line `ensure_experts` / `copy_missing` / `_expert_gemm` prefix
  appears in both `_decode_routed` and the new `_cached_extend_routed`, so an nvfp4-only assert
  meant for the second went into the first and broke bf16 decode -- caught only because
  `tests/moe/test_offload.py` covers it. When patching by text, anchor on something unique to
  the target (a docstring line), or assert the match count is 1 AND that it is in the right
  function.
- **The tree usually already documents the constraint you are about to hit.** Pointing the
  grouped prefill GEMM at the decode slot cache faulted with an illegal memory access (sgl
  `moe_align_block_size` over ~1,800 "experts"); `_expert_gemm`'s ds_fp4 branch had said for
  months that its small-chunk slot path uses the GEMV because "sorting over the full slot cache
  would drown in padding". Read the neighbouring branch's comment before generalising a kernel.
- **A benchmark that reuses one prompt measures the radix tree, not the model.** `base + tail`
  repeated across arms is served from cache and an m-token extend silently becomes a 1-token
  one. Give every timed call a fresh tail and key the row on the recorded `"<m>/extend"` forward
  bucket, so the shape is proven rather than assumed.
- **Greedy equivalence has to be probed with `ignore_eos=False`.** The 131K needle arm diverged
  at token 0 -- but `ignore_eos=True` forces both arms past `<|im_end|>` into a degenerate
  repeat whose phase carries no signal. With natural stop, the realistic shape (a long prompt
  whose last chunk is short) was token-identical, and only the all-new-kernel case drifted, at
  token 60 of 255.
- **Ship the arm that works and file the faster one.** The GEMV variant was 23-27x on the MoE
  and took one line; chasing the grouped GEMM's further ~2.3x of HBM would have cost another
  session and had a fault to design around. Same discipline as the scheduler's third attempt.

## 2026-09-05 (building `--speculative ngram` — three measurement traps in one feature)
- **A feature that "does nothing" and a feature that "never fires" look identical from the
  outside.** The first measured run reported `draft_rate 0.0` on the copy class where the
  offline replay said 0.353, and the only reason it was diagnosable at all was adding
  per-reason decline counters (`declined_no_slot`, `declined_stale_match`, `declined_budget`,
  `declined_shape`). **Instrument every early `return False` in a gate with its own counter
  before measuring it**, not after the number confuses you.
- **`LinearStatePool.num_free_slots` is normally ZERO during steady-state decode.** A request
  holds live + 2 ping-pong and the radix tree owns every donated snapshot, so "is there a free
  slot?" is the wrong question — the right one is "can `ensure_mamba_slots(1)` make one?"
  (tier 2, LRU-evict an unlocked tree snapshot). Any new consumer of that pool must escalate,
  and must NOT escalate to tier 3 (`reserve_mamba_slots` spills a session lease) for an
  optimisation.
- **The design doc said "reuse the already-reserved ping-pong pair"; both slots were live.**
  One holds the tool-call anchor freeze, the other is the next radix chunk snapshot's
  destination. **A slot described as "scratch" in a design note is not scratch until you have
  read every writer of it.**
- **A/B without a repeat of arm A is not an A/B.** `on != off` was unattributable until an
  `off2` control arm ran last and came back identical to `off` on 4/4 prompts — which is what
  turned "speculation might be buggy" into "the engine is deterministic and speculation
  genuinely changes the stream". One extra arm, no extra model load.
- **When two paths must agree but cannot be compared end-to-end, compare the INTERMEDIATE they
  share.** A greedy diff cannot separate "the state commit is wrong" from "extend kernels
  reduce in a different order than decode kernels". Replaying all m positions and comparing
  against what the verify forward itself wrote (`SpecScanCapture.replay_error`) does: it came
  back **0.000e+00** and closed the question in one run. Build that check while you build the
  thing, not after a confusing result.
- **Warm the arm, not just the cache.** The first run measured off=84 tok/s against on=123 and
  the "1.46x" was entirely a cold first arm: a 1-token generate warms the prefix tree but not
  the expert cache or the decode path. A short decode warm-up per prompt made both arms land
  within 1 %.
- **Measure over the window the offline study used.** The copy class drafted on 0.8 % of steps
  at `--max-tokens 256` and 8 % at 1 024 — the model spends its first ~200 tokens reasoning and
  only then starts copying. A short generation had measured the preamble, not the phenomenon.
- **A cost ratio measured at one context length is not a constant.** A verify step's extend
  attention reads the KV history once per query token, so it is ~3.6x a decode step at short
  context and **~10x at 131K** — which flips a +11 % win into a −20 % loss. The fix is to
  estimate the ratio online (the gap between consecutive scheduler steps IS a decode step's
  wall time) rather than to threshold on context length, because the ratio also depends on KV
  dtype, attention backend and acceptance.
- **Speculative decoding on this engine is NOT token-identical to non-speculative decoding, and
  no implementation can make it so.** The verify step argmaxes extend-path logits and commits
  state with the SSD scan; a decode step uses the graphed decode kernels and the recurrent step.
  Say this in the write-up as a property of the architecture, so the next person does not spend
  a session hunting a bug that is a reduction order.
- **A self-tuning gate needs the RIGHT estimator on each side, and "robust" is not one answer.**
  The break-even gate compares a verify step against a decode step. `verify_ms` must be a
  running MINIMUM (a few dozen samples, and the first pays Triton autotune -- as an EWMA that
  one-off read 11.35 where the steady state is 4-5, closed the gate, and then starved it of
  the samples that would reopen it). `decode_ms` must be an EWMA (hundreds of samples, and the
  scheduler loop's gap is not uniform, so its minimum sits far below a real step -- as a floor
  it gated out 264 of 278 drafts and turned +11 % into -0.3 %). Each wrong choice cost a GPU
  run. **Ask how many samples a quantity gets and what its outliers look like before picking.**
- **Make a gate hard to close and easy to leave open when a false close costs more than a
  false open.** Requiring two timed samples plus a 25 % margin took the copy-class declines
  from 264/278 to 5/59 while still shutting off at 131K.
- **Measure the feature over the window the projection assumed.** The go/no-go's 1.63x used
  1 023 output tokens; measuring at 256 caught only the model's reasoning preamble and put the
  copy-class draft rate at 0.8 % instead of 8 %.
- **Back the per-step cost out of the throughput before believing a component measurement.**
  The extend forward is 27-30 ms, but `631*7.4 + 54*V = 7.48 s` puts the end-to-end verify
  step at ~52 ms -- **~40 % of it is not the forward** (drain, batch prep, a 46-launch eager
  commit). A component benchmark is a lower bound on a feature's cost, never the cost.

## 2026-09-05 (MoE prefill GEMM — the redundant *load*, not the arithmetic)
- **A per-block quantization scale is a `K/QBLOCK`-sized tensor. Load it at that size and
  broadcast; never at the tile's K.** The NVFP4 prefill K-loop indexed the scale with
  `byte_idx // 8`, producing a `[BLOCK_KB, BLOCK_N]` tile in which every row repeats eight
  times — one e4m3 scale covers 16 k-values = 8 packed bytes. Loading the distinct
  `[BLOCK_KB/8, BLOCK_N]` rows and `broadcast_to(...).reshape(...)` was **1.73x on its own**
  and took the kernel's shared memory 28 KB -> 12 KB. The *multiply* has to happen at the
  tile's K; the *load* does not, and the compiler cannot CSE a load.
- **Ablate to a wrong-but-fast bound before optimizing further.** A `FAST=16` arm that skips
  the scale entirely (garbage output) ran at 17.41 ms against the fixed kernel's 16.67 —
  i.e. the fixed kernel is already past the "scale is free" bound, and every further idea on
  that axis (fp16 product, serializing the two dequant tiles) was measured at 0 %. One
  deliberately incorrect arm retired four hypotheses.
- **Look for a hardware instruction before hand-rolling the bit twiddle — and look in the
  competitor's source.** `cvt.rn.f16x2.e2m1x2` (PTX 8.6, Blackwell) decodes a packed NVFP4
  byte to an `f16x2` in one op; it was found by grepping the installed *flashinfer* wheel
  (`moe_w4a16_fp4_helpers.py`) for `e2m1`, which is where b12x's 78 TFLOP/s came from. In
  Triton it is 12 lines of `tl.inline_asm_elementwise` and it is bit-identical to the
  arithmetic decode for all 256 byte values, so it needs no accuracy gate at all. (`.reg .b8`
  needs `cvt.u8.u32`, not `cvt.u16.u32` — ptxas says only "Arguments mismatch".)
- **A tile sweep can be a *negative* result that is worth its GPU minutes.** 68 tiles at
  M=8192 bought 1.12x against a 3.6x gap; that is what said "the structure, not the launch"
  and redirected the effort. Report the exhausted sweep, do not quietly drop it.
- **`nan` in a comparison table is usually a missing row, not a NaN.** The ticket "b12x
  returned nan at M=8192" was `bench_nvfp4_moe_kernels.py` printing `float("nan")` for every
  column of a backend that contributed no rows. Before debugging a numeric fault, check
  whether the number was ever computed — one grep of the reporting code, no GPU.
- **A skew knob is cheaper than a real capture, and it can retire the requirement.**
  Dirichlet-alpha routing at 0.25/1.0/4.0 moved M=8192 by 0.7 %, because 49,152 routed rows
  over 128 experts leave every expert with hundreds whatever the skew. That justified *not*
  spending a model load on capturing real `topk_ids` — say which experiment you skipped and
  why, with the number that licenses it.
- **`pgrep -f <script name>` matches the `bash -c` wrapper that contains the name**, so
  "is my job still running?" answered yes for 35 minutes after it had finished. Same family
  as the 2026-09-04 `pkill -f` rule: check `ps -o etime -p <pid>` on the *real* pid, or look
  at the artifact's mtime. The tuner had written its config tables 16 minutes in.
- **`gpu_lock.sh` kills the job before Python flushes a buffered stdout**, so a wrapped
  script that does not `exec >` its own log (or run `python -u`) leaves a **0-byte** log even
  on a fully successful run. Both halves of the rule are needed: redirect *inside* the
  wrapped script AND use `-u`.
- **When the whole change is bit-exact, say so and drop the agreement gate.** The scale
  broadcast reads the same values and the hardware FP4 decode produces the same codes, so
  `max|Δ| = 0.000e+00` at every tile and the 131K/262K needle answers are byte-identical
  between the two arms. That is a stronger statement than any `assert_close` tolerance, and
  it is what makes a launch-table rewrite safe to ship without a quality re-gate.

## 2026-09-05 (the 1M cross-engine oracle: a second engine that cannot be run is not an oracle)
- **Verify prompt identity on CPU with `record --build-only` BEFORE acquiring the GPU lock.**
  8 s of tokenizer work reproduced the FreeToken 1M haystack sha (`38f517c3e3b06ae3`) and
  proved the two legs would be comparable. Doing this after a 20-minute prefill is how a run
  gets thrown away.
- **A capacity failure on a GPU does not always look like an OOM. On WSL2 it looks like a
  silent 3–13x slowdown.** `llama-server -c 1052672` started fine, answered `/health`, and
  processed its first 4,096-token chunk in 27.3 s instead of 2.05 s, because the driver pages
  the overflow to host RAM instead of failing the allocation. **Always compare the first
  `prompt processing` line against a known-good run at a shorter context** — that one line
  said "this will take 20 hours" 90 seconds into a job budgeted at 4.
- **`nvidia-smi` free memory is useless as a residency check under WSL2 oversubscription.** It
  read 15,956–15,960 MiB used / 18–22 MiB free at `--n-cpu-moe` 14, 20 *and* 23 — i.e. at three
  weight footprints 4 GiB apart, all three of which behaved very differently. The honest
  residency probe is **throughput on the first chunk**, not a memory counter.
- **Find the offload floor from the file, not by bisecting runs.** `gguf.GGUFReader` gave
  685.1 MiB of routed experts per MoE block × 23 blocks in ten seconds, which says exactly what
  `--n-cpu-moe 23` can buy (4.1 GiB over the documented 14) and that 23 is the floor. Two of
  the three GPU attempts were avoidable with that number in hand.
- **When the top rung is unreachable, look for the rung where the phenomenon *starts*.** The
  1M question was "is the direct-probe collapse ours or the model's". The collapse begins at
  524K (direct 5/6 → 1/6 between 262K and 524K), which llama.cpp *can* run — so the 1M question
  was settled at 1/20 the cost and with a real oracle instead of no oracle. Ask "what is the
  cheapest length that reproduces the shape" before defending the expensive one.
- **Two engines returning the *same wrong number* is the strongest possible negative result.**
  Four direct probes at 524K returned the key's near-duplicate twin byte-for-byte in both
  NVFP4-FreeToken and Q4_0-llama.cpp. No amount of single-engine classification is worth that
  one observation.
- **A runbook whose serve line omits the flag its own check depends on will condemn good
  runs.** `docs/oracle.md`'s Phase-A profile has no `--enable-cache-report`, so the 524K
  recording came back `cached_tokens: 0` on every turn — and the same document says that means
  "the run is measuring re-prefill, not recall". It was not: TTFT was 2.46 s on a 524K prompt
  and the server's own log said `#new-token: 55, #cached-token: 524287`. **When a documented
  red flag fires, corroborate it with an independent measurement before throwing the run away
  — and check whether the flag can even be true given how the server was started.**
- **A zero that can mean three things is a design bug, not a data point.** `openai_api.py`
  returns 0 when cache reporting is off and then *omits* `prompt_tokens_details` entirely, so
  "reporting disabled", "genuinely nothing cached" and "field absent" are the same bytes on
  the wire. I spent a subagent proving there was no regression to find. Fields that gate on a
  config flag should say so in the payload, or the consumer should record "not enabled".
