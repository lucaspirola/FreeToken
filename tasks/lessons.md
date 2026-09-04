
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
