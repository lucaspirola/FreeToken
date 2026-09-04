
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
