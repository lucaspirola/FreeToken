
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
