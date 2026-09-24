# ts-local: expandable_segments on WSL, owner's RTX 5080, 2026-09-24/25

* `repro-wsl.txt` and `repro-wsl-ebf63e6.txt`: `wsl_expandable_repro.py` at 0 MiB.
  * cuMemCreate with NONE and POSIX_FD works; FABRIC returns INVALID_VALUE.
  * torch 2.11 expandable_segments works through the env var (IPC unset, 0 or 1).
  * It also works through the engine's runtime `_set_allocator_settings` path (IPC unset or 0).
* `tsw-mirror-1m-*`: the WSL server gate on ebf63e6, i.e. 583afc8 plus expandable_segments on WSL.
  * Nemotron serve-default, auto pool, reserve 256, 8K + 1M, two passes, main venv.
  * Run from the detached worktree `ts-measure`, at ratio 1.00 and port 1920, 0 MiB before the arm.
  * Measured with `measure.sh`, with `cgroup-sampler.sh` (exp/dynamic-transient 434750d)
    alongside: `tsw-cgroup-samples*.tsv` and the per-mapping snapshots in `tsw-maps/`.

## Gate (a2c850f7's order)

| check | result | |
|---|---|---|
| 1. journal | "Enabled expandable_segments", no "WSL detected", no "unknown error" | PASS |
| 2. transient vs allocator peak | "Prefill headroom: transient 0.59 GiB ... allocator peak 0.59 GiB"; it was 1.00 vs 0.59 | PASS |
| 3. captures / faults / OOM at 1M | captures=1, coverage faults 0, starved 0, no OOM, no Traceback, 0 prefills at the decode level | PASS |
| 4. decode / prefill vs ck4m-mirror-1m (583afc8) | 8K 193.0 / 180.0 vs 180.5 / 174.7; 1M 88.7 / 98.7 vs 88.5 / 92.1; 1M TTFT 793 / 794 s vs 794 / 795 s | PASS |

The measured transient shrank by 0.41 GiB, and that VRAM went to the arena: 2189 GPU residents
against 2173, and 1850 pool rows / 9.69 GiB against 1866 / 9.77.

The first start logs one FutureWarning:
`torch.cuda._set_allocator_settings is deprecated. Use torch._C._accelerator_setAllocatorSettings`.

ram_gib is 13.39. It is host-wide and was taken on a busy host. Cgroup anon is 12.70 at ready
and 13.35 after 1M p2.

## The aux-process anon growth during 1M (coordinator's question)

Processes in the unit:
* `ft serve` main process (pid 151646): HTTP server and tokenizer manager.
* scheduler (151794): gloo/tcpstore threads.
* detokenizer (151795): a spawned child with 4 ZMQ contexts.

Anon in GiB, sampled at the first sample after each request (`tsw-cgroup-samples.procs.tsv`):

| process | ready | during 1M p1 | after 1M p1 + 8K p2 | after 1M p2 |
|---|---|---|---|---|
| main / tokenizer | 0.40 | 0.97 | 0.75 | 0.92 |
| scheduler | 11.48 | 11.48 | 11.48 | 11.48 |
| detokenizer | 0.59 | 0.59 | 0.94 (13 s after 1M p1 finished) | 0.94 |

* Tokenizer manager:
  * It gains about 0.57 GiB as soon as the 1M prompt arrives, in unnamed anonymous mappings,
    and the `[heap]` grows only a little (158 to 182 MiB).
  * These are the working buffers for tokenizing and holding a 1M-token prompt. About 0.2 GiB is
    returned after the request.
* Detokenizer:
  * Its `[heap]` grows 374 to 740 MiB (+0.36 GiB) when the finished 1M request reaches it.
  * It does not grow again on the second 1M request, so it is a glibc heap high-water mark that
    is kept, not a per-request leak.
* The scheduler holds the pool and is flat.

Bounded: the second 1M request adds nothing beyond the first.
