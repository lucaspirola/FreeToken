# RAM gate re-run: mirror-1m on exp/reorg 31b8efb, local RTX 5080 WSL (2026-09-25)

`checkpoint1.sh ck4g` with `ARMS=mirror-1m` (8K + 1M, two passes, needles/recall post), the cgroup
sampler (`cg.tsv`), host load every 60 s (`hostload.txt`) and page-cache residency (mincore) of
every file under the candidate directories before start, at ready, after 1M p1 and at the end
(`pcache-*.tsv.gz`, `pcache-diff-*.txt`; tools `pcache.py`, `pdiff.py`, driver `run.sh`, dirs in
`dirs.txt`). The kernel .so files are round 2's (built after a658c8c), as ck4r2 used.

## Preconditions
* GPU 0 MiB, freetoken-serve inactive, embedder stopped, no pytest (preflight).
* Waited for a quiet host (1-min load < 3 and MemAvailable >= 23 GiB, polled every 3 min):
  load 10.5 / 4.3 / 8.4 at the first polls (the owner's piro_agentd at ~760% CPU, a cargo build),
  quiet at 15:11:14 (load 1.76). Started at once.
* Load over the start-to-ready window (the ram_gib window): 1.76 -> 1.22. Over the whole arm:
  1-min load median 4.5, max 14.7 (51 samples); SM clock 2880-2895 MHz while decoding.

## Result: the gate is met, at the band's edge
| | ck4g 31b8efb (this run) | ck4r2 01548e7 | ck4w 26956dc | ck4n f03c081 |
|---|---|---|---|---|
| ram_gib (MemAvailable delta at ready) | **12.86** | 13.06 | 12.86 | 11.71 |
| rss_ready_gib | 13.70 | 13.81 | 13.80 | 13.76 |
| cgroup anon_gib, end of arm | 13.29 | 13.08 | 13.05 | 13.07 |
| cgroup file_gib, end of arm | 0.44 | 0.96 | 0.90 | 0.55 |
| rss_gib (arm processes), end | 13.63 | 14.19 | 14.17 | 14.14 |

* 12.86 is within 12.26 ± 0.6 (|12.86 - 12.26| = 0.60). `ck4g-box-compare.txt`: "0 point(s)
  outside the band".
* MemAvailable (hostload.txt): 28.75 GiB before start (15:11:15), 16.11 GiB at ready (15:12:50);
  the sampler's first ready row reads 15.72 GiB two seconds later. measure.sh's own pair gives
  12.86.
* The server's memory (sampler, largest process = scheduler): RssAnon 11.47 GiB at ready, 1M p1
  and 1M p2, the same as ck4r2 (11.47-11.48) and ck4n (11.48). Cgroup anon at ready / 1M p1 /
  1M p2: 12.83 / 13.22 / 13.24 (ck4r2 12.47 / 13.34 / 13.00, ck4n 12.70 / 13.19 / 12.99).
* The same server memory gave ram_gib 11.71, 12.86, 13.06 and 12.86 across four runs: the
  host-wide MemAvailable delta moves by about ±0.7 GiB with what the rest of the host does.
* Correctness: coverage_faults 0, starved 0, R3 `captures=1 kv_grows=46 tracebacks=0`, R6 ok.
  Needles/recall against ck4r2-whole-1m (the round-2 whole-model 1M reference; this checkpoint ran
  no whole arm, so the script's default `ck4g-whole` reference did not exist):
  0 differences (`ck4g-needles-compare.txt`).

## Decode against ck4r2's whole arm (same host, 10 hours earlier, NOT back to back)
| | ck4g mirror-1m | ck4r2 whole-1m | ratio | ck4r2 mirror-1m |
|---|---|---|---|---|
| 8K p1 | 179.9 | 182.3 | 98.7% | 173.9 |
| 8K p2 | 172.6 | 166.6 | 103.6% | 164.1 |
| 1M p1 | 88.9 | 86.4 | 102.9% | 85.5 |
| 1M p2 | 99.6 | 93.3 | 106.8% | 92.7 |

1M prefill 528.6 / 528.8 s (ck4r2 mirror 527.6 / 527.9). Host load differs between the two
sessions, so these ratios carry the drift described in ../ck4r2-local/README.md.

## The +1.2 GiB of cgroup page cache: shared libraries and the CUDA JIT cache
Page cache that appeared between "before start" and "ready" (`pcache-diff-ready.txt`), 1.39 GiB
in 2551 files (cgroup file at ready: 1.68 GiB, 0.44 GiB at the first sample):

| Source | GiB |
|---|---|
| venv shared libraries (.so): libtriton.so 366 MiB, cutlass DSL `_cutlass_ir` 112 MiB, libcublasLt 51, libtorch_cpu 49, libtorch_cuda 40, libnvrtc 23, libnvJitLink 20, ... | 0.721 |
| CUDA driver JIT cache `~/.nv/ComputeCache` (kernels JIT-compiled from PTX, read back at start) | 0.584 |
| venv .py / .pyc / data | 0.049 |
| model dir (tokenizer.json 16 MiB; the weights are read O_DIRECT or were already cached) | 0.019 |
| triton cache, flashinfer / tvm-ffi cache, torch extensions, worktree .so | 0.018 |
| session-spill (`~/.cache/freetoken/session-spill`) | 0.000 |

* It shrinks on its own: 1.08 GiB still resident after 1M p1, 0.19 GiB at the end (cgroup file
  1.68 -> 1.33 -> 1.20 GiB), i.e. clean, reclaimable cache.
* The server's mapped file pages (RssFile) are 0.41-0.44 GiB; the rest is readahead around
  them, and the JIT cache files are read, not mapped.
* It does NOT belong in the RAM metric. It is program text and the driver's compiled-kernel
  cache, the same for any model and any expert-RAM configuration, shared with every process on
  the host, and reclaimable. MemAvailable (ram_gib) already counts reclaimable cache as available,
  so it hardly enters ram_gib. It does enter cgroup `memory.current`, which is why
  compare_ram_cgroup.py's memory.current rows must not be read as server memory. A cgroup is charged
  for a cached page only if it is the first to fault it. That fits ck4n's 0.57 GiB (the files were
  still cached from before) against 1.78 in ck4r2 and 1.68 here, but this was not checked for ck4n.
* 0.29 GiB of the 1.68 GiB cgroup file charge at ready lies outside the scanned directories
  (dirs.txt: model, venv, ~/.cache/{flashinfer,tvm-ffi,torch,torch_extensions,freetoken},
  ~/.triton, ~/.nv, the worktree's python/, /usr/lib/wsl, /usr/local/cuda, /dev/shm); most
  likely system Python and /usr/lib libraries, not attributed.
