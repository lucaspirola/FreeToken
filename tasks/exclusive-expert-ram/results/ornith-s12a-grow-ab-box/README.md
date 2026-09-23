# Ornith S12a grow A/B on native Linux (rented RTX 5080, 2026-09-23)

Diagnostics only, not R-S12a records: the owner's numbers come from the owner's machine.

Box: RTX 5080 16 GB, driver 595.58.03, native Linux in Docker (no WDDM, no systemd),
256 GB RAM / 171 GiB cgroup, RLIMIT_MEMLOCK 64 KiB. Code: exp/reorg fc04dca + box-runner
(measure.sh FT_LAUNCHER=nohup, probe with time.monotonic()). venv /root/venv (torch 2.11+cu130),
`freetoken.__file__` = /root/FreeToken-reorg/python/freetoken/__init__.py. Model: Ornith NVFP4
ornith-ai@94e431d (22 files, sizes identical to the owner's copy). Runner:
`FT_RATIO=0.85 ARMS="grow64-a grow128 grow64-b" tasks/exclusive-expert-ram/s12a-grow-ab-box.sh`.
`*-stamped.txt` = server log with time.monotonic() seconds since launch (logstamp.py).

## 1. Startup: no stall, so no regression in fc04dca's startup path

Seconds since launch (monotonic, logstamp.py). "banks done" = `--moe-cache-auto resolved`, the
first line after the expert load + cudaHostRegister settle; "ready" = `API server is ready`.

| run | expert path | weights | banks start | banks done | KV alloc | graphs | warmup | ready | first completion |
|---|---|---|---|---|---|---|---|---|---|
| startup-cold (1.00, cold JIT) | parallel | 21.4 | 21.8 | 69.5 | 71.9 | OOM (LinearStatePool 780 MiB) | | | |
| startup-warm (1.00) | parallel | 21.6 | 22.0 | 48.4 | 49.8 | OOM (same) | | | |
| startup-r099 | parallel | 21.2 | 21.6 | 48.0 | 49.4 | 79.1 (cold triton) | OOM 256 MiB | | |
| serial-r099 (`--expert-load serial`) | serial | 21.4 | 21.6 | 49.8 | 51.0 | 52.2 | OOM 256 MiB | | |
| startup-r097 | parallel | 21.6 | 22.0 | 50.2 | 51.6 | 54.0 | 71.3 | 72.1 | 159 s (cold triton, ~90 s) then OOM on 8K |
| grow64-a (0.85) | parallel | 21.1 | 21.5 | 48.3 | 49.7 | 52.1 | 53.5 | 54.3 | 52 s |
| grow128 (0.85) | parallel | 21.4 | 21.8 | 48.0 | 49.4 | 51.8 | 53.4 | 54.2 | 51 s |
| grow64-b (0.85) | parallel | 21.4 | 21.8 | 48.0 | 49.6 | 51.0 | 52.4 | 53.2 | 52 s |

(~10 s before "Parsed arguments" is `uv run` + imports.) The serial path, the one the local host
took, finishes its 3 shards and the pin settle in 28 s: `Loading Qwen3.5 NVFP4 experts 3/3`
and `ninja: no work to do` / `--moe-cache-auto resolved` land in the same 0.2 s stamp.
Locally the good start of 18:00 (ornith-s12a-attn-ab) did the same step in 33 s; the three
stalled local arms (ornith-s12a-grow-ab) printed 3/3 after 42-135 s and then nothing for
13 min, with 22.3 GiB memory peak and 1.7 GiB swap peak in the unit. The local tvm-ffi JIT
cache already held this tree's kernels (built 22:39-22:55) and no JIT dir was created or
touched during the stalls, so a cold nvcc build is not the local stall either.
Pinning on the box: no pin/mlock/"settled pageable" line in any journal; at readiness the arm
held 17.95 GiB of /dev/zero RSS (the pinned banks), smaps `Locked:` 0 (cudaHostRegister does
not count as VmLck; no OS mlock was attempted), ram_gib 21.09.

Box-only setup issue: the reorg clone had no install-time extension
(`freetoken.kernel._pinned_tensor is not installed`, startup-fail-noext); fixed by
`python setup.py build_ext --inplace` in /root/FreeToken-reorg (ignored *.so, sources identical
to /root/FreeToken's).

## 2. Ratio 1.00 does not run on native Linux

The same plan that WSL starts with `Free memory after initialization: 0.00 GiB` fails here:

| ratio | free after init | failure |
|---|---|---|
| 1.00 | - | OOM at LinearStatePool (780 MiB, 720 MiB free), before graphs |
| 0.99 | 0.09 GiB | OOM in prefill warmup (256 MiB) |
| 0.97 | 0.40 GiB | ready, OOM on first 8K prefill (conv, 126 MiB, 130 MiB free) |
| 0.94 | 0.85 GiB | ready, OOM on first 8K prefill (GDN chunk, 126 MiB, 94 MiB free) |
| 0.91 | 1.30 GiB | 8K/32K OK; 80K OOM in the chunk right after `KV grew 65536 -> 131072` (pre-commit 1.27 GiB free, 0.66 commit, 0.91 required; then 32 MiB with 36 MiB free) |
| 0.85 | ~2.2 GiB | all sizes OK (grow pre-commit 2.18 GiB free, no arena shrink) |

So an Ornith 8192-token prefill chunk needs > 0.61 GiB of transient VRAM beyond what the
allocator holds, while the growable-KV commit only guarantees
`VMM_COMMIT_CUSHION_BYTES + 128 MiB` (0.375 GiB) after the commit when it has to shrink the
arena. Locally (ratio 1.00) every 80K grow takes that shrink path
("arena shrink 6078 -> 5448, 1.04 GiB uncommitted" then a 0.66 GiB commit), i.e. the chunk
after the grow runs with less free VRAM than it needs by WSL's own ledger. Native Linux
turns that into an OOM; WSL/WDDM can page instead, which fits the slow, half-power post-grow
chunk. The warmup only prefills lengths [80, 128], so the startup plan never sees an 8192 chunk.

## 3. A/B at ratio 0.85 (monotonic TTFT = wall TTFT to the ms on this host)

| arm | 8K p1 | 8K p2 | 32K p1 | 32K p2 | 80K p1 | 80K p2 | 128K p1 | 128K p2 |
|---|---|---|---|---|---|---|---|---|
| grow64-a | 0.18 s* | 0.79 s / 10,203 | 3.88 / 8,264 | 3.90 / 8,206 | 15.43 / 5,185 | **15.41 / 5,193** | 35.84 / 3,572 | 35.43 / 3,613 |
| grow128 | 0.18 s* | 0.74 / 10,856 | 3.87 / 8,265 | 3.84 / 8,336 | 15.36 / 5,211 | **15.36 / 5,211** | 35.36 / 3,621 | 35.34 / 3,623 |
| grow64-b | 0.19 s* | 0.74 / 10,872 | 3.87 / 8,265 | 3.84 / 8,346 | 15.36 / 5,208 | **15.36 / 5,211** | 35.36 / 3,620 | 35.32 / 3,625 |

TTFT s / prefill tok/s. *8K pass 1 hits the warm-up's prefix cache. Six 80K requests,
15.36-15.43 s: no bimodality, and no difference between growing mid-prefill (grow64) and
never growing (grow128). The 80K prefill spans 13.4 s from first to last chunk line in both
arms; the grow (65536 -> 131072, no arena shrink at this ratio) moves ~4 s of log spacing
around the post-grow chunk but adds nothing to the total.

dmon, 80K pass 2 (all three arms): SM 99-100 % through the whole prefill, 300-322 W early,
drifting to 260-285 W over the later (attention-heavy) chunks identically in grow128, which
never grows; PCIe RX bursts ~30 GB/s. No half-power interval after the grow.

## Reading

* H3 (the grow itself costs time): rejected on native Linux: grow64 = grow128 within 0.05 s.
* H1 (WDDM paging at 100 % VRAM): leading. Native Linux shows no bimodality, and it cannot run
  the local memory level at all -- every step of the local profile's headroom that WSL tolerates
  is an OOM here, the post-grow chunk included. Caveat: the A/B had to run at 0.85, so the
  arena-shrink grow path that local takes was not exercised without an OOM.
* F3 (monotonic clock) is on this branch in probe_decode.py; it changes nothing on native Linux.
