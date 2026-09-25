# Round 2 (exp/reorg-round2): local ck4, the suite fault, the decode check (2026-09-25)

## What round 2 contains
Round 2 is exp/reorg-next with these on top:
* exp/wb-schedule, merged at f98c7c5 and again at 26956dc (01548e7);
* exp/prefix-reuse (59287aa);
* the six flashinfer extend commits (e954d20..6894623 from exp/ornith-exl3), cherry-picked as
  their own series (80dd4ac..d31a680);
* the GGUF MMQ pad fixes from exp/mmq-ypad: fe95df1, then 90b8dd7 (d56a058) and 01593a0
  (f528b2e), which complete it (see "The illegal memory access" below).

## Local ck4 (01548e7, RTX 5080 WSL, ratio 1.00, empty GPU)
The ck4 runs 8K/80K ×3, 1M whole and mirror with needles, the cgroup sampler, and one Claude Code
replay pass.

* **8K p1/p2** medians: 96.3 / 97.6%. **80K**: 97.0 / 97.1%. Every arm passes
  (`ck4r2-compare.txt`).
* **1M p1**:
  * whole / mirror: 86.4 / 85.5 tok/s (99%);
  * p2: 93.3 / 92.7;
  * needles: 0 differences (`../ck4r2-needles-compare.txt`);
  * R3/R6: pass.
* **ram_gib** is 13.06, outside the 12.26±0.6 band (ck4w had 12.86). The server's own memory
  did not grow: the gap is the host's MemAvailable baseline (see "RAM gate" below).
* **1M prefill** takes 523 s against 788 s (flashinfer extend).
* **Replay** matches exp/prefix-reuse's own results (tasks/prefix-reuse/README.md).

## The illegal memory access (ft-dev suite on d31a680): a latent GGUF MMQ bug, now fixed
* **Symptom**:
  * The d31a680 suite fails with 329 failed and 36 errors.
  * The first failure is `tests/kernels/test_gguf_mma.py::test_mma_matches_reference[7-12]`,
    which raises cudaErrorIllegalAddress at `.item()`. Everything after it is poisoned.
* **Reproduction**:
  * Deterministic on d31a680: run E1 in a fresh tree with fresh JIT gave the same counts.
  * tests/moe + tests/kernels (292 s) is enough to trigger it.
  * tests/kernels alone is clean (569 passed).
  * Hidden by CUDA_LAUNCH_BLOCKING=1: the full sequence passes, 3049.
  * 01548e7 is clean in the same tree and build: 3049 passed.
* **Where it fires**:
  * A pytest plugin syncs the device at every test teardown.
  * Every test through test_gguf_dispatch syncs clean.
  * The first poisoned sync is the teardown of test_gguf_mma[7-12], so the access is inside that
    test's own MMQ call.
* **Cause** (python/freetoken/kernel/csrc/gguf_mmq):
  * mmq_ext.cu pads y_q by `ggml_cuda_mmq_get_J_max(type, fallback, cc, ne11)` blocks.
  * `ggml_cuda_mmq_get_J_max` (mmq.cuh) rounds `min(ne11, 512)` down to a multiple of 8, which
    gives 0 blocks for ne11 < 8.
  * The MoE entry always passed ne11 = 1.
  * The launcher still selects J = 8, and the y-tile loads read J whole columns per K group with
    no bound. For 7 tokens of Q4_K that reads 144 B past a 4032 B buffer.
* **Why only sometimes**: whether that read faults depends only on what the caching allocator
  placed after y_q.
  * Preceding tests (tests/moe) and the timing of cross-stream frees change that placement.
  * memcheck on the file alone sees torch's 2 MB segments, not the tensor, and reports nothing.
  * The d31a680 vs 01548e7 split is placement luck.
  * The code dates from 242c37a / c81d074 / 55de75a, which are already on reorg-next.
  * Neither wb-schedule nor flashinfer is involved.
* **Ring-full write-back fallback is ruled out** by a direct test:
  * 102 mirror tests under memcheck with FREETOKEN_MIRROR_WB_STAGE_ROWS=1 give 0 errors.
  * The one failure is a test that asserts the default ring size.
* **Fix, in three commits** (exp/mmq-ypad; round 2 carries them as fe95df1, 90b8dd7, 01593a0):
  * fe95df1 pads y_q by the column tile J actually launched. That was not enough: memcheck
    with `PYTORCH_NO_CUDA_MEMORY_CACHING=1` on test_gguf_mma (d31a680 + fe95df1, ft-dev
    /root/wbfix/V1-memcheck.log) still reported 333 errors: "Invalid __global__ read of size 4" in
    `mul_mat_q<Q4_K, 8, true>+0x5070` from `ggml_mul_mat_a8_mma`, warp 6 of block 13,
    65..157 B past a 1728 B allocation (the 1-token y_q: 4 data + 8 pad blocks).
  * d56a058: each y-tile load copies `J*MMQ_TILE_Y_K` ints in steps of nthreads with no bound on
    the index, so it reads that count rounded UP to nthreads: 512 ints (14.2 blocks) for J = 8
    on the 256-thread config, not 8 blocks. The pad is now
    `ceil(round_up(J*36, nthreads) / 36)` blocks (15 for J = 8, 128 for J = 128, the same
    18 KiB upstream's `mmq_x_max` pad gives).
  * f528b2e: with y fixed, memcheck (/root/r2fix/V1-memcheck.log) reported 139 errors; all
    100 printed ones are in `mul_mat_q<Q4_K, 16, true>+0xee0` from `ggml_moe_a8_mma`, warp 0,
    1..49 B past a 144 B allocation. That is ids_dst of the 9-token top-4 MoE call (36 int32): the shared ids_dst
    fill copies J entries from each tile's first column with no bound (only the write-back
    bounds j), so the last expert's last tile reads up to J - 1 entries past it. Upstream's
    pool allocator rounds buffers up and hides this; ids_dst is now allocated with
    `ne_get_rows + J` entries.
  * The regression test `test_activation_padding_covers_the_launched_tile` checks both sizing
    rules against the kernel's real read extent (the hook returns J, nthreads, the y pad and
    the ids_dst pad) for 1..299 and large column counts.
* **Verification** (ft-dev, /root/r2fix2/status):
  * V1, d31a680 + the three commits, memcheck with `PYTORCH_NO_CUDA_MEMORY_CACHING=1` over
    test_gguf_mma, test_gguf_dispatch, test_gguf_quant_types and
    tests/moe/test_cpu_moe_mixed_gguf.py: 142 passed, `ERROR SUMMARY: 0 errors`.
  * V2.1 / V2.2, d31a680 + the fix (the placement that faulted), full 7-directory sequence
    with the per-test sync plugin, fresh JIT each: 3057 passed, 25 skipped, clean sync, 0
    POISONED, 0 "illegal memory access", both times.
  * Round 2 suite (8cc98cf = 01548e7 + fe95df1, plus d56a058 and f528b2e; same python/ and
    tests/ as 01593a0), the full 7-directory suite with a fresh JIT: rc=0, 3057 passed,
    25 skipped, 0 "illegal memory access" (01548e7 had 3049; the fix adds 8 gguf_mma cases).
  * Copies: `mmq-fix/` (the status files, run script, the three V1 memcheck logs, md5-checked
    against the box).
* Found by the mirror worker (a2c850f7), from the per-test sync plus a source read.
* Evidence on ft-dev:
  * /root/fault-hunt4/{F-lb,G-kernels,H-moe-kernels}.txt;
  * /root/wbfix/status, P-*.txt, M-*.

## Decode: no regression in round 2 (the ck4r2-vs-ck4w 1M gap is host drift)
* **Arena and VRAM are equal** (`decode-ab/`, 512-token decode, two fresh servers per arm):
  * Startup arena is 2032 / 2040 slots.
  * Decode has 2104 slots at 8K and 1952 at 300K in every arm.
  * Decode hit rates, from `--moe-collect-stats` missing/active:

    | Arm | Hit rate |
    |---|---|
    | base e2f136f | 95.72% |
    | r2 01548e7 | 95.86% |
    | r2 with FREETOKEN_EXTEND_BACKEND=triton | 95.82% |
    | r2 with FREETOKEN_CHUNK_SNAPSHOTS=0 | 95.78% |

* **Decode tok/s**, mean of 2:

  | Arm | 300K p1 | 300K p2 | 8K p1 | 8K p2 |
  |---|---|---|---|---|
  | base | 154.9 | 157.7 | 186.9 | 180.9 |
  | r2 | 154.4 | 158.9 | 187.6 | 174.6 |
  | r2-triton | 156.2 | 158.2 | 184.1 | 180.6 |
  | r2-nosnap | 154.4 | 156.1 | 184.4 | 177.4 |

  * Each sample spreads by about ±4 tok/s.
  * 300K prefill runs at 4650 tok/s with flashinfer against 3426 with triton.
* **nsys 1M pair, back to back** (`nsys-1m-pair/`, whole arm, trace from token 33 over 200
  tokens):
  * Graph launch period median: base 10.26 ms, r2 10.38 ms (+1.1%).
  * Arena: 1456 -> 1528 slots in both.
  * In-graph ms/token:

    | Component | base | r2 |
    |---|---|---|
    | attention (stage 1 + stage 2) | 3.79 | 3.87 |
    | expert copies (fast_index_copy_multi + ensure_experts) | 1.69 | 1.82 |
    | MoE/linear GEMMs | 2.84 | 2.90 |
    | other | 1.03 | 1.05 |
    | total | 9.36 | 9.65 |

  * Nothing overlaps attention.
  * Over steps 1-128 the two are equal: 10.31 vs 10.26 ms.
  * From step 129 on, r2 fetches more experts: 16.0 vs 11.9 copies per step, 11.5 vs 10.5 ms.
* **Host load during the pair** (`hostload.txt`):
  * load average 4-10 in both arms;
  * the owner's rustc builds were running during both;
  * SM clock 2872-2880 MHz in both.
* **Base today vs earlier**: base itself now runs at 10.26 ms, against 9.72 ms for the same
  whole arm on 3d6f249 earlier (`../nsys1m-local/README.md`). The 15% gap between ck4r2
  (86.4) and ck4w (102.0) at 1M therefore does not reproduce back to back. It is host drift,
  not round 2.

## RAM gate: ram_gib 13.06 is the host baseline, not the server
ram_gib is `MemAvailable before start - MemAvailable at ready`, host-wide. The server's own
memory is the same in ck4n, ck4w and ck4r2 to within 0.05 GiB; only the baseline moved.

End-of-arm record (measure.sh, mirror-1m arm, after 1M p2):

| | ck4n f03c081 | ck4w 26956dc | ck4r2 01548e7 |
|---|---|---|---|
| ram_gib (MemAvailable delta at ready) | 11.71 | 12.86 | 13.06 |
| rss_ready_gib (arm processes at ready) | 13.76 | 13.80 | 13.81 |
| rss_gib (arm processes, end) | 14.14 | 14.17 | 14.19 |
| cgroup anon_gib (end) | 13.07 | 13.05 | 13.08 |
| cgroup file_gib (end) | 0.55 | 0.90 | 0.96 |

cgroup sampler (`cg.tsv` and `../ck4n-local/cgroup-samples.tsv`), same mirror-1m arm, GiB:

| Point | Arm | cg current | cg anon | cg file | scheduler VmRSS | its RssAnon | host MemAvailable |
|---|---|---|---|---|---|---|---|
| ready | ck4n | 13.31 | 12.70 | 0.57 | 12.03 | 11.48 | 17.23 |
| ready | ck4r2 | 14.31 | 12.47 | 1.78 | 12.04 | 11.47 | 17.42 |
| 1M p1 | ck4n | 13.81 | 13.19 | 0.58 | 12.04 | 11.48 | 14.83 |
| 1M p1 | ck4r2 | 15.23 | 13.34 | 1.83 | 12.05 | 11.48 | 16.45 |
| 1M p2 | ck4n | 13.62 | 12.99 | 0.58 | 12.04 | 11.48 | 16.36 |
| 1M p2 | ck4r2 | 14.89 | 13.00 | 1.83 | 12.05 | 11.48 | 17.04 |

* The scheduler process's anon RSS is 11.47-11.48 GiB in both rounds at every point, and the
  cgroup's anon differs by -0.23 / +0.15 / +0.01 GiB. Round 2's prefix-reuse snapshots,
  session spill and flashinfer did not add host memory the server holds.
* At ready the host had MORE memory available in ck4r2 (17.42 vs 17.23 GiB). The ram_gib gap
  is in the "before" reading: ram_gib + MemAvailable at ready gives about 30.48 GiB before ck4r2
  against 28.94 before ck4n, i.e. other processes (the owner's builds and agents) held
  about 1.5 GiB less when ck4r2 started and more by the time it was ready.
* The only server-side difference is +1.2 GiB of cgroup file cache in ck4r2 (1.78-1.83 vs
  0.57-0.58). It is not mapped by the server (its RssFile is 0.48 vs 0.47), it is
  reclaimable page cache that MemAvailable counts as available, and it had dropped to 0.96 by
  the end-of-arm record (ck4w 0.90). Which files it caches was not attributed.
* So ram_gib misses the band, but the gate is explained with evidence: the server itself
  holds the same RAM as ck4w/ck4n.
* **Re-run on exp/reorg 31b8efb, quiet host (`../ck4g-local/README.md`): ram_gib 12.86, inside
  12.26 ± 0.6 (at the edge), scheduler RssAnon 11.47 GiB as here; needles 0 differences, 0
  faults, 0 starved, one capture.** The +1.2 GiB of cgroup file cache is shared libraries
  (0.72 GiB, libtriton.so alone 0.36) and the CUDA driver's JIT cache ~/.nv/ComputeCache
  (0.58 GiB). It is reclaimable and does not belong in the metric.

## Script fixes
* `compare_records.py`: an arm that did not run in this checkpoint (ARMS) is skipped, not a
  traceback.
* `checkpoint1.sh`: the needles reference defaults to `$CK-whole-1m` when only that arm ran
  with needles.
