# Round 2 (exp/reorg-round2): local ck4, the suite fault, the decode check (2026-09-25)

## What round 2 contains
Round 2 is exp/reorg-next with these on top:
* exp/wb-schedule, merged at f98c7c5 and again at 26956dc (01548e7);
* exp/prefix-reuse (59287aa);
* the six flashinfer extend commits (e954d20..6894623 from exp/ornith-exl3), cherry-picked as
  their own series (80dd4ac..d31a680);
* the GGUF MMQ y_q pad fix, fe95df1 from exp/mmq-ypad (see "The illegal memory access" below).

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
* **ram_gib** is 13.06, outside the 12.26±0.6 band (ck4w had 12.86).
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
* **Fix**: fe95df1 (exp/mmq-ypad) pads y_q by the column tile actually launched (at most 128
  blocks) and adds a regression test.
* **Verification**: see the suite line in the status below.
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

## Script fixes
* `compare_records.py`: an arm that did not run in this checkpoint (ARMS) is skipped, not a
  traceback.
* `checkpoint1.sh`: the needles reference defaults to `$CK-whole-1m` when only that arm ran
  with needles.
