# Final numbers: Ornith-1.5-35B-A3B EXL3 5.0 bpw on the local RTX 5080 (2026-09-27)

Branch `exp/final-numbers` (from exp/reorg 84ae232). The server code is the same in every arm:
`python/` at cf26ac4 (`results/status.txt`, `code=` column). The harness is 1dfad58 + cf26ac4 +
this commit. Model: `~/ai/models/Ornith-1.5-35B-A3B-exl3-5.0bpw-hq` (codebook mul1).

Every arm runs `scripts/serve-default.sh` on :1920 as a systemd --user unit, driven by `arm.sh`
(→ `benchmarks/.../measure.sh`), one arm at a time under the GPU host lock.
* Settings: q8_0 KV unless stated, ratio 1.00, pin budget 20.
* Saver = FT_ROWS=-1 (mirror, auto rows). Whole = FT_ROWS=0.
* The probe does 2 passes (pass 1 at 8K is a prefix hit on the warm-up; pass 2 uses fresh
  prompts), with 512 generated tokens, timed on the monotonic clock.
* Natural text = 5 tasks of `natural_gen.py`, 3500 max tokens.
* Per-arm load and SM clock are in `results/status.txt`, with 5-s samples in `results/load5s.txt`.

All tables below come from `table.py results` → `results/tables.md`. Per-arm raw files are
`results/<arm>-{probe.jsonl,record.json,journal.txt,stats.json,acceptance-R3.txt}`.

Every arm had coverage faults 0, starved writebacks 0, captures=1 and tracebacks=0
(`results/status.txt`, `*-acceptance-R3.txt`).

## Timing drift (read first)

Between ~10:00 and ~12:00 this host ran the same work ~9-10% slower. The output was identical.
* Before (06:49-09:58, every first-run arm) and after 12:06 the speeds agree.
* Evidence:

  | arm | start | 256K prefill p2 | 256K decode p2 | vs first run |
  |---|---|---:|---:|---|
  | kv-q8q8-256k | 07:35 | 3569 | 123.1 | reference |
  | kv-q8q8-256k-rep | 10:02 | 3218 | 110.4 | -10% / -10%, same out_sha1 |
  | p1-*-3/4 (all four) | 10:22-10:54 | 3208-3256 | 110.3-115.2 | -9% / -9% |
  | kv-q8q8-256k-r2 | 11:29 | 3271 | 112.1 | -8% / -9%, same out_sha1 |
  | kv-q6q5-256k-r2 | 11:44 | 1340 | 103.0 | -7% / -7% |
  | kv-q8q8-384k-r2 | 12:06 | 2521 (384K) | 92.4 | +2% / +2% (normal) |
  | kv-q8q6-384k-r2, kv-q4q4-384k-r2 | 12:14-13:04 | 546, 627 | 79.7, 95.7 | +2% (normal) |

  Natural text at 11:12-11:28 was unchanged: whole 167.9/168.3, saver 153.7/153.6 tok/s.
* **What it is not:**
  * **Host load.** p1-saver-3/4 ran at 1-min load 1.2-1.3 and were slow; first-run arms at load
    3.7-6.5 were fast.
  * **The logged SM clock.** The mean under load is 2833-2919 MHz in every 10-min bucket, before and
    after (`results/load5s.txt`).
  * **Time-slicing by another GPU context.** A 40 s sustained fp16 GEMM at 11:58 ran 118.5 TF/s
    median and 120.7 TF/s over the wall clock, with max call 9.49 ms vs median 9.28 ms, so no gaps
    (`profile/sustained_gemm.jsonl`, `sustained_gemm.py`). The only foreign process is the Windows
    Codex app (`ChatGPT.exe`, C+G, 17 MiB), present from ~10:36 on and still present during the
    normal-speed arms after 12:06.
  * **PCIe.** Pinned DMA at 10:56 measured H2D 56.6 GB/s and D2H 41.5 GB/s (`profile/pcie_bw.log`).
  * **Thermal.** 42-43 °C, and HW/SW thermal slowdown counters stay at 0
    (`profile/gpu-counters.txt`).
* **What it is: the GPU executing the same kernels more slowly.**
  * The identical whole-model 8K decode graph was traced at 12:00 with round 5's script
    (`nsys-decode.sh`). It took **4962 µs of GPU time per step, against 4742 µs** on 2026-09-26 on
    this host (+4.6%).
  * Every kernel outside the graph is slower by about the same share: index_elementwise 7.3 vs
    7.0 µs, reduce 4.7 vs 4.5 µs (`profile/decode-nsys/steps-vs-round5.txt`; traces in
    `_orch/final/profile/decode-nsys/`, `_orch/r5/nsys/`).
  * A uniform per-kernel slowdown with an unchanged reported clock points below what nvidia-smi
    reports from WSL (effective vs. requested clock). That is a hypothesis; the physical cause is
    **not identified**.
* A 5-s sampler of clocks, power and event reasons (`profile/gpu5s.csv`, 12:03-13:13) only
  covers normal-speed arms. It shows long prefill at 340 W against the 360 W cap, reason 0x4
  (SW power cap), SM ~2860 MHz.
* **Consequence for the tables:** saver/whole ratios are taken **within a period** (arms 1-2
  vs 1-2, arms 3-4 vs 3-4). Lanes are compared on their first runs, which all fall in the normal
  period 07:35-09:58. The `-r2`/`-rep` repeats are listed as repeats.

## Step 1: EXL3 saver vs whole, q8_0 KV, 262144 ceiling

Pass 2, mean of two arms per period (`results/tables.md`, "Part 1"):

| prompt | saver prefill (arms 1-2) | whole prefill | saver / whole | saver decode | whole decode | saver / whole | TTFT saver / whole |
|---|---:|---:|---:|---:|---:|---:|---|
| 8K | 8154 | 8462 | 96.4% | 194.7 | 204.9 | 95.0% | 0.98 / 0.95 s |
| 32K | 7733 | 7902 | 97.9% | 184.2 | 194.0 | 94.9% | 4.14 / 4.05 s |
| 80K | 6136 | 6330 | 96.9% | 159.6 | 175.1 | 91.1% | 13.04 / 12.64 s |
| 128K | 5166 | 5228 | 98.8% | 153.3 | 158.1 | 97.0% | 24.78 / 24.49 s |
| 256K | 3552 | 3586 | 99.1% | 123.2 | 126.4 | 97.4% | 72.07 / 71.39 s |

In the slow period (arms 3-4) the ratios are 97.4-99.6% for prefill and 94.6-96.5% for decode
(same file).

Other results:
* **Pass 1** (the 8K prefix hit): TTFT 0.3 s saver vs 0.4 s whole. The per-arm tables show the
  pass-1 figures in brackets.
* **Natural text** (5 tasks each, md5 5/5 identical to `p1nat-whole-1` in all 8 arms):

  | residency | tok/s per arm | mean |
  |---|---|---:|
  | saver | 154.6, 154.5, 153.7, 153.6 | 154.1 |
  | whole | 167.4, 168.2, 167.9, 168.3 | 167.95 |

  Saver / whole = **91.8%**: 92.1% in the first pair of periods, 91.4% in the second.
* **RAM:**
  * saver: ram_gib 16.62-16.76, RSS 17.4-17.6; the pool is 7290 rows = 13.45 GiB;
  * whole: ram_gib 19.87-22.24, RSS 22.6-23.0.
* **VRAM:** gpu_mib 13978-13994. Expert slots: saver 4864 at start, 3768 once KV has grown to
  256K (2.73 GiB KV); whole 4928 → 3832.

### R-S12a-EXL3 (saver prefill ≥ 5000 tok/s at 32K and 80K, chunk 8192): **PASS**
* Measured: **7733 tok/s at 32K, 6136 tok/s at 80K** (arms 1-2). The slow period gave 6970 / 5658.

**Roofline** (`roofline_prefill.py profile` → `profile/roofline-prefill.txt`):
* Peaks measured on this GPU: fp16 tensor core 121.8 TF/s, DRAM 830 GB/s (`profile/roofline.log`).
* The bound per kernel group is the larger of FLOP/peak and bytes/peak.
* The traced requests are from `nsys-prefill.sh`, taken in the slow period, one request per trace
  after an untraced pass.

| saver | traced TTFT | roofline | measured / bound |
|---|---|---:|---:|
| 32K | 4.71 s = 6793 tok/s | 16355 tok/s | 2.43× |
| 80K | 14.53 s = 5509 tok/s | 10704 tok/s | 1.94× |

Whole has the same shape: 2.33× and 1.88×.

**Residual gap, compute stream, saver 80K** (the 32K split is similar):

| bucket | measured s | bound s | over bound | share of the 7.0 s gap |
|---|---:|---:|---:|---:|
| routed experts: exl3_gemm 2.49, reconstruct_experts 1.04, had_rows 0.69, splitk_combine 0.25 | 4.57 | 1.32 | 3.45× | 46% |
| copies/casts (torch direct_copy, elementwise) | 1.19 | – | – | 17% |
| dense projections (cuBLAS/CUTLASS fp16) | 2.54 | 1.85 | 1.38× | 10% |
| idle / launch gaps | 0.68 | 0 | – | 10% |
| GDN + conv + norms | 0.61 | – | – | 9% |
| attention (flashinfer SinglePrefill 4.78) | 4.86 | 4.31 | 1.13× | 8% |

The saver's layer assembly (fast_index_copy_multi, 2.91 s) runs on side stream 25, overlapped
with compute. Saver and whole compute streams differ by 0.3 s at 80K.

**Why the routed experts sit at 3.45× their bound** (ncu, `ncu.sh`/`ncu_moe.py`):
* Details are in `profile/ncu-moe-details.txt` (report in `_orch/final/profile/ncu-moe.ncu-rep`).
* ncu locks base clocks, SM 2.48-2.59 GHz.

| kernel | duration | limiter (ncu) | occupancy | notes |
|---|---:|---|---:|---|
| `_exl3_gemm_kernel` (2297×8 grid) | 470 µs | Tensor pipe 82.2% ("over-utilized"), 10.4 of 22.9 cycles/issue stalled on the math pipe; DRAM 54% | 32% (shared-memory limited) | SASS: HMMA.16816.F32 (fp16 in, fp32 accumulate) |
| `_reconstruct_experts_kernel` | 238 µs | **FP64 pipe 67.3%**; DRAM 64% | 81% | see below |
| `_had_rows_kernel` | 1.64 ms | Tensor pipe 78.7%, 3.1 cycles/issue waiting on the pipe | **16.7%** (255 registers/thread) | DRAM 37% |

* **The FP64 pipe on a GeForce part comes from the mul1 trellis decode** (`_exl3_decode`,
  `python/freetoken/kernel/triton/exl3.py`, CB == 2 branch). It builds the 16-bit code in uint64
  (`a << 32 | b`, `* 0x83DCD12D`, byte sum). `(s + 1024).to(tl.float32)` then converts a uint64,
  which compiles to `I2F.U64`, an instruction the FP64 unit executes.
* The SASS confirms it: `cuobjdump -sass` of the cached cubins shows 16× `I2F.U64` in the
  `_reconstruct_experts_kernel` variants and 32× in two of the four cached `_exl3_gemm_kernel`
  variants (the ones with 288 HMMA, i.e. inline decode). One cached reconstruct variant, compiled
  with other constants, has none.
* Reported, not changed ("no new optimisation work"). A 32-bit decode path would move this off
  the FP64 pipe. It has to be proven bit-exact, since the reconstruct output feeds the GEMM.

## Step 2: KV lanes, saver, at 256K and 384K

Lanes: q8q8 = q8_0/q8_0 (reference), q8q6 = K q8_0/V q6_0, q6q5 = K q6_0/V q5_0, q4q4 = q4_0
(control only). 384K = `--rope-yarn-factor 2`, ceiling 393216. First runs, normal period
(`results/tables.md`, "Part 2"). The output columns record identity to q8q8 only; quality is
not judged here.

| lane | ceiling | big-prompt prefill tok/s | TTFT s | big decode | 8K decode | natural | output vs q8q8 (probe; natural) | KV GiB | expert slots start / at max KV | pool rows (GiB) | ram_gib |
|---|---|---:|---:|---:|---:|---:|---|---:|---|---:|---:|
| q8q8 | 256K | **3569** | 71.7 | **123.1** | 194.4 | 154.1 | reference | 2.73 | 4864 / 3768 | 7290 (13.45) | 17.68 |
| q8q6 | 256K | 855 | 299.5 | 107.3 | 197.3 | 150.8 | all 4 differ; 0/5 | 2.42 | 4912 / 3936 | 7118 (13.14) | 16.30 |
| q6q5 | 256K | 1441 | 177.7 | 111.3 | 197.3 | 152.1 | all 4 differ; 0/5 | 1.95 | 4976 / 4192 | 6860 (12.66) | 15.80 |
| q4q4 | 256K | 971 | 263.6 | 114.3 | 201.2 | 152.5 | all 4 differ; 0/5 | 1.48 | 5040 / 4448 | 6602 (12.18) | 15.37 |
| q8q8 | 384K | **2466** | 155.7 | 90.3 | 169.1 | 153.8 | reference | 4.06 | 4832 / 3024 | 8042 (14.84) | 18.16 |
| q8q6 | 384K | 533 | 719.9 | 77.9 | 177.0 | 152.9 | all 4 differ; 0/5 | 3.59 | 4888 / 3280 | 7784 (14.36) | 17.53 |
| q6q5 | 384K | 919 | 417.7 | 81.9 | 176.8 | 152.9 | all 4 differ; 0/5 | 2.89 | 4944 / 3656 | 7397 (13.65) | 17.09 |
| q4q4 | 384K | 612 | 627.9 | 94.4 | 179.0 | 151.4 | all 4 differ; 0/5 | 2.19 | 5008 / 4040 | 7010 (12.94) | 16.19 |

gpu_mib is 13966-13990 in every lane.

Repeats reproduce each first run's out_sha1 exactly: q8q8 256K ×3, q6q5 256K, and q8q8, q8q6,
q4q4 at 384K. The runs are deterministic.

**Why every quantized-V lane has slow prefill:**
* `kernel/extend_flashinfer.py eligible()` accepts only `k_format`/`v_format` in (0, 1):
  unquantized or q8_0/fp8. q6_0, q5_0 and q4_0 fall back to the Triton extend kernel.
* Journals: q8q8 logs "extend attention: flashinfer 0.6.17"; the q8q6 journal has only the Triton
  extend launch line (`results/kv-*-journal.txt`).
* Control `kv-q8q8-256k-triton` (`control.sh`, q8q8 with `FREETOKEN_EXTEND_BACKEND=triton`, run
  in the slow period):
  * 1859 tok/s at 256K, vs 3218-3271 for q8q8 on flashinfer in the same period. Triton costs ~43%
    of prefill before any sub-8-bit dequant.
  * Its output differs from flashinfer's (a24504c26b vs a983be5886), so the extend kernel alone
    changes the numerics.
* The quantized lanes are slower still: 855/1441/971 tok/s.

**Why decode is slower on q8q6/q6q5 at 256K** despite 168-424 more expert slots: this is the
Triton decode-attention dequant cost. Not profiled further. q4q4 decodes fastest at 384K
(94.4 vs 90.3) but is a control.

**Verdict: q8_0/q8_0 stays.**
* It is the fastest lane for prefill at both ceilings (2.5-4.6×) and for decode at 256K.
* It is the only lane on the flashinfer extend path.
* The lower lanes free 0.31-0.78 GiB of KV at 256K, which does not buy speed.

## Step 3: serve-default.sh → Ornith EXL3, RAM saver, q8_0 KV (this branch only, not main)

`scripts/serve-default.sh`:
* Model default: `~/ai/models/Ornith-1.5-35B-A3B-exl3-5.0bpw-hq`, `--text-model-only`, served as
  `ornith` with `--reasoning-parser qwen3 --tool-call-parser qwen3_coder`. The Nemotron name and
  aliases are removed.
* Context: 262144 (the model's native maximum); 384K is documented as an `FREETOKEN_EXTRA_ARGS`
  option.
* KV: `--kv-cache-dtype q8_0`.
* Saver on by default: `FREETOKEN_MIRROR_EXPERT_RAM=1`, `FREETOKEN_MIRROR_HOST_ROWS=-1`.
* `FREETOKEN_PIN_BUDGET_GB` default 20 (≥ 19.28 GiB of Ornith expert banks, for the whole-model
  switch).
* The comments carry the numbers above.

**Memory ratio.** `scripts/tune-memory-ratio.sh` has been retired into `verify-memory-ratio.sh`
(owner decision 2026-09-24): ratio 1.00, no bisection. It ran via `verify.sh` with
`VERIFY_LAUNCHER=nohup`, port 1920, in unit `ft-final-verify`.
* Result: **PASS** at ratio 1.00 (`results/verify.log`, `results/verify-memory-ratio.tsv`,
  server log `results/verify-server.log`).
* Ready in 81 s, 1 capture, 8K/80K/256K served, decode 147.9/145.1/109.2 tok/s (its own probe
  settings).
* 3.35 GiB free after capture; prefill transient predicted 0.88 GiB vs measured 1.10 GiB;
  0 prediction WARNs.
* Nothing was written to serve.env. The owner's serve.env already holds ratio 1.00 and the saver.

## Files
* **Harness:**
  * `arm.sh`, `chain.sh` (part 1 + first-run lanes), `chain2.sh` (period repeats), `control.sh`;
  * `table.py` (all tables), `natural_gen.py`;
  * `profile.sh`/`nsys-prefill.sh`/`nsys_kernels.py`/`roofline_prefill.py` (prefill roofline);
  * `ncu.sh`/`ncu_moe.py`;
  * `nsys-decode.sh`/`nsys_steps.py` (decode step trace, round 5's script, unit name now unique);
  * `sustained.sh`/`sustained_gemm.py`, `pcie_bw.py`, `verify.sh`.
* **Probe:** `scripts/probe_decode.py` now records `out_sha1` and takes a `PROBE_TAG` prefix (1dfad58, cf26ac4).
* **Aborted runs:**
  * `aborted-20260927-0649/`: the first arm, where probe_decode.py had lost its exec bit (fixed, rerun).
  * `profile/decode-nsys.failed-reused-unit/`: nsys-decode.sh reused round 5's unit name, so the
    "API server is ready" grep matched an old journal line. Fixed with a unique name.
* **Large binaries**, not committed: `_orch/final/profile/` (nsys .nsys-rep/.sqlite, ncu .ncu-rep).

## Open
* The physical cause of the 10:00-12:00 slowdown is not identified; the evidence above rules out
  load, reported clock, time-slicing, PCIe and thermal.
* The routed-expert prefill kernels hold 46% of the gap to the roofline. The known causes are
  listed above: I2F.U64 on the FP64 pipe, tensor-pipe saturation in exl3_gemm, and had_rows
  register pressure. None were changed.
