# popt-exl3: EXL3 prefill kernels, EXL3 dense layers, GDN and norm copies (bit-exact)

Branch `exp/popt-exl3` (worktree `FreeToken-wt/popt-exl3`), base `757caee` (exp/reorg after kfix).
Assignment (2026-09-27): find what now bounds the 8192-token expert path and attack it, then the
EXL3 dense layers and GDN + norms. Also D3 (shared-expert fusion) if it helps bit-exactly. Owner
shapes: omp file reads (5-80K tool results appended to a deep context) and compactions (20-40K fresh
prefill).

**Verdict.** Faster prefill on every owner shape, with identical outputs (every probe/extend
out_sha1 and every natural-text md5 equal to 757caee). Decode is within noise. The same kernels are
bit-exact and faster on the Ornith 4.0bpw checkpoint. Gate and suite results: see
[Gate and suite](#gate-and-suite).

| commit | change |
|---|---|
| bc23b40 | MoE prefill: bitstream-order reconstruct, group-ranged decoded GEMM launches, L2-sized decode groups |
| 337b2fa | dense EXL3 folded prefill: `gemm_cast` (fused bf16→fp16 input / fp16→bf16 output casts) from 1152 rows, bitstream decode in `reconstruct_folded` |
| 382bdd1 | GDN prefill: tiled Triton transposes into the conv and out to contiguous q/k/v |
| f99bad5 | GDN prefill: gated output norm reads the strided z slice in place |
| 6459d2f | server ABBA results |

The server ABBA ran on 382bdd1 (`pnew-exl3` snapshot). f99bad5 (norm, −0.11 ms per GDN layer at 8K)
came later and is covered by its micro-benchmark, its bitwise test, and the gate, which ran on
f99bad5.

## 1. What bounded the 8192-token expert path (ncu, `profile/ncu-moe8192-details.txt`)

At 757caee, one MoE prefill at 8192 tokens (256 experts, top-8, 5-bit mul1) took 11.4 ms:

* **Decoded grouped GEMM** (`_exl3_gemm_kernel`): tensor-pipe bound, 82% for gate_up and 67% for
  down. The hi/lo Hadamard epilogue that keeps it exact adds 12.5% (gate_up) and 50% (down) of the
  GEMM FLOPs. Each launch also carried ~60 µs of early-exit CTAs (grid over all row blocks of all
  experts; `results/empty-grid.log`).
* **`_had_rows_kernel`**: 86% tensor pipe (~114 TF/s of the ~121.8 TF/s fp16/fp32-acc peak).
* **`_reconstruct_experts_kernel`**: DRAM-write bound. W_hat of a 32-expert group is larger than L2,
  so the GEMM re-read it from DRAM.
* **`_splitk_combine_kernel`**: DRAM bound (fp32 o, 537 MB).

That is why 757caee was 3-5x faster than before at 9-200 tokens but only −3% at 8192. At 8K the
path is tensor-pipe work plus a W_hat round trip through DRAM, not launch or latency bound.

### Fixes (all bitwise equal; `tests/kernels/test_exl3_bitexact.py`)

1. **Ranged launches.** The GEMM grid covers only the group's row blocks. The block range comes
   from a device-side `group_blocks` prefix (no host sync), and the kernel loops
   `pid_m in range(mb0 + pid, mb1, nprog)`.
2. **L2-sized decode groups.** `PREFILL_DECODE_L2_FRACTION=0.5` of the device's L2 sets the group
   size: 8 experts for gate_up and 16 for down on the 5080 (64 MB L2). W_hat is then written and
   read back in L2. The group is derived from `torch.cuda.get_device_properties().L2_cache_size`,
   capped at 32 with a floor of 4, so it adapts to any GPU and to any bit width or shape.
   Sweep: `results/ab-group-ranged.log`, `results/ab-l2frac.log`.
3. **Bitstream-order decode.** Code t = 32·cl + i sits at r = 8·i1 + 4·i4 + 2·i3 + i0 and
   c = 8·i2 + cl. `_decode_bands` reads the trellis words in stream order and forms the tile with a
   register reshape/permute instead of per-element index math. `RECON_KT=2` bands per program.
   Test: bits 1-8 × 3 codebooks × 3 layouts against a verbatim copy of the old indexed kernel.

Profile after the fixes (`results/prof-f50.log`, 8192 tokens): GEMM 5.64, had_rows 1.42,
reconstruct 1.19, combine 0.61, act 0.20 ms (total 9.13 ms). nsys shows 0.22 ms of gaps
(`results/nsys-moe.log`). What is left is at the tensor-pipe and DRAM bounds of an exact
computation.

### Rejected (measured)

* **Reconstruct on a side stream, overlapped with the GEMM**: 26-106% slower. Both fight for the
  same SMs and the L2 (`results/ab-overlap.log`).
* **Decode inside the GEMM at 8K**: best config 9.3 ms vs 5.3 ms for reconstruct + GEMM
  (`results/sweep-inl-8k-gu.log`).
* **GEMM tile sweeps**: the production config was already the best (`results/sweep-dec-gemm-gu.log`,
  `results/sweep-dec-gemm-dn.log`).
* **Sharing gate/up `suh`**: not possible, the vectors differ in the checkpoint.

## 2. EXL3 dense layers

For ≥ 1152 rows, `_forward_folded` runs `gemm_cast`: a Triton sequential-k fp32-accumulate GEMM that
reads the bf16 activations, casts them to fp16 in registers, and writes `acc.to(fp16).to(bf16)`. The
double rounding of the old cast → cuBLAS → cast chain is kept.

It is bitwise equal to cuBLAS wherever cuBLAS does not choose split-K. For the Ornith shapes on this
box, cuBLAS picks split-K only at K4096→2048 for M = 1024 and 1059-1088 (`results/split-map2.log`),
hence `FUSED_CAST_MIN_ROWS = 1152`. Below that, the cuBLAS path is unchanged.

Tests: `test_gemm_cast_equals_cast_cublas_cast`, `test_folded_fused_cast_path_equals_cublas_path`.

## 3. GDN + norms

In the 80K fresh saver trace at 757caee (`profile/base-buckets-popt-sched-traces.txt`, popt-sched's
nsys traces), generic torch copies took 1136 ms, with the transposes running at ~270 GB/s. Most of
them came from my layers:

* `conv_in.transpose(0,1).contiguous()` before the causal conv (279 ms), and FLA `input_guard`
  making the transposed q/k/v contiguous (~290 ms). These are now **tiled Triton transposes**
  (`kernel/triton/transpose.py`): a pure copy, 959 → 355 µs per 8192² bf16
  (`results/bench-transpose.log`). The conv writes straight out to contiguous q/k/v, so
  `input_guard` has nothing left to copy. One GDN prefill layer at 8192 tokens (conv to chunk
  output) went 3.71 → 2.59 ms (`results/bench-gdn.log`).
* The z copy before the gated RMSNorm (47 ms). z is a column slice of the in_proj output.
  `rms_norm_gated_heads` launches the same fla kernel with tokens as rows and heads as column groups
  (weight repeated per head, rows per block of the old launch): 405 → 293 µs per layer at 8192
  (`results/bench-norm.log`). Prefill only; decode paths are unchanged.
* The fp16↔bf16 casts around the dense cuBLAS GEMMs (~300 ms) are gone at ≥ 1152 rows through
  `gemm_cast` (section 2).

Tests: `tests/kernels/test_gdn_prefill_copies.py` (36 passed, `results/t4-gdncopies.log`). They check
the transpose against `.t().contiguous()` over shapes, strides and dtypes, and the whole GDN prefill
path against the old one, including conv states, recurrent states and chunk output. They also check
the heads norm against the row norm over T ∈ {1, 3, 17, 300, 8192} and three head layouts.

Not pursued: FLA chunk-kernel tuning (chunk_h, chunk_o, kkt, w_u; ~380 ms at 80K together). A
`num_warps` change alters the `tl.sum` reduction order, so bits could change for well under 1%.

## 4. D3 (shared-expert fusion): not pursued, with numbers

* **Prefill.** At 8192 rows the shared expert is 0.39 ms (2048→1024) + 0.22 ms (512→2048), against a
  9.8 ms routed MoE (`results/ab-dense.md`). Its sigmoid-gated add is already one kernel
  (`fused_shared_expert_add_`). The routed GEMMs are tensor-pipe bound at 8K, so overlap has nothing
  to fill. Fusing it as a 9th route needs mixed bit widths in one grouped launch, which saves launch
  tails only.
* **Decode.** It already runs on a side stream (`moe.py _overlap_shared`).

## 5. Kernel A/B vs 757caee (tree switch, CUDA events, sha1 of outputs)

5-bit (production Ornith), 4 ABBA rounds (`results/ab-dense.tsv` → `results/ab-dense.md`):

| shape | 757caee ms | new ms | change | sha1 |
|---|---:|---:|---:|---|
| MoE 8192 tok uniform / skewed | 11.37 / 11.42 | 9.95 / 9.96 | −12.5% / −12.7% | identical |
| MoE 2048 | 5.42 / 5.46 | 3.81 / 4.24 | −29.7% / −22.4% | identical |
| MoE 512 | 4.31 / 4.32 | 2.89 / 3.10 | −33.0% / −28.4% | identical |
| MoE 64 | 2.43 / 1.83 | 2.42 / 1.82 | 0% | identical |
| dense 2048→12288 (GDN in_proj) @8192 rows | 4.50 | 4.11 | −8.8% | identical |
| dense 4096→2048 (out_proj) @8192 | 1.55 | 1.35 | −13.1% | identical |
| dense 2048→1024 / 512→2048 (shared) @8192 | 0.457 / 0.280 | 0.389 / 0.223 | −14.8% / −20.7% | identical |
| dense @1808 rows | | | −7 to −35% | identical |
| dense @1100 rows (cuBLAS path kept) | | | −5% to +8%, sub-10 µs noise | identical |

## 6. Server ABBA (:1920, q8_0 KV, 262144, ratio 1.00, `abchain.sh`; b = 757caee, n = 382bdd1; order b n n b)

Cells are pass 2 of the probe, means of the two arms per tree. Per-arm tables are in the
`compare-*-arms.txt` files next to each summary.

### Fresh prefill / TTFT and file-read extends

Saver (`results/ab1/compare-saver.txt`):

| shape | 757caee | new | change |
|---|---:|---:|---:|
| TTFT 300 tok | 0.377 s | 0.338 s | −10% |
| TTFT 1000 tok | 0.353 s | 0.317 s | −10% |
| 8K prefill | 8476 tok/s | 9506 | +12.2% |
| 32K prefill | 7752 | 8302 | +7.1% |
| 80K prefill | 6018 | 6746 | +12.1% |
| 100K context (fresh) | 5538 / 4366 | 6045 / 5124 | +9% / +17% |
| extend 100K+10K | 3562 / 3590 | 3635 / 3866 | +2% / +8% |
| extend 100K+30K | 3438 / 3456 | 3717 / 3756 | +8.1% / +8.7% |
| extend 200K+30K | 2270 / 2302 | 2409 / 2444 | +6.1% / +6.2% |

For the extend rows, the two cells are pass 1 / pass 2; each pass uses its own context prefix.
Extends at depth are attention-dominated (4.8 s of attention against 1.5 s of MoE at 100K+30K), and
attention belongs to popt-sched.

Whole (`results/ab1/compare-whole.txt`):

| shape | 757caee | new | change |
|---|---:|---:|---:|
| TTFT 300 tok | 0.413 s | 0.823 s (0.426 s without n-w1) | see note |
| TTFT 1000 tok | 0.383 s | 0.395 s | +3% (noise band) |
| 8K prefill | 8754 | 9908 | +13.2% |
| 32K prefill | 7814 | 8876 | +13.6% |
| 80K prefill | 6214 | 6679 | +7.5% |
| extend 100K+10K | 3639 / 3738 | 3922 / 3854 | +7.8% / +3.1% |
| extend 100K+30K | 3528 / 3586 | 3682 / 3708 | +4.4% / +3.4% |
| extend 200K+30K | 2344 / 2380 | 2392 / 2426 | +2.0% / +1.9% |

Note on the whole-model 300-token TTFT: arm n-w1 pass 2 measured 1.216 s once. Its pass 1 was
0.406 s, and the other three arms were 0.404 / 0.429 / 0.423 s (`results/ab1/compare-whole-arms.txt`).
It is a single outlier. Whole-mode small-prompt TTFT is flat.

Where that TTFT goes (base trace `_orch/popt/profile-base/ext-whole-fresh-1000.sqlite`): about 2.5 ms
of compute-stream idle between `_router_triton_kernel` and `moe_align_block_size` in each of 39 MoE
layers (95.8 ms total). That is the prefill layer-stream wait
(`layers/moe.py _wait_prefill_overlap`, offload cache), popt-sched's area, not the EXL3 kernels. The
fused EXL3 path has no host sync before `moe_align`.

### Ornith 4.0bpw checkpoint (saver, `results/ab4/compare.txt`)

| shape | 757caee | new | change |
|---|---:|---:|---:|
| 8K prefill | 8325 tok/s | 9784 | +17.5% |
| 32K prefill | 7513 | 8820 | +17.4% |
| 8K decode | 202.9 | 208.5 | +2.8% (noise) |

All out_sha1 are identical. b-q1 was the first start on this checkpoint and is slower than b-q2
(`results/ab4/compare-arms.txt`). Against b-q2 alone, 8K prefill is +12% and 32K is +14%.

Kernel A/B at 4 bits (`results/ab-dense-4bit.md`): MoE 8192 −12.3%, 2048 −30%, 512 −33%; every hash
identical.

Checkpoint: `ultimatechris/Ornith-1.5-35B-A3B-EXL3-4bpw`, revision 6507c778b466b78eaaad0d7d552f98625bd97881
(19.8 GB), in `~/ai/models/Ornith-1.5-35B-A3B-exl3-4.0bpw`. Its `quantization_config`:

```
{'quant_method': 'exl3', 'version': '1.4.2', 'bits': 4.0, 'head_bits': 6, 'calibration': {'rows': 180, 'cols': 2048}, 'out_scales': 'always', 'codebook': 'mul1', 'mtp_bits': 4}
```

The codebook is **mul1**, the same as the 5.0bpw hq checkpoint. Another candidate exists:
`yeasah/Ornith-1.5-35B-A3B-exl3`, branch 4.00bpw (4.08 bits, v1.4.9, mul1, 20.2 GB).

### Decode (no-regression bar ~2%) and natural text

`results/ab3/compare-saver.txt` / `compare-whole.txt`, pass 2:

| mode | 8K | 80K | 256K |
|---|---:|---:|---:|
| saver 757caee → new | 171.3 → 176.9 | 159.2 → 159.2 | 115.9 → 114.2 (−1.5%) |
| whole 757caee → new | 184.2 → 192.0 | 161.9 → 156.9 | 119.7 → 120.5 |

Every probe decode sample (both passes, rounds ab1 + ab3, n = 8 per tree; `results/decode-passes.txt`):

| mode, prompt | change |
|---|---:|
| saver 8K | +2.6% |
| saver 80K | +1.0% |
| whole 8K | +1.1% |
| whole 80K | −1.6% (sd 3.9 / 4.7 tok/s) |

None of the changes touch a decode kernel. The GDN and norm changes are prefill-only, and dense
decode stays below 1152 rows.

Natural text, 5 tasks × 3500 tokens (`results/ab3/natural-compare.txt`):

| mode | 757caee | new |
|---|---:|---:|
| saver | 154.7 / 152.0 tok/s | 154.9 / 155.0 |
| whole | 167.7 / 167.7 | 167.9 / 166.6 |

md5 is identical 5/5 in all 8 arms.

## Gate and suite

GATE_PLACEHOLDER

## popt-sched's handoff items (`_orch/popt/handoff-popt-sched.md`)

1. **Copies around the dense GEMMs and GDN**: done, see sections 2 and 3.
2. **Routed experts at 3.2-3.4x roofline at extend depth**: the extend's MoE is the same prefill
   path, now −12% at 8192 and −22 to −30% at 2048 routes per chunk. The rest of the gap to a plain
   roofline is the exactness cost: the hi/lo Hadamard epilogue FLOPs, the W_hat reconstruct, and
   the fp32 combine (section 1).
3. **Triton recompiles per token count**: checked. No EXL3, GDN or norm kernel here takes the token
   count as a `tl.constexpr`. Rows, n_tiles and npad are runtime args, and the constexprs are
   bucketed block sizes, bit widths and flags. The new kernels (`_transpose_kernel`, `gemm_cast`,
   `rms_norm_gated_heads`, whose rows per block take ≤ 3 values) follow the same rule. Triton's
   integer specialisation (divisible by 16 / equals 1) bounds the variants per config.

## Knowledge for a successor

* The EXL3 MoE prefill at 8K now sits at exact-computation bounds: tensor pipe for GEMM and
  had_rows, DRAM for reconstruct and combine. Further gains need fewer FLOPs, i.e. a different
  exact formulation of the Hadamard epilogue, or a lower precision, which is rejected.
* The L2 group rule is device-derived. On a GPU with a smaller L2 the group floor is 4. Re-run
  `sweep_group.py` there if prefill looks off.
* `FUSED_CAST_MIN_ROWS=1152` comes from this box's cuBLAS heuristics (split-K at M ≈ 1024-1088 for
  K4096). Another GPU or cuBLAS version can pick split-K elsewhere; `split_map2.py` maps it. Below the
  threshold the old path runs, so a wrong threshold changes bits only where cuBLAS itself would
  split, and the folded-path test would catch it.
* Scripts: `go.sh` / `gpu.sh` (GPU lock, unit), `abtrees.sh` + `abtable.py` (kernel tree A/B),
  `abchain.sh` + `arm.sh` (server ABBA), `gate-chain.sh` (gate + suite), `decode_passes.py`.
