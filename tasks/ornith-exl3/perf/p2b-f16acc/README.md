# P2b: fp16-accumulate prefill GEMMs (Ornith EXL3 5.0bpw, box ft-dev, RTX 5080 gen4)

These are box numbers. The flag is `FREETOKEN_EXL3_F16ACC` (see "Gate" below for its default).

## The change

- **Dense projections at >= 1024 rows:**
  - `gemm_f16acc` (triton) multiplies the raw bf16 activations by each folded 2048-column W_full slab.
  - It accumulates in fp16, as exllamav3 does, and writes bf16 straight into the output slice.
  - This replaces the bf16->fp16 cast, the cuBLAS fp32-accumulate GEMM and the fp16->bf16 slab copy.
- **MoE decoded-slab GEMM:** the accumulator is fp16. The tiles were re-swept: BM 16/32/64 by rows per expert, BK 64, 2 stages, 8 warps.
- **MoE down output:** it is written per route in fp16. The combine sums in fp32 either way and now reads half the bytes.

## Per-kernel budget at 8192 rows (`bench_f16acc.py`, `bench_moe_prefill.py 8192 --quick`)

| kernel | fp32 accumulate (P2) | fp16 accumulate (P2b) |
|---|---:|---:|
| GDN in_proj GEMM 2048x12288 | 3423 us (120.5 TF/s, cuBLAS) | 1896 us (217.5 TF/s) |
| attn qkv GEMM 2048x9216 | 2634 us (117.4 TF/s) | 1479 us (209.1 TF/s) |
| o_proj GEMM 4096x2048 | 1194 us (115.1 TF/s) | 700 us (196.4 TF/s) |
| shared gate/up GEMM 2048x1024 | 311 us | 203 us |
| shared down GEMM 512x2048 | 165 us | 100 us |
| MoE layer total | 12.20 ms (P1) / 10.82 ms (P2 tiles) | **9.72 ms** |
| of which exl3_gemm (decoded slab) | 6.37 / 5.67 ms | 4.77 ms |
| of which reconstruct / had_rows / combine | 2.64 / 1.68 / 0.65 ms | 2.64 / 1.69 / 0.34 ms |

The dense-GEMM rel err against fp32 accumulation is 0.8-2.4e-3 on random data.

## End to end (whole model + saver, ratio 1.00, flashinfer extend, PREROT=1, pass 2; `fuse/job-e2e.sh`)

| prompt | P4 (fp32 acc) prefill tok/s | P2b (fp16 acc) | decode tok/s P4 / P2b |
|---:|---:|---:|---:|
| 8K | 7101 | **8523** | 126.0 / 126.0 |
| 32K | 6733 | **7963** | 127.6 / 128.3 |
| 80K | 5412 | **6232** | 116.4 / 118.0 |
| 128K | 4516 | **5078** | 108.8 / 109.0 |
| 256K | 3114 | **3371** | 86.3 / 85.9 |

- Both runs had captures=1 and no tracebacks.
- The prefill transient is 1.06 GiB, so the arena gains slots: 5336 against 5304.
- Tests: 318 passed.
- Results: `results/exl3e2e-p2b`.

## Gate: the flag stays OFF by default

### Logits (`fuse/job-logits.sh`, ratio 0.90, teacher-forced against exllamav3; `results/exl3logits-p2b`)

With the flag on, both sets stay inside the spread the earlier kernels show against exllamav3:
- short: pre1 KL mean 2.48e-2, top-1 0.903.
- long: pre1 KL mean 4.20e-2, top-1 0.935.
- For comparison, the long set gave P1 3.57e-2, pre-P1 4.34e-2 and P2 1.04e-2.
- Saver vs whole is bit-equal.

### Logits at 83K (`fuse/job-logits-f16acc.sh`; `results/exl3logits-f16acc-p2b`)

Setup: 83221 tokens, 31 positions teacher-forced on the fp32-accumulate greedy ids. The noise yardstick is the same fp32 build with the triton extend tile instead of flashinfer.

| vs fp32 accumulate | top-1 | top-5 | KL mean / max |
|---|---:|---:|---:|
| fp16 accumulate | 30/31 | 0.948 | 1.05e-2 / 1.17e-1 |
| noise yardstick | 29/31 | 0.955 | 7.28e-3 / 4.71e-2 |

### Greedy output (`fuse/job-greedy-f16acc.sh`; `results/exl3greedy-f16acc{,-multi}`)

Setup:
- Five haystack prompts (85K-124K tokens), 1024 tokens each, temperature 0.
- Every arm is run-to-run identical.
- The reference is fp32 accumulation; the noise arm is the fp32 build on the triton tile.

| | fp16 accumulate | noise yardstick |
|---|---|---|
| first divergent token (per prompt) | 5, 102, 0, 0, 0 | 7, 93, 0, 0, 5 |
| quoted facts right (of 15; the reference: 14) | **13** | 14 |

- The divergence positions are the same as the noise yardstick's.
- In substance, fp16 accumulation made one recall error that neither other arm made. On the 107K-token prompt it quoted record 4872 as "Chiara left a lantern in **Nagoya**"; the reference and the noise arm have Porto, which is correct.
- The other four prompts agree in substance: same records, same cities.
- One extra error in 15 facts is not proof of a recall regression, but it is not "within noise" either.

So `FREETOKEN_EXL3_F16ACC` stays off by default. Before it can become the default, a larger recall run is needed, such as the ck4 needles/recall suite on this build with the flag on and off.
