# Step 6: split-K reduction inside the EXL3 GEMV and fused decode epilogues (box ft-dev, RTX 5080, 2026-09-24)

All numbers here are box numbers: rented RTX 5080, ratio 1.00, port 30100, one run at a time. The code is exp/ornith-exl3
9b896eb. The E2E runs used scratch/exl3-t3, which is scratch/exl3-dma b85402f with the same change (cf0ab06).

- `FREETOKEN_EXL3_SPLITK_INKERNEL` (default on) moves the split-K reduction into the GEMV. The
  last-arriving split sums the fp32 planes in split order, so the result is deterministic. Setting it to 0 restores
  the separate `torch.sum`.
- `FREETOKEN_EXL3_FUSED_EPILOGUE` (default on) fuses two groups of steps. `splitk_silu_had` fuses the cast, the
  silu*up and the down-projection input rotation. `splitk_combine` fuses the top-k combine. Setting it to 0
  restores the separate ops.

## Layer bench (`t3-bench.txt`, CUDA graph replay)

| | unfused | in-kernel sum | + fused epilogues |
|---|---|---|---|
| MoE layer (H 2048, I 512, top-8) | 58.4 us | 56.9 us | 51.7 us |
| dense o_proj 4096->2048 | 25.3 us | 24.0 us | 22.8 us |

The largest difference from the unfused path is 9.8e-4 (fp16 rounding), and 0 for the dense path. Tests: 299 passed
(`t3-tests.txt`).

## End to end (s5 probe, Ornith EXL3 5.0bpw + RAM saver, ring rule 66 rows)

Decode tok/s at 8K/32K/80K/128K:

| code | p1 | p2 |
|---|---|---|
| rule control (b85402f) | 81.2/98.0/94.6/92.3 | 118.1/115.0/103.0/99.6 |
| **+ task 3 (cf0ab06)** | **84.4/101.9/98.6/97.4** | **122.6/120.8/107.8/102.0** |

That is +2.4 to 5.5% at every point. The greedy chats differ in wording from the control. The
logits below show why.

## Logits (`logits/`, teacher-forced on the old path's ids, 31 tokens)

| prompt | top-1 agreement | KL mean / max | max abs dlogit |
|---|---|---|---|
| short | 30/30 | 5.9e-3 / 3.8e-2 | 2.16 |
| long (repeat 40) | 29/30 | 3.3e-3 / 2.3e-2 | 2.73 |

The same tool gives a kernel-noise yardstick. Two valid triton attention tiles on Ornith (the attn-prefill
logits run, 54K prompt) differ by a mean KL of 1.4e-2 and top-1 of 0.905. The change reorders
fp32 sums and rounds once less. Its logits stay within that noise; they are not identical.
