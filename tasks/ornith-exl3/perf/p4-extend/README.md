# P4: flashinfer extend attention on the EXL3 stack (Ornith EXL3 5.0bpw, box ft-dev, RTX 5080 gen4)

These are box numbers.

## The change

The following commits from exp/attn-prefill were cherry-picked onto exp/ornith-exl3:

| exp/attn-prefill | exp/ornith-exl3 |
|---|---|
| e954d20 | 8ee0ca0 |
| 320b4ab | eaabf81 |
| 94875cf | f8c8a7a |
| c0a9214 | 29a0ea5 |
| 5ea76b1 | 8700110 |
| 6894623 | f822192 |

The kernel's own results are in exp/attn-prefill f86057a, `tasks/attn-prefill/results`. The per-layer speedup at a 120K prefix on Ornith's 16q/2kv D256 geometry was 350.8 -> 154.9 ms (x2.26).

How the path works:
- The q8_0 prefix is dequantized into 16K-token bf16 blocks.
- flashinfer's FA2 prefill attends each block and returns the LSE.
- The blocks are merged with the causal pass over the chunk's own keys.
- `FREETOKEN_EXTEND_BACKEND=triton` restores the triton kernel.

Why this was prioritised: before P4, the 80K prefill profile (P2 tree, whole model, `results/exl3roof-p2d/prefill-80k-whole.txt`) put attention at 9.26 s of GPU time. That is 36.7%, and the largest single item: at an 80K prompt the attention per 8K chunk grows from 80 ms (chunk 0) to ~1.7 s by the last chunk.

## End to end (same P2 tree, triton vs flashinfer, whole + saver, ratio 1.00, `fuse/job-e2e.sh`)

Pass 2 of record:

| prompt | prefill tok/s, triton | prefill tok/s, flashinfer | TTFT | decode tok/s triton / flashinfer |
|---:|---:|---:|---:|---:|
| 8K | 6856 | 7101 | 1.17 -> 1.13 s | 126.2 / 126.0 |
| 32K | 6014 | 6733 | 5.32 -> 4.76 s | 129.3 / 127.6 |
| 80K | 4189 | **5412** | 19.11 -> 14.79 s | 117.3 / 116.4 |
| 128K | 3095 | **4516** | 41.36 -> 28.35 s | 107.7 / 108.8 |
| 256K | 1833 | **3114** | 139.71 -> 82.21 s | 87.8 / 86.3 |

- The target was 80K >= 5000 tok/s, and it is met with 5412.
- Pass 1 gives the same picture: 80K 4121 -> 5320, 128K 3094 -> 4501, 256K 1830 -> 3103.
- Both arms had captures=1 and no tracebacks or OOM, with the prefill transient at 1.12 GiB and arena 5304 of 6076 slots in both. The measurement runs on an 8192-token prefix, and the 16K dequant block evidently fits the headroom through 256K.
- Tests: 450 passed. That covers test_exl3, test_extend_flashinfer, test_triton_attention and test_kv_quant.

## Logits (`fuse/job-logits-extend.sh`, results/exl3logits-extend-p4)

Setup:
- The prompt is 76821 tokens, which is 10 prefill chunks.
- The triton run decodes 32 tokens greedily, and flashinfer is teacher-forced on those ids.
- A second valid triton tile (M64 N64 w8 s1) is also teacher-forced. It is the same math reordered, so it measures kernel noise.

| vs triton (the reference) | top-1 agreement | top-5 overlap | KL mean / max | max abs dlogit |
|---|---:|---:|---:|---:|
| flashinfer | 0.968 (30/31) | 0.929 | 6.39e-3 / 3.60e-2 | 5.79 |
| other triton tile (noise) | 0.968 (30/31) | 0.968 | 6.21e-3 / 4.14e-2 | 5.93 |

The flashinfer path deviates from triton about as much as a second triton tile does, so it is within kernel noise.
