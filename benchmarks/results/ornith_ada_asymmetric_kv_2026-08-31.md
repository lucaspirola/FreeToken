# Ornith RTX 2000 Ada asymmetric-KV port

Date: 2026-08-31  
Host: NVIDIA RTX 2000 Ada Generation (sm_89), 16 GiB, 70 W, WSL 2  
Stack: Windows driver 595.95, CUDA toolkit 13.1, PyTorch 2.11.0+cu130,
Triton 3.6.0

The growable, independently quantized K/V machinery added and validated on an
RTX 5080 is architecture-neutral.  Its decode launch tables were not: on sm_89,
both new mixed formats fell through to the conservative eight-split geometry.
This left batch-one long-context attention substantially under-parallelized.

## Accepted Ada launch policy

- Q8-K/Q6-V: 32 splits, 32-token KV tile, four warps.  Batch four caps the
  realized split count at 16 because its four streams already expose enough
  stage-one work.
- Q6-K/Q5-V: 64 splits, 32-token KV tile, eight warps.  A 128-split launch was
  narrowly faster for some near-262K multi-stream cases but regressed shorter
  contexts; 64 was the robust 16K--262K choice.

The sweep covered 16K, 64K, 131K and near-262K contexts, batches 1--4, splits
8/16/32/64/128, 32/64-token tiles, and four/eight warps. Every retained case
matched the dequantized BF16 oracle (`rtol=atol=2e-2`); observed maximum absolute
errors were at most 0.0002.

## Production-kernel results

Median isolated full-attention-layer latency, before (generic eight splits) and
after the accepted sm_89 table:

| KV pair / context / batch | Before | After | Reduction |
|---|---:|---:|---:|
| Q8-K/Q6-V, 65,536, 1 | 0.9226 ms | 0.3912 ms | 57.6% |
| Q8-K/Q6-V, 261,888, 1 | 3.2983 ms | 1.3670 ms | 58.6% |
| Q8-K/Q6-V, 261,888, 3 | 5.2347 ms | 4.0059 ms | 23.5% |
| Q8-K/Q6-V, 261,888, 4 | 5.5316 ms | 5.2746 ms | 4.6% |
| Q6-K/Q5-V, 65,536, 1 | 0.9984 ms | 0.4137 ms | 58.6% |
| Q6-K/Q5-V, 261,888, 1 | 3.9526 ms | 1.5708 ms | 60.3% |
| Q6-K/Q5-V, 261,888, 3 | 6.7174 ms | 4.5978 ms | 31.6% |
| Q6-K/Q5-V, 261,888, 4 | 6.8649 ms | 6.1143 ms | 10.9% |

## Live coherence gates

Both gates used the Q6_K Ornith checkpoint, one 32,768-token cold synthetic
needle prompt, 128 forced output tokens, 8K prefill chunks, a 65,536-token
growable ceiling, and a 32K growth step. Both crossed the growth boundary,
recaptured the CUDA graph, and recovered exact passcode `5663623`.

| KV pair | End-to-end prefill | Engine average | Decode | Final MoE slots | Result |
|---|---:|---:|---:|---:|---|
| Q8-K/Q6-V | 455.17 tok/s | 445.25 tok/s | 23.98 tok/s | 4,481 | PASS |
| Q6-K/Q5-V | 483.04 tok/s | 469.82 tok/s | 23.56 tok/s | 4,530 | PASS |

The end-to-end decode window includes the one-time KV-growth graph transition;
steady server status samples settled around 39 tok/s for both. These short live
runs are coherence/capacity gates, not replacements for the final long-context
pair benchmarks.

