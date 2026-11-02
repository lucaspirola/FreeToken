# LFM2.5-24B-A2B on the RTX 5080: profile and numbers (2026-11-02)

## Profile

`scripts/serve-lfm25.sh`, model `LFM2.5-24B-A2B-Uncensored-Heretic-exl3-5.0bpw-hq`.

- `--dtype float16`. The residual stream is fp32 (`add_rmsnorm_fp32` in
  `python/freetoken/models/lfm2_moe/model.py`).
- RAM saver on: the mirror pool holds the experts in about 4 GiB of pinned RAM.
- 131072-token context, q8_0 KV, single lane, prefill chunk 8192, memory ratio 1.00.

Branch `feat/lfm2-moe` contains these fixes:

- `fix/vram-cliff`. The engine reads free VRAM from NVML and keeps a 0.40 GiB WSL floor.
  This branch also contains `fix/lfm25-saver`, `fix/lfm25-predictor` and
  `perf/exl3-decode-group`.
- `perf/spill-at-idle`. The scheduler writes soft-session checkpoints while the GPU is idle.
- `fix/lfm25-router-precision`. The residual stream stays in fp32.

## Parity

The router worker measured the KL against exllamav3 (`.work/lfm25/router`, prompts up to
16K tokens). The mean KL was 0.154 with bf16 activations and 0.106 with fp16 activations.
The exllamav3 control is 0.106. That run measured no decode cost for fp16.

## fp16 overflow check at 112K tokens

The check used two prompts of about 112K tokens each: the Vim user manual (prose) and 22
Python 3.12 stdlib modules (code). Each request generated 256 tokens at temperature 0.

- Server log: no NaN, no inf, no traceback.
- Output: both answers are readable English. They are on topic and similar in quality to a
  bf16 control on the same input. No token repeats in sequence (longest run 1).
- In one fp16 code run with a different request nonce, the model repeated a list of six
  module names three times. The bf16 control was not run on that input.
- Decode at 112K: fp16 134.6 / 140.2 tok/s, bf16 136.0 / 142.1 tok/s (prose / code).

An offline probe recorded the max abs activation per layer during the 112K prefill.

| value | max abs | layer |
|---|---|---|
| residual (fp32) | 150.9 | 36 |
| feed-forward output (fp16) | 150.1 | 34 |
| mixer output (fp16) | 6.4 | 39 |

The values are the same for both prompts. The fp16 limit is 65504, so the margin is about
430 times. The probe does not see the values inside the EXL3 MoE kernels.

## Benchmark

`bench_api.py`, port 1931, prompts with unique prefixes, two passes, 5 s idle before each
request, 256 output tokens. The client uses `time.monotonic`. The server column is the
request trace `ttft_ms`, from arrival to first chunk. Raw rows: `bench-2026-11-02.jsonl`.

| prompt tokens | pass | TTFT s | prefill tok/s | decode tok/s | server TTFT ms |
|---|---|---|---|---|---|
| 9352 | 0 | 2.93 | 3188 | 184.1 | 2919 |
| 37232 | 0 | 3.43 | 10841 | 173.8 | 3432 |
| 74159 | 0 | 8.18 | 9062 | 150.5 | 8180 |
| 112448 | 0 | 13.37 | 8410 | 139.4 | 13368 |
| 9352 | 1 | 0.94 | 9939 | 187.5 | 939 |
| 37232 | 1 | 4.58 | 8132 | 177.8 | 4576 |
| 74159 | 1 | 7.88 | 9409 | 153.0 | 7879 |
| 112448 | 1 | 13.48 | 8342 | 139.9 | 13477 |

The first request after the start is slow (3188 tok/s) because the expert banks are still
being built. The numbers in `bench-real-2026-11-01.jsonl` were measured before the clock
fix of 2026-11-02 00:41. Do not compare them with these numbers.

## Soak

`.work/vram-cliff/bench.py` sizes 10.6K/20.8K/41.7K/62.3K/83.0K, 24 passes, no idle, with
`FREETOKEN_PREFILL_MEM_PROBE=1`.

- 120 of 120 requests completed. There was no NOT_READY and no traceback.
- Measured prefill transient: 0.84 GiB per 8192-token chunk.
- Lowest NVML free in a chunk: 0.873 GiB. Largest NVML drop in a chunk: 0.844 GiB.
- No chunk used more than the transient plus 0.2 GiB.
- Median prefill 8.4K to 11.3K tok/s. Median decode 191 tok/s at 10.6K and 154 tok/s at 83.0K.

## Open risks

- The fp16 check covers two prompts at 112K tokens. Values inside the EXL3 MoE kernels
  were not measured.
- The 2026-11-01 crash is not explained. In that crash, one chunk used at least 1.5 GiB
  against a 1.10 GiB transient. This soak did not reproduce it.
- The 37K request of pass 1 had 25% lower prefill throughput than pass 0. An idle-time checkpoint write
  may overlap the next request. This was not measured.
