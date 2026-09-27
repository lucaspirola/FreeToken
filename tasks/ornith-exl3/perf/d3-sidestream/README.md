# D3a: shared expert on a side stream (Ornith EXL3 5.0bpw, box ft-dev, RTX 5080 gen4)

These are box numbers, and every arm ran with nothing else on the GPU.

## The change

In EXL3 decode, each MoE layer's shared expert is a 7 bpw SwiGLU made of 2 GEMVs: gate/up 2048x1024 and down 512x2048. Before this change they ran on the main stream, serially with the router, the expert loads and the routed GEMVs. At ~8 + ~6 us they cost ~14 us per layer, and alone they reach only ~0.2 of the DRAM bound (see `../d2-gemv`).

What changes:
- **`models/qwen3_5_moe/moe.py`:** in EXL3 decode, the shared expert now runs on a per-device side stream. The side stream forks with `side.wait_stream(main)` and joins with `main.wait_stream(side)` before `fused_shared_expert_add_`. The fork/join is captured into the decode CUDA graph (captures=1). In eager mode, `record_stream` keeps the allocator away from tensors shared across the streams.
  - The EXL3 routed path returns a fresh tensor, so both streams may read `hidden_states`.
  - `FREETOKEN_SHARED_EXPERT_OVERLAP=0` restores the serial path.
- **`kernel/triton/exl3.py` (split-K counter lanes):** the in-kernel split-K reduction counts arrivals in a global int32 buffer. Each launch leaves that buffer zeroed. One buffer per device is only correct for GEMVs issued in stream order. Two GEMVs running concurrently count into the same slots, so the "last" split can sum another launch's planes. `split_counter_lane(1)` gives the side stream's GEMVs their own buffer.
  - Test: `test_concurrent_split_k_gemvs_on_their_own_counter_lanes_match_serial`. It runs dense GEMVs on a side stream against the routed decode on main, 50 iterations, and requires them to be bit-equal to serial.

**The first D3a run is void.** It had no lanes and measured 166.9 / 156.4 / 141.7 tok/s (`results/exl3e2e-d3ov1`). That run changed the model's output:
- The greedy haystack answer (84966-token prompt) came out as 1023 tokens with sha1 `3ca253284e`.
- The serial path gives 818 tokens with sha1 `ed5c265a64` (`results/exl3greedy-f16acc-ov{1,0}`).
- The cause is the shared-counter race described above. Racing splits that exit early also explain the "too good" speed.

## Output gate

Setup: greedy, 1024 tokens, temperature 0, 84966-token haystack prompt (`fuse/job-greedy-f16acc.sh ref`, SIZE 62000). With lanes:

| arm | tokens | sha1 |
|---|---:|---|
| overlap on (`results/exl3greedy-f16acc-ov1b`) | 818 | ed5c265a64dc |
| overlap off (`results/exl3greedy-f16acc-ov0b`) | 818 | ed5c265a64dc |

- The two arms are token-identical, as they must be: the change only moves work to another stream, and each GEMV's split order is fixed.
- Tests: 319 passed (`results/exl3tests-d3a2`, test_exl3 including the new lane test).

## (a) Saver E2E (mirror, ratio 1.00, flashinfer extend, PREROT=1, pass 2; bracketed off / on / off)

| prompt | off 1 | **on** | off 2 | change |
|---:|---:|---:|---:|---:|
| 8K decode tok/s | 134.7 | **146.3** | 134.8 | +8.6% |
| 32K | 128.9 | **139.0** | 129.1 | +7.8% |
| 80K | 110.9 | **117.8** | 110.9 | +6.2% |

- Prefill is unchanged: 7385 / 7429 / 7380 at 8K.
- Every arm had captures=1 and 0 tracebacks.
- The off arms reproduce D2's 134.7 / 128.9 / 111.0 exactly.

## (b) Whole-model E2E (`--expert-residency whole`: all experts in pinned RAM behind the GPU LFU cache)

| prompt | off | **on** | change |
|---:|---:|---:|---:|
| 8K decode tok/s | 145.5 | **159.8** | +9.8% |
| 32K | 136.5 | **148.7** | +8.9% |
| 80K | 124.0 | **134.1** | +8.1% |

Ornith EXL3's 20.2 GB of routed experts do not fit the 16 GB GPU, so this mode also crosses PCIe on misses. It is not a kernel-only number.

## (c) Profiled 8K decode (saver, torch.profiler, 127 tokens; `fuse/job-prof.sh`)

| per token, GPU time | D2 (serial, `results/exl3prof-d2`) | D3a (overlap, `results/exl3prof-d3a2`) |
|---|---:|---:|
| exl3 GEMV | 4.04 ms | 4.10 ms |
| copy: expert miss loads (`fast_index_copy_kinds`) | 1.70 ms | 1.69 ms |
| copy: saver writebacks (memcpy DtoH) | 0.62 ms | 0.62 ms |
| rest | 1.92 ms | 2.12 ms |
| GPU busy / decode window | 8.27 / 8.96 ms (92%) | 8.53 / 8.37 ms (102%: the streams overlap) |

- The GEMVs do the same work in both columns. Run concurrently they take 0.06 ms/token longer in total, but they are off the critical path.
- The copies are at the PCIe bound, 23-26 GB/s on gen4 (`bench_mirror_copy`). The mirror counters are identical in both runs: 2276 swaps and 739 writebacks over the request.

**Derived no-miss decode at 8K** (D3a):

| basis | ms/token | tok/s |
|---|---:|---:|
| E2E ms/token (1/146.3) minus the miss loads | 6.84 - 1.69 = 5.15 | **194** |
| the same, also minus the writebacks | 4.53 | 221 |
| lower bound: profiled window (profiler overhead included) minus both copies | 8.37 - 2.31 = 6.07 | 165 |

- Every basis clears the ft-dev kernel target of >= 150 tok/s.
- The first basis assumes the miss loads sit serially on the critical path, which is where the main stream issues them.

## Dual-BITS fusion (D3b) is skipped

The side stream saves 1/134.7 - 1/146.3 = **0.59 ms/token** on the saver, and 0.61 ms/token on the whole-model run. That is 14.7 us per MoE layer over 40 layers: the whole ~14 us/layer the shared expert cost. Nothing is left to recover by fusing the 7 bpw shared expert into the 5 bpw routed launch (dual BITS). That fusion would add a second codebook path to the grouped kernel for no gain.

The per-token work that remains is:
- the copies (1.69 + 0.62 ms), which are PCIe;
- "rest" (2.12 ms), which is D3c: router, `_ensure_experts_sized_kernel_v2`, cuBLAS split-K reduce and small elementwise ops.
