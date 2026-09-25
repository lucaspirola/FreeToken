# overlap-sync: the per-step host sync in triton `prepare_metadata` (BOX NUMBERS, ft-dev, 2026-09-25)

Box: ft-dev, Vast 52296107, RTX 5080 on PCIe gen4, 84 SMs. Every number below comes from that box.
Base: exp/reorg-round2 01548e7. A = /root/FT-os (01548e7). B = /root/FT-os2 (A + this fix + test + tools).
Ornith EXL3 arms use A = /root/FT-exl3p8 (exp/ornith-exl3 52020bc content) and B = /root/FT-exl3p9 (p8 + the same fix).
Box chains: /root/K/chain50 (A baselines), chain51 (tests, greedy, nsys, E2E) and chain51b (clean moe/kernels rerun).
Results are archived in `~/ai/box-archive/ft-dev-exl3/results` (md5-verified).

## Cause (direct evidence)

Decode under the overlap scheduler left the GPU idle for ~0.54-0.64 ms between consecutive graph replays.
`decode_timeline.py` sync mode (`torch.cuda.set_sync_debug_mode(1)` plus a warnings hook) records every
synchronizing call made while serving one 8K request with 128 decode steps:

| sync site | Nemotron whole | Nemotron saver | Ornith saver |
|---|---:|---:|---:|
| `attention/triton.py:319` `prefix_lens = torch.tensor(list, device=cuda)` | 128x | 128x | 128x |
| `attention/triton.py:321` `indptr = torch.tensor(list, device=cuda)` | 128x | 128x | 128x |
| everything else (prefill-only: `_invalidate_prefill_buffer`, `_arena_shrink`, fla `prepare_chunk_indices`) | 0 per decode step | 0 per decode step | 0 per decode step |

(`timeline-{nem-whole,nem-saver,orn-saver}-sync/sites.txt`)

`torch.tensor(list, device=cuda)` makes a pageable host->device copy, and that copy synchronizes the current
stream. In the overlap loop the current stream is ordered behind the forward still in flight, so preparing
step N+1 waited for step N to end. The GPU then sat idle while the host built N+1.

## Fix

`python/freetoken/attention/triton.py`: build the host lists in pinned CPU tensors and copy them with
`non_blocking=True`, as the fa / fi / trtllm backends already do. The `cumsum_` runs on the CPU tensor.
The caching host allocator keeps each pinned buffer alive until its copy event completes. The attention
computation is untouched.
Test: `tests/kernels/test_triton_attention.py::test_triton_prepare_metadata_never_synchronizes`
(decode and prefill-with-prefix). It runs `prepare_metadata` under `set_sync_debug_mode("error")`.
Negative control: the same test run against A's code (`PYTHONPATH=/root/FT-os/python`) fails 2 of 2 with
`RuntimeError: called a synchronizing CUDA operation` at `triton.py:319` (`tests-os-syncneg/pytest.txt`).

The fix is scheduler/attention machinery. It is model-agnostic: the same line is on every Triton-attention model's path.

## Results

### Greedy output gate (8K prompt, 1023 greedy tokens; `timeline-*-greedy-{A,B}/greedy.txt`)

| arm | A sha1 | B sha1 |
|---|---|---|
| Nemotron whole | 6972907068 | 6972907068 |
| Nemotron saver | 6972907068 | 6972907068 |
| Ornith EXL3 saver | 3554d1fe8c | 3554d1fe8c |

Identical in all three arms.

### nsys (`--cuda-graph-trace=graph`, 8K, 127 decode steps; `timeline-*-nsys*/split.txt`)

| arm | gap between replays A → B (median µs) | idle ms/token A → B | window ms/token A → B |
|---|---:|---:|---:|
| Nemotron whole | 644.4 → **68.2** | 0.783 → 0.208 | 11.688 → 11.117 |
| Nemotron saver | 538.7 → **75.0** | 0.751 → 0.205 | 12.273 → 11.739 |
| Ornith EXL3 saver | 560.6 → **78.5** | 0.533 → 0.047 | 7.706 → 7.129 |

The remaining ~70 µs gap is the host launch path between replays. No other per-step sync site is left:
the sync-mode lists above show only prefill sites besides these two.

### E2E, decode tok/s, pass 2 of record, bracketed A1 / B / A2

Nemotron 3.5 Lightning NVFP4, `scripts/serve-default.sh` on port 1920, ratio 1.00 (`nem-e2e-*`):

| residency | ctx | A1 | **B** | A2 | B vs max(A) |
|---|---:|---:|---:|---:|---:|
| whole | 8K | 172.8 | **181.7** | 172.8 | +5.2% |
| whole | 32K | 167.1 | **176.3** | 167.9 | +5.0% |
| whole | 80K | 161.6 | **169.4** | 161.7 | +4.8% |
| saver | 8K | 167.7 | **176.0** | 167.7 | +5.0% |
| saver | 32K | 161.0 | **171.7** | 163.8 | +4.8% |
| saver | 80K | 155.4 | **162.4** | 155.3 | +4.5% |

Ornith EXL3 5.0bpw saver, `tasks/ornith-exl3/fuse/job-e2e.sh`, PREROT=1 (`exl3e2e-os-orn-*`):

| ctx | A1 | **B** | A2 | B vs max(A) |
|---:|---:|---:|---:|---:|
| 8K | 151.7 | **159.2** | 152.0 | +4.7% |
| 32K | 144.3 | **151.1** | 144.6 | +4.5% |
| 80K | 121.9 | **126.8** | 122.0 | +3.9% |

Prefill is unchanged within noise: every B value is within 1% of A.
Every E2E run shows captures=1, tracebacks=0 and oom=0.
The gain is smaller than the ~8% the idle share suggested. Part of each replay gap is launch overhead that remains.

Saver counters (`stats.json`, mirror):

| run | swaps | stage_redirects | ring_full_fallbacks |
|---|---:|---:|---:|
| Nemotron A1 / B / A2 | 4891 / 4891 / 4891 | 3 / 3 / 3 | 0 / 0 / 0 |
| Ornith A1 / B / A2 | 54800 / 54800 / 54800 | 336 / 284 / 332 | 761 / 756 / 761 |

Neither redirects nor ring fallbacks rise with the earlier-issued metadata.

### Tests (box, under /root/gpu.lock)

| suite | A (FT-os) | B (FT-os2) |
|---|---|---|
| tests/moe | 411 passed, 5 skipped | 411 passed, 5 skipped (`tests-os2c-moe`) |
| tests/kernels | 569 passed, 7 skipped | 571 passed, 7 skipped (`tests-os2c-kernels`; +2 = the new test) |
| tests/engine | — | 259 passed, 2 skipped |
| tests/scheduler | — | 412 passed, 1 skipped |

The first B run of moe/kernels (`tests-os2-{moe,kernels}`) is void. Twelve stray EXL3 files had been copied
into the B tree: `QuantKind.EXL3` does not exist in round 2, which caused 178 failures and a collection error.
Those files were moved to `/root/K/os2-stray`, and chain51b reran both suites clean.

## Tools here

- `decode_timeline.py`, modes `nsys` / `sync` / `greedy`.
- `job-timeline.sh` (`MODEL=nemotron|ornith-exl3 RESIDENCY=whole|mirror MODE=...`).
- `nsys_split.py`: graph replay / eager / idle split and the gap between replays.
- `job-e2e-nemotron.sh`: serve-default.sh on :1920, warm-up, then 8K/32K/80K x2.
- `job-tests.sh`.

## Status

Verified on the box. Not merged: overlap-sync rides round 3 under the owner's checkpoint gate on the local machine.
