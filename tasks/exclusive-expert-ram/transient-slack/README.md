# Prefill transient slack (box ft-dev, exp/dt-dma 583afc8 + diagnostics, 2026-09-24)

Question: the owner's WSL host measures the prefill transient at 1.00 GiB with an allocator peak of
0.59 GiB (Nemotron, `ck4-mirror-1m-journal.txt`). That is 0.41 GiB of free VRAM held through every
prefill.

## Cause

- **Native Linux already runs expandable segments.** `engine._ensure_expandable_segments` sets
  `expandable_segments:True` by default, and skips WSL because the first CUDA allocation failed there.
- **Forcing the native caching allocator on ft-dev reproduces WSL exactly.** Transient from
  `ts-box/*-server.log`, 8K chunk:

| model | allocator | transient (reserved rise) | allocator peak |
|---|---|---|---|
| Nemotron (mirror) | expandable (Linux default) | 0.59 GiB (602 MiB) | 0.59 GiB (603 MiB) |
| Nemotron (mirror) | native (`-ne`, = WSL) | **1.00 GiB (1026 MiB)** | 0.59 GiB |
| Ornith NVFP4 | expandable | 0.98 GiB | 0.97 GiB |
| Ornith NVFP4 | native | 1.07 GiB | 0.97 GiB |

**What the snapshot shows** (`ts-box/nemo-mirror-ne-analysis.txt`, from `FREETOKEN_TRANSIENT_SNAPSHOT`
+ `snap_analyze.py`). The native run creates 11 large segments, each exactly its request's size, so
there is no rounding slack.

- The Mamba/attention temporaries come first, and each gets its own segment: fp8 gemm 162, chunk_state
  128, conv 96, mamba2 64, embedding/rmsnorm 42+42, bmm 32, 16+16 MiB.
- Once freed they stay cached as separate segments.
- The MoE prefill buffers (`fused_nvfp4.py:685/696`, 174 and 252 MiB) fit in none of those segments.
  They get two new segments while 681-759 MiB sits free in the cache.
- Result: reserved 1026 MiB for a live peak of 603 MiB.

## Candidate fixes (native allocator, Nemotron mirror 8K chunk; `ts-box/captest*-status`)

| fix | reserved rise | vs peak | chunk time (prefix 0 / 8192) |
|---|---|---|---|
| none | 1026 MiB | +422 | 814 / 892 ms |
| (b) one cached segment of the peak (`FREETOKEN_PREFILL_WORKSPACE_TEST`) | 860 MiB | +256 | - |
| (c) allocator cap at reserved + peak + pad, pad 0-64 MiB | OOM | - | - |
| (c) cap, pad 96-128 MiB | 682 MiB | +78 | 834 / 917 ms (+2.4%) |
| (c) cap, pad 192-256 MiB | 774 MiB | +170 | 827 / 911 ms |

- **Ornith, cap pad 0:** reserved rise 838 MiB (blocks come from segments cached before the chunk),
  675.6 / 836.7 ms vs 667.7 / 829.4 ms uncapped (+1.2% / +0.9%).
- **The cap fails the gate on Nemotron:** +78 MiB over the peak and +2.4% chunk time. Segments that
  hold long-lived blocks cannot be released, so capping still leaves fragmentation.
- **Expandable segments on Linux (`nemo-mirror-cap`)** already meet the target: 602 vs 603 MiB,
  0 retries.

## WSL expandable-segments failure: the torch call that differs from our arena

In torch 2.11 (`c10/cuda/CUDACachingAllocator.cpp` @70d99e9, `ExpandableSegment::map`, ~lines
409-465), `cuMemCreate` runs with these handle types:

- It first tries `CU_MEM_HANDLE_TYPE_FABRIC`. A non-OOM error falls back.
- It then uses `CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR`, checked hard with `C10_CUDA_DRIVER_CHECK`.
- `TORCH_CUDA_EXPANDABLE_SEGMENTS_IPC=0` leaves `requestedHandleTypes = 0` (NONE).

Our arena (`kernel/csrc/vmm_tensor.cpp:67,127,216`) uses `CU_MEM_HANDLE_TYPE_NONE`.

`wsl_expandable_repro.py` checks this directly:

- **Part 1** calls `cuMemCreate` with each handle type, then reserves 1.125x VRAM of address space
  (torch's reservation per segment).
- **Part 2** runs torch in one process per setting: native, expandable, expandable with IPC=0, and
  expandable with IPC=1.

On ft-dev every torch setting passes. The driver-level results were NONE and POSIX_FD OK, FABRIC
NOT_PERMITTED, reserve OK (`ts-box/repro-native.txt`). The WSL run is still pending.
