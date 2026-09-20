# What is in here

`sweep-memcurrent-superseded.tsv` is the first capacity sweep. Its RAM column is
`memory.current`, which counts page cache, and the arms do not fill page cache
the same way (the baseline reads its expert banks through the page cache, the
mirror pool reads with O_DIRECT). Two identical baseline arms measured 23.77 and
20.93 GiB that way, and the mirror arms came out *lower* the *bigger* the pinned
pool was. The decode/TTFT columns in that file are real; the RAM column is not a
measurement of what this branch changes. Superseded by `sweep.tsv`, which
records anonymous memory.

`sweep.tsv`'s RAM column went through three definitions before one answered the
question. The first two are recorded above and in measure.sh; the third,
`ram_gib`, is the MemAvailable the arm takes away from the host, measured
before the unit starts and again at steady state after the probe. It is the
only one that is indifferent to whether a profile pins anonymous or file-backed
pages, which is the difference between the baseline (mmap'd banks, charged to
`file`) and the mirror (a pinned pool). `anon_gib`/`file_gib`/`rss_gib`/
`locked_gib` are kept beside it as attribution, not as the answer.

`nemotron-r-baseline` (02:00) is CONTAMINATED and must not be read as the
baseline: NVFP4 kernel microbenchmarks were run on the same GPU while it was
measuring. Its 80K TTFT came out 11.19 s against 9.36 s for the identical arm
run 20 minutes earlier with the GPU to itself, and its prefill rate fell with
it. `nemotron-r-baseline2`, run after the sweep with nothing else on the
device, is the baseline of record. The RAM column is unaffected (the
contamination was GPU-side) and both agree.

## Where the host RAM actually lives (2026-09-21)

Classified by summing every resident page of every process in the arm's cgroup
and grouping by what backs the mapping (`/proc/<pid>/smaps`):

    baseline (rows=0)   15865 MiB /dev/zero   1921 [heap]  1052 [anon]  -> 19815 RSS
    mirror   rows=2100  18543 MiB /dev/zero   1899 [heap]  1054 [anon]  -> 22478 RSS

`/dev/zero` is what CUDA pinned host memory looks like under WSL2, which is
also why `free` reports it as `shared` and why the cgroup charges it to `file`
rather than `anon` -- the reason the second metric (anon) read 3 GiB for a
profile holding 15 GiB of banks.

The baseline's pinned region is 15.49 GiB, i.e. the expert banks (15.41) and
nothing else: that profile is exactly what it claims to be.

The mirror arm's is 18.11 GiB for a pool of 10.99 GiB. The 7.1 GiB difference
is NOT the expert banks -- the mirror path builds `banks` from `meta` tensors
and never calls `load_expert_banks` -- and it is not any pinned allocation in
the serving code, all of which are kilobytes.

### The 7.1 GiB, answered: power-of-two rounding in the pinned allocator

`MirrorExpertPool` allocated its banks with `torch.empty(..., pin_memory=True)`.
torch's caching host allocator rounds EVERY allocation up to the next power of
two before asking CUDA for it (`ATen/core/CachingHostAllocator.h`:
`size_t roundSize = c10::llvm::PowerOf2Ceil(size);` then
`allocate_host_memory(roundSize, &ptr)`). The pool asks it for two banks of
several GiB each, so:

    pool rows   pool size   actually pinned   the two big banks
     1700        8.90 GiB    9.11 GiB          3.95 -> 4 GiB each
     2100       10.99 GiB   18.11 GiB          4.88 -> 8 GiB each
     2500       13.08 GiB   18.12 GiB          5.81 -> 8 GiB each
     2944       15.41 GiB   18.12 GiB          6.85 -> 8 GiB each

Three of the four capacities land in the same (4 GiB, 8 GiB] bucket, so they
pin an identical 18.1 GiB -- 2.6 GiB MORE than the baseline's 15.49 -- and the
RAM knob appears not to move. That is the whole of the "mirror saves nothing"
result, and it is an allocator artefact, not the design. The baseline escapes
it because `load_expert_banks` allocates per layer, where the rounding has
little room to bite (15.41 -> 15.49).

Verified directly before changing anything, on this host: a
`pin_memory=True` tensor of 4.88 GiB grew resident `/dev/zero` by **8.07 GiB**;
the same allocation made pageable and pinned with `cudaHostRegister` over the
exact byte range grew it by **0.00 GiB** of overhead and still reports
`is_pinned() == True`. The pool now allocates pageable and calls
`freetoken.kernel.pinned.host_register` on the exact range -- the same call the
baseline's banks already go through -- and unregisters in `close()`.
Commit: "Mirror: pin the bytes the pool asked for, not the next power of two".

### superseded/pre-pinning-fix/

Every arm measured before that commit is in there. Its decode, TTFT and free-
eviction columns are honest (they were measured on the retention fix) but its
RAM column measures the allocator's rounding, not the pool, so no point on that
curve can be chosen from. The curve of record is the one re-measured after it.
