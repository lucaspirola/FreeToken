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
