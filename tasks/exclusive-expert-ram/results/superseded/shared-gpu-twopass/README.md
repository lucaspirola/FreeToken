# Superseded: the two-pass re-run of the 2026-09-21 arms (shared GPU)

These are the `PROBE_PASSES=2` re-runs of the 1700 / 2100 / 2500 / 2944 /
baseline arms. **Every decode number here is void**: the piro-board embedder
held 4.3 GB of VRAM throughout, and `--memory-ratio` is a fraction of FREE
VRAM, so these servers sized themselves to a smaller card than the arm names
imply.

They are kept, rather than committed over the 2026-09-21 files or deleted,
because they are the evidence for two things that ARE valid: the two-pass
probe fixed the 46.4 tok/s outlier at 80K (a one-off KV-growth VMM commit and
graph recapture landing before or after the first token, depending on the
pass), and `ram_gib` moved by only ~0.2 GiB between runs, which is what makes
the host-RAM metric trustworthy.

The numbers of record are the ratio-1.00 arms on an EMPTY GPU:
`nemotron-baseline-r100`, `nemotron-auto-r100`, `nemotron-baseline-r100b`.
