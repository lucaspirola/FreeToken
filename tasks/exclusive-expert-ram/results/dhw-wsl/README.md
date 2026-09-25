# Decode-level headroom probe, owner's RTX 5080 under WSL2 (local numbers, 2026-09-25)

Run by aa6ee56e with `dhw.sh` (this directory) on exp/decode-headroom 83b6c19 (= a2f08c0 + box results),
Nemotron NVFP4 mirror, ratio 1.00, rows -1, reserve 256, `FREETOKEN_DECODE_MEM_PROBE=1`, measure.sh.
Files copied from that worker's scratchpad (md5-verified); the `dhw-1m` row in `../sweep.tsv` is the
measure.sh append from this run.

* Every "Decode memory window closed" line (15, all arms): drop 0.00 GiB, reserved rise 0.00; allocated
  peak rise at most 0.01 GiB (1M).
* Every KV commit (34): overhead 0.00 GiB against the 0.25 GiB cushion.
* 0 coverage faults, 0 starved write-backs, one graph capture, 0 tracebacks per arm.

Decode tok/s p1 / p2 (local; host load not bracketed, single run each):

| Arm | 8K | 80K | 1M |
|---|---|---|---|
| dhw-np (default target) | 169.4 / 167.5 | 151.5 / 159.0 | - |
| dhw-128 (`FREETOKEN_DECODE_FREE_TARGET_MB=128`) | 159.2 / 163.5 | 146.4 / 153.4 | - |
| dhw-1m (default target) | 166.9 / 161.3 | - | 79.7 / 85.9 |

Reading: a 128 MiB decode target is feasible on WSL (no window needed more than the default leaves);
np vs 128 is one unbracketed pair each, so the speed difference is not attributable. The bracketed
comparison is the box A/B in `../dt*-box` (see `../dh2-box/README.md`).
