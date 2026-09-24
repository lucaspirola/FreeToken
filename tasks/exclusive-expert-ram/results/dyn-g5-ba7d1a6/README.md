# dyn-g5 rerun at ba7d1a6: the merge gate for exp/dynamic-transient (box ft-g5, 2026-09-24)

All numbers here are box numbers. ft-g5 is a rented RTX 5080 (16303 MiB, driver 580.126.09, torch
2.11.0+cu130, 62 GiB RAM).

## Setup

- Code: exp/dynamic-transient ba7d1a6, which adds the fix for the OOM found in `../dyn-g5/`. The
  fix: a KV shrink keeps the decode level, so the next prefill reserves headroom again. The
  prebuilt `.so` files were copied in.
- Model: Nemotron-3.5-Lightning NVFP4, ratio 1.00, port 1920.
- Harness: `checkpoint-box.sh dyn2` with `ARMS="whole mirror-nd"` and `FT_SIZES="8000 80000"`,
  followed by the needles/recall post-pass. The driver script is `dyn2-g5.sh`, its log is `dyn2-g5.log`.

## Gate

| criterion | whole | mirror-nd |
|---|---|---|
| prefills that start at the decode level without a reserve (`scan_noreserve.py`) | 0 of 192 | **0 of 192** |
| "Prefill headroom reserved" / "released to decode" lines | 24 / 28 | 24 / 28 |
| OOM / tracebacks | 0 / 0 | 0 / 0 |
| captures (acceptance R3) | 1 | 1 |
| coverage faults | n/a | 0 |
| needles/recall vs whole (`dyn2-needles-compare.txt`) | reference | **0 differences** (17 SAME) |

- The 4 releases without a matching reserve are each followed by a KV commit or by no prefill at
  all. The scan counts only prefills, and it reports 0 at the decode level.
- The previous run (`../dyn-g5/`, efbaa08) OOM'd on mirror-nd at the needle prefill right after a KV
  shrink. That sequence (needle prefills after shrinks) now completes.

## Decode and prefill, tok/s (box)

| arm | 8K p1 | 80K p1 | 8K p2 | 80K p2 | 80K p2 prefill |
|---|---|---|---|---|---|
| whole | 182.1 | 165.4 | 184.7 | 168.5 | 7261 |
| mirror-nd | 168.4 | 152.6 | 171.9 | 157.8 | 7072 |

`dyn2-records-compare.txt` has a traceback because the script expects a mirror-1m arm, which this
two-arm run does not have. It says nothing about the gate.
