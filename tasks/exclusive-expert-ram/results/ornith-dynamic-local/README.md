# R-S12a on dynamic prefill headroom: Ornith 1.5 35B-A3B NVFP4, owner's RTX 5080 (WSL)

Run 2026-09-24 by `headroom-local.sh`, with HR_OUT set to this directory. Settings: ratio 1.00, port 1920,
262144-token KV, prompts 8K/32K/80K/128K, two passes, GPU at 0 MiB before every arm.

| arm | code | profile |
|---|---|---|
| ornith-dyn-whole / -saver | dynamic-transient 0f3505e / efbaa08 | dynamic prefill headroom (default on the branch) |
| ornith-st-whole | dynamic-transient efbaa08 | `FREETOKEN_DYNAMIC_PREFILL_HEADROOM=0` (static reservation) |
| ornith-st-saver | dt-measure worktree at efbaa08 | same |

A first ornith-st-saver start imported uncommitted compaction edits from the branch
worktree. It was stopped and re-run from a detached measurement worktree at efbaa08. None of
these arms has arena compaction (5374b4e).

Startup is identical on both profiles. The transient is 1.07 GiB, measured on an 8192-token
chunk. The arena is 512 -> 5352 of 6078 slots for whole and 4168 -> 5344 for the saver.
Dynamic arms release to 5824 slots for decode and reserve back down to about 5336 before each
prefill.

## Prefill (tok/s, monotonic; R-S12a is the 80K point)

| point | st-whole | dyn-whole | st-saver | dyn-saver |
|---|---|---|---|---|
| 32K p1 / p2 | 9887 / 9876 | 9444 / 9712 | 9530 / 9957 | 9265 / 9420 |
| 80K p1 / p2 | 5742 / 5740 | 5834 / 5647 | 5485 / 5621 | 5427 / 5562 |
| 128K p1 / p2 | 3890 / 3896 | 3899 / 3909 | 3850 / 3874 | 3822 / 3842 |

**R-S12a holds.** Dynamic 80K prefill is within 2% of static on both arms (-1.6% whole p2,
-1.1% saver p2). The reserve before each prefill costs no measurable prefill time.

## Decode (tok/s, monotonic, 127 generated tokens)

| point | st-whole | dyn-whole | st-saver | dyn-saver |
|---|---|---|---|---|
| 8K p2 | 158.9 | 155.2 | 149.3 | 144.8 |
| 32K p2 | 154.2 | 146.0 | 151.1 | 135.5 |
| 80K p2 | 138.8 | 134.3 | 121.9 | 122.0 |
| 128K p2 | 127.5 | 125.9 | 118.8 | 114.9 |

Without compaction, dynamic decodes 1-10% slower on Ornith, although it holds about 490 more
slots during decode. The journal shows why. Each reserve evicts the top 488 slots whatever
their rank. Under the saver, each reserve also re-reads 61-218 GPU-only experts from the
checkpoint ("mirror restored coverage for N experts"). Each request's 127-token decode then
starts by re-admitting the hot experts it just lost. Arena compaction (5374b4e) exists to
remove exactly this cost: the dynamic arms are re-run with it on the same host.

## With compaction and the teardown fix (ornith-dc-*, dt-measure at ba7d1a6)

`scan_noreserve.py` finds 0 of 65 prefill batches started at the decode level on both arms.
The dyn-* arms above had 3 each, from the teardown-shrink bug fixed in ba7d1a6.

| point | st-whole | dc-whole | st-saver | dc-saver |
|---|---|---|---|---|
| prefill 32K p2 | 9876 | 9717 | 9957 | 10116 |
| prefill 80K p1 / p2 | 5742 / 5740 | 5816 / 5806 | 5485 / 5621 | 5639 / 5669 |
| prefill 128K p2 | 3896 | 3948 | 3874 | 3945 |
| decode 8K p2 | 158.9 | 147.2 | 149.3 | 155.4 |
| decode 32K p2 | 154.2 | 149.3 | 151.1 | 147.5 |
| decode 80K p2 | 138.8 | 134.5 | 121.9 | 122.0 |
| decode 128K p2 | 127.5 | 124.9 | 118.8 | 113.0 |

**R-S12a holds on the fixed code.** 80K prefill is 1% above static on both arms.

Compaction over 14 reserves:
* whole: 2401 experts moved (4.26 GB D2D, 2.6 ms per call).
* saver: 1502 experts moved (2.67 GB), 4735 duplicates dropped, 337 written back from the
  GPU and 0 re-read from the checkpoint.

Decode against static: the saver is within -5% to +4%. The whole arm is still 2-7% below
static. These arms run without `--moe-collect-stats`, so they have no hit rates. The Nemotron
8K re-check (recheck-local.sh) measures the per-request release gap
directly (gap1_ms).
