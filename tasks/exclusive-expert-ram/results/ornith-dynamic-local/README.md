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
