# Dynamic prefill headroom on ft-g5 (native Linux, PCIe gen 5, RTX 5080), 2026-09-24

All numbers here are box numbers.

- Code: exp/dynamic-transient efbaa08, which is 06298b6 plus scripts only (`git diff 06298b6 efbaa08 -- python` is empty). It ran in the box worktree /root/FreeToken-dyn, with the prebuilt .so files copied in.
- Run: `FT_SIZES="8000 80000" ARMS="whole mirror-nd" checkpoint-box.sh dyn` (Nemotron NVFP4, ratio 1.00, port 1920). Driven by `dyn-g5.sh`, 12:02-12:29Z.

## Result: the start is fine; the pool arm OOMs at a later prefill

| arm | start | captures | 8K p1/p2 | 80K p1/p2 | needles/recall | OOM |
|---|---|---|---|---|---|---|
| whole | ok (arena parked 2298 -> 256, transient 0.65 GiB measured) | 1 | 165.7 / 185.2 | 168.6 / 172.0 | done | 0 |
| mirror-nd | ok (parked 2298 -> 1496) | 1 | 164.9 / 170.3 | 150.7 / 155.6 | not run (server dead) | 1 |

The journal shows "Prefill headroom reserved / released to decode" cycling as designed in both arms, for example `MoE slots 2200 -> 2128 (0.38 GiB released, 0.79 GiB free, target 0.77 GiB)`.

**mirror-nd failure** (`dyn-mirror-nd-journal.txt`, line 136):
1. The 80K probe completes. The KV shrinks `131072 -> 65536`, and the arena takes the headroom back: `Prefill headroom released to decode: MoE slots 2088 -> 2168`.
2. The next request is the first needle prompt, a 21K prefill. It arrives with no preceding `Prefill headroom reserved` line.
3. `mamba2_prefill` asks for 32 MiB with 24 MiB free and fails: `torch.OutOfMemoryError`, under `_prefill_scan`.

So the arena holds the prefill headroom when a prefill starts right after a KV shrink or teardown. That path skips the reservation. The whole arm walked the same sequence (80K, then needles) without failing.

R6 fails as usual on Vast, where memlock is capped at 64 KiB. R3 fails on mirror-nd only because of the traceback.
