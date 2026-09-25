# Round 3 checkpoint: exp/reorg-round3 ceb482c, local RTX 5080 WSL + ft-dev suite (2026-09-25)

Round 3 = exp/reorg 3b19f4a + exp/mirror-reserve-min 7f7b11d (09bb7b1) + exp/lfu-natural 41aeabd
(fe9c422) + exp/decode-headroom 90f393a (ed515b2) + the 128 MiB decode free target as the default for
every model (ceb482c; `FREETOKEN_DECODE_FREE_TARGET_MB` still overrides). overlap-sync 068b80d is
not in round 3 (round 4). Kernel .so files are round 2's build. All decode numbers below are local
RTX 5080 numbers, ratio 1.00, embedder stopped, GPU otherwise empty.

## Verdicts
| gate | result | verdict |
|---|---|---|
| ft-dev full 7-dir suite, fresh JIT, /root/gpu.lock | rc 0, 3072 passed, 25 skipped, 0 failed, 0 illegal access (`box-suite/`) | PASS |
| 8K/80K whole vs mirror alternated x3 (median ratio, bar >= 91%) | 8K p1 95.4%, p2 98.6%; 80K p1 98.6%, p2 97.5% (`ck5-compare.txt`) | PASS |
| 1M mirror RAM, 12.26 +/- 0.6 GiB | 12.77 (chain, quiet host); 12.86 / 12.84 / 12.84 (bracket) | PASS |
| coverage faults / starved writebacks | 0 / 0 in every mirror arm | PASS |
| one graph capture | captures=1, kv_grows=46, tracebacks=0 (R3, whole and mirror); one capture in every bracket arm | PASS |
| memlock (R6) | ok, no pageable fallback | PASS |
| needles/recall vs whole-1m on the same commit | 0 differences, 17 items (`../ck5-needles-compare.txt`) | PASS |
| 1M decode within 9% of whole, pass by pass | chain: p2 90.6% (host load 7.9 vs 4.8); back-to-back bracket on a quiet host: all points >= 91.6% (below) | PASS (bracket) |
| natural text decode, whole vs mirror x2 | whole 167.0 / 169.8, mirror 164.2 / 164.4 tok/s (98.3%, 96.8%); outputs md5-identical in all 4 arms | PASS |
| Claude Code / omp / Codex replay | identical cached counts to ck4r2: CC clear 16609/18999 (TTFT 0.376 s vs cold 2.644 s), repeat/next 18913; omp 8619; Codex reports no usage (as in ck4r2), TTFT 0.33 s vs cold 1.32 s (`ck5-replay.json`) | PASS |

## The 1M decode point, and why the bracket was needed
The chain's mirror-1m (`../ck5-box-compare.txt`) read 1M p2 94.3 against whole-1m 104.1 (90.6%),
with 1-min host load 7.9 at the mirror's p2 decode window vs 4.8 at the whole's. Two candidate
causes: host noise, or the LFU aging default (64 steps, from exp/lfu-natural) on the
repeated-sentence probe. A first bracket (`bracket/`) was killed at 19:23 by the owner's driver
reset. `bracket2/` reran it after the WSL restart: a warm-up start, then whole-a, mirror-a,
mirror with `FREETOKEN_LFU_HALVE_STEPS=256` (the old period), mirror-b and whole-b, back to back,
one at a time. Before each arm: host lock, GPU <= 32 MiB (17 MiB = the owner's Windows
ChatGPT.exe, seen with the Windows nvidia-smi), MemAvailable >= 23 GiB. Load and SM clock were
logged every 5 s (`load5s.txt`). The table comes from `brtab.py`; the load and clock are the
samples at the start of each 1M decode window.

| arm | 8K p1 | 1M p1 (gap, load) | 8K p2 | 1M p2 (gap, load) | ram_gib |
|---|---|---|---|---|---|
| whole-a | 191.5 | 97.2 (9.8 ms, 1.06) | 178.6 | 98.4 (9.6 ms, 1.00) | 18.62 |
| mirror-a | 176.5 | 92.6 (10.4 ms, 1.03) | 172.9 | 99.6 (9.4 ms, 1.07) | 12.86 |
| mirror LFU 256 | 187.1 | 96.3 (9.7 ms, 1.09) | 185.6 | 95.1 (9.9 ms, 1.04) | 12.84 |
| mirror-b | 176.8 | 91.2 (10.4 ms, 1.12) | 173.6 | 95.0 (9.8 ms, 1.00) | 12.84 |
| whole-b | 193.1 | 98.2 (9.7 ms, 1.03) | 179.9 | 100.3 (9.5 ms, 1.40) | 18.57 |

SM clock was 2880-2932 MHz in every decode window.

Mirror vs the adjacent whole arm:
* mirror-a / whole-a: 8K p1 92.2%, 1M p1 95.3%, 8K p2 96.8%, 1M p2 101.2%.
* mirror-b / whole-b: 8K p1 91.6%, 1M p1 92.9%, 8K p2 96.5%, 1M p2 94.7%.

Against the mean of the two whole arms, mirror a/b read:
* 8K p1: 91.8% / 91.9%.
* 1M p1: 94.8% / 93.3%.
* 8K p2: 96.5% / 96.8%.
* 1M p2: 100.3% / 95.6%.

Every point is within 9%. At the same 1M p2 point, the chain's miss had a host load of 7.9, and the
bracket's arms had about 1.0. Conclusion: host noise.

LFU 256 vs 64 does not discriminate. It is better at 8K p1/p2 and 1M p1, but worse at 1M p2 than
mirror-a (95.1 vs 99.6), and inside the mirror-a/mirror-b spread at 1M p2. So nothing is
reverted; the 64-step default stays. The tightest point is mirror 8K p1 (the first request after
start), at 91.6-92.2% back to back. The x3 alternation's 8K p1 median is 95.4%.

## Chain arms (before the reset)
* whole-1m, ceb482c, 16:37-17:37 (chain start load 1.35):
  * 8K p1/p2: 193.6 / 176.9.
  * 1M p1/p2: 96.9 / 104.1.
  * ram_gib 18.55.
* mirror-1m, started at the first quiet poll (17:50:33, 1-min load 2.13):
  * 8K p1/p2: 181.1 / 175.8.
  * 1M p1/p2: 92.7 / 94.3.
  * ram_gib 12.77, rss_ready 13.81 GiB, cgroup anon 13.07 GiB.
  * cgroup sampler: `cg.tsv*`; host load: `hostload.txt`.
* Decode free target: releasing to decode now leaves 128 MiB free. That gives +48 slots
  (2024 -> 2136, against 2088 with the old 0.375 GiB target).
* Logs: `checkpoint-{whole,mirror}-1m.log`, `chain-status.txt`, `recheck-8k.log`.
  * Top-level files: `../ck5-{whole,mirror}-1m-*`, `../ck5-box-compare.txt`, `../ck5-needles-compare.txt`.
  * The whole-1m log's needles-compare traceback is expected: at that point the mirror arm had
    not run. The mirror run's compare is the one that counts (0 differences).
