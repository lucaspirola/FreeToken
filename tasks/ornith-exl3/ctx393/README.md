# ctx393: 1.5x context (393216 tokens, YaRN factor 2 over the native 262144), measured

Branch `exp/ctx393` (worktree `FreeToken-wt/ctx393`), code = main `c0f73ad6`, unchanged (measurement
only; `python/` is not modified and `scripts/serve-default.sh` stays at 262144). Assignment
(owner-approved 2026-09-28): measure `--num-tokens 393216 --max-seq-len-override 393216
--rope-yarn-factor 2 --rope-yarn-original-context 262144` on top of the default profile (q8_0 KV, ratio
1.00), saver and whole, to the protocol of the popt/kfix READMEs.

**Verdict: 1.5x context is usable.** Exact recall holds to the ceiling. A single needle is found at
10/50/85/98% depth in 300K, 380K and 390K prompts (the deepest at token ~382,130 of 389,964), and
three needles at 2/50/97% come back in order: 15/15, saver and whole give identical answers. Speed at
the new sizes is what the attention cost predicts: 380K fresh prefill 2702 tok/s saver (TTFT 141 s),
decode 94 tok/s. Nothing below 262144 got slower from serving with YaRN (same-day control, section 4).
RAM/VRAM fit, but the saver pins 1.39 GiB more RAM, and a full-length context has 20% fewer GPU
expert slots.

What does change: static YaRN rescales RoPE at **every** position, so outputs differ from the 262144
profile even for short prompts. All 5 natural-text md5 differ, and every probe out_sha1 differs. The
texts are coherent and correct (the math task gives the same 21 h / 8:00 answer). The gate's needle
battery scores 13/14 with YaRN against 12/14 for the non-YaRN ck9c reference. Recall is identical. I
found no point where recall starts failing; the stretched range was probed up to 99% of the ceiling.

## How it was run

`chain.sh` (unit `ctx393-chain`) ran 9 arms back to back with `arm.sh` (popt-exl3's, plus
`YARN_ORIG` and a `NEEDLE=1` post). Each arm took `gpu-host.lock`, waited for nvidia-smi 0 MiB and
MemAvailable >= 23 GiB, and served on :1920 as a `ft-measure-<arm>` user unit. y-* = 393216 with YaRN
2/262144; c-* = same-day control at 262144 without YaRN. Every start line (load, GPU MiB,
MemAvailable, SM clock, commit) is in `results/status.txt`. Host samplers: `results/load5s.txt`
(load, SM clock, util) and `results/mem5s.txt` (GPU MiB, MemAvailable, swap), every 5 s.
`report.py` builds every table below into `results/report.md`.

| arm | what | rows |
|---|---|---|
| c-sp / c-wp | probe 1K/8K/32K/80K/128K/256K, 2 passes, 262144 no YaRN | saver / whole |
| y-sp / y-wp | probe 1K/8K/32K/80K/128K/256K/262K/300K/380K, 2 passes, 393216 YaRN | saver / whole |
| y-sq / y-wq | 8K probe, then the gate's `needles.py 21000 120000` (thinking budget 65536) + `recall.py 21000 120000 240000`, then `depth_needles.py` | saver / whole |
| c-sn / y-sn / y-wn | natural text, 5 tasks x 3500 tokens (`natural_gen.py`) | saver / saver / whole |

Probe prompt sizes leave room for the output: the 380000 target is 380015 prompt tokens + 128
generated. `depth_needles.py` builds its haystacks with the model's tokenizer, so a 390000 prompt is
389964-389971 tokens + at most 96 answer tokens < 393216. recall.py's word-count guess would have
missed this; its 240000 is 197993 tokens.

## 1. Start, RAM and VRAM (`results/report.md` "Arms", journals, records)

Clean start in all 9 arms: captures=1, 0 tracebacks (`*-acceptance-R3.txt`), 0 coverage faults and
0 starved writebacks in every saver arm (`*-record.json`). The journal confirms YaRN
(`rope_yarn_factor=2.0, rope_yarn_original_context=262144`). Memory prediction checks
(`*-journal.txt`), identical in both profiles:

| check | 262144 (c-sp) | 393216 YaRN (y-sp) |
|---|---|---|
| non-expert weights | 2.50 vs 2.69 GiB (-7%) | 2.50 vs 2.73 GiB (-8%) |
| CUDA graph pool | 0.03 vs 0.04 GiB (-24%) | 0.03 vs 0.04 GiB (-27%) |
| prefill transient (8192 chunk, gdn) | 0.88 vs 0.94 GiB (-7%) | 0.88 vs 0.96 GiB (-9%) |
| linear-state pool | 0.78 vs 0.78 GiB (+0%) | same |

No prediction WARN in any journal. The graph pool's -27% is 0.01 GiB absolute; popt-exl3 read
-24% on the same line.

| | c-sp saver 262144 | y-sp saver 393216 | c-wp whole 262144 | y-wp whole 393216 |
|---|---:|---:|---:|---:|
| KV physical at the ceiling | 2.73 GiB | **4.06 GiB** | 2.73 | 4.06 |
| GPU expert slots: start -> at full KV | 4952 -> 3856 | 4912 -> **3088** | 5016 -> 3920 | 4984 -> 3168 |
| saver pool pinned (`*-geometry.txt`) | 7290 rows, 13.45 GiB | **8042 rows, 14.84 GiB** | – | – |
| ram_gib / rss at ready (record) | 17.29 / 17.43 | 17.84 / 18.92 | 21.97 / 22.87 | 21.97 / 22.82 |
| peak cgroup memory | 18.79 | 19.84 (y-sq 21.46) | 27.35 | 25.85 (y-wq 26.19) |
| min MemAvailable during the arm (`mem5s.txt`) | 10.22 GiB | 8.57 (y-sq 7.16) | 4.96 | 4.70 (y-wq 4.53) |
| GPU used, peak | 15098 MiB | 15102 | 15082 | 15098 |

* **VRAM**: ratio 1.00 fills the free card in every arm (14.1-15.1 GiB used). The 393216 ceiling is
  paid in expert slots: KV grows in 65536-token steps, and each step takes ~360 slots. From 262144 to
  393216 the ceiling costs 1.33 GiB of KV = 768 fewer slots (3856 -> 3088, -20%), only once a
  context actually passes 262144 (`Committed growable KV` lines in `y-sp-journal.txt`).
* **RAM, saver**: the pool pins +752 rows (+1.39 GiB), because it must cover the complement of the
  smaller GPU resident set at full KV. It fits: MemAvailable never went below 7.2 GiB on this 33 GiB
  host.
* **RAM, whole**: unchanged (21.97 GiB either way). MemAvailable bottomed at 4.5-5.0 GiB in both
  profiles, so the ceiling adds no host-RAM risk there.

## 2. Fresh prefill, TTFT and decode (probe pass 2 of record, `y-sp-probe.jsonl`, `y-wp-probe.jsonl`)

| target | prompt tok | saver TTFT | saver prefill tok/s | whole TTFT | whole prefill | saver/whole prefill |
|---|---:|---:|---:|---:|---:|---:|
| 1K | 1019 | 0.51 s | 1989 | 0.41 s | 2502 | 79.5% |
| 8K | 8011 | 0.90 s | 8861 | 0.82 s | 9793 | 90.5% |
| 32K | 32024 | 4.01 s | 7994 | 3.69 s | 8683 | 92.1% |
| 80K | 80025 | 12.73 s | 6286 | 11.93 s | 6710 | 93.7% |
| 128K | 128027 | 24.77 s | 5168 | 23.42 s | 5467 | 94.5% |
| 256K | 256022 | 72.75 s | 3519 | 69.83 s | 3666 | 96.0% |
| 262K (native ceiling) | 262025 | 75.48 s | 3471 | 72.77 s | 3601 | 96.4% |
| **300K** | 300021 | **94.44 s** | **3177** | 91.20 s | 3290 | 96.6% |
| **380K** | 380015 | **140.65 s** | **2702** | 137.04 s | 2773 | 97.4% |

Pass 1 is within 0.6% of pass 2 from 32K up in saver, and within 3.4% in whole (`results/report.md`). At 8K, pass 1 is a prefix hit on
measure.sh's warm-up and does not count. The 1K cell is sub-second TTFT jitter.

Decode tok/s (pass 1 / pass 2, 128 tokens after the prompt):

| context | saver | whole | saver/whole p2 |
|---|---:|---:|---:|
| 8K | 190.3 / 178.8 | 205.1 / 186.9 | 95.7% |
| 80K | 152.5 / 152.6 | 147.5 / 158.4 | 96.3% |
| 256K | 112.4 / 112.0 | 116.6 / 116.7 | 96.0% |
| 262K | 101.4 / 98.8 | 105.3 / 106.6 | 92.7% |
| 300K | 104.0 / 97.2 | 107.7 / 104.2 | 93.3% |
| **380K** | **91.6 / 94.2** | **97.7 / 99.3** | 94.9% |

The step from 256K (112) to 262K (99-101) is the KV step, not YaRN: 262025 + 128 tokens crosses
262144, so KV commits the 5th step (327680) and the arena drops from ~3816 to ~3456 slots. Past that,
decode falls with attention length (97 at 300K, 94 at 380K). Every out_sha1 is identical between
saver and whole at every size (`results/report.md`, "saver / whole, YaRN").

serve-default.sh's comment quotes an earlier 384K measurement: "384K prefill 2466-2521 tok/s, decode
90-92 tok/s" (pre-popt tree). Today 380K is 2702-2773 tok/s prefill and 94-99 tok/s decode.

Natural text, 5 tasks x 3500 tokens (`*-natural.json`):

| arm | tok/s | md5 |
|---|---:|---|
| c-sn saver 262144 | 154.8 | 5/5 = ck9c and popt-exl3 ab3 (8a3ec4b3, ef957d71, 1c460027, e243f5b3, a450eb53) |
| y-sn saver YaRN | 153.8 | 5/5 different from c-sn (186498b3, e063bebf, 20ce984f, 38300dbf, a4f2137c) |
| y-wn whole YaRN | 166.5 | 5/5 = y-sn |

Saver/whole natural = 92.4% (popt-exl3 ab3: 92.2%, ck9c 90.2-91.0%).

## 3. Needle and recall correctness in the stretched range (`y-sq-depth.jsonl`, `y-wq-depth.jsonl`)

`depth_needles.py`: thinking off, temperature 0, one fresh prefill per request, each prompt with its
own header line (no shared prefix). Single = "The vault passphrase is X." at the given depth, asked
for alone. Multi = three authorization codes at 2/50/97%, "list them in order".

| kind | prompt tok | depth | planted | saver answer | s | whole answer | s |
|---|---:|---|---|---|---:|---|---:|
| single | 299965 | 10% | AMBER-LYNX-3255 | AMBER-LYNX-3255 | 93.9 | same | 91.3 |
| single | 299966 | 50% | SAFFRON-MARTEN-3242 | SAFFRON-MARTEN-3242 | 93.5 | same | 91.0 |
| single | 299966 | 85% | SAFFRON-MARTEN-4943 | SAFFRON-MARTEN-4943 | 93.4 | same | 91.1 |
| single | 299965 | 98% | AMBER-OTTER-7515 | AMBER-OTTER-7515 | 93.1 | same | 91.3 |
| single | 379966 | 10% | INDIGO-OTTER-7382 | INDIGO-OTTER-7382 | 138.9 | same | 136.9 |
| single | 379966 | 50% | COBALT-MARTEN-6404 | COBALT-MARTEN-6404 | 139.0 | same | 135.8 |
| single | 379966 | 85% | COBALT-MARTEN-2636 | COBALT-MARTEN-2636 | 139.4 | same | 137.1 |
| single | 379966 | 98% | SAFFRON-MARTEN-8147 | SAFFRON-MARTEN-8147 | 139.3 | same | 137.0 |
| single | 389966 | 10% | INDIGO-FALCON-2515 | INDIGO-FALCON-2515 | 145.4 | same | 143.1 |
| single | 389966 | 50% | INDIGO-FALCON-8697 | INDIGO-FALCON-8697 | 145.5 | same | 143.0 |
| single | 389966 | 85% | SAFFRON-HERON-4658 | SAFFRON-HERON-4658 | 145.4 | same | 143.0 |
| single | 389964 | 98% | VIOLET-HERON-7299 | VIOLET-HERON-7299 | 145.4 | same | 143.2 |
| multi | 299970 | 2/50/97% | INDIGO-HERON-4406, COBALT-LYNX-1793, VIOLET-LYNX-4431 | all three, in order | 93.0 | same | 91.3 |
| multi | 379971 | 2/50/97% | VIOLET-HERON-6365, SAFFRON-HERON-4324, SAFFRON-LYNX-9979 | all three, in order | 138.4 | same | 136.0 |
| multi | 389971 | 2/50/97% | SAFFRON-FALCON-5175, SAFFRON-FALCON-3130, SAFFRON-LYNX-7473 | all three, in order | 145.0 | same | 142.4 |

**15/15 PASS in both modes; the saver and whole answers are character-identical.** The 390K / 98%
needle sits at token ~382,130 (382125 tokens of text precede it, plus the chat template). Nothing degrades toward the ceiling in this test: the answers are
exact, with no extra text, and finish_reason=stop everywhere. Limits of this test: it is literal
retrieval from a uniform filler. The reasoning-type questions (below) only ran at <= 120K, the
battery's sizes.

**Gate battery under YaRN** (`y-sq-needles.json`, `y-sq-post.txt`; the reference is ck9c's
non-YaRN whole arm, copied to `results/ref-ck9c-whole-*`):

| size | YaRN (saver = whole) | ck9c reference (262144, no YaRN) |
|---|---|---|
| 21K (17647 tok) | 6/7: counting FAIL (answered 6, truth 7) | 6/7: ordering TRUNC (65535 tokens of reasoning) |
| 120K (99207 tok) | 7/7 (ordering in 3523 tokens) | 6/7: ordering TRUNC; counting "ok" but also hit 65535 |
| recall.py 21K/120K/240K | 3/3 codes each, answers identical to the reference | 3/3 each |

* saver vs whole under YaRN: `results/y-needles-compare.txt`, **0 differences** (every field of
  every question plus the recall lines).
* YaRN vs the non-YaRN reference: `results/y-sq-vs-ck9c-needles-compare.txt`, 13 differences. These
  are the wording and length of the thinking answers, expected when RoPE changes. The recall lines
  are SAME.

The one miss (21K counting, 6 vs 7) is a single greedy sample, and it is identical in saver and
whole, so it comes from the model under YaRN, not the residency. On the other side, YaRN finished
both `ordering` questions where the reference ran out its 65536-token budget. I read this as "no
measurable quality loss on this battery", not as a proof of equality.

## 4. 262144-native numbers vs popt-exl3 (sanity check, no gate re-run)

popt-exl3 README, round ab3 (382bdd1, the same lineage; mean of 2 arms, pass 2; probe sequence
8K/80K/256K) against today's arms (pass 2):

| | popt-exl3 ab3 | c-* today, 262144 | y-* today, YaRN 393216 |
|---|---:|---:|---:|
| saver prefill 8K / 80K / 256K | 8959 / 6621 / 3659 | 9153 / 6669 / 3596 | 8861 / 6286 / 3519 |
| saver decode 8K / 80K / 256K | 176.9 / 159.2 / 114.2 | 170.9 / 153.0 / 105.2 (p1 110.2) | 178.8 / 152.6 / 112.0 |
| whole prefill 8K / 80K / 256K | 9672 / 6755 / 3723 | 10071 / 6457 / 3637 | 9793 / 6710 / 3666 |
| whole decode 8K / 80K / 256K | 192.0 / 156.9 / 120.5 | 190.2 / 148.1 / 104.4 (p1 120.9) | 186.9 / 158.4 / 116.7 |

YaRN / same-day control, pass 2 (`results/report.md`):

* **Prefill**: saver 94.3-97.9% (32K-256K), whole 97.1-103.9%.
* **Decode**: saver 97.6-106.5%, whole 96.6-111.8%.

Single arms on both sides, and the controls themselves spread by as much. c-wp's 256K decode went
120.9 -> 104.4 between its two passes. Against popt-exl3 the YaRN arms are within -5.1% (saver 80K
prefill) to +1.3%, and today's control shows the same 80K decode offset (153 vs 159). YaRN only
changes the cos/sin table, so no compute is added per token. Conclusion: **no regression at <= 262K
from serving with YaRN.** A saver prefill gap of about 3% at 32K-256K would need an ABBA to separate
from arm spread.

Lineage check: c-sn's natural text is md5-identical to ck9c and popt-exl3 (5/5), so today's main
without YaRN produces exactly the gated outputs. Probe out_sha1 cannot be compared across rounds
because it depends on the probe's size sequence. The same commit gives different 8K hashes in
popt-exl3's ab5 (sizes 300/1000/8K/...) and its control-256k (8K/256K). Within this round, saver ==
whole at every size.

## Open points

* The saver pool is sized for the ceiling (+1.39 GiB pinned) whether or not a session ever passes
  262144. If ctx393 becomes an option, that RAM is the price even for short sessions.
* Short-context outputs change under static YaRN (section 2 md5, section 3 battery wording). This run
  measured no quality loss, but it is one sample per question. A dynamic/NTK-by-parts YaRN that only
  applies beyond 262144 would keep short outputs bit-identical. That is a code change and out of
  scope here.
* Not run (out of scope by the assignment): ck9x gate, ABBA replicates, file-read extends at depth
  (`extend_probe.py`) beyond 262144.

## Files

`chain.sh`, `arm.sh`, `depth_needles.py`, `report.py`, `natural_gen.py` (popt-exl3 copy);
`results/`: per arm `*-probe.jsonl`, `*-record.json`, `*-journal.txt`, `*-geometry.txt`,
`*-acceptance-R3.txt`, `*-measure.log`, `*-stats.json`; needles `y-{s,w}q-depth.jsonl`,
`y-{s,w}q-needles.json`, `y-{s,w}q-post.txt`, `y-needles-compare.txt`,
`y-sq-vs-ck9c-needles-compare.txt`; natural `*-natural.json` + texts; `status.txt`, `load5s.txt`,
`mem5s.txt`, `report.md`.
