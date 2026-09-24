# R-S12a on the owner's RTX 5080 (WSL2) — reorg-headroom 6545f7b, 2026-09-24

Runner `tasks/exclusive-expert-ram/headroom-local.sh` (SKIP_CK4=1), transient units
`ft-measure-ornith-hr-*`, port 1920, ratio 1.00, nvidia-smi 0 MiB checked before each arm,
Ornith NVFP4, chunk 8192, two passes. Prefill = prompt tokens / monotonic TTFT
(`prefill_tok_s_mono`; the wall-clock field is unreliable on WSL2, see probe_decode.py).

| arm | 32K p2 prefill | 80K p2 prefill | 128K p2 prefill | 80K p2 decode |
|---|---|---|---|---|
| ornith-hr-whole-a (rows 0) | 10,599 | 6,119 | 4,159 | 145.0 |
| ornith-hr-whole-b (rows 0) | 10,315 | 6,046 | 4,148 | 143.8 |
| ornith-hr-saver (rows -1)  | 10,058 | 6,008 | 4,116 | 134.9 |

**R-S12a (≥ 5,000 tok/s at 32K and 80K): PASS on all three arms**, both passes (80K p1:
5,990 / 6,075 / 5,858). No half-speed chunks: the "bimodal" 80K runs are gone.
Startup measured the prefill transient at 1.07 GiB per 8192-token chunk (headroom 1.07 + 0.12 GiB
margin), so the arena stops 726 slots below the ratio plan (5352 of 6078).
Raw files: `*-probe.jsonl`, `*-record.json`, `*-journal.txt`, `*-dmon.txt`, `*-geometry.txt`.
