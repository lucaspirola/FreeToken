# Mirror expert RAM (FREETOKEN_MIRROR_EXPERT_RAM=1) — first full numbers
Date: 2026-09-18. Branch: exp/exclusive-expert-ram @ 5778445.

## Configuration
Same default profile as the baseline (`scripts/serve-default.sh`, memory-ratio
1.00, LFU, growable KV), except the host side is a bounded pinned mirror
(1605 rows = 7.73 GiB) that swaps rows with the GPU cache instead of holding
every expert. Decode is EAGER (graphs off) — see Known issue.

## Results (RTX 5080, Nemotron-3.5-Lightning-30B-A3B-NVFP4)

| metric                       | baseline | disk-backed (rejected) | mirror |
|------------------------------|----------:|----------------------:|-------:|
| host RAM (cgroup)            |   20.2 GiB |               14.3 GiB | 11.9 GiB |
| RAM saved vs baseline        |           — |                5.9 GiB |  8.3 GiB |
| decode tok/s (127-tok probe) |      176.4 |                  19.9 |    ~72 |
| TTFT (short prompt)          |     0.36 s |                3.12 s | 0.15 s* |
| 21K factual recall           |        ok |               corrupt |     ok |
| 74K factual recall            |        ok |                  n/a  |     ok |
| coverage faults / starved     |        n/a |               n/a    |  0 / 0 |

* TTFT 0.15 s with graphs; eager decode lengthens TTFT on longer prompts.

## Correctness evidence
- swap_smoke (byte-exact over routed steps, prefill, arena shrinks): PASS
- unit tests tests/moe/test_mirror_pool.py: 3/3; tests/scheduler: 380/380
- 21K-token recall (code planted mid-context): "SIERRA-7741" — correct
- 74K-token recall with KV-arena growth mid-request: model cites the planted
  code; 0 coverage faults, 0 starved writebacks, 38.5k swaps
- same-seed greedy vs baseline, 3 short prompts: 2/3 byte-identical; 1 diverges
  at char 299 into a coherent paraphrase (slot-order changes float reduction
  order — expected for any cache-layout change, not corruption)
- baseline re-run from this branch's tree: correct, full speed with graphs

## Bugs found and fixed on the way (all with repros)
1. materialize kernels published no victims/prior owners
2. "already in place" must come from prior_ids, not id_of_slot
3. slot relocation corrupted slot_for_id of overwritten owners
4. D2D relocation had to run before the H2D uploads
5. free stack rebuilt from stale HOST maps (236 wrong experts -> 0)
6. arena-shrink refill had the same host/device divergence (237 wrong -> 0)
7. prefill writebacks drained the free stack (4017 starved) -> prefill drops
8. post-sweep restore needs total-E rows -> boundary is a full warm start
9. mirror_warm_start must clear the maps first (83 wrong -> 0)

## Known issue
CUDA graphs must stay OFF: with graphs on, 21K answers come out degenerate
while every device counter reads clean — the eager swap copies race the
replayed GEMM on the bank rows. Fix (capture the copies into the replay or add
an explicit dependency) is the main follow-up; it should close most of the
72 -> 176 tok/s gap.

## Open follow-ups
- fix the graph-replay race, re-enable decode graphs
- 200K/600K prompts, acceptance.sh R3/R6, tests/moe promotion of swap_smoke
- one warm start per prefill->decode transition costs a full checkpoint
  re-read (~4 s); reuse rows still valid instead
