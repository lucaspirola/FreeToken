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

## Graphs-mode root cause + fix (same day)
The graphs-on corruption was NOT a race: a pure CUDA-graph replay never runs
host code, and the prefill->decode boundary warm start (host + disk, inside
ensure_experts) could therefore never fire after any post-capture prefill —
every replayed miss hit an empty mirror and read pool row 0 silently.
Repro: tasks/exclusive-expert-ram/graph_race_repro.py (91 wrong over 60 pure
replays, no server needed). Fix: Scheduler._forward consumes the boundary
flag at the batch boundary (always host-visible), before the replay is
admitted; the gate re-enables graphs.

Also fixed the 600K mid-flight death generically: MirrorExpertPool derives
its coverage bound from its own geometry, and the growable-KV arena may not
shrink below it (a funding failure names FREETOKEN_MIRROR_HOST_ROWS instead
of dying hours later). No model constants anywhere; floor verified across
three synthetic geometries.

## Final numbers (graphs ON, default sizing)

| metric | baseline | mirror |
|---|---:|---:|
| RAM at startup | 20.2 GiB | ~13.2 GiB |
| decode (127-tok probe) | 176.4* | **229** |
| 21K recall | ok | SIERRA-7741 ok |
| 240K recall (200K ask) | ok | ORION-3391 ok |
| 713K recall (600K ask, 1M ceiling) | ok | VEGA-5527 ok, 0 faults |
| coverage faults / starved over all runs | n/a | 0 / 0 |

* baseline re-run from this tree measured 271 warm; 176.4 was the campaign
  number with cold-warmup amortized.

Decode during a 713K session reads ~77 tok/s while the expert cache sits at
its arena floor (1144 slots); the floor is a KV-growth tradeoff, restored to
1520 when the session releases KV (log: "MoE slots 1144 -> 1520").

## Acceptance
- R3: captures=1, grows=10, tracebacks=0 (this run) -> PASS
- R6: LimitMEMLOCK=infinity, banks pinned, 0 "settled pageable" -> PASS
- tests/scheduler + tests/moe: 600 passed, 5 skipped
- swap_smoke + graph_race_repro promoted to tests/moe/test_mirror_device.py

## Open follow-ups
- output equality vs baseline beyond 3 short prompts (slot-order paraphrase
  is expected, corruption is not)
- optional: trim boundary warm start (GPU re-fill dominates; ~4 s per
  transition, noise inside real prefills)
