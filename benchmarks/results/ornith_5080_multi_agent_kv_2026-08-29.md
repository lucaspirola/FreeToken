# Ornith RTX 5080 elastic multi-agent KV validation

Date: 2026-08-29
Host: NVIDIA GeForce RTX 5080 (sm_120), 16 GiB
Workload: two independent 70,000-token prompts, unique numeric needles, agent 0
established one second before agent 1, 8,192-token prefill chunks, LFU, greedy
decode with EOS ignored to keep the teardown timing deterministic.

Growable KV now uses one shared VMM arena with an independent page-table row per
request. Admission still reserves each request's complete remaining context, while
physical growth is driven by aggregate batch demand. When both prefill and decode
are runnable, the scheduler runs 32 decode steps between prefill chunks. This keeps
the established agent responsive without replacing efficient homogeneous prefill
and decode kernels with a mixed-phase batch.

At request teardown FreeToken evicts unlocked finished prefixes, compacts only the
surviving requests' private tail pages into low free holes, decommits complete CUDA
VMM mappings, and rebuilds the expert cache upward. Radix-owned protected pages are
not moved; if one pins a high segment, shrink stops conservatively at that segment.

| Gate | Q6_K + Q8_0 KV | Q4_K_M + INT4 KV |
|---|---:|---:|
| Growth step | 131,072 | 65,536 |
| Live KV before helper exit | 262,144 | 196,608 |
| MoE slots before helper exit | 3,483 | 5,644 |
| Live KV after helper exit | 131,072 | 131,072 |
| VRAM physically returned | 1.33 GiB | 0.35 GiB |
| MoE slots after helper exit | 4,036 | 5,842 |
| Main tokens after helper exit | 161 | 161 |
| Post-teardown decode | ~130–131 tok/s server steady state | 154.04 tok/s final 80-token tail |
| Own needle / foreign needle | PASS / absent | PASS / absent |
| Aggregate prompt throughput | 669.49 tok/s | 307.50 tok/s |

The Q4 prompt throughput is specific to this repetitive two-agent isolation text
and is not a replacement for the retained long-context performance gate. Its purpose
here is lifecycle correctness across two growth events and the reverse transition.

The low-level gate also verifies that each growth step is a distinct physical CUDA
mapping. CUDA on this host rejects partial unmapping of one allocation, so shrink is
implemented as the exact reverse of whole step commits. Four VMM/MHA tests cover
commit, pointer stability, decommit, data preservation, and recommit. Scheduler tests
cover aggregate demand, prefix-before-growth policy, private-page compaction, suffix
removal, and bounded decode starvation.
