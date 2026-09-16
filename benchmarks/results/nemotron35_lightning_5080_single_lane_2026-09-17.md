# Nemotron 3.5 Lightning 30B-A3B NVFP4 on RTX 5080 / WSL2 — single-lane default profile (2026-09-17)

Profile: `scripts/serve-default.sh` at 5915936 (system unit `freetoken-serve`).
`--max-running-requests 1 --linear-state-slots 13 --kv-grow-step-tokens 65536 --num-tokens 1048576
--max-seq-len-override 1048576 --kv-cache-dtype q8_0 --moe-backend offload --moe-cache-auto
--moe-cache-policy lfu --memory-ratio 0.91 --max-prefill-length 8192 --host-ram-reserve-gb 0
--session-spill-ram-gb 1 --session-spill-disk-gb 50`, env `FREETOKEN_EXPERT_ARENA=1
FREETOKEN_GROWABLE_OVERLAP=1 FREETOKEN_PIN_BUDGET_GB=17`. One session resident on the GPU,
every other session checkpointed to RAM/disk and swapped back in. Raw files (gitignored):
`benchmarks/switchyard_soak/runs/single-lane-df06962/`.

## Decode / prefill vs prompt size (one request, 128 generated tokens, thinking off)

| prompt tokens | prefill tok/s (chunk 4096) | prefill tok/s (chunk 8192) | TTFT 8192 | decode tok/s |
|---:|---:|---:|---:|---:|
| 8K   | 3 783 | 3 908 | 2.1 s | 148–150 |
| 32K  | 8 788 | 9 922 | 3.2 s | 157–174 |
| 80K  | 6 211 | 8 794 | 9.1 s | 151–172 |
| 128K | 6 621 | 6 980 | 18.3 s | 163–164 |
| 256K | 3 814 | 3 969 | 64.5 s | 150–151 |

Decode is flat in prompt size (Mamba-2 + q8_0 KV). Chunk 8192 is the default (prefill +10–40 %,
decode unchanged). Previous profile (16 lanes, main 4cad61f): 75 tok/s for one request alone,
16–41 tok/s per lane under load.

## Expert residency / KV growth

MoE cache 1924 slots (1985 with 6 GDN slots; 1023 on the 16-lane profile). KV grew
64K → 256K → 64K on demand during the sweeps and 24 times during the soak, always with the
expert arena at full size and **one** CUDA-graph capture per process (startup) — zero recaptures.

## Soak (switchyard-soak, 16 concurrent clients, 5 scenarios, private Switchyard on :4001)

| run | duration | requests | errors | rps | p50 | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 16-lane stage (4cad61f) | 20 min | 470 | 0 | 0.33 | 20.2 s | 179 s | 332 s |
| single-lane stage | 10 min | 319 | 0 | 0.51 | 24.0 s | 77 s | 94 s |
| single-lane passthrough | 10 min | 725 | 0 | 1.18 | 13.5 s | 18.6 s | 20.0 s |
| single-lane passthrough, 13 GDN slots | 5 min | 353 | 0 | 1.14 | 13.6 s | 20.8 s | 22.3 s |

Server counters over the single-lane soaks: 0 tracebacks, 0 scheduler-invariant violations,
spills 1320 / 0 failed, restores 405 (56 failed with 6 GDN slots — "no GDN snapshot slot
available for cold session restore", fixed by 13 slots: 164 restores / 0 failed). Host
MemAvailable stayed 5.4–7.2 GB (reserve 0 by the owner's choice).

## Crash fixed on the way

The 16-lane passthrough soak killed the scheduler after 12 clean minutes
(`core.py:130 append_host: assert m <= max_device_len`): `overlap_loop` kept a stale local
`last_data` after a message-path cold-restore drain, so a pending shrink drained the same batch
twice (double `append_host`). Fixed in 8f4b584 with a regression test
(`tests/scheduler/test_growable_overlap_drain.py`).
