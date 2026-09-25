# Step 3 — `ft serve` on the full Ornith EXL3 5.0bpw-hq checkpoint, RAM saver on (ft-dev RTX 5080 box, 2026-09-24)

Launcher: `step3.sh` (flags modelled on scripts/serve-default.sh: single lane, growable KV
step 65536 up to 262144 tokens, q8_0 KV, triton attention, offload MoE + `--moe-cache-auto`
lfu, prefill chunk 8192, expert arena + growable overlap) plus `--text-model-only
--expert-residency mirror --moe-mirror-host-rows -1` (auto pool) and
`--reasoning-parser qwen3 --tool-call-parser qwen3_coder`. Port 30100. Code c10600a.

## Startup attempts

| ratio | linear-state slots | outcome |
|---|---|---|
| 1.00 | 13 (serve-default's Nemotron knob) | OOM at LinearStatePool: 780 MiB asked, 748 MiB free (`s3-serve-q8_0-r1.00-slots13.log`) |
| 1.00 | default | OOM at LinearStatePool: 540 MiB asked, 526 MiB free (`s3-serve-q8_0-r1.00.log`) |
| 0.95 | default | ready in 80 s; 5665 GPU expert slots, mirror pool 6651 rows / 12.27 GiB pinned, 768 reserve, KV ceiling 262144 validated (`s3-serve-q8_0-r0.95.log`) |

The auto plan does subtract the GDN state pool (`engine._resolve_auto_moe_cache_size`,
`state_pool_bytes`); both 1.00 failures miss by only 14–32 MiB of unaccounted overhead —
the per-host edge `scripts/tune-memory-ratio.sh` exists to find. Not an EXL3 issue; left alone.

## Chat (ratio 0.95, temperature 0) — coherent, correct

- `s3-chat1-r0.95.json`: "why is the sky blue + Rayleigh dependence" -> correct paragraph,
  `I ∝ 1/λ^4`, (700/450)^4 ≈ 10; 292 reasoning + 250 content tokens, finish=stop,
  reasoning split out by the qwen3 parser.
- `s3-chat2-r0.95.json`: iterative Fibonacci with docstring -> correct code, O(1) space,
  examples F(10)=55, F(20)=6765; finish=stop.

## Then: first 8K probe at 0.95 killed the server

OOM inside the GDN FLA prefill (`wy_fast.recompute_w_u_fwd`, 64 MiB asked, 28 MiB free) on
the first 8000-token request: at 0.95 with the 262K growable-KV reserve the plan leaves no
transient prefill workspace. Shared scheduler headroom (same family as the post-grow prefill
headroom work in the other workstream), not EXL3. Step 4 numbers were taken at a lower ratio.
