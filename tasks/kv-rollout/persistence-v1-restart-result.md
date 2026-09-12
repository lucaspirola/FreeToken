# Planned persistence restart — 53c3c92

**Result:** restarted once, engine ready, and idle for root’s independent two-session restore verification. Operations did not send an inference request or an administrative checkpoint request.

## Checkpoint authorization and pre-stop verification

The HSR receipt `persistence-checkpoint-v1/attempt-1.json` recorded a complete, durable checkpoint operation on old instance `a963bbba-a5da-4410-956c-439992077602`: durable count `142`, aggregate digest `b0cd2f52e8aebbeb97de053032437b5e14f994e88111115d426677a9414cef35`, and drain complete `True`. The paired `stop-approval.json` approved stopping only `freetoken-serve.service` for that exact instance. Its aggregate receipt does not attest per-session resident membership; restore verification remains separate.

Operations independently confirmed that exact instance was idle before stopping: zero active requests and four completed seed requests. It stopped and released the port in one second. No claim is made about the old resident cache; the persistence result rests on the durable checkpoint receipt and post-restart adoption record.

## Admission and restart

The same identity-pinned launcher `resume-1m-grow-8192-mr091-53c3c92.sh` was invoked exactly once. It uses commit `53c3c92`, Python tree `f706c2450376b1c4b69d1cb71d3ecfcc49e4a853`, and `uv run --locked` with the verified ignored-lock hash.

Ten consecutive two-second samples met both admission floors: MemAvailable 29580988–29662472 KiB (floor 29,360,128) and spill filesystem free 137619240–137621896 KiB (floor 62,914,560). The launcher then performed its final checks before process creation.

## New engine and adoption

- New instance: `8d2ea844-51e4-41bc-8943-ec033b5776aa`
- Engine journal readiness marker: present.
- Spill adoption: `Session spill root ... adopted 142 checkpoint(s), removed 0 stale entries`.
- API: `0` active and `0` completed requests.
- Ready geometry: logical KV 1048576; MoE cache 2016; usable Mamba slots 5; policy lfu.
- Profile preserved: one lane, 1,048,576 logical context/KV, grow step 8192, .91 memory ratio, prefill 4096, six requested linear slots, host reserve 8 GiB, spill 1/50/50 GiB, CUDA telemetry.

Current cgroup/host/GPU snapshot:

```
timestamp=2026-09-12T23:35:40+04:00
pid=714611
control_group=/user.slice/user-1000.slice/user@1000.service/app.slice/freetoken-serve.service
memory.current=20158509056

memory.peak=23536209920

memory.swap.current=0

MemAvailable:=10300684 KiB
SwapFree:=2106380 KiB
13946, 2032, 16303
```

This server is held idle for root’s two-session restore verifier. Adoption count establishes matching disk records were found; it does not itself establish successful restoration of every session.

Evidence: `persistence-v1-pre-stop-stats.json`, `persistence-v1-pre-stop-journal.txt`, `persistence-v1-restart-settling-samples.tsv`, `persistence-v1-restart-launch.txt`, `persistence-v1-restart-journal.txt`, `persistence-v1-restart-ready-marker.txt`, `persistence-v1-restart-spill-adoption.txt`, `persistence-v1-restart-ready-stats.json`, `persistence-v1-restart-cache-status.json`, and `persistence-v1-restart-resources.txt`.
