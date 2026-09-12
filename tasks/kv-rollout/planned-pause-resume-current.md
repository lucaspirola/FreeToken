# Current planned pause / resume procedure

Reviewed 2026-09-12. Production identity: 53c3c92; grow8192, logical1M,
memory ratio .91, one lane, prefill4096, spill RAM/disk/total1/50/50 GiB.
This is the current elastic profile, not the historical static fallback.

Before a planned server stop or host reboot:

1. Pause HSR dispatch through its campaign_control CLI. Wait for the existing
   operation to become terminal; never replay an ambiguous claim. Preserve its
   artifacts and reconcile a dead claim as unknown only after verifying death.
2. Verify the owned FreeToken instance and idle state. From the HSR root, use
   guarded `src.prepare_planned_stop` with a NEW output directory and operation
   ID, the actual expected instance and `--execute-owned-local-checkpoint`.
   Inspect its complete durable receipt and generated stop-approval artifact.
   A stale receipt from a previous instance/operation is not reusable.
3. Only after that exact durable acknowledgement, stop the owned
   freetoken-serve.service. Do not issue inference after the sealed barrier.
   A receipt timeout/unknown is not permission to assume persistence; use the
   client's bounded same-operation reconciliation or investigate without replay.

After boot or planned stop, leave campaign dispatch paused. The embedding server
must not compete with FreeToken for the GPU. Coordinate the resource window and
verify ten consecutive two-second post-release samples with MemAvailable >=28GiB
and spill-volume free >=60GiB, followed by the launcher's final admission checks.
This settling window is an operator procedure; the launcher itself checks the
final sample, not the full window.

Validated launcher:

```
/home/lucas/ai/FreeToken/tasks/kv-rollout/resume-1m-grow-8192-mr091-53c3c92.sh --dry-run
/home/lucas/ai/FreeToken/tasks/kv-rollout/resume-1m-grow-8192-mr091-53c3c92.sh
```

It refuses an active owned service or occupied port; checks the approved
production tree and dependency identities; uses the verified ignored uv.lock
with `uv run --locked`; never stops another service; and applies the final
28GiB/60GiB admission floors. A dry-run does not launch or prove available RAM.
Do not lower these floors or silently retry a failed candidate startup.

Verify engine readiness in the new journal, exact configuration, new instance,
checkpoint adoption and idle status before rebinding a NEW campaign run. Never
reuse a frozen design bound to the old instance. Adoption alone does not prove
session restore; the verified two-session test is documented in HSR
analysis/experiments/elastic-handoff/planned-restart-qualification-v1.md.

The two synthetic 8K/32K sessions survived the explicit barrier and coordinated
service restart. This does not qualify full1M workloads, physical host reboot,
power loss, every stored session or long-context reasoning quality. RAM/resident
state without a successful durable barrier remains vulnerable to abrupt loss.
