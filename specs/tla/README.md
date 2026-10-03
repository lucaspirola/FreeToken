# Single-lane admission and reclaim

`Admission.tla` models the pre-fix scheduler protocol at main
`e9ddad4dd4cfa92c1337a4ca60755ab999eee6a3` and the implemented admission fixes.
`Fixed = FALSE` preserves the pre-fix explicit-lease exclusion and the
own-lease length guard. `Fixed = TRUE` permits checkpointing explicit leases and
uses matched-path identity for own-lease reclaim. `PinState` selects a KV-only pin
(0) or an independent KV-plus-GDN pin (1). `ReleaseStatePins = TRUE` encodes
admission-driven release of a state-bearing pin for state pressure, extending the
pre-fix KV-only fallback. The owner approved this pin policy together with explicit
lease checkpointing and matched-path own-lease reclaim. `PredictMatchState = TRUE` also
aligns reclaim pressure with the post-match GDN gate; the check below shows why
changing only the B/D eligibility predicates is insufficient in this abstraction.
The state-bearing-pin PASS also requires `ReleaseStatePins`; the focused pin
counterexample separates that requirement from matched-state pressure prediction.

## Run

From this directory, create local scratch directories and run TLC sequentially:

```bash
mkdir -p results/java-tmp
java -Xmx2g -XX:+UseParallelGC -Djava.io.tmpdir=results/java-tmp -cp /usr/local/lib/tlaplus/tla2tools.jar tlc2.TLC -workers 4 -lncheck final -config MCB.cfg -metadir results/MCB Admission.tla
java -Xmx2g -XX:+UseParallelGC -Djava.io.tmpdir=results/java-tmp -cp /usr/local/lib/tlaplus/tla2tools.jar tlc2.TLC -workers 4 -lncheck final -config MCD.cfg -metadir results/MCD Admission.tla
java -Xmx2g -XX:+UseParallelGC -Djava.io.tmpdir=results/java-tmp -cp /usr/local/lib/tlaplus/tla2tools.jar tlc2.TLC -workers 4 -lncheck final -config MCBQueue.cfg -metadir results/MCBQueue Admission.tla
java -Xmx2g -XX:+UseParallelGC -Djava.io.tmpdir=results/java-tmp -cp /usr/local/lib/tlaplus/tla2tools.jar tlc2.TLC -workers 4 -lncheck final -config MCStatePressure.cfg -metadir results/MCStatePressure Admission.tla
java -Xmx2g -XX:+UseParallelGC -Djava.io.tmpdir=results/java-tmp -cp /usr/local/lib/tlaplus/tla2tools.jar tlc2.TLC -workers 4 -lncheck final -config MCFixedPredicates.cfg -metadir results/MCFixedPredicates Admission.tla
java -Xmx2g -XX:+UseParallelGC -Djava.io.tmpdir=results/java-tmp -cp /usr/local/lib/tlaplus/tla2tools.jar tlc2.TLC -workers 4 -lncheck final -config MCFixedB.cfg -metadir results/MCFixedB Admission.tla
java -Xmx2g -XX:+UseParallelGC -Djava.io.tmpdir=results/java-tmp -cp /usr/local/lib/tlaplus/tla2tools.jar tlc2.TLC -workers 4 -lncheck final -config MCFixedD.cfg -metadir results/MCFixedD Admission.tla
java -Xmx2g -XX:+UseParallelGC -Djava.io.tmpdir=results/java-tmp -cp /usr/local/lib/tlaplus/tla2tools.jar tlc2.TLC -workers 4 -lncheck final -config MCFixedBQueue.cfg -metadir results/MCFixedBQueue Admission.tla
java -Xmx2g -XX:+UseParallelGC -Djava.io.tmpdir=results/java-tmp -cp /usr/local/lib/tlaplus/tla2tools.jar tlc2.TLC -workers 4 -lncheck final -config MCFixedStatePressure.cfg -metadir results/MCFixedStatePressure Admission.tla
java -Xmx2g -XX:+UseParallelGC -Djava.io.tmpdir=results/java-tmp -cp /usr/local/lib/tlaplus/tla2tools.jar tlc2.TLC -workers 4 -lncheck final -config Admission.cfg -metadir results/AdmissionPinCheck Admission.tla
java -Xmx2g -XX:+UseParallelGC -Djava.io.tmpdir=results/java-tmp -cp /usr/local/lib/tlaplus/tla2tools.jar tlc2.TLC -workers 4 -lncheck final -config MC4.cfg -metadir results/MC4Drain Admission.tla
java -Xmx2g -XX:+UseParallelGC -Djava.io.tmpdir=results/java-tmp -cp /usr/local/lib/tlaplus/tla2tools.jar tlc2.TLC -workers 4 -lncheck final -config MCWide.cfg -metadir results/MCWideAuto Admission.tla
java -Xmx2g -XX:+UseParallelGC -Djava.io.tmpdir=results/java-tmp -cp /usr/local/lib/tlaplus/tla2tools.jar tlc2.TLC -workers 4 -lncheck final -config MC3KV.cfg -metadir results/MC3KV Admission.tla
java -Xmx2g -XX:+UseParallelGC -Djava.io.tmpdir=results/java-tmp -cp /usr/local/lib/tlaplus/tla2tools.jar tlc2.TLC -workers 4 -lncheck final -config MCPinState.cfg -metadir results/MCPinStatePinCheck Admission.tla
java -Xmx2g -XX:+UseParallelGC -Djava.io.tmpdir=results/java-tmp -cp /usr/local/lib/tlaplus/tla2tools.jar tlc2.TLC -workers 4 -lncheck final -config MCFixedPinGuard.cfg -metadir results/MCFixedPinGuardPinCheck Admission.tla
java -Xmx2g -XX:+UseParallelGC -Djava.io.tmpdir=results/java-tmp -cp /usr/local/lib/tlaplus/tla2tools.jar tlc2.TLC -workers 4 -lncheck final -config MCFixedPin.cfg -metadir results/MCFixedPinPinCheck Admission.tla
java -Xmx2g -XX:+UseParallelGC -Djava.io.tmpdir=results/java-tmp -cp /usr/local/lib/tlaplus/tla2tools.jar tlc2.TLC -workers 4 -lncheck final -config MC4Pin.cfg -metadir results/MC4PinPinCheck Admission.tla
```

Expected failing configurations exit 13 with `Temporal properties were violated`;
passing configurations exit 0 with `Model checking completed. No error has been found`.
`-lncheck final` checks liveness on the complete graph, avoiding repeated partial-graph checks.
Keep the 2 GiB heap and four-worker cap. No server, GPU, torch, network service, or
client fairness is involved. `results/` is ignored; selected complete TLC logs are
kept under `evidence/`. The temporary-directory option keeps Java's extracted TLA
standard modules inside the worktree, too.

## State and bounds

A FIFO sequence records arrival order. Fresh requests have no matched prefix
and are still bound to a modelled lease id. Admission may bypass a blocked entry,
as `_reclaim_for_blocked_prefill` and prefill admission actually do. A session has
at most one active uid; repeated requests for its session arrive after that uid
finishes. Sibling-session creation is represented by a separate session id.
`owner = 0` is idle, an owner in the queue is busy (and, with a GPU handle, parked),
and `owner = running` is executing. Busy leases never expire. Automatic resident
leases are demand-evicted; explicit idle leases and cold idle leases can expire.

`MCWide` adds a third queued session with automatic leases, exercising back-to-front
choice between two parked victims; `MC4` covers repeat turns and all combinations
of automatic/explicit leases with larger retained prefixes. The unconstrained
three-session mixed probe was stopped during exploration because its growth
projected a longer sweep; it has no complete safety or liveness verdict; its configuration is kept as `MCWideMixed.cfg` and its log
(originally run under the name `MCWide.cfg`) is archived as incomplete.

`Admission`, `MC3KV`, `MC4`, and `MC4Pin` enumerate every assignment of request kinds, initial
lease types, residency (`gpu`, `cold`, `none`), history/summary branch, lease size
1..2, and an optional one-page pin, subject to pool capacity. Session ids alternate
with uid. Requests are finite, each arrives at most once, and arrivals can stop
forever. The fixed runs cover repeat turns as well as competing sessions.

| Configuration | Sessions | Requests | Usable GDN slots | KV pages | Full request demand | Retained lease pages |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Admission / MC3KV | 2 | 3 | 4 | 5 | fresh 1, summary 2, diverge 3, continue 4 | 1..2 |
| MC4 / MC4Pin | 2 | 4 | 4 | 5 | same | 1..2 |
| MCWide | 3 | 3 | 4 | 5 | same | 1 |
| B / fixed B | 1 | 1 | 4 | 4 | 3 | initial 2 |
| D / fixed D | 1 | 1 | 4 | 4 | 4 | initial 1 |
| BQueue / fixed BQueue | 2 | 2 | 4 | 5 | 4, 3 | initial 1, 2 |
| StatePressure / Pin variants | 1 | 1 | 4 | 5 | 4 | initial 1 |

`Admission` and `MC4Pin` use one GDN slot for a live pin; `MC3KV`, `MC4`,
and `MCWide` use KV-only pins. Focused Pin variants use one KV-plus-GDN pin.

Slot capacity excludes permanently reserved padding. Admission reserves three
slots (one live, two ping-pong); a distinct matched snapshot additionally removes
one slot from the evictable pool. A lease snapshot costs one slot. A matched lease
on the request's path already owns that snapshot and its pages: no double charge.
`freePages` and `freeSlots` mean allocatable capacity (free plus evictable), not
only physically empty allocator entries. Whole-request demand includes matched
pages and the remaining prefill/decode reservation. `runPages` holds that full
reservation, so a restore cannot consume pages promised to an executing request.

The model distinguishes history from summary and gives each fresh/divergent
request its own branch. History matched outside the current lease is symbolic,
including closed-session and `~prev` history. `matched` represents its potential
prefix size; tree topology and eviction of individual history nodes are not
modelled. KV locking charges that match before the gate. A finishing request
retains at most two abstract pages, rather than its literal token sequence.

## Properties and code anchors

All Python paths below are under `python/freetoken/`. Source comments in the spec
anchor each protocol abstraction to its implementing function.

| Property / abstraction | Model | Code |
| --- | --- | --- |
| Slot conservation, three-slot seat | `SlotConservation`, `SeatSlots` | `scheduler/prefill.py:PrefillAdder._try_allocate_one`; `scheduler/cache.py:reserve_mamba_slots`, `ensure_mamba_slots`, `mamba_available_size`; `kvcache/hybrid_radix_cache.py:inc_lock` |
| Page conservation, match lock and finishability | `PageConservation`, `Need`, `LockDelta`, `runPages` | `scheduler/prefill.py:PrefillAdder._kv_gate_ok`, `_try_allocate_one`; `scheduler/cache.py:lock_delta`; `scheduler/scheduler.py:_restore_cold_session` |
| Live lock owners and uid transfer | `LiveLocks`, `RequestAccounting` | `scheduler/cache.py:lock`, `unlock`; `scheduler/scheduler.py:_process_one_msg`, `_free_req_resources`, `_close_session` |
| Queue-relative parked protection | `QueueOrder`, `Eligible`, `releasedAhead` | `scheduler/scheduler.py:_reclaim_soft_sessions_for_pending` |
| One executing uid | `SingleLane` | single-lane configuration; `scheduler/prefill.py:PrefillAdder._try_allocate_one` |
| Checkpoint before protected unlock | `Reclaim`, `cold` | `scheduler/scheduler.py:_release_soft_session_handle`, `_spill_soft_session`, `_reclaim_soft_sessions_for_state_slot` |
| Restore opportunities and preserved branches | `restoreDue`, `Restore`, `Path`, `Finish` | `scheduler/scheduler.py:_process_one_msg`, `_reclaim_for_blocked_prefill`, `_restore_cold_session`, `_spill_replaced_conversation`, `_free_req_resources` |
| Expiry without busy-request rescue | `Expire` | `scheduler/scheduler.py:_expire_sessions`, `_close_session`; `_process_one_msg` sets `expires_at = None` |
| Pin ownership / fallback | `PinState`, `ReleasePin` | `scheduler/cache.py:pin_prefix`, `_release_pin`, `release_pins_for_admission`; `scheduler/scheduler.py:_reclaim_soft_sessions_for_pending` |
| Cold-tier latency abstraction | `cold` | `scheduler/scheduler.py:_prefetch_queued_session` (RAM/disk promotion does not free GPU resources) |
| Admission progress | `EventualAdmission`, `DrainFittingQueue` | `scheduler/scheduler.py:_reclaim_for_blocked_prefill`; `scheduler/prefill.py:PrefillAdder._try_allocate_one` |
| Useful deadlock freedom | `ProtocolDeadlockFree` | same admission/reclaim/restore/finish/expiry service actions |

`EventualAdmission` says every queued uid with whole demand no larger than the
page pool and at least four usable GDN slots eventually enters `admitted`.
This expresses the intended post-checkpoint progress contract, including explicit
leases in the fixed policy. It does not promise admission of oversized workloads.
`MC4.cfg`, `MC4Pin.cfg`, and `MCWide.cfg` check the equivalent `DrainFittingQueue` formula to avoid TLC making
one copy per request of the temporal graph. `Fits` is constant per uid, there are finitely
many arrivals, a uid leaves the queue only by admission, and `admitted` never
shrinks. Thus eventual drain implies each fitting queued uid is admitted; conversely,
per-uid progress empties the finite fitting set after its last arrival. All safety
invariants and fairness conditions are unchanged. The focused configurations and `Admission.cfg` check the
explicit per-uid formula directly. `MC4PerRequest.cfg` retains that original encoding;
its run was stopped after exceeding ten minutes with low-heap liveness warnings,
after completing the 4,809,860-state safety graph. This interrupted run is not a PASS.
The archived interrupted log originally used the name `MC4.cfg`; its configuration
is preserved as `MC4PerRequest.cfg`.

Four slots are needed when a cached recurrent source must survive beside the three
new request slots; this is not a proof for three-slot pools or other architectures.

Scheduler admission has strong fairness per uid because restores may temporarily
remove its seat. Finish, reclaim per caller/victim, pin release, and idle expiry
have weak fairness. There is no fairness on arrivals, cancellation, deletion, or
restore success. Checkpoint creation succeeds atomically; bounded/failed storage,
permanent GPU shrink, and I/O failure are outside the progress premise. Restore
is optional, only on receipt or following that caller's successful reclaim, as in
the source. An unrestricted background restore action produced a spurious loop:
repeatedly restore/reclaim a later request before the head could reclaim its own
unrelated lease. Such a restore loop is not the source protocol.

`Poll` allows legitimate idle termination. Consequently TLC's built-in deadlock
check alone would be vacuous. Every fixed configuration also checks the stronger
`ProtocolDeadlockFree`: with outstanding work, an actual scheduler service action
must be enabled, excluding polling and client arrivals. Temporal checking excludes
fair infinite polling and other fair starvation cycles.

## Current-main counterexamples

These traces start at pre-existing resident leases, a state reached by the pinned
Python replays. Pool units preserve the capacity/branch relationships, not their
literal token counts. No safety invariant failed in the counterexample runs.

* **B, `MCB`:** explicit history lease A has size 2 in a four-page pool.
  `Arrive(1)` queues A's divergent demand-3 turn and makes the lease busy.
  Two pages remain, so admission fails. Explicit exclusion prevents checkpointing;
  busy expiry is disabled. The trace then stutters forever.
  Maps to `test_open_bug_explicit_session_turn_that_does_not_fit_beside_its_own_lease`.
* **D, `MCD`:** automatic summary lease A has size 1; its continuing request matches
  size 2 of history elsewhere and has total demand 4. `Arrive(1)` leaves three
  allocatable pages; locking history leaves one, but the request needs two more.
  `own_len > cached_len` is `1 > 2`, false. Busy automatic expiry cannot rescue it;
  the trace stutters. Maps to
  `test_open_bug_own_summary_lease_starves_the_turn_that_continues_the_history`
  (17-token summary, 28-token history match, one page short).
* **B with queue ordering, `MCBQueue`:** A's automatic history lease has size 1,
  B's explicit history lease size 2, with five total pages. `Arrive(1)` queues A's
  continuation, then `Arrive(2)` queues B's divergence. Both are one page short.
  A cannot reclaim explicit B, B cannot reclaim its own explicit lease, and B
  must not reclaim A's parked lease ahead. Both remain queued, then stutter.
  Maps to `test_open_bug_explicit_lease_of_a_queued_diverged_turn_starves_the_head`.
* **Matched-state pressure gap, `MCStatePressure`:** use the D seed with five pages.
  A's off-path summary holds one slot; matched history needs one more source slot.
  KV fits exactly after locking. Three GDN slots are allocatable before matching,
  but only two after matching. Admission needs three new slots. Reclaim's
  pre-match `mamba_available_size < 3` test is false, and the state-slot hook only
  reaches idle leases; A is busy. `Arrive(1)` is followed by infinite stuttering.
  `MCFixedPredicates` reproduces this with the B/D eligibility changes alone.
  This is a model-derived finding, not an existing xfail or a hardware replay.
  Direct evidence: `cache.py:lock` calls `hybrid_radix_cache.py:inc_lock`, which
  subtracts an unlocked matching snapshot from `mamba_evictable_size`;
  `prefill.py:_try_allocate_one` then reserves three slots; reclaim predicts KV
  lock delta but tests the pre-match GDN availability against three.

* **State-bearing pin, `MCPinState`:** A has a one-page history lease on the
  request's matched path; an independent off-path pin owns one page and one GDN
  snapshot. `Arrive(1)` queues a total-demand-4 continuation in a five-page pool.
  The remaining three KV pages fit its three-page tail, but only two GDN slots
  are allocatable. Own-path lease reclaim is correctly forbidden, while pin
  fallback is incorrectly inactive because KV is not short. Infinite stuttering
  follows. `MCFixedPinGuard` still fails with B/D and matched-state prediction
  fixed; `MCFixedPin` passes when state-bearing pins can be released for GDN
  pressure. This is also model-derived, without a Python/hardware replay or
  existing xfail. `pin_prefix` selects snapshot-bearing path nodes and `inc_lock`
  protects their GDN state; both the scheduler call site and
  `release_pins_for_admission`'s loop are guarded only by KV shortage today.

A and C are intentionally below this model's evidence level. A's restore-cut
`mamba_ref_count`/insert ordering would require radix nodes and snapshot identity;
C's stale `token_ids` would require token/checkpoint content and spill validation.
Here restore creates a valid owned snapshot and checkpoint content is coherent.
The fixed proof therefore does not resolve either bug, even though C can produce
an observable parked-restore deadlock when those assumptions fail.

## Results and implementation implications

Results were obtained with TLC 2.19 (2024-08-08), OpenJDK 25.0.4.1, Linux amd64,
four workers, and the exact commands above. State counts include all initial
assignments, all scheduler interleavings, and arbitrary finite client arrival
prefixes. Complete temporal checking, safety checking, and useful deadlock
checking are required for a PASS; a stopped sweep is not a PASS.

| Configuration | Verdict | Generated states | Distinct states | Runtime |
| --- | --- | ---: | ---: | --- |
| MCB | B liveness counterexample | 14 | 7 | <1s |
| MCD | D liveness counterexample | 4 | 2 | <1s |
| MCBQueue | B queued-head counterexample | 38 | 17 | <1s |
| MCStatePressure | matched-state counterexample | 4 | 2 | <1s |
| MCFixedPredicates | B/D changes alone still fail | 4 | 2 | <1s |
| MCPinState | state-bearing-pin counterexample | 4 | 2 | <1s |
| MCFixedPinGuard | B/D plus match prediction still fail | 4 | 2 | <1s |
| MCFixedB | PASS | 19 | 9 | <1s |
| MCFixedD | PASS | 13 | 5 | <1s |
| MCFixedBQueue | PASS | 110 | 39 | <1s |
| MCFixedStatePressure | PASS | 11 | 5 | <1s |
| MCFixedPin | PASS | 10 | 5 | <1s |
| Admission | PASS, per-uid progress, GDN pin | 3,237,004 | 947,937 | 1m48s |
| MC3KV | PASS, per-uid progress, KV-only pin | 3,681,463 | 1,120,824 | 2m03s |
| MC4Pin | PASS, drain progress, GDN pin | 13,599,787 | 3,993,047 | 11m15s |
| MC4 | PASS, drain progress, KV-only pin | 15,782,935 | 4,809,860 | 14m05s |
| MCWide | PASS, drain progress, three sessions | 4,912,436 | 1,096,772 | 2m04s |

Complete raw outputs are in [evidence/](evidence/), named after each configuration.
`MC4Pin` is the largest state-bearing-pin bound completed near the requested
ten-minute budget (11m15s). The KV-only four-request sweep overshot that budget;
its complete result is reported rather than described as a ten-minute run.
The interrupted `MC4PerRequest` and `MCWideMixed` probes have no temporal verdict.

The earlier `MC3KV` check used the name `Admission.cfg` and metadata directory
`results/Admission`; the configuration was retained as `MC3KV.cfg` when the default
was extended to GDN-bearing pins. Its archived output is `evidence/MC3KV.log`.
The MC3KV command above reproduces that transition system under its final name.
The interrupted per-request and mixed probes used respectively `-config MC4.cfg
-metadir results/MC4` and `-config MCWide.cfg -metadir results/MCWide`, with the same
Java prefix and `Admission.tla` suffix as above, before those names were reused.
Their exact configurations are preserved as `MC4PerRequest.cfg` and
`MCWideMixed.cfg`; no other run in this table is interrupted.

Explicit protection must mean a preserved checkpoint, not an irrevocable GPU
lock: change candidate selection, own-lease selection, and the release helper's
explicit exclusion together. Require a valid checkpoint before unlocking an
explicit lease; do not turn checkpoint failure into silent discard. Preserve
busy uid ownership while moving its handle to cold storage. Prefetch's explicit
exclusion may also need updating to obtain the intended latency, though prefetch
is not required for liveness here.

Compare the own handle against the actual matched path, not lengths. A short
summary and long historical match can be different branches. Preserve idle-first,
back-to-front parked reclaim, and never reclaim ahead; these are source rules,
not new owner policy. Predict GDN headroom after locking the matched snapshot as
well as KV lock delta. The fixed PASS includes this alignment, and must not be
read as proof that B/D eligibility edits alone suffice. Python regressions now confirm
both the state-pressure and pin findings.

The finite sweep is not an unbounded proof. It omits radix sharing across leases,
multi-snapshot paths, byte fidelity, host/disk capacity and eviction, asynchronous
copies/shrink, table/SWA limits, chunked execution (the whole footprint is reserved
atomically), client cancellation/deletion, and infinite arrival streams. Pin state
is an independent cache-owned lock, either KV-only or one KV-plus-GDN snapshot;
shared paths, multiple pinned snapshots, and pin creation during a workload are omitted.
Expiry abstracts time as fair eventual service without modelling a numeric TTL.
