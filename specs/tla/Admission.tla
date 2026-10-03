----------------------------- MODULE Admission -----------------------------
EXTENDS Naturals, Integers, Sequences, FiniteSets, TLC
CONSTANTS Fixed, PredictMatchState, PinState, ReleaseStatePins, Scenario,
          NSessions, NRequests, SlotCapacity, PageCapacity,
          MaxDemand, MaxLease
Sessions == 1..NSessions
Requests == 1..NRequests
Kinds == {"continue", "diverge", "summary", "fresh"}
Sid(r) == 1 + ((r - 1) % NSessions)
\* scheduler.py:_prefetch_queued_session only changes RAM/disk latency; both are cold here.
VARIABLES queue, running, arrived, admitted, done, kind, demand, matched,
          automatic, handle, branch, size, owner, cold, restoreDue,
          pins, locks, freeSlots, freePages, runPages, releasedAhead
vars == <<queue, running, arrived, admitted, done, kind, demand, matched,
          automatic, handle, branch, size, owner, cold, restoreDue,
          pins, locks, freeSlots, freePages, runPages, releasedAhead>>
RECURSIVE Sum(_, _)
Sum(f, domain) == IF domain = {} THEN 0
                 ELSE LET s == CHOOSE x \in domain : TRUE
                      IN f[s] + Sum(f, domain \ {s})
\* cache.py:ensure_mamba_slots/reserve_mamba_slots fold evictable nodes into free capacity.
Resident == {s \in Sessions : handle[s] = "gpu"}
LeaseLock(s) == <<"lease", s>>
ReqLock(r) == <<"request", r>>
\* cache.py:pin_prefix/_release_pin can protect KV alone or a distinct GDN snapshot too.
PinLock == <<"pin", 0>>
Range(q) == {q[i] : i \in 1..Len(q)}
Position(r) == CHOOSE i \in 1..Len(queue) : queue[i] = r
Remove(q, r) == SelectSeq(q, LAMBDA x : x # r)

\* scheduler.py:_free_req_resources/_spill_replaced_conversation preserve branch identity.
Path(r) == CASE kind[r] = "continue" -> <<"history", 0>>
                [] kind[r] = "summary" -> <<"summary", 0>>
                [] OTHER -> <<kind[r], r>>
OnPath(r) == handle[Sid(r)] = "gpu" /\ Path(r) = branch[Sid(r)]
CacheLen(r) == IF OnPath(r) THEN size[Sid(r)] ELSE matched[r]
LockDelta(r) == IF OnPath(r) THEN 0 ELSE matched[r]
Need(r) == demand[r] - CacheLen(r)
Cost(r) == Need(r) + LockDelta(r)
\* prefill.py:_try_allocate_one/_kv_gate_ok lock the match before the second KV gate.
KVShort(r) == Need(r) > freePages - LockDelta(r)
\* cache.py:lock and kvcache/hybrid_radix_cache.py:inc_lock protect the matched snapshot.
MatchSlots(r) == IF ~OnPath(r) /\ matched[r] > 0 THEN 1 ELSE 0
SeatSlots(r) == 3 + MatchSlots(r)
Pressure(r) == KVShort(r) \/ freeSlots < SeatSlots(r)
\* scheduler.py:_reclaim_soft_sessions_for_pending predicts KV and GDN match-lock deltas in the fixed implementation.
ReclaimPressure(r) == KVShort(r) \/
 freeSlots < (IF Fixed /\ PredictMatchState THEN SeatSlots(r) ELSE 3)
Spillable(s) == automatic[s] \/ Fixed

\* scheduler.py:_process_one_msg creates busy leases with expires_at None; clients are finite.
Init ==
 /\ queue = <<>> /\ running = 0 /\ arrived = {} /\ admitted = {} /\ done = {}
 /\ kind \in [Requests -> Kinds]
 /\ demand = [r \in Requests |->
       IF Scenario = "B" THEN 3 ELSE IF Scenario = "D" THEN 4
       ELSE CASE kind[r] = "fresh" -> 1
              [] kind[r] = "summary" -> 2
              [] kind[r] = "diverge" -> 3
              [] OTHER -> MaxDemand]
 /\ matched = [r \in Requests |-> IF kind[r] = "continue" THEN MaxLease ELSE 0]
 /\ automatic \in [Sessions -> BOOLEAN]
 /\ handle \in [Sessions -> {"gpu", "cold", "none"}]
 /\ branch \in [Sessions -> {<<"history", 0>>, <<"summary", 0>>}]
 /\ size \in [Sessions -> 1..MaxLease]
 /\ owner = [s \in Sessions |-> 0]
 /\ cold = [s \in Sessions |-> handle[s] = "cold"]
 /\ restoreDue = {}
 /\ pins \in 0..1 /\ runPages = 0
 /\ freeSlots = SlotCapacity - Cardinality(Resident) - PinState * pins
 /\ freePages = PageCapacity - Sum(size, Resident) - pins
 /\ freeSlots >= 0 /\ freePages >= 0
 /\ locks = {LeaseLock(s) : s \in Resident}
              \cup (IF pins = 1 THEN {PinLock} ELSE {})
 /\ releasedAhead = FALSE
 /\ \A r \in Requests : matched[r] < demand[r]
 /\ IF Scenario = "B" THEN
       /\ automatic[1] = FALSE /\ handle[1] = "gpu" /\ size[1] = 2
       /\ branch[1] = <<"history", 0>> /\ kind[1] = "diverge"
       /\ demand[1] = 3 /\ matched[1] = 0 /\ pins = 0
     ELSE IF Scenario = "D" THEN
       /\ automatic[1] = TRUE /\ handle[1] = "gpu" /\ size[1] = 1
       /\ branch[1] = <<"summary", 0>> /\ kind[1] = "continue"
       /\ demand[1] = 4 /\ matched[1] = 2 /\ pins = 0
     ELSE IF Scenario = "BQueue" THEN
       /\ automatic = <<TRUE, FALSE>> /\ handle = <<"gpu", "gpu">>
       /\ size = <<1, 2>> /\ branch = <<<<"history", 0>>, <<"history", 0>>>>
       /\ kind = <<"continue", "diverge">> /\ pins = 0
     ELSE IF Scenario = "Pin" THEN
       /\ automatic[1] = TRUE /\ handle[1] = "gpu" /\ size[1] = 1
       /\ branch[1] = <<"history", 0>> /\ kind[1] = "continue" /\ pins = 1
     ELSE IF Scenario = "Wide" THEN automatic = [s \in Sessions |-> TRUE]
     ELSE TRUE

Arrive(r) ==
 /\ r \notin arrived /\ owner[Sid(r)] = 0
 /\ \A p \in Requests : p < r => p \in arrived
 /\ queue' = Append(queue, r) /\ arrived' = arrived \cup {r}
 /\ owner' = [owner EXCEPT ![Sid(r)] = r]
 /\ restoreDue' = restoreDue \cup {r}
 /\ UNCHANGED <<running, admitted, done, kind, demand, matched, automatic,
                handle, branch, size, cold, pins, locks,
                freeSlots, freePages, runPages, releasedAhead>>

\* scheduler.py:_reclaim_soft_sessions_for_pending permits only idle or behind owners.
Eligible(r, s) ==
 /\ handle[s] = "gpu" /\ Spillable(s)
 /\ IF s = Sid(r) THEN
       /\ owner[s] \in {0, r}
       /\ IF Fixed THEN ~OnPath(r) ELSE size[s] > CacheLen(r)
     ELSE owner[s] = 0 \/
          (owner[s] \in Range(queue) /\ Position(owner[s]) > Position(r))

\* scheduler.py:_release_soft_session_handle requires a successful checkpoint before unlock.
Reclaim(r, s) ==
 /\ r \in Range(queue) /\ Eligible(r, s)
 \* scheduler.py:_reclaim_soft_sessions_for_state_slot reaches idle leases after match locking.
 /\ (ReclaimPressure(r) \/ (Pressure(r) /\ owner[s] = 0))
 /\ IF s = Sid(r) THEN
       ~\E t \in Sessions \ {s} : Eligible(r, t)
     ELSE IF owner[s] # 0 THEN
       /\ ~\E t \in Sessions \ {Sid(r)} : Eligible(r, t) /\ owner[t] = 0
       /\ ~\E t \in Sessions \ {Sid(r)} : Eligible(r, t) /\
                 owner[t] \in Range(queue) /\ Position(owner[t]) > Position(owner[s])
     ELSE TRUE
 /\ handle' = [handle EXCEPT ![s] = "cold"]
 /\ cold' = [cold EXCEPT ![s] = TRUE]
 /\ restoreDue' = restoreDue \cup {r}
 /\ locks' = locks \ {LeaseLock(s)}
 /\ freeSlots' = freeSlots + 1 /\ freePages' = freePages + size[s]
 /\ releasedAhead' = releasedAhead \/
       (s # Sid(r) /\ owner[s] \in Range(queue) /\ Position(owner[s]) < Position(r))
 /\ UNCHANGED <<queue, running, arrived, admitted, done, kind, demand, matched,
                automatic, branch, size, owner, pins, runPages>>

\* cache.py:release_pins_for_admission frees pins for KV and GDN admission pressure in the fixed implementation.
\* ReleaseStatePins enables the same fallback for state-bearing pins under GDN pressure.
ReleasePin(r) ==
 /\ r \in Range(queue) /\ pins = 1
 /\ (KVShort(r) \/ (Fixed /\ ReleaseStatePins /\ PinState = 1 /\ freeSlots < SeatSlots(r)))
 /\ ~\E s \in Sessions : Eligible(r, s)
 /\ pins' = 0 /\ locks' = locks \ {PinLock} /\ freePages' = freePages + 1
 /\ freeSlots' = freeSlots + PinState
 /\ UNCHANGED <<queue, running, arrived, admitted, done, kind, demand, matched,
                automatic, handle, branch, size, owner, cold, restoreDue,
                runPages, releasedAhead>>

\* scheduler.py:_process_one_msg/_reclaim_for_blocked_prefill restore only on receipt/retry.
\* scheduler.py:_restore_cold_session charges restore against outstanding finishability.
Restore(r) ==
 /\ r \in Range(queue) /\ r \in restoreDue
 /\ handle[Sid(r)] # "gpu" /\ cold[Sid(r)] /\ Path(r) = branch[Sid(r)]
 /\ freeSlots >= 1 /\ freePages >= size[Sid(r)]
 /\ handle' = [handle EXCEPT ![Sid(r)] = "gpu"]
 /\ restoreDue' = restoreDue \ {r}
 /\ locks' = locks \cup {LeaseLock(Sid(r))}
 /\ freeSlots' = freeSlots - 1 /\ freePages' = freePages - size[Sid(r)]
 /\ UNCHANGED <<queue, running, arrived, admitted, done, kind, demand, matched,
                automatic, branch, size, owner, cold, pins, runPages, releasedAhead>>

\* prefill.py:_try_allocate_one reserves one live and two ping-pong GDN slots.
Admit(r) ==
 /\ running = 0 /\ r \in Range(queue) /\ ~Pressure(r)
 /\ queue' = Remove(queue, r) /\ running' = r /\ admitted' = admitted \cup {r}
 /\ locks' = locks \cup {ReqLock(r)}
 /\ freeSlots' = freeSlots - SeatSlots(r) /\ freePages' = freePages - Cost(r)
 /\ runPages' = Cost(r)
 /\ UNCHANGED <<arrived, done, kind, demand, matched, automatic, handle,
                branch, size, owner, cold, restoreDue, pins, releasedAhead>>

\* scheduler.py:_free_req_resources/_spill_replaced_conversation retain the new branch.
\* The old family checkpoint is abstracted by matched history, outside the current lease.
Finish(r) ==
 /\ running = r /\ r # 0
 /\ LET s == Sid(r)
        oldPages == IF handle[s] = "gpu" THEN size[s] ELSE 0
        oldSlot == IF handle[s] = "gpu" THEN 1 ELSE 0
        kept == IF demand[r] < MaxLease THEN demand[r] ELSE MaxLease
    IN /\ handle' = [handle EXCEPT ![s] = "gpu"]
       /\ size' = [size EXCEPT ![s] = kept]
       /\ branch' = [branch EXCEPT ![s] = Path(r)]
       /\ owner' = [owner EXCEPT ![s] = 0]
       /\ cold' = [cold EXCEPT ![s] = FALSE]
       /\ freeSlots' = freeSlots + SeatSlots(r) + oldSlot - 1
       /\ freePages' = freePages + runPages + oldPages - kept
       /\ locks' = (locks \ {ReqLock(r)}) \cup {LeaseLock(s)}
 /\ running' = 0 /\ runPages' = 0 /\ done' = done \cup {r}
 /\ UNCHANGED <<queue, arrived, admitted, kind, demand, matched, automatic,
                restoreDue, pins, releasedAhead>>

\* scheduler.py:_expire_sessions/_close_session never expire busy or resident auto leases.
Expire(s) ==
 /\ owner[s] = 0 /\ handle[s] # "none"
 /\ (~automatic[s] \/ handle[s] # "gpu")
 /\ freeSlots' = freeSlots + (IF handle[s] = "gpu" THEN 1 ELSE 0)
 /\ freePages' = freePages + (IF handle[s] = "gpu" THEN size[s] ELSE 0)
 /\ handle' = [handle EXCEPT ![s] = "none"]
 /\ locks' = locks \ {LeaseLock(s)}
 /\ UNCHANGED <<queue, running, arrived, admitted, done, kind, demand, matched,
                automatic, branch, size, owner, cold, restoreDue, pins,
                runPages, releasedAhead>>

\* scheduler.py:_reclaim_for_blocked_prefill retries admission; quiescence is normal.
Poll == UNCHANGED vars
Service == (\E r \in Requests : Restore(r) \/ Admit(r) \/ Finish(r)
                                \/ ReleasePin(r) \/ (\E s \in Sessions : Reclaim(r, s)))
           \/ (\E s \in Sessions : Expire(s))
Next == (\E r \in Requests : Arrive(r)) \/ Service \/ Poll

Spec == Init /\ [][Next]_vars
 /\ (\A r \in Requests : SF_vars(Admit(r)) /\ WF_vars(Finish(r))
                         /\ WF_vars(ReleasePin(r)))
 /\ (\A r \in Requests, s \in Sessions : WF_vars(Reclaim(r, s)))
 /\ (\A s \in Sessions : WF_vars(Expire(s)))

SlotConservation == freeSlots >= 0 /\ freeSlots <= SlotCapacity
 /\ freeSlots + Cardinality(Resident) + PinState * pins
    + (IF running = 0 THEN 0 ELSE SeatSlots(running)) = SlotCapacity
PageConservation == freePages >= 0 /\ freePages <= PageCapacity
 /\ freePages + Sum(size, Resident) + pins + runPages = PageCapacity
LiveLocks == locks = {LeaseLock(s) : s \in Resident}
 \cup (IF running = 0 THEN {} ELSE {ReqLock(running)})
 \cup (IF pins = 0 THEN {} ELSE {PinLock})
 /\ \A s \in Sessions : owner[s] # 0 => owner[s] \in Range(queue) \cup {running}
SingleLane == running \in Requests \cup {0} /\ running \notin Range(queue)
QueueOrder == ~releasedAhead
\* prefill.py:_kv_gate_ok; fitting includes the matched path plus its live GDN snapshot.
Fits(r) == demand[r] <= PageCapacity /\ SlotCapacity >= 4
EventualAdmission == \A r \in Requests : (r \in Range(queue) /\ Fits(r)) ~> (r \in admitted)
\* scheduler.py:_process_one_msg/_free_req_resources bound arrivals and remove uids by admission.
\* With finite arrivals, draining fitting uids is equivalent to the per-uid progress contract.
FittingQueue == {r \in Range(queue) : Fits(r)}
DrainFittingQueue == (FittingQueue # {}) ~> (FittingQueue = {})
\* Poll permits idle termination, so test useful progress separately from TLC's stutter check.
ProtocolDeadlockFree == (running = 0 /\ queue = <<>>) \/ ENABLED Service
\* scheduler.py:_process_one_msg/_free_req_resources transfer, rather than duplicate, uids.
RequestAccounting ==
 /\ Cardinality(Range(queue)) = Len(queue)
 /\ done \subseteq admitted /\ admitted \subseteq arrived
 /\ arrived = Range(queue) \cup done \cup (IF running = 0 THEN {} ELSE {running})
 /\ Range(queue) \cap admitted = {}
 /\ \A r \in (Range(queue) \cup {running}) \ {0} : owner[Sid(r)] = r
=============================================================================
