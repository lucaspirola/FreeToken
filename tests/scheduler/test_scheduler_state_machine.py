"""Stateful property test of the scheduler's session / restore / admission machinery.

A Hypothesis ``RuleBasedStateMachine`` drives the REAL ``Scheduler.normal_loop`` (message
handling, admission through ``PrefillManager``/``PrefillAdder``, the lease reclaim hooks,
finish/abort/close/expiry, cold restore) over the REAL ``CacheManager`` (hybrid radix tree,
``LinearStatePool`` and ``MHAKVCache`` on CPU) and a REAL ``SessionSpillStore`` (RAM and
disk tiers in a temp dir). Pools are tiny (48 KV pages, 13 state slots, one lane), so
pressure is constant.

Simulated, because it needs the GPU model: ``_prepare_batch`` (only its pool effects --
``allocate_paged`` and the GDN chunk-boundary tracking of ``attention/linear.py``) and
``_forward`` (``complete_one`` plus a scripted next token). Time is a fake clock, disk
writes and prefetches are waited for after every step, and host MemAvailable is a rule,
so an example replays deterministically.

Checked after every step: GDN slot conservation, radix lock ownership, KV page
conservation, spill-store accounting, lease token/prefix consistency and busy-lease
liveness; after the loop runs, bounded admission progress. The motivating bug is 2fae797a
(a cold restore overwrote a lease's locked handle; one GDN slot leaked per compaction turn
until admission refused everything).

The default run is derandomized. ``FREETOKEN_SM_PROFILE=heavy`` searches longer with a
random seed; ``FREETOKEN_SM_STRICT=1`` fails on the open bugs in ``KNOWN_BUGS`` too.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time as _real_time
from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, assume, event, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402
from hypothesis.control import currently_in_test_context  # noqa: E402
from hypothesis.stateful import (  # noqa: E402
    RuleBasedStateMachine,
    invariant,
    precondition,
    rule,
)

from freetoken.core import SamplingParams  # noqa: E402
from freetoken.distributed.info import DistributedInfo  # noqa: E402
from freetoken.kvcache.linear_state_pool import LinearStatePool  # noqa: E402
from freetoken.kvcache.mha_pool import MHAKVCache  # noqa: E402
from freetoken.kvcache.quant import Q6_0, Q8_0  # noqa: E402
from freetoken.message import (  # noqa: E402
    AbortBackendMsg,
    CloseSessionBackendMsg,
    UnpinPrefixesBackendMsg,
    UserMsg,
)
from freetoken.message.tokenizer import DetokenizeMsg, ErrorReplyMsg  # noqa: E402
from freetoken.models.config import LinearGatedDeltaGroupConfig  # noqa: E402
from freetoken.scheduler import session_spill  # noqa: E402
from freetoken.scheduler.cache import CacheManager  # noqa: E402
from freetoken.scheduler.decode import DecodeManager  # noqa: E402
from freetoken.scheduler.prefill import ChunkedReq, PrefillManager  # noqa: E402
from freetoken.scheduler.scheduler import ForwardInput, Scheduler  # noqa: E402
from freetoken.scheduler.session_spill import SessionSpillStore  # noqa: E402
from freetoken.scheduler.table import TableManager  # noqa: E402

PAGES = 48  # KV pool, page_size 1
STATE_SLOTS = 13  # --linear-state-slots 13: padding + 12, as served
MAX_RUNNING = 1  # single lane, as served
CHUNK = 4  # GDN snapshot boundary (track_chunk_size)
PREFILL_BUDGET = 16  # forces chunked prefill of every longer prompt
MAX_SEQ = PAGES
MAX_PROMPT = 36
PIN_MIN = 6  # auto-pin the shared system prompt once two sessions match through it
FAMILIES = ("A", "B", "C")
SYSTEM = list(range(10, 16))
SUMMARY = [20, 21]
EOS = 2
PROGRESS_PASSES = 4
MEM_PLENTY = 1 << 40

# Open bugs this machine found (see the xfail tests at the bottom). An example that reaches
# one's exact signature is discarded instead of failed, so the search keeps looking for new
# ones; FREETOKEN_SM_STRICT=1 fails on them instead. Drop an entry with its fix.
EXPLICIT_LEASE_STARVES_OWN_TURN = "explicit lease starves its own diverged turn"
RESTORE_KEEPS_STALE_TOKEN_IDS = (
    "a restored lease keeps the token_ids of the turn before the restore")
SHORT_OWN_LEASE_NEVER_RELEASED = (
    "a diverged own lease shorter than the request's match elsewhere is never released")
KNOWN_BUGS = {
    SHORT_OWN_LEASE_NEVER_RELEASED,
    EXPLICIT_LEASE_STARVES_OWN_TURN,
    RESTORE_KEEPS_STALE_TOKEN_IDS,
}


class _FakeTime:
    """``time`` for the scheduler, cache and spill modules: a clock the rules advance, and
    strictly increasing ``monotonic_ns``/``time`` so LRU ties never depend on the host."""

    def __init__(self) -> None:
        self.now = 1_000.0
        self.ticks = 0

    def monotonic(self) -> float:
        return self.now

    def monotonic_ns(self) -> int:
        self.ticks += 1
        return self.ticks

    def time(self) -> float:
        self.ticks += 1
        return 1.7e9 + self.now + self.ticks * 1e-6

    def sleep(self, _seconds: float) -> None:
        pass

    def __getattr__(self, name):
        return getattr(_real_time, name)


class _Done:
    def synchronize(self) -> None:
        pass


def _setup_context() -> None:
    from freetoken.core import Context, get_global_ctx, set_global_ctx

    try:
        get_global_ctx()
    except AssertionError:
        set_global_ctx(Context(page_size=1))


def _pools():
    group = LinearGatedDeltaGroupConfig(
        name="linear",
        layer_ids=(1, 3),
        num_key_heads=2,
        num_value_heads=2,
        key_head_dim=16,
        value_head_dim=16,
        conv_kernel_dim=4,
        output_gate=True,
        track_chunk_size=CHUNK,
    )
    linear = LinearStatePool(group, STATE_SLOTS, torch.bfloat16, torch.device("cpu"), tp_size=1)
    with patch(
        "freetoken.kvcache.mha_pool.get_tp_info",
        return_value=DistributedInfo(rank=0, size=1),
    ):
        kv = MHAKVCache(
            num_kv_heads=2,
            num_layers=4,
            head_dim=64,
            num_pages=PAGES + 1,
            page_size=1,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
            quant_k=Q8_0,
            quant_v=Q6_0,
        )
    return kv, linear


class SchedulerSessionMachine(RuleBasedStateMachine):
    strict = os.environ.get("FREETOKEN_SM_STRICT") == "1"

    def __init__(self) -> None:
        super().__init__()
        _setup_context()
        self.clock = _FakeTime()
        self.mem_available = MEM_PLENTY
        self._patches = [
            patch(f"{mod}.time", self.clock)
            for mod in (
                "freetoken.scheduler.scheduler",
                "freetoken.scheduler.cache",
                "freetoken.scheduler.session_spill",
                "freetoken.kvcache.radix_cache",
                "freetoken.kvcache.hybrid_radix_cache",
            )
        ]
        self._patches.append(
            patch.object(session_spill, "_mem_available_bytes", lambda: self.mem_available)
        )
        # Several KV chunks per checkpoint, so incremental disk checkpoints reuse files.
        self._patches.append(patch.object(session_spill, "SPILL_CHUNK_PAGES", 8))
        for p in self._patches:
            p.start()
        self.spill_dir = tempfile.mkdtemp(prefix="ft-sm-spill-")
        self.s = self._build()
        self.inbox: list = []
        self.next_uid = 1
        self.next_token = 1_000
        self.next_close = 0
        # uid -> (client session id, prompt, scripted output, updates the family history)
        self.reqs: dict[int, tuple] = {}
        self.outstanding: dict[int, str | None] = {}
        self.generated: dict[int, list[int]] = {}
        self.history: dict[str, list[int]] = {}
        self.scheduled_prefill = False
        self.seen: Counter = Counter()

    # ------------------------------------------------------------------ construction
    def _build(self) -> Scheduler:
        kv, linear = _pools()
        pt = torch.zeros((MAX_RUNNING + 1, 128), dtype=torch.int32)
        cm = CacheManager(
            PAGES,
            1,
            pt,
            "hybrid_radix",
            linear_state_pool=linear,
            swa_pool=kv,
            pin_prefix_min_tokens=PIN_MIN,
            pin_prefix_scope="shared",
            pin_working_set_slots=4 * MAX_RUNNING,
        )
        tm = TableManager(MAX_RUNNING, pt)
        dm = DecodeManager(page_size=1)
        pm = PrefillManager(cm, tm, dm)
        per_session = kv.session_spill_bytes(24) + 2 * linear.bytes_per_slot()
        store = SessionSpillStore(
            kv,
            linear,
            directory=self.spill_dir,
            ram_budget_bytes=int(per_session * 1.5),
            disk_budget_bytes=per_session * 200,
            host_reserve_bytes=0,
            state_stride_tokens=CHUNK,
            max_states=8,
        )

        s = Scheduler.__new__(Scheduler)
        s.engine = SimpleNamespace(
            kv_cache=kv,
            linear_state_pool=linear,
            encoder_cache=None,
            max_seq_len=MAX_SEQ,
            page_table=pt,
            hidden_states=None,
        )
        s.device = torch.device("cpu")
        s.config = SimpleNamespace(
            auto_session_grace_seconds=0.0,
            kv_grow_step_tokens=0,
            host_ram_reserve_gb=0.0,
            adaptive_scheduler=False,
            offline_mode=True,
            page_size=1,
            moe_collect_stats=False,
            max_running_req=MAX_RUNNING,
            cuda_memory_telemetry=False,
        )
        s.table_manager, s.cache_manager, s.decode_manager, s.prefill_manager = tm, cm, dm, pm
        cm.mamba_reclaim_hook = s._reclaim_soft_sessions_for_state_slot
        s.token_pool = tm.token_pool
        s.prefill_budget = PREFILL_BUDGET
        s.finished_reqs = set()
        s._pending_abort_acks = set()
        s._pending_session_close_acks = {}
        s._abort_tombstones = {}
        s._session_aliases = {}
        s._last_data = None
        s._pending_rebuild = None
        s._pending_durable_checkpoint = None
        s._durable_checkpoint_result = None
        s._growable_shrink_pending = False
        s._growable_shrink_in_flight = False
        s._growable_handoff_pending = None
        s._growable_decode_steps = 0
        s._growable_decode_burst = 32
        s._admission_stalled = False
        s._forward_iter = 0
        s._cold_restore_retry = False
        s._sessions = {}
        s._session_spill_store = store
        s._session_spill_last_pressure_check = 0.0
        s._state_capture = None
        s._spec = None
        s._model_is_mrope = False
        s.eos_token_ids = {EOS}
        s.toolcall_anchor_id = None
        s.status_reporter = SimpleNamespace(report_batch=lambda *a, **k: None)
        s.send_result = self._on_results
        s.receive_msg = self._receive
        s._prepare_batch = self._sim_prepare
        s._forward = self._sim_forward
        s._gpu_mem_bytes = lambda: 0
        s._cuda_memory_telemetry = lambda: None
        real_enforce = s._enforce_session_host_reserve
        s._enforce_session_host_reserve = lambda: self._checked_enforce(real_enforce)
        real_release = s._release_soft_session_handle
        real_restore = s._restore_cold_session

        def release(session_id, reason, **kw):
            released = real_release(session_id, reason, **kw)
            self.seen[f"lease released: {reason}" if released else "lease release refused"] += 1
            return released

        def restore(session_id, input_ids):
            lease = s._sessions.get(session_id)
            held = lease is not None and lease.handle is not None
            restored = real_restore(session_id, input_ids)
            if restored:
                self.seen["restore into a lease holding a prefix" if held else "restore"] += 1
            return restored

        s._release_soft_session_handle = release
        s._restore_cold_session = restore
        return s

    def teardown(self) -> None:
        store = self.s._session_spill_store
        for name in ("spills", "spills_incremental", "restores", "restores_diverged",
                     "restores_deferred", "restores_failed", "demote_writes", "prefetches"):
            if getattr(store.counters, name):
                self.seen[f"store.{name}"] += 1
        pc = self.s.cache_manager.prefix_counters
        for name in ("pin_evictions", "pin_admission_releases"):
            if getattr(pc, name):
                self.seen[f"pins.{name}"] += 1
        ids = [*self.s._sessions, *(r.session_id for r in store._records)]
        if any("~" in sid for sid in ids):
            self.seen["sibling or ~prev id"] += 1
        if currently_in_test_context():  # not when a test below replays a sequence
            for what in self.seen:
                event(what)
        try:
            self.s._session_spill_store.shutdown()
        finally:
            for p in reversed(self._patches):
                p.stop()
            shutil.rmtree(self.spill_dir, ignore_errors=True)

    # ------------------------------------------------------------------ simulated model
    def _receive(self, blocking: bool = False) -> list:
        msgs, self.inbox = self.inbox, []
        return msgs

    def _sim_prepare(self, batch) -> ForwardInput:
        """Pool effects of the real ``_prepare_batch`` (no SWA, no growth on this arena)."""
        s = self.s
        s._forward_iter += 1
        if batch.is_prefill:
            self.scheduled_prefill = True
        s.cache_manager.allocate_paged(batch.reqs)
        if batch.is_prefill:
            # attention/linear.py: the forward freezes the state at the deepest track
            # boundary strictly inside the extend into the next ping-pong slot.
            for r in batch.reqs:
                if r.mamba_ping_pong is None:
                    continue
                c = (r.extend_len - 1) // CHUNK
                if c < 1:
                    continue
                r.mamba_last_track_seqlen = r.cached_len + c * CHUNK
                r.mamba_next_track_idx = 1 - r.mamba_next_track_idx
        return ForwardInput(batch=batch, sample_args=None, input_tuple=None, write_tuple=None)

    def _sim_forward(self, forward_input: ForwardInput):
        batch = forward_input.batch
        for req in batch.reqs:
            req.complete_one()
        tokens = []
        for req in batch.reqs:
            if isinstance(req, ChunkedReq):
                tokens.append(0)
                continue
            plan = self.reqs[req.uid][2]
            done = len(self.generated.get(req.uid, ()))
            tokens.append(plan[done] if done < len(plan) else 7)
        self.s.decode_manager.filter_reqs(batch.reqs)
        return (None, torch.tensor(tokens, dtype=torch.int32), _Done())

    def _on_results(self, msgs) -> None:
        for m in msgs:
            if isinstance(m, DetokenizeMsg):
                self.generated.setdefault(m.uid, []).append(m.next_token)
                if m.finished:
                    self._terminal(m.uid, finished=True)
            elif isinstance(m, ErrorReplyMsg):
                self._terminal(m.uid, finished=False)

    def _terminal(self, uid: int, *, finished: bool) -> None:
        self.outstanding.pop(uid, None)
        info = self.reqs.get(uid)
        if not finished or info is None:
            return
        session_id, prompt, _plan, updates = info
        if updates and session_id is not None:
            self.history[session_id] = prompt + self.generated.get(uid, [])

    def _checked_enforce(self, real_enforce) -> None:
        before = self.s._session_spill_last_pressure_check
        real_enforce()
        if self.s._session_spill_last_pressure_check == before:
            return  # rate-limited: the cleanup did not run this iteration
        for sid, lease in self.s._sessions.items():
            # Invariant 5b: the cleanup step leaves no lease pointing at a dead record.
            assert lease.spill is None or lease.spill.valid, (
                f"session {sid} still points at an invalid checkpoint after cleanup"
            )

    def _settle_threads(self) -> None:
        store = self.s._session_spill_store
        for write in list(store._pending_writes):
            write.done.wait(timeout=30)
        for state in (store._prefetch, store._write_behind):
            if state is not None and state.thread is not None:
                state.thread.join(timeout=30)

    def _fresh(self, n: int) -> list[int]:
        out = list(range(self.next_token, self.next_token + n))
        self.next_token += n
        return out

    def _iterate(self) -> bool:
        """One real ``normal_loop`` iteration; True when it scheduled a prefill batch."""
        self.clock.now += 0.05
        self.scheduled_prefill = False
        self.s.normal_loop()
        self._settle_threads()
        return self.scheduled_prefill

    # ------------------------------------------------------------------ rules
    def _send(self, family, kind, new_len, cut, out_len, stop_early, reclaimable, ttl) -> int:
        past = self.history.get(family)
        if kind in ("continue", "diverge") and (past is None or len(past) + new_len > MAX_PROMPT):
            kind = "new"
        if kind == "continue":
            prompt = past + self._fresh(new_len)
        elif kind == "diverge":
            # a retokenized or edited turn: shares a prefix of the conversation, then drifts
            keep = len(SYSTEM) + cut % max(1, len(past) - len(SYSTEM))
            prompt = past[:keep] + self._fresh(new_len)
        elif kind == "summary":
            # omp's compaction summary: same session id, a prompt that does not continue it
            prompt = SYSTEM + SUMMARY + self._fresh(new_len)
        else:
            prompt = SYSTEM + self._fresh(new_len + 6)
        plan = self._fresh(out_len)
        if stop_early and out_len > 1:
            plan[-1] = EOS
        uid = self.next_uid
        self.next_uid += 1
        session_id = None if kind == "nosession" else family
        self.reqs[uid] = (session_id, prompt, plan, kind != "summary")
        self.outstanding[uid] = session_id
        self.inbox.append(
            UserMsg(
                uid=uid,
                input_ids=torch.tensor(prompt, dtype=torch.int32),
                sampling_params=SamplingParams(max_tokens=out_len),
                session_id=session_id,
                session_ttl_seconds=ttl,
                session_reclaimable=reclaimable,
            )
        )
        return uid

    def _run_until_done(self, uids, limit: int = 40) -> None:
        for _ in range(limit):
            if not any(uid in self.outstanding for uid in uids):
                break
            self._iterate()
        self._check_progress()

    KINDS = st.sampled_from(["new", "continue", "continue", "continue", "diverge", "summary",
                             "nosession"])
    RECLAIMABLE = st.sampled_from([True, True, True, False])
    TTL = st.sampled_from([1.0, 30.0, 300.0])

    @rule(family=st.sampled_from(FAMILIES), kind=KINDS, new_len=st.integers(1, 14),
          cut=st.integers(0, 40), out_len=st.integers(1, 4), stop_early=st.booleans(),
          reclaimable=RECLAIMABLE, ttl=TTL)
    def arrive(self, family, kind, new_len, cut, out_len, stop_early, reclaimable, ttl):
        self._send(family, kind, new_len, cut, out_len, stop_early, reclaimable, ttl)

    @rule(family=st.sampled_from(FAMILIES), kind=KINDS, new_len=st.integers(1, 14),
          cut=st.integers(0, 40), out_len=st.integers(1, 4), reclaimable=RECLAIMABLE)
    def turn(self, family, kind, new_len, cut, out_len, reclaimable):
        """A whole turn: the request arrives and the loop runs until it ends."""
        uid = self._send(family, kind, new_len, cut, out_len, False, reclaimable, 300.0)
        self._run_until_done([uid])

    @rule(family=st.sampled_from(FAMILIES), summary_len=st.integers(1, 8),
          new_len=st.integers(1, 8), summary_first=st.booleans(), run=st.booleans())
    def compaction(self, family, summary_len, new_len, summary_first, run):
        """omp compaction: the summary request and the continuing turn arrive together under
        one automatic session id (the second is rebound to a sibling ``id~N`` lease)."""
        summary = (family, "summary", summary_len, 0, 2, False, True, 300.0)
        turn = (family, "continue", new_len, 0, 2, False, True, 300.0)
        first, second = (summary, turn) if summary_first else (turn, summary)
        uids = [self._send(*first), self._send(*second)]
        if run:
            self._run_until_done(uids)

    @rule(passes=st.integers(1, 6))
    def run_loop(self, passes):
        for _ in range(passes):
            self._iterate()
        self._check_progress()

    @precondition(lambda self: self.outstanding)
    @rule(pick=st.integers(0, 1 << 16), ahead=st.booleans())
    def abort(self, pick, ahead):
        uids = sorted(self.outstanding)
        uid = uids[pick % len(uids)]
        msg = AbortBackendMsg(uid=uid, session_id=self.outstanding[uid])
        if ahead:
            # Several tokenizer workers: an abort may overtake its own request.
            self.inbox.insert(0, msg)
        else:
            self.inbox.append(msg)

    @rule(pick=st.integers(0, 1 << 16))
    def close(self, pick):
        ids = sorted(set(FAMILIES) | set(self.s._sessions))
        self.next_close += 1
        self.inbox.append(
            CloseSessionBackendMsg(
                session_id=ids[pick % len(ids)], request_id=f"c{self.next_close}"
            )
        )

    @rule(seconds=st.sampled_from([0.5, 2.0, 40.0, 400.0]))
    def advance_clock(self, seconds):
        self.clock.now += seconds

    @rule()
    def unpin(self):
        self.next_close += 1
        self.inbox.append(UnpinPrefixesBackendMsg(request_id=f"u{self.next_close}"))

    @rule(pressure=st.booleans())
    def host_memory(self, pressure):
        self.mem_available = 0 if pressure else MEM_PLENTY

    @rule(n=st.integers(1, STATE_SLOTS - 1))
    def state_slot_demand(self, n):
        # What a chunk commit or a restore does when it needs a GDN slot: free-list,
        # snapshot eviction, then the lease-spill hook (mamba_reclaim_hook).
        self.s.cache_manager.reserve_mamba_slots(n)
        self._settle_threads()

    # ------------------------------------------------------------------ progress (4)
    def _check_progress(self) -> None:
        """(4) Bounded progress. With requests queued, nothing running and no message in
        flight, the loop must admit within ``PROGRESS_PASSES`` passes -- or after every idle
        lease's TTL has run out, since an idle explicit lease is released only by its TTL and
        the reclaim deliberately never robs a parked restore of a request queued AHEAD of
        the one it serves. A queue still refused after that never moves again: then the
        reclaim must not have been able to do better, and every queued request must be
        genuinely unseatable."""
        s = self.s
        pm, dm = s.prefill_manager, s.decode_manager

        def idle() -> bool:
            return not dm.running_reqs and not any(p.chunked_req for p in pm.pending_list)

        def admits(passes: int) -> bool:
            for _ in range(passes):
                if self._iterate() or not pm.pending_list or not idle():
                    return True
            return False

        if self.inbox or not pm.pending_list or not idle() or admits(PROGRESS_PASSES):
            return
        self.clock.now += 2 * 86_400.0
        if admits(PROGRESS_PASSES):
            self.seen["waited out an idle lease's TTL"] += 1
            return
        queued = [(p.uid, p.session_id, p.input_len, p.output_len) for p in pm.pending_list]
        state = self._describe()
        stale = [
            sid for sid, lease in s._sessions.items()
            if lease.reclaimable and lease.handle is not None and lease.handle.cached_len
            and (lease.token_ids is None or len(lease.token_ids) != lease.handle.cached_len)
        ]
        short_own = []
        for p in pm.pending_list:
            own = s._sessions.get(p.session_id) if p.session_id else None
            if (own is not None and own.reclaimable and own.active_uid == p.uid
                    and own.handle is not None and own.token_ids is not None
                    and len(own.token_ids) == own.handle.cached_len
                    and not s._extends(own, p.input_ids)
                    and own.handle.cached_len
                    <= s.cache_manager.match_req(p).cuda_handle.cached_len):
                short_own.append(p.uid)
        # An explicit lease its own queued turn diverged from: admission would move it to
        # ~prev anyway, but until then nothing may release it, whoever it starves.
        explicit_diverged = [
            p.uid for p in pm.pending_list
            if p.session_id in s._sessions
            and not s._sessions[p.session_id].reclaimable
            and s._sessions[p.session_id].active_uid == p.uid
            and s._sessions[p.session_id].handle is not None
            and not s._extends(s._sessions[p.session_id], p.input_ids)
        ]
        for sid, lease in list(s._sessions.items()):
            if lease.reclaimable and lease.handle is not None:
                s._release_soft_session_handle(sid, "progress oracle", owner_uid=lease.active_uid)
        s.cache_manager.unpin_all()
        admitted = admits(1)
        if admitted and stale:
            # The release the reclaim needed is the one it is refused (no checkpoint).
            self._known_bug(RESTORE_KEEPS_STALE_TOKEN_IDS)
        if admitted and explicit_diverged:
            self._known_bug(EXPLICIT_LEASE_STARVES_OWN_TURN)
        if admitted and short_own:
            # Its own lease is not on its path, but no longer than the match: not "diverged".
            self._known_bug(SHORT_OWN_LEASE_NEVER_RELEASED)
        assert not admitted, (
            f"permanent stall: never admitted, yet admits once the automatic leases and pins "
            f"are released; queue={queued}\n{state}"
        )
        cm = s.cache_manager
        for p in pm.pending_list:
            own = s._sessions.get(p.session_id) if p.session_id else None
            if own is not None and not own.reclaimable and own.active_uid == p.uid and (
                own.handle is not None
            ):
                # Its own explicit lease: busy, so no TTL; explicit, so never reclaimed.
                handle = cm.match_req(p).cuda_handle
                need = p.input_len + p.output_len - handle.cached_len
                if need > cm.available_size - cm.lock_delta(handle):
                    self._known_bug(EXPLICIT_LEASE_STARVES_OWN_TURN)
        for p in pm.pending_list:
            handle = cm.match_req(p).cuda_handle
            need = p.input_len + p.output_len - handle.cached_len
            room = cm.available_size - cm.lock_delta(handle)
            assert need > cm.committed_pages, (
                f"permanent stall: uid={p.uid} needs {need} pages of a {cm.committed_pages}-page "
                f"pool ({room} obtainable, {cm.mamba_available_size} state slots) and is never "
                f"admitted; queue={queued}\n{state}"
            )
        self.seen["refused: larger than the pool"] += 1

    def _known_bug(self, name: str) -> None:
        """Discard an example that reached an open bug's signature (unless strict)."""
        if name in KNOWN_BUGS and not self.strict:
            event(f"known bug: {name}")
            assume(False)

    def _describe(self) -> str:
        s, cm = self.s, self.s.cache_manager
        leases = {
            sid: (
                getattr(lease.handle, "cached_len", None),
                lease.active_uid,
                lease.reclaimable,
                None if lease.spill is None else lease.spill.valid,
            )
            for sid, lease in s._sessions.items()
        }
        return (
            f"leases={leases} free_pages={len(cm.free_slots)} "
            f"free_slots={cm.linear_state_pool.num_free_slots} "
            f"mamba_available={cm.mamba_available_size} pins={len(cm._pins)}"
        )

    # ------------------------------------------------------------------ ownership
    def _live_reqs(self) -> list:
        s = self.s
        reqs = list(s.decode_manager.running_reqs)
        reqs += [p.chunked_req for p in s.prefill_manager.pending_list if p.chunked_req is not None]
        return [r for r in {id(r): r for r in reqs}.values() if r.table_idx != -1]

    def _lock_owners(self) -> list[tuple[str, object]]:
        """(owner, locked node) for every holder of a radix lock."""
        s, cm = self.s, self.s.cache_manager
        owners = [
            (f"lease {sid}", lease.handle.node)
            for sid, lease in s._sessions.items()
            if lease.handle is not None
        ]
        owners += [(f"req {r.uid}", r.cache_handle.node) for r in self._live_reqs()]
        owners += [("pin", node) for node in cm._pin_locked]
        return [(who, node) for who, node in owners if not node.is_root()]

    @invariant()
    def gdn_slots_are_conserved(self):
        """(1) free-list + tree snapshots + live requests' slots == pool, none twice."""
        cm = self.s.cache_manager
        pool, tree = cm.linear_state_pool, cm.prefix_cache
        owned: dict[int, str] = {}

        def own(slot, who):
            assert slot is not None and slot != pool.padding_slot, f"{who} holds slot {slot}"
            assert slot not in owned, f"GDN slot {slot} owned by {owned[slot]} and {who}"
            owned[slot] = who

        for slot in pool._free_slots:
            own(slot, "free-list")
        for node in tree._snapshot_nodes():
            own(node.mamba_value, f"tree node@{tree._path_len(node)}")
        for r in self._live_reqs():
            if r.linear_slot_idx is not None:
                own(r.linear_slot_idx, f"req {r.uid} live")
            for slot in r.mamba_ping_pong or ():
                own(slot, f"req {r.uid} ping-pong")
            if r.spec_scratch_slot is not None:
                own(r.spec_scratch_slot, f"req {r.uid} scratch")
        assert len(owned) == pool.num_slots - 1, (
            f"GDN slots: {len(owned)} accounted of {pool.num_slots - 1} "
            f"(missing {sorted(set(range(1, pool.num_slots)) - set(owned))})"
        )
        assert tree.mamba_evictable + tree.mamba_protected == len(tree._snapshot_nodes())

    @invariant()
    def radix_locks_match_their_owners(self):
        """(2) every node's ref_count is exactly the owners locking it or a descendant."""
        tree = self.s.cache_manager.prefix_cache
        expected: dict[int, int] = {}
        direct: dict[int, list[str]] = {}
        for who, node in self._lock_owners():
            cur = node
            while not cur.is_root():
                parent = cur.parent
                assert parent is not None and parent.children.get(tree.key_fn(cur._key)) is cur, (
                    f"{who} locks a node that is no longer in the tree"
                )
                expected[id(cur)] = expected.get(id(cur), 0) + 1
                cur = parent
            assert cur is tree.root, f"{who} locks a node of another tree"
            direct.setdefault(id(node), []).append(who)
        for node in tree._all_nodes():
            want = expected.get(id(node), 0)
            assert node.ref_count == want, (
                f"node@{tree._path_len(node)} ref_count {node.ref_count} != {want} owners "
                f"({direct.get(id(node), [])})"
            )
            holders = direct.get(id(node), [])
            if node.mamba_value is None:
                assert node.mamba_ref_count == 0, f"tombstone@{tree._path_len(node)} has mamba refs"
                # Leases and requests lock a matched snapshot node; only a pin may lock KV alone.
                if any(not who.startswith("pin") for who in holders):
                    raise AssertionError(
                        f"snapshot@{tree._path_len(node)} was evicted under {holders}"
                    )
                continue
            assert node.mamba_ref_count <= len(holders), (
                f"snapshot@{tree._path_len(node)} mamba_ref_count {node.mamba_ref_count} > "
                f"its {len(holders)} owners {holders}"
            )
            # A lease or a request reads this snapshot; it must not be evictable under it.
            # (A pin that locked the node before it had a snapshot is a KV-only pin there.)
            if any(not who.startswith("pin") for who in holders):
                assert node.mamba_ref_count >= 1, (
                    f"snapshot@{tree._path_len(node)} held by {holders} is evictable"
                )
        evictable = sum(n.length for n in tree._all_nodes() if n.ref_count == 0)
        assert tree.full_evictable == evictable, (tree.full_evictable, evictable)

    @invariant()
    def kv_pages_are_conserved(self):
        """(3) free pages + tree pages + live requests' private pages == pool, none twice."""
        cm = self.s.cache_manager
        owned: dict[int, str] = {}

        def own(pages, who):
            for page in pages.tolist():
                assert 0 <= page < cm.committed_pages, f"{who} holds page {page}"
                assert page not in owned, f"KV page {page} owned by {owned[page]} and {who}"
                owned[page] = who

        own(cm.free_slots, "free-list")
        tree = cm.prefix_cache
        for node in tree._all_nodes():
            own(node.value, f"tree node@{tree._path_len(node)}")
        for r in self._live_reqs():
            private = cm.page_table[r.table_idx, r.cache_handle.cached_len : r.cached_len]
            own(private, f"req {r.uid}")
        assert len(owned) == cm.committed_pages, (
            f"KV pages: {len(owned)} accounted of {cm.committed_pages}"
        )

    @invariant()
    def spill_store_accounting_matches_records(self):
        """(5) ram_bytes / disk_bytes are exactly what the valid records hold."""
        store = self.s._session_spill_store
        records = store._records
        assert all(r.valid for r in records), "an invalid record is still tracked"
        ram = sum(r.byte_size for r in records if r.tier == "ram")
        ram += sum(r.cached_bytes for r in records if r.tier == "disk")
        ram += sum(w.nbytes for w in store._pending_writes if w.charged)
        disk = sum(r.byte_size for r in records if r.tier == "disk")
        disk += sum(r.byte_size for r in records if r.tier == "ram" and r.backing is not None)
        assert (store.ram_bytes, store.disk_bytes) == (ram, disk), (
            f"store charges ram={store.ram_bytes} disk={store.disk_bytes}, records hold "
            f"ram={ram} disk={disk}"
        )
        for sid, record in store._by_session.items():
            assert any(r is record for r in records) and record.session_id == sid

    @invariant()
    def lease_tokens_describe_their_prefix(self):
        """A lease checkpoints ``token_ids`` with its handle's pages
        (``_spill_soft_session``): equal lengths with different tokens would write a
        checkpoint whose KV belongs to another prompt."""
        for sid, lease in self.s._sessions.items():
            handle, tokens = lease.handle, lease.token_ids
            if handle is None or tokens is None or len(tokens) != handle.cached_len:
                continue
            keys, node = [], handle.node
            while not node.is_root():
                keys.append(node._key)
                node = node.parent
            path = torch.cat(keys[::-1]) if keys else torch.empty(0, dtype=torch.int32)
            if not torch.equal(path.to(torch.int32), tokens.to(torch.int32)):
                self._known_bug(RESTORE_KEEPS_STALE_TOKEN_IDS)
            assert torch.equal(path.to(torch.int32), tokens.to(torch.int32)), (
                f"session {sid}: token_ids do not match the {handle.cached_len}-token prefix "
                f"its handle locks"
            )

    @invariant()
    def busy_leases_have_a_live_request(self):
        """A lease marked busy must belong to a request still queued or running; otherwise
        every later turn of that session is refused as busy forever."""
        s = self.s
        live = {p.uid for p in s.prefill_manager.pending_list}
        live |= {r.uid for r in s.decode_manager.running_reqs}
        for sid, lease in s._sessions.items():
            assert lease.active_uid is None or lease.active_uid in live, (
                f"session {sid} is busy with uid {lease.active_uid}, which is gone"
            )


def _profile():
    # The default run is derandomized so the suite is reproducible; the heavy one explores.
    profile = os.environ.get("FREETOKEN_SM_PROFILE")
    if profile == "heavy":
        return dict(max_examples=3_000, stateful_step_count=100)
    if profile == "quick":
        return dict(max_examples=25, stateful_step_count=30, derandomize=True)
    return dict(max_examples=300, stateful_step_count=50, derandomize=True)


SchedulerSessionMachine.TestCase.settings = settings(
    **_profile(),
    deadline=None,
    database=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
TestSchedulerSessionMachine = SchedulerSessionMachine.TestCase


# ---------------------------------------------------------------------- shrunk sequences
INVARIANTS = (
    "gdn_slots_are_conserved",
    "radix_locks_match_their_owners",
    "kv_pages_are_conserved",
    "spill_store_accounting_matches_records",
    "lease_tokens_describe_their_prefix",
    "busy_leases_have_a_live_request",
)


def _replay(steps) -> None:
    """Run a sequence Hypothesis shrunk, strictly, checking every invariant after each step."""
    machine = SchedulerSessionMachine()
    machine.strict = True
    try:
        for name, kwargs in steps:
            getattr(machine, name)(**kwargs)
            for check in INVARIANTS:
                getattr(machine, check)()
    finally:
        machine.teardown()


def _compaction(family, new_len, summary_len, *, run, summary_first=False):
    return ("compaction", dict(family=family, new_len=new_len, summary_len=summary_len,
                               run=run, summary_first=summary_first))


def _arrive(family, kind, new_len, out_len, *, cut=0, reclaimable=True):
    return ("arrive", dict(family=family, kind=kind, new_len=new_len, cut=cut, out_len=out_len,
                           stop_early=False, reclaimable=reclaimable, ttl=1.0))


def _loop(passes):
    return ("run_loop", dict(passes=passes))


def test_compaction_summary_then_restore_into_the_same_lease_leaks_no_lock():
    # 2fae797a: with its scheduler.py hunk reverted this fails at the last step with
    # "node@6 ref_count 5 != 4 owners" -- one locked prefix leaked per compaction turn.
    _replay([
        _compaction("A", 4, 4, run=False),
        _loop(1),
        _compaction("A", 1, 6, run=True),
        _compaction("A", 1, 6, run=True),
    ])


def test_open_bug_sibling_restore_cut_inside_a_resident_node():
    _replay([
        _arrive("A", "new", 1, 1),
        _arrive("B", "new", 1, 3),
        _loop(1),
        _arrive("A", "new", 13, 3),
        _arrive("A", "new", 9, 1),
        _loop(6),
        _loop(1),
        _arrive("B", "continue", 1, 2),
        _arrive("B", "diverge", 1, 1, cut=6),
        _loop(1),
    ])


def test_open_bug_restored_snapshot_is_evictable_under_its_lease(tmp_path):
    _setup_context()
    kv, linear = _pools()
    cm = CacheManager(PAGES, 1, torch.zeros((2, 64), dtype=torch.int32), "hybrid_radix",
                      linear_state_pool=linear, swa_pool=kv)
    store = SessionSpillStore(kv, linear, directory=str(tmp_path), ram_budget_bytes=1 << 30,
                              disk_budget_bytes=0, host_reserve_bytes=0,
                              state_stride_tokens=CHUNK)
    try:
        tokens = torch.arange(100, 115, dtype=torch.int32)
        pages = cm._page_to_token(cm._allocate(len(tokens)))
        at12, at15 = linear.alloc(2)
        cm.prefix_cache.insert(tokens[:12], pages[:12], at12)
        cm.prefix_cache.insert(tokens, pages, at15)
        handle = cm.retain_prefix(tokens, len(tokens))
        record = store.spill("B", tokens, pages, at15,
                             extra_states=cm.hybrid_session_state_boundaries(handle))
        cm.unlock(handle)
        cm.evict_all_unlocked_prefixes()
        resident = cm.restore_hybrid_session_prefix(record, store, 15)  # lease B
        # lease B~1 cuts the same record at 12, inside the node lease B just restored
        sibling = cm.restore_hybrid_session_prefix(record, store, 12)
        assert sibling.node.mamba_ref_count == 1  # was 0: evict_mamba frees it under the lease
        assert sibling.node.ref_count == 2  # sibling plus the descendant resident lease
        slot = sibling.node.mamba_value
        cm.prefix_cache.evict_mamba(STATE_SLOTS)
        assert sibling.node.mamba_value == slot
        cm.unlock(sibling)
        assert sibling.node.mamba_ref_count == 0
        assert sibling.node.ref_count == 1
        cm.unlock(resident)
        assert sibling.node.ref_count == 0
    finally:
        store.shutdown()


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=EXPLICIT_LEASE_STARVES_OWN_TURN + (
    ": the lease is busy (expires_at None, so no TTL) and explicit (the own-lease release in "
    "_reclaim_soft_sessions_for_pending requires own.reclaimable)"))
def test_open_bug_explicit_session_turn_that_does_not_fit_beside_its_own_lease():
    _replay([
        _arrive("A", "new", 12, 1, reclaimable=False),
        _loop(2),
        _arrive("A", "new", 9, 4, reclaimable=False),
        _loop(1),
    ])


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=RESTORE_KEEPS_STALE_TOKEN_IDS + (
    ": _restore_cold_session sets session.handle but not session.token_ids; with a different "
    "length _spill_soft_session refuses, so no require_checkpoint release frees a parked restore"))
def test_open_bug_two_parked_restores_block_each_other():
    _replay([
        _loop(1),
        _compaction("A", 1, 1, run=False),
        _compaction("A", 4, 1, run=True),
        _compaction("A", 7, 1, run=True),
        _compaction("C", 6, 1, run=False),
        _loop(1),
        _compaction("A", 2, 1, run=True),
        _compaction("A", 1, 1, run=False),
        _loop(1),
        _compaction("A", 3, 1, run=False),
        _compaction("A", 1, 1, run=False),
        _loop(1),
        _compaction("C", 3, 1, run=True, summary_first=True),
    ])


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=RESTORE_KEEPS_STALE_TOKEN_IDS + (
    ": with the same length _spill_soft_session accepts them and checkpoints the restored KV "
    "under the previous turn's token ids"))
def test_open_bug_restored_lease_would_checkpoint_kv_under_another_prompt():
    _replay([
        _arrive("A", "new", 11, 1),
        _arrive("A", "nosession", 10, 4),
        _compaction("A", 2, 6, run=True),
        _compaction("A", 1, 1, run=False),
        _compaction("A", 1, 1, run=False),
        _loop(1),
    ])


def _turn(family, kind, new_len, out_len):
    return ("turn", dict(family=family, kind=kind, new_len=new_len, cut=0, out_len=out_len,
                         reclaimable=True))


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=SHORT_OWN_LEASE_NEVER_RELEASED + (
    ": _reclaim_soft_sessions_for_pending releases the own lease only if own_len > cached_len, "
    "a length test; here the lease is a 17-token summary branch and the request matches 28 "
    "tokens of the closed session's history, 1 page short of fitting beside the lease"))
def test_open_bug_own_summary_lease_starves_the_turn_that_continues_the_history():
    _replay([
        _turn("A", "new", 13, 4),
        ("close", dict(pick=0)),
        _turn("A", "summary", 6, 4),
        _turn("A", "continue", 5, 4),
    ])


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=EXPLICIT_LEASE_STARVES_OWN_TURN + (
    ": the head (A) is 1 page short; B's explicit lease, which B's queued diverged turn would "
    "move to B~prev at admission, is never released, and A's lease is parked ahead of B"))
def test_open_bug_explicit_lease_of_a_queued_diverged_turn_starves_the_head():
    _replay([
        _arrive("B", "new", 14, 4, reclaimable=False),
        _turn("A", "new", 1, 3),
        _arrive("A", "continue", 6, 4),
        ("turn", dict(family="B", kind="new", new_len=1, cut=0, out_len=1, reclaimable=False)),
    ])
