#!/usr/bin/env python3
"""CPU replay of the Switchyard *stage* route against FreeToken's scheduler.

Two ways to run it:

  * ``--gate`` -- the CI regression gate (.github/workflows/cpu-checks.yml). Runs the four
    fixed scenarios below and exits non-zero if throughput, completions or (on the
    residency profiles) the error rate fall outside the recorded floors, if the scheduler
    raises at all, if it DEADLOCKS, or if the finishability invariant is violated on any
    pass. Takes ~1 min.
  * everything else -- a single ad-hoc run that prints one JSON line, for bisecting a
    scheduler regression or measuring a candidate fix (``--diagnose`` adds the
    per-refusal breakdown that identified the prefill admission-gate starvation).

No GPU, no model. Drives the real PrefillManager / CacheManager / TableManager /
DecodeManager through the traffic shape that starved on the stage route
(benchmarks/results/nemotron35_lightning_5080_switchyard_soak_2026-09-04.md, R5):

  * 16 concurrent clients, each looping request -> 256 decode tokens -> next request
  * scenario mix and prompt lengths taken from the soak's per-scenario counts
    (long-context 118K tokens, tool catalogs ~9K, bursts ~4K, prefix-reuse ~2K)
  * ~75% shared prefix per conversation family, so the radix cache really matches
  * 262,144-token pool, page_size 1, growing in 65,536-token steps (the soak's
    --num-tokens / --kv-grow-step-tokens)
  * 8,192-token prefill budget, 32-step decode burst (Scheduler._growable_decode_burst)

The ``switchyard-stage`` profile adds the piece the other three do not model, and without
which this replay cannot see the failure of soak report S: SESSION RESIDENCY. Its clients
drive persistent sessions, so a finished turn retains (LOCKS) its prefix in the radix tree
until an idle timeout or the scheduler's demand reclaim releases it. Locked pages are
neither free nor evictable, so they leave ``available_size`` -- and a fresh-admit gate that
charges the pool MAXIMUM cannot tell that state apart from an empty pool. It also models the
soak's 600 s client timeout, so a starved request becomes an error rather than an eternal
wait.

The ``switchyard-deadlock`` profile is what soak report T needed and neither of the above
had: the ~40K mid-sized chunked prompts that deadlocked the pool against each other, plus
the FAITHFUL client-timeout model (``abort_inflight=False``; see ``abandon_starved``). A
client giving up does not give the engine its KV back -- FastAPI never sees the disconnect
of a request that has yielded no chunk -- and modelling the timeout as a free abort was the
escape hatch that let this replay pass both trees that then deadlocked live.

Metrics: lanes per prefill batch, prefill budget utilisation, wait-to-first-chunk per
request, and any fatal raised out of the scheduler path; plus, for the residency profiles,
the stall signature (``stall_frac`` / ``stall_usage_p50`` -- share of the wall clock with no
batch, and the pool occupancy while it is silent), ``timeouts`` / ``error_rate``, the
session-lease lifecycle (``lease_reclaims`` / ``lease_expiries``), and ``match_calls`` /
``match_tokens_per_prefill_pass`` as the wall-clock proxy for the O(queue x prompt) radix
walks that a refused pass repeats.

Two of them exist because a throughput number could not see soak report T at all:

  * ``deadlock`` / ``trailing_silence`` / ``trailing_silence_frac`` -- a deadlock produces
    ZERO gaps BETWEEN batch lines, because the silence starts after the last one and never
    ends. Measure the trailing half or the run that never recovers scores best.
  * ``invariant_violations`` / ``slack_min`` -- the finishability invariant of the ADMITTED
    SET, sampled every tick:

        owed = SUM over in-flight chunked prefills of (input_len - forwarded) + output_len
             + DecodeManager.inflight_tokens
        owed <= CacheManager.available_size + reclaimable idle lease tokens

    Violated = the pool has promised more than it can ever hand over, and since the
    chunked-prefill scheduler advances every lane together, nothing completes to break the
    tie. This is the check that separates the trees: ``ea7ed7c`` beats every throughput
    floor here and violates it on 566 switchyard-stage passes (short by up to 192,242
    tokens) and 2,181 switchyard-deadlock passes. The scheduler carries the same statement
    as an env-gated assertion, ``FREETOKEN_SCHEDULER_INVARIANT=warn|raise``.

Deliberately API-defensive: it runs unchanged at every commit that has
PrefillManager.schedule_next_batch(budget), and reports which knobs were absent.
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import random
import statistics
import sys
import time
import traceback

import torch

POOL = 262_144          # --num-tokens
GROW_STEP = 65_536      # --kv-grow-step-tokens
PAGE_SIZE = 1
MAX_RUNNING = 16        # --max-running-requests
PREFILL_BUDGET = 8_192  # --max-prefill-length
DECODE_BURST = 32       # Scheduler._growable_decode_burst
OUTPUT_LEN = 256        # --max-output-tokens on the soak client
WIDTH = 131_072 + 512   # --max-seq-len-override + output headroom

# Prompt length per scenario and its share of the stage route's 278 requests (R3).
SCENARIOS = [
    ("prefix-reuse",         2_048,  60),
    ("growing-conversation", 6_144,  59),
    ("tool-call-burst",      4_096,  59),
    ("large-tool-catalog",   9_216,  54),
    ("long-context",       118_000,  46),
]
# "pressure" profile: the steady state a closed 16-client loop drifts into on the stage
# route -- long prompts outlive short ones, so the queue fills with them. Drives the pool
# to the >=0.97 occupancy at which the pre-fad1fc4 fatal fired.
SCENARIOS_PRESSURE = [
    ("large-tool-catalog",   9_216,  10),
    ("long-context",       118_000,  90),
]
# "fanout" profile: the batch shape the pre-fad1fc4 fatal actually died on
# (server.log: "#new-seq: 14, #new-token: 7168" = 14 chunked lanes x 512, token usage 0.99).
# Sixteen medium prompts that all chunk, in a pool just large enough to admit them all --
# so every lane is a CONTINUATION, and continuations were checked by nothing.
SCENARIOS_FANOUT = [
    ("medium-chunked", 8_192, 100),
]
# "switchyard-stage" profile: the soak's own geometry, including the piece the other three
# profiles do not model at all -- SESSION RESIDENCY. Every Switchyard client drives a
# persistent session, so each finished turn ends in
# ``Scheduler._free_req_resources(retain_session=True)`` -> ``CacheManager.retain_prefix``,
# which LOCKS the turn's prefix in the radix tree. Locked pages leave ``evictable_size`` and
# therefore leave ``available_size``: they are neither free nor reclaimable by ordinary
# eviction, and a lease is only released on an idle timeout or by the scheduler's demand
# reclaim (``_reclaim_soft_sessions_for_pending``, which skips sessions that are ACTIVE).
#
# That is exactly the state 81ab30e's finishability gate could not see (soak report S5):
# it charged fresh admits against ``max_size`` minus the in-flight prefills, i.e. the whole
# pool, while sixteen retained session prefixes plus the decoding requests had already given
# most of that pool away. Admissions kept arriving until token usage 1.00, then no lane could
# buy its next chunk and nothing could complete to free one.
SCENARIOS_SWITCHYARD = [
    ("prefix-reuse",         2_048,  58),
    ("growing-conversation", 6_144,  57),
    ("tool-call-burst",      4_096,  57),
    ("large-tool-catalog",   9_216,  48),
    ("long-context",       118_000,  48),
]
# "switchyard-deadlock" profile: the geometry of the permanent deadlock of soak report T5.
#
# The failure there was NOT long-context prompts. It was FOURTEEN mid-sized chunked
# prefills, admitted one at a time through a gate that re-counted the same reclaimable
# lease tokens for each of them, which between them had forwarded 237,819 tokens and still
# owed 222,538 -- a combined footprint of 1.76x a 262,144-token pool. Nothing was decoding,
# so nothing could complete to hand a page back, and no lane could buy the page it needed
# to finish. 222,538 / 14 = 15.9K owed each on ~33K prompts; the scenario length is set a
# little above that, at the point where sixteen of them cannot possibly fit in the pool and
# six comfortably can, which is where a gate that re-sells the same capacity separates
# hardest from one that does not.
#
# Session residency is kept (it is the reclaimable capacity the broken gate double-counts)
# but the prompts are short enough that sixteen RETAINED prefixes cannot fill the pool on
# their own -- so the only way this profile can wedge is by over-admitting prefills, which
# is exactly the property under test. Its client-timeout model is the faithful one
# (``abort_inflight=False``, see abandon_starved): the deadlock is permanent, as it was
# live.
SCENARIOS_DEADLOCK = [
    ("chunked-mid", 40_000, 100),
]
# "switchyard-restore" profile: the soak §W6 geometry -- the same scenario mix and session
# residency as "switchyard-stage", with the half of the session cycle none of the profiles
# above modelled: a reclaimed lease is CHECKPOINTED (``_release_soft_session_handle`` spills
# before it unlocks), and the session's next turn RESTORES it.
#
# That closes a hole in the accounting, not just in the traffic. While a prefix sits on
# disk it costs the pool nothing, so ``available_size`` counts its tokens as free and the
# admission gate sells them to chunked prefills; ``_restore_cold_session`` then takes them
# straight back -- allocated and locked, outside every admission gate, and AFTER that gate
# proved the admitted set finishable against an ``available_size`` the restore is about to
# shrink. Nothing re-checks the proof, so the invariant goes negative by exactly the
# restore's footprint and then tracks the admitted set in lockstep as it drains.
#
# Live (§W6): nine ``finishability invariant`` warnings in a 19 s window of the ca7e74b
# passthrough tail, a CONSTANT 1,401-token shortfall while owed and available fell by one
# 8,192-token chunk per pass -- the pool over-promised exactly once. The two seconds before
# the first warning hold four ``Restored cold session`` lines, one of 79,104 tokens.
SCENARIOS_RESTORE = SCENARIOS_SWITCHYARD
# "ornith-ada" profile: the growable-GGUF SINGLE-LANE geometry, which none of the profiles
# above reach. ``_resolve_max_prefill_seqs`` returns 1 -- and therefore
# ``PrefillManager.max_batch_seqs = 1`` -- only for growable quantized-GGUF MoE
# (``model_config.gguf_expert_types is not None``) with ``max_running_req > 1``; every
# Nemotron tree the four profiles above model runs with ``max_batch_seqs == 0``, so the
# lane cap and the short-prompt grouping escape hatch beside it (fork/main's
# ``_auto_small_prompt_group_tokens``: 1,536 tokens on sm_89, 1,280 elsewhere) are
# untested here.
#
# That matters after the seatable-lanes divisor (812bc57): the divisor is
# ``_seatable_lanes(adder, lane_cap, ...)``, which STOPS at ``lane_cap`` -- so on this path
# it is 1 whenever the cap holds and the whole budget goes to the one lane, while a group of
# short fresh prompts lifts the cap to 0 and the divisor becomes the real seat count again.
# Both arms are exercised here.
#
# Prompt mix and sizing: benchmarks/results/ornith_ada_multi_agent_scheduler_2026-08-31.md
# (four concurrent agents; 1,024/1,536 group, 2,048/4,096 serialize) and
# ornith_ada_prefill_chunk_2026-08-31.md (the 32,768-token cold gate). Four agents, a
# 4,096-token chunk (fork/main's Ada auto-resolution) and a 65,536-token pool are that
# report's own configuration, so this profile measures the tree at the geometry the Ada
# numbers were taken at rather than at the Switchyard soak's.
SCENARIOS_ORNITH = [
    ("agent-1k",       1_024, 25),
    ("agent-1k5",      1_536, 25),
    ("agent-2k",       2_048, 25),
    ("agent-4k",       4_096, 15),
    ("agent-32k",     32_768, 10),
]
# The "trace" profile: not a hand-written geometry at all, but one derived from a captured
# server trace by benchmarks/trace_to_profile.py (--profile-file). It exists so this gate
# can be run against the traffic shape that actually occurred rather than against the soak's
# synthetic mix -- the five profiles above encode failures we have already seen, and a trace
# encodes the load we have not yet reasoned about. Installed by apply_profile_file(), which
# also overrides the module-level constants below that a profile file carries.
TRACE_PROFILE: dict | None = None
#: Bumped when the profile-file schema changes meaning; trace_to_profile.py stamps it.
PROFILE_VERSION = 1

# Per-profile overrides of the module-level Switchyard constants. Absent keys keep them.
PROFILE_KNOBS = {
    # Model the restore half of the session cycle (see SCENARIOS_RESTORE). Off everywhere
    # else so the four profiles whose floors were measured against live-soak evidence stay
    # bit-identical: a reclaim still records its checkpoint, but nothing ever reads it.
    "switchyard-restore": {
        "restore_spilled": True,
    },
    "ornith-ada": {
        "agents": 4,                         # --max-running-requests on the Ada runs
        "prefill_budget": 4_096,             # Engine._resolve_auto_prefill_chunk on sm_89
        "pool_pages": 65_536,                # --num-tokens
        "max_batch_seqs": 1,                 # _resolve_max_prefill_seqs for growable GGUF
        "small_prompt_group_tokens": 1_536,  # _auto_small_prompt_group_tokens on sm_89
    },
}
# Session geometry, from the soak driver (S1) and the server's session defaults.
SESSION_TTL = 300.0       # UserMsg.session_ttl_seconds default (scheduler.py:875)
CLIENT_TIMEOUT = 600.0    # switchyard soak --request-timeout; a starved request is abandoned
TURN_GROWTH = 512         # tokens a conversation gains per turn on top of its own output
STALL_POLL = 0.01         # scheduler poll quantum: a stalled loop still burns wall clock
FAMILIES = 4
REUSE_FRAC = 0.75  # stage route measured 73.6% prefix reuse pre-fix


def apply_profile_file(path: str) -> str:
    """Install a trace-derived profile under the name ``trace``; return that name.

    The module's constants (OUTPUT_LEN, TURN_GROWTH, FAMILIES, WIDTH, the two timeouts) are
    read as globals by ``Traffic``, ``build_pool`` and ``run``, so a file-driven profile
    installs itself by rebinding them rather than by threading a spec object through every
    call site -- which keeps the five hand-written profiles bit-identical to what their
    recorded floors were measured under.

    WIDTH is raised, never lowered below what the scenarios need: ``Req`` asserts
    ``max_device_len <=`` the page-table row width, so a trace with longer contexts than
    the Switchyard soak's would otherwise assert deep inside the scheduler.
    """
    global TRACE_PROFILE, OUTPUT_LEN, TURN_GROWTH, FAMILIES, CLIENT_TIMEOUT
    global SESSION_TTL, WIDTH
    with open(path, encoding="utf-8") as fh:
        spec = json.load(fh)
    if spec.get("profile_version") != PROFILE_VERSION:
        raise SystemExit(
            f"{path}: profile_version {spec.get('profile_version')!r}, expected "
            f"{PROFILE_VERSION}; regenerate it with benchmarks/trace_to_profile.py")
    if not spec.get("scenarios"):
        raise SystemExit(f"{path}: no scenarios")
    TRACE_PROFILE = spec
    OUTPUT_LEN = max(1, int(spec.get("output_len", OUTPUT_LEN)))
    TURN_GROWTH = max(0, int(spec.get("turn_growth", TURN_GROWTH)))
    FAMILIES = max(1, int(spec.get("families", FAMILIES)))
    CLIENT_TIMEOUT = float(spec.get("client_timeout", CLIENT_TIMEOUT))
    SESSION_TTL = float(spec.get("session_ttl", SESSION_TTL))
    longest = max(int(row[1]) for row in spec["scenarios"])
    WIDTH = max(int(spec.get("width", WIDTH)), longest + OUTPUT_LEN + TURN_GROWTH + 1)
    PROFILE_KNOBS["trace"] = dict(spec.get("knobs") or {})
    return "trace"


def build_pool(pool_pages: int = POOL, interleave: bool = True,
               max_running: int = MAX_RUNNING, max_batch_seqs: int = 0,
               small_prompt_group_tokens: int = 0):
    from freetoken.core import Context, get_global_ctx, set_global_ctx
    try:
        get_global_ctx()
    except Exception:
        set_global_ctx(Context(page_size=PAGE_SIZE))

    from freetoken.scheduler.cache import CacheManager
    from freetoken.scheduler.decode import DecodeManager
    from freetoken.scheduler.prefill import PrefillManager
    from freetoken.scheduler.table import TableManager

    pt = torch.zeros((max_running + 1, WIDTH), dtype=torch.int32, device="cpu")
    kw = {}
    sig = inspect.signature(CacheManager.__init__).parameters
    grows = "committed_pages" in sig
    # Only start under-committed when this tree also has the growth trigger the scheduler
    # drives (committed_pages_required); otherwise the pool could never grow and the
    # comparison would measure the harness, not the tree.
    if grows and hasattr(CacheManager, "committed_pages_required"):
        kw["committed_pages"] = min(GROW_STEP, pool_pages)
    cm = CacheManager(num_pages=pool_pages, page_size=PAGE_SIZE, page_table=pt,
                      type="radix", **kw)
    tm = TableManager(max_running_reqs=max_running, page_table=pt)
    dm = DecodeManager(page_size=PAGE_SIZE)
    pm = PrefillManager(cm, tm, dm)
    caps = {"committed_pages": grows,
            "interleave_chunks": hasattr(pm, "interleave_chunks"),
            "max_batch_seqs": hasattr(pm, "max_batch_seqs"),
            "small_prompt_group_tokens": hasattr(pm, "small_prompt_group_tokens"),
            "committed_pages_required": hasattr(cm, "committed_pages_required"),
            # Does this tree charge a cold session restore against the finishability
            # reservation the admission gate proved? (soak §W6; absent before the fix)
            "restore_charged": hasattr(PrefillManager, "finishability_reservation")}
    if caps["interleave_chunks"]:
        pm.interleave_chunks = interleave   # growable multi-agent mode, as the soak ran
    # The GGUF lane cap and its short-prompt escape hatch; both absent on older trees, and
    # both no-ops at their defaults, so every existing profile stays bit-identical.
    if caps["max_batch_seqs"]:
        pm.max_batch_seqs = max_batch_seqs
    if caps["small_prompt_group_tokens"]:
        pm.small_prompt_group_tokens = small_prompt_group_tokens
    return cm, tm, dm, pm, caps


class Traffic:
    """Deterministic stage-route generator with per-family shared prefixes.

    ``sessions=True`` (the switchyard profile) switches to the soak's real client shape: a
    slot is one persistent Switchyard *session* running one sticky scenario, and each turn's
    prompt is the previous turn's prompt plus its output plus a little new text. Turn k+1's
    prompt is therefore a strict extension of what turn k left in the tree, which is what
    makes the session lease worth retaining -- and what makes sixteen of them hold the pool.
    """

    def __init__(self, seed: int, profile: str = "stage"):
        self.rng = random.Random(seed)
        self.uid = 0
        table = {"pressure": SCENARIOS_PRESSURE, "fanout": SCENARIOS_FANOUT,
                 "switchyard-stage": SCENARIOS_SWITCHYARD,
                 "switchyard-deadlock": SCENARIOS_DEADLOCK,
                 "switchyard-restore": SCENARIOS_RESTORE,
                 "ornith-ada": SCENARIOS_ORNITH}.get(profile, SCENARIOS)
        spec = TRACE_PROFILE if profile == "trace" else None
        if spec is not None:
            table = [(str(n), int(length), int(w)) for n, length, w in spec["scenarios"]]
        self.sessions = profile.startswith("switchyard") or bool(
            spec and spec.get("sessions"))
        self.reuse = 0.0 if profile == "fanout" else float(
            spec.get("reuse", REUSE_FRAC) if spec else REUSE_FRAC)
        self.jitter = (
            tuple(spec["jitter"]) if spec and spec.get("jitter")
            else ((0.95, 1.05) if profile == "fanout" else (0.8, 1.2))
        )
        # Turns per conversation before a client rotates to a new one. Fixed at 2-5 for the
        # hand-written profiles; a trace measures it.
        self.turns = tuple(spec["turns"]) if spec and spec.get("turns") else (2, 5)
        self.pop = [s for s in table for _ in range(s[2])]
        self.prefix = {
            f: torch.arange(f * 10_000_000, f * 10_000_000 + WIDTH, dtype=torch.int32)
            for f in range(FAMILIES)
        }
        self.tail = 900_000_000
        self.slot_state: dict[int, dict] = {}
        self.generation = 0

    def _session_slot(self, slot: int):
        """One conversation at a time per client; a conversation is a few growing turns.

        The session POPULATION is deliberately larger than the concurrency, as it is in the
        soak: sixteen clients are active at any instant, but each finishes its conversation
        after a handful of turns and starts a new one, so finished conversations accumulate
        as *idle* leases. That distinction is load-bearing -- the scheduler's demand reclaim
        (``_reclaim_soft_sessions_for_pending``) only ever releases an IDLE lease, so a model
        in which every session stays active forever has no reclaim path at all and stalls at
        every commit, which measures the harness rather than the tree.
        """
        cap = WIDTH - OUTPUT_LEN - 1          # Req asserts max_device_len <= the row width
        st = self.slot_state.get(slot)
        rotate = (
            st is None
            or st["turn"] >= st["turns"]
            or st["len"] + OUTPUT_LEN + TURN_GROWTH > cap
        )
        if rotate:
            name, length, _ = self.rng.choice(self.pop)
            length = max(64, int(length * self.rng.uniform(*self.jitter)))
            length = min(length, cap - OUTPUT_LEN - TURN_GROWTH)
            self.generation += 1
            st = {
                "name": name,
                "len": length,
                "turn": 0,
                "turns": self.rng.randint(*self.turns),
                "sid": f"sess-{slot}-{self.generation}",
                "fam": self.generation % FAMILIES,
                # The shared head is the family's system prompt / tool catalog: it is what
                # a NEW conversation matches in the tree. Without it every new session pays
                # full price for its prompt and the replay runs at ~0% prefix reuse, against
                # the 74-84% the soak measured -- which alone decides whether the pool can
                # keep up, independently of any admission policy.
                "shared": int(length * self.reuse),
                "base": self.generation * 2_000_000,
            }
            self.slot_state[slot] = st
        else:
            st["len"] += OUTPUT_LEN + TURN_GROWTH
        st["turn"] += 1
        shared = st["shared"]
        uniq = st["len"] - shared
        ids = torch.cat((
            self.prefix[st["fam"]][:shared],
            torch.arange(st["base"], st["base"] + uniq, dtype=torch.int32),
        ))
        return st["name"], ids, st["sid"]

    def next(self, slot: int):
        if self.sessions:
            name, ids, sid = self._session_slot(slot)
            self.uid += 1
            return self.uid, name, ids, sid
        name, length, _ = self.rng.choice(self.pop)
        length = max(64, int(length * self.rng.uniform(*self.jitter)))
        length = min(length, WIDTH - OUTPUT_LEN - 1)
        fam = slot % FAMILIES
        shared = int(length * self.reuse)
        uniq = length - shared
        ids = torch.cat((
            self.prefix[fam][:shared],
            torch.arange(self.tail, self.tail + uniq, dtype=torch.int32),
        ))
        self.tail += uniq
        self.uid += 1
        return self.uid, name, ids, None


class _Msg:
    """Minimal UserMsg stand-in for PrefillManager.add_one_req."""

    def __init__(self, uid, ids, session_id=None):
        from freetoken.core import SamplingParams
        self.uid = uid
        self.input_ids = ids
        self.sampling_params = SamplingParams(max_tokens=OUTPUT_LEN)
        self.mm_embeds = None
        self.session_id = session_id
        self.session_ttl_seconds = SESSION_TTL if session_id else None
        self.hidden_states = None
        self.no_prefix_cache = False


class _Lease:
    """``Scheduler.SessionLease``, cut down to the fields the KV accounting depends on.

    ``handle`` is the locked radix handle ``retain_prefix`` returned: while it is held the
    session's prefix is *protected*, so it counts against neither ``free_slots`` nor
    ``evictable_size`` and is invisible to ``available_size`` and to eviction.

    ``spill`` is the other half of the lifecycle: ``_release_soft_session_handle``
    checkpoints the prefix (``_spill_soft_session``) *before* it unlocks, so a reclaimed
    lease is a debt the pool owes back the moment the session's next turn arrives, not a
    loss. It holds the checkpoint's token count; 0 means nothing is on disk.
    """

    __slots__ = ("handle", "expires_at", "active_uid", "last_used_at", "spill")

    def __init__(self):
        self.handle = None
        self.expires_at = None
        self.active_uid = None
        self.last_used_at = 0.0
        self.spill = 0


def install_match_counter(cm):
    """Count admission prefix walks -- the wall-clock proxy for a refused pass.

    A refused pass runs a full ``match_req`` per fresh candidate and forwards nothing, so
    during the soak's stalls the whole CPU budget went into 118K-token radix walks
    (soak report S5: ``fast_compare_key`` on four of five py-spy samples). ``match_prefix``
    itself is O(matched_len), but the cost that matters is the one the scheduler *repeats*:
    tokens offered to the matcher per pass. Returns a dict updated in place.
    """
    stats = {"calls": 0, "tokens": 0, "per_pass": []}
    orig = cm.match_req

    def wrapper(req):
        stats["calls"] += 1
        stats["tokens"] += int(req.input_len)
        return orig(req)

    cm.match_req = wrapper
    return stats


def install_diagnostics(pm=None):
    """Record why each prefill pass stopped admitting.

    ``schedule_next_batch`` breaks at the first refusal, so a pass has at most one.
    Returns (counters, restore_fn).
    """
    from freetoken.scheduler import prefill as P

    A = P.PrefillAdder
    orig_try = A.try_add_one
    orig_alloc = A._try_allocate_one
    takes_limit = "chunk_limit" in inspect.signature(orig_try).parameters
    stop = {"budget_exhausted": 0, "fresh_table_slot": 0, "fresh_admit_gate": 0,
            "fresh_chunk_cap": 0, "continuation_chunk_cap": 0}
    # Counterfactual for the fresh-admit gate: how much headroom was there, and how much
    # would ONE interleave chunk have cost, at each refusal.
    gate_evidence = []
    limits = []
    # Head-of-line blocking: schedule_next_batch ``break``s at the first refusal, so every
    # queued request BEHIND the refused one is skipped for the whole pass. Count how many of
    # those the pools could actually have seated.
    skipped_total = []
    skipped_admissible = []

    def alloc_wrap(self, req, *a, **k):
        self._diag_table = self.table_manager.available_size == 0
        r = orig_alloc(self, req, *a, **k)
        self._diag_alloc_none = r is None
        return r

    def try_wrap(self, pending_req, chunk_limit=None):
        cont = pending_req.chunked_req is not None
        self._diag_alloc_none = False
        self._diag_table = False
        if chunk_limit is not None:
            limits.append(int(chunk_limit))
        if self.token_budget <= 0:
            stop["budget_exhausted"] += 1
            return None
        r = orig_try(self, pending_req, chunk_limit=chunk_limit) if takes_limit \
            else orig_try(self, pending_req)
        if r is None:
            if cont:
                stop["continuation_chunk_cap"] += 1
            elif self._diag_table:
                stop["fresh_table_slot"] += 1
            elif self._diag_alloc_none:
                stop["fresh_admit_gate"] += 1
                if pm is not None:
                    q = pm.pending_list
                    try:
                        i = next(k for k, x in enumerate(q) if x is pending_req)
                    except StopIteration:
                        i = len(q) - 1
                    behind = q[i + 1:]
                    skipped_total.append(len(behind))
                    av = self.cache_manager.available_size
                    rs = self.reserved_size
                    n = 0
                    for x in behind:
                        if x.chunked_req is not None:
                            n += 1  # a continuation is never checked against the pool at all
                        elif x.input_len + x.output_len + rs <= av:
                            n += 1
                    skipped_admissible.append(n)
                gate_evidence.append((
                    int(pending_req.input_len),
                    int(self.cache_manager.available_size),
                    int(self.reserved_size),
                    int(chunk_limit or self.token_budget),
                ))
            else:
                stop["fresh_chunk_cap"] += 1
        return r

    A.try_add_one = try_wrap
    A._try_allocate_one = alloc_wrap

    def restore():
        A.try_add_one = orig_try
        A._try_allocate_one = orig_alloc

    return (stop, gate_evidence, limits, skipped_total, skipped_admissible), restore


def run(ticks: int, seed: int, verbose: bool = False, profile: str = "stage",
        diagnose: bool = False, pool_pages: int = POOL,
        interleave: bool = True, abort_inflight: bool | None = None):
    # See ``abandon_starved``. False is what the real server does; True is a deliberate
    # simplification that keeps a starved run measurable (the throughput/error-rate
    # profiles need the run to continue past a stall) and it is what the three original
    # profiles' recorded floors were measured under.
    if abort_inflight is None:
        abort_inflight = profile != "switchyard-deadlock"
    knobs = PROFILE_KNOBS.get(profile, {})
    max_running = knobs.get("agents", MAX_RUNNING)
    prefill_budget = knobs.get("prefill_budget", PREFILL_BUDGET)
    # An explicit --pool still wins; the profile's own pool applies only at the default.
    if pool_pages == POOL:
        pool_pages = knobs.get("pool_pages", POOL)
    cm, tm, dm, pm, caps = build_pool(
        pool_pages,
        interleave,
        max_running=max_running,
        max_batch_seqs=knobs.get("max_batch_seqs", 0),
        small_prompt_group_tokens=knobs.get("small_prompt_group_tokens", 0),
    )
    traffic = Traffic(seed, profile)
    match_stats = install_match_counter(cm)
    stop = gate_evidence = None
    limits = []
    skipped_total = []
    skipped_admissible = []

    def restore():
        return None
    if diagnose:
        (stop, gate_evidence, limits, skipped_total,
         skipped_admissible), restore = install_diagnostics(pm)

    # Synthetic clock so waits are comparable with the soak's p95 (R3/R4):
    # prefill ran at ~1,800 tok/s instant, decode at ~160 tok/s aggregate over 16 lanes.
    PREFILL_TPS = 1_800.0
    DECODE_SEC = 0.10

    state = {"clock": 0.0}
    submitted = {}     # uid -> submit time
    outstanding = {}   # uid -> submit time, for the uids still in flight (<= MAX_RUNNING)
    first_chunk = {}   # uid -> wait to first forwarded chunk
    finished = {}      # uid -> total latency
    scenario = {}      # uid -> name
    slot_of = {}
    free_slots = list(range(max_running))

    lanes_hist, util_hist, newtok_hist = [], [], []
    margin_prefill, margin_decode = [], []

    def alloc_margin(reqs):
        """allocatable - needed for this batch: how far it sits from the
        ``committed_pages_required`` fatal (negative would raise)."""
        from freetoken.utils import div_ceil
        needed = sum(div_ceil(r.device_len, PAGE_SIZE) - div_ceil(r.cached_len, PAGE_SIZE)
                     for r in reqs)
        pc = cm.prefix_cache
        ev = (pc.full_evictable_size if (cm.is_hybrid or cm.is_swa)
              else pc.size_info.evictable_size)
        allocatable = len(cm.free_slots) + ev // PAGE_SIZE
        allocatable += getattr(cm, "num_pages", 0) - getattr(cm, "committed_pages",
                                                             getattr(cm, "num_pages", 0))
        return allocatable - needed
    empty_prefill_passes = 0
    decode_batches = 0
    fatal = None
    tick = 0

    # ---- session residency (the piece the pre-existing profiles do not model) ----
    leases: dict[str, _Lease] = {}
    session_of: dict[int, str] = {}
    timed_out: set[int] = set()
    # Requests the client gave up on that the SERVER never learned about (see
    # abandon_starved): still in the pending list, still holding their forwarded pages.
    abandoned_inflight: set[int] = set()
    stalls: list[float] = []
    stall_usage: list[float] = []
    stall_total = 0.0
    reclaims = 0
    expiries = 0
    # Session checkpoint traffic (``/v1/stats.scheduler.session_spill``): every reclaim
    # spills, and ``restore_spilled`` profiles take the tokens back on the session's next
    # turn. ``restores_deferred`` is the fix under test refusing to do so.
    spills = 0
    restores = 0
    restores_deferred = 0
    restores_failed = 0
    restored_tokens = 0
    restore_spilled = bool(knobs.get("restore_spilled", False))
    pool_tokens = pool_pages * PAGE_SIZE
    # ---- finishability invariant of the ADMITTED SET (soak report T5) ----
    #
    #   owed = SUM over in-flight chunked prefills of (input_len - forwarded) + output_len
    #        + DecodeManager.inflight_tokens                (the running decodes' remainder)
    #   owed <= cache_manager.available_size + reclaimable idle lease tokens
    #
    # The right-hand side is every token admission can still OBTAIN: what is free or
    # evictable now, plus the idle session leases demand reclaim would release. A violation
    # says the requests already admitted cannot all be driven to completion even if every
    # reclaimable thing in the pool is handed to them -- and since the chunked-prefill
    # scheduler advances all lanes together, nothing completes to break the tie. That is the
    # deadlock precondition, and it is what "a budget checked only at admission" cannot see:
    # each admit passed its own check against a capacity the next admit then spent again.
    #
    # Live (T5): 14 chunked prefills had forwarded 237,819 tokens of a 262,144-token pool and
    # still owed 222,538 -- a 1.76x peak footprint -- with #running-req 0 and nothing
    # reclaimable. Every continuation deferred at kv_pages == 0, forever.
    owed_samples: list[int] = []
    slack_samples: list[int] = []
    invariant_violations = 0
    last_batch_clock = 0.0
    deadlock = False
    deadlock_state = None

    def admitted_owed() -> int:
        owed = 0
        for pending in pm.pending_list:
            chunked = pending.chunked_req
            if chunked is not None:
                owed += max(0, pending.input_len - chunked.cached_len) + pending.output_len
        return owed + dm.inflight_tokens

    def obtainable() -> int:
        return cm.available_size + reclaimable_tokens()

    def reclaimable_tokens():
        """``Scheduler._reclaimable_session_tokens``: locked, but buyable on demand."""
        return sum(int(x.handle.cached_len) for x in leases.values()
                   if x.handle is not None and x.active_uid is None)

    if hasattr(type(cm), "reclaimable_tokens_hook"):
        cm.reclaimable_tokens_hook = reclaimable_tokens

    def retain_session(req):
        """``_free_req_resources(retain_session=True)``: donate, then LOCK the prefix."""
        sid = session_of.get(req.uid)
        if sid is None:
            return
        lease = leases[sid]
        if lease.active_uid != req.uid:
            return
        handle = cm.retain_prefix(req.input_ids, req.cached_len)
        old, lease.handle = lease.handle, handle
        lease.active_uid = None
        lease.expires_at = state["clock"] + SESSION_TTL
        lease.last_used_at = state["clock"]
        if old is not None:
            cm.unlock(old)

    def release_lease(lease):
        """``Scheduler._release_soft_session_handle``: CHECKPOINT, then unlock.

        The real reclaim never simply drops a lease -- ``_spill_soft_session`` writes the
        prefix to the spill store first, so the tokens it hands back to the pool are a debt
        the session reclaims the moment its next turn arrives. Recording that debt is what
        lets ``restore_session`` model the other half of the cycle; profiles without
        ``restore_spilled`` only ever read the count.
        """
        nonlocal spills
        if lease.handle is not None:
            lease.spill = int(lease.handle.cached_len)
            spills += 1
        cm.unlock(lease.handle)
        lease.handle = None
        lease.expires_at = None

    def restore_session(lease, ids):
        """``Scheduler._restore_cold_session`` -> ``CacheManager.restore_hybrid_session_prefix``.

        The half of the session cycle the replay was missing, and the mechanism behind the
        nine ``finishability invariant`` warnings of soak §W6.

        While a prefix sits on disk it costs the pool nothing: ``available_size`` counts its
        tokens as free or evictable and the admission gate sells them to chunked prefills.
        Materialising it takes them straight back -- ALLOCATED AND LOCKED, from the message
        path (``scheduler.py``'s ``_process_one_msg``, before ``add_one_req``) and from
        ``_reclaim_for_blocked_prefill``, neither of which is an admission gate. Nothing
        re-checks the proof the gate already made, so ``owed`` is unchanged while
        ``available_size`` drops by the restore's whole footprint, and the invariant goes
        negative by exactly that much -- once -- and then tracks the admitted set in
        lockstep as it drains. Live: a constant 1,401-token shortfall over nine passes,
        with four ``Restored cold session`` lines (one of 79,104 tokens) in the two seconds
        before the first.

        The page arithmetic mirrors ``restore_hybrid_session_prefix``: lock what is already
        resident so ``_allocate`` cannot evict it, allocate only the missing tail, insert,
        return whatever the tree had gained meanwhile, then lock the result.
        """
        nonlocal restores, restores_deferred, restores_failed, restored_tokens
        # ``length <= len(ids) - 1``: the final prompt token must still run through prefill.
        length = min(int(lease.spill), max(0, len(ids) - 1))
        if length <= 0:
            lease.spill = 0
            return
        tokens = ids[:length]
        matched = cm.prefix_cache.match_prefix(tokens).cuda_handle
        resident = int(matched.cached_len)
        missing = length - resident
        # What the restore takes out of ``available_size``: the tail it must allocate PLUS
        # the resident-but-evictable part it is about to lock. Re-locking an evictable
        # prefix costs the pool exactly as much as allocating it -- ``available_size``
        # counts ``evictable_size`` and ``lock`` moves those tokens into ``protected_size``
        # -- which is why a restore can over-promise the pool without allocating a single
        # new page. Computed by the manager itself, so the replay charges a restore exactly
        # what ``_restore_cold_session`` charges it.
        footprint = getattr(cm, "session_restore_footprint", None)
        cost = footprint(tokens) if footprint is not None else length
        # THE FIX UNDER TEST. ``PrefillManager.finishability_reservation()`` is the exact
        # left-hand side of ``_check_finishability``; charging the restore against it is
        # what stops a restore from retroactively invalidating a finishability the
        # admission gate already proved. A restore that does not fit is DEFERRED with its
        # checkpoint intact -- reuse is an optimization, so recomputing the prefix is
        # always correct, and the next turn (or ``_reclaim_for_blocked_prefill``'s retry
        # after a release) picks it up once the reservation has drained. The attribute is
        # absent on trees before the fix, and that is what reproduces the soak's warnings.
        reserve = getattr(pm, "finishability_reservation", None)
        if reserve is not None and cost > cm.available_size - reserve():
            restores_deferred += 1
            return
        try:
            if missing > 0:
                # ``_restore_cold_session`` grows the committed pool first when free +
                # evictable cannot cover the restore (Engine.grow_runtime_kv).
                if caps["committed_pages"]:
                    allocatable = (len(cm.free_slots)
                                   + cm.prefix_cache.size_info.evictable_size)
                    if missing > allocatable:
                        required = cm.committed_pages + missing - allocatable
                        new_total = min(pool_pages,
                                        int(math.ceil(required / GROW_STEP)) * GROW_STEP)
                        cm.add_committed_pages(new_total)
                if resident:
                    cm.lock(matched)          # inc_lock: do not evict what we just matched
                allocated = cm._allocate(missing)
                all_pages = (torch.cat((matched.get_matched_indices(), allocated))
                             if resident else allocated)
                cached_len, _inserted = cm.prefix_cache.insert_prefix(tokens, all_pages)
                cm._free(allocated[: max(0, cached_len - resident)])
                if resident:
                    cm.unlock(matched)
            lease.handle = cm.retain_prefix(ids, length)
        except Exception:  # noqa: BLE001 -- reuse is an optimization, never an admission gate
            restores_failed += 1
            return                            # keep the checkpoint; retried on the next turn
        lease.spill = 0
        lease.expires_at = None
        restores += 1
        restored_tokens += cost

    def expire_leases():
        """``Scheduler._expire_sessions``: an IDLE lease past its TTL gives its KV back."""
        nonlocal expiries
        now = state["clock"]
        for lease in leases.values():
            if (lease.handle is not None and lease.active_uid is None
                    and lease.expires_at is not None and lease.expires_at <= now):
                release_lease(lease)
                expiries += 1

    def reclaim_for_blocked_prefill():
        """``Scheduler._reclaim_for_blocked_prefill`` -> ``_reclaim_soft_sessions_for_pending``.

        LRU over leases that are resident AND idle. A session whose request is queued or
        decoding is ACTIVE (``active_uid is not None``) and is deliberately skipped -- which
        is why a closed loop of sixteen busy clients can leave this returning False with the
        pool at 1.00, exactly as the soak's log shows.
        """
        if not pm.runnable:
            return False
        best = None
        for lease in leases.values():
            if lease.handle is None or lease.active_uid is not None:
                continue
            if best is None or lease.last_used_at < best.last_used_at:
                best = lease
        if best is None:
            return False
        release_lease(best)
        return True

    def reclaim_for_admission(pending, sid):
        """``Scheduler._reclaim_soft_sessions_for_admission``, run on every arriving message.

        This is the *main* pressure valve -- the soak logged 749 of these in 50 minutes
        against a handful from the blocked-prefill path -- so a replay that models only the
        blocked-prefill reclaim leaves the pool permanently over-subscribed and stalls at
        every commit. Releases LRU idle leases (never this request's own) until the pool can
        seat what this admission still needs.
        """
        nonlocal reclaims
        candidates = sorted(
            ((lease.last_used_at, key) for key, lease in leases.items()
             if key != sid and lease.active_uid is None and lease.handle is not None),
            key=lambda item: item[0],
        )
        if not candidates:  # cheap gate: skip the prefix match when nothing can be freed
            return
        cached = cm.match_req(pending).cuda_handle.cached_len
        needed = max(0, pending.input_len - cached) + pending.output_len
        for _last_used, key in candidates:
            if needed <= cm.available_size:
                break
            release_lease(leases[key])
            reclaims += 1

    def next_release_time():
        """When the outside world next frees something: an idle TTL, or a client giving up."""
        times = [lease.expires_at for lease in leases.values()
                 if lease.handle is not None and lease.active_uid is None
                 and lease.expires_at is not None]
        times += [t + CLIENT_TIMEOUT for t in outstanding.values()]
        return min(times) if times else None

    def abandon_starved():
        """The soak's 600 s client timeout: what turns a stall into an error rate.

        A client giving up does **not** in general give the engine its KV back, and getting
        that wrong is what hid soak report T from this replay for two rewrites. FastAPI's
        disconnect detection lives in ``ApiServer.stream_with_cancellation``, which checks
        ``request.is_disconnected()`` only *after* the response generator has yielded a
        chunk (python/freetoken/server/api_server.py:419). A request that is still in
        prefill has yielded nothing, so the loop is parked on the ack queue, the disconnect
        is never observed, no ``AbortMsg`` is ever sent, and the scheduler keeps the
        request -- its pending-list entry, its table slot and every KV page it has already
        forwarded -- for the rest of the server's life.

        Modelling the timeout as a free abort gave every deadlock an escape hatch the real
        server does not have: the replay healed after 600 s and reported an error rate,
        where the soak reported 2,616 s of unbroken silence to the end of the run.
        """
        now = state["clock"]
        for uid, t in list(outstanding.items()):
            if now - t < CLIENT_TIMEOUT:
                continue
            timed_out.add(uid)
            outstanding.pop(uid, None)
            pending = next((p for p in pm.pending_list if p.uid == uid), None)
            if (pending is not None and not abort_inflight
                    and pending.chunked_req is not None):
                # Mid-prefill and holding forwarded pages, and it has streamed nothing, so
                # the server never learns the client is gone. Count the client-side error
                # and leave the engine holding the KV -- for good.
                #
                # Scoped to a request that has actually forwarded a chunk: one still purely
                # queued holds no KV at all, so whether the model keeps it or drops it
                # cannot change the pool, and dropping it is what keeps the closed
                # 16-client loop from silting up with entries that are only an artefact of
                # the slot abstraction.
                abandoned_inflight.add(uid)
            elif pending is not None:
                chunked = pm.abort_req(uid)
                if chunked is not None:
                    cm.cache_req(chunked, finished=True)
                    tm.free(chunked.table_idx)
            else:
                # Already decoding, so it has streamed: the next yield sees the
                # disconnect and aborts for real.
                req = dm.abort_req(uid)
                if req is not None:
                    cm.cache_req(req, finished=True)
                    tm.free(req.table_idx)
            sid = session_of.get(uid)
            if (sid is not None and leases[sid].active_uid == uid
                    and uid not in abandoned_inflight):
                # The turn is gone, so the lease goes idle and starts ageing out. Its
                # retained KV is NOT released here -- that is what _expire_sessions is for.
                #
                # A request the server never learned about is the exception: its session is
                # still ACTIVE as far as the scheduler is concerned, so demand reclaim skips
                # it forever. Soak report T6 counted 112 idle expiries during the deadlock
                # and none of them freed anything, for exactly this reason.
                leases[sid].active_uid = None
                leases[sid].expires_at = now + SESSION_TTL
                leases[sid].last_used_at = now
            # The client itself moves on either way (closed loop), but admit_new() is
            # bounded by pending+running, so an abandoned-but-resident request keeps its
            # slot in practice.
            free_slots.append(slot_of[uid])

    def admit_new():
        while free_slots and len(pm.pending_list) + len(dm.running_reqs) < max_running:
            slot = free_slots.pop()
            uid, name, ids, sid = traffic.next(slot)
            slot_of[uid] = slot
            scenario[uid] = name
            submitted[uid] = state["clock"]
            outstanding[uid] = state["clock"]
            if sid is not None:
                lease = leases.setdefault(sid, _Lease())
                lease.active_uid = uid
                lease.expires_at = None
                lease.last_used_at = state["clock"]
                session_of[uid] = sid
            pm.add_one_req(_Msg(uid, ids, sid))
            if sid is not None:
                # scheduler.py's order on a UserMsg carrying a session id:
                # _reclaim_soft_sessions_for_admission (make room), then
                # _restore_cold_session (take the checkpoint back), then add_one_req.
                reclaim_for_admission(pm.pending_list[-1], sid)
                if restore_spilled and lease.handle is None and lease.spill:
                    restore_session(lease, ids)

    growable_decode_steps = 0
    try:
        admit_new()
        while tick < ticks:
            tick += 1
            owed = admitted_owed()
            slack = obtainable() - owed
            owed_samples.append(owed)
            slack_samples.append(slack)
            if slack < 0:
                invariant_violations += 1
            prefill_runnable = pm.runnable
            decode_runnable = dm.runnable
            batch = None
            if prefill_runnable and decode_runnable:
                if growable_decode_steps < DECODE_BURST:
                    batch = dm.schedule_next_batch()
                    growable_decode_steps += 1
                else:
                    batch = pm.schedule_next_batch(prefill_budget)
                    growable_decode_steps = 0
                    if batch is None:
                        empty_prefill_passes += 1
                        batch = dm.schedule_next_batch()
            elif prefill_runnable:
                batch = pm.schedule_next_batch(prefill_budget)
                growable_decode_steps = 0
                if batch is None:
                    empty_prefill_passes += 1
            elif decode_runnable:
                batch = dm.schedule_next_batch()
            if batch is None:
                if not pm.runnable and not dm.runnable:
                    admit_new()
                    if not pm.runnable:
                        break
                    continue
                # The scheduler's escape ladder, in the order scheduler.py runs it.
                if reclaim_for_blocked_prefill():
                    reclaims += 1
                    continue
                # Nothing reclaimable: the pool is held by leases whose sessions are ACTIVE
                # and by requests that cannot buy their next chunk. The server spins here
                # (py-spy: a full radix walk per pending prompt, forever) and only two
                # outside events end it -- an idle TTL, or a client giving up. Jump to
                # whichever is first and charge the gap as stall time. This is the 492 s /
                # 515 s / 624 s silence of soak report S5, and it is the whole signature.
                nxt = next_release_time()
                if nxt is None:
                    # Permanent: work is outstanding, nothing is schedulable, and no
                    # outside event (idle TTL, client giving up on a request the server
                    # can still see) will ever release a token. This is soak report T --
                    # the last batch line of the run, followed by silence to the end.
                    deadlock = True
                    lanes = [(max(0, q.input_len - q.chunked_req.cached_len)
                              + q.output_len)
                             for q in pm.pending_list if q.chunked_req is not None]
                    deadlock_state = {
                        "chunked_lanes": len(lanes),
                        "owed": admitted_owed(),
                        "owed_min_lane": min(lanes) if lanes else None,
                        "forwarded": sum(q.chunked_req.cached_len
                                         for q in pm.pending_list
                                         if q.chunked_req is not None),
                        "available": cm.available_size,
                        "reclaimable": reclaimable_tokens(),
                        "running": len(dm.running_reqs),
                        "queued_fresh": sum(1 for q in pm.pending_list
                                            if q.chunked_req is None),
                    }
                    fatal = "DEADLOCK: no batch schedulable and nothing left to release"
                    break
                if verbose and len(stalls) < 8:
                    held = sum(int(x.handle.cached_len) for x in leases.values()
                               if x.handle is not None)
                    print("STALL#%d clock=%.1f avail=%d free=%d evictable=%d leases=%d "
                          "held=%d run=%d queue=%d chunked=%d" % (
                              len(stalls), state["clock"], cm.available_size,
                              len(cm.free_slots),
                              cm.prefix_cache.size_info.evictable_size,
                              sum(1 for x in leases.values() if x.handle is not None),
                              held, len(dm.running_reqs), len(pm.pending_list),
                              sum(1 for q in pm.pending_list
                                  if q.chunked_req is not None)),
                          file=sys.stderr)
                nxt = max(nxt, state["clock"] + STALL_POLL)
                stall = max(0.0, nxt - state["clock"])
                if stall > 0:
                    stalls.append(stall)
                    stall_total += stall
                    stall_usage.append(
                        round(1.0 - cm.available_size / pool_tokens, 4))
                state["clock"] = nxt
                expire_leases()
                abandon_starved()
                admit_new()
                continue

            # A batch line in the server log: the clock at the last one is what the
            # trailing-silence measurement below is taken against.
            last_batch_clock = state["clock"]
            # ---- _prepare_batch: growth check, then allocation ----
            if caps["committed_pages_required"]:
                required = cm.committed_pages_required(batch.reqs)
                if required > cm.committed_pages:
                    new_total = min(pool_pages, int(math.ceil(required / GROW_STEP)) * GROW_STEP)
                    cm.add_committed_pages(new_total)
            (margin_prefill if batch.is_prefill else margin_decode).append(
                alloc_margin(batch.reqs))
            cm.allocate_paged(batch.reqs)

            if batch.is_prefill:
                n = len(batch.reqs)
                new_tok = int(getattr(batch, "log_new_tokens", 0)) or sum(
                    r.extend_len for r in batch.reqs)
                lanes_hist.append(n)
                newtok_hist.append(new_tok)
                util_hist.append(new_tok / prefill_budget)
                state["clock"] += new_tok / PREFILL_TPS
                for r in batch.reqs:
                    if r.uid not in first_chunk:
                        first_chunk[r.uid] = state["clock"] - submitted[r.uid]
                    r.complete_one()
                dm.filter_reqs(batch.reqs)
            else:
                decode_batches += 1
                state["clock"] += DECODE_SEC
                for r in batch.reqs:
                    r.complete_one()
                done = [r for r in batch.reqs if r.remain_len <= 0]
                for r in done:
                    dm.remove_req(r)
                    cm.cache_req(r, finished=True)
                    retain_session(r)
                    tm.free(r.table_idx)
                    finished[r.uid] = state["clock"] - submitted[r.uid]
                    outstanding.pop(r.uid, None)
                    free_slots.append(slot_of[r.uid])
                if done:
                    admit_new()
            # The clock moved, so leases can age out and clients can give up on the normal
            # path too, not only out of a stall.
            if leases:
                expire_leases()
                abandon_starved()
            if verbose and state.get("probe") is None and cm.available_size < dm.inflight_tokens:
                state["probe"] = 1
                print("BREACH after %s batch n=%d newtok=%s avail=%d inflight=%d "
                      "free=%d committed=%d num=%d running=%d queue=%d chunked=%d" % (
                          batch.phase, len(batch.reqs),
                          getattr(batch, "log_new_tokens", None), cm.available_size,
                          dm.inflight_tokens, len(cm.free_slots), cm.committed_pages,
                          cm.num_pages, len(dm.running_reqs), len(pm.pending_list),
                          sum(1 for q in pm.pending_list if q.chunked_req is not None)),
                      file=sys.stderr)
            if verbose and tick % 500 == 0:
                print(f"  tick {tick} clock {state['clock']:7.1f}s "
                      f"queue {len(pm.pending_list)} run {len(dm.running_reqs)} "
                      f"done {len(finished)} avail {cm.available_size}", file=sys.stderr)
    except Exception as e:  # noqa: BLE001
        fatal = f"{type(e).__name__}: {e}"
        try:
            pc = cm.prefix_cache
            ev = (pc.full_evictable_size if (cm.is_hybrid or cm.is_swa)
                  else pc.size_info.evictable_size)
            print("FATAL phase=%s n=%d reqs=%s free=%d evictable=%d committed=%d num=%d "
                  "avail=%d inflight=%d running=%d queue=%d" % (
                      batch.phase, len(batch.reqs),
                      [(r.cached_len, r.device_len) for r in batch.reqs],
                      len(cm.free_slots), ev, cm.committed_pages, cm.num_pages,
                      cm.available_size, dm.inflight_tokens, len(dm.running_reqs),
                      len(pm.pending_list)),
                  file=sys.stderr)
        except Exception:  # noqa: BLE001
            pass
        traceback.print_exc()

    restore()
    clock = state["clock"]
    open_waits = {u: clock - t for u, t in submitted.items() if u not in first_chunk}

    def pct(xs, p):
        if not xs:
            return None
        xs = sorted(xs)
        k = min(len(xs) - 1, int(round((p / 100) * (len(xs) - 1))))
        return xs[k]

    lc_waits = [w for u, w in first_chunk.items() if scenario[u] == "long-context"]
    lc_done = [finished[u] for u in finished if scenario[u] == "long-context"]
    return {
        "caps": caps,
        "ticks": tick,
        "sim_seconds": round(clock, 1),
        "prefill_batches": len(lanes_hist),
        "decode_batches": decode_batches,
        "empty_prefill_passes": empty_prefill_passes,
        "lanes_mean": round(statistics.mean(lanes_hist), 3) if lanes_hist else None,
        "lanes_p50": pct(lanes_hist, 50),
        "lanes_max": max(lanes_hist) if lanes_hist else None,
        "single_lane_frac": round(sum(1 for n in lanes_hist if n == 1) / len(lanes_hist), 3)
        if lanes_hist else None,
        "util_mean": round(statistics.mean(util_hist), 4) if util_hist else None,
        "util_p50": round(pct(util_hist, 50), 4) if util_hist else None,
        "newtok_p50": pct(newtok_hist, 50),
        "prefilled_tokens": sum(newtok_hist),
        "admitted": len(submitted),
        "started": len(first_chunk),
        "completed": len(finished),
        "lc_completed": len(lc_done),
        "wait_first_chunk_p50": round(pct(list(first_chunk.values()), 50), 2) if first_chunk else None,
        "wait_first_chunk_p95": round(pct(list(first_chunk.values()), 95), 2) if first_chunk else None,
        "wait_first_chunk_max": round(max(first_chunk.values()), 2) if first_chunk else None,
        "lc_wait_max": round(max(lc_waits), 2) if lc_waits else None,
        "open_wait_max": round(max(open_waits.values()), 2) if open_waits else None,
        "latency_p95": round(pct(list(finished.values()), 95), 2) if finished else None,
        # ---- session residency / stall signature (switchyard profile) ----
        # ``stall_frac`` is the headline: soak report S5 measured 53% (stage) and 52%
        # (passthrough) of the wall clock with no batch emitted at all, at token usage 1.00.
        "stall_seconds": round(stall_total, 1),
        "stall_frac": round(stall_total / clock, 4) if clock > 0 else None,
        "stall_episodes": len(stalls),
        "stall_max": round(max(stalls), 1) if stalls else None,
        "stall_usage_p50": pct(stall_usage, 50),
        # ---- deadlock signature (soak report T) ----
        # ``stall_*`` measures silence BETWEEN two batches, and a deadlock produces none of
        # that: the silence starts after the LAST batch and never ends, so the soak's
        # gaps>=30s analyzer scored zero gaps on the run that never recovered. These three
        # are the trailing half of the same measurement.
        "deadlock": deadlock,
        "deadlock_state": deadlock_state,
        "trailing_silence": round(clock - last_batch_clock, 1),
        "trailing_silence_frac": (round((clock - last_batch_clock) / clock, 4)
                                  if clock > 0 else None),
        # ---- finishability invariant (see admitted_owed) ----
        "owed_max": max(owed_samples) if owed_samples else None,
        "owed_p95": pct(owed_samples, 95),
        "slack_min": min(slack_samples) if slack_samples else None,
        "slack_p05": pct(slack_samples, 5),
        "invariant_violations": invariant_violations,
        "invariant_violation_frac": (round(invariant_violations / len(owed_samples), 4)
                                     if owed_samples else None),
        "abandoned_inflight": len(abandoned_inflight),
        "timeouts": len(timed_out),
        "error_rate": (round(len(timed_out) / (len(finished) + len(timed_out)), 4)
                       if (finished or timed_out) else None),
        "lease_reclaims": reclaims,
        "lease_expiries": expiries,
        # ---- session checkpoint traffic (soak §W6) ----
        # ``restores_deferred`` is the fix: a restore charged against the standing
        # reservation and refused rather than allowed to invalidate it. On a tree without
        # ``PrefillManager.finishability_reservation`` it is 0 by construction and the
        # violations show up in ``invariant_violations`` instead.
        "session_spills": spills,
        "session_restores": restores,
        "session_restores_deferred": restores_deferred,
        "session_restores_failed": restores_failed,
        "session_restored_tokens": restored_tokens,
        "leases_resident": sum(1 for x in leases.values() if x.handle is not None),
        # ---- wall-clock proxy for a refused pass (soak report S5: O(queue x prompt)) ----
        "match_calls": match_stats["calls"],
        "match_tokens": match_stats["tokens"],
        "match_tokens_per_prefill_pass": (
            int(match_stats["tokens"] / (len(lanes_hist) + empty_prefill_passes))
            if (len(lanes_hist) + empty_prefill_passes) else None),
        "fatal": fatal,
        "stop_reasons": stop,
        # (prompt_len, available_size, reserved_size, chunk_limit) at each fresh-admit refusal:
        # how many of them a chunk-sized charge would have admitted.
        "gate_chunk_would_fit": (
            sum(1 for L, av, rs, cl in gate_evidence if cl + rs <= av)
            if gate_evidence else 0),
        "gate_refusals": len(gate_evidence) if gate_evidence is not None else 0,
        "gate_median_headroom": (
            int(statistics.median([av - rs for _, av, rs, _ in gate_evidence]))
            if gate_evidence else None),
        "gate_median_promptlen": (
            int(statistics.median([L for L, _, _, _ in gate_evidence]))
            if gate_evidence else None),
        "skipped_behind_p50": (int(statistics.median(skipped_total))
                               if skipped_total else None),
        "skipped_admissible_p50": (int(statistics.median(skipped_admissible))
                                   if skipped_admissible else None),
        "skipped_admissible_total": sum(skipped_admissible) if skipped_admissible else 0,
        "min_margin_prefill": min(margin_prefill) if margin_prefill else None,
        "min_margin_decode": min(margin_decode) if margin_decode else None,
        "p05_margin_prefill": pct(margin_prefill, 5) if margin_prefill else None,
        "chunk_limit_p50": int(statistics.median(limits)) if limits else None,
        "chunk_limit_min": min(limits) if limits else None,
        "chunk_limit_max": max(limits) if limits else None,
    }


# ---------------------------------------------------------------------------
# Regression gate
# ---------------------------------------------------------------------------
# Floors for --gate. Each is a *floor*, not the measurement: the measured value is in the
# comment and the floor sits ~5% under it, so ordinary scheduling jitter between
# Python/torch versions does not trip the gate while a real starvation regression (which
# halves throughput) does.
#
# Measured on the RESTORED d685e99 admission gate (fresh admits charged their whole
# remaining footprint against ``available_size``), seed 7, 20,000 forwards, torch CPU,
# Python 3.12; the run is deterministic:
#   stage            prefilled_tokens 2,814,602  completed 181
#   pressure         prefilled_tokens 5,000,774  completed  60
#   switchyard-stage prefilled_tokens 1,873,120  completed 219  error_rate 0.3578
#
# Re-measured after the chunk share was made to divide by the lanes the pass will SEAT
# instead of by the queue depth (soak §R7 ticket 1 / §U8; the divisor came from f3c3ac4).
# Same seed, ticks and interpreter; 797d23e -> this tree:
#   stage            2,756,687 -> 5,024,311  (+82%)  completed 183 -> 297  util 0.144 -> 0.851
#   pressure         5,000,774 -> 8,238,023  (+65%)  completed  60 ->  91  util 0.121 -> 0.917
#   switchyard-stage 1,995,035 -> 2,398,186  (+20%)  completed 247 -> 256  util 0.088 -> 0.620
#   switchyard-dead    903,157 -> 1,163,345  (+29%)  completed 118 -> 185  util 0.115 -> 0.707
# ``invariant_violations`` 0 and ``deadlock`` False on all four, before and after: the
# change moves no admission gate, it only stops under-spending the prefill budget on the
# lanes a pass has already decided to seat.
#
# d685e99 is the ONLY tree that has passed the live 16-way Switchyard soak
# (stage route: 471 req / 0 err / 1 STALLED). Two successive rewrites scored far higher
# here and failed live, so these floors are deliberately set to the *live-passing* tree
# and not to the best replay number ever recorded:
#   upstream bd372b6   stage 7,103,059 / 404   pressure  8,577,078 / 83
#   81ab30e (reverted) stage 7,049,549 / 373   pressure 10,071,808 / 99   -- soak FAIL
#   ea7ed7c (reverted) stage 6,194,304 / 375   pressure  8,094,693 / 96   -- soak DEADLOCK
# Both reverted trees beat this gate on raw throughput. Raw throughput on the two
# residency-free profiles is therefore NOT a sufficient gate, and neither is the
# switchyard profile's error rate on its own: what discriminates a deadlock is the
# ``deadlock_violations`` invariant below (the owed footprint of the admitted set).
#
GATE_TICKS = 20_000
GATE_SEED = 7
GATE_CASES = [
    # profile, min prefilled_tokens, min completed, max error_rate (None = not checked),
    # min session restores (None = the profile does not model them)
    ("stage",            4_773_000, 282, None, None),
    ("pressure",         7_826_000,  86, None, None),
    # The residency profile is graded on goodput AND on the soak's own acceptance metric.
    # The error ceiling moved 0.376 -> 0.410 with the chunk-share fix, and that is jitter,
    # not a regression: over seeds {1,3,5,7,11,13,17,23} the mean error rate is 0.288
    # before and 0.285 after, with the per-seed spread 0.214-0.338 before and 0.198-0.391
    # after -- seed 7 is simply this profile's worst seed on the new tree, while its
    # completions rise 263 -> 337 (+28%) on the same eight seeds. The ceiling tracks the
    # seed-7 measurement (+5%) like every other floor here; the SPREAD is what says whether
    # a future move is real, so re-run the sweep before touching this number again.
    ("switchyard-stage", 2_278_000, 243, 0.410, None),
    # The deadlock profile is the regression test for soak report T. Its floors come from
    # the shipped tree -- but the checks that matter on it are the two every case now
    # carries, ``deadlock`` and ``invariant_violations``. ea7ed7c beats every throughput
    # floor here (1,716,024 / 278, error rate 0.1965) and fails those two: 2,181
    # violations, the admitted set owing up to 42,477 tokens more than the pool could
    # obtain. Seed 7 is this profile's BEST seed for error rate (0.1355 against a
    # 0.136-0.246 spread over the eight seeds above), so the 0.143 ceiling is tight by
    # construction; widen it from a sweep, not from a single failing run.
    ("switchyard-deadlock", 1_105_000, 176, 0.143, None),
    # The restore profile is the regression test for soak §W6, and like the deadlock
    # profile its point is ``invariant_violations``, not its throughput. Measured on this
    # tree at seed 7 / 20,000 forwards: 2,383,276 tokens, 308 completed, error 0.3391,
    # 138 spills / 12 restores / 2 deferred, slack_min +169, 0 violations. The floors sit
    # ~5% under those.
    #
    # The A/B that makes it a test: deleting ``PrefillManager.finishability_reservation``
    # (the tree before the fix) on the SAME replay code takes seed 7 to 43 violations, the
    # admitted set owing 84,234 tokens more than the pool could obtain -- one restore of a
    # 127,204-token prefix, of which 121,865 tokens were evictable and therefore still
    # counted in ``available_size`` when the gate proved the in-flight prefills finishable.
    # Over the eight seeds {1,3,5,7,11,13,17,23} the fixed tree is 0 violations and
    # ``deadlock`` False on all eight; the unfixed one violates on seed 7 alone -- which is
    # why the live soak saw this once in 41 minutes over 642 restores, and why §V's 441
    # restores produced none.
    #
    # ``min_restores`` guards the test itself: a change that stopped the profile spilling
    # or restoring would otherwise pass it by doing nothing. The eight-seed spread is
    # 10-27 restores, so the floor sits well under the low end.
    ("switchyard-restore", 2_264_000, 292, 0.356, 8),
]


def gate(ticks: int = GATE_TICKS, seed: int = GATE_SEED, verbose: bool = False) -> int:
    """Run the fixed gate cases and report. Returns a process exit code."""
    failures = []
    for profile, min_tokens, min_completed, max_error, min_restores in GATE_CASES:
        t0 = time.perf_counter()
        out = run(ticks, seed, verbose=verbose, profile=profile)
        elapsed = time.perf_counter() - t0
        tokens = out["prefilled_tokens"]
        completed = out["completed"]
        bad = []
        if out["fatal"] is not None:
            bad.append(f"fatal: {out['fatal']}")
        if tokens < min_tokens:
            bad.append(f"prefilled_tokens {tokens:,} < {min_tokens:,}")
        if completed < min_completed:
            bad.append(f"completed {completed} < {min_completed}")
        err = out.get("error_rate")
        if max_error is not None and err is not None and err > max_error:
            bad.append(f"error_rate {err} > {max_error}")
        # Not a floor on the tree -- a floor on the HARNESS. A profile whose whole purpose
        # is one mechanism has to be shown still exercising it, or it grades nothing.
        restores = out.get("session_restores")
        if min_restores is not None and (restores or 0) < min_restores:
            bad.append(
                f"session_restores {restores} < {min_restores}: this profile is no longer "
                f"exercising cold restores, so its invariant check proves nothing"
            )
        # The two checks that a throughput floor cannot make. Both reverted trees beat this
        # gate's numbers and failed the live soak; these are what they fail.
        if out.get("deadlock"):
            bad.append(
                "permanent deadlock: last batch at "
                f"{out['sim_seconds'] - out['trailing_silence']:.1f}s, then "
                f"{out['trailing_silence']:.1f}s "
                f"({out['trailing_silence_frac']:.0%}) of silence -- "
                f"{out['deadlock_state']}"
            )
        if out.get("invariant_violations"):
            bad.append(
                f"finishability invariant violated on {out['invariant_violations']} "
                f"passes ({out['invariant_violation_frac']:.1%}); the admitted set owed "
                f"up to {-out['slack_min']:,} tokens more than the pool could obtain"
            )
        print(f"{'FAIL' if bad else 'ok  '} {profile:<19} "
              f"tokens={tokens:>10,} (min {min_tokens:>9,})  "
              f"completed={completed:>4} (min {min_completed:>4})  "
              f"lanes={out['lanes_mean']}  util={out['util_mean']}  "
              f"err={out.get('error_rate')}  stallUsage={out.get('stall_usage_p50')}  "
              f"restores={out.get('session_restores')}/"
              f"{out.get('session_restores_deferred')}def  "
              f"viol={out.get('invariant_violations')}  "
              f"deadlock={out.get('deadlock')}  {elapsed:.1f}s")
        for line in bad:
            print(f"       {line}")
        failures.extend(f"{profile}: {line}" for line in bad)
    if failures:
        print(f"\nscheduler replay gate FAILED ({len(failures)} check(s)):")
        for line in failures:
            print(f"  - {line}")
        print("\nReproduce a single case with:\n"
              f"  uv run --no-project python benchmarks/scheduler_replay.py --ticks {ticks} "
              f"--seed {seed} --profile "
              "<stage|pressure|switchyard-stage|switchyard-deadlock|switchyard-restore>"
              " --diagnose")
        return 1
    print("\nscheduler replay gate passed")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gate", action="store_true",
                    help="run the CI regression gate (fixed profiles/seed) and exit "
                         "non-zero on a regression")
    ap.add_argument("--ticks", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--label", default="")
    ap.add_argument("--profile", default="stage",
                    choices=["stage", "pressure", "fanout", "switchyard-stage",
                             "switchyard-deadlock", "switchyard-restore", "ornith-ada",
                             "trace"])
    ap.add_argument("--profile-file", default="",
                    help="a benchmarks/trace_to_profile.py profile: run the traffic shape "
                         "of a captured trace instead of a hand-written scenario mix "
                         "(implies --profile trace)")
    ap.add_argument("--diagnose", action="store_true")
    ap.add_argument("--abort-inflight", dest="abort_inflight",
                    action="store_true", default=None,
                    help="model a client timeout as freeing an in-flight prefill "
                         "(the server does NOT: see abandon_starved)")
    ap.add_argument("--no-abort-inflight", dest="abort_inflight",
                    action="store_false",
                    help="faithful client-timeout model; default on "
                         "switchyard-deadlock")
    ap.add_argument("--pool", type=int, default=POOL)
    ap.add_argument("--no-interleave", action="store_true")
    a = ap.parse_args()
    if a.profile_file:
        a.profile = apply_profile_file(a.profile_file)
    elif a.profile == "trace":
        ap.error("--profile trace requires --profile-file")
    if a.gate:
        # --ticks/--seed stay honoured so the gate can be shortened while bisecting;
        # CI runs it at the defaults the floors above were measured at.
        sys.exit(gate(a.ticks if a.ticks != 4000 else GATE_TICKS, a.seed, a.verbose))
    out = run(a.ticks, a.seed, a.verbose, a.profile, a.diagnose, a.pool,
              not a.no_interleave, a.abort_inflight)
    out["label"] = a.label
    out["profile"] = a.profile
    # The EFFECTIVE pool: a profile with its own geometry (PROFILE_KNOBS) overrides POOL
    # when --pool was left at the default, and the JSON line has to say what actually ran.
    out["pool"] = (
        a.pool if a.pool != POOL
        else PROFILE_KNOBS.get(a.profile, {}).get("pool_pages", POOL)
    )
    out["interleave"] = not a.no_interleave
    if a.profile_file:
        out["profile_file"] = a.profile_file
        out["profile_source"] = (TRACE_PROFILE or {}).get("source")
    print(json.dumps(out))
