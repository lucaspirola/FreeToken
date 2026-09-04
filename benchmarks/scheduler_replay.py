#!/usr/bin/env python3
"""CPU replay of the Switchyard *stage* route against FreeToken's scheduler.

Two ways to run it:

  * ``--gate`` -- the CI regression gate (.github/workflows/cpu-checks.yml). Runs the
    two fixed scenarios below and exits non-zero if throughput or completions fall
    below the recorded floors, or if the scheduler raises at all. Takes ~1 min.
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

Metrics: lanes per prefill batch, prefill budget utilisation, wait-to-first-chunk
per request, and any fatal raised out of the scheduler path.

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
FAMILIES = 4
REUSE_FRAC = 0.75  # stage route measured 73.6% prefix reuse pre-fix


def build_pool(pool_pages: int = POOL, interleave: bool = True):
    from freetoken.core import Context, get_global_ctx, set_global_ctx
    try:
        get_global_ctx()
    except Exception:
        set_global_ctx(Context(page_size=PAGE_SIZE))

    from freetoken.scheduler.cache import CacheManager
    from freetoken.scheduler.decode import DecodeManager
    from freetoken.scheduler.prefill import PrefillManager
    from freetoken.scheduler.table import TableManager

    pt = torch.zeros((MAX_RUNNING + 1, WIDTH), dtype=torch.int32, device="cpu")
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
    tm = TableManager(max_running_reqs=MAX_RUNNING, page_table=pt)
    dm = DecodeManager(page_size=PAGE_SIZE)
    pm = PrefillManager(cm, tm, dm)
    caps = {"committed_pages": grows,
            "interleave_chunks": hasattr(pm, "interleave_chunks"),
            "max_batch_seqs": hasattr(pm, "max_batch_seqs"),
            "committed_pages_required": hasattr(cm, "committed_pages_required")}
    if caps["interleave_chunks"]:
        pm.interleave_chunks = interleave   # growable multi-agent mode, as the soak ran
    return cm, tm, dm, pm, caps


class Traffic:
    """Deterministic stage-route generator with per-family shared prefixes."""

    def __init__(self, seed: int, profile: str = "stage"):
        self.rng = random.Random(seed)
        self.uid = 0
        table = {"pressure": SCENARIOS_PRESSURE, "fanout": SCENARIOS_FANOUT}.get(
            profile, SCENARIOS)
        self.reuse = 0.0 if profile == "fanout" else REUSE_FRAC
        self.jitter = (0.95, 1.05) if profile == "fanout" else (0.8, 1.2)
        self.pop = [s for s in table for _ in range(s[2])]
        self.prefix = {
            f: torch.arange(f * 10_000_000, f * 10_000_000 + WIDTH, dtype=torch.int32)
            for f in range(FAMILIES)
        }
        self.tail = 900_000_000

    def next(self, slot: int):
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
        return self.uid, name, ids


class _Msg:
    """Minimal UserMsg stand-in for PrefillManager.add_one_req."""

    def __init__(self, uid, ids):
        from freetoken.core import SamplingParams
        self.uid = uid
        self.input_ids = ids
        self.sampling_params = SamplingParams(max_tokens=OUTPUT_LEN)
        self.mm_embeds = None
        self.session_id = None
        self.session_ttl_seconds = None
        self.hidden_states = None
        self.no_prefix_cache = False


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
        interleave: bool = True):
    cm, tm, dm, pm, caps = build_pool(pool_pages, interleave)
    traffic = Traffic(seed, profile)
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
    first_chunk = {}   # uid -> wait to first forwarded chunk
    finished = {}      # uid -> total latency
    scenario = {}      # uid -> name
    slot_of = {}
    free_slots = list(range(MAX_RUNNING))

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

    def admit_new():
        while free_slots and len(pm.pending_list) + len(dm.running_reqs) < MAX_RUNNING:
            slot = free_slots.pop()
            uid, name, ids = traffic.next(slot)
            slot_of[uid] = slot
            scenario[uid] = name
            submitted[uid] = state["clock"]
            pm.add_one_req(_Msg(uid, ids))

    growable_decode_steps = 0
    try:
        admit_new()
        while tick < ticks:
            tick += 1
            prefill_runnable = pm.runnable
            decode_runnable = dm.runnable
            batch = None
            if prefill_runnable and decode_runnable:
                if growable_decode_steps < DECODE_BURST:
                    batch = dm.schedule_next_batch()
                    growable_decode_steps += 1
                else:
                    batch = pm.schedule_next_batch(PREFILL_BUDGET)
                    growable_decode_steps = 0
                    if batch is None:
                        empty_prefill_passes += 1
                        batch = dm.schedule_next_batch()
            elif prefill_runnable:
                batch = pm.schedule_next_batch(PREFILL_BUDGET)
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
                fatal = "LIVELOCK: no batch schedulable with work outstanding"
                break

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
                util_hist.append(new_tok / PREFILL_BUDGET)
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
                    tm.free(r.table_idx)
                    finished[r.uid] = state["clock"] - submitted[r.uid]
                    free_slots.append(slot_of[r.uid])
                if done:
                    admit_new()
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
# Floors for --gate. Each is a *floor*, not the measurement: the measured value at
# 81ab30e ("Scheduler: gate fresh admits on finishability + this chunk") is in the
# comment, and the floor sits ~8-10% under it so ordinary scheduling jitter between
# Python/torch versions does not trip the gate while a real starvation regression
# (which halves throughput) does.
#
# Measured at 81ab30e ("Scheduler: gate fresh admits on finishability + this chunk"),
# seed 7, 20,000 forwards, torch 2.11 CPU, Python 3.12 -- identical numbers on 2 and on
# 12 cores, so the run is deterministic and the floors are portable:
#   stage    prefilled_tokens  7,049,549   completed 373   (1.8 s on 2 cores)
#   pressure prefilled_tokens 10,071,808   completed  99   (0.4 s on 2 cores)
#
# The pre-fix trees this gate protects against sat at roughly half the stage
# throughput, or raised out of _prepare_batch outright (fatal != None).
GATE_TICKS = 20_000
GATE_SEED = 7
GATE_CASES = [
    # profile, min prefilled_tokens, min completed
    ("stage",    6_500_000, 350),
    ("pressure", 8_500_000,  85),
]


def gate(ticks: int = GATE_TICKS, seed: int = GATE_SEED, verbose: bool = False) -> int:
    """Run the fixed gate cases and report. Returns a process exit code."""
    failures = []
    for profile, min_tokens, min_completed in GATE_CASES:
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
        print(f"{'FAIL' if bad else 'ok  '} {profile:<9} "
              f"tokens={tokens:>10,} (min {min_tokens:>9,})  "
              f"completed={completed:>4} (min {min_completed:>4})  "
              f"lanes={out['lanes_mean']}  util={out['util_mean']}  {elapsed:.1f}s")
        for line in bad:
            print(f"       {line}")
        failures.extend(f"{profile}: {line}" for line in bad)
    if failures:
        print(f"\nscheduler replay gate FAILED ({len(failures)} check(s)):")
        for line in failures:
            print(f"  - {line}")
        print("\nReproduce a single case with:\n"
              f"  uv run --no-project python benchmarks/scheduler_replay.py --ticks {ticks} "
              f"--seed {seed} --profile <stage|pressure> --diagnose")
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
    ap.add_argument("--profile", default="stage", choices=["stage", "pressure", "fanout"])
    ap.add_argument("--diagnose", action="store_true")
    ap.add_argument("--pool", type=int, default=POOL)
    ap.add_argument("--no-interleave", action="store_true")
    a = ap.parse_args()
    if a.gate:
        # --ticks/--seed stay honoured so the gate can be shortened while bisecting;
        # CI runs it at the defaults the floors above were measured at.
        sys.exit(gate(a.ticks if a.ticks != 4000 else GATE_TICKS, a.seed, a.verbose))
    out = run(a.ticks, a.seed, a.verbose, a.profile, a.diagnose, a.pool,
              not a.no_interleave)
    out["label"] = a.label
    out["profile"] = a.profile
    out["pool"] = a.pool
    out["interleave"] = not a.no_interleave
    print(json.dumps(out))
