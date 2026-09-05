#!/usr/bin/env python3
"""Parse a FreeToken server.log (or a per-phase slice of one) from the Switchyard soak
into the numbers the results file wants: decode/prefill throughput, KV+mamba occupancy,
lanes per prefill batch, the §R7-ticket-1 starvation signature, back-pressure markers,
and the pressure episodes that used to be fatal.

Also reads the ``stats_*.json`` snapshots ``run.sh`` curls off ``/v1/stats`` at each phase
boundary, which carry the cumulative scheduler counters the log cannot give: aborts by
reason, chunked-prefill deferrals and ``max_chunked_prefills`` hits, the seatable-lanes
divisor histogram, finishability-invariant violations (counted whether or not
FREETOKEN_SCHEDULER_INVARIANT is set), speculative-decode declines by reason with the
accepted-token histogram, and session spill/restore/prefetch traffic including failures.
Snapshots are cumulative for the server process, so consecutive ones on one command line
also print the per-phase delta between them.

Usage: analyze.py <log|stats.json> [<log|stats.json> ...]
       analyze.py runs/<id>/server.log runs/<id>/stats_after_soak*.json
"""
from __future__ import annotations

import json
import re
import statistics as st
import sys

DEC = re.compile(
    r"Decode batch, #running-req: (\d+), #token: (\d+), token usage: ([\d.]+), "
    r"#mamba-slot: (\d+)/(\d+), mamba usage: ([\d.]+), gen throughput \(token/s\): ([\d.]+), "
    r"#queue-req: (\d+)"
)
PRE = re.compile(
    r"Prefill batch, #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+), "
    r"token usage: ([\d.]+), #mamba-slot: (\d+)/(\d+), mamba usage: ([\d.]+), "
    r"#running-req: (\d+), #queue-req: (\d+), "
    # Added after 13af13d; optional so this script still reads older soak logs.
    r"(?:#seatable-lane: (\d+), )?(?:#chunked-inflight: (\d+), )?"
    r"input throughput \(token/s\): ([\d.]+) instant"
)
# Column indices into the PRE tuple, named where the new optional ones would otherwise be
# off-by-one bait.
P_SEATABLE, P_CHUNKED, P_INSTANT_TPS = 9, 10, 11
MARKERS = (
    ("invariant_violated", "finishability invariant violated"),
    ("committed_pages", "committed_pages_required"),
    ("linear_exhausted", "LinearStatePool exhausted"),
    ("eviction_did_not_free", "Eviction did not free enough space"),
    ("oversize_skip", "can never be admitted"),
    ("traceback", "Traceback (most recent call last)"),
    ("released_admission", "KV protection (admission pressure)"),
    ("released_gdn", "KV protection (GDN state-slot pressure)"),
    ("released_grace", "KV protection (grace expired)"),
    ("kv_grew", "KV grew"),
    ("kv_shrank", "KV shrank"),
    ("session_expired", "expired after idle timeout"),
    ("restored_cold", "Restored cold session"),
    ("discarded_cold", "client tokens diverge"),
    ("scheduler_idle", "Scheduler is idle"),
)


def q(vals, name):
    if not vals:
        return f"{name}: n=0"
    return (f"{name}: n={len(vals)} median={st.median(vals):.2f} "
            f"mean={st.mean(vals):.2f} max={max(vals):.2f}")


def analyze(path: str) -> None:
    decodes, prefills, markers = [], [], {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = DEC.search(line)
            if m:
                decodes.append(tuple(float(x) for x in m.groups()))
                continue
            m = PRE.search(line)
            if m:
                # Optional groups are None on a pre-13af13d log; nan keeps the tuple shape
                # and drops out of every aggregate below via the `== x` filters.
                prefills.append(
                    tuple(float(x) if x is not None else float("nan") for x in m.groups())
                )
                continue
            for key, pat in MARKERS:
                if pat in line:
                    markers[key] = markers.get(key, 0) + 1

    print(f"=== {path}")
    print(f"decode batches: {len(decodes)}   prefill batches: {len(prefills)}")
    if decodes:
        run = [d[0] for d in decodes]
        print(q([d[6] for d in decodes], "decode aggregate tok/s (all)"))
        print(q([d[6] / d[0] for d in decodes if d[0] > 0], "decode per-stream tok/s (all)"))
        print(f"mean #running-req = {st.mean(run):.2f}")
        for sel, label in (([d for d in decodes if d[0] >= 12], ">= 12"),
                           ([d for d in decodes if d[0] == 16], "== 16")):
            if sel:
                print(f"  #running-req {label}: n={len(sel)} "
                      f"agg median={st.median([d[6] for d in sel]):.1f} "
                      f"mean={st.mean([d[6] for d in sel]):.1f} "
                      f"max={max(d[6] for d in sel):.1f} "
                      f"per-stream median={st.median([d[6] / d[0] for d in sel]):.2f}")
        print(q([d[2] for d in decodes], "decode token usage"))
        print(q([d[3] for d in decodes], "decode mamba slots used"))
        print(f"decode batches at mamba usage 1.00: {sum(1 for d in decodes if d[5] >= 1.0)}")
    if prefills:
        lanes = [p[0] for p in prefills]
        starv = [p for p in prefills if p[0] == 1 and p[1] <= 512 and p[8] >= 8]
        fresh = [p for p in prefills if p[2] > 0]
        print(q([p[P_INSTANT_TPS] for p in prefills], "prefill instant tok/s"))
        print(q([p[3] for p in prefills], "prefill token usage"))
        print(q([p[1] for p in prefills], "prefill #new-token"))
        print(f"lanes per prefill batch: mean={st.mean(lanes):.2f} "
              f"median={st.median(lanes):.0f} max={max(lanes):.0f}")
        print(f"STARVATION SIGNATURE (#new-seq==1, #new-token<=512, #queue-req>=8): "
              f"{len(starv)}/{len(prefills)} = {100 * len(starv) / len(prefills):.1f}%")
        print(f"passes with #cached-token>0 (proves a FRESH admit, so chunked_inflight<cap): "
              f"{len(fresh)}")
        print(f"prefill batches at mamba usage 1.00: {sum(1 for p in prefills if p[6] >= 1.0)}")
        print("prefill batches at 96/96 mamba slots: "
              f"{sum(1 for p in prefills if p[4] == p[5] and p[5] >= 96)}")
        seatable = [p[P_SEATABLE] for p in prefills if p[P_SEATABLE] == p[P_SEATABLE]]
        chunked = [p[P_CHUNKED] for p in prefills if p[P_CHUNKED] == p[P_CHUNKED]]
        if seatable:
            print(q(seatable, "seatable lanes (the interleave share's divisor)"))
            print(f"  passes whose divisor was 1: {sum(1 for v in seatable if v == 1)}"
                  f"/{len(seatable)}")
        if chunked:
            print(q(chunked, "chunked prefills in flight at pass start"))

    hot_d = [d for d in decodes if d[2] >= 0.98 and d[7] > 0]
    hot_p = [p for p in prefills if p[3] >= 0.98 and p[8] > 0]
    print(f"\npressure episodes (token usage >= 0.98 AND #queue-req > 0): "
          f"decode {len(hot_d)}, prefill {len(hot_p)}")
    print(f"  token usage >= 0.99: decode {sum(1 for d in decodes if d[2] >= 0.99)}, "
          f"prefill {sum(1 for p in prefills if p[3] >= 0.99)}")
    capped = [p for p in prefills
              if p[3] >= 0.95 and p[8] > 0 and 0 < p[1] < 8192 and p[0] == 1]
    print(f"  single-seq prefill chunks < 8192 tokens at usage >= 0.95 with queue: {len(capped)}")
    print("\nmarkers: " + (", ".join(f"{k}={v}" for k, v in sorted(markers.items())) or "none"))
    print()


# --------------------------------------------------------------------------------------
# /v1/stats snapshots (run.sh writes stats_after_<phase>.json and stats_{before,after}_probe)
# --------------------------------------------------------------------------------------


def _hist(counts: dict, label: str, indent: str = "  ") -> str:
    """A ``k=v`` histogram line with the empty buckets dropped, in numeric key order."""
    live = {k: v for k, v in (counts or {}).items() if v}
    if not live:
        return f"{indent}{label}: none"
    order = sorted(live, key=lambda k: (not k[:1].isdigit(), _lead_int(k), k))
    return f"{indent}{label}: " + " ".join(f"{k}={live[k]}" for k in order)


def _lead_int(key: str) -> int:
    lead = key.split("-")[0].rstrip("+")
    return int(lead) if lead.isdigit() else 0


def _kv(counts: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted((counts or {}).items())) or "none"


def _flat(doc: dict, prefix: str = "") -> dict:
    """Every integer leaf of a stats document, keyed by its dotted path -- what the delta
    between two cumulative snapshots is computed over."""
    out = {}
    for key, value in (doc or {}).items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flat(value, f"{path}."))
        elif isinstance(value, int) and not isinstance(value, bool):
            out[path] = value
    return out


def report_stats(path: str, previous: tuple[str, dict] | None) -> dict:
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)

    print(f"=== {path} (/v1/stats snapshot)")
    reqs = doc.get("requests") or {}
    print(f"uptime {doc.get('uptime_s', 0)} s, completed {reqs.get('completed', 0)}, "
          f"active {reqs.get('active', 0)}, p95 {reqs.get('p95_ms', 0)} ms, "
          f"ttft mean {reqs.get('ttft_mean_ms', 0)} ms")
    print(f"aborts: {_kv(reqs.get('aborts'))}")

    sched = doc.get("scheduler")
    if sched is None:
        # Either an engine from before these counters existed, or one that never published
        # (offline / non-primary TP rank). Not the same as "all zero".
        print("scheduler counters: NOT REPORTED by this engine")
    else:
        pre = sched.get("prefill")
        if pre:
            print(f"prefill passes: {pre.get('passes', 0)}  "
                  f"cap(max_chunked_prefills)={pre.get('max_chunked_prefills', 0)}")
            print(f"  fresh admits blocked by the cap: "
                  f"{pre.get('fresh_admits_blocked_by_cap', 0)}"
                  "   <- nonzero means the cap BINDS; the reservation arithmetic is wrong")
            print(f"  deferred chunks: {pre.get('deferred_chunks', 0)}   "
                  f"refusals (pool/table/budget): {pre.get('refusals', 0)}")
            print(f"  chunked prefills in flight: last={pre.get('chunked_inflight', 0)} "
                  f"max={pre.get('chunked_inflight_max', 0)}")
            print(f"  seatable lanes: last={pre.get('seatable_lanes_last', 0)}")
            print(_hist(pre.get("seatable_lanes"), "seatable-lane histogram", indent="    "))
            inv = pre.get("invariant") or {}
            print(f"  finishability invariant: checks={inv.get('checks', 0)} "
                  f"VIOLATIONS={inv.get('violations', 0)} "
                  f"worst shortfall={inv.get('worst_shortfall', 0)} tokens")
        spec = sched.get("spec")
        if spec is None:
            print("spec decode: off")
        else:
            print(f"spec decode: verify steps={spec.get('verify_steps', 0)} "
                  f"plain peeks={spec.get('plain_peeks', 0)} "
                  f"accept rate={spec.get('accept_rate', 0)} "
                  f"tokens/verify={spec.get('tokens_per_verify', 0)}")
            print(f"  declined: {_kv(spec.get('declined'))}")
            print(_hist(spec.get("accepted_hist"), "accepted-token histogram", indent="  "))
        spill = sched.get("session_spill")
        if spill is None:
            print("session spill: off")
        else:
            print(f"session spill: spills={spill.get('spills', 0)}"
                  f"/failed {spill.get('spills_failed', 0)}  "
                  f"restores={spill.get('restores', 0)}"
                  f"/failed {spill.get('restores_failed', 0)}"
                  f"/diverged {spill.get('restores_diverged', 0)}  "
                  f"prefetches={spill.get('prefetches', 0)}"
                  f"/failed {spill.get('prefetches_failed', 0)}"
                  f"/collected {spill.get('prefetches_collected', 0)}")

    flat = _flat(doc)
    if previous is not None:
        prev_path, prev_flat = previous
        moved = {
            k: flat[k] - prev_flat[k]
            for k in flat
            if k in prev_flat and flat[k] != prev_flat[k] and k != "uptime_s"
        }
        # Counters are cumulative for the server process, so the phase's own numbers are
        # the difference against the snapshot taken at the previous phase boundary.
        print(f"\ndelta vs {prev_path}:")
        print("  " + ("  ".join(f"{k}+{v}" if v > 0 else f"{k}{v}"
                                for k, v in sorted(moved.items())) or "nothing moved"))
    print()
    return flat


previous_stats: tuple[str, dict] | None = None
for arg in sys.argv[1:]:
    if arg.endswith(".json"):
        previous_stats = (arg, report_stats(arg, previous_stats))
    else:
        analyze(arg)
