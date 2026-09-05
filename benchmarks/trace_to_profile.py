#!/usr/bin/env python3
"""Turn a captured request trace into a benchmarks/scheduler_replay.py profile.

    python benchmarks/trace_to_profile.py --trace /var/tmp/ft-trace --out trace.profile.json
    python benchmarks/scheduler_replay.py --profile-file trace.profile.json --ticks 4000

``scheduler_replay.py``'s five profiles are hand-written geometries, each one the shape of a
failure we have already seen (an admission-gate starvation, a committed-pages fatal, a
permanent pool deadlock). They are the right regression gate and they are, by construction,
a museum: nothing in them describes traffic we have not yet had a bad day with. This reads
a real trace and emits the same kind of object -- arrival pattern, prompt lengths, cached
fractions, output lengths, session retention -- so the CPU gate can also be run on the shape
that actually arrived.

What is derived, and from what
------------------------------
``scenarios``       prompt-token quantile buckets; each entry is
                    ``(name, median prompt tokens, request count)``, which is exactly the
                    ``(name, length, weight)`` shape ``Traffic`` samples from.
``reuse``           median ``cached_tokens / prompt_tokens``. This is the single number
                    that decides whether the radix cache can keep up, and the synthetic
                    profiles guess it at 0.75.
``output_len``      median ``output_tokens`` -- what a decode lane costs.
``sessions``        whether the traffic is conversational at all.
``turns``           the (p10, p90) count of turns a session runs before it goes idle, which
                    is what decides how many leases accumulate as *reclaimable*.
``turn_growth``     median ``prompt[k+1] - prompt[k] - output[k]`` within a session: how
                    much new text a turn adds on top of what the last one left in the tree.
``families``        distinct first-message hashes = distinct system prompts / tool catalogs,
                    i.e. how many separate shared prefixes the tree has to hold.
``knobs.agents``    peak concurrency, swept from the arrival/finish intervals -- the real
                    ``--max-running-requests`` the traffic demanded, not the flag's value.
``jitter``          the within-bucket length spread, so a bucket is not a single length.

Not derived, because a trace cannot see it: pool size, prefill budget and the lane cap are
server flags, and are passed through from ``--pool-pages`` / ``--prefill-budget`` (defaulting
to the P2 serving profile's). ``client_timeout`` likewise defaults to the soak's 600 s.

Stdlib only -- no torch, so this runs anywhere the trace file does.
"""

from __future__ import annotations

import argparse
import json

import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_replay import RT  # noqa: E402 -- one definition of the trace format

PROFILE_VERSION = 1

# Server-side geometry a trace cannot observe; the P2 Switchyard serving profile's values.
DEFAULT_POOL_PAGES = 262_144      # --num-tokens
DEFAULT_PREFILL_BUDGET = 8_192    # --max-prefill-length
DEFAULT_CLIENT_TIMEOUT = 600.0    # switchyard-soak --request-timeout
DEFAULT_SESSION_TTL = 300.0       # UserMsg.session_ttl_seconds


def _pct(vals: list[float], p: float) -> float:
    s = sorted(vals)
    if not s:
        return 0.0
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def peak_concurrency(records: list[dict[str, Any]]) -> int:
    """Max requests in flight, from the arrival/finish interval sweep.

    This is the concurrency the traffic *demanded*. It can exceed the server's
    ``--max-running-requests``, because a request is in flight from the moment it arrives
    -- queued time included -- and that is the number the replay wants: it models the
    client population, and the scheduler under test is what decides how many of them run.
    """
    events: list[tuple[float, int]] = []
    for r in records:
        t = r.get("t") or 0.0
        events.append((t, +1))
        events.append((t + (r.get("duration_ms") or 0.0) / 1e3, -1))
    events.sort()
    cur = peak = 0
    for _, delta in events:
        cur += delta
        peak = max(peak, cur)
    return max(1, peak)


def bucket_scenarios(records: list[dict[str, Any]], buckets: int) -> list[list[Any]]:
    """Quantile-bucket prompt lengths into ``(name, median length, count)`` rows.

    Quantiles rather than fixed edges because the interesting traces are heavy-tailed: a
    Switchyard mix puts half its requests under 8K and a tail at 118K, and equal-width bins
    would give one bucket 95 % of the weight and the tail no representation at all.
    """
    lengths = sorted(r["prompt_tokens"] for r in records if (r.get("prompt_tokens") or 0) > 0)
    if not lengths:
        return []
    n = min(buckets, len(set(lengths)))
    edges = [_pct(lengths, i / n) for i in range(1, n)]
    groups: list[list[int]] = [[] for _ in range(n)]
    for length in lengths:
        idx = 0
        while idx < len(edges) and length > edges[idx]:
            idx += 1
        groups[idx].append(length)
    rows: list[list[Any]] = []
    for g in groups:
        if not g:
            continue
        med = int(statistics.median(g))
        rows.append([_name(med), max(1, med), len(g)])
    # Two buckets can round to the same median on a spiky trace; merge rather than emit a
    # duplicate name that would read as two scenarios in the replay's output.
    merged: dict[str, list[Any]] = {}
    for name, length, weight in rows:
        if name in merged:
            merged[name][2] += weight
        else:
            merged[name] = [name, length, weight]
    return list(merged.values())


def _name(tokens: int) -> str:
    if tokens >= 1_000_000:
        return f"prompt-{tokens // 1_000_000}m"
    if tokens >= 1_000:
        return f"prompt-{tokens // 1_000}k"
    return f"prompt-{tokens}"


def session_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Turns per session, and how much a turn adds on top of the previous one."""
    by_session: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        sid = r.get("session")
        if sid:
            by_session.setdefault(sid, []).append(r)
    if not by_session:
        return {"sessions": False, "turns": [1, 1], "turn_growth": 0, "session_count": 0}
    turn_counts = [len(v) for v in by_session.values()]
    growth: list[int] = []
    for turns in by_session.values():
        turns.sort(key=lambda r: r.get("t") or 0.0)
        for prev, cur in zip(turns, turns[1:]):
            d = ((cur.get("prompt_tokens") or 0) - (prev.get("prompt_tokens") or 0)
                 - (prev.get("output_tokens") or 0))
            if d >= 0:
                growth.append(d)
    lo = max(1, int(round(_pct([float(c) for c in turn_counts], 0.10))))
    hi = max(lo, int(round(_pct([float(c) for c in turn_counts], 0.90))))
    return {
        "sessions": True,
        "turns": [lo, hi],
        "turn_growth": int(statistics.median(growth)) if growth else 0,
        "session_count": len(by_session),
    }


def build_profile(records: list[dict[str, Any]], a: argparse.Namespace) -> dict[str, Any]:
    served = [r for r in records if (r.get("prompt_tokens") or 0) > 0]
    if not served:
        raise SystemExit("trace has no request with a prompt_tokens count")
    scenarios = bucket_scenarios(served, a.buckets)
    reuse_samples = [min(1.0, (r.get("cached_tokens") or 0) / r["prompt_tokens"])
                     for r in served]
    outputs = [r.get("output_tokens") or 0 for r in served if (r.get("output_tokens") or 0) > 0]
    output_len = int(statistics.median(outputs)) if outputs else 256
    sess = session_stats(served)
    # Distinct first messages = distinct shared heads (a system prompt or tool catalog).
    # Capped: Traffic allocates a WIDTH-token id tensor per family, so an unbounded count
    # off a long trace would allocate gigabytes to model a distinction the pool cannot see.
    fams = {(r.get("msg_chain") or [None])[0] for r in served}
    fams.discard(None)
    families = max(1, min(a.max_families, len(fams)))
    lengths = [float(r["prompt_tokens"]) for r in served]
    med = statistics.median(lengths) or 1.0
    jitter = [max(0.2, round(_pct(lengths, 0.10) / med, 3)),
              max(1.0, round(_pct(lengths, 0.90) / med, 3))]
    span = (max(r["t"] for r in records) - min(r["t"] for r in records)) or 1.0
    errors = sum(1 for r in records if r.get("status") != "ok")
    return {
        "profile_version": PROFILE_VERSION,
        "name": a.name or Path(a.trace).name,
        "source": {
            "trace": str(Path(a.trace).resolve()),
            "requests": len(records),
            "served": len(served),
            "errors": errors,
            "span_s": round(span, 3),
            "arrival_rate_hz": round(len(records) / span, 4),
            "routes": sorted({r.get("route") or "?" for r in records}),
            "models": sorted({r.get("model") or "?" for r in records}),
        },
        "scenarios": scenarios,
        "sessions": sess["sessions"],
        "turns": sess["turns"],
        "turn_growth": sess["turn_growth"],
        "reuse": round(statistics.median(reuse_samples), 4) if reuse_samples else 0.0,
        "output_len": output_len,
        "families": families,
        "jitter": jitter,
        "client_timeout": a.client_timeout,
        "session_ttl": a.session_ttl,
        "width": a.width or (max(int(s[1]) for s in scenarios) + output_len + 512 + 1),
        "knobs": {
            "agents": a.agents or peak_concurrency(records),
            "prefill_budget": a.prefill_budget,
            "pool_pages": a.pool_pages,
        },
        # Kept beside the geometry so a scheduler_replay run can be read against what the
        # server actually delivered on this traffic, not only against the other profiles.
        "observed": {
            "ttft_ms_p50": round(_pct([r["ttft_ms"] for r in served
                                       if r.get("ttft_ms") is not None], 0.50), 2),
            "ttft_ms_p95": round(_pct([r["ttft_ms"] for r in served
                                       if r.get("ttft_ms") is not None], 0.95), 2),
            "latency_ms_p50": round(_pct([r["duration_ms"] for r in served
                                          if r.get("duration_ms") is not None], 0.50), 2),
            "latency_ms_p95": round(_pct([r["duration_ms"] for r in served
                                          if r.get("duration_ms") is not None], 0.95), 2),
            "output_tok_s": round(sum(outputs) / span, 3),
            "cached_frac": round(
                sum(r.get("cached_tokens") or 0 for r in served)
                / max(1, sum(r["prompt_tokens"] for r in served)), 4),
            "session_count": sess["session_count"],
            "peak_concurrency": peak_concurrency(records),
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trace", required=True, help="trace .jsonl file or --trace-dir dir")
    ap.add_argument("--out", default="", help="write the profile here (default: stdout)")
    ap.add_argument("--name", default="", help="profile name (default: the trace's)")
    ap.add_argument("--buckets", type=int, default=5,
                    help="prompt-length quantile buckets = scenarios (default 5, the "
                         "number the Switchyard soak's own scenario mix has)")
    ap.add_argument("--max-families", type=int, default=16,
                    help="cap on distinct shared prefixes modelled (memory: one "
                         "WIDTH-token id tensor each)")
    ap.add_argument("--route", default="", help="only requests on this route")
    ap.add_argument("--agents", type=int, default=0,
                    help="override the derived peak concurrency")
    ap.add_argument("--pool-pages", type=int, default=DEFAULT_POOL_PAGES,
                    help="the server's --num-tokens; a trace cannot observe it")
    ap.add_argument("--prefill-budget", type=int, default=DEFAULT_PREFILL_BUDGET,
                    help="the server's --max-prefill-length; a trace cannot observe it")
    ap.add_argument("--client-timeout", type=float, default=DEFAULT_CLIENT_TIMEOUT)
    ap.add_argument("--session-ttl", type=float, default=DEFAULT_SESSION_TTL)
    ap.add_argument("--width", type=int, default=0,
                    help="page-table row width; 0 derives it from the longest scenario")
    a = ap.parse_args(argv)

    records = list(RT.read_trace(a.trace))
    if a.route:
        records = [r for r in records if (r.get("route") or "") == a.route]
    if not records:
        print(f"no v{RT.TRACE_VERSION} records in {a.trace}", file=sys.stderr)
        return 2
    profile = build_profile(records, a)
    text = json.dumps(profile, indent=2)
    if a.out:
        Path(a.out).write_text(text + "\n")
        obs = profile["observed"]
        print(f"{a.out}: {len(profile['scenarios'])} scenarios from "
              f"{profile['source']['served']} requests, reuse={profile['reuse']:.2f}, "
              f"output_len={profile['output_len']}, agents={profile['knobs']['agents']}, "
              f"sessions={profile['sessions']} ({obs['session_count']})")
        print(f"  run it: python benchmarks/scheduler_replay.py --profile-file {a.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
