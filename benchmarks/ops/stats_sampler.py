#!/usr/bin/env python3
"""Sample ``GET /v1/stats`` on a schedule, and summarize the samples into per-hour deltas.

    # capture (systemd --user, or by hand)
    python benchmarks/ops/stats_sampler.py sample --base-url http://127.0.0.1:1919
    # read it back
    python benchmarks/ops/stats_sampler.py summarize ~/.cache/freetoken/stats/

The soak reads ``/v1/stats`` exactly twice per phase (``run.sh`` curls a snapshot at each
phase boundary) and ``switchyard_soak/analyze.py`` differences the two. That works for a
40-minute run whose start and end are known; it does not work for a server that has been up
for a week, where the question is *when* the counters moved, not by how much in total.

This is the same arithmetic on a time series. Every counter in the document is cumulative
for the server process, so a rate is a difference between two snapshots; sampling every 60 s
and bucketing the differences by hour turns "restores_deferred = 41209" into "restores were
deferred between 14:00 and 15:00, and not before or since".

Three things it has to get right, and does:

*Restarts.* The counters are per-process, so a restart resets them to zero and a naive
difference goes hugely negative. ``uptime_s`` going backwards is the signal; the sample
after a restart is counted from zero rather than against the pre-restart snapshot, and the
restart is reported.

*The server being down.* A sampler that dies when the thing it samples dies is worse than
useless -- the outage is exactly the interval you wanted recorded. Every fetch failure is
written as a record with ``ok: false`` and the loop continues; nothing in the sample path
raises out of :func:`run_sampler`.

*Gauges vs counters.* ``requests.p95_ms`` and ``requests.ttft_mean_ms`` are windowed gauges
over the request ring, not cumulative totals -- differencing them is meaningless. They are
summarized as a distribution *across the samples in the bucket* (p50/p95 of the sampled
value), which is the only honest reading of a gauge polled at a fixed cadence.

Stdlib only, and no torch: it must run as an unprivileged long-lived service next to a
server whose venv it does not share. The cumulative-counter flattening and the key-value
rendering are imported from ``benchmarks/switchyard_soak/analyze.py`` rather than
re-derived, so a counter renamed in ``freetoken/scheduler/counters.py`` breaks both tools at
once instead of silently splitting them.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Iterable

_REPO = Path(__file__).resolve().parents[2]


def _load_analyze():
    """Import ``benchmarks/switchyard_soak/analyze.py`` by path.

    By path because the directory is not a package and ``benchmarks`` is not importable as
    one; the module itself is stdlib-only and guarded by ``if __name__ == "__main__"``.
    """
    path = _REPO / "benchmarks" / "switchyard_soak" / "analyze.py"
    spec = importlib.util.spec_from_file_location("_ft_soak_analyze", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


AZ = _load_analyze()

#: strftime pattern, expanded at every sample -- which is what rotates the file at midnight
#: without a rotation timer: the writer simply opens a different name once the date changes.
#: A pattern with no ``%`` in it never rotates, which is what an explicit ``--out`` means.
DEFAULT_OUT = "~/.cache/freetoken/stats/%Y-%m-%d.jsonl"
DEFAULT_BASE_URL = "http://127.0.0.1:1919"
DEFAULT_INTERVAL = 60.0


# --------------------------------------------------------------------------------- sample


def fetch_json(url: str, timeout: float) -> Any:
    req = urllib.request.Request(url, headers={"accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 -- operator URL
        return json.loads(resp.read().decode("utf-8"))


def sample_once(
    base_url: str,
    timeout: float,
    *,
    cursor: int | None = None,
    limit: int = 512,
    now: float | None = None,
) -> tuple[dict[str, Any], int | None]:
    """One sample record, and the ``/v1/requests`` cursor to poll with next.

    ``cursor`` is ``None`` when request pulling is off. It survives a failed poll unchanged,
    so an outage does not silently skip the rows that were in the ring across it -- the ring
    is bounded, so some are lost anyway, but not because this advanced past them.
    """
    ts = time.time() if now is None else now
    rec: dict[str, Any] = {
        "ts": round(ts, 3),
        "iso": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(ts)),
    }
    base = base_url.rstrip("/")
    try:
        rec["stats"] = fetch_json(f"{base}/v1/stats", timeout)
        rec["ok"] = True
    except Exception as exc:  # noqa: BLE001 -- the outage IS the datum; never propagate
        rec["ok"] = False
        rec["error"] = f"{type(exc).__name__}: {exc}"
        return rec, cursor
    if cursor is not None:
        try:
            doc = fetch_json(f"{base}/v1/requests?since={cursor}&limit={limit}", timeout)
            rec["requests"] = doc.get("entries") or []
            nxt = doc.get("next_cursor")
            if isinstance(nxt, int):
                cursor = nxt
        except Exception as exc:  # noqa: BLE001 -- a stats sample is still worth keeping
            rec["requests_error"] = f"{type(exc).__name__}: {exc}"
    return rec, cursor


def append_record(out_pattern: str, rec: dict[str, Any], now: float | None = None) -> str:
    """Append one JSON line to the (date-expanded) output path. Returns the path used."""
    ts = rec.get("ts") if now is None else now
    path = time.strftime(os.path.expanduser(out_pattern), time.localtime(ts))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # Open/append/close per sample: a sampler is read while it runs, and a kill -9 must not
    # cost the last hour of samples. One open per minute is free.
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    return path


def run_sampler(a: argparse.Namespace) -> int:
    """The poll loop. Returns only on ``--samples`` exhaustion or SIGINT; never raises."""
    cursor: int | None = 0 if a.requests else None
    seen = 0
    # Sleep to the next multiple of the interval rather than for the interval: a fixed sleep
    # accumulates the fetch time as drift, and after a day of 60 s samples the "hourly"
    # buckets no longer line up with the hour.
    started = time.time()
    while True:
        rec, cursor = sample_once(a.base_url, a.timeout, cursor=cursor, limit=a.requests_limit)
        try:
            path = append_record(a.out, rec)
        except Exception as exc:  # noqa: BLE001 -- a full/unwritable disk must not kill it
            print(f"stats_sampler: cannot write sample: {exc}", file=sys.stderr, flush=True)
            path = a.out
        if a.verbose:
            state = "ok" if rec.get("ok") else f"DOWN ({rec.get('error')})"
            print(f"{rec['iso']} {state} -> {path}", flush=True)
        seen += 1
        if a.samples and seen >= a.samples:
            return 0
        elapsed = time.time() - started
        delay = a.interval - (elapsed % a.interval)
        try:
            time.sleep(delay)
        except KeyboardInterrupt:
            return 0


# ------------------------------------------------------------------------------ summarize


def read_samples(paths: Iterable[str]) -> list[dict[str, Any]]:
    """Every sample record under the given files/globs/directories, in timestamp order.

    A partial last line (the sampler was killed mid-write) is skipped rather than fatal, as
    ``request_trace.read_trace`` does for the same reason.
    """
    files: list[str] = []
    for raw in paths:
        p = os.path.expanduser(raw)
        if os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, "*.jsonl"))))
        else:
            files.extend(sorted(glob.glob(p)) or [p])
    recs: list[dict[str, Any]] = []
    for f in files:
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if isinstance(rec, dict) and "ts" in rec:
                    recs.append(rec)
    recs.sort(key=lambda r: r.get("ts") or 0.0)
    return recs


#: Integer leaves of the stats document that are NOT cumulative counters: gauges,
#: high-water marks, derived ratios and static configuration. ``_flat`` cannot tell them
#: apart -- it flattens every int -- so the distinction lives here, and differencing one
#: anyway is how a report grows a "requests.p95_ms +300" column that means nothing.
#: High-water marks (``*_max``, ``worst_shortfall``) are monotonic but their *difference*
#: is not a rate; they are read as gauges instead.
GAUGE_PREFIXES = ("kv.", "mamba.", "swa.")
GAUGE_PATHS = frozenset({
    "uptime_s",
    "vram_bytes",
    "model.ctx",
    "requests.active",
    "requests.p95_ms",
    "requests.ttft_mean_ms",
    "scheduler.prefill.max_chunked_prefills",
    "scheduler.prefill.chunked_inflight",
    "scheduler.prefill.chunked_inflight_max",
    "scheduler.prefill.seatable_lanes_last",
    "scheduler.prefill.match.tokens_per_pass",
    "scheduler.prefill.invariant.worst_shortfall",
    "scheduler.moe.extend_cache.threshold_tokens",
})


def is_counter(path: str) -> bool:
    """True when a flattened stats path is a cumulative counter worth differencing."""
    return path not in GAUGE_PATHS and not path.startswith(GAUGE_PREFIXES)


def _pct(vals: list[float], p: float) -> float:
    s = sorted(vals)
    if not s:
        return 0.0
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


class Bucket:
    """One time bucket: summed counter deltas, plus the gauges sampled inside it."""

    __slots__ = ("key", "start", "samples", "down", "restarts", "delta", "p95_ms", "ttft_ms",
                 "active", "decode_tps")

    def __init__(self, key: str, start: float) -> None:
        self.key = key
        self.start = start
        self.samples = 0
        self.down = 0
        self.restarts = 0
        self.delta: dict[str, int] = {}
        self.p95_ms: list[float] = []
        self.ttft_ms: list[float] = []
        self.active: list[float] = []
        self.decode_tps: list[float] = []

    def add_delta(self, moved: dict[str, int]) -> None:
        for k, v in moved.items():
            self.delta[k] = self.delta.get(k, 0) + v

    def add_gauges(self, stats: dict[str, Any]) -> None:
        reqs = stats.get("requests") or {}
        for field, sink in (("p95_ms", self.p95_ms), ("ttft_mean_ms", self.ttft_ms),
                            ("active", self.active)):
            v = reqs.get(field)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                sink.append(float(v))
        tps = (stats.get("throughput") or {}).get("decode_tps")
        if isinstance(tps, (int, float)) and not isinstance(tps, bool):
            self.decode_tps.append(float(tps))


def bucket_samples(recs: list[dict[str, Any]], bucket_s: float) -> list[Bucket]:
    """Difference consecutive snapshots and accumulate them into wall-clock buckets.

    The delta belongs to the bucket of the *later* sample: it covers the interval ending
    there, and attributing it to the earlier one would credit the hour before an event with
    the event.
    """
    buckets: dict[str, Bucket] = {}
    fmt = "%Y-%m-%d %H:%M" if bucket_s < 3600 else "%Y-%m-%d %H:00"

    def bucket_for(ts: float) -> Bucket:
        start = ts - (ts % bucket_s)
        key = time.strftime(fmt, time.localtime(start))
        b = buckets.get(key)
        if b is None:
            b = buckets[key] = Bucket(key, start)
        return b

    prev: dict[str, int] | None = None
    prev_uptime = -1.0
    for rec in recs:
        b = bucket_for(rec.get("ts") or 0.0)
        b.samples += 1
        if not rec.get("ok"):
            b.down += 1
            # A gap of unknown length: the next successful sample becomes a fresh baseline,
            # or work done during the outage would be credited to the recovery minute.
            prev, prev_uptime = None, -1.0
            continue
        stats = rec.get("stats") or {}
        b.add_gauges(stats)
        uptime = float(stats.get("uptime_s") or 0)
        flat = {k: v for k, v in AZ._flat(stats).items() if is_counter(k)}
        if prev is None:
            moved: dict[str, int] = {}
        elif uptime < prev_uptime:
            # Restart: the counters restarted from 0, so this snapshot's absolute values ARE
            # the delta for the new process. Differencing against the old one is negative.
            b.restarts += 1
            moved = {k: v for k, v in flat.items() if v}
        else:
            moved = {k: flat[k] - prev[k] for k in flat
                     if k in prev and flat[k] != prev[k]}
        b.add_delta(moved)
        prev, prev_uptime = flat, uptime
    return [buckets[k] for k in sorted(buckets)]


#: ``(column header, dotted stats path)`` for the cumulative counters worth a column. Order
#: is the order of the table. A path this engine does not publish stays 0 -- which is
#: indistinguishable from "published and idle", exactly as in analyze.py, and is why the
#: NOT REPORTED line below exists.
COUNTER_COLUMNS: tuple[tuple[str, str], ...] = (
    ("req", "requests.completed"),
    ("err", "requests.aborts.error"),
    ("disc", "requests.aborts.client_disconnect"),
    ("refus", "scheduler.prefill.refusals"),
    ("freshdef", "scheduler.prefill.fresh_admits_deferred"),
    ("capblk", "scheduler.prefill.fresh_admits_blocked_by_cap"),
    ("spill", "scheduler.session_spill.spills"),
    ("restore", "scheduler.session_spill.restores"),
    ("restdef", "scheduler.session_spill.restores_deferred"),
    ("memohit", "scheduler.prefill.match.memo_hits"),
    ("inv", "scheduler.prefill.invariant.violations"),
)

#: Ratios, as ``(header, numerator path, second path, kind)``. ``kind`` is ``hits_misses``
#: when the second path is the complement and ``shortfall`` when it is a subtrahend
#: (``1 - missing/active``). Computed from the summed DELTAS, never read off a snapshot's
#: own lifetime ratio -- a rate already divided over a process lifetime cannot be
#: differenced back into a window's rate, which is why counters.py publishes raw counts.
RATIO_COLUMNS: tuple[tuple[str, str, str, str], ...] = (
    ("moe_hit%", "scheduler.moe.decode.active", "scheduler.moe.decode.missing", "shortfall"),
    ("xcache%", "scheduler.moe.extend_cache.hits", "scheduler.moe.extend_cache.misses",
     "hits_misses"),
    ("memo%", "scheduler.prefill.match.memo_hits", "scheduler.prefill.match.calls",
     "hits_misses"),
)


def ratio(b: Bucket, hit: str, other: str, kind: str) -> float | None:
    h, o = b.delta.get(hit, 0), b.delta.get(other, 0)
    if kind == "shortfall":  # h = active, o = missing; the hit rate is 1 - missing/active
        return (100.0 * (h - o) / h) if h else None
    total = h + o
    return (100.0 * h / total) if total else None


def _num(v: int) -> str:
    return f"{v:,}" if v else "-"


def summarize(recs: list[dict[str, Any]], bucket_s: float, *, as_json: bool = False) -> int:
    if not recs:
        print("no samples", file=sys.stderr)
        return 1
    buckets = bucket_samples(recs, bucket_s)
    ok = [r for r in recs if r.get("ok")]
    span = (recs[0].get("iso", "?"), recs[-1].get("iso", "?"))
    if as_json:
        print(json.dumps({
            "samples": len(recs), "ok": len(ok), "span": list(span),
            "bucket_s": bucket_s,
            "buckets": [{
                "bucket": b.key, "samples": b.samples, "unreachable": b.down,
                "restarts": b.restarts, "delta": b.delta,
                "p95_ms": {"p50": _pct(b.p95_ms, 0.5), "p95": _pct(b.p95_ms, 0.95)},
                "ttft_mean_ms": {"p50": _pct(b.ttft_ms, 0.5), "p95": _pct(b.ttft_ms, 0.95)},
                "active": {"p50": _pct(b.active, 0.5), "max": max(b.active or [0])},
                "ratios": {name: ratio(b, h, o, k) for name, h, o, k in RATIO_COLUMNS},
            } for b in buckets],
        }, indent=2, sort_keys=True))
        return 0

    print(f"samples {len(recs)} (reachable {len(ok)}, unreachable {len(recs) - len(ok)})")
    print(f"span {span[0]} .. {span[1]}   bucket {int(bucket_s)} s")
    restarts = sum(b.restarts for b in buckets)
    if restarts:
        when = ", ".join(b.key for b in buckets if b.restarts)
        print(f"server restarts (uptime_s went backwards): {restarts}  at {when}")
    if ok and (ok[-1].get("stats") or {}).get("scheduler") is None:
        print("scheduler counters: NOT REPORTED by this engine (offline / non-primary TP rank)")
    print()

    heads = (["bucket", "n", "down"] + [c[0] for c in COUNTER_COLUMNS]
             + [c[0] for c in RATIO_COLUMNS] + ["p50 lat", "p95 lat", "ttft", "act"])
    rows: list[list[str]] = []
    for b in buckets:
        row = [b.key, str(b.samples), str(b.down or 0)]
        row += [_num(b.delta.get(path, 0)) for _h, path in COUNTER_COLUMNS]
        for _h, hit, other, kind in RATIO_COLUMNS:
            r = ratio(b, hit, other, kind)
            row.append("-" if r is None else f"{r:.1f}")
        row += [f"{_pct(b.p95_ms, 0.5):.0f}", f"{_pct(b.p95_ms, 0.95):.0f}",
                f"{_pct(b.ttft_ms, 0.5):.0f}", f"{max(b.active or [0]):.0f}"]
        rows.append(row)
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(heads)]
    print("  ".join(h.rjust(w) for h, w in zip(heads, widths)))
    for row in rows:
        print("  ".join(c.rjust(w) for c, w in zip(row, widths)))

    print()
    print("columns: req=requests.completed  err/disc=aborts.error/client_disconnect  "
          "refus=prefill.refusals")
    print("         freshdef=fresh_admits_deferred  capblk=fresh_admits_blocked_by_cap  "
          "inv=finishability violations")
    print("         spill/restore/restdef=session_spill.{spills,restores,restores_deferred}")
    print("         moe_hit%=1-missing/active over the window  xcache%=extend-cache gate  "
          "memo%=radix match memo")
    print("         p50/p95 lat = quantiles of the SAMPLED requests.p95_ms gauge; "
          "ttft = p50 of ttft_mean_ms; act = max active")
    total: dict[str, int] = {}
    for b in buckets:
        for k, v in b.delta.items():
            total[k] = total.get(k, 0) + v
    hot = {k: v for k, v in total.items() if v and (
        "invariant.violations" in k or "aborts" in k or "failed" in k
        or k.endswith("restores_deferred") or k.endswith("fresh_admits_blocked_by_cap"))}
    print("\nwhole-span totals worth a look: " + (AZ._kv(hot) if hot else "none"))
    return 0


# ------------------------------------------------------------------------------------ cli


def build_parser(prog: str = "stats_sampler.py") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog,
        description="Poll /v1/stats into a JSONL file; summarize it into per-hour deltas.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("sample", help="poll the server and append one JSON line per sample")
    s.add_argument("--base-url", default=DEFAULT_BASE_URL)
    s.add_argument("--interval", type=float, default=DEFAULT_INTERVAL,
                   help="seconds between samples (default: 60)")
    s.add_argument("--out", default=DEFAULT_OUT,
                   # argparse %-formats help strings, so every literal % in the default
                   # path has to be doubled or `--help` dies on `%Y`.
                   help="output path; strftime escapes are expanded per sample, which is "
                        f"what rotates it daily (default: {DEFAULT_OUT.replace('%', '%%')})")
    s.add_argument("--timeout", type=float, default=10.0, help="HTTP timeout in seconds")
    s.add_argument("--requests", action="store_true",
                   help="also pull /v1/requests?since=<cursor> and store the new rows")
    s.add_argument("--requests-limit", type=int, default=512,
                   help="rows per /v1/requests poll (the server caps it at 512)")
    s.add_argument("--samples", type=int, default=0,
                   help="stop after N samples (0 = run forever)")
    s.add_argument("--verbose", action="store_true", help="one line per sample on stdout")

    z = sub.add_parser("summarize", help="per-bucket deltas of a sample file")
    z.add_argument("paths", nargs="+", help="sample .jsonl files, globs, or directories")
    z.add_argument("--bucket", type=float, default=3600.0,
                   help="bucket width in seconds (default: 3600 = per hour)")
    z.add_argument("--json", action="store_true", dest="as_json")
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    if a.command == "sample":
        if a.interval <= 0:
            print("--interval must be > 0", file=sys.stderr)
            return 2
        return run_sampler(a)
    return summarize(read_samples(a.paths), a.bucket, as_json=a.as_json)


if __name__ == "__main__":
    raise SystemExit(main())
