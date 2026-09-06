#!/usr/bin/env python3
"""Compare a captured request trace against the load the Switchyard soak assumes.

    python benchmarks/trace_load_report.py --trace /var/tmp/ft-trace
    python benchmarks/trace_load_report.py --trace /var/tmp/ft-trace \
        --soak-run benchmarks/switchyard_soak/runs/20260905T120000Z

The soak (docs/switchyard.md §8) is a closed 16-client loop over five fixed scenarios with
``--prompt-bytes 16384`` and ``--max-output-tokens 256``. Every scheduler decision the gate
protects was tuned against that shape. Nothing has ever checked whether the shape resembles
the traffic the deployment actually gets -- and if it does not, a green gate is evidence
about a workload nobody runs.

This reads a ``--trace-dir`` capture (§9) and prints the real load's arrival rate,
concurrency, prompt/output length distributions, prefix reuse, per-route split, session
structure and abort rate, each beside the soak's corresponding number, with the ratio
between them. Then a verdict: which of the soak's assumptions hold, which do not, and the
exact ``trace_to_profile.py`` command that turns this trace into a replay-gate profile
sized to what was measured.

Where the soak's numbers come from
----------------------------------
Two sources, and the report always says which one a row used.

*A soak run directory* (``--soak-run``), when given: ``run.sh`` curls ``/v1/stats`` at every
phase boundary, so the phase's own traffic is the difference between consecutive snapshots
-- the same arithmetic ``switchyard_soak/analyze.py`` prints, reusing its ``_flat`` here so
the two cannot drift. That yields the soak's measured requests, mean prompt and output
tokens, wall duration (from ``uptime_s``, which is in the snapshot) and abort counts.
``soak*/results-*/summary.json`` adds the soak client's own verdict.

*The client's configuration* otherwise: the flags ``switchyard_e2e.py soak`` passes, which
are constants of the harness, not of a run. These are marked ``(client config)`` and the
prompt-token figure derived from ``--prompt-bytes`` is explicitly an approximation -- bytes
are not tokens, and the divisor is a flag here rather than a hidden constant.

Stdlib only, no torch: the trace format reader, the quantile helper and the concurrency
sweep are imported from ``trace_to_profile.py`` / ``trace_replay.py`` so that one definition
of the format serves the replay, the profile and this report.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_HERE))

import trace_to_profile as T2P  # noqa: E402 -- one definition of the trace format

RT = T2P.RT
_pct = T2P._pct


def _load_analyze():
    """Import ``switchyard_soak/analyze.py`` by path (the directory is not a package)."""
    path = _REPO / "benchmarks" / "switchyard_soak" / "analyze.py"
    spec = importlib.util.spec_from_file_location("_ft_soak_analyze", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


AZ = _load_analyze()

#: What ``scripts/switchyard_e2e.py soak`` passes to ``switchyard-soak`` (docs §8). These
#: are properties of the harness, so they are the fallback baseline when no run dir is
#: given -- and the honest label for every row derived from them is "client config".
SOAK_CLIENT: dict[str, Any] = {
    "concurrency": 16,
    "max_output_tokens": 256,
    "prompt_bytes": 16_384,
    "context_window_tokens": 131_072,
    "request_timeout_s": 600.0,
    "max_error_rate": 0.0,
    "scenarios": 5,
    "routes": ("switchyard/stage", "switchyard/passthrough"),
    "per_phase_duration": "20m",
}

#: Bytes per token used to turn ``--prompt-bytes`` into a token figure when there is no
#: soak run to measure it from. English prose on this tokenizer sits near 4; a trace of
#: code or CJK does not, which is why this is a flag and every row it feeds says "approx".
DEFAULT_CHARS_PER_TOKEN = 4.0


# ------------------------------------------------------------------------- distributions


def dist(vals: Iterable[float]) -> dict[str, float]:
    """p50/p90/p99/max/mean/n of a sample. Empty gives all-zero rather than raising: a
    trace with no streaming request has no TTFT, and that is a fact to print, not a crash."""
    v = [float(x) for x in vals]
    if not v:
        return {"n": 0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "n": len(v),
        "p50": _pct(v, 0.50),
        "p90": _pct(v, 0.90),
        "p99": _pct(v, 0.99),
        "max": max(v),
        "mean": statistics.fmean(v),
    }


def per_bucket(records: list[dict[str, Any]], bucket_s: float) -> dict[str, Any]:
    """Arrivals, completions and distinct active sessions per wall-clock bucket.

    Arrivals and completions are counted separately on purpose: a queue that is filling
    shows as arrivals persistently above completions, which no single "request rate" number
    can express and which is precisely the shape the admission gate is judged on.
    """
    if not records:
        return {"buckets": 0, "arrivals": [], "completions": [], "sessions": []}
    t0 = min(r.get("t") or 0.0 for r in records)
    t1 = max((r.get("t") or 0.0) + (r.get("duration_ms") or 0.0) / 1e3 for r in records)
    n = max(1, int((t1 - t0) // bucket_s) + 1)
    arrivals = [0] * n
    completions = [0] * n
    sessions: list[set[str]] = [set() for _ in range(n)]
    for r in records:
        start = (r.get("t") or 0.0) - t0
        end = start + (r.get("duration_ms") or 0.0) / 1e3
        i = min(n - 1, max(0, int(start // bucket_s)))
        j = min(n - 1, max(0, int(end // bucket_s)))
        arrivals[i] += 1
        completions[j] += 1
        sid = r.get("session")
        if sid:
            # A long request keeps its session active in every bucket it spans, which is
            # what "sessions active per minute" has to mean for a lease-holding scheduler.
            for k in range(i, j + 1):
                sessions[k].add(sid)
    return {
        "buckets": n,
        "arrivals": arrivals,
        "completions": completions,
        "sessions": [len(s) for s in sessions],
    }


def session_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Session count, turns per session, and lifetime (first arrival to last completion)."""
    by: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        sid = r.get("session")
        if sid:
            by.setdefault(sid, []).append(r)
    if not by:
        return {"count": 0, "bound_frac": 0.0, "turns": dist([]), "lifetime_s": dist([])}
    lifetimes = []
    for turns in by.values():
        first = min(t.get("t") or 0.0 for t in turns)
        last = max((t.get("t") or 0.0) + (t.get("duration_ms") or 0.0) / 1e3 for t in turns)
        lifetimes.append(last - first)
    bound = sum(len(v) for v in by.values())
    return {
        "count": len(by),
        "bound_frac": bound / max(1, len(records)),
        "turns": dist(len(v) for v in by.values()),
        "lifetime_s": dist(lifetimes),
    }


def route_split(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        by.setdefault(str(r.get("route") or "?"), []).append(r)
    rows = []
    for route, rs in sorted(by.items()):
        prompts = [r["prompt_tokens"] for r in rs if (r.get("prompt_tokens") or 0) > 0]
        outs = [r.get("output_tokens") or 0 for r in rs]
        rows.append({
            "route": route,
            "requests": len(rs),
            "share": len(rs) / max(1, len(records)),
            "prompt_p50": _pct([float(p) for p in prompts], 0.50),
            "output_p50": _pct([float(o) for o in outs], 0.50),
        })
    return rows


def build_trace_report(records: list[dict[str, Any]], bucket_s: float) -> dict[str, Any]:
    """Everything the report knows about the real load, as plain data."""
    served = [r for r in records if (r.get("prompt_tokens") or 0) > 0]
    ts = [r.get("t") or 0.0 for r in records]
    span = (max(ts) - min(ts)) if len(ts) > 1 else 0.0
    prompts = [float(r["prompt_tokens"]) for r in served]
    cached = [float(r.get("cached_tokens") or 0) for r in served]
    fresh = [max(0.0, p - c) for p, c in zip(prompts, cached)]
    outputs = [float(r.get("output_tokens") or 0) for r in served]
    buckets = per_bucket(records, bucket_s)
    statuses: dict[str, int] = {}
    for r in records:
        s = str(r.get("status") or "ok")
        statuses[s] = statuses.get(s, 0) + 1
    ttfts = [float(r["ttft_ms"]) for r in records if r.get("ttft_ms") is not None]
    durs = [float(r["duration_ms"]) for r in records if r.get("duration_ms") is not None]
    total_prompt = sum(prompts)
    return {
        "requests": len(records),
        "served": len(served),
        "span_s": span,
        "routes": sorted({str(r.get("route") or "?") for r in records}),
        "models": sorted({str(r.get("model") or "?") for r in records}),
        "bucket_s": bucket_s,
        "rate_per_bucket": dist([float(x) for x in buckets["arrivals"]]),
        "completions_per_bucket": dist([float(x) for x in buckets["completions"]]),
        "sessions_per_bucket": dist([float(x) for x in buckets["sessions"]]),
        "req_per_min": (len(records) / span * 60.0) if span > 0 else 0.0,
        "peak_concurrency": T2P.peak_concurrency(records),
        "mean_inflight": (sum(r.get("duration_ms") or 0.0 for r in records) / 1e3 / span)
        if span > 0 else 0.0,
        "prompt_tokens": dist(prompts),
        "fresh_tokens": dist(fresh),
        "output_tokens": dist(outputs),
        "cached_frac_agg": (sum(cached) / total_prompt) if total_prompt else 0.0,
        "cached_frac_median": statistics.median(
            [min(1.0, c / p) for p, c in zip(prompts, cached)]) if prompts else 0.0,
        "has_cached": any(r.get("cached_tokens") for r in served),
        "routes_split": route_split(records),
        "sessions": session_report(records),
        "ttft_ms": dist(ttfts),
        "duration_ms": dist(durs),
        "statuses": statuses,
        "abort_frac": statuses.get("abort", 0) / max(1, len(records)),
        "error_frac": statuses.get("error", 0) / max(1, len(records)),
        "stream_frac": sum(1 for r in records if r.get("stream")) / max(1, len(records)),
    }


# ------------------------------------------------------------------------ soak baseline


def _snapshot_order(paths: list[str]) -> list[tuple[str, dict[str, Any]]]:
    """Load ``stats_*.json`` snapshots and order them by ``uptime_s``.

    Not by filename and not by mtime: ``stats_after_soakStage`` / ``stats_after_soakPass``
    / ``stats_before_probe`` do not sort alphabetically into their chronological order, and
    a copied run directory has lost its mtimes. ``uptime_s`` is monotonic within the server
    process, so it *is* the chronology, and it comes free inside the document.
    """
    docs = []
    for p in paths:
        try:
            with open(p, encoding="utf-8") as fh:
                doc = json.load(fh)
        except (OSError, ValueError):
            continue
        if isinstance(doc, dict):
            docs.append((p, doc))
    docs.sort(key=lambda pd: float(pd[1].get("uptime_s") or 0))
    return docs


def load_soak_run(run_dir: str) -> dict[str, Any]:
    """Measured soak numbers from a ``runs/<tag>/`` directory.

    The traffic phases are the ``stats_after_soak*`` snapshots; each phase's counters are
    the delta against the previous snapshot (the first one's absolute values, because the
    process started at zero). ``uptime_s`` differences give the phase's wall duration, so
    the request rate needs neither ``driver.log`` nor the client's own clock.
    """
    run = Path(os.path.expanduser(run_dir))
    docs = _snapshot_order(sorted(glob.glob(str(run / "stats_*.json"))))
    phases: list[dict[str, Any]] = []
    prev_flat: dict[str, int] | None = None
    prev_uptime = 0.0
    for path, doc in docs:
        flat = AZ._flat(doc)
        uptime = float(doc.get("uptime_s") or 0)
        if prev_flat is None:
            delta = {k: v for k, v in flat.items() if k != "uptime_s"}
            duration = uptime
        else:
            delta = {k: flat[k] - prev_flat.get(k, 0) for k in flat if k != "uptime_s"}
            duration = uptime - prev_uptime
        prev_flat, prev_uptime = flat, uptime
        name = Path(path).stem
        if not name.startswith("stats_after_soak"):
            continue  # probe snapshots carry no soak traffic
        reqs = delta.get("requests.completed", 0)
        phases.append({
            "phase": name.replace("stats_after_", ""),
            "requests": reqs,
            "duration_s": max(0.0, duration),
            "prompt_tokens": delta.get("requests.prompt_tokens_total", 0),
            "output_tokens": delta.get("requests.completion_tokens_total", 0),
            "aborts": sum(v for k, v in delta.items() if k.startswith("requests.aborts.")),
            "refusals": delta.get("scheduler.prefill.refusals", 0),
            "p95_ms": (doc.get("requests") or {}).get("p95_ms", 0),
            "ttft_mean_ms": (doc.get("requests") or {}).get("ttft_mean_ms", 0),
        })

    summaries = []
    for sp in sorted(glob.glob(str(run / "soak*" / "results-*" / "summary.json"))):
        try:
            with open(sp, encoding="utf-8") as fh:
                summaries.append(json.load(fh))
        except (OSError, ValueError):
            continue

    total_req = sum(p["requests"] for p in phases)
    total_s = sum(p["duration_s"] for p in phases)
    out: dict[str, Any] = {
        "run_dir": str(run),
        "phases": phases,
        "summaries": summaries,
        "requests": total_req,
        "duration_s": total_s,
        "req_per_min": (total_req / total_s * 60.0) if total_s > 0 else 0.0,
        "mean_prompt_tokens": (sum(p["prompt_tokens"] for p in phases) / total_req)
        if total_req else 0.0,
        "mean_output_tokens": (sum(p["output_tokens"] for p in phases) / total_req)
        if total_req else 0.0,
        "aborts": sum(p["aborts"] for p in phases),
        "abort_frac": (sum(p["aborts"] for p in phases) / total_req) if total_req else 0.0,
        "p95_ms": max((p["p95_ms"] or 0) for p in phases) if phases else 0,
        "ttft_mean_ms": max((p["ttft_mean_ms"] or 0) for p in phases) if phases else 0,
    }
    if summaries:
        out["client_requests"] = sum(s.get("requests") or 0 for s in summaries)
        out["client_failures"] = sum(s.get("failures") or 0 for s in summaries)
        out["client_p95_ms"] = max(s.get("latency_p95_ms") or 0 for s in summaries)
        out["client_passed"] = all(s.get("passed") is True for s in summaries)
    return out


def soak_baseline(run: dict[str, Any] | None, chars_per_token: float) -> dict[str, Any]:
    """One value per comparable metric, each tagged with where it came from.

    ``None`` for a value means "the soak has no counterpart", which the table prints as
    ``-`` rather than inventing a comparison. That is the difference between this report
    and a dashboard: an unmeasured baseline is stated as unmeasured.
    """
    cfg = SOAK_CLIENT
    base: dict[str, tuple[float | None, str]] = {
        "req_per_min": (None, "client config (closed loop: rate is not a soak input)"),
        "peak_concurrency": (float(cfg["concurrency"]), "client config (--concurrency)"),
        "prompt_tokens_mean": (cfg["prompt_bytes"] / chars_per_token,
                               f"approx: --prompt-bytes/{chars_per_token:g} chars-per-token"),
        "output_tokens_mean": (float(cfg["max_output_tokens"]),
                               "client config (--max-output-tokens, an upper bound)"),
        "cached_frac": (None, "not observable from the soak's own counters"),
        "abort_frac": (float(cfg["max_error_rate"]), "client config (--max-error-rate 0)"),
        "ttft_mean_ms": (None, "no soak run given"),
        "p95_ms": (None, "no soak run given"),
        "sessions": (1.0, "client config (growing-conversation binds sessions)"),
    }
    if run and run.get("requests"):
        base["req_per_min"] = (run["req_per_min"], f"measured: {run['run_dir']}")
        base["prompt_tokens_mean"] = (run["mean_prompt_tokens"],
                                      "measured: prompt_tokens_total delta / completed")
        base["output_tokens_mean"] = (run["mean_output_tokens"],
                                      "measured: completion_tokens_total delta / completed")
        base["abort_frac"] = (run["abort_frac"], "measured: requests.aborts delta / completed")
        base["ttft_mean_ms"] = (float(run["ttft_mean_ms"]), "measured: last phase snapshot")
        base["p95_ms"] = (float(run["p95_ms"]), "measured: last phase snapshot")
    return base


# ---------------------------------------------------------------------------- comparison

#: ``(row label, trace-report key path, baseline key)``. The trace value is read with
#: :func:`_dig` so a nested distribution field is one string here.
COMPARE_ROWS: tuple[tuple[str, str, str], ...] = (
    ("request rate (req/min)", "req_per_min", "req_per_min"),
    ("peak concurrency", "peak_concurrency", "peak_concurrency"),
    ("prompt tokens (mean)", "prompt_tokens.mean", "prompt_tokens_mean"),
    ("output tokens (mean)", "output_tokens.mean", "output_tokens_mean"),
    ("prefix reuse (cached frac)", "cached_frac_agg", "cached_frac"),
    ("abort+disconnect rate", "abort_frac", "abort_frac"),
    ("TTFT mean (ms)", "ttft_ms.mean", "ttft_mean_ms"),
    ("latency p95 (ms)", "duration_ms.p90", "p95_ms"),
)


def _dig(doc: dict[str, Any], path: str) -> float:
    cur: Any = doc
    for part in path.split("."):
        cur = (cur or {}).get(part) if isinstance(cur, dict) else None
    return float(cur or 0.0)


def compare(report: dict[str, Any], base: dict[str, tuple[float | None, str]],
            tolerance: float) -> list[dict[str, Any]]:
    """One row per comparable metric: real, soak, ratio and a hold/differ verdict."""
    rows = []
    for label, key, bkey in COMPARE_ROWS:
        real = _dig(report, key)
        soak, source = base.get(bkey, (None, ""))
        ratio = None
        verdict = "n/a"
        if soak is not None:
            if soak == 0:
                # A zero baseline (the soak requires zero aborts) has no ratio, only a
                # yes/no: anything above zero is already the assumption failing.
                verdict = "HOLDS" if real <= 0 else "DIFFERS"
            else:
                ratio = real / soak
                verdict = "HOLDS" if abs(ratio - 1.0) <= tolerance else "DIFFERS"
        rows.append({"metric": label, "real": real, "soak": soak, "ratio": ratio,
                     "verdict": verdict, "source": source})
    return rows


def _fmt(v: float | None) -> str:
    if v is None:
        return "-"
    if v == 0:
        return "0"
    if abs(v) < 1:
        return f"{v:.3f}"
    if abs(v) < 1000:
        return f"{v:,.1f}"
    return f"{v:,.0f}"


def _phrase(row: dict[str, Any]) -> str:
    """The one-line human form of a differing row -- 'real X is 3.1x the soak's'."""
    r = row["ratio"]
    both = f"({_fmt(row['real'])} vs {_fmt(row['soak'])})"
    if r is None:
        return (f"{row['metric']}: real {_fmt(row['real'])} vs soak {_fmt(row['soak'])} "
                "(the soak requires zero)")
    if r == 0:
        # A real value of exactly zero against a non-zero soak: there is no "x LOWER"
        # factor, and 1/0 is how a report crashes on the good news.
        return f"real {row['metric']} is zero where the soak had {_fmt(row['soak'])}"
    if r >= 1:
        return f"real {row['metric']} is {r:.2f}x the soak's {both}"
    return f"real {row['metric']} is {1 / r:.2f}x LOWER than the soak's {both}"


# -------------------------------------------------------------------------------- output


def _table(heads: list[str], rows: list[list[str]], left: tuple[int, ...] = (0,)) -> str:
    """Numbers right-aligned so magnitudes line up; ``left`` names the text columns."""
    if not rows:
        return "  (none)"
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(heads)]

    def line(cells: list[str]) -> str:
        return "  ".join(c.ljust(w) if i in left else c.rjust(w)
                         for i, (c, w) in enumerate(zip(cells, widths))).rstrip()

    return "\n".join([line(heads)] + [line(r) for r in rows])


def _dist_rows(report: dict[str, Any], keys: list[tuple[str, str]]) -> list[list[str]]:
    rows = []
    for label, key in keys:
        d = report.get(key) or {}
        rows.append([label, str(int(d.get("n", 0))), _fmt(d.get("p50")), _fmt(d.get("p90")),
                     _fmt(d.get("p99")), _fmt(d.get("max")), _fmt(d.get("mean"))])
    return rows


def profile_command(trace: str, report: dict[str, Any], out: str) -> list[str]:
    """The ready-to-run ``trace_to_profile.py`` line, sized from what was measured."""
    agents = max(1, int(report["peak_concurrency"]))
    buckets = min(8, max(3, len(report["routes_split"]) + 3))
    return [
        f"python benchmarks/trace_to_profile.py --trace {trace} \\",
        f"    --out {out} --agents {agents} --buckets {buckets}",
        f"python benchmarks/scheduler_replay.py --profile-file {out} --ticks 4000",
    ]


def render(report: dict[str, Any], rows: list[dict[str, Any]], run: dict[str, Any] | None,
           trace: str, profile_out: str) -> str:
    b = int(report["bucket_s"])
    out: list[str] = []
    out.append(f"=== {trace}")
    out.append(f"{report['requests']} requests ({report['served']} with a prompt-token "
               f"count) over {report['span_s'] / 60:.1f} min")
    out.append(f"routes: {', '.join(report['routes'])}    "
               f"models: {', '.join(report['models'])}    "
               f"streaming: {100 * report['stream_frac']:.0f}%")
    if not report["has_cached"]:
        out.append("NOTE: no request in this trace carries cached_tokens; prefix-reuse rows "
                   "read 0 because the field is absent, not because reuse was zero.")
    out.append("")

    out.append(f"--- load, per {b} s bucket")
    out.append(_table(["series", "n", "p50", "p90", "p99", "max", "mean"],
                      _dist_rows(report, [("arrivals", "rate_per_bucket"),
                                          ("completions", "completions_per_bucket"),
                                          ("active sessions", "sessions_per_bucket")])))
    out.append(f"  overall {report['req_per_min']:.2f} req/min   "
               f"peak concurrency {report['peak_concurrency']}   "
               f"mean in flight {report['mean_inflight']:.2f}")
    out.append("")

    out.append("--- request shape (tokens)")
    out.append(_table(["series", "n", "p50", "p90", "p99", "max", "mean"],
                      _dist_rows(report, [("prompt", "prompt_tokens"),
                                          ("new (prompt-cached)", "fresh_tokens"),
                                          ("output", "output_tokens")])))
    out.append(f"  prefix reuse: aggregate {report['cached_frac_agg']:.3f}, "
               f"median per request {report['cached_frac_median']:.3f}")
    out.append("")

    out.append("--- per-route split")
    out.append(_table(["route", "requests", "share", "prompt p50", "output p50"],
                      [[r["route"], f"{r['requests']:,}", f"{100 * r['share']:.1f}%",
                        _fmt(r["prompt_p50"]), _fmt(r["output_p50"])]
                       for r in report["routes_split"]]))
    out.append("")

    s = report["sessions"]
    out.append("--- sessions")
    out.append(f"  {s['count']} distinct sessions; {100 * s['bound_frac']:.1f}% of requests "
               "carried one")
    out.append(_table(["series", "n", "p50", "p90", "p99", "max", "mean"],
                      [["turns/session", str(int(s["turns"]["n"])), _fmt(s["turns"]["p50"]),
                        _fmt(s["turns"]["p90"]), _fmt(s["turns"]["p99"]),
                        _fmt(s["turns"]["max"]), _fmt(s["turns"]["mean"])],
                       ["lifetime (s)", str(int(s["lifetime_s"]["n"])),
                        _fmt(s["lifetime_s"]["p50"]), _fmt(s["lifetime_s"]["p90"]),
                        _fmt(s["lifetime_s"]["p99"]), _fmt(s["lifetime_s"]["max"]),
                        _fmt(s["lifetime_s"]["mean"])]]))
    out.append("")

    out.append("--- latency and outcomes")
    out.append(_table(["series", "n", "p50", "p90", "p99", "max", "mean"],
                      _dist_rows(report, [("ttft (ms, streaming)", "ttft_ms"),
                                          ("duration (ms)", "duration_ms")])))
    out.append("  status: " + (AZ._kv(report["statuses"]) or "none")
               + f"   abort {100 * report['abort_frac']:.2f}%"
                 f"   error {100 * report['error_frac']:.2f}%")
    out.append("")

    if run:
        out.append(f"--- soak baseline: {run['run_dir']}")
        out.append(_table(["phase", "requests", "duration s", "req/min", "aborts", "refusals"],
                          [[p["phase"], f"{p['requests']:,}", f"{p['duration_s']:.0f}",
                            f"{p['requests'] / p['duration_s'] * 60:.1f}"
                            if p["duration_s"] else "-",
                            f"{p['aborts']:,}", f"{p['refusals']:,}"]
                           for p in run["phases"]]))
        if run.get("summaries"):
            out.append(f"  soak client: {run.get('client_requests')} requests, "
                       f"{run.get('client_failures')} failures, "
                       f"p95 {run.get('client_p95_ms')} ms, "
                       f"passed={run.get('client_passed')}")
    else:
        out.append("--- soak baseline: client config only (no --soak-run given)")
        out.append(f"  concurrency {SOAK_CLIENT['concurrency']}, "
                   f"--max-output-tokens {SOAK_CLIENT['max_output_tokens']}, "
                   f"--prompt-bytes {SOAK_CLIENT['prompt_bytes']}, "
                   f"--request-timeout {SOAK_CLIENT['request_timeout_s']:.0f}s, "
                   f"{SOAK_CLIENT['scenarios']} scenarios")
    out.append("")

    out.append("--- differs from soak by")
    out.append(_table(["metric", "real", "soak", "ratio", "verdict", "soak number from"],
                      [[r["metric"], _fmt(r["real"]), _fmt(r["soak"]),
                        "-" if r["ratio"] is None else f"{r['ratio']:.2f}x",
                        r["verdict"], r["source"]] for r in rows],
                      left=(0, 4, 5)))
    out.append("")

    holds = [r for r in rows if r["verdict"] == "HOLDS"]
    differs = [r for r in rows if r["verdict"] == "DIFFERS"]
    unknown = [r for r in rows if r["verdict"] == "n/a"]
    out.append("=== VERDICT")
    for r in holds:
        out.append(f"  HOLDS    {r['metric']}: real {_fmt(r['real'])} vs soak "
                   f"{_fmt(r['soak'])}")
    for r in differs:
        out.append(f"  DIFFERS  {_phrase(r)}")
    # The tail-vs-mean comparison the table cannot make: the soak's counters give a mean
    # prompt and nothing else, so the real p99 has no like-for-like baseline. Saying so
    # explicitly is more useful than omitting the row that usually matters most.
    soak_mean = next((r["soak"] for r in rows if r["metric"] == "prompt tokens (mean)"), None)
    p99 = report["prompt_tokens"]["p99"]
    if soak_mean:
        out.append(f"  TAIL     real p99 prompt is {p99 / soak_mean:.2f}x the soak's MEAN "
                   f"prompt ({_fmt(p99)} vs {_fmt(soak_mean)} tokens) -- the soak has no "
                   "p99 of its own; only the mean is derivable from its counters")
    if report["sessions"]["count"] == 0:
        out.append("  DIFFERS  no request in this trace was session-bound; the soak's "
                   "growing-conversation scenario assumes leases accumulate")
    for r in unknown:
        out.append(f"  UNKNOWN  {r['metric']}: {r['source']}")
    out.append("")
    out.append("=== turn this trace into a replay-gate profile")
    out.extend("  " + line for line in profile_command(trace, report, profile_out))
    return "\n".join(out)


# ------------------------------------------------------------------------------------ cli


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trace", required=True, help="trace .jsonl file or --trace-dir dir")
    ap.add_argument("--soak-run", default="",
                    help="benchmarks/switchyard_soak/runs/<tag>/ to take the soak's own "
                         "measured numbers from (default: the client config's constants)")
    ap.add_argument("--route", default="", help="only requests on this route")
    ap.add_argument("--bucket", type=float, default=60.0,
                    help="rate/concurrency bucket in seconds (default: 60)")
    ap.add_argument("--tolerance", type=float, default=0.25,
                    help="fractional band around 1.0x inside which an assumption HOLDS "
                         "(default: 0.25, i.e. 0.75x-1.25x)")
    ap.add_argument("--chars-per-token", type=float, default=DEFAULT_CHARS_PER_TOKEN,
                    help="divisor turning the soak's --prompt-bytes into tokens when there "
                         f"is no run to measure (default: {DEFAULT_CHARS_PER_TOKEN:g})")
    ap.add_argument("--profile-out", default="trace.profile.json",
                    help="the --out this report's trace_to_profile.py command suggests")
    ap.add_argument("--json", action="store_true", dest="as_json",
                    help="emit the whole report as JSON instead of text")
    a = ap.parse_args(argv)

    records = list(RT.read_trace(a.trace))
    if a.route:
        records = [r for r in records if (r.get("route") or "") == a.route]
    if not records:
        print(f"no v{RT.TRACE_VERSION} records in {a.trace}", file=sys.stderr)
        return 2

    report = build_trace_report(records, a.bucket)
    run = load_soak_run(a.soak_run) if a.soak_run else None
    base = soak_baseline(run, a.chars_per_token)
    rows = compare(report, base, a.tolerance)
    if a.as_json:
        print(json.dumps({"trace": a.trace, "report": report, "soak": run,
                          "comparison": rows,
                          "profile_command": profile_command(a.trace, report, a.profile_out)},
                         indent=2, sort_keys=True, default=str))
    else:
        print(render(report, rows, run, a.trace, a.profile_out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
