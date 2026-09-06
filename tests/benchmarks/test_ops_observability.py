"""CPU tests for the two production-observability tools.

``benchmarks/ops/stats_sampler.py`` (poll /v1/stats -> JSONL, then per-bucket deltas) and
``benchmarks/trace_load_report.py`` (a real trace against the soak's assumed load).

Torch-free, and nothing here mocks a format: the sampler runs against a stdlib HTTP server
serving a real-shaped ``/v1/stats`` document built from
``freetoken.scheduler.counters.build_scheduler_counters``, and the load report reads a trace
written by the REAL writer (``freetoken/server/request_trace.py``) -- the same arrangement
``test_trace_replay.py`` uses, and for the same reason: a field renamed on one side has to
break here rather than in production.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "benchmarks"))

import trace_load_report as TLR  # noqa: E402


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SS = _load("_ft_stats_sampler", "benchmarks/ops/stats_sampler.py")
RT = TLR.RT


# --------------------------------------------------------------------------- stats doc


def stats_doc(uptime: int, *, completed: int, disconnects: int = 0, errors: int = 0,
              refusals: int = 0, deferred: int = 0, restores_deferred: int = 0,
              memo_hits: int = 0, match_calls: int = 0, active_experts: int = 0,
              missing_experts: int = 0, p95_ms: int = 100, ttft_ms: int = 20,
              violations: int = 0, scheduler: bool = True) -> dict:
    """A ``/v1/stats`` document with the same shape ``server/stats.py`` builds.

    The ``scheduler`` block is produced by the real ``build_scheduler_counters`` off stub
    managers -- the same duck-typed path the low-level scheduler tests use -- so a counter
    that moves in ``counters.py`` moves here too.
    """
    sched = None
    if scheduler:
        sys.path.insert(0, str(_ROOT / "python"))
        from freetoken.scheduler.counters import (  # noqa: PLC0415 -- torch-free by design
            PrefillCounters,
            SpillCounters,
            build_scheduler_counters,
        )

        pre = PrefillCounters()
        pre.passes = 10
        pre.refusals = refusals
        pre.fresh_admits_deferred = deferred
        pre.match_calls = match_calls
        pre.match_memo_hits = memo_hits
        pre.invariant_violations = violations
        spill = SpillCounters()
        spill.spills = 3
        spill.restores = 2
        spill.restores_deferred = restores_deferred

        class _PM:
            counters = pre
            max_chunked_prefills = 4

        class _SS:
            counters = spill

        class _MoE:
            extend_cache_hits = 8
            extend_cache_misses = 2
            extend_cache_tokens = 256

            @staticmethod
            def decode_stat_totals():
                return {"layer_calls": 1, "active": active_experts,
                        "missing": missing_experts, "fetched": missing_experts,
                        "prefill_rows": 0, "prefill_hit_rows": 0,
                        "pageable_stage_calls": 0, "pageable_rows": 0}

        sched = build_scheduler_counters(_PM(), None, _SS(), _MoE(), True)
    return {
        "instance_id": "i-1",
        "model": {"id": "m", "ctx": 131072, "attn": "hybrid_linear", "moe": True},
        "uptime_s": uptime,
        "kv": {"used_pages": 1, "total_pages": 100, "page_size": 64},
        "mamba": None, "swa": None, "vram_bytes": 1,
        "throughput": {"decode_tps": 12.5, "prefill_tps": 0.0},
        "requests": {
            "active": 2, "completed": completed, "p95_ms": p95_ms, "ttft_mean_ms": ttft_ms,
            "prompt_tokens_total": completed * 1000,
            "completion_tokens_total": completed * 100,
            "aborts": {"client_disconnect": disconnects, "explicit": 0, "error": errors},
        },
        "scheduler": sched,
    }


# --------------------------------------------------------------------------- fake server


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):  # keep pytest output clean
        pass

    def do_GET(self):  # noqa: N802
        srv = self.server
        if self.path.startswith("/v1/stats"):
            with srv.lock:
                srv.polls += 1
                doc = srv.docs[min(srv.polls - 1, len(srv.docs) - 1)]
            return self._json(doc)
        if self.path.startswith("/v1/requests"):
            return self._json({"entries": [{"ts": "t", "method": "POST",
                                            "path": "/v1/completions", "status": 200,
                                            "model": "m", "duration_ms": 5, "ttft_ms": 1,
                                            "prompt_tokens": 7, "completion_tokens": 2,
                                            "stream": False, "error": None}],
                               "next_cursor": 9})
        self.send_error(404)

    def _json(self, doc):
        body = json.dumps(doc).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    srv.lock = threading.Lock()
    srv.polls = 0
    srv.docs = [stats_doc(10, completed=1), stats_doc(70, completed=5)]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv
    srv.shutdown()
    srv.server_close()


# --------------------------------------------------------------------- sampler: capture


def test_sample_once_records_the_document(server):
    url = f"http://127.0.0.1:{server.server_address[1]}"
    rec, cursor = SS.sample_once(url, 5.0)
    assert rec["ok"] is True
    assert rec["stats"]["requests"]["completed"] == 1
    assert cursor is None  # request pulling off by default
    assert "requests" not in rec


def test_sample_once_pulls_requests_and_advances_the_cursor(server):
    url = f"http://127.0.0.1:{server.server_address[1]}"
    rec, cursor = SS.sample_once(url, 5.0, cursor=0)
    assert cursor == 9
    assert rec["requests"][0]["prompt_tokens"] == 7


def test_sample_once_never_raises_when_the_server_is_down():
    """The outage is the datum. A sampler that propagated this would stop recording the
    only interval anyone cares about."""
    rec, cursor = SS.sample_once("http://127.0.0.1:1", 0.5, cursor=3)
    assert rec["ok"] is False
    assert "error" in rec and rec["error"]
    assert cursor == 3  # unchanged: never skip rows we did not read


def test_append_record_expands_strftime_and_rotates_by_day(tmp_path):
    pattern = str(tmp_path / "%Y-%m-%d.jsonl")
    day1 = time.mktime((2026, 9, 6, 12, 0, 0, 0, 0, -1))
    day2 = time.mktime((2026, 9, 7, 12, 0, 0, 0, 0, -1))
    p1 = SS.append_record(pattern, {"ts": day1, "ok": True})
    p2 = SS.append_record(pattern, {"ts": day1, "ok": True})
    p3 = SS.append_record(pattern, {"ts": day2, "ok": True})
    assert p1 == p2 != p3
    assert Path(p1).read_text().count("\n") == 2
    assert Path(p3).read_text().count("\n") == 1


def test_run_sampler_writes_n_lines_and_stops(server, tmp_path):
    out = tmp_path / "s.jsonl"
    a = SS.build_parser().parse_args(
        ["sample", "--base-url", f"http://127.0.0.1:{server.server_address[1]}",
         "--out", str(out), "--interval", "0.05", "--samples", "2", "--requests"])
    assert SS.run_sampler(a) == 0
    lines = [json.loads(x) for x in out.read_text().splitlines()]
    assert len(lines) == 2
    assert [x["stats"]["requests"]["completed"] for x in lines] == [1, 5]


def test_run_sampler_survives_a_dead_server(tmp_path):
    out = tmp_path / "s.jsonl"
    a = SS.build_parser().parse_args(
        ["sample", "--base-url", "http://127.0.0.1:1", "--out", str(out),
         "--interval", "0.05", "--samples", "2", "--timeout", "0.5"])
    assert SS.run_sampler(a) == 0
    lines = [json.loads(x) for x in out.read_text().splitlines()]
    assert len(lines) == 2 and all(x["ok"] is False for x in lines)


# ------------------------------------------------------------------- sampler: summarize


def _samples(tmp_path, docs, step=60.0, t0=None):
    t0 = t0 or time.mktime((2026, 9, 6, 0, 0, 0, 0, 0, -1))
    path = tmp_path / "2026-09-06.jsonl"
    with path.open("w") as fh:
        for i, doc in enumerate(docs):
            rec = {"ts": t0 + i * step,
                   "iso": time.strftime("%Y-%m-%dT%H:%M:%S%z",
                                        time.localtime(t0 + i * step))}
            if doc is None:
                rec["ok"] = False
                rec["error"] = "URLError: refused"
            else:
                rec["ok"] = True
                rec["stats"] = doc
            fh.write(json.dumps(rec) + "\n")
    return path


def test_summarize_differences_cumulative_counters(tmp_path, capsys):
    docs = [stats_doc(60 * i, completed=10 * i, disconnects=i, refusals=2 * i,
                      restores_deferred=5 * i) for i in range(1, 6)]
    path = _samples(tmp_path, docs)
    recs = SS.read_samples([str(path)])
    assert len(recs) == 5
    buckets = SS.bucket_samples(recs, 3600.0)
    assert len(buckets) == 1
    d = buckets[0].delta
    # Four intervals of +10 completed, +1 disconnect, +2 refusals, +5 restores_deferred.
    assert d["requests.completed"] == 40
    assert d["requests.aborts.client_disconnect"] == 4
    assert d["scheduler.prefill.refusals"] == 8
    assert d["scheduler.session_spill.restores_deferred"] == 20
    assert SS.summarize(recs, 3600.0) == 0
    text = capsys.readouterr().out
    assert "restdef" in text and "40" in text


def test_summarize_buckets_by_hour(tmp_path):
    docs = [stats_doc(1800 * i, completed=10 * i) for i in range(1, 7)]
    recs = SS.read_samples([str(_samples(tmp_path, docs, step=1800.0))])
    buckets = SS.bucket_samples(recs, 3600.0)
    assert [b.samples for b in buckets] == [2, 2, 2]
    # The delta belongs to the bucket of the LATER sample, so the first bucket holds only
    # the one interval that ends inside it.
    assert buckets[0].delta["requests.completed"] == 10
    assert buckets[1].delta["requests.completed"] == 20


def test_restart_is_detected_and_counted_from_zero(tmp_path):
    """uptime_s going backwards means a new process whose counters restarted at 0.
    Differencing across it would emit a large negative delta."""
    docs = [stats_doc(600, completed=100), stats_doc(660, completed=120),
            stats_doc(30, completed=5), stats_doc(90, completed=9)]
    recs = SS.read_samples([str(_samples(tmp_path, docs))])
    b = SS.bucket_samples(recs, 3600.0)[0]
    assert b.restarts == 1
    # 20 before the restart, then the new process's 5 absolute, then +4.
    assert b.delta["requests.completed"] == 29
    assert all(v >= 0 for v in b.delta.values())


def test_outage_resets_the_baseline(tmp_path):
    docs = [stats_doc(60, completed=10), None, stats_doc(600, completed=900)]
    recs = SS.read_samples([str(_samples(tmp_path, docs))])
    b = SS.bucket_samples(recs, 3600.0)[0]
    assert b.down == 1
    # Nothing is attributed across the gap: the work could have happened at any point in it.
    assert b.delta.get("requests.completed", 0) == 0


def test_ratios_come_from_the_window_not_the_lifetime(tmp_path):
    """MoE decode hit rate is 1 - missing/active over the DELTAS, which is why counters.py
    publishes raw counts and never a pre-divided ratio."""
    docs = [stats_doc(60, completed=1, active_experts=100, missing_experts=50),
            stats_doc(120, completed=2, active_experts=200, missing_experts=60)]
    recs = SS.read_samples([str(_samples(tmp_path, docs))])
    b = SS.bucket_samples(recs, 3600.0)[0]
    # window: active +100, missing +10 -> 90 %, not the lifetime 200/60 = 70 %.
    assert SS.ratio(b, "scheduler.moe.decode.active",
                    "scheduler.moe.decode.missing", "shortfall") == pytest.approx(90.0)


def test_gauges_are_summarized_as_a_distribution_not_differenced(tmp_path):
    docs = [stats_doc(60 * i, completed=i, p95_ms=100 * i, ttft_ms=10 * i)
            for i in range(1, 5)]
    recs = SS.read_samples([str(_samples(tmp_path, docs))])
    b = SS.bucket_samples(recs, 3600.0)[0]
    assert b.p95_ms == [100.0, 200.0, 300.0, 400.0]
    assert SS._pct(b.p95_ms, 0.5) == pytest.approx(250.0)
    assert "requests.p95_ms" not in b.delta


def test_summarize_reports_an_engine_that_publishes_no_scheduler_block(tmp_path, capsys):
    docs = [stats_doc(60, completed=1, scheduler=False),
            stats_doc(120, completed=3, scheduler=False)]
    recs = SS.read_samples([str(_samples(tmp_path, docs))])
    assert SS.summarize(recs, 3600.0) == 0
    assert "NOT REPORTED" in capsys.readouterr().out


def test_summarize_json_and_directory_input(tmp_path, capsys):
    docs = [stats_doc(60, completed=1), stats_doc(120, completed=4)]
    _samples(tmp_path, docs)
    assert SS.main(["summarize", str(tmp_path), "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["samples"] == 2
    assert doc["buckets"][0]["delta"]["requests.completed"] == 3


def test_summarize_tolerates_a_truncated_last_line(tmp_path):
    path = _samples(tmp_path, [stats_doc(60, completed=1)])
    with path.open("a") as fh:
        fh.write('{"ts": 1, "ok": tru')
    assert len(SS.read_samples([str(path)])) == 1


def test_gauges_and_high_water_marks_are_not_differenced():
    """``_flat`` flattens every int, so the counter/gauge distinction lives in the sampler.
    Differencing a gauge is how a report grows a 'requests.p95_ms +300' column."""
    assert SS.is_counter("requests.completed")
    assert SS.is_counter("scheduler.session_spill.restores_deferred")
    for gauge in ("uptime_s", "requests.p95_ms", "requests.active", "kv.used_pages",
                  "scheduler.prefill.chunked_inflight_max",
                  "scheduler.prefill.match.tokens_per_pass",
                  "scheduler.prefill.invariant.worst_shortfall"):
        assert not SS.is_counter(gauge), gauge


@pytest.mark.parametrize("argv", [["--help"], ["sample", "--help"], ["summarize", "--help"]])
def test_help_renders(argv):
    with pytest.raises(SystemExit) as exc:
        SS.main(argv)
    assert exc.value.code == 0


def test_sample_help_shows_the_rotating_default(capsys):
    """argparse %-formats every help string, so the strftime default has to be doubled in
    the help text; an un-doubled ``%Y`` made ``--help`` itself raise ValueError."""
    with pytest.raises(SystemExit):
        SS.main(["sample", "--help"])
    assert "%Y-%m-%d.jsonl" in capsys.readouterr().out


def test_analyze_is_importable_without_running_its_argv_loop():
    """The guard this work added to analyze.py. Without it, importing the module would
    re-run its CLI over the *importer's* argv."""
    assert SS.AZ._flat({"a": {"b": 3}, "c": True}) == {"a.b": 3}
    assert callable(SS.AZ.main)


# ---------------------------------------------------------------- trace load report


def _write_trace(tmp_path, *, sessions=3, turns=3, base_prompt=1000, route="/v1/chat/completions"):
    """Drive the real request-trace writer, as test_trace_replay.py does."""
    sys.path.insert(0, str(_ROOT / "python"))
    trace_dir = tmp_path / "trace"
    RT._reset_for_tests()
    RT.configure(str(trace_dir))
    t = 1_000_000.0
    for s in range(sessions):
        msgs = [{"role": "system", "content": f"sys {s}"}]
        for k in range(turns):
            msgs = msgs + [{"role": "user", "content": f"turn {k} " + "x " * (50 * (k + 1))}]
            prompt = base_prompt * (k + 1) + 100 * s
            RT.record(route=route, arrival=t, messages=msgs,
                      request_id=f"r{s}-{k}", model="m", session_id=f"s{s}", stream=True,
                      prompt_tokens=prompt, cached_tokens=prompt // 2,
                      max_tokens=512, output_tokens=64 + 8 * k,
                      ttft=t + 0.2, finished=t + 2.0, finish_reason="stop", status="ok")
            t += 5.0
    RT.flush()
    RT.close()
    RT._reset_for_tests()
    return trace_dir


@pytest.fixture
def trace_dir(tmp_path):
    return _write_trace(tmp_path)


def test_report_reads_a_real_trace(trace_dir, capsys):
    assert TLR.main(["--trace", str(trace_dir)]) == 0
    text = capsys.readouterr().out
    assert "differs from soak by" in text
    assert "=== VERDICT" in text
    assert "trace_to_profile.py" in text
    assert "scheduler_replay.py" in text


def test_report_shapes_match_the_trace(trace_dir):
    recs = list(RT.read_trace(str(trace_dir)))
    rep = TLR.build_trace_report(recs, 60.0)
    assert rep["requests"] == 9 and rep["served"] == 9
    assert rep["sessions"]["count"] == 3
    assert rep["sessions"]["turns"]["p50"] == 3
    assert rep["prompt_tokens"]["max"] == 3200  # 1000*3 + 100*2
    # cached_tokens is prompt//2 on every record, so both reuse readings must be 0.5.
    assert rep["cached_frac_agg"] == pytest.approx(0.5, abs=1e-3)
    assert rep["cached_frac_median"] == pytest.approx(0.5, abs=1e-3)
    assert rep["fresh_tokens"]["mean"] == pytest.approx(rep["prompt_tokens"]["mean"] / 2,
                                                       rel=1e-3)
    assert rep["ttft_ms"]["p50"] == pytest.approx(200.0, rel=1e-3)


def test_per_bucket_counts_arrivals_completions_and_sessions():
    recs = [{"t": 0.0, "duration_ms": 1000.0, "session": "a"},
            {"t": 30.0, "duration_ms": 90_000.0, "session": "b"},
            {"t": 65.0, "duration_ms": 1000.0, "session": "a"}]
    b = TLR.per_bucket(recs, 60.0)
    assert b["arrivals"] == [2, 1, 0]
    # The 90 s request arrives in bucket 0 and completes in bucket 2.
    assert b["completions"] == [1, 1, 1]
    # b spans all three buckets; a is active in 0 and 1.
    assert b["sessions"] == [2, 2, 1]


def test_compare_flags_a_load_that_differs():
    report = {"req_per_min": 120.0, "peak_concurrency": 48,
              "prompt_tokens": {"mean": 40_000.0, "p99": 128_000.0},
              "output_tokens": {"mean": 900.0},
              "cached_frac_agg": 0.8, "abort_frac": 0.02,
              "ttft_ms": {"mean": 500.0}, "duration_ms": {"p90": 9000.0}}
    base = TLR.soak_baseline(None, 4.0)
    rows = {r["metric"]: r for r in TLR.compare(report, base, 0.25)}
    assert rows["peak concurrency"]["verdict"] == "DIFFERS"
    assert rows["peak concurrency"]["ratio"] == pytest.approx(3.0)
    # 40 000 vs 16 384/4 = 4 096.
    assert rows["prompt tokens (mean)"]["ratio"] == pytest.approx(40_000 / 4096)
    # The soak requires zero aborts, so there is no ratio -- only a yes/no.
    assert rows["abort+disconnect rate"]["verdict"] == "DIFFERS"
    assert rows["abort+disconnect rate"]["ratio"] is None
    # No run dir, so these have no baseline at all and must not be invented.
    assert rows["latency p95 (ms)"]["verdict"] == "n/a"
    assert "3.00x" in TLR._phrase(rows["peak concurrency"])


def test_compare_holds_when_the_trace_matches_the_soak():
    report = {"req_per_min": 0.0, "peak_concurrency": 16,
              "prompt_tokens": {"mean": 4096.0, "p99": 4096.0},
              "output_tokens": {"mean": 256.0}, "cached_frac_agg": 0.5, "abort_frac": 0.0,
              "ttft_ms": {"mean": 0.0}, "duration_ms": {"p90": 0.0}}
    rows = {r["metric"]: r for r in TLR.compare(report, TLR.soak_baseline(None, 4.0), 0.25)}
    assert rows["peak concurrency"]["verdict"] == "HOLDS"
    assert rows["prompt tokens (mean)"]["verdict"] == "HOLDS"
    assert rows["output tokens (mean)"]["verdict"] == "HOLDS"
    assert rows["abort+disconnect rate"]["verdict"] == "HOLDS"


def _soak_run(tmp_path):
    """A runs/<tag>/ with the two phase snapshots run.sh curls, plus a probe snapshot in
    between whose name does not sort into chronological order."""
    run = tmp_path / "runs" / "tag"
    run.mkdir(parents=True)
    (run / "stats_after_soakStage.json").write_text(json.dumps(
        stats_doc(1200, completed=600, p95_ms=3000, ttft_ms=400)))
    (run / "stats_after_soakPass.json").write_text(json.dumps(
        stats_doc(2400, completed=1500, p95_ms=4000, ttft_ms=500, disconnects=3)))
    (run / "stats_after_probe.json").write_text(json.dumps(
        stats_doc(2500, completed=1502, disconnects=5)))
    results = run / "soakStage" / "results-1"
    results.mkdir(parents=True)
    (results / "summary.json").write_text(json.dumps(
        {"passed": True, "requests": 900, "failures": 0, "latency_p95_ms": 3100}))
    return run


def test_soak_run_deltas_come_from_the_snapshots(tmp_path):
    run = TLR.load_soak_run(str(_soak_run(tmp_path)))
    assert [p["phase"] for p in run["phases"]] == ["soakStage", "soakPass"]
    assert [p["requests"] for p in run["phases"]] == [600, 900]
    assert [p["duration_s"] for p in run["phases"]] == [1200.0, 1200.0]
    assert run["requests"] == 1500
    assert run["req_per_min"] == pytest.approx(1500 / 2400 * 60)
    # prompt_tokens_total is completed*1000 in the fixture, so the mean is exactly 1000.
    assert run["mean_prompt_tokens"] == pytest.approx(1000.0)
    assert run["mean_output_tokens"] == pytest.approx(100.0)
    assert run["abort_frac"] == pytest.approx(3 / 1500)
    assert run["client_requests"] == 900 and run["client_passed"] is True


def test_snapshots_are_ordered_by_uptime_not_filename(tmp_path):
    """``stats_after_soakPass`` sorts before ``stats_after_soakStage`` alphabetically, and
    a copied run directory has no usable mtimes. uptime_s is the only real chronology."""
    run = _soak_run(tmp_path)
    ordered = TLR._snapshot_order(sorted(str(p) for p in run.glob("stats_*.json")))
    assert [d["uptime_s"] for _p, d in ordered] == [1200, 2400, 2500]


def test_report_against_a_soak_run_uses_measured_numbers(trace_dir, tmp_path, capsys):
    run = _soak_run(tmp_path)
    assert TLR.main(["--trace", str(trace_dir), "--soak-run", str(run)]) == 0
    text = capsys.readouterr().out
    assert "measured: prompt_tokens_total delta / completed" in text
    assert "soak client: 900 requests" in text
    assert "TAIL" in text


def test_report_json_mode(trace_dir, capsys):
    assert TLR.main(["--trace", str(trace_dir), "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["report"]["requests"] == 9
    assert doc["soak"] is None
    assert any("trace_to_profile.py" in line for line in doc["profile_command"])


def test_profile_command_is_sized_from_the_trace(trace_dir):
    recs = list(RT.read_trace(str(trace_dir)))
    rep = TLR.build_trace_report(recs, 60.0)
    cmd = " ".join(TLR.profile_command(str(trace_dir), rep, "p.json"))
    assert f"--agents {rep['peak_concurrency']}" in cmd
    assert "--out p.json" in cmd


def test_report_help_renders():
    with pytest.raises(SystemExit) as exc:
        TLR.main(["--help"])
    assert exc.value.code == 0


def test_report_refuses_an_empty_trace(tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert TLR.main(["--trace", str(empty)]) == 2
    assert "no v" in capsys.readouterr().err


def test_report_flags_a_trace_with_no_sessions(tmp_path, capsys):
    d = tmp_path / "t"
    RT._reset_for_tests()
    RT.configure(str(d))
    for i in range(4):
        RT.record(route="/v1/completions", arrival=1_000_000.0 + i,
                  messages=[f"prompt {i}"], model="m", session_id=None, stream=False,
                  prompt_tokens=500, cached_tokens=0, output_tokens=20,
                  finished=1_000_000.0 + i + 1, status="abort")
    RT.flush()
    RT.close()
    RT._reset_for_tests()
    assert TLR.main(["--trace", str(d)]) == 0
    text = capsys.readouterr().out
    assert "no request in this trace was session-bound" in text
    assert "cached_tokens" in text  # the "reuse reads 0 because the field is absent" note
    assert "abort" in text
