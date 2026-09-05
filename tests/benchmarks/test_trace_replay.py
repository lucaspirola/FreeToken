"""Trace capture -> replay -> metrics, and trace -> scheduler_replay profile.

Torch-free (except the one test that actually drives ``scheduler_replay``, which skips
without it): the trace is written by the REAL writer
(``python/freetoken/server/request_trace.py``), replayed by the REAL
``benchmarks/trace_replay.py`` against a stdlib HTTP server that speaks just enough
OpenAI SSE, and converted by the REAL ``benchmarks/trace_to_profile.py``. Nothing here
mocks the format, so a field renamed on one side breaks here rather than in a soak.

The fake server implements a *deterministic* tokenizer -- ``prompt_tokens = words +
7*messages + 3`` -- which is what makes the fidelity assertions meaningful: the replay's
three calibration probes must recover slope 1.0, intercept 3 and 7 tokens per turn,
and the fitted ``scale``
must then reproduce the traced prompt lengths to within a few percent.
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "benchmarks"))

import trace_replay as TRP  # noqa: E402
import trace_to_profile as T2P  # noqa: E402

RT = TRP.RT


# --------------------------------------------------------------------------- fake server


def fake_prompt_tokens(messages) -> int:
    words = sum(len(str(m.get("content", "")).split()) for m in messages)
    return words + 7 * len(messages) + 3


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):  # keep pytest output clean
        pass

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        sid = self.headers.get("x-switchyard-session-id")
        srv = self.server
        with srv.lock:
            srv.seen.append({"session": sid, "path": self.path,
                             "prompt_tokens": fake_prompt_tokens(body.get("messages") or []),
                             "max_tokens": body.get("max_tokens"),
                             "model": body.get("model"),
                             "temperature": body.get("temperature")})
            if sid:
                srv.live[sid] = srv.live.get(sid, 0) + 1
                srv.max_live[sid] = max(srv.max_live.get(sid, 0), srv.live[sid])
        try:
            pt = fake_prompt_tokens(body.get("messages") or [])
            n_out = min(int(body.get("max_tokens") or 4), 4)
            usage = {"prompt_tokens": pt, "completion_tokens": n_out,
                     "total_tokens": pt + n_out,
                     "prompt_tokens_details": {"cached_tokens": pt // 2}}
            if body.get("stream"):
                self._sse(n_out, usage)
            else:
                self._json({"id": "chatcmpl-1", "object": "chat.completion",
                            "choices": [{"index": 0, "finish_reason": "stop",
                                         "message": {"role": "assistant", "content": "ok"}}],
                            "usage": usage})
        finally:
            with srv.lock:
                if sid:
                    srv.live[sid] -= 1

    def _json(self, payload):
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _sse(self, n_out, usage):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        chunks = [{"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""},
                                "finish_reason": None}]}]
        chunks += [{"choices": [{"index": 0, "delta": {"content": "tok"},
                                 "finish_reason": None}]} for _ in range(n_out)]
        chunks.append({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
        chunks.append({"choices": [], "usage": usage})
        payload = b"".join(f"data: {json.dumps(c)}\n\n".encode() for c in chunks)
        payload += b"data: [DONE]\n\n"
        self.wfile.write(b"%x\r\n%s\r\n0\r\n\r\n" % (len(payload), payload))


@pytest.fixture()
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    srv.lock = threading.Lock()
    srv.seen, srv.live, srv.max_live = [], {}, {}
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    srv.base_url = f"http://127.0.0.1:{srv.server_address[1]}"
    yield srv
    srv.shutdown()
    srv.server_close()


# --------------------------------------------------------------------------- a trace


def _write_trace(tmp_path, *, sessions=3, turns=4, include_text=False):
    """Capture a synthetic conversation workload THROUGH THE REAL WRITER."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_ft_rt_bench", _ROOT / "python" / "freetoken" / "server" / "request_trace.py")
    rt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rt)
    d = tmp_path / "trace"
    rt.configure(str(d), include_text=include_text)
    t = 1_000_000.0
    for s in range(sessions):
        # Two families of system prompt, so the trace has cross-session prefix sharing.
        history = [{"role": "system", "content": f"system prompt family {s % 2} " + "x " * 40}]
        for turn in range(turns):
            history = history + [{"role": "user", "content": f"turn {turn} " + "word " * (30 * (turn + 1))}]
            pt = fake_prompt_tokens(history)
            trace = rt.Trace("/v1/chat/completions", list(history),
                             "unit-model", f"sess-{s}", True, 64,
                             {"temperature": 0.7, "top_p": 0.95})
            trace.arrival = t
            trace.ttft = t + 0.05
            trace.seal(request_id=f"chatcmpl-{s}-{turn}", prompt_tokens=pt,
                       cached_tokens=fake_prompt_tokens(history[:-1]) if turn else 0,
                       output_tokens=16, finish_reason="stop", finished=t + 0.3)
            history = history + [{"role": "assistant", "content": "reply " * 16}]
            t += 0.05
    assert rt.flush(timeout=5.0)
    rt.close()
    rt._reset_for_tests()
    return d


@pytest.fixture()
def trace_dir(tmp_path):
    return _write_trace(tmp_path)


def test_capture_produces_a_readable_trace(trace_dir):
    recs = list(RT.read_trace(str(trace_dir)))
    assert len(recs) == 12
    assert {r["session"] for r in recs} == {"sess-0", "sess-1", "sess-2"}
    assert all(r["status"] == "ok" for r in recs)
    assert recs == sorted(recs, key=lambda r: r["t"])


# --------------------------------------------------------------------------- reconstruction


def test_dry_run_reconstructs_lengths_and_prefix_sharing(trace_dir, tmp_path, capsys):
    out = tmp_path / "dry.json"
    assert TRP.main(["--trace", str(trace_dir), "--dry-run", "--out", str(out)]) == 0
    res = json.loads(out.read_text())
    f = res["fidelity"]
    assert res["records"] == 12 and res["sessions"] == 3
    assert f["predicted_prompt_tokens_err_p50"] < 0.10
    # The traced traffic is conversational, so most of what is sent is a repeat of a
    # message an earlier turn already sent. If this collapses to ~0 the pure-length rule
    # has broken and the replay is measuring a cold cache against a warm trace.
    assert f["reconstructed_shared_word_frac"] > 0.5
    assert res["original"]["cached_frac"] > 0.4
    capsys.readouterr()


def test_regenerated_prefixes_are_byte_identical_across_turns(trace_dir):
    """The property the whole design rests on: message i of turn k+1 must be the exact
    text message i of turn k was, or the server's radix cache cannot hit."""
    recs = list(RT.read_trace(str(trace_dir)))
    builder = TRP.PromptBuilder(TRP.fit_scale(recs, TRP.Calibration(1.0, 3.0, 7.0)))
    by_session = {}
    for r in recs:
        by_session.setdefault(r["session"], []).append(r)
    for turns in by_session.values():
        built = [builder.build(r) for r in turns]
        for prev, cur in zip(built, built[1:]):
            assert len(cur) > len(prev)
            assert cur[: len(prev)] == prev
    # And across sessions: two sessions sharing a system prompt share message 0.
    a = builder.build(by_session["sess-0"][0])
    c = builder.build(by_session["sess-2"][0])
    assert a[0] == c[0], "the shared system prompt must regenerate identically"


def test_stored_text_is_replayed_verbatim(tmp_path):
    d = _write_trace(tmp_path, sessions=1, turns=2, include_text=True)
    recs = list(RT.read_trace(str(d)))
    builder = TRP.PromptBuilder(0.2)
    assert builder.build(recs[0]) == recs[0]["messages"]


# --------------------------------------------------------------------------- replay


def test_replay_against_a_server_reports_both_sides(trace_dir, server, tmp_path):
    out = tmp_path / "replay.json"
    rc = TRP.main(["--trace", str(trace_dir), "--base-url", server.base_url,
                   "--speed", "50", "--out", str(out)])
    assert rc == 0
    res = json.loads(out.read_text())
    # The three-probe fit must recover the fake tokenizer exactly:
    # prompt_tokens = 1*words + 3 + 7*messages.
    cal = res["calibration"]
    assert cal["tokens_per_word"] == pytest.approx(1.0, abs=1e-6)
    assert cal["per_message"] == pytest.approx(7.0, abs=1e-6)
    assert cal["overhead"] == pytest.approx(3.0, abs=1e-6)
    rep, orig = res["replay"], res["original"]
    assert rep["requests"] == orig["requests"] == 12
    assert rep["errors"] == 0
    for key in ("ttft_ms_p50", "ttft_ms_p95", "ttft_ms_p99",
                "latency_ms_p50", "latency_ms_p95", "latency_ms_p99"):
        assert rep[key] is not None and orig[key] is not None
    assert rep["output_tokens"] == 12 * 4
    # The replayed prompts land on the traced lengths, which is the point of the fit.
    assert rep["prompt_tokens_err_p50"] < 0.05
    # 12 replayed requests reached the server, none of them empty.
    assert len(server.seen) == 12 + 3  # + the three calibration probes
    assert all(s["max_tokens"] == 64 for s in server.seen[3:])
    assert all(s["temperature"] == 0.7 for s in server.seen[3:])


def test_replay_preserves_session_affinity_and_serializes_turns(trace_dir, server):
    assert TRP.main(["--trace", str(trace_dir), "--base-url", server.base_url,
                     "--speed", "1000"]) == 0
    sessions = [s["session"] for s in server.seen if s["session"]]
    assert set(sessions) == {"sess-0", "sess-1", "sess-2"}
    assert len(sessions) == 12
    # Turns of one conversation must never overlap: overlapping them would let turn k+1
    # start before turn k's output existed, which is not a conversation and would not
    # reproduce the prefix growth the trace recorded.
    assert max(server.max_live.values()) == 1, server.max_live


def test_replay_records_transport_failures_as_error_rows(trace_dir, server, tmp_path):
    """A dead upstream is a measurement, not a crash: the run must finish and report the
    error rate, which is one of the soak's own pass criteria."""
    port = server.server_address[1]
    server.shutdown()
    server.server_close()
    out = tmp_path / "err.json"
    rc = TRP.main(["--trace", str(trace_dir), "--base-url", f"http://127.0.0.1:{port}",
                   "--tokens-per-word", "1.0", "--template-overhead", "3",
                   "--per-message-overhead", "7", "--timeout", "2",
                   "--speed", "1000", "--out", str(out)])
    assert rc == 0
    res = json.loads(out.read_text())
    assert res["replay"]["errors"] == 12
    assert res["replay"]["error_rate"] == 1.0
    assert res["original"]["errors"] == 0


def test_empty_trace_is_an_error_not_a_traceback(tmp_path):
    (tmp_path / "trace-empty.jsonl").write_text("")
    assert TRP.main(["--trace", str(tmp_path), "--dry-run"]) == 2


# --------------------------------------------------------------------------- profile


def test_trace_to_profile_shape(trace_dir, tmp_path):
    out = tmp_path / "p.json"
    assert T2P.main(["--trace", str(trace_dir), "--out", str(out), "--buckets", "3"]) == 0
    p = json.loads(out.read_text())
    assert p["profile_version"] == T2P.PROFILE_VERSION
    assert 1 <= len(p["scenarios"]) <= 3
    assert sum(row[2] for row in p["scenarios"]) == 12
    assert all(isinstance(row[0], str) and row[1] > 0 for row in p["scenarios"])
    assert p["sessions"] is True
    assert p["turns"] == [4, 4]          # every session ran exactly four turns
    assert p["output_len"] == 16
    assert 0.0 < p["reuse"] < 1.0
    assert p["families"] == 2            # two system-prompt families
    assert p["knobs"]["agents"] >= 1
    assert p["source"]["served"] == 12
    assert p["observed"]["session_count"] == 3
    assert p["width"] > max(row[1] for row in p["scenarios"])


def test_profile_weights_follow_the_length_distribution(tmp_path):
    """A heavy tail must survive bucketing: quantile edges, not equal-width bins."""
    d = _write_trace(tmp_path, sessions=8, turns=5)
    out = tmp_path / "p.json"
    assert T2P.main(["--trace", str(d), "--out", str(out), "--buckets", "5"]) == 0
    p = json.loads(out.read_text())
    lengths = [row[1] for row in p["scenarios"]]
    assert lengths == sorted(lengths)
    assert max(lengths) > 2 * min(lengths)


def test_profile_carries_every_key_scheduler_replay_reads():
    """Torch-free contract check between the two files.

    ``apply_profile_file`` reads these by name and silently keeps its own default for a
    key that is missing, so a rename on either side would not fail -- it would quietly
    replay the Switchyard soak's geometry under a trace's name. Grepping the reader's
    source is the only way to assert this without importing torch.
    """
    src = (_ROOT / "benchmarks" / "scheduler_replay.py").read_text()
    reader = src.split("def apply_profile_file", 1)[1].split("\ndef ", 1)[0]
    assert f"PROFILE_VERSION = {T2P.PROFILE_VERSION}\n" in src, \
        "trace_to_profile and scheduler_replay disagree on the profile version"
    for key in ("scenarios", "output_len", "turn_growth", "families", "client_timeout",
                "session_ttl", "width", "knobs", "profile_version"):
        assert f'"{key}"' in reader, f"apply_profile_file no longer reads {key!r}"
    traffic = src.split("class Traffic", 1)[1].split("\nclass ", 1)[0]
    for key in ("sessions", "reuse", "jitter", "turns"):
        assert f'"{key}"' in traffic, f"Traffic no longer reads {key!r}"


def test_peak_concurrency_sweeps_intervals():
    recs = [{"t": 0.0, "duration_ms": 1000.0},
            {"t": 0.5, "duration_ms": 1000.0},
            {"t": 0.6, "duration_ms": 10.0},
            {"t": 5.0, "duration_ms": 10.0}]
    assert T2P.peak_concurrency(recs) == 3


# --------------------------------------------------------------------------- the gate


def test_profile_drives_scheduler_replay(trace_dir, tmp_path):
    """trace -> profile -> a real scheduler_replay run.

    Skipped without torch: scheduler_replay drives the actual PrefillManager /
    CacheManager, which is the point of it. Kept short (--ticks) -- this asserts the
    profile is *accepted and used*, not a throughput floor.
    """
    pytest.importorskip("torch")
    out = tmp_path / "p.json"
    assert T2P.main(["--trace", str(trace_dir), "--out", str(out),
                     "--pool-pages", "32768", "--agents", "4"]) == 0
    import scheduler_replay as SR

    assert SR.apply_profile_file(str(out)) == "trace"
    assert SR.TRACE_PROFILE is not None
    assert SR.OUTPUT_LEN == 16
    assert SR.PROFILE_KNOBS["trace"]["pool_pages"] == 32768
    traffic = SR.Traffic(7, "trace")
    assert traffic.sessions is True
    uid, name, ids, sid = traffic.next(0)
    assert sid and len(ids) > 0
    res = SR.run(200, 7, profile="trace", pool_pages=32768)
    assert res["fatal"] is None
    assert res["prefilled_tokens"] > 0
