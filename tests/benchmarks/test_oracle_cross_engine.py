"""Cross-engine oracle: verdicts, the report, and the transport, without a GPU.

The comparison logic is what turns two recordings into "reopen the engine bug", so it
is tested against hand-written recordings; the transport (SSE parsing, the logprobs
capability probe, leak accounting across turns) is tested against a throwaway HTTP
server rather than a monkeypatched ``urlopen``, so the streaming parser is really
exercised.
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


BENCH = Path(__file__).parents[2] / "benchmarks"
sys.path.insert(0, str(BENCH))
import bench_multi_needle as mn  # noqa: E402
import oracle_cross_engine as oracle  # noqa: E402


# ------------------------------------------------------------------ fake server


class FakeEngine:
    """A minimal OpenAI-compatible server: /v1/models and a streamed chat completion.

    ``answers`` maps a substring of the last user message to the reply. ``logprobs``
    selects whether the server honours ``top_logprobs`` (llama.cpp-shaped) or rejects
    it with a 400 the way FreeToken's chat endpoint does.
    """

    def __init__(self, answers: dict[str, str], *, logprobs: bool = False,
                 model_id: str = "fake-model") -> None:
        self.answers = answers
        self.logprobs = logprobs
        self.model_id = model_id
        self.requests: list[dict] = []
        engine = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_GET(self):
                if self.path.endswith("/v1/models"):
                    self._json(200, {"data": [{"id": engine.model_id}]})
                elif self.path.endswith("/health"):
                    self._json(200, {"status": "ok"})
                else:
                    self._json(404, {"error": "no"})

            def do_POST(self):
                body = json.loads(
                    self.rfile.read(int(self.headers["Content-Length"] or 0)) or b"{}"
                )
                engine.requests.append(body)
                if body.get("top_logprobs") and not engine.logprobs:
                    self._json(400, {"error": {
                        "message": "logprobs are not supported; omit top_logprobs"}})
                    return
                text = engine.reply_for(body)
                if not body.get("stream"):
                    payload = {"choices": [{"message": {"content": text}}]}
                    if body.get("top_logprobs"):
                        payload["choices"][0]["logprobs"] = {
                            "content": engine.logprob_entries(text)}
                    self._json(200, payload)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                entries = engine.logprob_entries(text) if body.get("top_logprobs") else []
                for index, piece in enumerate(text.split(" ")):
                    chunk = {"choices": [{"delta": {
                        "content": piece if index == 0 else " " + piece}}]}
                    if index < len(entries):
                        chunk["choices"][0]["logprobs"] = {"content": [entries[index]]}
                    self._write(chunk)
                self._write({"choices": [], "usage": {
                    "prompt_tokens": 1000 + len(json.dumps(body["messages"])),
                    "completion_tokens": len(text.split(" ")),
                    "prompt_tokens_details": {"cached_tokens": 900}}})
                self.wfile.write(b"data: [DONE]\n\n")

            def _write(self, chunk):
                self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
                self.wfile.flush()

            def _json(self, status, payload):
                blob = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(blob)))
                self.end_headers()
                self.wfile.write(blob)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def reply_for(self, body: dict) -> str:
        last = body["messages"][-1]["content"]
        for needle, answer in self.answers.items():
            if needle in last:
                return answer
        return "I do not know."

    def logprob_entries(self, text: str) -> list[dict]:
        return [{"token": word, "logprob": -0.5 * (index + 1),
                 "top_logprobs": [{"token": word, "logprob": -0.5 * (index + 1)}]}
                for index, word in enumerate(text.split(" "))]

    @property
    def origin(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_exc):
        self.server.shutdown()
        self.server.server_close()


# ------------------------------------------------------------- capability probe


def test_a_server_that_rejects_top_logprobs_is_reported_unsupported():
    with FakeEngine({}, logprobs=False) as engine:
        capability = oracle.probe_logprobs(engine.origin, engine.model_id)
    assert capability["supported"] is False
    assert "400" in capability["reason"]


def test_a_server_that_returns_logprobs_content_is_reported_supported():
    with FakeEngine({"ok": "ok"}, logprobs=True) as engine:
        capability = oracle.probe_logprobs(engine.origin, engine.model_id)
    assert capability["supported"] is True


def test_a_server_that_accepts_and_ignores_top_logprobs_is_still_unsupported():
    class Silent(FakeEngine):
        def logprob_entries(self, text):
            return []

    with Silent({"ok": "ok"}, logprobs=True) as engine:
        capability = oracle.probe_logprobs(engine.origin, engine.model_id)
    assert capability["supported"] is False
    assert "no logprobs.content" in capability["reason"]


# ------------------------------------------------------------- suite over HTTP


def _perfect_answers() -> dict[str, str]:
    answers = {}
    for needle in mn.NEEDLES:
        answers[f"What is the {needle.name} code?"] = (
            f"The {needle.name} code is {needle.code}."
        )
        answers[f"the code {needle.code}."] = f"That is the {needle.name}."
    for item in mn.questions():
        if item["shape"] == "combined":
            answers[item["text"]] = (
                f"The {item['expect_larger']} one is larger. "
                f"The sum is {item['expect']}."
            )
    answers[f"{mn.CONTROL_KEY} {mn.NEEDLE_KIND} code?"] = (
        f"No {mn.CONTROL_KEY} {mn.NEEDLE_KIND} code appears in the records."
    )
    return answers


def test_a_perfect_engine_passes_every_turn_and_is_graded_recall():
    with FakeEngine(_perfect_answers()) as engine:
        rows, _ = mn.run_suite(engine.origin, engine.model_id, "HAYSTACK",
                               mn.questions(), session_id="t", decode=32, timeout=30)
    assert len(rows) == len(mn.questions())
    failures = [r["question_id"] for r in rows if not r.get("verdict_pass")]
    assert failures == []
    assert {e["class"] for e in mn.classify_all(rows)} == {"recall"}


def test_the_conversation_grows_and_only_turn_one_carries_the_haystack():
    with FakeEngine(_perfect_answers()) as engine:
        mn.run_suite(engine.origin, engine.model_id, "HAYSTACK", mn.questions()[:3],
                     session_id="t", decode=32, timeout=30)
        bodies = engine.requests
    assert "HAYSTACK" in bodies[0]["messages"][0]["content"]
    assert len(bodies[0]["messages"]) == 1
    assert len(bodies[1]["messages"]) == 3  # user, assistant, user
    assert all("HAYSTACK" not in m["content"]
               for m in bodies[1]["messages"][1:])


def test_leak_accounting_marks_a_probe_after_its_code_has_been_printed():
    with FakeEngine(_perfect_answers()) as engine:
        rows, _ = mn.run_suite(engine.origin, engine.model_id, "HAYSTACK",
                               mn.questions(), session_id="t", decode=32, timeout=30)
    by_id = {row["question_id"]: row for row in rows}
    # Directs run first, so each needle's code is fresh when it is asked...
    assert all(by_id[f"direct:{n.key}"]["leak_free"] for n in mn.NEEDLES)
    # ...and every later probe for the same needle is not, once it answered.
    assert not by_id["reverse:orchard"]["leak_free"]
    assert not by_id["combined:orchard+harbour"]["leak_free"]


def test_a_transport_error_stops_the_run_and_is_recorded():
    with FakeEngine({}, logprobs=False) as engine:
        rows, _ = mn.run_suite(engine.origin, engine.model_id, "HAYSTACK",
                               mn.questions(), session_id="t", decode=32, timeout=30,
                               top_logprobs=4)
    assert len(rows) == 1 and "error" in rows[0]
    assert "400" in rows[0]["error"]


def test_streamed_logprobs_are_collected_when_the_engine_emits_them():
    with FakeEngine(_perfect_answers(), logprobs=True) as engine:
        row = mn.chat(engine.origin, engine.model_id, "t",
                      [{"role": "user", "content": "What is the orchard ledger code?"}],
                      32, 30, top_logprobs=4)
    assert [e["token"] for e in row["logprobs"]] == row["text"].split(" ")


def test_generic_prompts_are_graded_against_their_expected_fragments():
    answers = {"17 times 23": "391", "Australia": "Canberra",
               "prime numbers": "2, 3, 5, 7, 11", "def add": "a + b",
               "'benchmark'": "9"}
    with FakeEngine(answers) as engine:
        rows = oracle.run_generic(engine.origin, engine.model_id, decode=16,
                                  timeout=30, top_logprobs=0, session_prefix="t")
    assert len(rows) == len(oracle.GENERIC_PROMPTS)
    assert all(row["verdict_pass"] for row in rows)

    with FakeEngine({}) as engine:
        rows = oracle.run_generic(engine.origin, engine.model_id, decode=16,
                                  timeout=30, top_logprobs=0, session_prefix="t")
    assert not any(row["verdict_pass"] for row in rows)


# ---------------------------------------------------------------- llama-server


def test_the_llama_server_command_carries_the_262k_reference_flags():
    server = oracle.LlamaServer(binary="/bin/llama-server", gguf="/m.gguf", port=8080,
                                ctx=270_336, n_cpu_moe=14, threads=16, log_path="/dev/null",
                                extra_args=["--extra", "1"])
    command = server.command()
    joined = " ".join(command)
    assert command[:3] == ["/bin/llama-server", "-m", "/m.gguf"]
    for fragment in ("-c 270336", "-np 1", "--no-context-shift", "--cache-ram 0",
                     "-ngl 999", "--n-cpu-moe 14", "-fa on", "-ctk q8_0", "-ctv q8_0",
                     "--jinja", "--no-warmup", "--extra 1"):
        assert fragment in joined
    assert server.base_url == "http://127.0.0.1:8080"


def test_a_missing_binary_fails_before_anything_is_launched(tmp_path):
    server = oracle.LlamaServer(binary=str(tmp_path / "nope"), gguf=str(tmp_path),
                                port=1, ctx=8, n_cpu_moe=0, threads=1,
                                log_path=str(tmp_path / "log"))
    try:
        server.__enter__()
    except SystemExit as exit_:
        assert "llama-server not found" in str(exit_)
    else:  # pragma: no cover
        server.__exit__(None, None, None)
        raise AssertionError("a missing binary must not start a run")


# ------------------------------------------------------------------ comparison


def _recording(engine: str, passes: dict[str, bool], *, digest="abc",
               logprobs=False, classification=None, texts=None) -> dict:
    rows = []
    for qid, passed in passes.items():
        rows.append({"question_id": qid, "shape": qid.split(":")[0],
                     "owner": qid.split(":")[-1], "depth": 0.25, "expect": "x",
                     "leak_free": True, "verdict_pass": passed,
                     "verdict_detail": "ok" if passed else "missed",
                     "text": (texts or {}).get(qid, f"{engine}-{qid}-{passed}")})
    return {
        "engine": engine, "label": engine, "model_id": f"{engine}-model",
        "model_dir": "/weights", "generated": "2026-09-04T00:00:00",
        "build": {"gguf": "/m.gguf"},
        "suite": {"target_prompt_tokens": 262_144, "haystack_tokens": 262_100,
                  "haystack_sha256": digest, "filler_cursor": 0,
                  "needles": [{"key": "harbour", "role": "needle",
                               "actual_depth": 0.25}],
                  "question_ids": list(passes)},
        "logprobs": {"supported": logprobs, "reason": "probe"},
        "hidden_states": None, "rows": rows, "generic": [],
        "classification": classification or [],
    }


def test_the_four_verdicts_come_out_of_the_pass_matrix():
    passes_ft = {"direct:a": True, "direct:b": False, "direct:c": False,
                 "direct:d": True}
    passes_lc = {"direct:a": True, "direct:b": True, "direct:c": False,
                 "direct:d": False}
    result = oracle.compare(_recording("freetoken", passes_ft),
                            _recording("llama.cpp", passes_lc), logprob_positions=8)
    verdicts = {e["question_id"]: e["verdict"] for e in result["comparisons"]}
    assert verdicts == {"direct:a": "agree", "direct:b": "freetoken-only-miss",
                        "direct:c": "both-miss", "direct:d": "llamacpp-only-miss"}
    assert result["matrix"] == {"agree": 1, "both-miss": 1, "freetoken-only-miss": 1,
                                "llamacpp-only-miss": 1, "missing": 0}


def test_a_question_only_one_engine_answered_is_missing_not_a_miss():
    result = oracle.compare(_recording("freetoken", {"direct:a": True}),
                            _recording("llama.cpp", {"direct:b": True}),
                            logprob_positions=8)
    verdicts = {e["question_id"]: e["verdict"] for e in result["comparisons"]}
    assert verdicts == {"direct:a": "missing", "direct:b": "missing"}
    assert result["matrix"]["missing"] == 2


def test_an_engine_error_row_is_missing_rather_than_a_silent_failure():
    left = _recording("freetoken", {"direct:a": True})
    left["rows"][0] = {"question_id": "direct:a", "error": "HTTP 500"}
    result = oracle.compare(left, _recording("llama.cpp", {"direct:a": True}),
                            logprob_positions=8)
    assert result["comparisons"][0]["verdict"] == "missing"
    assert result["comparisons"][0]["note"] == "engine error"


def test_recordings_of_different_prompts_are_refused():
    result = oracle.compare(_recording("freetoken", {"direct:a": True}, digest="aaa"),
                            _recording("llama.cpp", {"direct:a": True}, digest="bbb"),
                            logprob_positions=8)
    assert any("haystack_sha256" in line for line in result["prompt_mismatch"])
    assert "void until this is fixed" in oracle.render_markdown(result)


def test_identical_answers_are_flagged_so_a_shared_bug_is_visible():
    texts = {"direct:a": "the same words"}
    result = oracle.compare(
        _recording("freetoken", {"direct:a": False}, texts=texts),
        _recording("llama.cpp", {"direct:a": False}, texts=texts),
        logprob_positions=8)
    assert result["comparisons"][0]["answers_identical"] is True
    assert result["comparisons"][0]["verdict"] == "both-miss"


# -------------------------------------------------------------------- logprobs


def test_logprobs_are_only_compared_when_both_engines_expose_them():
    off = oracle.compare(_recording("freetoken", {"direct:a": True}, logprobs=False),
                         _recording("llama.cpp", {"direct:a": True}, logprobs=True),
                         logprob_positions=8)
    assert off["logprobs_compared"] is False
    assert "logprobs" not in off["comparisons"][0]
    report = oracle.render_markdown(off)
    assert "Not compared." in report
    assert "SamplingParams" in report

    on = oracle.compare(_recording("freetoken", {"direct:a": True}, logprobs=True),
                        _recording("llama.cpp", {"direct:a": True}, logprobs=True),
                        logprob_positions=8)
    assert on["logprobs_compared"] is True
    assert "logprobs" in on["comparisons"][0]


def test_logprob_comparison_finds_the_first_divergent_position():
    left = [{"token": "a", "logprob": -0.1}, {"token": "b", "logprob": -0.2},
            {"token": "c", "logprob": -0.3}]
    right = [{"token": "a", "logprob": -0.3}, {"token": "X", "logprob": -9.0},
             {"token": "c", "logprob": -0.3}]
    result = oracle.compare_logprobs(left, right, positions=8)
    assert result["positions"] == 3
    assert result["top1_agree"] == 2
    assert result["first_divergence"] == {"position": 1, "freetoken": "b",
                                          "llamacpp": "X"}
    assert abs(result["mean_abs_logprob_delta"] - 0.1) < 1e-9


def test_logprob_comparison_is_capped_at_the_requested_positions():
    entries = [{"token": str(i), "logprob": -0.1} for i in range(50)]
    assert oracle.compare_logprobs(entries, entries, positions=4)["positions"] == 4
    assert oracle.compare_logprobs([], [], positions=4)["positions"] == 0


# ---------------------------------------------------------------------- report


def _classification(key, cls, in_state):
    return {"key": key, "class": cls, "in_state": in_state, "evidence": f"{cls} here",
            "probes": {}}


def test_the_report_names_the_quantization_confound_and_the_needle_disagreement():
    result = oracle.compare(
        _recording("freetoken", {"direct:harbour": False},
                   classification=[_classification("harbour", "interference-cross",
                                                   True)]),
        _recording("llama.cpp", {"direct:harbour": True},
                   classification=[_classification("harbour", "recall", True)]),
        logprob_positions=8)
    report = oracle.render_markdown(result)

    assert "engine and quantization move together" in report
    assert "`freetoken-only-miss`" in report
    assert "interference-cross" in report and "recall" in report
    assert "**harbour**" in report  # the disagreement is called out by name
    # The agreement matrix must be present and correctly oriented.
    assert "| **FreeToken FAIL** | 1 | 0 |" in report


def test_the_report_survives_a_recording_with_no_classification_or_needles():
    result = oracle.compare(_recording("freetoken", {"generic:arith": True}),
                            _recording("llama.cpp", {"generic:arith": True}),
                            logprob_positions=8)
    report = oracle.render_markdown(result)
    assert "## Needle classification" in report
    assert "## Verdicts" in report


def test_compare_main_writes_both_artifacts_and_exits_two_on_a_freetoken_miss(tmp_path):
    left = tmp_path / "ft.json"
    right = tmp_path / "lc.json"
    left.write_text(json.dumps(_recording("freetoken", {"direct:a": False})))
    right.write_text(json.dumps(_recording("llama.cpp", {"direct:a": True})))
    markdown = tmp_path / "report.md"
    merged = tmp_path / "merged.json"
    code = oracle.main(["compare", "--freetoken", str(left), "--llamacpp", str(right),
                        "--markdown", str(markdown), "--json", str(merged)])
    assert code == 2
    assert "freetoken-only-miss" in markdown.read_text()
    assert json.loads(merged.read_text())["matrix"]["freetoken-only-miss"] == 1


def test_compare_main_exits_zero_when_the_engines_agree(tmp_path):
    left = tmp_path / "ft.json"
    right = tmp_path / "lc.json"
    left.write_text(json.dumps(_recording("freetoken", {"direct:a": True})))
    right.write_text(json.dumps(_recording("llama.cpp", {"direct:a": True})))
    assert oracle.main(["compare", "--freetoken", str(left), "--llamacpp", str(right),
                        "--markdown", str(tmp_path / "r.md")]) == 0


def test_compare_main_exits_three_when_the_prompts_did_not_match(tmp_path):
    left = tmp_path / "ft.json"
    right = tmp_path / "lc.json"
    left.write_text(json.dumps(_recording("freetoken", {"direct:a": True},
                                          digest="aaa")))
    right.write_text(json.dumps(_recording("llama.cpp", {"direct:a": True},
                                           digest="bbb")))
    assert oracle.main(["compare", "--freetoken", str(left), "--llamacpp", str(right),
                        "--markdown", str(tmp_path / "r.md")]) == 3


# ------------------------------------------------------------------ end to end


def test_record_then_compare_produces_a_report_from_two_live_servers(tmp_path,
                                                                    monkeypatch):
    """The whole pipeline on fake engines: build, drive, grade, merge, render.

    llama.cpp answers everything; FreeToken returns the orchard code for the harbour
    question -- the 2026-09-04 1M failure shape -- so the report must land on
    ``freetoken-only-miss`` and classify it as interference rather than retention.
    """
    from test_bench_multi_needle import CharTokenizer

    monkeypatch.setattr(oracle, "load_tokenizer", lambda _dir: CharTokenizer())

    good = _perfect_answers()
    bad = dict(good)
    harbour = next(n for n in mn.NEEDLES if n.key == "harbour")
    orchard = next(n for n in mn.NEEDLES if n.key == "orchard")
    bad[f"What is the {harbour.name} code?"] = (
        f"The {orchard.name} code is {orchard.code}."
    )

    generic = {"17 times 23": "391", "Australia": "Canberra",
               "prime numbers": "2, 3, 5, 7, 11", "def add": "a + b",
               "'benchmark'": "9"}
    recordings = {}
    for engine_name, answers in (("freetoken", {**bad, **generic}),
                                 ("llama.cpp", {**good, **generic})):
        with FakeEngine(answers) as engine:
            out = tmp_path / f"{engine_name.replace('.', '')}.json"
            code = oracle.main([
                "record", "--engine", "freetoken", "--out", str(out),
                "--base-url", engine.origin, "--model-dir", "/fake",
                "--target-prompt-tokens", "40000", "--decode", "32",
                "--timeout", "60", "--label", engine_name,
            ])
        assert code == 0
        recordings[engine_name] = json.loads(out.read_text())
        recordings[engine_name]["engine"] = engine_name
        out.write_text(json.dumps(recordings[engine_name]))

    ft, lc = recordings["freetoken"], recordings["llama.cpp"]
    assert ft["suite"]["haystack_sha256"] == lc["suite"]["haystack_sha256"]
    assert ft["logprobs"]["supported"] is False  # the fake rejects it, as FreeToken does

    result = oracle.compare(ft, lc, logprob_positions=8)
    verdicts = {e["question_id"]: e["verdict"] for e in result["comparisons"]}
    assert verdicts["direct:harbour"] == "freetoken-only-miss"
    assert verdicts["direct:quarry"] == "agree"
    assert verdicts["generic:arith"] == "agree"
    assert result["matrix"]["freetoken-only-miss"] == 1

    by_key = {n["key"]: n for n in result["needles"]}
    assert by_key["harbour"]["freetoken_class"] == "interference-cross"
    assert by_key["harbour"]["freetoken_in_state"] is True
    assert by_key["harbour"]["llamacpp_class"] == "recall"

    report = oracle.render_markdown(result)
    assert "freetoken-only-miss" in report and "**harbour**" in report
    assert "engine and quantization move together" in report
