"""/health must fail when the backend is gone, and the stop must be bounded.

Regression for the two follow-on defects of the Switchyard soak crash: `/health` answered
`{"status":"ok"}` for nine minutes after the scheduler process died (every soak liveness probe
passed against a server that answered nothing), and the ensuing stop wedged for 38 minutes in
uvicorn's untimed "Waiting for background tasks to complete" while still holding the GPU and
~20 GB of pinned expert banks.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from freetoken.server import api_server, control_api


class _Proc:
    """The subset of multiprocessing.Process the server touches."""

    def __init__(self, name: str, alive: bool = True, *, ignores_sigterm: bool = False):
        self.name = name
        self._alive = alive
        self._ignores_sigterm = ignores_sigterm
        self.terminated = False
        self.killed = False

    def is_alive(self):
        return self._alive

    def terminate(self):
        self.terminated = True
        if not self._ignores_sigterm:
            self._alive = False

    def kill(self):
        self.killed = True
        self._alive = False

    def join(self, timeout=None):
        if not self._ignores_sigterm:
            self._alive = False


def _state(**kw):
    base = dict(
        instance_id="inst-1",
        fatal_error=None,
        maintenance_state="serving",
        config=SimpleNamespace(served_model_name="m"),
        ready_at=time.monotonic(),
        load_progress=None,
        backend_processes=[],
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _client(state):
    from fastapi.testclient import TestClient

    api_server._GLOBAL_STATE = state
    return TestClient(api_server.app)


# ------------------------------------------------------------------ /health liveness


def test_health_is_ok_while_every_worker_is_alive():
    doc = control_api.build_health(
        _state(backend_processes=[_Proc("freetoken-TP0-scheduler")]), "v"
    )
    assert doc["status"] == "ok"


def test_health_reports_error_when_the_scheduler_process_is_dead():
    doc = control_api.build_health(
        _state(
            backend_processes=[
                _Proc("freetoken-tokenizer-0"),
                _Proc("freetoken-TP0-scheduler", alive=False),
            ]
        ),
        "v",
    )
    assert doc["status"] == "error"
    assert "freetoken-TP0-scheduler" in doc["message"]


def test_health_liveness_does_not_wait_for_the_supervisor_to_latch_fatal_error():
    """The supervisor thread returns after one failure; a later death must still be seen."""
    state = _state(
        fatal_error=None,                       # never latched: the watch thread already exited
        maintenance_state="serving",            # still "serving", as observed in the soak
        backend_processes=[_Proc("freetoken-TP0-scheduler", alive=False)],
    )
    assert control_api.build_health(state, "v")["status"] == "error"


def test_health_endpoint_returns_503_with_a_reason_when_the_backend_died():
    saved = api_server._GLOBAL_STATE
    try:
        client = _client(
            _state(backend_processes=[_Proc("freetoken-TP0-scheduler", alive=False)])
        )
        r = client.get("/health")
        assert r.status_code == 503
        body = r.json()
        assert body["status"] == "error"
        assert "freetoken-TP0-scheduler" in body["message"]

        client = _client(_state(backend_processes=[_Proc("freetoken-TP0-scheduler")]))
        r = client.get("/health")
        assert r.status_code == 200 and r.json()["status"] == "ok"
    finally:
        api_server._GLOBAL_STATE = saved


def test_health_endpoint_still_answers_200_while_loading():
    """The readiness poll waits for status == "ok"; loading must not become a 503."""
    saved = api_server._GLOBAL_STATE
    try:
        client = _client(
            _state(
                maintenance_state="loading",
                load_progress=SimpleNamespace(phase="weights", done_bytes=1, total_bytes=2),
            )
        )
        r = client.get("/health")
        assert r.status_code == 200 and r.json()["status"] == "loading"
    finally:
        api_server._GLOBAL_STATE = saved


def test_health_endpoint_503s_on_a_supervisor_latched_fatal_error():
    saved = api_server._GLOBAL_STATE
    try:
        client = _client(_state(fatal_error="backend worker x exited", maintenance_state="failed"))
        r = client.get("/health")
        assert r.status_code == 503 and r.json()["message"] == "backend worker x exited"
    finally:
        api_server._GLOBAL_STATE = saved


def test_dead_backend_reason_tolerates_unqueryable_handles():
    class _Broken:
        name = "broken"

        def is_alive(self):
            raise OSError("no such process")

    assert control_api.dead_backend_reason(_state(backend_processes=[_Broken()])) is None
    assert control_api.dead_backend_reason(_state(backend_processes=None)) is None


# ------------------------------------------------------------------ bounded shutdown


def test_uvicorn_graceful_shutdown_is_bounded():
    """uvicorn defaults timeout_graceful_shutdown to None == wait forever. Never here."""
    assert api_server.SHUTDOWN_GRACE_S > 0
    src = Path(api_server.__file__).read_text()
    assert src.count("timeout_graceful_shutdown=SHUTDOWN_GRACE_S") == 2  # ft serve + shell


def test_reap_bounds_the_total_wait_across_workers():
    procs = [_Proc(f"w{i}", ignores_sigterm=True) for i in range(4)]
    start = time.monotonic()
    api_server._reap_backend_workers(procs, timeout=0.2)
    elapsed = time.monotonic() - start
    assert elapsed < 1.0                       # a per-process timeout would be 4x the budget
    assert all(p.killed for p in procs)        # SIGTERM ignored -> SIGKILL, every one of them


def test_frontend_shutdown_terminates_and_reaps_the_workers():
    procs = [_Proc("freetoken-TP0-scheduler", ignores_sigterm=True), _Proc("freetoken-tokenizer-0")]
    state = api_server.FrontendManager.__new__(api_server.FrontendManager)
    state.send_tokenizer = SimpleNamespace(stop=lambda: None)
    state.recv_tokenizer = SimpleNamespace(stop=lambda: None)
    state.backend_processes = procs

    state.shutdown()
    assert all(p.terminated for p in procs)
    assert procs[0].killed                     # the wedged one is SIGKILLed, not left behind
    assert not any(p.is_alive() for p in procs)


def test_backend_death_arms_a_hard_exit_backstop(monkeypatch):
    """The graceful stop can wedge on ASGI tasks a dead backend will never answer."""
    signalled, armed = [], []
    monkeypatch.setattr(api_server.os, "kill", lambda pid, sig: signalled.append(sig))
    monkeypatch.setattr(
        api_server, "_force_exit_after", lambda d: armed.append(d) or threading.Timer(0, lambda: None)
    )
    api_server._SHUTTING_DOWN.clear()

    timer = api_server._exit_after_backend_death(0.0)
    timer.join(timeout=5)
    deadline = time.monotonic() + 5
    while not signalled and time.monotonic() < deadline:
        time.sleep(0.01)

    assert signalled == [api_server.signal.SIGTERM]
    assert armed and armed[0] == api_server.SHUTDOWN_HARD_DEADLINE_S
    assert armed[0] < 120                      # 38 minutes was the bug; keep it minutes-free


def test_force_exit_backstop_kills_the_workers_before_exiting(monkeypatch):
    procs = [_Proc("freetoken-TP0-scheduler", ignores_sigterm=True)]
    exits = []
    saved = api_server._GLOBAL_STATE
    try:
        api_server._GLOBAL_STATE = _state(backend_processes=procs)
        monkeypatch.setattr(api_server.os, "_exit", lambda code: exits.append(code))
        timer = api_server._force_exit_after(0.0)
        timer.join(timeout=5)
        deadline = time.monotonic() + 5
        while not exits and time.monotonic() < deadline:
            time.sleep(0.01)
        assert procs[0].killed and exits == [1]
    finally:
        api_server._GLOBAL_STATE = saved


@pytest.mark.parametrize("processes", [None, []])
def test_shutdown_helpers_are_noops_without_workers(processes):
    api_server._terminate_backend_workers(processes)
    api_server._reap_backend_workers(processes, timeout=0.1)
