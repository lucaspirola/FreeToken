from __future__ import annotations

import asyncio
from types import SimpleNamespace

from freetoken.message import SessionClosedReply
from freetoken.server import api_server


def test_close_session_waits_for_scheduler_barrier():
    previous = api_server._GLOBAL_STATE
    state = SimpleNamespace(session_close_futures={})

    async def send_one(msg):
        state.session_close_futures[msg.request_id].set_result(
            SessionClosedReply(
                session_id=msg.session_id,
                request_id=msg.request_id,
                status="closed",
            )
        )

    state.send_one = send_one
    api_server._GLOBAL_STATE = state
    try:
        result = asyncio.run(api_server.close_session("helper-1"))
    finally:
        api_server._GLOBAL_STATE = previous

    assert result == {"id": "helper-1", "status": "closed"}
    assert state.session_close_futures == {}


def test_close_client_launch_closes_every_tracked_agent_session():
    previous = api_server._GLOBAL_STATE
    state = SimpleNamespace(
        session_close_futures={},
        client_launch_sessions={"launch-1": {"main", "child"}},
    )

    async def send_one(msg):
        state.session_close_futures[msg.request_id].set_result(
            SessionClosedReply(
                session_id=msg.session_id,
                request_id=msg.request_id,
                status="closed",
            )
        )

    state.send_one = send_one
    api_server._GLOBAL_STATE = state
    try:
        response = asyncio.run(api_server.close_client_launch_sessions("launch-1"))
    finally:
        api_server._GLOBAL_STATE = previous

    assert response.status_code == 200
    assert b'"id":"main"' in response.body
    assert b'"id":"child"' in response.body
    assert state.client_launch_sessions == {}
