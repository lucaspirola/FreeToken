"""`chat_session_id`: which conversation identifier a /v1/chat/completions turn
binds its KV session lease to.

Switchyard forwards the harness's own correlation headers verbatim and adds its
own overrides (`crates/protocol/src/metadata.rs`), so the same conversation can
arrive carrying two or three ids at once. Precedence has to be fixed and tested:
picking a different field on a later turn silently starts a second lease and
throws away the prefix the first one was holding.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from freetoken.server.client_sessions import chat_session_id


class FakeRequest:
    def __init__(self, **headers: str) -> None:
        # Starlette lowercases header lookups; the tests spell them as the wire does.
        self.headers = {k.replace("_", "-").lower(): v for k, v in headers.items()}


def req(session_id: str | None = None, prompt_cache_key: str | None = None):
    return SimpleNamespace(session_id=session_id, prompt_cache_key=prompt_cache_key)


def test_no_identity_at_all_binds_nothing():
    assert chat_session_id(req(), FakeRequest()) is None
    assert chat_session_id(req(), None) is None


def test_explicit_session_id_wins_over_every_header():
    request = FakeRequest(
        x_switchyard_session_id="sy",
        x_claude_code_session_id="cc",
        x_codex_session_id="cx",
    )
    assert chat_session_id(req(session_id="mine"), request) == "mine"


def test_precedence_order():
    """switchyard > claude-code > codex > prompt_cache_key > generic."""
    all_headers = dict(
        x_switchyard_session_id="sy",
        x_claude_code_session_id="cc",
        x_codex_session_id="cx",
        session_id="generic",
        x_session_id="opencode",
    )
    ordered = [
        "x_switchyard_session_id",
        "x_claude_code_session_id",
        "x_codex_session_id",
    ]
    seen = []
    headers = dict(all_headers)
    for name in ordered:
        bound = chat_session_id(req(prompt_cache_key="pck"), FakeRequest(**headers))
        assert bound is not None and bound.startswith("auto:")
        seen.append(bound)
        headers.pop(name)  # drop the winner; the next one down must take over

    # prompt_cache_key outranks the generic correlation headers ...
    with_key = chat_session_id(req(prompt_cache_key="pck"), FakeRequest(**headers))
    # ... which are themselves used when nothing above them is present.
    without_key = chat_session_id(req(), FakeRequest(**headers))
    seen.extend([with_key, without_key])

    assert len(set(seen)) == len(seen), seen  # every rung is a distinct lease


def test_generic_headers_rank_session_id_first():
    both = chat_session_id(req(), FakeRequest(session_id="a", x_session_id="b"))
    assert both == chat_session_id(req(), FakeRequest(session_id="a"))
    assert both != chat_session_id(req(), FakeRequest(x_session_id="b"))


def test_prompt_cache_key_binds_a_lease():
    """Switchyard's `prompt_cache_key` is the only affinity signal a bare OpenAI
    client sends; the same key must map to the same lease on every turn."""
    first = chat_session_id(req(prompt_cache_key="conv-7"), FakeRequest())
    second = chat_session_id(req(prompt_cache_key="conv-7"), FakeRequest())
    assert first is not None and first == second
    assert first != chat_session_id(req(prompt_cache_key="conv-8"), FakeRequest())


@pytest.mark.parametrize(
    "session_header, agent_header",
    [
        ("x_switchyard_session_id", "x_switchyard_agent_id"),
        ("x_claude_code_session_id", "x_claude_code_agent_id"),
    ],
)
def test_sub_agent_gets_its_own_lease(session_header, agent_header):
    """A sub-agent shares its parent's session id but not its conversation: one
    lease for both would serialize the two turns and evict each other's prefix."""
    parent = chat_session_id(req(), FakeRequest(**{session_header: "s1"}))
    child = chat_session_id(
        req(), FakeRequest(**{session_header: "s1", agent_header: "child"})
    )
    other_child = chat_session_id(
        req(), FakeRequest(**{session_header: "s1", agent_header: "other"})
    )
    assert len({parent, child, other_child}) == 3
    # ... and each child id is stable across its own turns.
    assert child == chat_session_id(
        req(), FakeRequest(**{session_header: "s1", agent_header: "child"})
    )


def test_agent_id_alone_binds_nothing():
    """An agent id without a conversation id is not an identity: binding one would
    make every unrelated turn of that agent share a lease."""
    assert chat_session_id(req(), FakeRequest(x_switchyard_agent_id="a")) is None


def test_blank_headers_and_keys_are_ignored():
    request = FakeRequest(x_switchyard_session_id="   ", x_claude_code_session_id="cc")
    assert chat_session_id(req(), request) == chat_session_id(
        req(), FakeRequest(x_claude_code_session_id="cc")
    )
    assert chat_session_id(req(prompt_cache_key="  "), FakeRequest()) is None


def test_derived_ids_are_marked_auto():
    """The `auto:` prefix is what the scheduler-side reclaim policy keys on, and
    what separates an id FreeToken inferred from one the client owns."""
    bound = chat_session_id(req(), FakeRequest(x_codex_session_id="cx"))
    assert bound is not None and bound.startswith("auto:codex:")
    assert chat_session_id(req(session_id="client-owned"), FakeRequest()) == "client-owned"
