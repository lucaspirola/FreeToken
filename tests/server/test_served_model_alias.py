"""`--served-model-alias`: one loaded model, several ids.

Drives the real parser (`parse_args`), the real `/v1/models` route and the real chat handler
against the `FakeState` pattern from test_openai_api.py. No model, no engine: the FakeState
replies are scripted `UserReply`s.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from freetoken.message import UserReply
from freetoken.server.args import parse_args
from freetoken.server.openai_api import handle_chat_completion, register_openai_routes
from freetoken.server.served_models import normalize_aliases, served_model_ids

from tests.server.test_openai_api import FakeState, chat_request

ANON_PATH = "/models/anon"


class _Config:
    def __init__(self, data: dict) -> None:
        self._data = data

    def to_dict(self) -> dict:
        return self._data


def _parse(argv: list[str]):
    config = _Config({"architectures": ["LlamaForCausalLM"], "torch_dtype": "bfloat16"})
    with patch("freetoken.utils.cached_load_hf_config", lambda _path: config):
        args, _ = parse_args(["--model", ANON_PATH, *argv])
    return args


# --- arg parsing ------------------------------------------------------------------------------


def test_alias_flag_is_repeatable_and_ordered():
    args = _parse(["--served-model-name", "primary", "--served-model-alias", "b",
                   "--served-model-alias", "a"])
    assert args.served_model_name == "primary"
    assert args.served_model_aliases == ("b", "a")
    assert served_model_ids(args) == ["primary", "b", "a"]
    assert args.strict_model_name is False


def test_no_alias_flag_means_no_aliases():
    args = _parse([])
    assert args.served_model_aliases == ()
    assert served_model_ids(args) == ["anon"]


def test_duplicate_alias_is_a_startup_error():
    with pytest.raises(SystemExit):
        _parse(["--served-model-alias", "x", "--served-model-alias", "x"])


def test_alias_equal_to_served_name_is_a_startup_error():
    with pytest.raises(SystemExit):
        _parse(["--served-model-name", "x", "--served-model-alias", "x"])


def test_empty_alias_is_a_startup_error():
    with pytest.raises(SystemExit):
        _parse(["--served-model-alias", "  "])


def test_normalize_aliases_strips_and_rejects():
    assert normalize_aliases("p", [" a ", "b"]) == ("a", "b")
    with pytest.raises(ValueError):
        normalize_aliases("p", ["a", "a"])
    with pytest.raises(ValueError):
        normalize_aliases("p", [""])


# --- /v1/models -------------------------------------------------------------------------------


def _client(state) -> TestClient:
    app = FastAPI()
    register_openai_routes(app, lambda: state, lambda: {})
    return TestClient(app)


def _aliased_state(*aliases: str, strict: bool = False) -> FakeState:
    state = FakeState([])
    state.config.served_model_aliases = aliases
    state.config.strict_model_name = strict
    state.config.max_seq_len = 4096
    return state


def test_models_lists_primary_first_then_each_alias_with_the_same_fields():
    cards = _client(_aliased_state("alias-a", "alias-b")).get("/v1/models").json()["data"]
    assert [c["id"] for c in cards] == ["unit-model", "alias-a", "alias-b"]
    for card in cards:
        assert card["root"] == "/models/unit-model"
        assert card["object"] == "model"
        assert card["max_model_len"] == 4096 and card["context_length"] == 4096


def test_models_by_id_resolves_alias_and_404s_unknown():
    client = _client(_aliased_state("alias-a"))
    assert client.get("/v1/models/alias-a").json()["id"] == "alias-a"
    assert client.get("/v1/models/unit-model").json()["id"] == "unit-model"
    missing = client.get("/v1/models/nope")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "model_not_found"


# --- chat under an alias ----------------------------------------------------------------------


def _reply() -> list[UserReply]:
    return [UserReply(uid=42, incremental_output="hi", finished=True,
                      prompt_tokens_delta=1, completion_tokens_delta=1)]


def _chat(state, model: str):
    return asyncio.run(handle_chat_completion(
        chat_request(model=model), request=None, state=state, model_sampling={},
    ))


def test_chat_under_alias_is_accepted_and_echoes_the_requested_name():
    state = _aliased_state("alias-a", strict=True)
    state.replies = _reply()
    response = _chat(state, "alias-a")
    assert state.sent is not None
    assert response["model"] == "alias-a"
    assert response["choices"][0]["message"]["content"] == "hi"


def test_chat_under_primary_is_accepted_when_strict():
    state = _aliased_state("alias-a", strict=True)
    state.replies = _reply()
    assert _chat(state, "unit-model")["model"] == "unit-model"


def test_chat_under_unknown_name_is_404_when_strict():
    state = _aliased_state("alias-a", strict=True)
    response = _chat(state, "nope")
    assert state.sent is None  # refused before anything reached the engine
    assert response.status_code == 404
    body = json.loads(response.body)
    assert body["error"]["code"] == "model_not_found"
    assert body["error"]["param"] == "model"


def test_chat_under_unknown_name_is_accepted_and_echoed_when_not_strict():
    """Today's contract, kept: Anthropic-protocol clients send `claude-*` to any proxy."""
    state = _aliased_state("alias-a")
    state.replies = _reply()
    assert _chat(state, "anything-goes")["model"] == "anything-goes"


def test_unknown_model_message_without_config_fields_is_permissive():
    """A config that predates the fields (older FakeStates) behaves as before."""
    from freetoken.server.served_models import unknown_model_message
    assert unknown_model_message(SimpleNamespace(served_model_name="m", model_path="/m"), "z") is None
