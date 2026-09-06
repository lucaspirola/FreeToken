"""``DELETE /v1/cache/pins`` and the ``--pin-prefix-*`` flags.

The endpoint is the same fire-and-reply shape as ``DELETE /v1/sessions/{id}``: a future keyed
by request id, one ``UnpinPrefixesMsg`` down the tokenizer link, and the scheduler's
``UnpinPrefixesReply`` resolving it in ``listen()``. Driven here against a state whose
``send_one`` answers inline, the way ``test_session_api.py`` does.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from freetoken.message import UnpinPrefixesMsg, UnpinPrefixesReply
from freetoken.server import api_server
from freetoken.server.args import parse_args

ANON_PATH = "/tmp/anon"


def _run_unpin(send_one):
    previous = api_server._GLOBAL_STATE
    state = SimpleNamespace(unpin_futures={})
    state.send_one = send_one
    api_server._GLOBAL_STATE = state
    try:
        return asyncio.run(api_server.unpin_prefixes()), state
    finally:
        api_server._GLOBAL_STATE = previous


def test_delete_pins_round_trips_and_reports_what_was_released():
    sent = []

    async def send_one(msg):
        sent.append(msg)
        assert isinstance(msg, UnpinPrefixesMsg)
        api_server._GLOBAL_STATE.unpin_futures[msg.request_id].set_result(
            UnpinPrefixesReply(request_id=msg.request_id, pinned_prefixes=2, pinned_tokens=4096)
        )

    result, state = _run_unpin(send_one)
    assert result == {
        "status": "ok", "released": {"pinned_prefixes": 2, "pinned_tokens": 4096},
    }
    assert len(sent) == 1 and state.unpin_futures == {}


def test_a_scheduler_that_never_answers_is_a_504_not_a_hang():
    async def send_one(msg):
        pass

    with patch.object(api_server.asyncio, "wait_for", side_effect=asyncio.TimeoutError):
        response, state = _run_unpin(send_one)
    assert response.status_code == 504 and state.unpin_futures == {}


# --------------------------------------------------------------------------- flags
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


def test_pin_flags_default_and_parse():
    args = _parse([])
    assert (args.pin_prefix_min_tokens, args.pin_prefix_max_tokens) == (1024, 65536)
    assert args.pin_prefix_max_slots == -1                    # auto: pool geometry minus 2
    args = _parse(["--pin-prefix-min-tokens", "0", "--pin-prefix-max-tokens", "0",
                   "--pin-prefix-max-slots", "3"])
    assert (args.pin_prefix_min_tokens, args.pin_prefix_max_tokens) == (0, 0)
    assert args.pin_prefix_max_slots == 3
    assert _parse(["--pin-prefix-max-slots", "0"]).pin_prefix_max_slots == 0   # no snapshot pins


@pytest.mark.parametrize("flag", ["--pin-prefix-min-tokens", "--pin-prefix-max-tokens"])
def test_negative_pin_flags_are_a_startup_error(flag):
    with pytest.raises(SystemExit):
        _parse([flag, "-1"])


def test_max_slots_below_auto_is_a_startup_error():
    with pytest.raises(SystemExit):
        _parse(["--pin-prefix-max-slots", "-2"])
