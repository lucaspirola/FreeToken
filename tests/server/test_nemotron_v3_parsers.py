"""Nemotron-3.x reasoning parser (`<think>` + a `<tool_call>` escape hatch) and the
`force_nonempty_content` answer swap -- one-shot, streaming, and end to end.

Lightning's chat template pre-opens `<think>\n` at the generation prompt when
`enable_thinking` is on (the default) and pre-closes `<think></think>` when it is
off, so the completion never carries an opening marker. Both gears are exercised
here, plus the malformed turn the plain `qwen3` parser cannot survive: no
`</think>` before `<tool_call>`, which would fold the whole tool block into
`reasoning_content` and leave the qwen3_coder detector with nothing to parse.
"""

from __future__ import annotations

import json

from freetoken.message import UserReply
from freetoken.server.openai_api import handle_chat_completion, stream_chat_completion_chunks
from freetoken.server.reasoning_parser import (
    NemotronV3ReasoningParser,
    ReasoningParser,
    ThinkReasoningParser,
)

from .test_openai_api import FakeState, chat_request, parse_sse, run

TOOL_BLOCK = (
    "<tool_call><function=get_weather><parameter=city>Paris</parameter>"
    "</function></tool_call>"
)


def nemotron_state(replies, **kwargs) -> FakeState:
    return FakeState(
        replies, tool_call_parser="qwen3_coder", reasoning_parser="nemotron_v3", **kwargs
    )


def reply(text: str, **kwargs) -> UserReply:
    return UserReply(uid=42, incremental_output=text, finished=False, **kwargs)


def done(text: str = "") -> UserReply:
    return UserReply(uid=42, incremental_output=text, finished=True, finish_reason="stop")


# ---------------------------------------------------------------------------
# Reasoning parser
# ---------------------------------------------------------------------------
def test_registered_under_its_own_name():
    assert ReasoningParser.ReasoningParserEnum["nemotron_v3"] is NemotronV3ReasoningParser
    assert issubclass(NemotronV3ReasoningParser, ThinkReasoningParser)
    assert NemotronV3ReasoningParser().tool_start_token == "<tool_call>"


def test_pre_opened_think_splits_at_the_closer():
    """thinking on: the template wrote `<think>`, the model emits only `</think>`."""
    p = NemotronV3ReasoningParser(force_reasoning=True)
    r = p.detect_and_parse("weigh the options</think>The answer is 4.")
    assert r.reasoning_text == "weigh the options"
    assert r.normal_text == "The answer is 4."


def test_thinking_off_is_pure_content():
    """thinking off: the template pre-closed `<think></think>`, so nothing the
    model writes is reasoning -- the caller builds the parser unforced."""
    p = NemotronV3ReasoningParser(force_reasoning=False)
    r = p.detect_and_parse("The answer is 4.")
    assert r.reasoning_text == ""
    assert r.normal_text == "The answer is 4."


def test_tool_call_without_a_closing_think_ends_reasoning():
    p = NemotronV3ReasoningParser(force_reasoning=True)
    r = p.detect_and_parse(f"I should look it up{TOOL_BLOCK}")
    assert r.reasoning_text == "I should look it up"
    assert r.normal_text == TOOL_BLOCK


def test_a_closing_think_still_beats_a_quoted_tool_marker():
    """A `<tool_call>` the model merely talks about inside its thought stays
    reasoning: the real closer arrives afterwards and wins."""
    p = NemotronV3ReasoningParser(force_reasoning=True)
    r = p.detect_and_parse("maybe emit <tool_call> here?</think>No, answering directly.")
    assert r.reasoning_text == "maybe emit <tool_call> here?"
    assert r.normal_text == "No, answering directly."


def test_streaming_tool_call_without_a_closer():
    p = NemotronV3ReasoningParser(force_reasoning=True)
    reasoning, content = [], []
    for chunk in ["I should look", " it up", "<tool_", "call><function=get_weather>", "..."]:
        r = p.parse_streaming_increment(chunk)
        reasoning.append(r.reasoning_text)
        content.append(r.normal_text)
    r = p.flush()
    reasoning.append(r.reasoning_text)
    content.append(r.normal_text)
    assert "".join(reasoning) == "I should look it up"
    assert "".join(content).startswith("<tool_call><function=get_weather>")


# ---------------------------------------------------------------------------
# Stray think markers outside a reasoning block are dropped
# ---------------------------------------------------------------------------
def test_a_second_closer_is_dropped_non_stream():
    """The block already closed, so a further `</think>` closes nothing: it must
    not reach `content` (it poisons the conversation the client echoes back and
    breaks cold-checkpoint prefix matching)."""
    p = NemotronV3ReasoningParser(force_reasoning=True)
    r = p.detect_and_parse("weigh it</think>The answer is 4.</think> Really.")
    assert r.reasoning_text == "weigh it"
    assert r.normal_text == "The answer is 4. Really."


def test_a_closer_with_no_block_open_is_dropped_non_stream():
    """thinking off: the template pre-closed the block, so a `</think>` the model
    emits anyway is stray from the very first token."""
    p = NemotronV3ReasoningParser(force_reasoning=False)
    r = p.detect_and_parse("The answer is 4.</think> Really.")
    assert r.reasoning_text == ""
    assert r.normal_text == "The answer is 4. Really."


def test_a_stray_opener_after_the_block_is_dropped_non_stream():
    """A `<think>` once the answer started opens nothing: dropped, not content and
    not reasoning."""
    p = NemotronV3ReasoningParser(force_reasoning=True)
    r = p.detect_and_parse("weigh it</think>The answer <think>is 4.")
    assert r.reasoning_text == "weigh it"
    assert r.normal_text == "The answer is 4."


def _stream(parser, chunks):
    reasoning, content = [], []
    for chunk in chunks:
        r = parser.parse_streaming_increment(chunk)
        reasoning.append(r.reasoning_text)
        content.append(r.normal_text)
    r = parser.flush()
    reasoning.append(r.reasoning_text)
    content.append(r.normal_text)
    return "".join(reasoning), "".join(content)


def test_a_second_closer_is_dropped_streaming():
    p = NemotronV3ReasoningParser(force_reasoning=True)
    reasoning, content = _stream(
        p, ["weigh it", "</think>", "The answer is 4.", "</think>", " Really."]
    )
    assert reasoning == "weigh it"
    assert content == "The answer is 4. Really."


def test_a_closer_split_across_chunks_is_dropped_streaming():
    """The marker's leading `<` arrives glued to the preceding token; the existing
    partial-hold reassembles it, and the whole marker is then dropped."""
    p = NemotronV3ReasoningParser(force_reasoning=True)
    reasoning, content = _stream(
        p, ["weigh it", "</think>", "The answer is 4.", "</th", "ink> Really."]
    )
    assert reasoning == "weigh it"
    assert content == "The answer is 4. Really."


def test_a_closer_with_no_block_open_is_dropped_streaming():
    p = NemotronV3ReasoningParser(force_reasoning=False)
    reasoning, content = _stream(p, ["The answer is 4.", "</think>", " Really."])
    assert reasoning == ""
    assert content == "The answer is 4. Really."


def test_a_stray_opener_after_the_block_is_dropped_streaming():
    """Dropped outright -- crucially it must NOT re-open a reasoning block and
    swallow the rest of the answer into `reasoning_content`."""
    p = NemotronV3ReasoningParser(force_reasoning=True)
    reasoning, content = _stream(
        p, ["weigh it", "</think>", "The answer ", "<think>", "is 4."]
    )
    assert reasoning == "weigh it"
    assert content == "The answer is 4."
    assert p.in_reasoning is False


def test_a_stray_opener_split_across_chunks_is_dropped_streaming():
    p = NemotronV3ReasoningParser(force_reasoning=True)
    reasoning, content = _stream(
        p, ["weigh it", "</think>", "The answer ", "<thi", "nk>is 4."]
    )
    assert reasoning == "weigh it"
    assert content == "The answer is 4."


def test_a_real_opening_think_still_opens_a_block():
    """The normal `<think>...</think>` turn is untouched: a thinking-off gear where
    the model opens a block of its own still yields reasoning, not content."""
    p = NemotronV3ReasoningParser(force_reasoning=False)
    reasoning, content = _stream(p, ["<think>", "a thought", "</think>", "an answer"])
    assert reasoning == "a thought"
    assert content == "an answer"


def test_stray_closer_never_reaches_the_client_end_to_end():
    state = nemotron_state([reply("a thought</think>an answer"), reply("</think>!"), done()])
    request = chat_request(tools=None)
    response = run(
        handle_chat_completion(request, request=None, state=state, model_sampling={})
    )
    message = response["choices"][0]["message"]
    assert message["content"] == "an answer!"
    assert message["reasoning_content"] == "a thought"


# ---------------------------------------------------------------------------
# End to end: the malformed turn must still produce a tool call
# ---------------------------------------------------------------------------
def test_missing_closer_still_yields_a_tool_call_non_stream():
    state = nemotron_state([reply("I should look it up"), reply(TOOL_BLOCK), done()])
    response = run(
        handle_chat_completion(chat_request(), request=None, state=state, model_sampling={})
    )
    choice = response["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    call = choice["message"]["tool_calls"][0]
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "Paris"}
    assert choice["message"]["reasoning_content"] == "I should look it up"


def test_missing_closer_still_yields_a_tool_call_streaming():
    state = nemotron_state([reply("I should look it up"), reply(TOOL_BLOCK), done()])
    events = parse_sse(
        run(_collect(stream_chat_completion_chunks(42, chat_request(stream=True), state)))
    )
    names = [
        tc["function"]["name"]
        for ev in events
        if isinstance(ev, dict)
        for choice in ev.get("choices", [])
        for tc in choice.get("delta", {}).get("tool_calls", [])
        if tc.get("function", {}).get("name")
    ]
    assert names == ["get_weather"]
    assert _finish_reason(events) == "tool_calls"


def test_thinking_off_turn_is_content_end_to_end():
    state = nemotron_state([reply("The answer is 4."), done()])
    request = chat_request(chat_template_kwargs={"enable_thinking": False}, tools=None)
    response = run(
        handle_chat_completion(request, request=None, state=state, model_sampling={})
    )
    message = response["choices"][0]["message"]
    assert message["content"] == "The answer is 4."
    assert "reasoning_content" not in message


# ---------------------------------------------------------------------------
# force_nonempty_content
# ---------------------------------------------------------------------------
def test_thinking_off_defaults_the_swap_on():
    """A thinking-OFF turn enables the swap without anyone asking: the template
    pre-closed the think block, so a model that opens one anyway and writes
    nothing after it would answer with an empty message."""
    # Thinking off, and the model opened a think block of its own anyway and wrote
    # nothing after it -- the empty answer this default exists to prevent.
    state = nemotron_state([reply("<think>only a thought</think>"), done()])
    request = chat_request(
        chat_template_kwargs={"enable_thinking": False},
        tools=None,
    )
    response = run(
        handle_chat_completion(request, request=None, state=state, model_sampling={})
    )
    assert state.sent.chat_template_kwargs == {"enable_thinking": False}
    assert response["choices"][0]["message"]["content"] == "only a thought"


def test_the_knob_never_reaches_the_chat_template():
    """`force_nonempty_content` is a serving knob, not a template variable: Jinja
    renders with **chat_template_kwargs, where an unknown name either raises or
    silently changes what the model sees."""
    state = nemotron_state([reply("a thought</think>an answer"), done()])
    request = chat_request(
        chat_template_kwargs={"enable_thinking": True, "force_nonempty_content": True},
        tools=None,
    )
    run(handle_chat_completion(request, request=None, state=state, model_sampling={}))
    assert state.sent.chat_template_kwargs == {"enable_thinking": True}
    # The request's own kwargs are untouched (the same dict may be reused).
    assert request.chat_template_kwargs["force_nonempty_content"] is True


def test_empty_content_swaps_in_the_reasoning_non_stream():
    state = nemotron_state([reply("the whole answer was a thought"), done()])
    request = chat_request(
        chat_template_kwargs={"force_nonempty_content": True}, tools=None
    )
    response = run(
        handle_chat_completion(request, request=None, state=state, model_sampling={})
    )
    message = response["choices"][0]["message"]
    assert message["content"] == "the whole answer was a thought"
    assert message["reasoning_content"] == "the whole answer was a thought"


def test_empty_content_swaps_in_the_reasoning_streaming():
    state = nemotron_state([reply("the whole answer was a thought"), done()])
    request = chat_request(
        stream=True, chat_template_kwargs={"force_nonempty_content": True}, tools=None
    )
    spec = _spec(request, state)
    events = parse_sse(run(_collect(stream_chat_completion_chunks(42, request, state, spec))))
    assert _joined(events, "content") == "the whole answer was a thought"
    assert _joined(events, "reasoning_content") == "the whole answer was a thought"
    # One trailing delta, and it lands before the finish chunk.
    contentful = [i for i, ev in enumerate(events) if _delta(ev).get("content")]
    assert len(contentful) == 1  # exactly one trailing delta, not one per reasoning chunk
    assert contentful[0] < _finish_index(events)


def test_the_swap_never_fires_when_there_is_real_content():
    state = nemotron_state([reply("a thought</think>a real answer"), done()])
    request = chat_request(
        chat_template_kwargs={"force_nonempty_content": True}, tools=None
    )
    response = run(
        handle_chat_completion(request, request=None, state=state, model_sampling={})
    )
    message = response["choices"][0]["message"]
    assert message["content"] == "a real answer"
    assert message["reasoning_content"] == "a thought"


def test_the_swap_never_fires_on_a_tool_call():
    """A tool-call turn has no visible content by design; swapping the thought in
    would make the assistant message look like a text answer to the client."""
    state = nemotron_state([reply("I should look it up"), reply(TOOL_BLOCK), done()])
    request = chat_request(chat_template_kwargs={"force_nonempty_content": True})
    response = run(
        handle_chat_completion(request, request=None, state=state, model_sampling={})
    )
    choice = response["choices"][0]
    assert choice["message"]["content"] == ""
    assert choice["finish_reason"] == "tool_calls"


def test_the_swap_is_off_by_default_for_a_thinking_turn():
    state = nemotron_state([reply("only a thought"), done()])
    response = run(
        handle_chat_completion(
            chat_request(tools=None), request=None, state=state, model_sampling={}
        )
    )
    assert response["choices"][0]["message"]["content"] == ""


def test_the_server_flag_turns_it_on_for_a_thinking_turn():
    state = nemotron_state([reply("only a thought"), done()])
    state.config.force_nonempty_content = True
    response = run(
        handle_chat_completion(
            chat_request(tools=None), request=None, state=state, model_sampling={}
        )
    )
    assert response["choices"][0]["message"]["content"] == "only a thought"


def test_a_request_can_override_the_server_flag_off():
    state = nemotron_state([reply("only a thought"), done()])
    state.config.force_nonempty_content = True
    request = chat_request(
        chat_template_kwargs={"force_nonempty_content": False}, tools=None
    )
    response = run(
        handle_chat_completion(request, request=None, state=state, model_sampling={})
    )
    assert response["choices"][0]["message"]["content"] == ""
    assert state.sent.chat_template_kwargs == {}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _spec(request, state):
    from freetoken.server.openai_api import chat_request_to_genspec

    return chat_request_to_genspec(
        request,
        {},
        force_nonempty_content=getattr(state.config, "force_nonempty_content", False),
    )


async def _collect(chunks):
    return [chunk async for chunk in chunks]


def _delta(event) -> dict:
    if not isinstance(event, dict):
        return {}
    choices = event.get("choices") or [{}]
    return choices[0].get("delta", {})


def _joined(events, key: str) -> str:
    return "".join(_delta(ev).get(key) or "" for ev in events)


def _finish_index(events) -> int:
    return next(
        i
        for i, ev in enumerate(events)
        if isinstance(ev, dict) and (ev.get("choices") or [{}])[0].get("finish_reason")
    )


def _finish_reason(events) -> str | None:
    return (events[_finish_index(events)]["choices"][0])["finish_reason"]
