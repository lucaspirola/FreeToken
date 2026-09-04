from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MessageContent(BaseModel):
    type: str
    text: str | None = None
    image_url: Any | None = None
    audio_url: Any | None = None


class Function(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None
    # OpenAI's structured-outputs flag. Typed so it round-trips instead of landing in
    # `extra`, but stripped before the chat template sees the tool (no template reads
    # it, and a stray key changes the rendered catalog byte-for-byte, which would
    # break prefix-cache reuse for clients that send it inconsistently).
    strict: bool | None = None


class Tool(BaseModel):
    type: Literal["function"] = "function"
    function: Function


class JsonSchemaFormat(BaseModel):
    """The `json_schema` payload of `response_format` (OpenAI structured outputs).

    Typed so the schema reaches JSON mode as a real dict instead of `extra`.
    `strict` is accepted and recorded but is advisory here: FreeToken has no
    constrained decoding, so a schema is enforced by validating the completion and
    retrying, not by masking the sampler (see `server/json_output.py`)."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    name: str | None = None
    description: str | None = None
    strict: bool | None = None
    # `schema` shadows BaseModel.schema(); the wire key stays "schema" via the alias.
    json_schema: dict[str, Any] | None = Field(default=None, alias="schema")


class ResponseFormat(BaseModel):
    """OpenAI `response_format`: text (the default), a bare JSON object, or a JSON
    object constrained to a schema. Switchyard's judge/classifier targets send
    `json_schema` (strict) or `json_object` verbatim."""

    model_config = ConfigDict(extra="allow")

    type: Literal["text", "json_object", "json_schema"] = "text"
    json_schema: JsonSchemaFormat | None = None

    @property
    def json_mode(self) -> bool:
        return self.type in ("json_object", "json_schema")

    @property
    def schema_dict(self) -> dict[str, Any] | None:
        """The JSON Schema to validate against, or None (json_object, or a
        json_schema wrapper with no schema body)."""
        if self.type != "json_schema" or self.json_schema is None:
            return None
        return self.json_schema.json_schema


class ToolChoiceFunction(BaseModel):
    name: str


class ToolChoiceObject(BaseModel):
    type: Literal["function"] = "function"
    function: ToolChoiceFunction


class StreamOptions(BaseModel):
    include_usage: bool = False


class FunctionCall(BaseModel):
    name: str | None = None
    arguments: str | dict[str, Any] | None = None


class ToolCall(BaseModel):
    id: str | None = None
    index: int | None = None
    type: Literal["function"] = "function"
    function: FunctionCall


class Message(BaseModel):
    role: str
    content: str | list[MessageContent] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    reasoning: str | None = None
    reasoning_content: str | None = None
    thinking: str | None = None
    tool_calls: list[ToolCall] | None = None


class KvTransferParams(BaseModel):
    """vLLM's hidden-state connector wire, as Switchyard's prefill router sends it.

    The router posts ``max_tokens: 1`` with top-level ``kv_transfer_params`` and reads
    ``kv_transfer_params.hidden_states_path`` off the response. ``hidden_states_path``
    is a *directory* on the way in and the written ``.safetensors`` file on the way out
    -- vLLM chooses the file name, and the reader is documented never to guess it.

    FreeToken adds ``layer_ids`` (vLLM spells this
    ``eagle_aux_hidden_state_layer_ids`` in the speculative-config at launch, which is
    not a per-request knob there); it defaults to every block of the model.
    ``include_output_tokens`` is accepted for wire compatibility and ignored: FreeToken
    only ever exports prompt positions, which is all the router mean-pools.

    This model is the only place ``kv_transfer_params`` is typed. /v1/completions,
    /v1/messages and /v1/responses do not declare it, so it lands in their ``extra`` and
    is ignored there -- the probe is a chat-completions feature.
    """

    model_config = ConfigDict(extra="forbid")

    hidden_states_path: str | None = None
    layer_ids: list[int] | None = None
    include_output_tokens: bool = False


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[Message]
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    n: int = 1
    stream: bool = False
    stream_options: StreamOptions | None = None
    stop: str | list[str] | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)
    reasoning_effort: str | None = None
    # DeepSeek-wire thinking toggle ({"type": "enabled"|"disabled"}). Any so a
    # foreign shape stays ignored (extra="allow" swallowed it before this field
    # existed) instead of becoming a bare 422 at the route boundary; the handler
    # reads the dict form and 400s only on an unknown "type" value.
    thinking: Any | None = None
    ignore_eos: bool = False
    tools: list[Tool] | None = None
    tool_choice: Literal["none", "auto", "required"] | ToolChoiceObject | None = None
    parallel_tool_calls: bool | None = None
    function_call: Any | None = None
    logit_bias: dict[str, float] | None = None
    # JSON mode. Typed (not a bare dict) so the handler reads the schema without
    # re-validating the wire shape on every request; /v1/completions keeps the
    # untyped field because it still rejects the feature.
    response_format: ResponseFormat | None = None
    # Accepted-and-ignored OpenAI fields Switchyard sends. Typed (not swallowed by
    # extra="allow") so they are visible to the handler: `prompt_cache_key` is a
    # session-affinity hint (bound to a KV session by the sessions layer),
    # `top_logprobs` is rejected only when > 0 since logprobs are unsupported.
    prompt_cache_key: str | None = None
    user: str | None = None
    top_logprobs: int | None = None
    # FreeToken extension: protect this conversation's completed KV until the next turn,
    # explicit close, disconnect/abort, or idle expiry.
    session_id: str | None = None
    session_ttl_seconds: float | None = None
    # Opt-in hidden-state export for Switchyard's prefill probe. Typed (not swallowed by
    # extra="allow") because it changes how the request is served: it bypasses prefix
    # reuse and binds no session. Requires --hidden-states-dir; see docs/switchyard.md.
    kv_transfer_params: KvTransferParams | None = None

    @model_validator(mode="after")
    def _sync_max_completion_tokens(self) -> "ChatCompletionRequest":
        # `max_completion_tokens` wins when a client sends both: it is the current
        # spelling and `max_tokens` is the deprecated alias (Switchyard only ever
        # sends the former).
        if self.max_completion_tokens is not None:
            self.max_tokens = self.max_completion_tokens
        return self


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    prompt: str | list[str] | list[int] | list[list[int]]
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    n: int = 1
    stream: bool = False
    stream_options: StreamOptions | None = None
    stop: str | list[str] | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    ignore_eos: bool = False
    logprobs: int | None = None
    echo: bool = False
    suffix: str | None = None
    logit_bias: dict[str, float] | None = None
    response_format: dict[str, Any] | None = None
    prompt_cache_key: str | None = None
    user: str | None = None
    session_id: str | None = None
    session_ttl_seconds: float | None = None

    @model_validator(mode="after")
    def _sync_max_completion_tokens(self) -> "CompletionRequest":
        # See ChatCompletionRequest._sync_max_completion_tokens.
        if self.max_completion_tokens is not None:
            self.max_tokens = self.max_completion_tokens
        return self


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "FreeToken"
    root: str
    # The model's own limit, not the KV budget in force. Two spellings of the same number:
    # `max_model_len` is vLLM/SGLang's, `context_length` what most other clients look for.
    max_model_len: int | None = None
    context_length: int | None = None
    # The checkpoint's probed effort vocabulary (freetoken.tokenizer.effort); None
    # (not []) when the model has no effort knob or the probe could not run.
    supported_reasoning_efforts: list[str] | None = None
    default_reasoning_effort: str | None = None


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard] = Field(default_factory=list)
