from __future__ import annotations

import importlib.util
import json
import os
import threading
from types import ModuleType
from typing import TYPE_CHECKING, Any, List

import torch
from freetoken.message import TokenizeMsg, UserMsg
from freetoken.message.tokenizer import IN_PLACE_SYSTEM_KEY
from freetoken.utils import init_logger
from transformers import PreTrainedTokenizerBase

if TYPE_CHECKING:
    from freetoken.mm.processor import MMProcessor

from .effort import (
    EffortProfile,
    ThinkingProfile,
    probe_effort_profile,
    probe_thinking_profile,
    quantize_effort,
)

logger = init_logger(__name__)


def resolve_thinking_mode(chat_template_kwargs: dict[str, Any] | None, tools: Any | None) -> str:
    """Resolve the thinking mode (``"thinking"`` or ``"chat"``) for a chat request.

    The single source of truth for this decision: the encode side
    (``_apply_dsv4_chat_encoder`` below) uses it to pick the prompt the model
    sees, and the frontend parse side (``server/openai_api.py``) imports it to
    decide whether the model's output begins inside a reasoning block. Keeping
    one implementation prevents the two sides from disagreeing. Thinking is on
    when tools are offered (dsv4 only emits well-formed tool calls in thinking
    mode) or when the caller requests it via ``chat_template_kwargs``.
    """
    ctk = chat_template_kwargs or {}
    mode = str(ctk.get("thinking_mode") or "chat")
    if tools or ctk.get("enable_thinking") or ctk.get("thinking"):
        mode = "thinking"
    if mode not in ("chat", "thinking"):
        mode = "chat"
    return mode


_EFFORT_PROBE_MESSAGES = [{"role": "user", "content": "ping"}]

_MID_SYSTEM_PROBE_USER = "FTPROBEUSERTURNONE"
_MID_SYSTEM_PROBE_SYSTEM = "FTPROBEMIDSYSTEM"
_MID_SYSTEM_PROBE_LAST = "FTPROBEUSERTURNTWO"


class TokenizeManager:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, mm_processor: MMProcessor | None = None) -> None:
        self.tokenizer = tokenizer
        self.mm_processor = mm_processor  # None: the model takes no images
        self._dsv4_encoder = _load_dsv4_encoder_if_needed(tokenizer)
        self._effort_profile: EffortProfile | None = None
        self._thinking_profile: ThinkingProfile | None = None
        self._effort_lock = threading.Lock()
        self._logged_effort_maps: set[tuple[Any, str | None]] = set()
        self._mid_system: bool | None = None
        # head render (system message + tools) -> its token ids; see segment_boundary.
        self._segment_ids: dict[int, torch.Tensor | None] = {}

    def tokenize(self, msgs: List[TokenizeMsg]) -> List[UserMsg]:
        results: List[UserMsg] = []
        # TODO: batch tokenization
        for msg in msgs:
            prompt = self.render_prompt(msg)
            # A jinja chat template owns every special token (HF's apply_chat_template
            # tokenizes with add_special_tokens=False for the same reason): tokenizers
            # that auto-add bos (muse-glimmer's, llama's) would otherwise double it --
            # the template already rendered one. Raw-string prompts and the dsv4
            # encoder path keep the default.
            templated = isinstance(msg.text, list) and self._dsv4_encoder is None
            input_ids: torch.Tensor = (  # type: ignore
                self.tokenizer.encode(
                    prompt, return_tensors="pt", add_special_tokens=not templated
                )
            )
            input_ids = input_ids.view(-1).to(torch.int32)
            boundary = (
                self.segment_boundary(msg, input_ids) if templated and not msg.images else None
            )
            if msg.images:
                if self.mm_processor is None:
                    raise ValueError("image input is not supported for this model")
                mm = self.mm_processor.apply(input_ids, msg.images)
                results.append(
                    UserMsg(
                        uid=msg.uid,
                        input_ids=mm.input_ids,
                        sampling_params=msg.sampling_params,
                        mm_items=mm.mm_items,
                        mrope_positions=mm.mrope_positions,
                        mrope_delta=mm.mrope_delta,
                        session_id=msg.session_id,
                        session_ttl_seconds=msg.session_ttl_seconds,
                        session_reclaimable=msg.session_reclaimable,
                        hidden_states=msg.hidden_states,
                        no_prefix_cache=msg.no_prefix_cache,
                        pin_key=msg.pin_key,
                    )
                )
            else:
                results.append(
                    UserMsg(
                        uid=msg.uid,
                        input_ids=input_ids,
                        sampling_params=msg.sampling_params,
                        session_id=msg.session_id,
                        session_ttl_seconds=msg.session_ttl_seconds,
                        session_reclaimable=msg.session_reclaimable,
                        hidden_states=msg.hidden_states,
                        no_prefix_cache=msg.no_prefix_cache,
                        pin_key=msg.pin_key,
                        prefix_boundary=boundary,
                    )
                )
        return results

    def render_prompt(self, msg: TokenizeMsg) -> str:
        """The template/encoder half of ``tokenize``, exposed so the frontend can
        validate a request before committing an SSE stream. Sanitizes
        ``reasoning_effort`` first: every render path (worker, frontend
        validation, count_tokens) must quantize identically."""
        if not isinstance(msg.text, list):
            return msg.text
        return self._render(
            self.place_system_messages(msg.text),
            msg.tools,
            self._sanitize_effort(msg.chat_template_kwargs or {}),
        )

    def mid_system_supported(self) -> bool:
        """Does this checkpoint's chat template render a system turn mid-conversation, in
        place? Probed once (a four-turn system/user/system/user render) and cached. A template
        that raises (Ornith 1.5, Qwen3.5: "System message must be at the beginning"), drops the
        turn or moves it elsewhere counts as no; so does the dsv4 encoder."""
        with self._effort_lock:
            if self._mid_system is None:
                self._mid_system = self._probe_mid_system()
                logger.info(
                    "chat template mid-conversation system turn: %s",
                    "kept in place" if self._mid_system else "folded into the adjacent user turn",
                )
            return self._mid_system

    def _probe_mid_system(self) -> bool:
        if self._dsv4_encoder is not None:
            return False
        probe = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": _MID_SYSTEM_PROBE_USER},
            {"role": "system", "content": _MID_SYSTEM_PROBE_SYSTEM},
            {"role": "user", "content": _MID_SYSTEM_PROBE_LAST},
        ]
        try:
            text = self._render(probe, None, {})
        except Exception:
            return False
        u = text.find(_MID_SYSTEM_PROBE_USER)
        m = text.find(_MID_SYSTEM_PROBE_SYSTEM)
        v = text.find(_MID_SYSTEM_PROBE_LAST)
        return 0 <= u < m < v

    def place_system_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Resolve the in-place system turns an adapter marked (:data:`IN_PLACE_SYSTEM_KEY`):
        kept as system turns when the template accepts them mid-conversation, else appended
        to the preceding user turn inside ``<system-reminder>`` tags (a standalone user turn
        when the preceding turn is not a user's). Either way the rendered prompt of one turn
        stays a prefix of the next turn's. Messages without the key pass through untouched."""
        if not any(isinstance(m, dict) and m.get(IN_PLACE_SYSTEM_KEY) for m in messages):
            return messages
        keep = self.mid_system_supported()
        out: list[dict[str, Any]] = []
        for m in messages:
            if not (isinstance(m, dict) and m.get(IN_PLACE_SYSTEM_KEY)):
                out.append(m)
                continue
            m = {k: v for k, v in m.items() if k != IN_PLACE_SYSTEM_KEY}
            if keep:
                out.append(m)
                continue
            text = f"<system-reminder>\n{m.get('content') or ''}\n</system-reminder>"
            prev = out[-1] if out else None
            if prev is not None and prev.get("role") == "user":
                content = prev.get("content")
                if isinstance(content, list):
                    merged = [*content, {"type": "text", "text": "\n\n" + text}]
                else:
                    merged = f"{content}\n\n{text}" if content else text
                out[-1] = {**prev, "content": merged}
            else:
                out.append({"role": "user", "content": text})
        return out

    def segment_boundary(self, msg: TokenizeMsg, input_ids: torch.Tensor) -> int | None:
        """Token length of the prompt's leading segment -- its first (system) message plus the
        tool list, as the template renders them with nothing after -- when those tokens are a
        strict prefix of ``input_ids``; else None. The segment is what separate conversations
        of one client share (system prompt + tools), so the scheduler ends a prefill chunk
        there and snapshots it. The head's token ids are cached per head render."""
        if not isinstance(msg.text, list) or self._dsv4_encoder is not None:
            return None
        messages = msg.text
        if len(messages) < 2 or not isinstance(messages[0], dict):
            return None
        if messages[0].get("role") != "system" or messages[0].get(IN_PLACE_SYSTEM_KEY):
            return None
        kwargs = self._sanitize_effort(msg.chat_template_kwargs or {})
        try:
            head = self._render([messages[0]], msg.tools, kwargs, add_generation_prompt=False)
        except Exception:
            return None
        key = hash(head)
        if key not in self._segment_ids:
            if len(self._segment_ids) > 256:
                self._segment_ids.clear()
            ids = self.tokenizer.encode(head, return_tensors="pt", add_special_tokens=False)
            self._segment_ids[key] = ids.view(-1).to(torch.int32) if ids.numel() else None
        ids = self._segment_ids[key]
        if ids is None:
            return None
        b = int(ids.numel())
        if b >= int(input_ids.numel()) or not torch.equal(input_ids[:b], ids):
            return None
        return b

    def _render(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        chat_template_kwargs: dict[str, Any],
        add_generation_prompt: bool = True,
    ) -> str:
        """Raw render, no effort sanitation — the probe needs unsupported values
        to actually reach the template so rejection is observable."""
        if self._dsv4_encoder is not None:
            return _apply_dsv4_chat_encoder(
                self._dsv4_encoder, messages, tools, chat_template_kwargs
            )
        # Broadcast the effort in every spelling the ecosystem's templates read
        # (muse-glimmer grades ``reasoning_strength``; Jinja ignores undeclared
        # variables) -- the same rule the thinking toggles use. An explicit
        # caller-provided spelling wins over the broadcast.
        if "reasoning_effort" in chat_template_kwargs:
            chat_template_kwargs = dict(chat_template_kwargs)
            chat_template_kwargs.setdefault(
                "reasoning_strength", chat_template_kwargs["reasoning_effort"]
            )
        if tools is not None:
            chat_template_kwargs = {**chat_template_kwargs, "tools": tools}
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **chat_template_kwargs,
        )
        assert isinstance(prompt, str)
        return prompt

    def effort_profile(self) -> EffortProfile:
        """The checkpoint's effort vocabulary, probed on first use and cached
        for the process lifetime."""
        with self._effort_lock:
            if self._effort_profile is None:
                self._effort_profile = probe_effort_profile(self._probe_render)
                logger.info(
                    "reasoning-effort profile: supported=%s default=%s",
                    sorted(self._effort_profile.supported) or "(none)",
                    self._effort_profile.default,
                )
            return self._effort_profile

    def thinking_profile(self) -> ThinkingProfile:
        """The checkpoint's thinking controls (toggle behavior + effort
        vocabulary), probed on first use and cached for the process lifetime.
        Feeds the /v1/cache/status gear derivation."""
        efforts = self.effort_profile()
        with self._effort_lock:
            if self._thinking_profile is None:
                self._thinking_profile = probe_thinking_profile(self._probe_render, efforts)
            return self._thinking_profile

    def _probe_render(
        self, kwargs: dict[str, Any], tools: list[dict[str, Any]] | None
    ) -> str:
        return self._render(_EFFORT_PROBE_MESSAGES, tools, kwargs)

    def _sanitize_effort(self, chat_template_kwargs: dict[str, Any]) -> dict[str, Any]:
        if "reasoning_effort" not in chat_template_kwargs:
            return chat_template_kwargs
        raw = chat_template_kwargs.get("reasoning_effort")
        mapped = quantize_effort(raw, self.effort_profile())
        if mapped == raw:
            return chat_template_kwargs
        # raw is client-controlled and may be unhashable (a JSON list/dict).
        key = (raw if isinstance(raw, str) else repr(raw), mapped)
        if key not in self._logged_effort_maps:
            self._logged_effort_maps.add(key)
            logger.info(
                "reasoning_effort %r is not supported by this checkpoint; using %s",
                raw,
                mapped if mapped is not None else "the template default",
            )
        sanitized = dict(chat_template_kwargs)
        if mapped is None:
            del sanitized["reasoning_effort"]
        else:
            sanitized["reasoning_effort"] = mapped
        return sanitized


def _load_dsv4_encoder_if_needed(tokenizer: PreTrainedTokenizerBase) -> ModuleType | None:
    if getattr(tokenizer, "chat_template", None):
        return None
    model_path = getattr(tokenizer, "name_or_path", None) or getattr(tokenizer, "_name_or_path", "")
    if not model_path:
        return None
    encoder_path = os.path.join(str(model_path), "encoding", "encoding_dsv4.py")
    if not os.path.isfile(encoder_path):
        return None
    spec = importlib.util.spec_from_file_location("encoding_dsv4", encoder_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "encode_messages"):
        return None
    return module


def _apply_dsv4_chat_encoder(
    encoder: ModuleType,
    messages: list[dict],
    tools: list[dict] | None,
    chat_template_kwargs: dict,
) -> str:
    rendered_messages = [dict(message) for message in messages]
    for message in rendered_messages:
        if message.get("tool_calls"):
            message["tool_calls"] = _dsv4_tool_calls(message["tool_calls"])
    if tools:
        _attach_tools_to_dsv4_messages(rendered_messages, tools)

    # No effort filtering here: the caller sanitized already, and the probe
    # needs raw values to reach the encoder's own validation.
    return encoder.encode_messages(
        rendered_messages,
        thinking_mode=resolve_thinking_mode(chat_template_kwargs, tools),
        reasoning_effort=chat_template_kwargs.get("reasoning_effort"),
    )


def _dsv4_tool_calls(tool_calls: list[dict]) -> list[dict]:
    """The dsv4 encoder's contract is ``function.arguments`` = JSON-object STRING
    (it json.loads then iterates .items()); a dict (what ``render_messages``
    produces for Jinja templates) trips its bare-except fallback, which wraps the
    whole payload in a bogus parameter literally named ``arguments``. Re-serialize
    here. Copies each tool-call dict: the outer message copy is shallow, so these
    are shared with the caller."""
    rendered = []
    for tc in tool_calls:
        tc = dict(tc)
        fn = dict(tc.get("function") or {})
        fn["arguments"] = _dsv4_arguments_str(fn.get("arguments"))
        tc["function"] = fn
        rendered.append(tc)
    return rendered


def _dsv4_arguments_str(arguments: Any) -> str:
    """Missing/empty means no arguments (vLLM parity); anything else that is not
    a JSON object is rejected -- ValueError becomes a per-request "could not
    encode request" error, never a worker crash -- matching sglang's
    validate-then-400. A non-object would otherwise raise uncaught in the
    encoder's .items() or be wrapped as garbage."""
    if arguments is None or (isinstance(arguments, str) and not arguments.strip()):
        return "{}"
    if isinstance(arguments, dict):
        return json.dumps(arguments, ensure_ascii=False)
    shown = f"{arguments!r:.200}"
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError as err:
            raise ValueError(
                f"tool call function.arguments must be valid JSON, got {shown}"
            ) from err
        if isinstance(parsed, dict):
            return arguments
    raise ValueError(f"tool call function.arguments must be a JSON object, got {shown}")


def _attach_tools_to_dsv4_messages(messages: list[dict], tools: list[dict]) -> None:
    for message in messages:
        if message.get("role") == "system":
            message["tools"] = tools
            return
    messages.insert(0, {"role": "system", "content": "", "tools": tools})
