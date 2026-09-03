"""JSON mode (`response_format`) support: prompt-side instruction, output repair,
schema validation.

FreeToken has no constrained decoding (no grammar backend in the dependency set,
and the sampler has no mask hook that survives CUDA-graph capture), so JSON mode
is *prompted*, then *enforced after the fact*:

1. the schema (or a bare "one JSON object only" rule) is appended to the system
   block and thinking is turned off by default, which is what makes a small model
   answer with the object and nothing else;
2. the completion is buffered, stripped of think residue and code fences, and the
   first balanced top-level JSON value is extracted and parsed;
3. with a ``json_schema`` format the parsed value is validated (``jsonschema``
   when it is importable, else the small built-in validator below);
4. a failure is retried once at temperature 0 with the error appended as a user
   message, and a *final* failure returns the raw text with HTTP 200 — never a
   400. Switchyard treats an unparseable judge verdict as a soft failure and
   falls through to a stronger target; a 5xx/4xx there would break the route.

Wire-agnostic on purpose: nothing here imports the OpenAI request models, so the
Anthropic/Responses adapters can reuse it.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

try:  # optional: a real implementation beats the built-in one when it is installed
    import jsonschema as _jsonschema
except Exception:  # noqa: BLE001 -- not a dependency; the built-in validator covers the rest
    _jsonschema = None


class JsonOutputError(Exception):
    """The completion was not usable JSON (unparseable, or schema-invalid). Carries
    the message shown to the model on the repair turn."""


#: Retries after the first attempt. One is the useful number: a model that fails
#: twice on the same prompt fails deterministically, and Switchyard's fallback is
#: cheaper than a third decode.
DEFAULT_JSON_RETRIES = 1


def json_retry_budget(state: Any) -> int:
    """How many repair attempts a JSON-mode request gets after the first one.
    ``FREETOKEN_JSON_RETRY`` (env) wins over ``config.json_retry``; both clamp to
    >= 0, and a non-numeric value falls back to the default rather than failing a
    request over an operator typo."""
    raw = os.environ.get("FREETOKEN_JSON_RETRY")
    if raw is None:
        raw = getattr(getattr(state, "config", None), "json_retry", None)
    if raw is None:
        return DEFAULT_JSON_RETRIES
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return DEFAULT_JSON_RETRIES


# --------------------------------------------------------------------------- #
# Prompt side
# --------------------------------------------------------------------------- #
_JSON_OBJECT_RULE = (
    "Respond with a single JSON object only. No prose, no explanation, "
    "no markdown code fences."
)


def schema_instruction(schema: dict[str, Any] | None) -> str:
    """The text appended to the system block for a JSON-mode call."""
    if not schema:
        return _JSON_OBJECT_RULE
    return (
        f"{_JSON_OBJECT_RULE} The object must validate against this JSON Schema:\n"
        f"{json.dumps(schema, ensure_ascii=False)}"
    )


def apply_json_instruction(
    messages: list[dict[str, Any]], instruction: str
) -> list[dict[str, Any]]:
    """Append ``instruction`` to the conversation's system block, creating one when
    the conversation has none. Appending (rather than prepending a second system
    turn) keeps the rendered prompt's shared prefix intact for clients that reuse
    a system block across calls."""
    out = [dict(m) for m in messages]
    for message in out:
        if message.get("role") != "system":
            continue
        content = message.get("content")
        message["content"] = f"{content}\n\n{instruction}" if content else instruction
        return out
    return [{"role": "system", "content": instruction}, *out]


def retry_user_message(error: str) -> dict[str, Any]:
    """The repair turn: what the model is told after an unusable reply."""
    return {
        "role": "user",
        "content": (
            f"Your previous reply was not accepted: {error}\n{_JSON_OBJECT_RULE}"
        ),
    }


# --------------------------------------------------------------------------- #
# Output side
# --------------------------------------------------------------------------- #
_THINK_BLOCK = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.DOTALL | re.IGNORECASE)
_THINK_CLOSE = re.compile(r"^.*?</think\s*>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```[a-zA-Z0-9_-]*\s*(.*?)\s*```", re.DOTALL)


def strip_wrappers(text: str) -> str:
    """Remove think residue and markdown code fences from a completion.

    The reasoning parser normally takes the think block out; this covers the turn
    where it is not configured, or where the model wrote a block the parser did
    not claim (an unopened ``</think>``, a second block after the answer)."""
    text = _THINK_BLOCK.sub(" ", text)
    if "</think" in text.lower():
        text = _THINK_CLOSE.sub("", text)
    text = re.sub(r"<think\b[^>]*>", " ", text, flags=re.IGNORECASE)
    fenced = _FENCE.search(text)
    if fenced is not None:
        return fenced.group(1).strip()
    return text.strip()


def extract_json_value(text: str) -> Any:
    """Parse the first complete JSON value in ``text``. Raises ``JsonOutputError``.

    Tries the whole (stripped) string first, then scans for the first ``{``/``[``
    and walks to its matching close, so trailing prose after a valid object does
    not sink the reply."""
    cleaned = strip_wrappers(text)
    if not cleaned:
        raise JsonOutputError("the reply was empty; expected a JSON object")
    try:
        return json.loads(cleaned)
    except ValueError:
        pass
    span = _first_balanced_span(cleaned)
    if span is None:
        raise JsonOutputError("the reply did not contain a JSON object")
    try:
        return json.loads(cleaned[span[0]:span[1]])
    except ValueError as exc:
        raise JsonOutputError(f"the reply was not valid JSON ({exc})") from exc


def _first_balanced_span(text: str) -> tuple[int, int] | None:
    """(start, end) of the first balanced ``{...}``/``[...]``, string-aware."""
    openers = {"{": "}", "[": "]"}
    start = next((i for i, ch in enumerate(text) if ch in openers), None)
    if start is None:
        return None
    stack: list[str] = []
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in openers:
            stack.append(openers[ch])
        elif stack and ch == stack[-1]:
            stack.pop()
            if not stack:
                return start, i + 1
        elif ch in ("}", "]"):
            return None  # unbalanced close: the text is not JSON at all
    return None


def canonical_json(value: Any) -> str:
    """The single spelling of a validated value that goes on the wire."""
    return json.dumps(value, ensure_ascii=False)


def coerce_json_content(text: str, schema: dict[str, Any] | None) -> tuple[str, str | None]:
    """``(content, error)``: the canonical JSON for ``text``, or the raw text plus
    the reason it could not be used. Never raises."""
    try:
        value = extract_json_value(text)
    except JsonOutputError as exc:
        return text, str(exc)
    if schema:
        error = schema_error(value, schema)
        if error is not None:
            return text, error
    return canonical_json(value), None


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def schema_error(value: Any, schema: dict[str, Any]) -> str | None:
    """None when ``value`` satisfies ``schema``, else a one-line reason."""
    if _jsonschema is not None:
        try:
            _jsonschema.validate(value, schema)
        except _jsonschema.ValidationError as exc:  # type: ignore[attr-defined]
            path = "/".join(str(p) for p in exc.absolute_path)
            return f"schema validation failed at {path or '<root>'}: {exc.message}"
        except Exception:  # noqa: BLE001 -- an invalid schema is not this reply's fault
            return None
        return None
    errors = _validate(value, schema, "<root>")
    return f"schema validation failed at {errors[0][0]}: {errors[0][1]}" if errors else None


_TYPES: dict[str, Any] = {
    "string": str,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _type_ok(value: Any, name: str) -> bool:
    if name == "null":
        return value is None
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if name == "boolean":
        return isinstance(value, bool)
    expected = _TYPES.get(name)
    if expected is None:
        return True  # unknown keyword: do not invent a failure
    return isinstance(value, expected) and not isinstance(value, bool)


def _validate(value: Any, schema: Any, path: str) -> list[tuple[str, str]]:
    """The built-in validator: the subset OpenAI structured outputs (and every
    Switchyard verdict schema) actually uses. Returns [(path, reason), ...]."""
    if not isinstance(schema, dict) or schema.get("$ref"):
        return []  # a $ref-bearing schema is beyond this validator; accept it
    for key in ("anyOf", "oneOf"):
        options = schema.get(key)
        if isinstance(options, list) and options:
            if any(not _validate(value, option, path) for option in options):
                continue
            return [(path, f"value does not match any {key} branch")]
    types = schema.get("type")
    if types is not None:
        names = types if isinstance(types, list) else [types]
        if not any(_type_ok(value, str(n)) for n in names):
            return [(path, f"expected type {'/'.join(map(str, names))}")]
    if "enum" in schema and value not in schema["enum"]:
        return [(path, f"value {value!r} is not one of {schema['enum']}")]
    if "const" in schema and value != schema["const"]:
        return [(path, f"value {value!r} must be {schema['const']!r}")]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            return [(path, f"{value} is below the minimum {schema['minimum']}")]
        if "maximum" in schema and value > schema["maximum"]:
            return [(path, f"{value} is above the maximum {schema['maximum']}")]
    # Length keywords: Switchyard's capability-classifier schema uses `minLength: 1`
    # to reject an empty `crux`, so the built-in path honors them too.
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            return [(path, f"string is shorter than minLength {schema['minLength']}")]
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            return [(path, f"string is longer than maxLength {schema['maxLength']}")]
    if isinstance(value, dict):
        properties = schema.get("properties") or {}
        for name in schema.get("required") or []:
            if name not in value:
                return [(path, f"required property {name!r} is missing")]
        if schema.get("additionalProperties") is False:
            extra = [k for k in value if k not in properties]
            if extra:
                return [(path, f"unexpected propert{'y' if len(extra) == 1 else 'ies'} {extra}")]
        for name, subschema in properties.items():
            if name in value:
                errors = _validate(value[name], subschema, f"{path}/{name}")
                if errors:
                    return errors
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            return [(path, f"array is shorter than minItems {schema['minItems']}")]
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            return [(path, f"array is longer than maxItems {schema['maxItems']}")]
        items = schema.get("items")
        if isinstance(items, dict):
            for index, item in enumerate(value):
                errors = _validate(item, items, f"{path}/{index}")
                if errors:
                    return errors
    return []
