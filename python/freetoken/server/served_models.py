"""The names the loaded model answers to.

One checkpoint, several ids: ``--served-model-name`` is the primary id and each
``--served-model-alias`` adds another. ``GET /v1/models`` lists them all, primary first, and
a request's ``model`` may be any of them. The response echoes the name the client sent, not
the primary, so a router (Switchyard) that keys on the id it dispatched with sees it back.

Whether an id *outside* that set is refused is ``--strict-model-name``: off by default, because
Anthropic-protocol clients send their own model names (``claude-*``) to whatever proxy they
are pointed at, and refusing them would break every such client on the day the flag lands.
"""

from __future__ import annotations

from typing import Any, Iterable


def normalize_aliases(primary: str, aliases: Iterable[str]) -> tuple[str, ...]:
    """Validate the alias list against the primary name.

    Raises ``ValueError`` on an empty/blank alias or a duplicate (between aliases or against
    the primary). Order is preserved: it is the order ``/v1/models`` lists them in.
    """
    seen = {primary}
    out: list[str] = []
    for raw in aliases:
        name = raw.strip() if isinstance(raw, str) else raw
        if not name:
            raise ValueError("--served-model-alias: empty name")
        if name in seen:
            raise ValueError(
                f"--served-model-alias {name!r}: duplicates the served model name or another alias"
            )
        seen.add(name)
        out.append(name)
    return tuple(out)


def served_model_ids(config: Any) -> list[str]:
    """Primary first, then the aliases in flag order."""
    primary = getattr(config, "served_model_name", None) or config.model_path
    return [primary, *getattr(config, "served_model_aliases", ())]


def unknown_model_message(config: Any, requested: str | None) -> str | None:
    """None when ``requested`` is accepted; otherwise the refusal text.

    Accepted: any served id, or any name at all unless ``strict_model_name`` is set.
    """
    if not getattr(config, "strict_model_name", False):
        return None
    ids = served_model_ids(config)
    if requested in ids:
        return None
    return f"The model {requested!r} does not exist; this server serves {ids}"
