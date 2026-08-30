"""Persistent, model-scoped pageable expert-bank placement profiles."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def _model_identity(model_path: str) -> dict[str, int | str]:
    path = Path(model_path).expanduser().resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def pageable_profile_path(model_path: str) -> Path:
    identity = _model_identity(model_path)
    digest = hashlib.sha256(str(identity["path"]).encode()).hexdigest()[:16]
    cache_root = Path(
        os.environ.get(
            "FREETOKEN_CACHE_DIR",
            Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
            / "freetoken",
        )
    )
    return cache_root / "pageable-placement" / f"{digest}.json"


def load_pageable_ranking(model_path: str, num_layers: int) -> tuple[int, ...] | None:
    """Load a profile only when it still describes this exact model file."""
    try:
        payload = json.loads(pageable_profile_path(model_path).read_text())
        ranking = tuple(int(layer) for layer in payload["ranking"])
        if payload.get("version") != 1 or payload.get("model") != _model_identity(
            model_path
        ):
            return None
        if len(ranking) != num_layers or set(ranking) != set(range(num_layers)):
            return None
        return ranking
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def save_pageable_ranking(
    model_path: str,
    ranking: list[int] | tuple[int, ...],
    costs_seconds: list[float] | tuple[float, ...],
) -> Path:
    """Atomically save an all-layer ranking measured at an idle boundary."""
    if len(ranking) != len(costs_seconds) or set(ranking) != set(range(len(ranking))):
        raise ValueError("pageable placement ranking must be a full layer permutation")
    path = pageable_profile_path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "model": _model_identity(model_path),
        "ranking": list(ranking),
        "cost_seconds_per_step": list(costs_seconds),
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
    os.replace(temporary, path)
    return path
