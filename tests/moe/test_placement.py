from __future__ import annotations

import json
from types import SimpleNamespace

from freetoken.moe.placement import (
    load_pageable_ranking,
    pageable_profile_path,
    save_pageable_ranking,
)


def test_pageable_placement_profile_round_trip(tmp_path, monkeypatch):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model-v1")
    monkeypatch.setenv("FREETOKEN_CACHE_DIR", str(tmp_path / "cache"))

    path = save_pageable_ranking(str(model), [2, 0, 1], [0.1, 0.2, 0.3])

    assert path == pageable_profile_path(str(model))
    assert load_pageable_ranking(str(model), 3) == (2, 0, 1)


def test_pageable_placement_profile_rejects_changed_model(tmp_path, monkeypatch):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model-v1")
    monkeypatch.setenv("FREETOKEN_CACHE_DIR", str(tmp_path / "cache"))
    path = save_pageable_ranking(str(model), [0, 1], [0.1, 0.2])

    model.write_bytes(b"model-version-two")

    assert path.exists()
    assert load_pageable_ranking(str(model), 2) is None


def test_pageable_placement_profile_rejects_invalid_permutation(tmp_path, monkeypatch):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model-v1")
    monkeypatch.setenv("FREETOKEN_CACHE_DIR", str(tmp_path / "cache"))
    path = pageable_profile_path(str(model))
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "model": {
                    "path": str(model.resolve()),
                    "size": model.stat().st_size,
                    "mtime_ns": model.stat().st_mtime_ns,
                },
                "ranking": [0, 0],
            }
        )
    )

    assert load_pageable_ranking(str(model), 2) is None


def test_pageable_profile_is_not_trained_in_off_or_read_modes():
    from freetoken.scheduler.scheduler import Scheduler

    for mode in ("off", "read"):
        scheduler = SimpleNamespace(
            config=SimpleNamespace(moe_pageable_profile=mode),
            engine=SimpleNamespace(moe_offload_cache=object()),
        )
        Scheduler._maybe_retune_pageable_layers(scheduler, [])
