"""A mirror reserve below 2E is refused at startup, naming the minimum (CPU only).

Ornith NVFP4 (E=256) with FREETOKEN_MIRROR_RESERVE_ROWS=256 started, parked the
arena at its coverage floor and then died in the warmup prefill with "bounded
expert mirror lost coverage before prefill" (ft-dev 2026-09-24, lfuo* arms):
the first prefill writes the 2E warm-start residents of the prefill double
buffer back to the pool, and at the floor the reserve is all the free rows
there are. ``min_reserve_rows`` is that 2E; ``resolve_reserve_rows`` (the one
place the variable is read, at engine startup) refuses anything smaller.
"""
from __future__ import annotations

import pytest

from freetoken.moe.mirror_pool import (
    default_reserve_rows,
    min_reserve_rows,
    prefill_buffer_slots,
    resolve_reserve_rows,
)

ENV = "FREETOKEN_MIRROR_RESERVE_ROWS"


@pytest.mark.parametrize("e", [128, 256])
def test_reserve_below_two_e_is_refused_with_the_minimum(monkeypatch, e):
    for raw in (str(e), str(2 * e - 1)):
        monkeypatch.setenv(ENV, raw)
        with pytest.raises(ValueError, match=rf"{ENV}.*minimum of {2 * e} rows"):
            resolve_reserve_rows(e)


@pytest.mark.parametrize("e", [128, 256])
def test_two_e_and_the_default_are_accepted(monkeypatch, e):
    assert min_reserve_rows(e) == prefill_buffer_slots(e) == 2 * e
    monkeypatch.setenv(ENV, str(2 * e))
    assert resolve_reserve_rows(e) == 2 * e
    monkeypatch.delenv(ENV)
    assert resolve_reserve_rows(e) == default_reserve_rows(e) >= min_reserve_rows(e)


def test_the_message_says_what_to_set(monkeypatch):
    monkeypatch.setenv(ENV, "256")
    with pytest.raises(ValueError) as err:
        resolve_reserve_rows(256)
    msg = str(err.value)
    assert "at least 512" in msg and "warmup prefill" in msg
