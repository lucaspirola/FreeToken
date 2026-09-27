"""The measured prefill transient is recorded per setup and prices the next start's
bounded mirror pool; a pool priced below the measurement fails the load, not the
request that reaches the KV ceiling. CPU-only: no engine, no GPU."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from freetoken.engine import engine as E
from freetoken.engine.growable_kv import PREFILL_HEADROOM_MARGIN_BYTES

MiB = 1 << 20


def _config(tmp_path, **kw):
    base = dict(
        model_path=str(tmp_path / "model"),
        max_extend_tokens=8192,
        attention_backend="triton",
        kv_cache_dtype="q8_0",
        kv_cache_dtype_k=None,
        kv_cache_dtype_v=None,
        expert_residency="mirror",
        tp_info=SimpleNamespace(rank=0, size=1),
        page_size=1,
    )
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.fixture
def record(tmp_path, monkeypatch):
    path = tmp_path / "rec" / "prefill-transient.json"
    monkeypatch.setenv("FREETOKEN_PREFILL_TRANSIENT_RECORD", str(path))
    monkeypatch.delenv("FREETOKEN_PREFILL_TRANSIENT_MB", raising=False)
    return path


def test_estimate_without_a_record_is_the_per_token_estimate(tmp_path, record):
    assert E._prefill_transient_estimate(_config(tmp_path)) == 8192 * 128 * 1024


def test_a_recorded_measurement_raises_the_estimate_but_never_lowers_it(tmp_path, record):
    cfg = _config(tmp_path)
    E._record_prefill_transient(cfg, 1146 * MiB)
    assert E._prefill_transient_estimate(cfg) == 1146 * MiB
    E._record_prefill_transient(cfg, 600 * MiB)  # the latest measurement is kept ...
    assert json.loads(record.read_text()) == {E._prefill_transient_key(cfg): 600 * MiB}
    assert E._prefill_transient_estimate(cfg) == 1024 * MiB  # ... the estimate floors it


def test_records_are_per_setup(tmp_path, record):
    cfg = _config(tmp_path)
    E._record_prefill_transient(cfg, 1146 * MiB)
    for other in (
        _config(tmp_path, kv_cache_dtype="q4_0"),
        _config(tmp_path, max_extend_tokens=4096),
        _config(tmp_path, expert_residency="whole"),
        _config(tmp_path, model_path=str(tmp_path / "other")),
    ):
        assert E._prefill_transient_recorded(other) is None
    assert E._prefill_transient_recorded(cfg) == 1146 * MiB


def test_env_override_wins_and_off_disables_the_record(tmp_path, record, monkeypatch):
    cfg = _config(tmp_path)
    E._record_prefill_transient(cfg, 1146 * MiB)
    monkeypatch.setenv("FREETOKEN_PREFILL_TRANSIENT_MB", "900")
    assert E._prefill_transient_estimate(cfg) == 900 * MiB
    monkeypatch.delenv("FREETOKEN_PREFILL_TRANSIENT_MB")
    monkeypatch.setenv("FREETOKEN_PREFILL_TRANSIENT_RECORD", "off")
    assert E._prefill_transient_estimate(cfg) == 1024 * MiB
    E._record_prefill_transient(cfg, 2000 * MiB)  # no-op
    monkeypatch.setenv("FREETOKEN_PREFILL_TRANSIENT_RECORD", str(record))
    assert E._prefill_transient_recorded(cfg) == 1146 * MiB


def test_a_corrupt_record_is_ignored_and_replaced(tmp_path, record):
    cfg = _config(tmp_path)
    record.parent.mkdir(parents=True)
    record.write_text("{not json")
    assert E._prefill_transient_recorded(cfg) is None
    E._record_prefill_transient(cfg, 1146 * MiB)
    assert E._prefill_transient_recorded(cfg) == 1146 * MiB


SLOT = 2 * MiB


def _engine(tmp_path, *, floor, sized, measured, plan_slack_bytes):
    """A stub engine whose ceiling plan leaves ``plan_slack_bytes`` above ``floor``
    before any extra reserve; each byte of extra reserve costs arena bytes."""
    eng = E.Engine.__new__(E.Engine)
    eng.config = _config(tmp_path)
    eng.num_pages = 262144
    eng.prefill_transient_sized = sized
    eng.prefill_transient_bytes = measured
    eng.prefill_transient_measured = True

    def plan(pages, *, extra_vmm_reserve_bytes=0):
        return floor + (plan_slack_bytes - extra_vmm_reserve_bytes) // SLOT, 0

    eng.growable_kv = SimpleNamespace(_plan_growable_kv=plan)
    eng.moe_offload_cache = SimpleNamespace(
        residency=SimpleNamespace(min_gpu_slots=lambda: floor, pool=SimpleNamespace(capacity=6737)),
        arena_layout=(6076, 8),
        class_arena_layouts=None,
    )
    return eng


def test_underpriced_pool_with_thin_slack_fails_the_load(tmp_path, record):
    eng = _engine(tmp_path, floor=4272, sized=1024 * MiB, measured=1146 * MiB,
                  plan_slack_bytes=16 * MiB)
    with pytest.raises(RuntimeError, match="so starting again sizes the pool for it"):
        eng._validate_growable_ceiling()


def test_underpriced_pool_with_enough_slack_loads(tmp_path, record):
    eng = _engine(tmp_path, floor=4272, sized=1024 * MiB, measured=1146 * MiB,
                  plan_slack_bytes=PREFILL_HEADROOM_MARGIN_BYTES + 16 * MiB)
    eng._validate_growable_ceiling()


def test_pool_priced_for_the_measurement_with_thin_slack_names_the_rows(tmp_path, record):
    # Priced at (or above) what was measured: a restart would not help, more rows would.
    eng = _engine(tmp_path, floor=4272, sized=1196 * MiB, measured=1146 * MiB,
                  plan_slack_bytes=16 * MiB)
    with pytest.raises(RuntimeError, match="raise --moe-mirror-host-rows to at least 6793"):
        eng._validate_growable_ceiling()
    eng = _engine(tmp_path, floor=4272, sized=1196 * MiB, measured=1146 * MiB,
                  plan_slack_bytes=PREFILL_HEADROOM_MARGIN_BYTES + 16 * MiB)
    eng._validate_growable_ceiling()


def test_whole_model_residency_is_never_refused(tmp_path, record):
    eng = _engine(tmp_path, floor=0, sized=1024 * MiB, measured=1146 * MiB,
                  plan_slack_bytes=0)
    eng._validate_growable_ceiling()


def _prediction(bytes_, chunk=8192, coarse=False):
    from freetoken.engine.memory_prediction import TransientPrediction

    return TransientPrediction(chunk_tokens=chunk, bytes=bytes_, peak_layer="gdn",
                               layers={"gdn": bytes_}, base=0, coarse=coarse)


def test_upper_bound_is_the_largest_in_band_measurement():
    from freetoken.engine import memory_prediction as mp

    pred = _prediction(897 * MiB)  # Ornith's closed form, 0.876 GiB
    bound = mp.transient_upper_bound(pred)
    assert not mp.compare(pred.bytes, bound)[1]  # in band at the bound ...
    assert mp.compare(pred.bytes, bound + MiB)[1]  # ... and out of it just above
    assert mp.transient_upper_bound(_prediction(897 * MiB, coarse=True)) is None
    assert mp.transient_upper_bound(None) is None


def test_prediction_bound_prices_the_first_start(tmp_path, record):
    cfg = _config(tmp_path)
    # Ornith: bound 1196 MiB covers the measured 1146 (ft-dev) / 1166 (owner) MiB.
    assert E._prefill_transient_estimate(cfg, _prediction(897 * MiB)) == 1196 * MiB
    # Nemotron: bound 0.90 GiB stays under the per-token estimate, which stands.
    assert E._prefill_transient_estimate(cfg, _prediction(693 * MiB)) == 1024 * MiB
    # A prediction for another chunk length or a coarse one is not used.
    assert E._prefill_transient_estimate(cfg, _prediction(897 * MiB, chunk=4096)) == 1024 * MiB
    assert E._prefill_transient_estimate(cfg, _prediction(897 * MiB, coarse=True)) == 1024 * MiB
    # A record above the bound still wins; the env override beats everything.
    E._record_prefill_transient(cfg, 1300 * MiB)
    assert E._prefill_transient_estimate(cfg, _prediction(897 * MiB)) == 1300 * MiB
