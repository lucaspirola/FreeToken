"""Lightweight RAM-accounting tests; intentionally does not import real torch."""

from __future__ import annotations

import importlib.util
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest import mock


SOURCE = Path(__file__).parents[2] / "python/freetoken/scheduler/session_spill.py"


class _Counters:
    def __init__(self):
        self.prefetches = 0
        self.prefetches_failed = 0
        self.prefetches_collected = 0


class _Logger:
    def __getattr__(self, _name):
        return lambda *_args, **_kwargs: None


def _load_module():
    torch = types.ModuleType("torch")
    torch.Tensor = object
    torch.inference_mode = lambda: (lambda function: function)
    torch.load = lambda *_args, **_kwargs: object()
    freetoken = types.ModuleType("freetoken")
    freetoken.__path__ = []
    scheduler = types.ModuleType("freetoken.scheduler")
    scheduler.__path__ = []
    utils = types.ModuleType("freetoken.utils")
    utils.init_logger = lambda _name: _Logger()
    counters = types.ModuleType("freetoken.scheduler.counters")
    counters.SpillCounters = _Counters
    name = "freetoken.scheduler.session_spill"
    stubs = {
        "torch": torch,
        "freetoken": freetoken,
        "freetoken.scheduler": scheduler,
        "freetoken.utils": utils,
        "freetoken.scheduler.counters": counters,
    }
    # Keep test doubles out of shared test collection after this module is loaded.
    with mock.patch.dict(sys.modules, stubs):
        spec = importlib.util.spec_from_file_location(name, SOURCE)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module, torch


spill, fake_torch = _load_module()


class PrefetchReservationTest(unittest.TestCase):
    def setUp(self):
        self.store = spill.SessionSpillStore.__new__(spill.SessionSpillStore)
        self.store.ram_budget_bytes = 100
        self.store.host_reserve_bytes = 0
        self.store.ram_bytes = 0
        self.store.disk_bytes = 80
        self.store._prefetch_reserved_bytes = 0
        self.store._prefetch = None
        self.store._promoted = None
        self.store._records = []
        self.store._by_session = {}
        self.store.counters = _Counters()
        self.record = spill.SessionSpillRecord(
            token_ids=object(),
            num_pages=1,
            byte_size=80,
            fingerprint=(),
            tier="disk",
            chunks=[spill.SpillChunk("kv", 0, 0, file=Path("chunk.pt"))],
            session_id="queued",
        )
        self.store._records.append(self.record)
        self.store._by_session["queued"] = self.record
        self.old_available = spill._mem_available_bytes
        spill._mem_available_bytes = lambda: 1 << 40

    def tearDown(self):
        spill._mem_available_bytes = self.old_available
        fake_torch.load = lambda *_args, **_kwargs: object()

    def test_active_prefetch_reservation_blocks_other_ram_admission(self):
        entered = threading.Event()
        release = threading.Event()

        def blocked_load(*_args, **_kwargs):
            entered.set()
            release.wait(2)
            return object()

        fake_torch.load = blocked_load
        self.assertTrue(self.store.start_prefetch("queued"))
        self.assertTrue(entered.wait(1))
        self.assertEqual(self.store._prefetch_reserved_bytes, 80)
        self.assertFalse(self.store._ram_has_room(21))
        self.assertTrue(self.store.cancel_prefetch())
        self.assertEqual(self.store._prefetch_reserved_bytes, 80)
        self.assertIsNone(self.store.collect_prefetch(wait=False))
        self.assertEqual(self.store._prefetch_reserved_bytes, 80)
        release.set()
        self.assertIsNone(self.store.collect_prefetch(wait=True))
        self.assertEqual(self.store._prefetch_reserved_bytes, 0)

    def test_collect_transfers_reservation_without_double_charging(self):
        self.assertTrue(self.store.start_prefetch("queued"))
        self.assertEqual(self.store.collect_prefetch(wait=True), "queued")
        self.assertEqual(self.store._prefetch_reserved_bytes, 0)
        self.assertEqual(self.store.ram_bytes, 80)
        self.assertEqual(self.store.disk_bytes, 0)
        self.assertEqual(self.record.tier, "ram")

    def test_record_larger_than_budget_never_starts_or_reserves(self):
        self.record.byte_size = 101
        self.assertFalse(self.store.start_prefetch("queued"))
        self.assertEqual(self.store._prefetch_reserved_bytes, 0)
        self.assertEqual(self.store.counters.prefetches_failed, 1)

    def test_failed_read_releases_reservation_when_reaped(self):
        def failed_load(*_args, **_kwargs):
            raise OSError("torn checkpoint")

        fake_torch.load = failed_load
        self.assertTrue(self.store.start_prefetch("queued"))
        self.assertIsNone(self.store.collect_prefetch(wait=True))
        self.assertEqual(self.store._prefetch_reserved_bytes, 0)
        self.assertEqual(self.store.counters.prefetches_failed, 1)

    def test_thread_start_failure_rolls_back_reservation(self):
        with mock.patch.object(threading.Thread, "start", side_effect=RuntimeError):
            self.assertFalse(self.store.start_prefetch("queued"))
        self.assertIsNone(self.store._prefetch)
        self.assertEqual(self.store._prefetch_reserved_bytes, 0)
        self.assertEqual(self.store.counters.prefetches_failed, 1)

    def test_collect_drops_promotion_when_current_host_reserve_is_low(self):
        self.assertTrue(self.store.start_prefetch("queued"))
        spill._mem_available_bytes = lambda: (256 << 20) - 1
        self.assertIsNone(self.store.collect_prefetch(wait=True))
        self.assertEqual(self.store._prefetch_reserved_bytes, 0)
        self.assertEqual(self.store.ram_bytes, 0)
        self.assertEqual(self.store.disk_bytes, 80)
        self.assertEqual(self.record.tier, "disk")
        self.assertEqual(self.store.counters.prefetches_failed, 1)


if __name__ == "__main__":
    unittest.main()
