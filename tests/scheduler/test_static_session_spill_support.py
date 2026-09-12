"""Focused policy tests for fixed-arena cold-session spill support.

These tests deliberately extract the two small decision paths from source and execute
them with plain Python stubs.  They validate policy parity only; they are not a 1M-token
or CUDA integration test.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import time
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).parents[2]
SPILL_SOURCE = ROOT / "python/freetoken/scheduler/session_spill.py"
SCHEDULER_SOURCE = ROOT / "python/freetoken/scheduler/scheduler.py"


def _load_spill_module():
    """Load the factory with a torch-shaped stub, not the real torch package."""
    torch = types.ModuleType("torch")
    torch.Tensor = object
    torch.inference_mode = lambda: (lambda function: function)
    freetoken = types.ModuleType("freetoken")
    freetoken.__path__ = []
    scheduler = types.ModuleType("freetoken.scheduler")
    scheduler.__path__ = []
    utils = types.ModuleType("freetoken.utils")
    utils.init_logger = lambda _name: SimpleNamespace()
    counters = types.ModuleType("freetoken.scheduler.counters")
    counters.SpillCounters = lambda: SimpleNamespace()
    name = "freetoken.scheduler.session_spill_static_policy"
    stubs = {
        "torch": torch,
        "freetoken": freetoken,
        "freetoken.scheduler": scheduler,
        "freetoken.utils": utils,
        "freetoken.scheduler.counters": counters,
    }
    with mock.patch.dict(sys.modules, stubs):
        spec = importlib.util.spec_from_file_location(name, SPILL_SOURCE)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


def _restore_cold_session():
    """Compile just Scheduler._restore_cold_session against lightweight stubs."""
    tree = ast.parse(SCHEDULER_SOURCE.read_text(encoding="utf-8"))
    scheduler = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Scheduler")
    function = next(node for node in scheduler.body if isinstance(node, ast.FunctionDef) and node.name == "_restore_cold_session")
    function.decorator_list = []
    ast.fix_missing_locations(function)
    namespace = {
        "_common_prefix_len": lambda _old, _new, maximum: maximum,
        "logger": SimpleNamespace(debug_rank0=lambda *_args: None, info_rank0=lambda *_args: None, warning=lambda *_args: None),
        "time": time,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SCHEDULER_SOURCE), "exec"), namespace)
    return namespace["_restore_cold_session"]


class StaticSpillFactoryTest(unittest.TestCase):
    def setUp(self):
        self.spill = _load_spill_module()
        self.pool = SimpleNamespace(growable=False, iter_session_spill_tensors=lambda: None)
        self.engine = SimpleNamespace(kv_cache=self.pool, linear_state_pool=object())
        self.config = SimpleNamespace(
            session_spill_dir="/unused", tp_info=SimpleNamespace(size=1), page_size=1,
            session_spill_ram_gb=1, session_spill_disk_gb=1, host_ram_reserve_gb=0,
            session_spill_limit_gb=1, session_spill_persist=False, model_path="model",
            session_spill_state_stride=1,
        )

    def test_static_page_one_hybrid_pool_is_supported(self):
        initialized = []
        with mock.patch.object(self.spill.SessionSpillStore, "__init__", lambda instance, *_args, **_kwargs: initialized.append(instance)):
            store = self.spill.SessionSpillStore.create_if_supported(self.engine, self.config)
        self.assertIsNotNone(store)
        self.assertEqual(len(initialized), 1)

    def test_factory_retains_non_growability_guards(self):
        for field, value in (("session_spill_dir", None), ("page_size", 2)):
            with self.subTest(field=field):
                setattr(self.config, field, value)
                self.assertIsNone(self.spill.SessionSpillStore.create_if_supported(self.engine, self.config))
                setattr(self.config, field, "/unused" if field == "session_spill_dir" else 1)
        self.config.tp_info.size = 2
        self.assertIsNone(self.spill.SessionSpillStore.create_if_supported(self.engine, self.config))
        self.config.tp_info.size = 1
        self.engine.linear_state_pool = None
        self.assertIsNone(self.spill.SessionSpillStore.create_if_supported(self.engine, self.config))


    def test_factory_rejects_pool_without_spill_export(self):
        self.engine.kv_cache = SimpleNamespace(growable=False)
        self.assertIsNone(self.spill.SessionSpillStore.create_if_supported(self.engine, self.config))


class StaticRestorePolicyTest(unittest.TestCase):
    def setUp(self):
        self.restore = _restore_cold_session()

    def _scheduler(self, *, growable: bool, grow_step: int):
        counters = SimpleNamespace(restores_deferred=0, restores=0, restores_failed=0)
        record = SimpleNamespace(
            valid=True, token_ids=[1, 2, 3, 4], num_pages=4, tier="disk", byte_size=4,
            restorable_length=lambda _matched: 4,
        )
        store = SimpleNamespace(
            counters=counters, get=lambda _sid: record,
            collect_prefetch=lambda *_args, **_kwargs: None, touch=lambda _record: None,
        )
        session = SimpleNamespace(spill=None, handle=None)
        cm = SimpleNamespace(
            available_size=100, committed_pages=16,
            session_restore_footprint=lambda _tokens: 0,
            hybrid_session_restore_geometry=lambda _tokens: (5, 4),
            add_committed_pages=lambda pages: setattr(cm, "added_pages", pages),
            restore_hybrid_session_prefix=lambda *_args: "restored-handle",
        )
        engine = SimpleNamespace(kv_cache=SimpleNamespace(growable=growable), grow_calls=[])
        engine.grow_runtime_kv = lambda required: (engine.grow_calls.append(required) or (16, 17))
        scheduler = SimpleNamespace(
            _sessions={"session": session}, _session_spill_store=store,
            prefill_manager=SimpleNamespace(finishability_reservation=lambda: 0),
            cache_manager=cm, engine=engine,
            config=SimpleNamespace(kv_grow_step_tokens=grow_step), discarded=[],
        )
        scheduler._discard_session_spill = lambda lease: scheduler.discarded.append(lease)
        return scheduler, session, record, counters, engine, cm

    def test_static_shortage_defers_without_growth_or_discard(self):
        scheduler, session, record, counters, engine, cm = self._scheduler(growable=False, grow_step=0)
        self.assertFalse(self.restore(scheduler, "session", [1, 2, 3, 4, 5]))
        self.assertEqual(engine.grow_calls, [])
        self.assertEqual(counters.restores_deferred, 1)
        self.assertEqual(scheduler.discarded, [])
        self.assertIs(session.spill, record)
        self.assertFalse(hasattr(cm, "added_pages"))

    def test_growable_shortage_still_grows_and_restores(self):
        scheduler, session, _record, counters, engine, cm = self._scheduler(growable=True, grow_step=64)
        self.assertTrue(self.restore(scheduler, "session", [1, 2, 3, 4, 5]))
        self.assertEqual(engine.grow_calls, [17])
        self.assertEqual(cm.added_pages, 17)
        self.assertEqual(counters.restores, 1)
        self.assertEqual(session.handle, "restored-handle")

    def test_growable_pool_with_zero_step_defers(self):
        scheduler, _session, _record, counters, engine, _cm = self._scheduler(growable=True, grow_step=0)
        self.assertFalse(self.restore(scheduler, "session", [1, 2, 3, 4, 5]))
        self.assertEqual(engine.grow_calls, [])
        self.assertEqual(counters.restores_deferred, 1)

    def test_static_restore_with_enough_space_never_grows(self):
        scheduler, session, _record, counters, engine, cm = self._scheduler(growable=False, grow_step=0)
        cm.hybrid_session_restore_geometry = lambda _tokens: (4, 4)
        self.assertTrue(self.restore(scheduler, "session", [1, 2, 3, 4, 5]))
        self.assertEqual(engine.grow_calls, [])
        self.assertEqual(counters.restores, 1)
        self.assertEqual(counters.restores_deferred, 0)
        self.assertEqual(session.handle, "restored-handle")
        self.assertEqual(scheduler.discarded, [session])

    def test_static_pool_positive_step_still_defers(self):
        scheduler, session, record, counters, engine, _cm = self._scheduler(growable=False, grow_step=64)
        self.assertFalse(self.restore(scheduler, "session", [1, 2, 3, 4, 5]))
        self.assertEqual(engine.grow_calls, [])
        self.assertEqual(counters.restores_deferred, 1)
        self.assertEqual(scheduler.discarded, [])
        self.assertIs(session.spill, record)


if __name__ == "__main__":
    unittest.main()
