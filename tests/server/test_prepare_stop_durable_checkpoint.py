import asyncio
from types import SimpleNamespace

import pytest

from freetoken.message import DurableCheckpointReply
from freetoken.server.accounting import AccountingDrainError, prepare_stop_accounting
from freetoken.server.api_server import FrontendManager
from freetoken.server.stats import StatsTracker


def _state(reply):
    calls = []

    async def checkpoint(operation_id, timeout_s):
        calls.append((operation_id, timeout_s))
        return reply

    return SimpleNamespace(
        maintenance_state="serving", instance_id="instance-1",
        config=SimpleNamespace(served_model_name="model"), stats=StatsTracker(),
        ready_at=None, durable_checkpoint=checkpoint, calls=calls,
    )


def test_accounting_only_seal_does_not_bypass_later_checkpoint_upgrade():
    reply = DurableCheckpointReply("op-1", "complete", 2, "digest", ["a", "b"])
    state = _state(reply)
    first = asyncio.run(prepare_stop_accounting(state))
    assert "checkpoint" not in first
    upgraded = asyncio.run(prepare_stop_accounting(
        state, checkpoint_sessions=True, checkpoint_operation_id="op-1"))
    assert upgraded["checkpoint"]["complete"] is True
    assert state.calls == [("op-1", 120.0)]
    assert asyncio.run(prepare_stop_accounting(
        state, checkpoint_sessions=True, checkpoint_operation_id="op-1")) == upgraded
    assert len(state.calls) == 1


def test_mismatched_or_failed_checkpoint_never_seals_checkpoint_evidence():
    state = _state(DurableCheckpointReply("wrong", "complete"))
    with pytest.raises(AccountingDrainError, match="correlation mismatch"):
        asyncio.run(prepare_stop_accounting(
            state, checkpoint_sessions=True, checkpoint_operation_id="op-1"))
    assert not hasattr(state, "_sealed_accounting")


def test_frontend_timeout_retry_is_single_operation_and_changed_id_conflicts():
    class Queue:
        def __init__(self): self.items = []
        async def put(self, item): self.items.append(item)

    async def exercise():
        queue = Queue()
        manager = FrontendManager(
            config=SimpleNamespace(), send_tokenizer=queue, recv_tokenizer=None,
            initialized=True)
        with pytest.raises(asyncio.TimeoutError):
            await manager.durable_checkpoint("op-1", 0)
        with pytest.raises(asyncio.TimeoutError):
            await manager.durable_checkpoint("op-1", 0)
        assert len(queue.items) == 1
        with pytest.raises(RuntimeError, match="pending"):
            await manager.durable_checkpoint("op-2", 0)
        future = manager.durable_checkpoint_futures.pop("op-1")
        reply = DurableCheckpointReply("op-1", "complete")
        manager.durable_checkpoint_results["op-1"] = reply
        if not future.done():
            future.set_result(reply)
        assert await manager.durable_checkpoint("op-1", 0) is reply

    asyncio.run(exercise())
