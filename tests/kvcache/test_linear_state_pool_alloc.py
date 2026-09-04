"""P1 unit: LinearStatePool free-list allocator (alloc/free/clear_slots/copy_from).
CPU-only, fast — pure slot bookkeeping + state copy/zero, no kernels."""
from __future__ import annotations

import pytest
import torch

from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.models.config import LinearGatedDeltaGroupConfig


def _pool(num_slots=8, device="cpu"):
    group = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0, 1),
        num_key_heads=2, num_value_heads=4,
        key_head_dim=16, value_head_dim=16, conv_kernel_dim=4, output_gate=True,
    )
    return LinearStatePool(group=group, num_slots=num_slots, dtype=torch.bfloat16,
                           device=torch.device(device), tp_size=1)


def test_alloc_free_roundtrip():
    pool = _pool(num_slots=8)
    assert pool.num_free_slots == 7          # slots 1..7 (slot 0 = padding)
    a = pool.alloc(3)
    assert len(set(a)) == 3 and all(1 <= s <= 7 for s in a)
    assert pool.padding_slot not in a        # slot 0 never allocated
    assert pool.num_free_slots == 4
    pool.free(a)
    assert pool.num_free_slots == 7
    # int and tensor free forms
    s = pool.alloc(1)[0]
    pool.free(s)
    s2 = pool.alloc(2)
    pool.free(torch.tensor(s2, dtype=torch.long))
    assert pool.num_free_slots == 7


def test_alloc_exhaustion_raises():
    pool = _pool(num_slots=4)                # 3 allocatable
    pool.alloc(3)
    with pytest.raises(RuntimeError, match="exhausted"):
        pool.alloc(1)


def test_clear_slots_zeros_all_layers():
    pool = _pool(num_slots=6)
    s = pool.alloc(1)[0]
    pool.conv_states[:, s] = 1.5
    pool.recurrent_states[:, s] = 2.0
    pool.clear_slots([s])
    assert pool.conv_states[:, s].abs().sum() == 0
    assert pool.recurrent_states[:, s].abs().sum() == 0


def test_copy_from_snapshot():
    pool = _pool(num_slots=6)
    src, dst = pool.alloc(2)
    torch.manual_seed(0)
    pool.conv_states[:, src] = torch.randn_like(pool.conv_states[:, src])
    pool.recurrent_states[:, src] = torch.randn_like(pool.recurrent_states[:, src])
    pool.copy_from(src, dst)
    assert torch.equal(pool.conv_states[:, dst], pool.conv_states[:, src])
    assert torch.equal(pool.recurrent_states[:, dst], pool.recurrent_states[:, src])


def test_resize_preserve_compacts_occupied_slots():
    pool = _pool(num_slots=10)
    occupied = pool.alloc(4)
    for value, slot in enumerate(occupied, 1):
        pool.conv_states[:, slot].fill_(value)
        pool.recurrent_states[:, slot].fill_(value + 10)
    remap = {slot: i + 1 for i, slot in enumerate(sorted(occupied))}
    pool.resize_preserve(6, remap)
    assert pool.num_slots == 6
    assert pool.occupied_slots == set(remap.values())
    for value, old in enumerate(occupied, 1):
        new = remap[old]
        assert bool((pool.conv_states[:, new] == value).all())
        assert bool((pool.recurrent_states[:, new] == value + 10).all())


def test_resize_preserve_rejects_incomplete_remap():
    pool = _pool(num_slots=6)
    pool.alloc(2)
    with pytest.raises(ValueError, match="remap covers"):
        pool.resize_preserve(8, {})


if __name__ == "__main__":
    test_alloc_free_roundtrip()
    test_alloc_exhaustion_raises()
    test_clear_slots_zeros_all_layers()
    test_copy_from_snapshot()
    print("LinearStatePool allocator unit: PASS")


# ---------------------------------------------------------------- state_layout geometry
def _lightning_group():
    """Nemotron-3.5 Lightning: 23 mamba layers, H=64 heads x P=64, N=128 d_state, G=8."""
    return LinearGatedDeltaGroupConfig(
        name="mamba", layer_ids=tuple(range(23)),
        num_key_heads=8, num_value_heads=64, key_head_dim=64, value_head_dim=128,
        conv_kernel_dim=4, output_gate=True, state_layout="mamba2", track_chunk_size=128,
    )


def test_mamba2_pool_is_the_native_hpn_block():
    """[L, slots, H, P, N] recurrent + a 6144-wide conv stream -- the SSD / flashinfer
    layout, so no kernel input or output is ever transposed."""
    pool = LinearStatePool(group=_lightning_group(), num_slots=5, dtype=torch.bfloat16,
                           device=torch.device("cpu"), tp_size=1)
    assert tuple(pool.recurrent_states.shape) == (23, 5, 64, 64, 128)
    assert pool.recurrent_states.dtype is torch.float32     # decode kernel requires fp32
    assert tuple(pool.conv_states.shape) == (23, 5, 6144, 3)
    assert pool.state_layout == "mamba2" and pool.track_chunk_size == 128
    assert pool.group is pool._group


def test_mamba2_state_bytes_per_slot():
    from freetoken.kvcache.linear_state_pool import linear_state_bytes_per_req

    group = _lightning_group()
    conv = 6144 * 3 * 2            # bf16 conv window
    rec = 64 * 64 * 128 * 4        # fp32 [H, P, N]
    assert linear_state_bytes_per_req(group, 1, torch.bfloat16) == 23 * (conv + rec)


def test_gdn_pool_geometry_is_unchanged():
    """The "kv" layout (Qwen3.5 / Ornith) keeps 2*Hk*Dk + Hv*Dv conv channels."""
    pool = _pool(num_slots=4)
    assert pool.state_layout == "kv" and pool.track_chunk_size == 64
    assert tuple(pool.conv_states.shape) == (2, 4, 2 * 2 * 16 + 4 * 16, 3)
    assert tuple(pool.recurrent_states.shape) == (2, 4, 4, 16, 16)
