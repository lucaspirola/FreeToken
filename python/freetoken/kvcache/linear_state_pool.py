from __future__ import annotations

import torch
from freetoken.distributed import get_tp_info
from freetoken.env import ENV
from freetoken.models.config import LinearGatedDeltaGroupConfig
from freetoken.utils import div_even

_SSM_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def ssm_state_dtype() -> torch.dtype:
    """Recurrent (SSM) state dtype, from FREETOKEN_MAMBA_SSM_DTYPE (default fp32)."""
    return _SSM_DTYPES.get(str(ENV.MAMBA_SSM_DTYPE).lower(), torch.float32)


def _linear_local_dims(
    group: LinearGatedDeltaGroupConfig, tp_size: int
) -> tuple[int, int, int]:
    """TP-local ``(n_layers, conv_dim, v_heads)`` for the recurrent state tensors -- the
    single source of the sharding math shared by the pool allocation and the byte estimate.

    The recurrent slot is always ``[v_heads, key_head_dim, value_head_dim]``; only the conv
    stream differs between the two layouts, because the projection that feeds the depthwise
    conv packs different things:

    ``kv`` (GDN)     q, k (``key_head_dim`` each, ``key_heads`` of them) then v.
    ``mamba2`` (SSD) x (``v_heads x key_head_dim`` = the SSM inner width) then B and C
                     (``key_heads x value_head_dim`` = G x N each) -- 4096 + 2*8*128 = 6144
                     for Nemotron-3.5 Lightning.
    """
    local_k_heads = div_even(group.num_key_heads, tp_size, allow_replicate=True)
    local_v_heads = div_even(group.num_value_heads, tp_size, allow_replicate=True)
    if group.state_layout == "mamba2":
        local_conv_dim = (
            local_v_heads * group.key_head_dim + 2 * local_k_heads * group.value_head_dim
        )
    else:
        local_conv_dim = (
            2 * local_k_heads * group.key_head_dim + local_v_heads * group.value_head_dim
        )
    return len(group.layer_ids), local_conv_dim, local_v_heads


class LinearStatePool:
    """Per-request recurrent state (conv + SSM) for GatedDeltaNet layers.

    Indexed by ``Req.table_idx`` (0..max_running_req), the same per-request slot the
    page table uses, so the scheduler's existing admit/free of ``table_idx`` covers the
    state's lifetime. One fixed slot per running request; no paging, no eviction.
    """

    def __init__(
        self,
        group: LinearGatedDeltaGroupConfig,
        num_slots: int,
        dtype: torch.dtype,
        device: torch.device,
        tp_size: int | None = None,
    ) -> None:
        if tp_size is None:
            tp_size = get_tp_info().size

        self._group = group
        self._num_slots = num_slots
        self._device = device
        self._conv_dtype = dtype

        n_layers, local_conv_dim, local_v_heads = _linear_local_dims(group, tp_size)

        # conv left-context: the last (kernel-1) timesteps of the conv input stream.
        self.conv_states = torch.zeros(
            (n_layers, num_slots, local_conv_dim, group.conv_kernel_dim - 1),
            dtype=dtype,
            device=device,
        )
        # SSM recurrent state. fp32 by default (matches HF mamba_ssm_dtype); the dtype is
        # overridable via FREETOKEN_MAMBA_SSM_DTYPE (see ssm_state_dtype).
        self.recurrent_states = torch.zeros(
            (n_layers, num_slots, local_v_heads, group.key_head_dim, group.value_head_dim),
            dtype=ssm_state_dtype(),
            device=device,
        )
        self._local_index = {layer_id: i for i, layer_id in enumerate(group.layer_ids)}

        # Free-list allocator over slots 1..num_slots-1 (slot 0 reserved as a padding sink,
        # sglang MambaPool convention). Live working slots, ping-pong track slots, and
        # radix-tree-donated snapshots are all drawn from this single free-list, so memory
        # flows between them by demand. Unused by the op harness (which assigns slots by hand).
        self.padding_slot = 0
        self._free_slots: list[int] = list(range(1, num_slots))

    @property
    def num_free_slots(self) -> int:
        return len(self._free_slots)

    def alloc(self, n: int = 1) -> list[int]:
        """Pop ``n`` free slot ids (LIFO). Raises if the pool is exhausted."""
        if n > len(self._free_slots):
            raise RuntimeError(
                f"LinearStatePool exhausted: need {n}, have {len(self._free_slots)}"
            )
        return [self._free_slots.pop() for _ in range(n)]

    def reclaim_all_slots(self) -> None:
        """Restore the free-list to all non-padding slots. Idle-only: the caller (e.g. a
        CacheManager rebuild that discards the tree owning donated snapshots) must guarantee no
        running request holds a slot, otherwise live state would be handed out twice."""
        self._free_slots = list(range(1, self._num_slots))

    def rebuild(self, num_slots: int) -> None:
        """Reallocate the conv + recurrent state tensors for ``num_slots`` slots IN PLACE.

        Geometry (layers, conv dim, head dims) and dtypes are taken from the existing
        tensors; only the slot count changes. Object identity is preserved so cached
        references (ctx.linear_state_pool) stay valid. Idle-only and destructive: every
        live/snapshot state is dropped, so the caller must guarantee no running request
        holds a slot and the radix tree owning donated snapshots is discarded too.
        """
        n_layers, _, local_conv_dim, km1 = self.conv_states.shape
        _, _, local_v_heads, key_head_dim, value_head_dim = self.recurrent_states.shape
        conv_dtype, rec_dtype = self.conv_states.dtype, self.recurrent_states.dtype
        device = self._device
        self.conv_states = None
        self.recurrent_states = None
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
        self.conv_states = torch.zeros(
            (n_layers, num_slots, local_conv_dim, km1), dtype=conv_dtype, device=device
        )
        self.recurrent_states = torch.zeros(
            (n_layers, num_slots, local_v_heads, key_head_dim, value_head_dim),
            dtype=rec_dtype,
            device=device,
        )
        self._num_slots = num_slots
        self._free_slots = list(range(1, num_slots))

    def resize_preserve(self, num_slots: int, remap: dict[int, int]) -> None:
        """Resize while preserving every occupied slot according to ``remap``.

        The caller owns all external slot references and updates them at the same
        no-forward-in-flight boundary.  Unlike :meth:`rebuild`, this supports live
        elastic-capacity changes without dropping an agent's recurrent state.
        """
        occupied = set(range(1, self._num_slots)) - set(self._free_slots)
        if set(remap) != occupied:
            raise ValueError(
                f"linear-state remap covers {len(remap)} slots, expected {len(occupied)}"
            )
        targets = list(remap.values())
        if len(set(targets)) != len(targets) or any(
            target <= 0 or target >= num_slots for target in targets
        ):
            raise ValueError(f"invalid linear-state remap for {num_slots} slots")

        n_layers, _, local_conv_dim, km1 = self.conv_states.shape
        _, _, local_v_heads, key_head_dim, value_head_dim = self.recurrent_states.shape
        new_conv = torch.zeros(
            (n_layers, num_slots, local_conv_dim, km1),
            dtype=self.conv_states.dtype,
            device=self._device,
        )
        new_recurrent = torch.zeros(
            (n_layers, num_slots, local_v_heads, key_head_dim, value_head_dim),
            dtype=self.recurrent_states.dtype,
            device=self._device,
        )
        if remap:
            src = torch.tensor(list(remap), dtype=torch.long, device=self._device)
            dst = torch.tensor(targets, dtype=torch.long, device=self._device)
            new_conv[:, dst] = self.conv_states[:, src]
            new_recurrent[:, dst] = self.recurrent_states[:, src]
        self.conv_states = new_conv
        self.recurrent_states = new_recurrent
        self._num_slots = num_slots
        used = set(targets)
        self._free_slots = [slot for slot in range(1, num_slots) if slot not in used]

    @property
    def occupied_slots(self) -> set[int]:
        return set(range(1, self._num_slots)) - set(self._free_slots)

    def free(self, slots) -> None:
        """Return slot ids to the free-list. Accepts an int, list, or 1-D tensor."""
        if isinstance(slots, torch.Tensor):
            slots = slots.flatten().tolist()
        elif isinstance(slots, int):
            slots = [slots]
        self._free_slots.extend(int(s) for s in slots)

    def clear_slots(self, slots) -> None:
        """Zero conv + recurrent state at ``slots`` across all linear layers (fresh sequence)."""
        if isinstance(slots, (list, tuple)):
            slots = torch.as_tensor(slots, dtype=torch.long, device=self._device)
        self.conv_states[:, slots] = 0
        self.recurrent_states[:, slots] = 0

    def copy_from(self, src: int, dst: int) -> None:
        """Copy a whole-sequence snapshot (conv + recurrent, all layers) from slot ``src`` to
        ``dst``. Used for COW-on-restore (donated snapshot -> fresh live slot)."""
        self.conv_states[:, dst].copy_(self.conv_states[:, src])
        self.recurrent_states[:, dst].copy_(self.recurrent_states[:, src])

    @property
    def group(self) -> LinearGatedDeltaGroupConfig:
        return self._group

    @property
    def state_layout(self) -> str:
        return self._group.state_layout

    @property
    def track_chunk_size(self) -> int:
        """Chunk granularity of the hybrid-radix state snapshots (64 GDN / 128 Mamba-2)."""
        return self._group.track_chunk_size

    def is_linear_layer(self, layer_id: int) -> bool:
        return layer_id in self._local_index

    def local_index(self, layer_id: int) -> int:
        return self._local_index[layer_id]

    def conv_state(self, layer_id: int, table_idx: int) -> torch.Tensor:
        return self.conv_states[self._local_index[layer_id], table_idx]

    def recurrent_state(self, layer_id: int, table_idx: int) -> torch.Tensor:
        return self.recurrent_states[self._local_index[layer_id], table_idx]

    def reset(self, table_idx: int) -> None:
        """Zero a slot across all linear layers (new request takes this table_idx)."""
        self.conv_states[:, table_idx].zero_()
        self.recurrent_states[:, table_idx].zero_()

    @property
    def num_linear_layers(self) -> int:
        return len(self._local_index)

    @property
    def num_slots(self) -> int:
        return self._num_slots

    @property
    def device(self) -> torch.device:
        return self._device

    def bytes_per_slot(self) -> int:
        """Total state bytes for one request (all linear layers)."""
        per = (
            self.conv_states[:, 0].numel() * self.conv_states.element_size()
            + self.recurrent_states[:, 0].numel() * self.recurrent_states.element_size()
        )
        return int(per)


def linear_state_bytes_per_req(
    group: LinearGatedDeltaGroupConfig,
    tp_size: int,
    dtype: torch.dtype,
) -> int:
    """Linear-state bytes for one request across all linear layers (TP-local)."""
    n_layers, local_conv_dim, local_v_heads = _linear_local_dims(group, tp_size)

    conv_elems = local_conv_dim * (group.conv_kernel_dim - 1)
    rec_elems = local_v_heads * group.key_head_dim * group.value_head_dim
    conv_bytes = conv_elems * dtype.itemsize  # conv state in model dtype
    rec_bytes = rec_elems * ssm_state_dtype().itemsize  # recurrent state (default fp32)
    return int(n_layers * (conv_bytes + rec_bytes))


__all__ = ["LinearStatePool", "linear_state_bytes_per_req"]


def state_pool_bytes(config, num_slots: int | None = None) -> int:
    """Total GDN state-pool bytes at ``num_slots`` PHYSICAL slots (default: the startup
    slot count). The engine adds this to the KV family's fixed cost when budgeting --
    the state pool is a sibling pool, not a KV tier."""
    linear_group = config.model_config.linear_attention_group()
    if linear_group is None:
        return 0
    slots = num_slots if num_slots is not None else _linear_pool_num_slots(config)
    return linear_state_bytes_per_req(linear_group, config.tp_info.size, config.dtype) * slots


def _linear_pool_num_slots(config) -> int:
    """LinearStatePool slot count. Hybrid-radix non-evictable peak is 4 slots per running request
    (1 live + 2 ping-pong + 1 committed snapshot locked through decode), plus a cross-request
    snapshot cache and a padding sink; naive GDN keeps the old (max_running_req + 1)."""
    mr = getattr(config, "elastic_initial_requests", None) or config.max_running_req
    override = getattr(config, "linear_state_slots_override", None)
    if override is not None:
        minimum = mr + 1 if config.cache_type != "hybrid_radix" else 4 * mr + 1
        if override < minimum:
            raise ValueError(
                f"--linear-state-slots {override} is below the {minimum}-slot "
                f"working-set floor for max_running_requests={mr} and "
                f"cache_type={config.cache_type!r}"
            )
        return int(override)
    if config.cache_type != "hybrid_radix":
        return mr + 1  # live + dummy/padding
    ratio = config.linear_state_cache_ratio
    n_cache = max(4, int(ratio * mr))
    return 4 * mr + n_cache + 1  # live + 2 ping-pong + locked committed snapshot + cache + padding


def linear_pool_slots_for_capacity(config, capacity: int) -> int:
    """Default physical slot geometry for an elastic request capacity."""
    if config.cache_type != "hybrid_radix":
        return capacity + 1
    n_cache = max(4, int(config.linear_state_cache_ratio * capacity))
    return 4 * capacity + n_cache + 1


def _linear_pool_min_slots(config) -> int:
    """Floor on LinearStatePool slots that still runs: the non-evictable working set with a
    zero snapshot cache. Hybrid-radix needs 4 per running request (1 live + 2 ping-pong + 1
    committed snapshot locked through decode) + the padding sink; naive needs 1 per request +
    padding. Below this, a full max_running_req batch can't get its slots and admission
    deadlocks -- so a runtime rebuild rejects a smaller request."""
    mr = config.max_running_req
    if config.cache_type != "hybrid_radix":
        return mr + 1
    return 4 * mr + 1
