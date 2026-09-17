"""GDN (gated-delta-rule) recurrent-state handling for a speculative verify forward.

This is the ``state_layout == "kv"`` twin of
``models/nemotron_h/spec_scan.py`` -- same contract, different kernel. A verify step
forwards ``m = k + 1`` tokens with ``fla.cache_indices`` pointed at a scratch slot
(the scheduler has already copied the live slot there), so the live recurrent and conv
state are never advanced speculatively. Once the accepted prefix length ``n`` is known,
:meth:`GdnSpecScanCapture.commit` advances the LIVE slot by exactly ``n`` tokens.

The GDN prefill kernel (``gdn_prefill_chunk_fla`` -> ``chunk_gated_delta_rule``) reads
``pool.recurrent_states[li][slot]`` as its initial state and writes the final state back
to the same slot IN PLACE, so the commit is a single re-run over the recorded
``q, k, v, g, beta`` prefix with ``state_source`` = the live slot. The conv window
(``pool.conv_states[li][slot]``, the last ``kernel-1`` raw conv inputs) is slid by ``n``
tokens over the recorded ``conv_in``.

One scan for every layer
------------------------
All GDN layers share the same head geometry and the fla kernel handles GQA in-kernel
(q/k at ``num_k_heads``, v/g/beta at ``num_v_heads``), so the layer axis folds onto the
K-head axis exactly as the mamba2 fused commit folds it onto the SSD head axis:
``L`` layers x ``HK`` k-heads is one ``(1, n, L*HK, K)`` sequence and ``L`` x ``HV``
v-heads is ``(1, n, L*HV, V)``. The per-layer ``A_log`` / ``dt_bias`` gate parameters
concatenate the same way. The fused path runs ONE chunk scan + ONE conv slide for the
whole model; :meth:`_commit_per_layer` is the kept reference it is checked against.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, NamedTuple

import torch

if TYPE_CHECKING:
    from freetoken.kvcache.linear_state_pool import LinearStatePool


class _LayerScan(NamedTuple):
    gdn: object                  # Qwen3_5GatedDeltaNet
    q: torch.Tensor              # [1, m, HK, K] bf16, post-conv, pre-l2norm
    k: torch.Tensor              # [1, m, HK, K] bf16
    v: torch.Tensor              # [1, m, HV, V] bf16
    g: torch.Tensor              # [1, m, HV] fp32 log-decay
    beta: torch.Tensor           # [1, m, HV] fp32
    conv_in: torch.Tensor        # [m, conv_dim] raw pre-conv projection stream


class _FusedPlan(NamedTuple):
    """Concatenated per-head gate params + the live-slot pool indices for one model."""

    A_log: torch.Tensor          # [L*HV] fp32
    dt_bias: torch.Tensor        # [L*HV] fp32
    rec_index: torch.Tensor      # [L] int64 into recurrent_states flattened [L*slots,...]
    conv_index: torch.Tensor     # [L] int64 into conv_states flattened [L*slots,...]


# (rec data_ptr, conv data_ptr, num_slots, live slot, layer ids, A_log storage id) -> plan.
# Keyed on the data pointer so an elastic pool rebuild (which reallocates both tensors)
# cannot hand back an index vector built for the old geometry.
_PLAN_CACHE: Dict[tuple, _FusedPlan] = {}


def _plan(pool: "LinearStatePool", layers: List[_LayerScan], live_slot: int) -> _FusedPlan | None:
    rs, cs = pool.recurrent_states, pool.conv_states
    first = layers[0].gdn
    key = (
        rs.data_ptr(), cs.data_ptr(), rs.shape[1], live_slot, id(first),
        first.A_log.data_ptr(), tuple(rec.gdn.layer_id for rec in layers),
    )
    plan = _PLAN_CACHE.get(key)
    if plan is not None:
        return plan
    for rec in layers:
        g = rec.gdn
        if (
            g.num_k_heads != first.num_k_heads
            or g.num_v_heads != first.num_v_heads
            or g.head_k_dim != first.head_k_dim
            or g.head_v_dim != first.head_v_dim
        ):
            return None  # non-uniform GDN geometry: the layer axis is not a valid head axis
    local = [pool.local_index(rec.gdn.layer_id) for rec in layers]
    device = rs.device
    plan = _FusedPlan(
        A_log=torch.cat([rec.gdn.A_log.reshape(-1) for rec in layers]).contiguous(),
        dt_bias=torch.cat([rec.gdn.dt_bias.reshape(-1) for rec in layers]).contiguous(),
        rec_index=torch.tensor(
            [li * rs.shape[1] + live_slot for li in local], dtype=torch.int64, device=device
        ),
        conv_index=torch.tensor(
            [li * cs.shape[1] + live_slot for li in local], dtype=torch.int64, device=device
        ),
    )
    if len(_PLAN_CACHE) > 64:
        _PLAN_CACHE.clear()
    _PLAN_CACHE[key] = plan
    return plan


class GdnSpecScanCapture:
    """Records the per-layer GDN scan inputs of one speculative verify forward."""

    def __init__(self, num_tokens: int, *, fused: bool = True) -> None:
        self.num_tokens = num_tokens
        self.fused = fused
        self.layers: List[_LayerScan] = []

    def record(self, gdn, q, k, v, g, beta, conv_in) -> None:
        assert q.shape[1] == self.num_tokens, (q.shape, self.num_tokens)
        self.layers.append(_LayerScan(gdn, q, k, v, g, beta, conv_in))

    # ------------------------------------------------------------------ commit

    def commit(
        self,
        pool: "LinearStatePool",
        live_slot: int,
        scratch_slot: int,
        n: int,
        *,
        force_replay: bool = False,
    ) -> None:
        """Advance the live state slot by the first ``n`` verify positions.

        ``n`` is ``accepted + 1`` (the tokens the sampler kept), ``1 <= n <= m``. At
        ``n == m`` the scratch slot already holds the answer, so it is copied back
        wholesale rather than recomputed -- ``force_replay`` disables that shortcut for
        :meth:`replay_error`.
        """
        assert 1 <= n <= self.num_tokens, (n, self.num_tokens)
        if not self.layers:
            return
        if n == self.num_tokens and not force_replay:
            pool.copy_from(scratch_slot, live_slot)
            return
        plan = _plan(pool, self.layers, live_slot) if self.fused else None
        if plan is None:
            self._commit_per_layer(pool, live_slot, n)
        else:
            self._commit_fused(pool, plan, n)

    # -- one scan for the whole model ------------------------------------------

    def _commit_fused(self, pool: "LinearStatePool", plan: _FusedPlan, n: int) -> None:
        from freetoken.models.qwen3_5_moe.gdn_kernels import gdn_prefill_chunk_fla

        layers = self.layers
        rs, cs = pool.recurrent_states, pool.conv_states
        first = layers[0].gdn
        HK, HV = first.num_k_heads, first.num_v_heads
        device = rs.device

        # Fold the layer axis onto the head axis: [1, n, L*HK, K] / [1, n, L*HV, ...].
        q = torch.cat([rec.q[:, :n] for rec in layers], dim=2)
        k = torch.cat([rec.k[:, :n] for rec in layers], dim=2)
        v = torch.cat([rec.v[:, :n] for rec in layers], dim=2)
        g = torch.cat([rec.g[:, :n] for rec in layers], dim=2)
        beta = torch.cat([rec.beta[:, :n] for rec in layers], dim=2)

        # One fused state view: [L, slots, HV, K, V] flattened over (L*slots) so the live
        # slot of every layer is addressable as one row, then one [1, L*HV, K, V] "pool".
        nl, num_slots, _, K, V = rs.shape
        rs_flat = rs.view(nl * num_slots, HV, K, V)
        fused_pool = rs_flat.index_select(0, plan.rec_index).reshape(1, nl * HV, K, V)
        # The chunk kernel writes its final state back into ``state_source[indices]``; the
        # index_select above is a COPY, so run the scan into a standalone buffer and
        # scatter it back. (The kernel's in-place write needs a real pool row.)
        scratch_state = fused_pool.clone()
        indices = torch.zeros(1, dtype=torch.int32, device=device)
        cu_seqlens = torch.tensor([0, n], dtype=torch.int64, device=device)
        gdn_prefill_chunk_fla(
            q, k, v, g, beta,
            state_source=scratch_state, indices=indices,
            cu_seqlens=cu_seqlens, scale=first.head_k_dim ** -0.5,
        )
        # scratch_state now holds the post-n state for every layer; scatter back to live.
        rs_flat.index_copy_(0, plan.rec_index, scratch_state.view(nl, HV, K, V).to(rs.dtype))
        self._commit_conv_fused(cs, plan.conv_index, n)

    def _commit_conv_fused(self, cs: torch.Tensor, index: torch.Tensor, n: int) -> None:
        """Slide every layer's conv window by ``n`` tokens in one gather/cat/scatter."""
        nl, num_slots, conv_dim, km1 = cs.shape
        cs_flat = cs.view(nl * num_slots, conv_dim, km1)
        # [L, n, conv_dim] -> [L, conv_dim, n]
        tail = torch.stack([rec.conv_in[:n] for rec in self.layers], dim=0)
        tail = tail.transpose(1, 2).to(cs.dtype)
        if n >= km1:
            new = tail[:, :, -km1:]
        else:
            win = cs_flat.index_select(0, index)
            new = torch.cat([win[:, :, n:], tail], dim=-1)
        cs_flat.index_copy_(0, index, new.contiguous())

    # -- the per-layer reference path ------------------------------------------

    def _commit_per_layer(self, pool: "LinearStatePool", live_slot: int, n: int) -> None:
        from freetoken.models.qwen3_5_moe.gdn_kernels import gdn_prefill_chunk_fla

        device = pool.recurrent_states.device
        indices = torch.tensor([live_slot], dtype=torch.int32, device=device)
        cu_seqlens = torch.tensor([0, n], dtype=torch.int64, device=device)
        for rec in self.layers:
            li = pool.local_index(rec.gdn.layer_id)
            self._commit_conv(pool, li, live_slot, rec.conv_in, n)
            gdn_prefill_chunk_fla(
                rec.q[:, :n].contiguous(),
                rec.k[:, :n].contiguous(),
                rec.v[:, :n].contiguous(),
                rec.g[:, :n].contiguous(),
                rec.beta[:, :n].contiguous(),
                state_source=pool.recurrent_states[li], indices=indices,
                cu_seqlens=cu_seqlens, scale=rec.gdn.head_k_dim ** -0.5,
            )

    @staticmethod
    def _commit_conv(pool, li: int, live_slot: int, conv_in: torch.Tensor, n: int) -> None:
        """Slide the live conv window by ``n`` tokens of ``conv_in`` ([m, conv_dim])."""
        win = pool.conv_states[li, live_slot]          # [conv_dim, kernel-1], a view
        km1 = win.shape[-1]
        tail = conv_in[:n].transpose(0, 1).to(win.dtype)
        if n >= km1:
            win.copy_(tail[:, -km1:])
        else:
            win.copy_(torch.cat([win[:, n:], tail], dim=-1))

    # ------------------------------------------------------------------ self-check

    def replay_error(
        self, pool: "LinearStatePool", live_slot: int, scratch_slot: int, spare_slot: int
    ) -> tuple[float, float]:
        """Max abs (recurrent, conv) disagreement between the replay and the forward.

        Same role as the mamba2 self-check: replay all ``m`` positions from the live slot
        into a spare slot (force_replay) and compare against what the verify forward left in
        the scratch slot. At ``n == m`` the two must agree to float noise.
        """
        pool.copy_from(live_slot, spare_slot)
        self.commit(pool, spare_slot, scratch_slot, self.num_tokens, force_replay=True)
        rec = (
            (pool.recurrent_states[:, spare_slot] - pool.recurrent_states[:, scratch_slot])
            .abs().max().item()
        )
        conv = (
            (pool.conv_states[:, spare_slot].float() - pool.conv_states[:, scratch_slot].float())
            .abs().max().item()
        )
        return rec, conv


__all__ = ["GdnSpecScanCapture"]
