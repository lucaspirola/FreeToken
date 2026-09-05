"""Mamba-2 recurrent-state handling for a speculative verify forward.

A verify step forwards ``m = k + 1`` tokens (the last committed token plus ``k`` drafted
ones) and then keeps only the accepted prefix. Attention needs no rollback -- rejected
positions simply keep their KV pages and are overwritten -- but the Mamba-2 recurrent and
conv state is a running summary: advancing it over a rejected token is not recoverable by
truncation.

The design (benchmarks/results/nemotron35_lightning_5080_ngram_spec_2026-09-05.md §5) is
to **never advance the live state speculatively**:

1. the scheduler copies the live slot into a private scratch slot and points the verify
   forward's ``fla.cache_indices`` at the scratch slot, so ``mamba2_prefill``'s scatter and
   ``causal_conv1d_varlen``'s conv write both land there;
2. every Mamba-2 mixer records its own scan inputs for the ``m`` verify positions into a
   :class:`SpecScanCapture` (``x``, ``dt``, ``B``, ``C`` and the raw conv input -- ~25 KiB
   per layer per token, so ~5 MiB at m = 9, against the 46.8 MiB a per-position state block
   would cost);
3. once the accepted count is known, :meth:`SpecScanCapture.commit` replays the SSD scan
   over the first ``n`` recorded positions, starting from the **live** slot and writing back
   into it, and slides the conv window by the same ``n`` tokens.

A full acceptance skips the replay entirely: the scratch slot already holds exactly the
state after all ``m`` positions, so it is copied back wholesale.

One scan for every layer (2026-09-05)
-------------------------------------
The first shipped commit ran step 3 **per layer**: 23 ``mamba2_prefill`` calls and 23 conv
window writes, each with its own metadata tensors, state gather and state scatter -- ~280
kernel launches for ~9 tokens of arithmetic, which measured as ~40 % of a verify step.

Mamba-2 heads are independent, and every Nemotron-H mixer has the same
``(head_dim, state_size, heads_per_group)``, so **the layer axis concatenates onto the head
axis**: 23 layers x 64 heads is one 1 472-head sequence, and 23 x 8 groups is 184 groups
with the same 8-heads-per-group mapping the kernel already assumes. ``A`` and ``dt_bias``
are per-head, so they concatenate too; ``D`` only feeds the scan *output*, which the commit
discards, so it is dropped entirely. What is left is **one** varlen SSD scan and **one**
gather/scatter pair for the whole model (:meth:`_commit_fused`), and the same trick folds
the 23 conv-window writes into one masked cat.

The per-layer path is kept as :meth:`_commit_per_layer` and is what runs when a model's
mixers are not uniform -- and it is the reference the fused path is checked against.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, NamedTuple, Tuple

import torch

if TYPE_CHECKING:
    from freetoken.kvcache.linear_state_pool import LinearStatePool


class _LayerScan(NamedTuple):
    mixer: object              # NemotronHMamba2Mixer
    x: torch.Tensor            # [m, H, P] post-conv SSM input
    dt: torch.Tensor           # [m, H] raw timestep (pre-bias, pre-softplus)
    B: torch.Tensor            # [m, G, N]
    C: torch.Tensor            # [m, G, N]
    conv_in: torch.Tensor      # [m, conv_dim] pre-conv projection stream


class _FusedPlan(NamedTuple):
    """Everything the fused commit needs that depends only on the model + the slot.

    Built once per (pool geometry, layer set, live slot) and cached: the concatenated
    per-head ``A`` / ``dt_bias`` are model constants, and the two index vectors address
    ``recurrent_states`` / ``conv_states`` flattened to ``[layers * slots, ...]``.
    """

    A: torch.Tensor                    # [L*H] fp32
    dt_bias: torch.Tensor | None       # [L*H] fp32
    dt_limit: Tuple[float, float]
    rec_index: torch.Tensor            # [L] int64 into recurrent_states.view(L*slots, ...)
    conv_index: torch.Tensor           # [L] int64 into conv_states.view(L*slots, ...)


# (recurrent data_ptr, num_slots, layer ids, live slot) -> plan. Keyed on the data pointer
# so an elastic pool rebuild (which reallocates both state tensors) cannot hand back an
# index vector built for the old geometry.
_PLAN_CACHE: Dict[tuple, _FusedPlan] = {}
# (device, chunk_size, n) -> (metadata, cu_seqlens). The chunk plan for a one-sequence
# batch of n tokens is a pure function of n, and a verify step visits at most k + 1 of them.
_META_CACHE: Dict[tuple, tuple] = {}


def _plan(pool: "LinearStatePool", layers: List[_LayerScan], live_slot: int) -> _FusedPlan | None:
    """The fused-commit plan for this layer set, or None if the mixers are not uniform."""
    rs, cs = pool.recurrent_states, pool.conv_states
    num_slots = rs.shape[1]
    first_mixer = layers[0].mixer
    # ``A`` is a cached ``-exp(A_log)`` on the module, so its storage identity is what says
    # "these are still the same weights"; a (re)load_state_dict rebuilds it and must not be
    # served a plan holding the old concatenation. The layer set and the pool geometry
    # complete the key.
    key = (rs.data_ptr(), cs.data_ptr(), num_slots, live_slot, id(first_mixer),
           first_mixer.A.data_ptr(), tuple(rec.mixer.layer_id for rec in layers))
    plan = _PLAN_CACHE.get(key)
    if plan is not None:
        return plan
    first = first_mixer
    heads_per_group = first.num_heads // first.n_groups
    for rec in layers:
        mx = rec.mixer
        if (
            mx.head_dim != first.head_dim
            or mx.state_size != first.state_size
            or mx.num_heads // mx.n_groups != heads_per_group
            or tuple(mx.dt_limit) != tuple(first.dt_limit)
            or (mx.dt_bias is None) != (first.dt_bias is None)
        ):
            return None  # non-uniform mixers: the head axis is not a valid layer axis
    local = [pool.local_index(rec.mixer.layer_id) for rec in layers]
    device = rs.device
    plan = _FusedPlan(
        A=torch.cat([rec.mixer.A.reshape(-1) for rec in layers]).contiguous(),
        dt_bias=(
            torch.cat([rec.mixer.dt_bias.reshape(-1) for rec in layers]).contiguous()
            if first.dt_bias is not None
            else None
        ),
        dt_limit=tuple(first.dt_limit),
        rec_index=torch.tensor(
            [li * num_slots + live_slot for li in local], dtype=torch.int64, device=device
        ),
        conv_index=torch.tensor(
            [li * cs.shape[1] + live_slot for li in local], dtype=torch.int64, device=device
        ),
    )
    if len(_PLAN_CACHE) > 64:  # bounded: one entry per (geometry, slot) actually used
        _PLAN_CACHE.clear()
    _PLAN_CACHE[key] = plan
    return plan


def _chunk_plan(device: torch.device, chunk_size: int, n: int):
    key = (str(device), chunk_size, n)
    got = _META_CACHE.get(key)
    if got is None:
        from freetoken.kernel.triton.mamba2 import build_mamba2_metadata

        meta = build_mamba2_metadata([0, n], chunk_size=chunk_size, device=device)
        cu = torch.tensor([0, n], dtype=torch.int32, device=device)
        got = (meta, cu, meta.last_chunk_indices.to(torch.long))
        _META_CACHE[key] = got
    return got


class SpecScanCapture:
    """Records the per-layer Mamba-2 scan inputs of one speculative verify forward."""

    def __init__(self, num_tokens: int, *, fused: bool = True) -> None:
        self.num_tokens = num_tokens
        self.fused = fused
        self.layers: List[_LayerScan] = []

    def record(self, mixer, x, dt, B, C, conv_in) -> None:
        assert x.shape[0] == self.num_tokens, (x.shape, self.num_tokens)
        self.layers.append(_LayerScan(mixer, x, dt, B, C, conv_in))

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

        ``n`` is ``accepted + 1`` (the tokens the sampler actually kept), ``1 <= n <= m``.
        At ``n == m`` the scratch slot is already the answer, so it is copied back rather
        than recomputed -- ``force_replay`` disables that shortcut, which is what
        :meth:`replay_error` uses to check the replay against the forward.
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
        from freetoken.kernel.triton.mamba2 import mamba_chunk_scan_combined_varlen

        layers = self.layers
        rs, cs = pool.recurrent_states, pool.conv_states
        nl, num_slots, heads, headdim, dstate = rs.shape
        rs_flat = rs.view(nl * num_slots, heads, headdim, dstate)

        # [n, L*H, P] / [n, L*H] / [n, L*G, N]: the layer axis folded onto the head axis.
        # ``cat`` over 23 slices is one kernel; the old path issued four per layer.
        x = torch.cat([rec.x[:n] for rec in layers], dim=1)
        dt = torch.cat([rec.dt[:n] for rec in layers], dim=1)
        B = torch.cat([rec.B[:n] for rec in layers], dim=1)
        C = torch.cat([rec.C[:n] for rec in layers], dim=1)

        meta, cu_seqlens, last_long = _chunk_plan(rs.device, pool.track_chunk_size, n)
        initial = rs_flat.index_select(0, plan.rec_index).reshape(
            1, len(layers) * heads, headdim, dstate
        )
        if initial.dtype != torch.float32:
            initial = initial.to(torch.float32)
        all_states = mamba_chunk_scan_combined_varlen(
            x,
            dt,
            plan.A,
            B,
            C,
            meta.chunk_size,
            cu_seqlens,
            meta.cu_chunk_seqlens,
            meta.last_chunk_indices,
            meta.seq_idx,
            torch.empty_like(x),
            D=None,  # the skip connection feeds the scan OUTPUT only, which is discarded
            z=None,
            dt_bias=plan.dt_bias,
            initial_states=initial,
            dt_softplus=True,
            dt_limit=plan.dt_limit,
            return_intermediate_states=True,
            state_dtype=torch.float32,
        )
        final = all_states.index_select(0, last_long)
        rs_flat.index_copy_(
            0, plan.rec_index, final.reshape(len(layers), heads, headdim, dstate).to(rs.dtype)
        )
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
        from freetoken.kernel.triton.mamba2 import build_mamba2_metadata, mamba2_prefill

        device = pool.recurrent_states.device
        meta = build_mamba2_metadata([0, n], chunk_size=pool.track_chunk_size, device=device)
        cu_seqlens = torch.tensor([0, n], dtype=torch.int32, device=device)
        indices = torch.tensor([live_slot], dtype=torch.int32, device=device)
        has_initial_state = torch.ones(1, dtype=torch.bool, device=device)

        for rec in self.layers:
            li = pool.local_index(rec.mixer.layer_id)
            self._commit_conv(pool, li, live_slot, rec.conv_in, n)
            mamba2_prefill(
                rec.x[:n].contiguous(),
                rec.dt[:n].contiguous(),
                rec.B[:n].contiguous(),
                rec.C[:n].contiguous(),
                A=rec.mixer.A,
                D=rec.mixer.D,
                dt_bias=rec.mixer.dt_bias,
                meta=meta,
                cu_seqlens=cu_seqlens,
                state_source=pool.recurrent_states[li],
                indices=indices,
                has_initial_state=has_initial_state,
                dt_softplus=True,
                dt_limit=rec.mixer.dt_limit,
            )

    @staticmethod
    def _commit_conv(pool, li: int, live_slot: int, conv_in: torch.Tensor, n: int) -> None:
        """Slide the live conv window by ``n`` tokens of ``conv_in`` ([m, conv_dim]).

        The window is the last ``kernel - 1`` timesteps of the conv input stream, so the
        committed window is the tail of ``old_window ++ conv_in[:n]``. The live slot was
        never written by the verify forward (it wrote the scratch slot), so ``old_window``
        here is still the pre-verify one.
        """
        win = pool.conv_states[li, live_slot]          # [conv_dim, kernel-1], a view
        km1 = win.shape[-1]
        tail = conv_in[:n].transpose(0, 1).to(win.dtype)
        if n >= km1:
            win.copy_(tail[:, -km1:])
        else:
            # torch.cat materializes before the copy, so reading `win` here is safe.
            win.copy_(torch.cat([win[:, n:], tail], dim=-1))

    # ------------------------------------------------------------------ self-check

    def replay_error(
        self, pool: "LinearStatePool", live_slot: int, scratch_slot: int, spare_slot: int
    ) -> tuple[float, float]:
        """Max abs (recurrent, conv) disagreement between the replay and the forward.

        The commit path recomputes a state the verify forward already computed once, from
        the same initial state, over the same tokens, with the same kernels -- so at
        ``n == m`` the two must agree. Anything but float noise here is a bug in the
        capture (wrong tensor, wrong order, wrong conv window), and it is the ONLY way to
        separate that from the ordinary extend-vs-decode kernel disagreement that a
        greedy-equivalence diff cannot attribute. Debug-only: costs a slot and a replay.

        It is also the gate on the fused commit, and a strong one: the verify forward ran
        the scan **per layer**, so a fused replay that agrees with it to 0 has proved that
        folding the layer axis onto the head axis changed nothing.
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


__all__ = ["SpecScanCapture"]
