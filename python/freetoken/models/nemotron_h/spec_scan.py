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
3. once the accepted count is known, :meth:`SpecScanCapture.commit` replays one varlen SSD
   scan per layer over the first ``n`` recorded positions, starting from the **live** slot
   and writing back into it, and slides the conv window by the same ``n`` tokens.

A full acceptance skips the replay entirely: the scratch slot already holds exactly the
state after all ``m`` positions, so it is copied back wholesale.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, NamedTuple

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


class SpecScanCapture:
    """Records the per-layer Mamba-2 scan inputs of one speculative verify forward."""

    def __init__(self, num_tokens: int) -> None:
        self.num_tokens = num_tokens
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
