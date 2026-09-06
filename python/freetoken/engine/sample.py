from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
from freetoken.utils import is_sm90_supported, nvtx_annotate

if TYPE_CHECKING:
    from freetoken.core import Batch


@dataclass
class FirstStepLogprobs:
    """One prefill batch's first-step logprobs, in flight to the host.

    ``rows[j]`` is the batch row the j-th entry belongs to; ``ks[j]`` its ``top_logprobs``.
    ``token_ids`` / ``logprobs`` are host tensors ``[len(rows), 1 + max(ks)]``: column 0 is
    the sampled token, columns 1..k the top-k of the same log_softmax (a row with a
    smaller k ignores its tail). Valid once the batch's copy_done event has fired --
    ``row(j)`` reads them, on the drain path, into plain python lists.
    """

    rows: List[int]
    ks: List[int]
    token_ids: torch.Tensor
    logprobs: torch.Tensor

    def row(self, j: int) -> dict:
        k = self.ks[j]
        return {
            "token_ids": self.token_ids[j, : 1 + k].tolist(),
            "logprobs": self.logprobs[j, : 1 + k].tolist(),
        }


def first_step_logprobs(
    logits: torch.Tensor, sampled: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """log_softmax over the full vocabulary of ``logits`` ``[n, vocab]`` (float32, the
    raw pre-temperature logits) for the sampled ids ``[n]`` and the ``k`` most likely
    tokens: ``(token_ids [n, 1 + k], logprobs [n, 1 + k])``, column 0 the sampled token.
    Stays on ``logits.device``; the caller makes the one small copy."""
    lp = torch.log_softmax(logits.float(), dim=-1)
    sampled = sampled.to(torch.int64).view(-1, 1)
    ids, vals = [sampled], [lp.gather(1, sampled)]
    if k > 0:
        top_vals, top_ids = torch.topk(lp, k, dim=-1)
        ids.append(top_ids)
        vals.append(top_vals)
    return torch.cat(ids, dim=1), torch.cat(vals, dim=1)


@dataclass
class BatchSamplingArgs:
    temperatures: torch.Tensor | None
    top_k: torch.Tensor | None = None
    top_p: torch.Tensor | None = None
    # Host-known upper bound lets the logits-domain sampler call torch.topk without a
    # device-to-host synchronization. None means top-k filtering is disabled.
    max_top_k: int | None = None


def make_device_tensor(data: List, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(data, dtype=dtype, pin_memory=True).to(device, non_blocking=True)


def sample_impl(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor | int | None,
    top_p: torch.Tensor | float | None,
    max_top_k: int | None = None,
) -> torch.Tensor:
    from freetoken.kernel.backend import is_flashinfer_installed

    # softmax is monotonic, so selecting top-k logits first and normalizing only those
    # candidates is mathematically identical to full-vocabulary softmax -> top-k ->
    # renormalize.  Ornith's 248K vocabulary makes avoiding the full probability tensor
    # and its repeated scans worthwhile. Keep very large k on the general implementation.
    if (
        not is_flashinfer_installed()
        and top_k is not None
        and max_top_k is not None
        and max_top_k <= 1024
    ):
        from freetoken.kernel.triton.sampling import top_k_top_p_sampling_from_logits

        return top_k_top_p_sampling_from_logits(
            logits, temperatures, top_k, top_p, max_top_k=max_top_k
        )

    if is_flashinfer_installed():
        import flashinfer.sampling as sampling
    else:
        import freetoken.kernel.triton.sampling as sampling

    probs = sampling.softmax(logits, temperatures, enable_pdl=is_sm90_supported())
    if top_k is None and top_p is None:
        return sampling.sampling_from_probs(probs)

    if top_p is None:
        assert top_k is not None
        return sampling.top_k_sampling_from_probs(probs, top_k)

    if top_k is None:
        assert top_p is not None
        return sampling.top_p_sampling_from_probs(probs, top_p)

    assert top_k is not None and top_p is not None
    return sampling.top_k_top_p_sampling_from_probs(probs, top_k, top_p)


@dataclass
class Sampler:
    device: torch.device
    vocab_size: int

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        params = [r.sampling_params for r in batch.reqs]
        if all(p.is_greedy for p in params):
            return BatchSamplingArgs(temperatures=None)

        MIN_P = MIN_T = 1e-6
        ts = [max(0.0 if p.is_greedy else p.temperature, MIN_T) for p in params]
        top_ks = [p.top_k if p.top_k >= 1 else self.vocab_size for p in params]
        top_ps = [min(max(p.top_p, MIN_P), 1.0) for p in params]
        temperatures = make_device_tensor(ts, torch.float32, self.device)
        top_k, top_p = None, None
        if any(k != self.vocab_size for k in top_ks):
            top_k = make_device_tensor(top_ks, torch.int32, self.device)
        if any(p < 1.0 for p in top_ps):
            top_p = make_device_tensor(top_ps, torch.float32, self.device)
        max_top_k = max(top_ks) if top_k is not None else None
        return BatchSamplingArgs(
            temperatures, top_k=top_k, top_p=top_p, max_top_k=max_top_k
        )

    def first_step_logprobs(
        self, batch: Batch, logits: torch.Tensor, sampled: torch.Tensor
    ) -> FirstStepLogprobs | None:
        """The first-step logprobs a prefill batch's plain (final-chunk) requests asked
        for, as host tensors in flight on the current stream; None when nobody did or
        for a decode batch. A ChunkedReq row is a continuation, not a first step."""
        from freetoken.scheduler.prefill import ChunkedReq

        if not batch.is_prefill:
            return None
        rows = [
            i for i, r in enumerate(batch.reqs)
            if r.sampling_params.logprobs and not isinstance(r, ChunkedReq)
        ]
        if not rows:
            return None
        ks = [batch.reqs[i].sampling_params.top_logprobs for i in rows]
        ids, vals = first_step_logprobs(logits[rows], sampled[rows], max(ks))
        return FirstStepLogprobs(
            rows, ks, ids.to("cpu", non_blocking=True), vals.to("cpu", non_blocking=True)
        )

    @nvtx_annotate("Sampler")
    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        with torch.cuda.nvtx.range("Sampler"):
            if args.temperatures is None:  # greedy sampling
                return torch.argmax(logits, dim=-1)
            return sample_impl(
                logits.float(), args.temperatures, args.top_k, args.top_p, args.max_top_k
            )
