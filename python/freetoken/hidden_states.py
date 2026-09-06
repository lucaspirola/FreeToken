"""Prompt hidden-state export -- the probe target of Switchyard's prefill router.

Switchyard's complexity router (``prefill_probe/scorer.rs``) asks a probe server for
one ``max_tokens: 1`` chat completion carrying top-level ``kv_transfer_params``, then
reads a ``.safetensors`` artifact off shared storage and mean-pools it into the feature
vector its learned head consumes. The artifact contract, copied from vLLM's
``ExampleHiddenStatesConnector`` (docs/vllm-serve-hidden-state.md) and pinned by that
reader:

    hidden_states  [prompt_tokens, layers, hidden]  BF16 (F32 also accepted)
    token_ids      [prompt_tokens]                  I64, optional but validated

``layers`` is the *raw post-decoder-layer residual stream* -- the value each block
leaves behind after adding its mixer output, NOT the final-norm output and not the
logits. vLLM names the captured set ``eagle_aux_hidden_state_layer_ids`` and its loader
requires them contiguous from 0 and ascending, so this module accepts nothing else.

For Nemotron-H every block is a "layer" here regardless of what it mixes: the
Nemotron-3.5-Lightning stack interleaves 23 mamba, 23 MoE and 6 attention blocks into
one 52-deep residual stream, and the router wants the stream, not a per-kind subset.

Capture is per prefill chunk. A probe request bypasses prefix reuse (so every prompt
token is actually forwarded) and its chunks are concatenated in forward order, which is
also token order. The whole path is opt-in: without ``Req.hidden_states`` no sink is
installed and a forward pays one attribute read.

``kv_transfer_params.pooling`` is the inline variant of the same capture: instead of
(or as well as) the file, the response carries per-layer ``mean`` and/or ``last``
vectors over the prompt positions, base64 float32, under ``kv_transfer_params.pooled``.
That path keeps only a running float32 sum and the last row per layer -- O(layers x
hidden) on the host -- so it has no token cap, needs no ``--hidden-states-dir``, and
accepts any ascending subset of layers (nothing indexes them positionally).
"""

from __future__ import annotations

import base64
import fcntl
import os
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

if TYPE_CHECKING:
    from freetoken.core import Batch

__all__ = [
    "DEFAULT_MAX_TOKENS",
    "POOLINGS",
    "HiddenStateCapture",
    "HiddenStateCollector",
    "HiddenStateSink",
    "HiddenStateSpec",
    "resolve_hidden_states_dir",
    "validate_layer_ids",
    "write_hidden_states",
]

#: Per-request prompt-token cap (``--hidden-states-max-tokens``). One 4096-token probe
#: over 52 layers at hidden 2688 is already ~1.1 GiB of BF16 on the wire and in the
#: reader's memory; the router's own probe prompts are two orders of magnitude shorter.
DEFAULT_MAX_TOKENS = 4096

#: The reader refuses anything else (``has_safetensors_extension``).
ARTIFACT_SUFFIX = ".safetensors"

_HIDDEN_STATES_KEY = "hidden_states"
_TOKEN_IDS_KEY = "token_ids"

#: ``kv_transfer_params.pooling`` values -> the ``pooled`` keys each one returns.
POOLINGS: dict[str, tuple[str, ...]] = {
    "mean": ("mean",),
    "last": ("last",),
    "both": ("mean", "last"),
}


@dataclass
class HiddenStateSpec:
    """One request's opt-in capture: where to write, which blocks to keep, what to pool.

    ``directory`` is already resolved against the server's ``--hidden-states-dir`` root
    (see :func:`resolve_hidden_states_dir`); nothing downstream re-derives it from
    client input. None means no artifact is written. ``layer_ids`` is ascending and
    unique, and contiguous from 0 whenever a file is written. ``pooling`` is the set of
    inline vectors to return (a ``POOLINGS`` value); empty for the file-only probe.
    """

    directory: str | None = None
    layer_ids: list[int] = field(default_factory=list)
    # A ``POOLINGS`` value; a list after the msgpack hop, so only ``in`` is used on it.
    pooling: Sequence[str] = ()


def validate_layer_ids(
    layer_ids: object, num_layers: int | None = None, *, contiguous: bool = True
) -> list[int]:
    """Normalize a client's ``layer_ids`` or raise ``ValueError``.

    Switchyard's artifact loader indexes the middle axis positionally against its
    checkpoint's ``layer_count``, so a non-contiguous or unsorted set would silently
    mislabel features rather than fail. Reject it here instead. The pooled response
    names its layers (``pooled.layer_ids``), so ``contiguous=False`` relaxes that to any
    ascending, unique subset.
    """
    if not isinstance(layer_ids, (list, tuple)) or isinstance(layer_ids, (str, bytes)):
        raise ValueError("kv_transfer_params.layer_ids must be a list of integers")
    ids: list[int] = []
    for value in layer_ids:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("kv_transfer_params.layer_ids must be a list of integers")
        ids.append(int(value))
    if not ids:
        raise ValueError("kv_transfer_params.layer_ids must not be empty")
    if contiguous and ids != list(range(len(ids))):
        raise ValueError(
            "kv_transfer_params.layer_ids must be contiguous from 0 and ascending "
            f"(got {ids!r}); Switchyard's artifact loader indexes them positionally"
        )
    if ids != sorted(set(ids)) or ids[0] < 0:
        raise ValueError(
            "kv_transfer_params.layer_ids must be ascending, unique and non-negative "
            f"(got {ids!r})"
        )
    if num_layers is not None and ids[-1] >= num_layers:
        raise ValueError(
            f"kv_transfer_params.layer_ids asks for layer {ids[-1]} but this model "
            f"has {num_layers}"
        )
    return ids


def resolve_hidden_states_dir(requested: str | None, root: str | None) -> str:
    """Canonicalize the client's target directory and refuse anything outside ``root``.

    ``root`` is the server's ``--hidden-states-dir``; without it the feature is off and
    every probe request is an error. The client may name ``root`` itself or a
    subdirectory of it -- ``..`` and symlinks are resolved before the containment test,
    which is the same check the reader repeats on its side.
    """
    if not root:
        raise ValueError(
            "hidden-state export is disabled; start the server with --hidden-states-dir"
        )
    root_path = os.path.realpath(root)
    if not os.path.isdir(root_path):
        raise ValueError(f"--hidden-states-dir {root!r} is not a directory")
    if requested is None or requested == "":
        return root_path
    if not isinstance(requested, str):
        raise ValueError("kv_transfer_params.hidden_states_path must be a string")
    target = os.path.realpath(
        requested if os.path.isabs(requested) else os.path.join(root_path, requested)
    )
    if target != root_path and not target.startswith(root_path + os.sep):
        raise ValueError(
            f"kv_transfer_params.hidden_states_path {requested!r} is outside the "
            f"server's --hidden-states-dir {root_path!r}"
        )
    if not os.path.isdir(target):
        raise ValueError(
            f"kv_transfer_params.hidden_states_path {requested!r} is not a directory"
        )
    return target


def write_hidden_states(
    directory: str, hidden_states: torch.Tensor, token_ids: torch.Tensor
) -> str:
    """Serialize one artifact under an exclusive ``flock`` and return its path.

    The reader polls for the path to exist and then takes ``LOCK_EX`` before reading, so
    the lock -- not an atomic rename -- is what keeps it from parsing a half-written
    file: it blocks on the lock the instant the (already visible) file appears.
    """
    from safetensors.torch import save

    if hidden_states.dim() != 3:
        raise ValueError(
            "hidden_states must be [prompt_tokens, layers, hidden]; got "
            f"{tuple(hidden_states.shape)}"
        )
    if token_ids.dim() != 1 or token_ids.numel() != hidden_states.shape[0]:
        raise ValueError(
            f"token_ids {tuple(token_ids.shape)} does not match hidden_states token "
            f"count {hidden_states.shape[0]}"
        )
    payload = save(
        {
            _HIDDEN_STATES_KEY: hidden_states.to(torch.bfloat16).contiguous(),
            _TOKEN_IDS_KEY: token_ids.to(torch.int64).contiguous(),
        }
    )
    path = os.path.join(directory, f"{uuid.uuid4().hex}{ARTIFACT_SUFFIX}")
    # 0o666 & ~umask: the reader opens read+write (it deletes the artifact once it has
    # scored it), so a read-only file would fail on its side, not ours.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    return path


class HiddenStateCapture:
    """One request's accumulator, on the host, for both shapes of the export.

    File: per-chunk ``[layers, chunk_tokens, hidden]`` BF16 buffers, concatenated at
    finish. Pooled: one float32 ``[layers, hidden]`` running sum and one ``[layers,
    hidden]`` last-row buffer, updated per chunk -- so a pooled request's footprint
    does not grow with the prompt. Either or both, per ``spec``.
    """

    __slots__ = (
        "spec", "_hidden_size", "_index", "_chunks", "_token_chunks", "_token_count",
        "_written", "_sum", "_last",
    )

    def __init__(self, spec: HiddenStateSpec, hidden_size: int):
        self.spec = spec
        self._hidden_size = hidden_size
        self._index = {layer_id: i for i, layer_id in enumerate(spec.layer_ids)}
        # [layers, chunk_tokens, hidden] per chunk -- layer-major so each block's D2H
        # copy lands in one contiguous host slab; transposed once at finish. Only
        # populated when an artifact is wanted.
        self._chunks: list[torch.Tensor] = []
        # The artifact's ``token_ids``; a pooled-only capture keeps just the count.
        self._token_chunks: list[torch.Tensor] = []
        self._token_count = 0
        # Distinct layer ids written into each chunk. The buffers are uninitialized, so
        # this is what separates "captured" from "a model that never calls the sink".
        self._written: list[set[int]] = []
        pooled = (len(self._index), hidden_size)
        self._sum = (
            torch.zeros(pooled, dtype=torch.float32) if "mean" in spec.pooling else None
        )
        self._last = (
            torch.empty(pooled, dtype=torch.float32) if "last" in spec.pooling else None
        )

    @property
    def token_count(self) -> int:
        return self._token_count

    def begin_chunk(self, token_ids: torch.Tensor) -> None:
        rows = int(token_ids.numel())
        self._token_count += rows
        if self.spec.directory is not None:
            self._token_chunks.append(token_ids.detach().to(torch.int64).clone())
            self._chunks.append(
                torch.empty(
                    len(self._index), rows, self._hidden_size, dtype=torch.bfloat16
                )
            )
        self._written.append(set())

    def write(self, layer_id: int, hidden: torch.Tensor) -> None:
        index = self._index.get(layer_id)
        if index is None:
            return
        # D2H per block: the probe path is opt-in and rare, and staging the whole
        # [chunk, layers, hidden] slab on the GPU first would cost ~1.1 GiB of VRAM at
        # the 4096-token cap for no benefit to a request that samples one token.
        # A second write of one layer in one chunk would double-count the sum.
        assert index not in self._written[-1], f"layer {layer_id} captured twice in a chunk"
        if self._chunks:
            self._chunks[-1][index].copy_(hidden)
        if self._sum is not None:
            # Reduce in float32 (on CUDA the half/bf16 -> float32 cast happens inside
            # the reduction kernel, so nothing [rows, hidden]-sized is materialized) and
            # accumulate across chunks on the host.
            self._sum[index] += hidden.sum(dim=0, dtype=torch.float32).to("cpu")
        if self._last is not None:
            # Each chunk overwrites it; the last chunk's last row is the final prompt
            # position, i.e. what ``token_ids[-1]`` names in the artifact.
            self._last[index].copy_(hidden[-1])
        self._written[-1].add(index)

    def _check_complete(self) -> None:
        if not self._written:
            raise ValueError("no prefill chunk was captured for this request")
        expected = len(self._index)
        missing = [i for i, seen in enumerate(self._written) if len(seen) != expected]
        if missing:
            # The served model's forward never called the sink for (all of) these
            # layers, so the buffers still hold uninitialized memory. Fail loudly
            # instead of handing the router plausible garbage.
            raise ValueError(
                f"prefill chunk(s) {missing} captured fewer than {expected} layers; "
                "this model does not implement the hidden-state hook"
            )

    def finish(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """``(hidden_states, token_ids)`` for the artifact; both None when the spec
        writes no file. Raises when a layer was never captured."""
        self._check_complete()
        if not self._chunks:
            return None, None
        return (
            torch.cat(self._chunks, dim=1).permute(1, 0, 2).contiguous(),
            torch.cat(self._token_chunks),
        )

    def pooled(self) -> dict | None:
        """The JSON-ready ``kv_transfer_params.pooled`` object; None when not pooling.

        Vectors are ``[len(layer_ids), hidden]`` float32, row-major, little-endian,
        base64 -- ``np.frombuffer(b64decode(v), dtype="<f4").reshape(len(layer_ids),
        hidden)`` on the client. ``mean`` is over every prompt position, chat-template
        tokens included, exactly what mean-pooling the artifact gives.
        """
        if not self.spec.pooling:
            return None
        self._check_complete()
        prompt_tokens = self.token_count
        payload: dict = {
            "layer_ids": list(self.spec.layer_ids),
            "hidden": self._hidden_size,
            "prompt_tokens": prompt_tokens,
            "dtype": "float32",
        }
        if self._sum is not None:
            payload["mean"] = _encode_f32(self._sum / prompt_tokens)
        if self._last is not None:
            payload["last"] = _encode_f32(self._last)
        return payload


def _encode_f32(vectors: torch.Tensor) -> str:
    return base64.b64encode(
        vectors.detach().to(torch.float32).contiguous().numpy().astype("<f4").tobytes()
    ).decode("ascii")


class HiddenStateSink:
    """Per-forward fan-out installed on ``Context.hidden_state_sink``.

    The model calls :meth:`capture` once per block with the whole batch's residual
    stream; the sink slices out each probe request's rows. It exists only for the
    duration of one prefill forward that actually has something to capture.
    """

    __slots__ = ("_targets",)

    def __init__(self, targets: list[tuple[HiddenStateCapture, int, int]]):
        self._targets = targets

    def capture(self, layer_id: int, hidden: torch.Tensor) -> None:
        for capture, start, stop in self._targets:
            capture.write(layer_id, hidden[start:stop])


class HiddenStateCollector:
    """Engine-side registry of in-flight captures, keyed by request uid.

    Chunked prefill builds a fresh ``Req`` per chunk, so the accumulator cannot live on
    the request object; the uid is what survives.
    """

    def __init__(self, hidden_size: int, num_layers: int):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self._captures: dict[int, HiddenStateCapture] = {}

    def default_layer_ids(self) -> list[int]:
        return list(range(self.num_layers))

    def begin_batch(self, batch: "Batch") -> HiddenStateSink | None:
        """Build this forward's sink, or None when no request in it wants capture."""
        if not batch.is_prefill:
            return None
        targets: list[tuple[HiddenStateCapture, int, int]] = []
        offset = 0
        for req in batch.padded_reqs:
            extend_len = req.extend_len
            spec = getattr(req, "hidden_states", None)
            if spec is not None and req.uid >= 0:
                capture = self._captures.get(req.uid)
                if capture is None:
                    capture = self._captures[req.uid] = HiddenStateCapture(
                        spec, self.hidden_size
                    )
                capture.begin_chunk(req.input_ids[req.cached_len : req.device_len])
                targets.append((capture, offset, offset + extend_len))
            offset += extend_len
        return HiddenStateSink(targets) if targets else None

    def finish(self, uid: int) -> dict | None:
        """Close ``uid``'s capture and return the response's ``kv_transfer_params``
        object (``hidden_states_path`` and/or ``pooled``); None if it captured nothing."""
        capture = self._captures.pop(uid, None)
        if capture is None:
            return None
        hidden, token_ids = capture.finish()
        result: dict = {}
        pooled = capture.pooled()
        if pooled is not None:
            result["pooled"] = pooled
        if hidden is not None:
            # The pooled half is already in hand; a write failure (full disk, a
            # directory that vanished) costs the client the path, not the vectors.
            try:
                result["hidden_states_path"] = write_hidden_states(
                    capture.spec.directory, hidden, token_ids
                )
            except Exception as exc:  # noqa: BLE001 -- never fail a sampled turn over this
                logger.warning_rank0(
                    f"Hidden-state artifact write failed for request {uid}: {exc}"
                )
        return result

    def discard(self, uid: int) -> None:
        self._captures.pop(uid, None)

    def __len__(self) -> int:
        return len(self._captures)
