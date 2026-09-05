from __future__ import annotations

from dataclasses import dataclass, field

from freetoken.engine import EngineConfig


def _get_pid_suffix() -> str:
    import os

    return f".pid={os.getpid()}"


@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    max_extend_tokens: int = 8192
    # CLI requests without an explicit chunk size may apply a measured
    # architecture/model-specific value after the CUDA device is selected.
    auto_prefill_chunk: bool = False
    # Independent prompt lanes per prefill forward. None auto-selects one for growable quantized
    # GGUF MoE with concurrency; zero explicitly keeps normal aggregate-token batching.
    max_prefill_seqs: int | None = None
    cache_type: str = "radix"
    # --- speculative decoding (scheduler/spec_ngram.py) ---
    # None disables it; "ngram" enables prompt-lookup (n-gram) speculation. Greedy-only and
    # single-stream in v1: a request with temperature > 0, or any step with more than one
    # running request, takes the ordinary decode path.
    speculative: str | None = None
    # n-gram order. 8, not the literature's 3: when verification costs 4.4x a decode step you
    # draft for precision, not recall (see benchmarks/results/..._ngram_spec_2026-09-05.md §2).
    spec_ngram_n: int = 8
    # Draft tokens per verify step; the forward carries spec_draft_len + 1 positions.
    spec_draft_len: int = 8
    # Halve the draft length after a rejection at position 0, restore it after a full accept.
    spec_adaptive: bool = True
    offline_mode: bool = False
    decode_log_interval: int = 40
    special_token_ckpt: bool = False

    # networking config
    _unique_suffix: str = field(default_factory=_get_pid_suffix)

    @property
    def zmq_backend_addr(self) -> str:
        return "ipc:///tmp/freetoken_0" + self._unique_suffix

    @property
    def zmq_detokenizer_addr(self) -> str:
        return "ipc:///tmp/freetoken_1" + self._unique_suffix

    @property
    def zmq_scheduler_broadcast_addr(self) -> str:
        return "ipc:///tmp/freetoken_2" + self._unique_suffix

    @property
    def max_forward_len(self) -> int:
        return self.max_extend_tokens

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return True
