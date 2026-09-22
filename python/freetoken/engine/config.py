from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import TYPE_CHECKING, List, Literal

import torch
from freetoken.distributed import DistributedInfo
from freetoken.hidden_states import DEFAULT_MAX_TOKENS as HIDDEN_STATES_MAX_TOKENS
from freetoken.layers.quantization import set_quant_config
from freetoken.mm.config import ENCODER_SECTIONS, MultimodalConfig
from freetoken.models.register import EncoderSpec, ModelSpec, _load_attr, checkpoint_quant_config, get_model_spec
from freetoken.utils import cached_load_hf_config, init_logger

if TYPE_CHECKING:
    from freetoken.models import ModelConfig

logger = init_logger(__name__)

# transformers writes an unbounded tokenizer window as VERY_LARGE_INTEGER (int(1e30)).
# Anything at or above this is a "no limit" sentinel, not a served ceiling.
_TOKENIZER_MAX_LEN_SENTINEL = int(1e20)


@dataclass(frozen=True)
class EngineConfig:
    model_path: str
    tp_info: DistributedInfo
    dtype: torch.dtype
    max_running_req: int = 4
    # In growable multi-agent mode, tune the prefill/decode time slices from measured
    # forward durations. The controller is active only while both phases are runnable.
    adaptive_scheduler: bool = True
    # Safety-net timer for automatic Claude Code/Codex sessions: after this many idle
    # seconds a completed prefix MAY be released, but only while another request is
    # queued or the pools are exhausted. 0 (default) disables the timer -- a resident
    # session is checkpointed on demand, when an admission actually fails. Explicit
    # client session_id leases remain protected until close/abort/TTL.
    auto_session_grace_seconds: float = 0.0
    # Idle automatic agent sessions may move their exact KV/GDN checkpoint out of
    # VRAM. ``auto`` uses a stable directory below the FreeToken cache; None disables
    # the cold tier. RAM is bounded independently and is admitted only while
    # MemAvailable remains above host_ram_reserve_gb; overflow goes to disk.
    session_spill_dir: str | None = "auto"
    # 4 GiB holds one look-ahead 1M checkpoint (~3 GiB KV + boundary states) beside the
    # one being written, which is what makes the queued-session prefetch worth doing.
    session_spill_ram_gb: float = 4.0
    session_spill_disk_gb: float = 64.0
    # Total retained checkpoint bytes (RAM + disk). A spill that would exceed it evicts
    # least-recently-used checkpoints instead of refusing; checkpoint lifetime is
    # therefore bounded by capacity and age, not by the session's lease TTL.
    session_spill_limit_gb: float = 50.0
    # Keep disk checkpoints across restarts (each carries a manifest with the model id,
    # K/V layout fingerprint and prompt-prefix hash; startup adopts the matching ones and
    # deletes the rest). False restores the old wipe-on-exit behavior.
    session_spill_persist: bool = True
    # Look-ahead spacing of the extra recurrent-state boundaries a checkpoint carries. A
    # restore cuts at the deepest stored boundary the client's tokens still match, so this
    # bounds the re-prefill a retokenization drift costs -- at ~47 MiB per boundary.
    session_spill_state_stride: int = 65_536
    # Copy the recurrent state to the host every ``session_spill_state_stride`` prefilled
    # tokens. None = automatic: on only when the state pool holds fewer than 6 slots per
    # running request, which is when a chunk commit cannot spare a slot to donate a snapshot
    # to the radix tree and a partial-prefix restore would otherwise have nothing to cut at.
    session_spill_capture_states: bool | None = None
    attention_backend: str = "auto"
    moe_strategy: str = "auto"
    # old name of moe_strategy; __post_init__ folds it in
    moe_backend: str | None = field(default=None, repr=False)
    # --quant-backend: layer[.kind]=kernel entries, comma separated
    quant_backend: str | None = None
    # PLE table backend: "disk" (default) reads rows from the checkpoint files per fill, "pinned" preloads the table into page-locked host RAM.
    ple_backend: str = "disk"
    # NVFP4 routed-expert GEMM backend (--nvfp4-backend): auto|marlin|flashinfer|triton.
    # (Upstream 477c860's rewrite of this file dropped the field; the fork's
    # moe/nvfp4_backends.py path still resolves through it.)
    nvfp4_backend: str = "triton"
    # Expert-bank host load (--expert-load): auto|serial|parallel. "auto" reads scattered
    # experts in parallel but falls back to serial when free RAM can't cover the banks + the
    # parallel reader's extra (non-reclaimable) whole-shard buffer; "serial" forces the
    # low-memory reclaimable read; "parallel" forces the fast read.
    expert_load: str = "auto"
    # Host memory kept outside resident expert banks. The pre-allocation gate rejects a
    # configuration that would consume this reserve instead of letting Linux's OOM killer
    # terminate the serving process (and, commonly, its launching terminal).
    host_ram_reserve_gb: float = 3.0
    moe_cache_size: int = 0
    moe_cache_rate: float | None = None
    moe_cache_auto: bool = False
    kv_reserve_tokens: int = 8192  # KV floor for --moe-cache-auto; small by design (MoE-priority)
    moe_cache_policy: str = "lru"
    moe_prefill_overlap: bool = True
    # Where an expert's bytes live while it is not on the GPU (moe/residency.py):
    # "whole" pins every expert row in host RAM; "mirror" bounds host expert RAM to
    # a pool of moe_mirror_host_rows rows. Resolved by server/args.py (the
    # FREETOKEN_MIRROR_EXPERT_RAM / FREETOKEN_MIRROR_HOST_ROWS aliases live there).
    expert_residency: Literal["whole", "mirror"] = "whole"
    # Bound host expert RAM to N mirror rows (0 = off / auto-size under
    # expert_residency="mirror", -1 = auto-size from model geometry + KV ceiling).
    # Native NVFP4 experts only.
    moe_mirror_host_rows: int = 0
    # Prefill hit/miss split: serve cache-resident experts D2D during prefill
    # prefetch instead of re-streaming the full layer over PCIe. Needs CUDA >= 12.8
    # (cudaMemcpyBatchAsync); no-op unless moe_cache_size > 2 * num_experts.
    moe_prefill_hit_d2d: bool = False
    # Extend (prefill-path) forwards carrying at most this many tokens take the DECODE
    # expert cache instead of streaming every expert of every layer over PCIe. The
    # prefill stream costs num_experts rows per layer per forward regardless of token
    # count (15.7 GiB per forward on Nemotron 3.5 Lightning, ~290 ms), which is free
    # behind an 8K chunk's GPU work and is the entire cost of a short extend. 0 disables.
    moe_extend_cache_tokens: int = 64
    moe_collect_stats: bool = False  # capture decode miss-rate counters into the cuda graph
    # Persistent pageable-layer profiles are deliberately opt-in. Production traffic can
    # have a very different expert-routing distribution from a tuning gate, so silently
    # training and applying a model-wide profile can regress every later server boot.
    # ``read`` applies an existing model-scoped profile; ``train`` also updates it at idle
    # boundaries (and enables the counters needed to do so).
    moe_pageable_profile: str = "off"  # off | read | train
    # CPU MoE backend (--moe-strategy cpu): number of CPU worker threads computing
    # the decode experts. 0 = auto (physical cores). Ignored by other backends.
    moe_cpu_threads: int = 0
    # Hybrid CPU/GPU decode (--moe-strategy offload only): which MoE layers decode on
    # the CPU executor instead of the GPU offload/PCIe path. Spec is an explicit id
    # list ("3,7,11"), a count ("8" -> 8 layers evenly strided across depth), or a
    # fraction ("0.5"). None/"" = all layers on GPU (plain offload). --moe-strategy cpu
    # already means all layers on CPU and ignores this.
    moe_cpu_layers: str | None = None
    # WSL fallback for expert banks that exceed the CUDA host-registration quota.
    # Overflow layers stay pageable in RAM; decode gathers each step's misses through
    # a small pinned staging buffer and still executes every expert on the GPU.
    # A CUDA host node gathers routed rows into mapped pinned staging, so decode
    # remains graph-replayable without copying an entire fixed-capacity buffer.
    moe_pageable_gpu: bool = False
    # Hybrid MoE backend (--moe-strategy hybrid): max experts fetched over PCIe per
    # (layer, decode step); the rest of that step's misses are computed on the CPU.
    # -1 (default) = auto: fetch the benched pcie_bw/cpu_bw fraction of each step's
    # misses so the PCIe fetch and the CPU compute finish together (perfect overlap);
    # falls back to a fixed cap of 1 without a usable `ft bench bw` profile.
    moe_hybrid_max_fetch: int = -1
    cuda_graph_bs: List[int] | None = None
    cuda_graph_max_bs: int | None = None
    page_size: int = 1
    memory_ratio: float = 0.9
    # Hybrid GDN models default to the HybridRadixCache (cross-request GDN-state prefix reuse);
    # `--cache-type naive` opts out. linear_state_cache_ratio sizes the GDN snapshot cache as
    # ceil(ratio * max_running_req) extra slots.
    linear_state_cache_ratio: float = 2.0
    # Window/full ratio for the SWA radix cache (`--cache-type radix` on SWA models) and the DSV4
    # window tier: the DEFAULT window-pool size = max(working-set floor, ratio x full-pool tokens).
    # < 1.0 trades retained window-prefix capacity for memory savings; must be in (0, 1]. It is the
    # DSV4 window/full ratio directly. Used only when swa_num_pages_override is None (a runtime
    # rebuild can pin an absolute window instead).
    swa_full_tokens_ratio: float = 0.2
    # Absolute window-pool size in the pool's own pages (usable, dummy excluded); None -> use the
    # ratio default above. A runtime cache rebuild sets this (num_swa_pages) to pin the window
    # regardless of the full anchor; the ratio is the startup default and the fallback.
    swa_num_pages_override: int | None = None
    distributed_timeout: float = 60.0
    use_dummy_weight: bool = False
    use_pynccl: bool = True
    max_seq_len_override: int | None = None
    # Optional runtime YaRN extension for checkpoints whose native metadata does
    # not encode the longer deployment context (notably Ornith GGUF).  The
    # original length defaults to the checkpoint rotary maximum.  This changes
    # the actual frequency table as well as its size; max_seq_len_override alone
    # must never be used to extend RoPE out of bounds.
    rope_yarn_factor: float | None = None
    rope_yarn_original_context: int | None = None
    # Physical GDN state slots, including the padding sink. None uses the normal
    # cache-ratio policy. A constrained dual-request deployment can request the
    # proven 4*max_running_req+1 minimum without paying for unused snapshots.
    linear_state_slots_override: int | None = None
    num_page_override: int | None = None  # if not None, will override the number of pages
    # KV capacity in tokens; resolved into num_page_override by _adjust_config once page_size
    # is final. Mutually exclusive with num_page_override.
    num_token_override: int | None = None
    # KV element storage (--kv-cache-dtype): "auto" keeps the compute dtype; q8_0 and
    # fp8_e4m3 store 8 bits, while int4/q4_0 use GGML Q4_0 with two values per byte.
    # Every quantized scheme carries a per-block scale. Resolved by the pools and cost model.
    kv_cache_dtype: str = "auto"
    # Optional independent formats. When omitted each inherits kv_cache_dtype. This is
    # useful because keys are more quality-sensitive than values. Validated pairs are
    # the high-fidelity Q8-K/Q6-V lane and the smaller Q6-K/Q5-V lane.
    kv_cache_dtype_k: str | None = None
    kv_cache_dtype_v: str | None = None
    # Reserve the full KV virtual range but physically commit it in chunks, shrinking the
    # GPU expert cache at each boundary. Zero keeps the conventional eager allocation.
    kv_grow_step_tokens: int = 0
    # Fixed-capacity VMM expert arena whose usable slot count shrinks/grows in place, so a
    # growable-KV resize never rebuilds the expert cache or recaptures decode graphs. The
    # engine publishes it to moe/offload_cache.py + offload_kernels.py at init. Resolved by
    # server/args.py (--expert-arena; FREETOKEN_EXPERT_ARENA=1 is its alias).
    expert_arena: bool = False
    # Tokenize each prompt frontend-side so an over-length one is answered with a 400
    # context_length_exceeded before it costs a queue slot (--no-context-preflight opts
    # out; FREETOKEN_CONTEXT_PREFLIGHT overrides both). The scheduler enforces the window
    # regardless -- this only decides where the client learns about it.
    context_preflight: bool = True
    # Repair attempts for response_format json_object/json_schema when the answer
    # is not valid JSON (--json-retry; FREETOKEN_JSON_RETRY overrides).
    json_retry: int = 1
    # Root directory for Switchyard prefill-probe hidden-state artifacts
    # (--hidden-states-dir). None disables the feature: a request that asks for one is
    # refused rather than served, and no other path may be written.
    hidden_states_dir: str | None = None
    # Per-probe prompt-token cap (--hidden-states-max-tokens); a longer prompt is a 400.
    hidden_states_max_tokens: int = HIDDEN_STATES_MAX_TOKENS
    # Root directory for the pooled hidden-state JSONL sink (--pooled-sink-dir). None
    # disables it: pooled vectors are only returned inline, and a request naming a
    # kv_transfer_params.pooled_sink is refused.
    pooled_sink_dir: str | None = None
    # Runtime knobs of the multimodal path; the architecture side (vision_config, mrope) lives in ModelConfig.
    mm: MultimodalConfig = field(default_factory=MultimodalConfig)

    def __post_init__(self):
        if self.moe_backend is None:
            return
        if self.moe_strategy != "auto":
            raise ValueError("moe_backend is the old name of moe_strategy; pass only moe_strategy")
        logger.warning("EngineConfig.moe_backend is deprecated; use moe_strategy")
        object.__setattr__(self, "moe_strategy", self.moe_backend)
        object.__setattr__(self, "moe_backend", None)

    @cached_property
    def kv_quant(self):
        from freetoken.kvcache.quant import resolve_kv_quant

        return resolve_kv_quant(self.kv_cache_dtype)

    @cached_property
    def kv_quant_k(self):
        from freetoken.kvcache.quant import resolve_kv_quant

        return resolve_kv_quant(self.kv_cache_dtype_k or self.kv_cache_dtype)

    @cached_property
    def kv_quant_v(self):
        from freetoken.kvcache.quant import resolve_kv_quant

        return resolve_kv_quant(self.kv_cache_dtype_v or self.kv_cache_dtype)

    @cached_property
    def hf_config(self):
        return cached_load_hf_config(self.model_path)

    @cached_property
    def model_spec(self) -> ModelSpec:
        return get_model_spec(self.hf_config.architectures[0])

    @cached_property
    def active_encoders(self) -> tuple[EncoderSpec, ...]:
        """The encoder towers this process builds: the family registers them, the checkpoint config carries their section, --mm-disable did not name them."""
        return tuple(
            e
            for e in self.model_spec.encoders
            if getattr(self.hf_config, e.config_key, None) is not None
            and e.kind not in self.mm.disabled_encoders
        )

    @cached_property
    def served_modalities(self) -> frozenset[str]:
        """Modalities this process accepts."""
        return frozenset(m for e in self.active_encoders for m in e.modalities)

    @cached_property
    def model_config(self) -> ModelConfig:
        # the parser sees no section for a tower this process does not build (for the vision tower that also means 1-D rope)
        hf_config = copy.copy(self.hf_config)
        built = {e.config_key for e in self.active_encoders}
        for key in set(ENCODER_SECTIONS) | {e.config_key for e in self.model_spec.encoders}:
            if key not in built:
                setattr(hf_config, key, None)
        spec = self.model_spec
        quant = checkpoint_quant_config(self.model_path, hf_config, spec)
        set_quant_config(quant)
        model_config = _load_attr(spec.module, spec.parse_config)(hf_config)
        config = replace(model_config, quant=quant)
        factor = self.rope_yarn_factor
        if factor is None:
            if self.rope_yarn_original_context is not None:
                raise ValueError(
                    "--rope-yarn-original-context requires --rope-yarn-factor"
                )
            return config
        if factor < 1.0:
            raise ValueError("--rope-yarn-factor must be >= 1")

        from freetoken.models.config import FullAttentionGroupConfig

        original = self.rope_yarn_original_context or config.rotary_config.max_position
        if original < 1:
            raise ValueError("--rope-yarn-original-context must be >= 1")
        scaled = int(round(original * factor))
        rotary = replace(
            config.rotary_config,
            max_position=scaled,
            scaling={
                "rope_type": "yarn",
                "factor": float(factor),
                "original_max_position_embeddings": int(original),
            },
        )
        groups = tuple(
            replace(group, rotary_config=rotary)
            if isinstance(group, FullAttentionGroupConfig)
            else group
            for group in config.attention_groups
        )
        return replace(config, rotary_config=rotary, attention_groups=groups)

    @cached_property
    def tokenizer_model_max_length(self) -> int | None:
        """``model_max_length`` from ``tokenizer_config.json``, or None.

        Checkpoints whose positional geometry is far larger than the window the
        tokenizer (and therefore the publisher) actually supports would otherwise
        advertise the geometry: Nemotron-3.5 Lightning carries
        ``max_position_embeddings`` 1,048,576 against a tokenizer
        ``model_max_length`` of 262,144. Transformers writes "unbounded" as a huge
        sentinel (``VERY_LARGE_INTEGER``, int(1e30)), which is not a real limit and
        is ignored here, as are absent/malformed values.
        """
        import json
        import os

        if os.path.isdir(self.model_path):
            path = os.path.join(self.model_path, "tokenizer_config.json")
            if not os.path.isfile(path):
                return None
        elif os.path.exists(self.model_path):
            return None  # a bare .gguf file carries its own metadata
        else:
            try:
                from huggingface_hub import hf_hub_download

                path = hf_hub_download(
                    repo_id=self.model_path, filename="tokenizer_config.json"
                )
            except Exception:
                return None
        try:
            with open(path, encoding="utf-8") as f:
                value = json.load(f).get("model_max_length")
        except Exception:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        if not math.isfinite(value) or value < 1 or value >= _TOKENIZER_MAX_LEN_SENTINEL:
            return None
        return int(value)

    @property
    def max_seq_len(self) -> int:
        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        max_position = self.model_config.rotary_config.max_position
        limit = self.tokenizer_model_max_length
        if limit is not None and limit < max_position:
            return limit
        return max_position

    @property
    def max_forward_len(self) -> int:
        return self.max_seq_len

    @property
    def distributed_addr(self) -> str:
        return "tcp://127.0.0.1:2333"
