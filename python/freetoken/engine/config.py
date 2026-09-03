from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property
from typing import TYPE_CHECKING, List

import torch
from freetoken.distributed import DistributedInfo
from freetoken.models.register import _load_attr, get_model_spec
from freetoken.utils import cached_load_hf_config

if TYPE_CHECKING:
    from freetoken.models import ModelConfig


@dataclass(frozen=True)
class EngineConfig:
    model_path: str
    tp_info: DistributedInfo
    dtype: torch.dtype
    max_running_req: int = 4
    # Optional smaller startup working set for elastic GDN serving. Admission still
    # accepts max_running_req requests, but recurrent-state/graph resources start at
    # this capacity and expand only when demand crosses it. Zero/None disables it.
    elastic_initial_requests: int | None = None
    # In growable multi-agent mode, tune the prefill/decode time slices from measured
    # forward durations. The controller is active only while both phases are runnable.
    adaptive_scheduler: bool = True
    # Automatic Claude Code/Codex sessions protect their completed prefix for this
    # grace period, then degrade to ordinary evictable radix state. Explicit client
    # session_id leases remain protected until close/abort/TTL.
    auto_session_grace_seconds: float = 30.0
    # Idle automatic agent sessions may move their exact KV/GDN checkpoint out of
    # VRAM. ``auto`` uses a private per-server directory below the FreeToken cache;
    # None disables the cold tier. RAM is bounded independently and is admitted
    # only while MemAvailable remains above host_ram_reserve_gb; overflow goes to disk.
    session_spill_dir: str | None = "auto"
    session_spill_ram_gb: float = 2.0
    session_spill_disk_gb: float = 64.0
    attention_backend: str = "auto"
    moe_backend: str = "auto"
    # NVFP4 routed-expert GEMM backend (--nvfp4-backend): auto|marlin|flashinfer|triton.
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
    # Prefill hit/miss split: serve cache-resident experts D2D during prefill
    # prefetch instead of re-streaming the full layer over PCIe. Needs CUDA >= 12.8
    # (cudaMemcpyBatchAsync); no-op unless moe_cache_size > 2 * num_experts.
    moe_prefill_hit_d2d: bool = False
    moe_collect_stats: bool = False  # capture decode miss-rate counters into the cuda graph
    # Persistent pageable-layer profiles are deliberately opt-in. Production traffic can
    # have a very different expert-routing distribution from a tuning gate, so silently
    # training and applying a model-wide profile can regress every later server boot.
    # ``read`` applies an existing model-scoped profile; ``train`` also updates it at idle
    # boundaries (and enables the counters needed to do so).
    moe_pageable_profile: str = "off"  # off | read | train
    # CPU MoE backend (--moe-backend cpu): number of CPU worker threads computing
    # the decode experts. 0 = auto (physical cores). Ignored by other backends.
    moe_cpu_threads: int = 0
    # Hybrid CPU/GPU decode (--moe-backend offload only): which MoE layers decode on
    # the CPU executor instead of the GPU offload/PCIe path. Spec is an explicit id
    # list ("3,7,11"), a count ("8" -> 8 layers evenly strided across depth), or a
    # fraction ("0.5"). None/"" = all layers on GPU (plain offload). --moe-backend cpu
    # already means all layers on CPU and ignores this.
    moe_cpu_layers: str | None = None
    # WSL fallback for expert banks that exceed the CUDA host-registration quota.
    # Overflow layers stay pageable in RAM; decode gathers each step's misses through
    # a small pinned staging buffer and still executes every expert on the GPU.
    # A CUDA host node gathers routed rows into mapped pinned staging, so decode
    # remains graph-replayable without copying an entire fixed-capacity buffer.
    moe_pageable_gpu: bool = False
    # Hybrid MoE backend (--moe-backend hybrid): max experts fetched over PCIe per
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
    # Tokenize each prompt frontend-side so an over-length one is answered with a 400
    # context_length_exceeded before it costs a queue slot (--no-context-preflight opts
    # out; FREETOKEN_CONTEXT_PREFLIGHT overrides both). The scheduler enforces the window
    # regardless -- this only decides where the client learns about it.
    context_preflight: bool = True

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
    def model_config(self) -> ModelConfig:
        spec = get_model_spec(self.hf_config.architectures[0])
        parse_config = _load_attr(spec.module, spec.parse_config)
        config = parse_config(self.hf_config)
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

    @property
    def max_seq_len(self) -> int:
        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        return self.model_config.rotary_config.max_position

    @property
    def max_forward_len(self) -> int:
        return self.max_seq_len

    @property
    def distributed_addr(self) -> str:
        return "tcp://127.0.0.1:2333"
