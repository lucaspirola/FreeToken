from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from freetoken.attention.linear import build_fla_metadata
from freetoken.core import get_global_ctx
from freetoken.kernel.causal_conv1d import causal_conv1d_decode, causal_conv1d_varlen
from freetoken.kernel.triton.mamba2 import (
    mamba2_decode,
    mamba2_gated_rmsnorm,
    mamba2_prefill,
)
from freetoken.layers import (
    BaseOP,
    LinearColParallelMerged,
    LinearReplicated,
    OPList,
    ParallelLMHead,
    RMSNorm,
    VocabParallelEmbedding,
    make_moe_layer,
)
from freetoken.models.blocks import BaseLLMModel
from freetoken.utils import nvtx_annotate

from .fp8_act_stats import ACT_STATS_PATH
from .mamba2_reference import reference_enabled
from .state_dump import STATE_DUMP_DIR

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig
    from .config import NemotronHArgs


def _act_stats_on(batch) -> bool:
    """FP8 activation statistics are collected on real prefill batches only: decode
    batches are CUDA-graph captured (a host sync inside capture is illegal) and the
    engine's warmup/profiling batches (``uid < 0``) carry dummy tokens whose magnitudes
    would pollute the recorded amax."""
    return (
        ACT_STATS_PATH is not None
        and not batch.is_decode
        and all(req.uid >= 0 for req in batch.reqs)
    )


def _linear(args: "NemotronHArgs", name: str, in_f: int, out_f: int):
    quant = args.module_quant(name)
    if quant == "fp8_pertensor":
        from freetoken.kernel.triton.fp8_pertensor_linear import Fp8PerTensorLinear

        return Fp8PerTensorLinear(in_f, out_f, has_bias=False)
    if quant == "nvfp4":
        # Dense NVFP4 (shared experts): served native W4A16 instead of dequantized to
        # bf16 at load, which costs ~1.6 GiB of resident weights on Lightning.
        from freetoken.kernel.triton.nvfp4_linear import Nvfp4DenseLinear

        return Nvfp4DenseLinear(in_f, out_f, has_bias=False)
    return LinearReplicated(in_f, out_f, has_bias=False)


class _DepthwiseConv1d(BaseOP):
    def __init__(self, dim: int, kernel: int):
        self.weight = torch.empty(dim, 1, kernel)
        self.bias = torch.empty(dim)


class _MambaGatedRMSNorm(BaseOP):
    """``norm(x * silu(z))`` per group -- one fused Triton kernel (kernel/fla
    layernorm_gated, pinned to the Mamba-2 flags by mamba2_gated_rmsnorm). The
    pure-PyTorch twin lives in mamba2_reference.reference_gated_rmsnorm."""

    def __init__(self, size: int, groups: int, eps: float):
        self.weight = torch.empty(size)
        self.groups = groups
        self.group_size = size // groups
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        if reference_enabled():
            from .mamba2_reference import reference_gated_rmsnorm

            return reference_gated_rmsnorm(x, gate, self.weight, self.eps, self.group_size)
        return mamba2_gated_rmsnorm(x, gate, self.weight, self.eps, self.group_size)


class NemotronHMamba2Mixer(BaseOP):
    def __init__(self, config: "ModelConfig", layer_id: int):
        args = config.nemotron_h_args
        assert args is not None
        self.layer_id = layer_id
        self.num_heads = args.mamba_num_heads
        self.head_dim = args.mamba_head_dim
        self.state_size = args.ssm_state_size
        self.n_groups = args.n_groups
        self.intermediate_size = args.mamba_intermediate_size
        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.state_size
        self.chunk_size = args.chunk_size
        # No lower clamp on the discretized timestep (args.time_step_min is 0.0 unless
        # FREETOKEN_NEMOTRON_DT_MIN overrides it): see config._dt_floor. Decode never
        # clamped, so prefill and decode now agree, as do vLLM and llama.cpp.
        self.dt_limit = (args.time_step_min, float("inf"))
        prefix = f"backbone.layers.{layer_id}.mixer"
        self.in_proj = _linear(
            args, f"{prefix}.in_proj", config.hidden_size,
            self.intermediate_size + self.conv_dim + self.num_heads,
        )
        self.conv1d = _DepthwiseConv1d(self.conv_dim, args.conv_kernel)
        self.dt_bias = torch.empty(self.num_heads, dtype=torch.float32)
        self.A_log = torch.empty(self.num_heads, dtype=torch.float32)
        self.norm = _MambaGatedRMSNorm(
            self.intermediate_size, self.n_groups, config.rms_norm_eps
        )
        self.D = torch.empty(self.num_heads, dtype=torch.float32)
        # Leading-underscore attrs are skipped by BaseOP.state_dict/load_state_dict.
        self._A_cache: tuple[torch.Tensor, torch.Tensor] | None = None
        self._out_buffers: dict[tuple, torch.Tensor] = {}
        self.out_proj = _linear(
            args, f"{prefix}.out_proj", self.intermediate_size, config.hidden_size
        )

    def _conv(self, conv_in: torch.Tensor, fla, pool, decode: bool) -> torch.Tensor:
        li = pool.local_index(self.layer_id)
        weight = self.conv1d.weight.squeeze(1)
        if decode:
            return causal_conv1d_decode(
                conv_in, pool.conv_states[li], weight, fla.cache_indices,
                bias=self.conv1d.bias,
            )
        return causal_conv1d_varlen(
            conv_in.transpose(0, 1).contiguous(), weight, pool.conv_states[li],
            fla.cu_seqlens, fla.cache_indices, fla.has_initial_state,
            bias=self.conv1d.bias,
        ).transpose(0, 1)

    @property
    def A(self) -> torch.Tensor:
        """``-exp(A_log)`` [H] fp32, cached on the module.

        Both kernels want the negated log-decay, and the flashinfer decode front end
        keys its stride-0 expanded views on ``id(A)`` -- rebuilding it every step would
        allocate on every call and defeat that cache. Keyed on the identity of
        ``A_log`` so a (re)load_state_dict is picked up."""
        cached = self._A_cache
        if cached is None or cached[0] is not self.A_log:
            cached = (self.A_log, -torch.exp(self.A_log.float()))
            self._A_cache = cached
        return cached[1]

    def _decode_out(self, x: torch.Tensor) -> torch.Tensor:
        """The [bs, H, P] decode output buffer, so a steady-state step allocates nothing.

        One buffer PER batch size, and an entry is never replaced once handed out. A
        single grow-only buffer would be a use-after-free: capture runs an eager warmup
        at every graph batch size, so each captured graph bakes in the address it saw,
        and a later *eager* decode wider than the largest captured size (elastic capacity
        raises max_running_requests above the captured sizes) would reallocate the buffer
        and free the block those graphs still write to on every replay.

        Keyed on (bs, dtype, device); the entry count is bounded by the graph batch-size
        list plus the eager sizes actually seen, at 2 * H * P bytes each (128 KiB at
        bs=16 for Lightning)."""
        key = (x.shape[0], x.dtype, x.device)
        buf = self._out_buffers.get(key)
        if buf is None:
            buf = torch.empty(x.shape[0], self.num_heads, self.head_dim,
                              dtype=x.dtype, device=x.device)
            self._out_buffers[key] = buf
        return buf

    def _prefill_scan(self, x, dt, B, C, fla, pool) -> torch.Tensor:
        """Chunked SSD scan over the whole varlen batch in one launch.

        The pool slot is already the kernels' native [H, P, N] block, so nothing is
        transposed; `has_initial_state` zeroes the carried state of fresh sequences
        inside the gather, replacing the old `fresh_state_indices` index_fill_."""
        li = pool.local_index(self.layer_id)
        state_source = pool.recurrent_states[li]
        assert fla.mamba2 is not None, "Mamba-2 prefill needs FLAMetadata.mamba2"
        track = fla.track_dst is not None
        out, states = mamba2_prefill(
            x, dt, B, C,
            A=self.A, D=self.D, dt_bias=self.dt_bias,
            meta=fla.mamba2, cu_seqlens=fla.cu_seqlens,
            state_source=state_source, indices=fla.cache_indices,
            has_initial_state=fla.has_initial_state,
            return_intermediate_states=track,
            dt_softplus=True, dt_limit=self.dt_limit,
        )
        if track:
            # Hybrid-radix mid-prefill snapshot: the state at the deepest interior chunk
            # boundary is already a row of the per-chunk block, so the donate is a gather
            # + scatter instead of the reference path's second scan.
            state_source.index_copy_(
                0, fla.track_dst, states[fla.track_h_row].to(state_source.dtype)
            )
        return out

    def _decode_scan(self, x, dt, B, C, fla, pool) -> torch.Tensor:
        li = pool.local_index(self.layer_id)
        return mamba2_decode(
            x, dt, B, C,
            A=self.A, D=self.D, dt_bias=self.dt_bias,
            state_source=pool.recurrent_states[li], indices=fla.cache_indices,
            out=self._decode_out(x), dt_softplus=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        batch, pool = ctx.batch, ctx.linear_state_pool
        assert pool is not None
        fla = batch.fla_metadata
        if fla is None:
            fla = build_fla_metadata(batch, hidden_states.device)
            batch.fla_metadata = fla
        if _act_stats_on(batch):
            from .fp8_act_stats import record

            record(
                f"backbone.layers.{self.layer_id}.mixer.in_proj",
                hidden_states,
                getattr(self.in_proj, "input_scale", None),
            )
        proj = self.in_proj.forward(hidden_states)
        gate, conv_in, dt = torch.split(
            proj, [self.intermediate_size, self.conv_dim, self.num_heads], dim=-1
        )
        mixed = self._conv(conv_in, fla, pool, batch.is_decode)
        if not batch.is_decode and fla.track_dst is not None:
            li = pool.local_index(self.layer_id)
            # The causal-conv pool itself is updated to the end of each extend. Preserve
            # the raw conv-input window at the earlier radix snapshot boundary separately.
            conv_window = conv_in[fla.track_conv_src].transpose(-1, -2).contiguous()
            pool.conv_states[li].index_copy_(
                0, fla.track_dst, conv_window.to(pool.conv_states.dtype)
            )
        x, B, C = torch.split(
            mixed,
            [self.intermediate_size, self.n_groups * self.state_size,
             self.n_groups * self.state_size],
            dim=-1,
        )
        x = x.view(-1, self.num_heads, self.head_dim)
        B = B.view(-1, self.n_groups, self.state_size)
        C = C.view(-1, self.n_groups, self.state_size)
        if reference_enabled():
            from . import mamba2_reference

            scan = (mamba2_reference.reference_decode_scan if batch.is_decode
                    else mamba2_reference.reference_prefill_scan)
            scanned = scan(self, x, dt, B, C, fla, pool)
        elif batch.is_decode:
            scanned = self._decode_scan(x, dt, B, C, fla, pool)
        else:
            scanned = self._prefill_scan(x, dt, B, C, fla, pool)
        # The SSD kernels return x's dtype; the pure-Torch reference accumulates in fp32
        # and is cast back here, matching HF before the gated norm and output projection.
        out = self.norm.forward(
            scanned.reshape(-1, self.intermediate_size).to(gate.dtype), gate
        )
        if _act_stats_on(batch):
            from .fp8_act_stats import record

            record(
                f"backbone.layers.{self.layer_id}.mixer.out_proj",
                out,
                getattr(self.out_proj, "input_scale", None),
            )
        return self.out_proj.forward(out)


class NemotronHAttention(BaseOP):
    def __init__(self, config: "ModelConfig", layer_id: int):
        args = config.nemotron_h_args
        self.layer_id = layer_id
        self.num_q = config.num_qo_heads
        self.num_kv = config.num_kv_heads
        self.head_dim = config.head_dim
        self.q_dim = self.num_q * self.head_dim
        self.kv_dim = self.num_kv * self.head_dim
        # Nemotron-H attention is deliberately position-free: unlike most transformer
        # blocks, it applies no RoPE. q/k/v are BF16 in this release; two o projections
        # are FP8 and follow the checkpoint's per-module quant map.
        self.qkv_proj = LinearColParallelMerged(
            config.hidden_size, [self.q_dim, self.kv_dim, self.kv_dim], has_bias=False
        )
        self.o_proj = _linear(
            args, f"backbone.layers.{layer_id}.mixer.o_proj",
            self.q_dim, config.hidden_size,
        )

    @nvtx_annotate("MHA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        q, k, v = torch.split(
            self.qkv_proj.forward(x), [self.q_dim, self.kv_dim, self.kv_dim], dim=-1
        )
        q = q.contiguous().view(-1, self.num_q, self.head_dim)
        out = ctx.attn_backend.forward(q, k.contiguous(), v.contiguous(), self.layer_id, ctx.batch)
        return self.o_proj.forward(out.reshape(-1, self.q_dim))


class NemotronHMLP(BaseOP):
    def __init__(self, config: "ModelConfig", layer_id: int, name: str, width: int):
        args = config.nemotron_h_args
        prefix = f"backbone.layers.{layer_id}.mixer.{name}"
        self.up_proj = _linear(args, f"{prefix}.up_proj", config.hidden_size, width)
        self.down_proj = _linear(args, f"{prefix}.down_proj", width, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj.forward(F.relu(self.up_proj.forward(x)).square())


class _NemotronRouter(BaseOP):
    def __init__(self, hidden_size: int, num_experts: int):
        self.weight = torch.empty(num_experts, hidden_size)
        self.e_score_correction_bias = torch.empty(num_experts, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.sigmoid(F.linear(x.float(), self.weight.float()))
        return scores, scores + self.e_score_correction_bias


class NemotronHMoE(BaseOP):
    def __init__(self, config: "ModelConfig", layer_id: int, bank_id: int):
        args = config.nemotron_h_args
        assert args is not None
        self.gate = _NemotronRouter(config.hidden_size, config.num_experts)
        # Nemotron-3-Super runs its experts in a narrower latent space (fc1/fc2 project
        # in and out); Nemotron-3.5-Lightning has ``moe_latent_size: null`` and routes
        # the residual stream straight through, so the projections must not exist at all
        # (they would demand checkpoint tensors that are not there).
        self.has_latent = args.moe_latent_size is not None
        if self.has_latent:
            self.fc1_latent_proj = _linear(
                args, f"backbone.layers.{layer_id}.mixer.fc1_latent_proj",
                config.hidden_size, args.moe_latent_size,
            )
        self.experts = make_moe_layer(
            config, layer_id=bank_id, activation=config.hidden_act, renormalize=False,
            hidden_size=config.expert_hidden_size,
            intermediate_size=config.moe_intermediate_size,
        )
        if self.has_latent:
            self.fc2_latent_proj = _linear(
                args, f"backbone.layers.{layer_id}.mixer.fc2_latent_proj",
                args.moe_latent_size, config.hidden_size,
            )
        self.shared_experts = NemotronHMLP(
            config, layer_id, "shared_experts", args.shared_intermediate_size
        )
        self.top_k = config.num_experts_per_tok
        self.scale = config.routed_scaling_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores, choice = self.gate.forward(x)
        ids = torch.topk(choice, self.top_k, dim=-1, sorted=False).indices
        weights = scores.gather(1, ids)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        weights = (weights * self.scale).float()
        latent = self.fc1_latent_proj.forward(x) if self.has_latent else x
        routed = self.experts.routed_forward(latent, weights, ids.to(torch.int32)).to(x.dtype)
        if self.has_latent:
            routed = self.fc2_latent_proj.forward(routed)
        return routed + self.shared_experts.forward(x)


class NemotronHBlock(BaseOP):
    def __init__(self, config: "ModelConfig", layer_id: int, moe_banks: dict[int, int]):
        self._layer_id = layer_id
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        kind = config.nemotron_h_args.layer_types[layer_id]
        if kind == "mamba":
            self.mixer = NemotronHMamba2Mixer(config, layer_id)
        elif kind == "attention":
            self.mixer = NemotronHAttention(config, layer_id)
        elif kind == "moe":
            self.mixer = NemotronHMoE(config, layer_id, moe_banks[layer_id])
        else:
            raise ValueError(f"unsupported Nemotron-H block type {kind!r}")

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mixer.forward(self.norm.forward(x))


class NemotronHBackbone(BaseOP):
    def __init__(self, config: "ModelConfig"):
        self.embeddings = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        moe_banks = {layer: bank for bank, layer in enumerate(config.moe_layer_ids or ())}
        self.layers = OPList([
            NemotronHBlock(config, layer, moe_banks) for layer in range(config.num_layers)
        ])
        self.norm_f = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run the block stack, optionally exporting each block's residual stream.

        ``ctx.hidden_state_sink`` is the generic model-side hook for Switchyard's
        prefill probe (freetoken/hidden_states.py): any model may call it after each
        block, and every Nemotron-H block counts as one "layer" here -- mamba, MoE and
        attention alike -- because the router consumes the residual stream, not a
        per-kind subset. The captured value is the post-block residual (``x`` after the
        block's own add), so it is neither normed by the next block nor by ``norm_f``,
        which is what vLLM's ``eagle_aux_hidden_state_layer_ids`` export means. The sink
        is None on every ordinary forward, making this one attribute read.
        """
        x = self.embeddings.forward(input_ids)
        sink = get_global_ctx().hidden_state_sink
        if sink is None:
            for layer in self.layers.op_list:
                x = layer.forward(x)
        else:
            for layer_id, layer in enumerate(self.layers.op_list):
                x = layer.forward(x)
                sink.capture(layer_id, x)
        return self.norm_f.forward(x)


class NemotronHForCausalLM(BaseLLMModel):
    def __init__(self, config: "ModelConfig"):
        self.backbone = NemotronHBackbone(config)
        if getattr(config, "lm_head_quant", "none") == "nvfp4":
            # The checkpoint stores the (untied) lm_head as NVFP4: keep it native (W4A16)
            # rather than dequantizing ~0.7 GiB of bf16 for the largest decode GEMV.
            from freetoken.kernel.triton.nvfp4_linear import Nvfp4LMHead

            assert not config.tie_word_embeddings, "NVFP4 lm_head assumes untied embeddings"
            self.lm_head = Nvfp4LMHead(
                num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
            )
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size, config.hidden_size,
                tie_word_embeddings=config.tie_word_embeddings,
                tied_embedding=(
                    self.backbone.embeddings if config.tie_word_embeddings else None
                ),
            )
        super().__init__()

    def forward(self) -> torch.Tensor:
        hidden = self.backbone.forward(get_global_ctx().batch.input_ids)
        logits = self.lm_head.forward(hidden)
        if STATE_DUMP_DIR is not None:
            # Debug A/B only (FREETOKEN_MAMBA2_STATE_DUMP=<dir>); see state_dump.py.
            from .state_dump import dump_prefill_state

            dump_prefill_state(logits)
        if _act_stats_on(get_global_ctx().batch):
            # Debug only (FREETOKEN_DEBUG_FP8_ACT_STATS=<file>); see fp8_act_stats.py.
            # Prefill only: a flush syncs the accumulators to the host, which is illegal
            # inside CUDA graph capture (the engine captures decode batches).
            from .fp8_act_stats import flush

            flush()
        return logits


__all__ = ["NemotronHForCausalLM"]
