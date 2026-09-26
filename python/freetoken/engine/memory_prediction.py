"""Startup VRAM prediction from the model config and the serving flags.

``--memory-ratio`` is 1.00 by default: the engine takes all of the free VRAM and
then reserves what the runtime needs. For growable KV, the arena is parked, the
prefill transient is measured, and the arena is filled back. Without growable KV,
the reserve comes out of the startup budget. This module predicts those runtime
needs from the config alone, before anything is allocated. The engine logs the
prediction next to what it later measures and warns when the two disagree. A new
model or GPU whose measurement leaves the prediction's band is then visible at its
first start and does not show up only as a long prompt's OOM.

Every term is a closed form in the config's geometry: hidden size, heads, head
dims, expert top-k and intermediate sizes, the linear mixers' state geometry and
the prefill chunk. Model-specific geometry reaches it only through the
onboarding interface:

* ``ModelConfig.attention_groups``, which each family's ``parse_config`` fills
  (full / SWA attention, GatedDeltaNet ``"kv"`` and Mamba-2 ``"mamba2"`` linear
  groups), plus the MoE / dense-MLP fields every family sets;
* ``ModelSpec.prefill_transient`` (``"module:function"``), an optional family hook
  returning ``{layer_kind: bytes}`` for mixers the generic terms do not model. Its
  kinds replace or add to the generic ones.

An attention-group kind the generic terms do not know, with no hook to price it,
makes the prediction *coarse*. It is still logged, but a non-growable start then
keeps the old 10%-of-free-VRAM reserve as a floor (``runtime_reserve_bytes``).

Calibration (8192-token chunk, bf16 activations): see ``tests/engine/
test_memory_prediction.py`` for the table of predicted vs measured transients the
WARN threshold is chosen from.
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass, field
from math import ceil
from typing import Any, Callable

MiB = 1024 * 1024

# WARN when |predicted - measured| > this fraction of the measured transient. The
# calibration residuals (test_memory_prediction.CALIBRATION) are at most 18% (Ornith
# on the owner's WSL host, where the allocator reserves ~10% above its peak).
# 25% is about 1.4 times that worst case: a new model or GPU whose transient leaves the
# band differs from the modelled kernels by more than allocator noise, which is
# exactly the case to flag.
PREDICTION_WARN_FRACTION = 0.25

# The captured decode graphs share one private memory pool (GraphRunner reuses the
# first graph's pool), sized by the largest batch: a fixed part plus the per-token
# activations of max_bs decode tokens. Every journal on record (Nemotron and Ornith,
# bs=[1], 400+ starts) shows 0.00-0.03 GiB between "before" and "after capturing CUDA
# graphs"; 32 MiB is that bound for the fixed part.
GRAPH_POOL_FIXED_BYTES = 32 * MiB

# The pre-ratio-1.00 implicit reserve: a coarse prediction keeps it as a floor.
COARSE_RESERVE_FRACTION = 0.10

GENERIC_GROUP_KINDS = ("full", "swa", "linear_gated_delta")


@dataclass(frozen=True)
class TransientPrediction:
    """One prefill chunk's predicted transient VRAM.

    ``layers`` maps each layer kind to the bytes live at its own peak (``base``
    included). The transient is the largest, because layers run one after another
    and free their intermediates on return."""

    chunk_tokens: int
    bytes: int
    peak_layer: str
    layers: dict[str, int]
    base: int
    coarse: bool
    unknown_kinds: tuple[str, ...] = ()


@dataclass(frozen=True)
class StartupPrediction:
    transient: TransientPrediction
    linear_state_bytes: int
    kv_floor_bytes: int
    kv_floor_tokens: int
    non_expert_weight_bytes: int | None
    graph_pool_bytes: int
    graph_max_bs: int
    notes: tuple[str, ...] = field(default_factory=tuple)

    def summary(self) -> str:
        from freetoken.utils import mem_GB

        t = self.transient
        parts = [
            f"prefill transient {mem_GB(t.bytes)} on a {t.chunk_tokens}-token chunk "
            f"(peak layer {t.peak_layer}; "
            + ", ".join(f"{k} {mem_GB(v)}" for k, v in sorted(t.layers.items()))
            + (f"; COARSE, unmodelled kinds {list(t.unknown_kinds)}" if t.coarse else "")
            + ")",
            f"graph pool {mem_GB(self.graph_pool_bytes)} (max bs {self.graph_max_bs})",
            f"linear-state pool {mem_GB(self.linear_state_bytes)}",
            f"KV floor {mem_GB(self.kv_floor_bytes)} ({self.kv_floor_tokens} tokens)",
            "non-expert weights "
            + ("n/a" if self.non_expert_weight_bytes is None else mem_GB(self.non_expert_weight_bytes)),
        ]
        return "; ".join(parts)


# ----------------------------------------------------------------------------------
# Prefill transient: per layer kind, the tensors live at that layer's peak.
# ----------------------------------------------------------------------------------


def _attention_bytes(T: int, H: int, nq: int, nkv: int, hd: int, a: int) -> int:
    """Fused qkv projection, the rotated/normed q/k copies, attention output and
    o_proj output."""
    qkv = T * (nq + 2 * nkv) * hd * a
    qk_copies = T * (nq + nkv) * hd * a
    attn_out = T * nq * hd * a
    return qkv + qk_copies + attn_out + T * H * a


def _gdn_bytes(T: int, H: int, g: Any, a: int) -> int:
    """GatedDeltaNet (``state_layout="kv"``), at the chunked scan's peak.

    in_proj output (q|k|v|z|b|a) and the causal conv output stay live. Inside
    ``chunk_gated_delta_rule`` the l2-normed q/k, the intra-chunk ``A`` and its
    inverse, ``w``/``u``, the per-chunk states ``h`` (bf16), ``v_new`` and ``o``
    are live together."""
    kh, vh, dk, dv = g.num_key_heads, g.num_value_heads, g.key_head_dim, g.value_head_dim
    chunk = g.track_chunk_size
    conv_dim = 2 * kh * dk + vh * dv
    in_proj = T * (conv_dim + vh * dv + 2 * vh) * a
    conv_out = T * conv_dim * a
    qk_l2 = 2 * T * kh * dk * a
    a_mats = 2 * T * vh * chunk * a
    w_u = T * vh * (dk + dv) * a
    h = ceil(T / chunk) * vh * dk * dv * a
    v_new_o = 2 * T * vh * dv * a
    return in_proj + conv_out + qk_l2 + a_mats + w_u + h + v_new_o


def _mamba2_bytes(T: int, H: int, g: Any, a: int) -> int:
    """Mamba-2 SSD (``state_layout="mamba2"``: value heads = SSM heads, key_head_dim
    = head dim P, value_head_dim = d_state N, key heads = B/C groups G), at the
    chunked scan's peak.

    in_proj output (gate|x B C|dt), the conv output, the per-chunk fp32 states
    (chunk states plus the passed/intermediate states), the fp32 ``C.B`` blocks,
    and the scan output."""
    nh, P, N, G = g.num_value_heads, g.key_head_dim, g.value_head_dim, g.num_key_heads
    chunk = g.track_chunk_size
    d_inner = nh * P
    conv_dim = d_inner + 2 * G * N
    nc = ceil(T / chunk)
    in_proj = T * (d_inner + conv_dim + nh) * a
    conv_out = T * conv_dim * a
    states = 2 * nc * nh * P * N * 4
    cb = nc * G * chunk * chunk * 4
    out = T * d_inner * a
    return in_proj + conv_out + states + cb + out


def _moe_bytes(T: int, H: int, mc: Any, a: int) -> int:
    """Fused routed-expert prefill (``fused_experts_nvfp4`` and siblings): gemm1's
    ``[T, k, two_i]`` output (activated in its epilogue), gemm2's ``[T, k, H]``
    output and the summed ``[T, H]``, plus the fp32 router logits and the shared
    expert's intermediate and output."""
    k = mc.num_experts_per_tok
    inter = mc.moe_intermediate_size
    two_i = inter if mc.hidden_act == "relu2" else 2 * inter
    routed = T * k * two_i * a + T * k * H * a + T * H * a
    router = T * mc.num_experts * 4 + T * k * 8
    shared = 0
    if mc.shared_expert_intermediate_size:
        si = mc.shared_expert_intermediate_size
        shared = T * (si if mc.hidden_act == "relu2" else 2 * si) * a + T * H * a
    return routed + router + shared


def _dense_mlp_bytes(T: int, H: int, mc: Any, a: int) -> int:
    inter = mc.intermediate_size
    return T * 2 * inter * a + T * inter * a + T * H * a


def _family_hook(model_config: Any) -> Callable[..., dict[str, int]] | None:
    archs = list(getattr(model_config, "architectures", None) or [])
    if not archs:
        return None
    from freetoken.models.register import _load_attr, get_model_spec

    try:
        spec = get_model_spec(archs[0])
    except ValueError:
        return None
    ref = getattr(spec, "prefill_transient", None)
    if not ref:
        return None
    module, _, attr = ref.partition(":")
    return _load_attr(module, attr)


def predict_prefill_transient(
    model_config: Any, chunk_tokens: int, act_bytes: int = 2
) -> TransientPrediction:
    """Predicted transient VRAM of one ``chunk_tokens`` prefill chunk (see module doc)."""
    mc, T, a = model_config, int(chunk_tokens), int(act_bytes)
    H = mc.hidden_size
    # Residual stream and the normed layer input, live across every layer.
    base = 2 * T * H * a
    layers: dict[str, int] = {}
    unknown: list[str] = []
    nq = mc.num_qo_heads
    if mc.num_qo_heads_per_layer:
        nq = max(mc.num_qo_heads_per_layer)
    for g in mc.attention_groups:
        if g.kind in ("full", "swa") and not getattr(g, "mla", False):
            layers[g.kind + "_attention"] = max(
                layers.get(g.kind + "_attention", 0),
                _attention_bytes(T, H, nq, g.num_kv_heads, g.head_dim, a),
            )
        elif g.kind == "linear_gated_delta" and g.state_layout == "kv":
            layers["gdn"] = max(layers.get("gdn", 0), _gdn_bytes(T, H, g, a))
        elif g.kind == "linear_gated_delta" and g.state_layout == "mamba2":
            layers["mamba2"] = max(layers.get("mamba2", 0), _mamba2_bytes(T, H, g, a))
        else:
            unknown.append(g.kind + ("/mla" if getattr(g, "mla", False) else ""))
    if not mc.attention_groups:
        layers["attention"] = _attention_bytes(T, H, nq, mc.num_kv_heads, mc.head_dim, a)
    if mc.num_experts and mc.moe_intermediate_size:
        layers["moe"] = _moe_bytes(T, H, mc, a)
    if mc.intermediate_size and (not mc.num_experts or mc.first_k_dense_replace):
        layers["dense_mlp"] = _dense_mlp_bytes(T, H, mc, a)
    hook = _family_hook(mc)
    if hook is not None:
        extra = hook(mc, T, a)
        layers.update({k: int(v) for k, v in extra.items()})
        # A family hook prices what it declares; groups it covers are no longer unknown.
        unknown = [u for u in unknown if u.split("/")[0] not in extra]
    layers = {k: base + v for k, v in layers.items()}
    peak = max(layers, key=layers.get) if layers else "none"
    return TransientPrediction(
        chunk_tokens=T,
        bytes=layers.get(peak, base),
        peak_layer=peak,
        layers=layers,
        base=base,
        coarse=bool(unknown),
        unknown_kinds=tuple(unknown),
    )


# ----------------------------------------------------------------------------------
# Non-expert weights: the checkpoint's own tensor table (headers only, no data).
# ----------------------------------------------------------------------------------


def _is_routed_expert(name: str) -> bool:
    return ".experts." in name and "shared_expert" not in name


def predict_non_expert_weight_bytes(model_path: str | None) -> int | None:
    """Bytes of every non-routed-expert tensor in the checkpoint's safetensors
    shards, from the shard headers. With expert offload these are what the GPU
    holds as "weights". Returns None for GGUF or unreadable checkpoints. Dtype
    conversions at load (dequantized modules, tied heads) are what the comparison
    with the measured weights shows."""
    if not model_path or not os.path.isdir(model_path):
        return None
    shards = sorted(f for f in os.listdir(model_path) if f.endswith(".safetensors"))
    if not shards:
        return None
    total = 0
    try:
        for shard in shards:
            with open(os.path.join(model_path, shard), "rb") as fh:
                (n,) = struct.unpack("<Q", fh.read(8))
                header = json.loads(fh.read(n))
            for name, meta in header.items():
                if name == "__metadata__" or _is_routed_expert(name):
                    continue
                if any(tok in name for tok in ("visual.", "vision_tower", "audio_tower", "mtp.")):
                    continue
                start, end = meta["data_offsets"]
                total += int(end) - int(start)
    except (OSError, ValueError, KeyError, struct.error):
        return None
    return total


# ----------------------------------------------------------------------------------
# The startup prediction and the reserve a non-growable start takes from the budget.
# ----------------------------------------------------------------------------------


def predict_startup(config: Any, *, act_bytes: int = 2) -> StartupPrediction:
    """Everything the engine predicts before it allocates (see module doc)."""
    from freetoken.kvcache.linear_state_pool import state_pool_bytes

    mc = config.model_config
    transient = predict_prefill_transient(mc, int(config.max_extend_tokens), act_bytes)
    try:
        linear_state = int(state_pool_bytes(config))
    except Exception:  # noqa: BLE001 - a prediction never fails a start
        linear_state = 0
    kv_floor_tokens = int(getattr(config, "kv_grow_step_tokens", 0) or 0)
    kv_floor_bytes = 0
    try:
        from freetoken.kvcache import resolve_pool_class

        per_page, fixed, _tok, _res = resolve_pool_class(mc).kv_cost(config)
        if kv_floor_tokens:
            kv_floor_bytes = int(per_page * ceil(kv_floor_tokens / config.page_size)) + int(fixed)
    except Exception:  # noqa: BLE001
        pass
    # GraphRunner resolves the batch list later (it depends on free VRAM); the
    # prediction takes the largest batch the flags allow.
    graph_bs = getattr(config, "cuda_graph_bs", None)
    max_bs = (
        max(graph_bs) if graph_bs
        else int(getattr(config, "cuda_graph_max_bs", None) or getattr(config, "max_running_req", 1) or 1)
    )
    per_token = transient.bytes // max(1, transient.chunk_tokens)
    graph_pool = GRAPH_POOL_FIXED_BYTES + max_bs * per_token
    return StartupPrediction(
        transient=transient,
        linear_state_bytes=linear_state,
        kv_floor_bytes=kv_floor_bytes,
        kv_floor_tokens=kv_floor_tokens,
        non_expert_weight_bytes=predict_non_expert_weight_bytes(getattr(config, "model_path", None)),
        graph_pool_bytes=graph_pool,
        graph_max_bs=max_bs,
    )


def runtime_reserve_bytes(
    prediction: StartupPrediction, *, growable: bool, baseline_free: int
) -> int:
    """VRAM a start must leave out of the ratio budget for the runtime.

    Growable KV reserves nothing here: its arena is parked while the transient is
    measured and filled back to leave the measured headroom (Engine.
    _settle_prefill_headroom). A fixed-size start cannot give memory back, so it
    reserves one predicted prefill chunk (at least the VMM commit cushion), the
    graph pools and the headroom margin. A coarse prediction keeps the old
    ``(1 - 0.90)`` share as a floor."""
    if growable:
        return 0
    from freetoken.engine.growable_kv import (
        PREFILL_HEADROOM_MARGIN_BYTES,
        growable_headroom_bytes,
    )

    reserve = (
        growable_headroom_bytes(prediction.transient.bytes)
        + prediction.graph_pool_bytes
        + PREFILL_HEADROOM_MARGIN_BYTES
    )
    if prediction.transient.coarse:
        reserve = max(reserve, int(COARSE_RESERVE_FRACTION * baseline_free))
    return int(reserve)


def transient_upper_bound(prediction: "TransientPrediction | None") -> int | None:
    """The largest measured transient this prediction accepts without a WARN.

    ``compare`` flags a measurement when ``(predicted - measured) / measured`` leaves
    ``+-PREDICTION_WARN_FRACTION``, so any in-band measurement is at most
    ``predicted / (1 - PREDICTION_WARN_FRACTION)``. Whatever must be priced before the
    measurement can run (the bounded mirror pool) can take this bound: a model whose
    measurement exceeds it is outside the calibrated band and WARNs at its first start.
    None for a coarse prediction (an unmodelled layer kind): it bounds nothing."""
    if prediction is None or prediction.coarse or prediction.bytes <= 0:
        return None
    return ceil(prediction.bytes / (1.0 - PREDICTION_WARN_FRACTION))


def compare(predicted: int, measured: int) -> tuple[float, bool]:
    """(relative difference vs the measurement, outside the WARN band)."""
    if measured <= 0:
        return 0.0, False
    rel = (predicted - measured) / measured
    return rel, abs(rel) > PREDICTION_WARN_FRACTION


__all__ = [
    "GRAPH_POOL_FIXED_BYTES",
    "PREDICTION_WARN_FRACTION",
    "StartupPrediction",
    "TransientPrediction",
    "compare",
    "predict_non_expert_weight_bytes",
    "predict_prefill_transient",
    "predict_startup",
    "runtime_reserve_bytes",
    "transient_upper_bound",
]
