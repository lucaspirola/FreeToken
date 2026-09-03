"""Per-layer numerical parity between FreeToken's Nemotron-H layers and HuggingFace.

For each requested layer this builds two implementations of the *same* block from the
*same* checkpoint tensors and feeds them one identical random hidden-state batch:

  reference   ``transformers.models.nemotron_h.modeling_nemotron_h`` modules with every
              quantized matrix expanded to bf16 (FP8: ``weight * weight_scale``; NVFP4:
              ``freetoken.models.qwen3_5_moe.weight._dequant_nvfp4_weight``), router in
              fp32, exactly as the checkpoint's producer intended it.
  candidate   FreeToken's own ``NemotronHMamba2Mixer`` / ``NemotronHMoE`` /
              ``NemotronHAttention``, driven standalone through a real global
              ``Context``: a real ``LinearStatePool`` for the Mamba scan and a real
              NVFP4 ``OffloadMoeCache`` (native W4A16 banks, Triton fused kernels,
              ``Nvfp4DenseLinear`` shared expert) for the MoE.

The one substitution is the attention *core*: ``ctx.attn_backend`` is an exact fp32
causal-GQA oracle instead of the paged Triton/FlashInfer backend, which needs a KV pool
and page table that only the engine builds. Everything Nemotron-specific about that
layer -- the merged qkv projection, the 32/2 GQA head layout, the deliberate absence of
RoPE, the o_proj quant path -- still goes through FreeToken's own module; only the
softmax itself is the oracle's. The paged kernels are covered by tests/kernels.

Gates (per layer): routed-expert id agreement >= 99.5%, output cosine > 0.999, and a
scaled max-abs error bound. Exits non-zero if any gate fails.

    uv run benchmarks/parity_nemotron_h_layers.py \
        --model ~/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
        --layers 0,1,5 --tokens 512
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import gc
import json
import math
import os
from typing import Any

import torch

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# The Nemotron-3.5-Lightning geometry these defaults target: layer 0 mamba, layer 1 moe,
# layer 5 attention. --layers overrides; the block kind always comes from the config.
DEFAULT_LAYERS = "0,1,5"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF checkpoint directory")
    parser.add_argument("--layers", default=DEFAULT_LAYERS, help="comma-separated layer ids")
    parser.add_argument("--tokens", type=int, default=512, help="sequence length T")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--bank-device",
        default=None,
        help="where the NVFP4 expert source banks live (default: --device)",
    )
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--min-expert-match", type=float, default=0.995)
    parser.add_argument("--max-scaled-err", type=float, default=0.05)
    parser.add_argument("--json", dest="json_out", help="append the report as a JSON line")
    parser.add_argument(
        "--result-md",
        nargs="?",
        const="",
        help="append a markdown section; bare flag writes benchmarks/results/"
        "nemotron35_lightning_5080_<date>.md",
    )
    return parser.parse_args(argv)


def parse_layers(spec: str) -> list[int]:
    layers = [int(piece) for piece in str(spec).split(",") if piece.strip()]
    if not layers or any(layer < 0 for layer in layers):
        raise ValueError(f"invalid layer list: {spec!r}")
    return layers


# ------------------------------------------------------------------- comparison


def comparison_metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    """Cosine, relative L2 and max-abs error scaled by the reference's own magnitude."""
    if reference.shape != actual.shape:
        raise ValueError(f"shape mismatch: {tuple(reference.shape)} vs {tuple(actual.shape)}")
    ref = reference.detach().reshape(-1).float()
    got = actual.detach().reshape(-1).float()
    ref_norm = float(torch.linalg.vector_norm(ref))
    got_norm = float(torch.linalg.vector_norm(got))
    denom = ref_norm * got_norm
    cosine = float(torch.dot(ref, got)) / denom if denom > 0 else float(ref_norm == got_norm)
    diff = (ref - got).abs()
    scale = float(ref.abs().max())
    return {
        "cosine": cosine,
        "max_abs_err": float(diff.max()),
        "scaled_max_abs_err": float(diff.max()) / scale if scale > 0 else 0.0,
        "rel_l2": float(torch.linalg.vector_norm(ref - got)) / ref_norm if ref_norm > 0 else 0.0,
        "ref_max_abs": scale,
    }


def expert_id_agreement(reference: torch.Tensor, actual: torch.Tensor) -> float:
    """Fraction of routed slots that agree, comparing each token's ids as a set.

    Both routers pick ``top_k`` experts with ``sorted=False``, so the order inside a
    token's row carries no meaning; only membership does.
    """
    if reference.shape != actual.shape:
        raise ValueError(f"shape mismatch: {tuple(reference.shape)} vs {tuple(actual.shape)}")
    if reference.numel() == 0:
        return 1.0
    ref = reference.detach().to(torch.int64).cpu()
    got = actual.detach().to(torch.int64).cpu()
    matched = 0
    for ref_row, got_row in zip(ref.tolist(), got.tolist()):
        matched += len(set(ref_row) & set(got_row))
    return matched / float(reference.numel())


def gate_failures(row: dict[str, Any], args: argparse.Namespace) -> list[str]:
    """Turn one layer's metrics into human-readable acceptance failures."""
    failures: list[str] = []
    label = f"layer {row.get('layer')} ({row.get('kind')})"
    cosine = row.get("cosine")
    if cosine is None or not math.isfinite(cosine) or cosine < args.min_cosine:
        failures.append(f"{label}: cosine {cosine!r} is below {args.min_cosine}")
    scaled = row.get("scaled_max_abs_err")
    if args.max_scaled_err is not None and (
        scaled is None or not math.isfinite(scaled) or scaled > args.max_scaled_err
    ):
        failures.append(
            f"{label}: scaled max abs error {scaled!r} exceeds {args.max_scaled_err}"
        )
    match = row.get("expert_match")
    if match is not None and match < args.min_expert_match:
        failures.append(
            f"{label}: routed expert agreement {match:.5f} is below {args.min_expert_match}"
        )
    return failures


def render_table(rows: list[dict[str, Any]]) -> str:
    """Fixed-width parity table for the terminal."""
    header = (
        f"{'layer':>5}  {'kind':<9}  {'cosine':>10}  {'expert ids':>10}  "
        f"{'max|err|':>10}  {'scaled':>9}  {'rel L2':>9}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        match = row.get("expert_match")
        lines.append(
            f"{row.get('layer'):>5}  {str(row.get('kind')):<9}  "
            f"{row.get('cosine', float('nan')):>10.6f}  "
            f"{'-' if match is None else f'{match:>10.5f}'}  "
            f"{row.get('max_abs_err', float('nan')):>10.3e}  "
            f"{row.get('scaled_max_abs_err', float('nan')):>9.3e}  "
            f"{row.get('rel_l2', float('nan')):>9.3e}"
        )
    return "\n".join(lines)


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"## Layer parity vs HuggingFace {report.get('date', '')}".rstrip(),
        "",
        f"Model: `{report.get('model', '')}`  ",
        f"Tokens: {report.get('tokens')}, seed {report.get('seed')}, "
        f"device `{report.get('device')}`",
        "",
        "| Layer | Kind | Cosine | Expert ids | max abs err | scaled | rel L2 |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in report.get("layers", []):
        match = row.get("expert_match")
        lines.append(
            f"| {row.get('layer')} | {row.get('kind')} | {row.get('cosine', float('nan')):.6f} "
            f"| {'—' if match is None else f'{match:.5f}'} "
            f"| {row.get('max_abs_err', float('nan')):.3e} "
            f"| {row.get('scaled_max_abs_err', float('nan')):.3e} "
            f"| {row.get('rel_l2', float('nan')):.3e} |"
        )
    failures = report.get("failures") or []
    lines += ["", f"Result: {'PASS' if report.get('accepted') else '**FAIL**'}"]
    if failures:
        lines += [""] + [f"- {failure}" for failure in failures]
    lines.append("")
    return "\n".join(lines)


def default_result_path() -> str:
    date = datetime.date.today().isoformat()
    return os.path.join(RESULTS_DIR, f"nemotron35_lightning_5080_{date}.md")


def write_result_markdown(path: str, text: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as handle:
        handle.write(text if text.endswith("\n") else text + "\n")
    return path


# ------------------------------------------------------------------- checkpoint


class Checkpoint:
    """Shard-aware safetensors reader keyed by the checkpoint's own tensor names."""

    def __init__(self, path: str):
        self.path = path
        index_path = os.path.join(path, "model.safetensors.index.json")
        with open(index_path) as handle:
            self.weight_map: dict[str, str] = json.load(handle)["weight_map"]
        self._handles: dict[str, Any] = {}

    def _handle(self, shard: str):
        from safetensors import safe_open

        if shard not in self._handles:
            self._handles[shard] = safe_open(
                os.path.join(self.path, shard), framework="pt", device="cpu"
            )
        return self._handles[shard]

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def get(self, name: str, device: str | torch.device = "cpu") -> torch.Tensor:
        shard = self.weight_map[name]
        return self._handle(shard).get_tensor(name).to(device)

    def module(self, prefix: str, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
        """Every tensor under ``prefix`` keyed by its name relative to the prefix."""
        head = prefix + "."
        return {
            name[len(head) :]: self.get(name, device)
            for name in self.weight_map
            if name.startswith(head)
        }

    def close(self) -> None:
        self._handles.clear()


def dequant_fp8(weight: torch.Tensor, weight_scale: torch.Tensor) -> torch.Tensor:
    """modelopt per-tensor FP8: ``w = code * scale`` (the scale is one scalar per tensor)."""
    return weight.to(torch.float32).mul(weight_scale.reshape(()).to(torch.float32)).to(
        torch.bfloat16
    )


def dequant_nvfp4(parts: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    """W4A16 NVFP4 (group 16) -> bf16, via FreeToken's own dequant kernel."""
    from freetoken.models.qwen3_5_moe.weight import _dequant_nvfp4_weight

    return _dequant_nvfp4_weight(
        parts["weight"].to(device),
        parts["weight_scale"].to(device),
        parts["weight_scale_2"].to(device),
    )


def nvfp4_global(parts: dict[str, torch.Tensor], rows: int) -> torch.Tensor:
    """``weight_scale_2`` (per-tensor) expanded to the per-output-row fp16 vector the
    FreeToken NVFP4 kernels read (mirrors ``qwen3_5_moe/weight.py::_nvfp4_parts``)."""
    return parts["weight_scale_2"].reshape(1).to(torch.float16).expand(rows).contiguous()


# ------------------------------------------------------------- FreeToken harness


class ExactAttnBackend:
    """fp32 causal GQA oracle standing in for the paged attention backend."""

    def __init__(self, num_q_heads: int, num_kv_heads: int, head_dim: int):
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

    def forward(self, q, k, v, layer_id, batch):  # noqa: ARG002 - backend signature
        tokens = q.shape[0]
        key = k.reshape(tokens, self.num_kv_heads, self.head_dim)
        value = v.reshape(tokens, self.num_kv_heads, self.head_dim)
        repeat = self.num_q_heads // self.num_kv_heads
        key = key.repeat_interleave(repeat, dim=1)
        value = value.repeat_interleave(repeat, dim=1)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.einsum("thd,shd->hts", q.float(), key.float()) * scale
        causal = torch.ones(tokens, tokens, dtype=torch.bool, device=q.device).tril()
        scores = scores.masked_fill(~causal, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out = torch.einsum("hts,shd->thd", probs, value.float())
        return out.to(q.dtype)


def fresh_context(**kwargs):
    """A global Context for this process; benchmarks own the whole process."""
    import freetoken.core as core
    from freetoken.core import Context, set_global_ctx

    core._GLOBAL_CTX = None  # bench-only: each layer scenario gets its own context
    ctx = Context(page_size=1, **kwargs)
    set_global_ctx(ctx)
    return ctx


def prefill_batch(tokens: int, slot: int):
    """A one-request prefill batch; SimpleNamespace because ``Req.extend_len`` is derived."""
    from types import SimpleNamespace

    from freetoken.core import Batch

    req = SimpleNamespace(
        extend_len=tokens,
        cached_len=0,
        table_idx=0,
        linear_slot_idx=slot,
        mamba_last_track_seqlen=None,
        mamba_ping_pong=None,
        mamba_next_track_idx=0,
    )
    batch = Batch(reqs=[req], phase="prefill")
    batch.padded_reqs = batch.reqs
    return batch


def load_module(module, tensors: dict[str, torch.Tensor], device: torch.device) -> None:
    """Load checkpoint tensors into a BaseOP, matching each buffer's dtype and shape.

    A per-tensor scale in the checkpoint (FP8 ``weight_scale``, NVFP4 ``weight_scale_2``)
    is a scalar; FreeToken's buffers are per-output-row, so a single value broadcasts.
    """
    payload: dict[str, torch.Tensor] = {}
    reference = module.state_dict()
    for name, want in reference.items():
        source = name
        if name == "weight_global" and "weight_global" not in tensors:
            source = "weight_scale_2"  # NVFP4 dense: per-tensor global -> per-row vector
        item = tensors[source]
        if item.numel() == 1 and want.numel() > 1:
            item = item.reshape(1).expand(want.numel()).reshape(want.shape)
        payload[name] = item.to(device=device, dtype=want.dtype).contiguous()
    for name, item in tensors.items():
        # modelopt's calibrated FP8 activation scale is not a declared buffer, so it never
        # appears in state_dict(); Fp8PerTensorLinear pops it by name during the load.
        if name.rsplit(".", 1)[-1] == "input_scale":
            payload[name] = item.to(device)
    module.load_state_dict(payload)


# ----------------------------------------------------------------------- layers


def hf_config(model_path: str):
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_path)
    config._attn_implementation = "eager"
    return config


def freetoken_config(config, *, moe_backend: str):
    from freetoken.models.nemotron_h.config import parse_config

    return dataclasses.replace(parse_config(config), moe_backend=moe_backend)


def random_hidden(tokens: int, hidden: int, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return (
        torch.randn(tokens, hidden, generator=generator, dtype=torch.float32).to(
            device=device, dtype=torch.bfloat16
        )
    )


def run_mamba_layer(
    checkpoint: Checkpoint, hf_cfg, ft_cfg, layer: int, hidden: torch.Tensor, device
) -> dict[str, Any]:
    from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHMamba2Mixer

    from freetoken.kvcache.linear_state_pool import LinearStatePool
    from freetoken.models.nemotron_h.model import NemotronHMamba2Mixer as FtMixer
    from freetoken.utils.torch_utils import torch_dtype

    prefix = f"backbone.layers.{layer}.mixer"
    raw = checkpoint.module(prefix)

    # ---- HuggingFace reference, every quantized matrix expanded to bf16.
    with torch.device(device), torch_dtype(torch.bfloat16):
        reference = NemotronHMamba2Mixer(
            hf_cfg, layer_idx=layer, initialize_mixer_weights=False
        )
    state = {
        "conv1d.weight": raw["conv1d.weight"],
        "conv1d.bias": raw["conv1d.bias"],
        "dt_bias": raw["dt_bias"],
        "A_log": raw["A_log"],
        "D": raw["D"],
        "norm.weight": raw["norm.weight"],
        "in_proj.weight": dequant_fp8(raw["in_proj.weight"], raw["in_proj.weight_scale"]),
        "out_proj.weight": dequant_fp8(raw["out_proj.weight"], raw["out_proj.weight_scale"]),
    }
    reference.load_state_dict(
        {key: value.to(device=device, dtype=torch.bfloat16) for key, value in state.items()},
        strict=True,
    )
    # FreeToken keeps the scan's scalar parameters in fp32 (the checkpoint stores them
    # bf16); upcast the reference's too so the comparison isolates the layer math.
    for name in ("dt_bias", "A_log", "D"):
        getattr(reference, name).data = raw[name].to(device=device, dtype=torch.float32)
    reference.eval()
    with torch.inference_mode():
        expected = reference(hidden.unsqueeze(0))[0]
    del reference
    free_device(device)

    # ---- FreeToken layer, driven through a real Context + LinearStatePool.
    group = ft_cfg.linear_attention_group()
    pool = LinearStatePool(
        group=group, num_slots=4, dtype=torch.bfloat16, device=torch.device(device), tp_size=1
    )
    ctx = fresh_context(linear_state_pool=pool)
    with torch.device(device), torch_dtype(torch.bfloat16):
        mixer = FtMixer(ft_cfg, layer)
    load_module(mixer, raw, torch.device(device))
    batch = prefill_batch(hidden.shape[0], slot=1)
    with ctx.forward_batch(batch), torch.inference_mode():
        actual = mixer.forward(hidden)
    del mixer, pool
    free_device(device)
    return {"expected": expected, "actual": actual}


def run_attention_layer(
    checkpoint: Checkpoint, hf_cfg, ft_cfg, layer: int, hidden: torch.Tensor, device
) -> dict[str, Any]:
    from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHAttention

    from freetoken.models.nemotron_h.model import NemotronHAttention as FtAttention
    from freetoken.utils.torch_utils import torch_dtype

    prefix = f"backbone.layers.{layer}.mixer"
    raw = checkpoint.module(prefix)
    tokens = hidden.shape[0]

    def projection(name: str) -> torch.Tensor:
        parts = {key[len(name) + 1 :]: value for key, value in raw.items() if key.startswith(name + ".")}
        if "weight_scale_2" in parts:
            return dequant_nvfp4(parts, torch.device(device)).to(torch.bfloat16)
        if "weight_scale" in parts:
            return dequant_fp8(parts["weight"], parts["weight_scale"])
        return parts["weight"].to(torch.bfloat16)

    with torch.device(device), torch_dtype(torch.bfloat16):
        reference = NemotronHAttention(hf_cfg, layer_idx=layer)
    reference.load_state_dict(
        {
            f"{name}.weight": projection(name).to(device=device, dtype=torch.bfloat16)
            for name in ("q_proj", "k_proj", "v_proj", "o_proj")
        },
        strict=True,
    )
    reference.eval()
    mask = torch.zeros(1, 1, tokens, tokens, device=device, dtype=torch.bfloat16)
    mask.masked_fill_(
        ~torch.ones(tokens, tokens, dtype=torch.bool, device=device).tril(), float("-inf")
    )
    with torch.inference_mode():
        expected = reference(hidden.unsqueeze(0), attention_mask=mask)[0][0]
    del reference
    free_device(device)

    ctx = fresh_context()
    ctx.attn_backend = ExactAttnBackend(
        ft_cfg.num_qo_heads, ft_cfg.num_kv_heads, ft_cfg.head_dim
    )
    with torch.device(device), torch_dtype(torch.bfloat16):
        attention = FtAttention(ft_cfg, layer)
    # FreeToken merges q/k/v into one column-parallel projection.
    merged = torch.cat(
        [raw["q_proj.weight"], raw["k_proj.weight"], raw["v_proj.weight"]], dim=0
    )
    load_module(
        attention.qkv_proj, {"weight": merged}, torch.device(device)
    )
    load_module(
        attention.o_proj,
        {key[len("o_proj.") :]: value for key, value in raw.items() if key.startswith("o_proj.")},
        torch.device(device),
    )
    batch = prefill_batch(tokens, slot=1)
    with ctx.forward_batch(batch), torch.inference_mode():
        actual = attention.forward(hidden)
    del attention
    free_device(device)
    return {"expected": expected, "actual": actual}


def _expert_banks(
    checkpoint: Checkpoint, layer: int, num_experts: int, bank_device: torch.device
) -> dict[str, list[torch.Tensor]]:
    """Stack the checkpoint's per-expert NVFP4 rows into the offload cache's bank layout.

    Nemotron's experts are ungated (up + down only), so ``gate_up_*`` carries exactly
    ``intermediate_size`` rows rather than ``2 * intermediate_size``.
    """
    banks: dict[str, list[torch.Tensor]] = {}
    packed, scales, globals_ = [], [], []
    down_packed, down_scales, down_globals = [], [], []
    for expert in range(num_experts):
        base = f"backbone.layers.{layer}.mixer.experts.{expert}"
        up = checkpoint.module(f"{base}.up_proj")
        down = checkpoint.module(f"{base}.down_proj")
        packed.append(up["weight"])
        scales.append(up["weight_scale"])
        globals_.append(nvfp4_global(up, up["weight"].shape[0]))
        down_packed.append(down["weight"])
        down_scales.append(down["weight_scale"])
        down_globals.append(nvfp4_global(down, down["weight"].shape[0]))
    stack = lambda rows: torch.stack(rows).to(bank_device).contiguous()  # noqa: E731
    banks["gate_up_packed"] = [stack(packed)]
    banks["gate_up_scale"] = [stack(scales)]
    banks["gate_up_global"] = [stack(globals_)]
    banks["down_packed"] = [stack(down_packed)]
    banks["down_scale"] = [stack(down_scales)]
    banks["down_global"] = [stack(down_globals)]
    return banks


def run_moe_layer(
    checkpoint: Checkpoint,
    hf_cfg,
    ft_cfg,
    layer: int,
    hidden: torch.Tensor,
    device,
    bank_device,
) -> dict[str, Any]:
    from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHMoE

    from freetoken.models.nemotron_h.model import NemotronHMoE as FtMoE
    from freetoken.moe.offload_cache import OffloadMoeCache
    from freetoken.utils.torch_utils import torch_dtype

    prefix = f"backbone.layers.{layer}.mixer"
    num_experts = int(ft_cfg.num_experts)
    router = checkpoint.module(f"{prefix}.gate")
    shared_up = checkpoint.module(f"{prefix}.shared_experts.up_proj")
    shared_down = checkpoint.module(f"{prefix}.shared_experts.down_proj")

    # ---- HuggingFace reference: all 128 experts expanded to bf16 (~2.5 GiB), one at a
    # time straight into the stacked parameter so the peak stays at the final tensor.
    with torch.device(device), torch_dtype(torch.bfloat16):
        reference = NemotronHMoE(hf_cfg, layer_idx=layer)
    reference.eval()
    with torch.no_grad():
        # The reference router stays fp32, exactly as the checkpoint stores it: any
        # routing divergence FreeToken's bf16 router weight causes must show up in the
        # expert-id agreement number rather than be cancelled out here.
        reference.gate.weight = torch.nn.Parameter(
            router["weight"].to(device=device, dtype=torch.float32), requires_grad=False
        )
        reference.gate.e_score_correction_bias = router["e_score_correction_bias"].to(
            device=device, dtype=torch.float32
        )
        reference.shared_experts.up_proj.weight.copy_(
            dequant_nvfp4(shared_up, torch.device(device))
        )
        reference.shared_experts.down_proj.weight.copy_(
            dequant_nvfp4(shared_down, torch.device(device))
        )
        for expert in range(num_experts):
            base = f"{prefix}.experts.{expert}"
            up = checkpoint.module(f"{base}.up_proj")
            down = checkpoint.module(f"{base}.down_proj")
            reference.experts.up_proj[expert].copy_(dequant_nvfp4(up, torch.device(device)))
            reference.experts.down_proj[expert].copy_(dequant_nvfp4(down, torch.device(device)))
            del up, down
    with torch.inference_mode():
        _, _, reference_ids = reference.gate(hidden.unsqueeze(0))
        expected = reference(hidden.unsqueeze(0))[0]
    reference_ids = reference_ids.detach().clone()
    del reference
    free_device(device)

    # ---- FreeToken layer: native NVFP4 banks through the real offload cache.
    banks = _expert_banks(checkpoint, layer, num_experts, torch.device(bank_device))
    cache = OffloadMoeCache(
        num_layers=1,
        num_experts=num_experts,
        cache_size=num_experts,
        device=torch.device(device),
        quant_format="nvfp4",
    )
    cache.set_bank_sources(banks)
    ctx = fresh_context(moe_offload_cache=cache)
    with torch.device(device), torch_dtype(torch.bfloat16):
        moe = FtMoE(ft_cfg, layer, 0)
    moe.experts.offload_cache = cache
    load_module(moe.gate, router, torch.device(device))
    load_module(moe.shared_experts.up_proj, shared_up, torch.device(device))
    load_module(moe.shared_experts.down_proj, shared_down, torch.device(device))
    batch = prefill_batch(hidden.shape[0], slot=1)
    with ctx.forward_batch(batch), torch.inference_mode():
        scores, choice = moe.gate.forward(hidden)
        actual_ids = torch.topk(choice, moe.top_k, dim=-1, sorted=False).indices
        actual = moe.forward(hidden)
    del moe, cache, banks
    free_device(device)
    return {
        "expected": expected,
        "actual": actual,
        "expected_ids": reference_ids.reshape(-1, reference_ids.shape[-1]),
        "actual_ids": actual_ids,
    }


def free_device(device) -> None:
    gc.collect()
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()


LAYER_RUNNERS = {
    "mamba": run_mamba_layer,
    "attention": run_attention_layer,
    "moe": run_moe_layer,
}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from freetoken.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)

    device = torch.device(args.device)
    bank_device = torch.device(args.bank_device or args.device)
    torch.manual_seed(args.seed)
    checkpoint = Checkpoint(args.model)
    config = hf_config(args.model)
    ft_cfg = freetoken_config(config, moe_backend="offload")
    kinds = ft_cfg.nemotron_h_args.layer_types

    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for layer in parse_layers(args.layers):
        kind = kinds[layer]
        runner = LAYER_RUNNERS[kind]
        hidden = random_hidden(args.tokens, ft_cfg.hidden_size, args.seed + layer, device)
        print(f"[parity] layer {layer} ({kind}) ...", flush=True)
        extra = {"bank_device": bank_device} if kind == "moe" else {}
        result = runner(checkpoint, config, ft_cfg, layer, hidden, device, **extra)
        row: dict[str, Any] = {"layer": layer, "kind": kind}
        row.update(comparison_metrics(result["expected"], result["actual"]))
        if "expected_ids" in result:
            row["expert_match"] = expert_id_agreement(
                result["expected_ids"], result["actual_ids"]
            )
        rows.append(row)
        failures += gate_failures(row, args)
        del result, hidden
        free_device(device)
    checkpoint.close()

    report = {
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "tokens": args.tokens,
        "seed": args.seed,
        "device": str(device),
        "layers": rows,
        "failures": failures,
        "accepted": not failures,
    }
    print()
    print(render_table(rows))
    print()
    for failure in failures:
        print(f"  - {failure}")
    print(f"[parity] overall: {'PASS' if report['accepted'] else 'FAIL'}", flush=True)
    if args.json_out:
        with open(args.json_out, "a") as handle:
            handle.write(json.dumps(report) + "\n")
    if args.result_md is not None:
        path = args.result_md or default_result_path()
        write_result_markdown(path, render_markdown(report))
        print(f"[parity] wrote {path}", flush=True)
    return 0 if report["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
