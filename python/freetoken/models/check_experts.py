"""``python -m freetoken.models.check_experts <checkpoint_dir>``

S8's model-conformance check (goal G4: adding a model must be easy and
checkable): can this checkpoint's routed NVFP4 experts actually be read by
this tree, and how would the engine serve them?

Reads ``config.json`` and the safetensors shard HEADERS only -- never a
tensor's data, never the GPU -- so it runs in seconds on a 16 GiB checkpoint.
It either prints a report or refuses with the exact reason (a
:class:`CheckFailed`, whose message names the offending kind / shape / shard).

This is the sequence a model needs to pass before it can serve NVFP4 routed
experts; ``docs/models.md`` writes it out as the "adding a model" checklist.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from dataclasses import dataclass, field

import torch

from freetoken.models.nvfp4_banks import (
    Nvfp4ExpertSourceSpec,
    Nvfp4RowLayout,
    _canon_kind,
    _num_moe_layers,
    expert_source_spec,
    nvfp4_expert_row_layout,
)
from freetoken.layers.quantization import set_quant_config
from freetoken.models.register import _load_attr, checkpoint_quant_config, get_model_spec
from freetoken.moe.offload_cache import _BANK_SCHEMAS
from freetoken.utils import cached_load_hf_config

# safetensors on-disk dtype string for each checkpoint_dtype this module cares about
# (the six the NVFP4 row layout ever names: packed codes, block scale, global scale).
_SAFETENSORS_DTYPE = {
    torch.uint8: "U8",
    torch.float8_e4m3fn: "F8_E4M3",
    torch.float32: "F32",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
}


class CheckFailed(Exception):
    """A conformance check refused the checkpoint. ``str(exc)`` is the exact reason."""


def _model_hook(spec, name: str):
    try:
        return _load_attr(spec.module, name)
    except AttributeError:
        return None


def resolve_expert_source_spec(model_path: str, config, arch_spec):
    """(spec, mechanism) for this checkpoint, or (None, None) if nothing resolves.

    Two mechanisms coexist in this tree (read from code, not assumed):

    1. The registry hook ``<family>.nvfp4_expert_spec(model_path, config)``,
       resolved the same way the real expert loader resolves it
       (``moe/expert_pieces.py:_model_hook`` via ``models.register.get_model_spec``
       keyed on ``config.architectures[0]``). This is what most families export
       today (qwen3_5_moe, qwen4_exp, minimax_m2, minimax_m3, glm4_moe, gemma4,
       glm5_next) and is preferred because it is what actually loads experts on
       the general (non-mirror) path, and because it can pick a checkpoint's
       quantization dialect at call time (glm5_next: ModelOpt vs compressed-
       tensors; qwen3_5_moe: dialect-aware kind names).
    2. The module-level constant ``NVFP4_EXPERT_SOURCE_SPEC``
       (``models.nvfp4_banks.expert_source_spec``, imported by ``model_type``
       rather than through the registry), which is what the bounded host mirror
       (``moe/mirror_pool.py``) resolves from a bare ``ModelConfig`` -- it has no
       ``model_path`` and no bound quant config, so it cannot call a dialect-aware
       hook. Only nemotron_h and qwen3_5_moe export it today; nemotron_h has no
       ``nvfp4_expert_spec`` hook at all (its experts are loaded straight by
       ``load_nvfp4_expert_sources``), so this is its only mechanism.
    """
    hook = _model_hook(arch_spec, "nvfp4_expert_spec")
    if hook is not None:
        spec = hook(model_path, config)
        if isinstance(spec, Nvfp4ExpertSourceSpec):
            return spec, f"{arch_spec.module}.nvfp4_expert_spec()"
    spec = expert_source_spec(config)
    if spec is not None:
        model_type = getattr(config, "model_type", "") or ""
        return spec, f"freetoken.models.{model_type}.weight.NVFP4_EXPERT_SOURCE_SPEC"
    return None, None


def _read_shard_header(path: str) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


@dataclass
class ExpertConformanceReport:
    model_path: str
    architecture: str
    model_type: str
    mechanism: str
    spec_desc: str
    gated: bool
    hidden_size: int
    moe_intermediate_size: int
    num_experts: int
    num_moe_layers: int
    row_bytes: int
    total_bank_bytes: int
    num_experts_checked: int
    cache_type: str
    pin_prefix_honoured: bool
    arena_supports_format: bool
    layout: Nvfp4RowLayout = field(repr=False)

    def render(self) -> str:
        gib = 1 << 30
        lines = [
            f"OK  {self.model_path}",
            f"    architecture       {self.architecture}  (model_type={self.model_type!r})",
            f"    expert source spec {self.spec_desc!r}  via {self.mechanism}",
            f"    gated              {self.gated}",
            f"    H={self.hidden_size} I={self.moe_intermediate_size} "
            f"E={self.num_experts} moe_layers={self.num_moe_layers}",
            f"    experts checked    {self.num_experts_checked} "
            f"(= {self.num_moe_layers} layers x {self.num_experts} experts), "
            "one shard each, tensors exactly as expected",
            f"    row bytes          {self.row_bytes:,} "
            f"({self.row_bytes / (1 << 20):.3f} MiB/expert)",
            f"    total bank bytes   {self.total_bank_bytes:,} "
            f"({self.total_bank_bytes / gib:.3f} GiB)",
            f"    cache type         {self.cache_type} "
            f"(--cache-type radix requested; the engine resolves hybrid_radix for any "
            "model with a linear-attention group)",
            f"    --pin-prefix-*     {'honoured' if self.pin_prefix_honoured else 'ignored (cache is not hybrid_radix)'}",
            f"    expert arena       {'supports' if self.arena_supports_format else 'does NOT support'} nvfp4",
        ]
        return "\n".join(lines)


def check_expert_tensors(
    model_path: str,
    spec: Nvfp4ExpertSourceSpec,
    config,
    *,
    mechanism: str,
) -> tuple[Nvfp4RowLayout, dict, int, int]:
    """The header-only tensor conformance core, given an already-resolved spec.

    Split out of :func:`check_experts` so the S8 parametrised test
    (``tests/models/test_expert_source_conformance.py``) can run this exact
    logic -- key-pattern matching, ``kind_map`` canonicalisation, the expected-
    tensor-set / shape / one-shard-per-expert checks, all against
    :func:`nvfp4_expert_row_layout` -- for every model family's spec against a
    small synthetic checkpoint, without needing a full, family-specific HF
    ``config.json`` fixture for each one (:func:`check_experts` still builds
    that real config for the two model families exercised end to end against
    real checkpoints on disk).

    Returns ``(layout, per_expert, num_moe_layers, expected_experts)``.
    """
    gated = bool(getattr(config, "expert_gated", True))
    if spec.gated != gated:
        raise CheckFailed(
            f"{model_path}: expert source spec gated={spec.gated} disagrees with "
            f"config.expert_gated={gated} ({mechanism})"
        )

    H = getattr(config, spec.hidden_size_attr) if spec.hidden_size_attr else config.hidden_size
    I = config.moe_intermediate_size
    E = config.num_experts
    num_moe_layers = _num_moe_layers(config)
    layout = nvfp4_expert_row_layout(H, I, gated=spec.gated, kind_map=spec.kind_map)

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        raise CheckFailed(
            f"{model_path}: no model.safetensors.index.json "
            "(a sharded NVFP4 checkpoint must ship one)"
        )
    with open(index_path, encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    # (bank_layer, expert) -> {(role, kind): (checkpoint_key, header_entry)}
    per_expert: dict[tuple[int, int], dict[tuple[str, str], tuple[str, dict]]] = {}
    shard_of_expert: dict[tuple[int, int], set[str]] = {}
    shard_headers: dict[str, dict] = {}

    for name, shard in weight_map.items():
        match = spec.key_pattern.match(name)
        if match is None:
            continue
        layer = int(match.group("layer"))
        bank_layer = spec.layer_to_bank(layer, config)
        if bank_layer is None:
            continue
        if not (0 <= bank_layer < num_moe_layers):
            raise CheckFailed(
                f"{model_path}: {name!r} maps to bank layer {bank_layer}, "
                f"outside [0, {num_moe_layers})"
            )
        expert = int(match.group("expert"))
        proj = match.group("proj")
        if proj not in spec.proj_to_role:
            raise CheckFailed(f"{model_path}: unknown expert projection {proj!r} in {name!r}")
        role = spec.proj_to_role[proj]
        raw_kind = match.group("kind")
        kind = _canon_kind(spec, raw_kind)
        if (role, kind) not in layout.tensors:
            expected = sorted(k for r, k in layout.tensors if r == role)
            raise CheckFailed(
                f"{model_path}: {name!r} canonicalises to tensor kind {kind!r} "
                f"(on-disk {raw_kind!r}) for role {role!r}, which the NVFP4 row "
                f"layout does not expect; expected one of {expected}"
            )
        key = (bank_layer, expert)
        shard_of_expert.setdefault(key, set()).add(shard)
        if shard not in shard_headers:
            shard_headers[shard] = _read_shard_header(os.path.join(model_path, shard))
        meta = shard_headers[shard].get(name)
        if meta is None:
            raise CheckFailed(f"{model_path}: {name!r} is in the index but not in {shard!r}'s header")
        per_expert.setdefault(key, {})[(role, kind)] = (name, meta)

    if not per_expert:
        raise CheckFailed(
            f"{model_path}: no checkpoint tensor matched {spec.desc}'s key pattern "
            f"{spec.key_pattern.pattern!r}"
        )

    expected_tensors = set(layout.tensors)
    for key in sorted(per_expert):
        present = per_expert[key]
        bank_layer, expert = key
        missing = expected_tensors - set(present)
        extra = set(present) - expected_tensors
        if missing or extra:
            parts = []
            if missing:
                parts.append(f"missing {sorted(missing)}")
            if extra:
                parts.append(f"unexpected {sorted(extra)}")
            raise CheckFailed(
                f"{model_path}: layer {bank_layer} expert {expert} has {' and '.join(parts)}"
            )
        for (role, kind), (name, meta) in present.items():
            expect = layout.tensors[(role, kind)]
            got_shape = tuple(meta["shape"])
            if got_shape != expect.checkpoint_shape:
                raise CheckFailed(
                    f"{model_path}: {name!r} has shape {got_shape}, "
                    f"expected {expect.checkpoint_shape}"
                )
            want_dtype = _SAFETENSORS_DTYPE.get(expect.checkpoint_dtype)
            if want_dtype is not None and meta.get("dtype") != want_dtype:
                raise CheckFailed(
                    f"{model_path}: {name!r} has dtype {meta.get('dtype')!r}, expected {want_dtype!r}"
                )
        shards = shard_of_expert[key]
        if len(shards) != 1:
            raise CheckFailed(
                f"{model_path}: layer {bank_layer} expert {expert} is split across "
                f"shards {sorted(shards)} (expected exactly one shard per expert)"
            )

    expected_experts = num_moe_layers * E
    if len(per_expert) != expected_experts:
        raise CheckFailed(
            f"{model_path}: found {len(per_expert)} (layer, expert) pairs with expert "
            f"tensors, expected {num_moe_layers} layers x {E} experts = {expected_experts}"
        )

    return layout, per_expert, num_moe_layers, expected_experts


def check_experts(model_path: str) -> ExpertConformanceReport:
    hf_config = cached_load_hf_config(model_path)
    architectures = list(getattr(hf_config, "architectures", None) or [])
    if not architectures:
        raise CheckFailed(f"{model_path}: config.json has no `architectures`")
    architecture = architectures[0]
    try:
        arch_spec = get_model_spec(architecture)
    except ValueError as exc:
        raise CheckFailed(str(exc)) from exc

    # Several families' nvfp4_expert_spec() hooks (qwen3_5_moe: dialect-aware kind
    # names) read the process-global QuantConfig (layers/quantization/configs) that
    # EngineConfig.model_config normally installs before any weight reader runs
    # (engine/config.py:264-267). This tool is not the engine, so it installs the
    # checkpoint's own QuantConfig the same way, read-only, before resolving the
    # spec -- otherwise those hooks raise "no QuantConfig installed".
    parse_config = _load_attr(arch_spec.module, arch_spec.parse_config)
    config = parse_config(hf_config)
    set_quant_config(checkpoint_quant_config(model_path, hf_config, arch_spec))

    spec, mechanism = resolve_expert_source_spec(model_path, config, arch_spec)
    if spec is None:
        raise CheckFailed(
            f"{model_path}: no NVFP4 expert source spec found for architecture "
            f"{architecture!r} (model_type {getattr(config, 'model_type', '?')!r}); "
            f"export nvfp4_expert_spec(model_path, config) from "
            f"{arch_spec.module}, or NVFP4_EXPERT_SOURCE_SPEC from "
            f"freetoken.models.<model_type>.weight"
        )

    H = getattr(config, spec.hidden_size_attr) if spec.hidden_size_attr else config.hidden_size
    I = config.moe_intermediate_size
    E = config.num_experts

    layout, per_expert, num_moe_layers, expected_experts = check_expert_tensors(
        model_path, spec, config, mechanism=mechanism
    )

    # The cache type the engine would resolve (Nemotron and Qwen3.5/Ornith both build
    # a LinearGatedDeltaGroupConfig -- Mamba-2 / gated-delta-net layers -- so both
    # resolve hybrid_radix, not radix; verified from engine.engine._resolve_cache_type,
    # not assumed). --pin-prefix-* is honoured only when the resolved cache is hybrid
    # (scheduler/cache.py: CacheManager.pin_prefix_enabled == is_hybrid and
    # pin_prefix_min_tokens > 0); this reports whether the *mechanism* is available,
    # not whether a specific --pin-prefix-min-tokens value was passed.
    from freetoken.engine.engine import _resolve_cache_type

    has_linear_attention = bool(getattr(config, "has_linear_attention", False))
    cache_type = _resolve_cache_type(has_linear_attention, "radix")
    pin_prefix_honoured = cache_type == "hybrid_radix"

    arena_supports_format = "nvfp4" in _BANK_SCHEMAS and set(_BANK_SCHEMAS["nvfp4"]) == set(
        layout.bank_shapes
    )

    return ExpertConformanceReport(
        model_path=model_path,
        architecture=architecture,
        model_type=getattr(config, "model_type", "") or "",
        mechanism=mechanism,
        spec_desc=spec.desc,
        gated=spec.gated,
        hidden_size=H,
        moe_intermediate_size=I,
        num_experts=E,
        num_moe_layers=num_moe_layers,
        row_bytes=layout.row_bytes,
        total_bank_bytes=layout.row_bytes * expected_experts,
        num_experts_checked=len(per_expert),
        cache_type=cache_type,
        pin_prefix_honoured=pin_prefix_honoured,
        arena_supports_format=arena_supports_format,
        layout=layout,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m freetoken.models.check_experts",
        description="Model conformance check: can this checkpoint's NVFP4 routed "
        "experts be read by this tree, and how would the engine serve them?",
    )
    parser.add_argument("checkpoint_dir")
    args = parser.parse_args(argv)

    try:
        report = check_experts(args.checkpoint_dir)
    except CheckFailed as exc:
        print(f"REFUSED {exc}", file=sys.stderr)
        return 1
    print(report.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
