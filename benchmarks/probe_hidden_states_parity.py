"""End-to-end parity for the Switchyard prefill-probe hidden-state export.

Sends ONE probe request through a running FreeToken server (started with
``--hidden-states-dir``), loads the ``.safetensors`` artifact the response points at,
and compares it against transformers' own ``NemotronHBlock`` modules run in bf16 over the
*same* token ids -- which the artifact carries, so no re-tokenization can drift the two
apart. The reference streams the checkpoint one block at a time (see
:func:`reference_hidden_states`): the released checkpoint is modelopt MIXED_PRECISION,
which transformers cannot load through ``from_pretrained``, and dense bf16 NemotronH is
58.8 GiB.

The served model and the reference do not fit in host RAM at the same time, so the run is
two-phase: ``--capture-only`` against the server, then ``--artifact <path>`` once it is
stopped.

What is compared is what Switchyard actually consumes: the mean over prompt tokens of
each layer's residual vector. Per-layer cosine must exceed ``--min-cosine`` (0.99 by
default), which is loose enough for NVFP4 experts and FP8 projections on the served side
against a bf16 reference, and tight enough that an off-by-one layer index, a final-norm
leak, or a dropped prefill chunk fails loudly.

    The reference records each ``NemotronHBlock``'s own return value through a forward
    hook, which *is* the quantity FreeToken exports (``residual + mixer``, before the next
    block's input norm and before ``norm_f``), so row ``i`` pairs with block ``i`` by
    construction -- no ``output_hidden_states`` off-by-one to reason about, and ``norm_f``
    never enters the comparison.

Run both phases under the GPU lock, against a P1-profile server:

    # phase A, server up
    uv run benchmarks/probe_hidden_states_parity.py \\
        --model ~/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \\
        --base-url http://127.0.0.1:1919 --hidden-states-dir /tmp/ft-hidden-states \\
        --prompt-tokens 300 --capture-only
    # phase B, server stopped
    scripts/gpu_lock.sh uv run benchmarks/probe_hidden_states_parity.py \\
        --model ~/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \\
        --hidden-states-dir /tmp/ft-hidden-states --artifact /tmp/ft-hidden-states/<uuid>.safetensors
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import urllib.error
import urllib.request

import torch

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

#: Deterministic filler; trimmed in token space to exactly --prompt-tokens.
_FILLER = (
    "The scheduler admits a request only when the KV pool, the recurrent state pool and "
    "the sliding-window pool can all seat it. Prefill is chunked; decode is graph "
    "captured. Experts stream over PCIe on a miss and stay resident on a hit. "
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF checkpoint directory")
    parser.add_argument("--base-url", default="http://127.0.0.1:1919")
    parser.add_argument(
        "--served-model-name",
        default=None,
        help="model id to send (default: read it from /v1/models)",
    )
    parser.add_argument(
        "--hidden-states-dir",
        required=True,
        help="the server's --hidden-states-dir, readable from here",
    )
    parser.add_argument("--prompt-tokens", type=int, default=300)
    parser.add_argument(
        "--capture-only",
        action="store_true",
        help="probe the server, keep the artifact, print its path and stop (no reference)",
    )
    parser.add_argument(
        "--artifact",
        default=None,
        help="score an artifact captured earlier instead of probing a server; the host "
        "cannot hold the served model and the CPU reference at once",
    )
    parser.add_argument("--layers", default=None, help="e.g. 0-51 (default: all)")
    parser.add_argument(
        "--reference-dt-min",
        type=float,
        default=0.0,
        help="dt floor for the reference Mamba-2 scan. transformers hard-codes "
        "config.time_step_min (1e-3), which is HF's dt_bias *initializer* range and not a "
        "runtime bound; FreeToken, vLLM and llama.cpp do not clamp. 0.0 matches the served "
        "engine; pass 1e-3 to reproduce transformers as shipped.",
    )
    parser.add_argument(
        "--reference-device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="device for the streamed HF reference (NVFP4 dequant needs CUDA)",
    )
    parser.add_argument("--min-cosine", type=float, default=0.99)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--keep-artifact",
        action="store_true",
        help="do not delete the artifact (Switchyard's reader normally consumes it)",
    )
    parser.add_argument("--json", dest="json_out", help="append the report as a JSON line")
    return parser.parse_args(argv)


# ------------------------------------------------------------------ server side


def _post(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        raise SystemExit(
            f"probe request failed: HTTP {error.code} {error.read().decode()[:400]}"
        ) from error


def _served_model_name(base_url: str, timeout: float) -> str:
    with urllib.request.urlopen(f"{base_url}/v1/models", timeout=timeout) as response:
        data = json.loads(response.read().decode())
    return data["data"][0]["id"]


def build_prompt(tokenizer, target_tokens: int) -> str:
    """A prompt whose *chat-templated* length is close to ``target_tokens``.

    Exactness is not required: the artifact carries the token ids that were actually
    forwarded, and the reference is driven from those.
    """
    text = ""
    while len(tokenizer(text).input_ids) < target_tokens:
        text += _FILLER
    ids = tokenizer(text).input_ids[:target_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def run_probe(args: argparse.Namespace, prompt: str, model_id: str) -> str:
    kv_transfer: dict = {
        "hidden_states_path": args.hidden_states_dir,
        "include_output_tokens": False,
    }
    if args.layers:
        kv_transfer["layer_ids"] = _parse_layer_spec(args.layers)
    response = _post(
        f"{args.base_url.rstrip('/')}/v1/chat/completions",
        {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1,
            "temperature": 0.0,
            "kv_transfer_params": kv_transfer,
        },
        args.timeout,
    )
    params = response.get("kv_transfer_params")
    if not params or not params.get("hidden_states_path"):
        raise SystemExit(
            "response carries no kv_transfer_params.hidden_states_path; is the server "
            "running with --hidden-states-dir?"
        )
    return params["hidden_states_path"]


def _parse_layer_spec(spec: str) -> list[int]:
    if "-" in spec:
        start, stop = spec.split("-", 1)
        return list(range(int(start), int(stop) + 1))
    return [int(piece) for piece in spec.split(",") if piece.strip()]


# --------------------------------------------------------------- reference side


def load_artifact(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    from safetensors import safe_open

    with safe_open(path, framework="pt") as handle:
        keys = sorted(handle.keys())
        if "hidden_states" not in keys:
            raise SystemExit(f"artifact {path} has no hidden_states tensor (keys {keys})")
        hidden = handle.get_tensor("hidden_states")
        if "token_ids" not in keys:
            raise SystemExit(f"artifact {path} has no token_ids tensor (keys {keys})")
        token_ids = handle.get_tensor("token_ids")
    if hidden.dim() != 3:
        raise SystemExit(
            f"hidden_states must be [prompt_tokens, layers, hidden]; got {tuple(hidden.shape)}"
        )
    if hidden.dtype not in (torch.bfloat16, torch.float32):
        raise SystemExit(f"Switchyard accepts only BF16/F32; got {hidden.dtype}")
    if token_ids.dtype is not torch.int64:
        raise SystemExit(f"token_ids must be I64; got {token_ids.dtype}")
    if token_ids.shape[0] != hidden.shape[0]:
        raise SystemExit(
            f"token_ids {tuple(token_ids.shape)} does not match hidden_states token "
            f"count {hidden.shape[0]}"
        )
    return hidden, token_ids


class _Shards:
    """Shard-aware safetensors reader keyed by the checkpoint's own tensor names."""

    def __init__(self, path: str):
        self.path = path
        with open(os.path.join(path, "model.safetensors.index.json")) as handle:
            self.weight_map: dict[str, str] = json.load(handle)["weight_map"]
        self._handles: dict[str, object] = {}

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def get(self, name: str) -> torch.Tensor:
        from safetensors import safe_open

        shard = self.weight_map[name]
        if shard not in self._handles:
            self._handles[shard] = safe_open(
                os.path.join(self.path, shard), framework="pt", device="cpu"
            )
        return self._handles[shard].get_tensor(name)

    def close(self) -> None:
        self._handles.clear()


def _dequant_fp8(shards: _Shards, prefix: str, device) -> torch.Tensor:
    """modelopt per-tensor FP8: ``w = code * scale``."""
    weight = shards.get(f"{prefix}.weight").to(device)
    scale = shards.get(f"{prefix}.weight_scale").reshape(()).to(device)
    return weight.to(torch.float32).mul(scale.to(torch.float32)).to(torch.bfloat16)


def _dequant_nvfp4(shards: _Shards, prefix: str, device) -> torch.Tensor:
    """W4A16 NVFP4 (group 16) -> bf16, through FreeToken's own dequant kernel (CUDA)."""
    from freetoken.models.qwen3_5_moe.weight import _dequant_nvfp4_weight

    return _dequant_nvfp4_weight(
        shards.get(f"{prefix}.weight").to(device),
        shards.get(f"{prefix}.weight_scale").to(device),
        shards.get(f"{prefix}.weight_scale_2").to(device),
    )


def _checkpoint_tensor(shards: _Shards, prefix: str, device) -> torch.Tensor:
    """One HF ``.weight``, dequantized according to the sibling scales the checkpoint has."""
    if not shards.has(f"{prefix}.weight"):
        raise SystemExit(f"checkpoint has no tensor for {prefix}.weight")
    if shards.has(f"{prefix}.weight_scale_2"):
        return _dequant_nvfp4(shards, prefix, device)
    if shards.has(f"{prefix}.weight_scale"):
        return _dequant_fp8(shards, prefix, device)
    return shards.get(f"{prefix}.weight").to(device)


def _layer_state_dict(shards: _Shards, layer: int, keys, device) -> dict[str, torch.Tensor]:
    """Build one ``NemotronHBlock``'s state dict out of the modelopt checkpoint.

    Name mapping: HF calls the trunk ``model.``, the checkpoint calls it ``backbone.``;
    HF stacks the 128 routed experts into one 3-D parameter, the checkpoint stores them
    per expert and NVFP4-packed.
    """
    root = f"backbone.layers.{layer}"
    state: dict[str, torch.Tensor] = {}
    for key in keys:
        if key in ("mixer.experts.up_proj", "mixer.experts.down_proj"):
            which = key.rsplit(".", 1)[1]
            stack = None
            expert = 0
            while shards.has(f"{root}.mixer.experts.{expert}.{which}.weight"):
                tile = _dequant_nvfp4(shards, f"{root}.mixer.experts.{expert}.{which}", device)
                if stack is None:
                    stack = torch.empty(
                        (128, *tile.shape), dtype=torch.bfloat16, device=tile.device
                    )
                stack[expert].copy_(tile)
                del tile
                expert += 1
            if stack is None:
                raise SystemExit(f"no experts under {root}.mixer.experts.*.{which}")
            state[key] = stack[:expert].clone() if expert != 128 else stack
            continue
        name = f"{root}.{key}"
        # The sibling-scale test comes first: a quantized ``.weight`` is present under its
        # own name too (FP8 codes / packed NVFP4 nibbles), so a plain `has()` would load
        # the undecoded tensor.
        if key.endswith(".weight"):
            state[key] = _checkpoint_tensor(shards, name[: -len(".weight")], device)
        elif shards.has(name):
            state[key] = shards.get(name).to(device)
        else:
            raise SystemExit(f"checkpoint has no tensor for {name}")
    return state


def reference_hidden_states(
    model_path: str, token_ids: torch.Tensor, device: str = "cuda", dt_min: float = 0.0
) -> torch.Tensor:
    """``[tokens, layers, hidden]`` of post-block residuals, from transformers' own
    ``NemotronHBlock`` modules with the checkpoint streamed one block at a time.

    ``AutoModelForCausalLM.from_pretrained`` cannot be used on this release: it is a
    modelopt ``MIXED_PRECISION`` checkpoint (FP8 mamba projections, W4A16 NVFP4 experts,
    a ``backbone.`` prefix and per-expert 2-D tensors), transformers 5.x has no
    ``modelopt`` quantizer, and a dense bf16 NemotronH is 58.8 GiB against a 34 GiB host.
    So the model is built on ``meta`` and each block's weights are materialized right
    before it runs and released right after -- transformers still owns the forward
    (masks, position ids, the pure-torch Mamba-2 scan, the residual adds), and the
    per-block output hook records exactly what FreeToken exports: ``residual + mixer``,
    before the next block's input norm and before ``norm_f``.
    """
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(model_path)
    config._attn_implementation = "eager"
    with torch.device("meta"):
        model = AutoModel.from_config(config, dtype=torch.bfloat16)
    model.eval()
    for module in model.modules():  # see --reference-dt-min
        if hasattr(module, "time_step_limit"):
            module.time_step_limit = (dt_min, float("inf"))

    shards = _Shards(model_path)
    torch_device = torch.device(device)

    # The trunk's own (small) weights are resident for the whole forward.
    model.embeddings.load_state_dict(
        {"weight": shards.get("backbone.embeddings.weight").to(torch_device)}, assign=True
    )
    model.norm_f.load_state_dict(
        {"weight": shards.get("backbone.norm_f.weight").to(torch_device)}, assign=True
    )

    captured: list[torch.Tensor] = []
    handles = []
    for index, block in enumerate(model.layers):
        empty = dict(block.state_dict())  # every entry is still on meta

        def pre_hook(module, args, kwargs, index=index):
            module.load_state_dict(
                _layer_state_dict(shards, index, list(module.state_dict()), torch_device),
                assign=True,
            )
            return None

        def post_hook(module, args, kwargs, output, index=index, empty=empty):
            hidden = output[0] if isinstance(output, tuple) else output
            captured.append(hidden.detach()[0].float().cpu())
            module.load_state_dict(empty, assign=True)
            if torch_device.type == "cuda":
                torch.cuda.empty_cache()
            return None

        handles.append(block.register_forward_pre_hook(pre_hook, with_kwargs=True))
        handles.append(block.register_forward_hook(post_hook, with_kwargs=True))

    with torch.inference_mode():
        model(input_ids=token_ids.unsqueeze(0).to(torch_device), use_cache=False)
    for handle in handles:
        handle.remove()
    shards.close()
    if len(captured) != len(model.layers):
        raise SystemExit(f"captured {len(captured)} blocks, expected {len(model.layers)}")
    return torch.stack(captured, dim=1)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    left, right = a.float().reshape(-1), b.float().reshape(-1)
    denom = float(torch.linalg.vector_norm(left)) * float(torch.linalg.vector_norm(right))
    if denom == 0.0:
        return 1.0 if torch.equal(left, right) else 0.0
    return float(torch.dot(left, right)) / denom


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.capture_only and args.artifact:
        raise SystemExit("--capture-only and --artifact are mutually exclusive")

    if args.artifact:
        path = args.artifact
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model_id = args.served_model_name or _served_model_name(args.base_url, args.timeout)
        prompt = build_prompt(tokenizer, args.prompt_tokens)
        path = run_probe(args, prompt, model_id)
    print(f"artifact: {path}")
    if args.capture_only:
        hidden, token_ids = load_artifact(path)
        print(
            f"shape: [{hidden.shape[0]} tokens, {hidden.shape[1]} layers, "
            f"{hidden.shape[2]} hidden] {hidden.dtype}"
        )
        print("capture-only: rerun with --artifact <path> once the server is stopped")
        return 0
    hidden, token_ids = load_artifact(path)
    prompt_tokens, layers, width = hidden.shape
    print(f"shape: [{prompt_tokens} tokens, {layers} layers, {width} hidden] {hidden.dtype}")

    reference = reference_hidden_states(
        args.model, token_ids, args.reference_device, args.reference_dt_min
    )
    if reference.shape[1] < layers:
        raise SystemExit(
            f"the artifact carries {layers} layers but the checkpoint has "
            f"{reference.shape[1]} blocks"
        )
    if reference.shape[2] != width:
        raise SystemExit(
            f"hidden size mismatch: artifact {width} vs checkpoint {reference.shape[2]}"
        )

    # Switchyard mean-pools over all prompt tokens, per layer, client-side.
    served_pool = hidden.float().mean(dim=0)          # [layers, hidden]
    reference_pool = reference[:, :layers].mean(dim=0)  # [layers, hidden]

    rows = []
    failures = []
    for layer_id in range(layers):
        value = cosine(reference_pool[layer_id], served_pool[layer_id])
        rows.append({"layer": layer_id, "cosine": value})
        if not (value > args.min_cosine):
            failures.append(f"layer {layer_id}: cosine {value:.6f} <= {args.min_cosine}")

    print(f"{'layer':>5}  {'cosine':>10}")
    print("-" * 17)
    for row in rows:
        print(f"{row['layer']:>5}  {row['cosine']:>10.6f}")
    worst = min(rows, key=lambda row: row["cosine"])
    print(f"\nworst: layer {worst['layer']} cosine {worst['cosine']:.6f}")

    if not args.keep_artifact:
        os.unlink(path)

    report = {
        "date": datetime.date.today().isoformat(),
        "model": args.model,
        "base_url": args.base_url,
        "prompt_tokens": prompt_tokens,
        "layers": layers,
        "hidden": width,
        "dtype": str(hidden.dtype),
        "min_cosine": args.min_cosine,
        "reference_dt_min": args.reference_dt_min,
        "worst_cosine": worst["cosine"],
        "per_layer": rows,
        "failures": failures,
        "accepted": not failures,
    }
    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)) or ".", exist_ok=True)
        with open(args.json_out, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(report) + "\n")

    if failures:
        print("\nFAIL")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
