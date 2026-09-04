"""End-to-end parity for the Switchyard prefill-probe hidden-state export.

Sends ONE probe request through a running FreeToken server (started with
``--hidden-states-dir``), loads the ``.safetensors`` artifact the response points at,
and compares it against ``transformers.AutoModelForCausalLM(output_hidden_states=True)``
run on the CPU in bf16 over the *same* token ids -- which the artifact carries, so no
re-tokenization can drift the two apart.

What is compared is what Switchyard actually consumes: the mean over prompt tokens of
each layer's residual vector. Per-layer cosine must exceed ``--min-cosine`` (0.99 by
default), which is loose enough for NVFP4 experts and FP8 projections on the served side
against a bf16 reference, and tight enough that an off-by-one layer index, a final-norm
leak, or a dropped prefill chunk fails loudly.

    HF's ``hidden_states`` has ``num_layers + 1`` entries: ``[0]`` is the embedding
    output and ``[i + 1]`` is the output of block ``i``. FreeToken's row ``i`` is block
    ``i``'s output, so the pairing is ``artifact[:, i] <-> hidden_states[i + 1]``, and
    ``hidden_states[-1]`` (which transformers passes through the final norm on some
    architectures) is never used.

Run it under the GPU lock, against a P1-profile server:

    scripts/gpu_lock.sh uv run benchmarks/probe_hidden_states_parity.py \\
        --model ~/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \\
        --base-url http://127.0.0.1:1919 --hidden-states-dir /tmp/ft-hidden-states \\
        --prompt-tokens 300
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
    parser.add_argument("--layers", default=None, help="e.g. 0-51 (default: all)")
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


def reference_hidden_states(model_path: str, token_ids: torch.Tensor) -> torch.Tensor:
    """``[tokens, layers, hidden]`` of post-block residuals, from transformers on CPU."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    )
    model.eval()
    with torch.no_grad():
        out = model(input_ids=token_ids.unsqueeze(0), output_hidden_states=True)
    # hidden_states[0] is the embedding output; [i + 1] is block i's output.
    blocks = out.hidden_states[1:]
    return torch.stack([layer[0].float() for layer in blocks], dim=1)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    left, right = a.float().reshape(-1), b.float().reshape(-1)
    denom = float(torch.linalg.vector_norm(left)) * float(torch.linalg.vector_norm(right))
    if denom == 0.0:
        return 1.0 if torch.equal(left, right) else 0.0
    return float(torch.dot(left, right)) / denom


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model_id = args.served_model_name or _served_model_name(args.base_url, args.timeout)
    prompt = build_prompt(tokenizer, args.prompt_tokens)

    path = run_probe(args, prompt, model_id)
    print(f"artifact: {path}")
    hidden, token_ids = load_artifact(path)
    prompt_tokens, layers, width = hidden.shape
    print(f"shape: [{prompt_tokens} tokens, {layers} layers, {width} hidden] {hidden.dtype}")

    reference = reference_hidden_states(args.model, token_ids)
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
