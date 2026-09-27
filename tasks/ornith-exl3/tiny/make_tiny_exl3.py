"""A tiny random EXL3 checkpoint with Ornith's exact structure, for whole-model tests without the real weights.

    python make_tiny_exl3.py <real_config_dir> <tokenizer_dir> <out_dir>

Takes Ornith's config.json (<real_config_dir>) and shrinks the text tower (4 layers: 3 GatedDeltaNet
+ 1 full attention, hidden 512, 16 experts top-4) while keeping every head dim, the vocabulary and
the tokenizer (<tokenizer_dir>). Every linear exllamav3 quantizes is written as EXL3 (random
trellis, mul1, the real bit widths: 7 attention/shared, 5 experts, 6 lm_head) with the same
``quantization_config.json`` ``tensor_storage`` table exllamav3 writes; everything else is stored
the way the real export stores it (bf16 norms/embeddings/GDN params, fp16 routers and in_proj_a/b).
The scales keep activations O(1): suh = +-1, svh = +-g/sqrt(K), so each linear roughly preserves
the RMS of its input. No vision tensors: serve it text-only.
"""
import json
import math
import os
import shutil
import sys

import torch
from safetensors.torch import save_file

MUL1 = 0x83DCD12D
PREFIX = "model.language_model"
SHRINK = dict(
    hidden_size=512, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
    linear_num_key_heads=4, linear_num_value_heads=8, num_experts=16, num_experts_per_tok=4,
    moe_intermediate_size=256, shared_expert_intermediate_size=256, mtp_num_hidden_layers=0,
)
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                   "chat_template.jinja", "generation_config.json", "preprocessor_config.json",
                   "processor_config.json", "video_preprocessor_config.json")


def main(real_dir, tok_dir, out):
    os.makedirs(out, exist_ok=True)
    cfg = json.load(open(os.path.join(real_dir, "config.json")))
    t = cfg["text_config"]
    t.update(SHRINK)
    t["layer_types"] = ["linear_attention"] * 3 + ["full_attention"]
    cfg["hidden_size"] = t["hidden_size"]
    cfg["vision_config"]["out_hidden_size"] = t["hidden_size"]
    for f in TOKENIZER_FILES:
        if os.path.exists(os.path.join(tok_dir, f)):
            shutil.copy(os.path.join(tok_dir, f), out)

    g = torch.Generator().manual_seed(20260924)
    H, V = t["hidden_size"], t["vocab_size"]
    hd, nq, nkv = t["head_dim"], t["num_attention_heads"], t["num_key_value_heads"]
    kd, vd, nk, nv = t["linear_key_head_dim"], t["linear_value_head_dim"], t["linear_num_key_heads"], t["linear_num_value_heads"]
    E, I, SI = t["num_experts"], t["moe_intermediate_size"], t["shared_expert_intermediate_size"]
    tensors, storage = {}, {}

    def sign(n):
        return (torch.randint(0, 2, (n,), generator=g) * 2 - 1).to(torch.float16)

    def exl3(name, k, n, bits, gain=1.0):
        tr = torch.randint(-(1 << 15), 1 << 15, (k // 16, n // 16, 16 * bits), generator=g, dtype=torch.int32).to(torch.int16)
        parts = {"trellis": tr, "suh": sign(k), "svh": (sign(n).float() * gain / math.sqrt(k)).half(),
                 "mul1": torch.tensor(MUL1 - (1 << 32), dtype=torch.int32)}
        entry = {"stored_tensors": {}, "quant_format": "exl3", "bits_per_weight": bits, "mul1_multiplier": MUL1}
        for kind, v in parts.items():
            tensors[f"{name}.{kind}"] = v
            entry["stored_tensors"][f"{name}.{kind}"] = {"shape": list(v.shape), "n_bytes": v.numel() * v.element_size(), "dtype": str(v.dtype)}
        storage[name] = entry

    def plain(name, value):
        tensors[name] = value
        module, _ = name.rsplit(".", 1)
        storage.setdefault(module, {"stored_tensors": {}})["stored_tensors"][name] = {
            "shape": list(value.shape), "n_bytes": value.numel() * value.element_size(), "dtype": str(value.dtype)}

    def randn(*shape, scale=1.0, dtype=torch.bfloat16):
        return (torch.randn(shape, generator=g) * scale).to(dtype)

    plain(f"{PREFIX}.embed_tokens.weight", randn(V, H))
    plain(f"{PREFIX}.norm.weight", randn(H, scale=0.1))
    for layer, kind in enumerate(t["layer_types"]):
        p = f"{PREFIX}.layers.{layer}"
        plain(f"{p}.input_layernorm.weight", randn(H, scale=0.1))
        plain(f"{p}.post_attention_layernorm.weight", randn(H, scale=0.1))
        if kind == "linear_attention":
            a = f"{p}.linear_attn"
            exl3(f"{a}.in_proj_qkv", H, 2 * nk * kd + nv * vd, 7)
            exl3(f"{a}.in_proj_z", H, nv * vd, 7)
            exl3(f"{a}.out_proj", nv * vd, H, 7)
            plain(f"{a}.in_proj_a.weight", randn(nv, H, scale=0.05, dtype=torch.float16))
            plain(f"{a}.in_proj_b.weight", randn(nv, H, scale=0.05, dtype=torch.float16))
            plain(f"{a}.A_log", torch.log(torch.rand(nv, generator=g) * 15 + 1).to(torch.bfloat16))
            plain(f"{a}.dt_bias", randn(nv, scale=0.5))
            plain(f"{a}.conv1d.weight", randn(2 * nk * kd + nv * vd, 1, t["linear_conv_kernel_dim"], scale=0.3))
            plain(f"{a}.norm.weight", randn(vd, scale=0.1))
        else:
            a = f"{p}.self_attn"
            exl3(f"{a}.q_proj", H, 2 * nq * hd, 7)  # q carries the output gate
            exl3(f"{a}.k_proj", H, nkv * hd, 7)
            exl3(f"{a}.v_proj", H, nkv * hd, 7)
            exl3(f"{a}.o_proj", nq * hd, H, 7)
            plain(f"{a}.q_norm.weight", randn(hd, scale=0.1))
            plain(f"{a}.k_norm.weight", randn(hd, scale=0.1))
        m = f"{p}.mlp"
        plain(f"{m}.gate.weight", randn(E, H, scale=0.1, dtype=torch.float16))
        plain(f"{m}.shared_expert_gate.weight", randn(1, H, scale=0.05, dtype=torch.float16))
        for proj, (k, n) in (("gate_proj", (H, SI)), ("up_proj", (H, SI)), ("down_proj", (SI, H))):
            exl3(f"{m}.shared_expert.{proj}", k, n, 7)
        for e in range(E):
            for proj, (k, n) in (("gate_proj", (H, I)), ("up_proj", (H, I)), ("down_proj", (I, H))):
                exl3(f"{m}.experts.{e}.{proj}", k, n, 5)
    exl3("lm_head", H, V, 6, gain=4.0)  # logits with a few units of spread, so top-1 is not a coin toss

    names = list(tensors)
    shards = {"model-00001-of-00002.safetensors": names[: len(names) // 2], "model-00002-of-00002.safetensors": names[len(names) // 2:]}
    weight_map = {}
    for shard, keys in shards.items():
        save_file({k: tensors[k].contiguous() for k in keys}, os.path.join(out, shard), metadata={"format": "pt"})
        weight_map.update({k: shard for k in keys})
    json.dump({"metadata": {}, "weight_map": weight_map}, open(os.path.join(out, "model.safetensors.index.json"), "w"))
    q = dict(cfg["quantization_config"])
    q["tensor_storage"] = storage
    json.dump(q, open(os.path.join(out, "quantization_config.json"), "w"))
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)
    print("wrote", len(tensors), "tensors to", out)


if __name__ == "__main__":
    main(*sys.argv[1:4])
