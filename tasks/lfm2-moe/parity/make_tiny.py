"""Tiny random lfm2_moe for the FreeToken vs exllamav3 parity check (FreeToken venv).

    python make_tiny.py <hf_dir_with_tokenizer> <out_dir>

Every Linear dim is a multiple of 128 (EXL3 tiles). The real tokenizer and vocab (128000)
are kept so the chat template works. Weights use std 0.06 (HF default 0.02 gives near-flat
logits, which makes top-1 agreement meaningless) and expert_bias is random so the
selection-only bias path is exercised.
"""
import json
import os
import shutil
import sys

import torch
from transformers import AutoModelForCausalLM, Lfm2MoeConfig


def main(src, out):
    c = json.load(open(f"{src}/config.json"))
    kinds = ["conv", "conv", "full_attention", "conv", "conv", "conv", "full_attention", "conv"]
    c.update(
        hidden_size=256, intermediate_size=512, moe_intermediate_size=256, num_attention_heads=4,
        num_key_value_heads=2, num_experts=16, num_experts_per_tok=4, num_hidden_layers=len(kinds),
        layer_types=kinds, num_dense_layers=2, initializer_range=0.06,
    )
    c.pop("auto_map", None)
    cfg = Lfm2MoeConfig(**{k: v for k, v in c.items() if k not in ("architectures", "model_type", "transformers_version")})
    torch.manual_seed(0)
    m = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32)
    with torch.no_grad():
        for name, p in [*m.named_parameters(), *m.named_buffers()]:
            if name.endswith("expert_bias"):
                p.copy_(torch.randn_like(p) * 0.05)
            elif name.endswith("feed_forward.gate.weight"):
                # TINY_GATE_STD makes routing decisive, so bf16 noise does not flip expert picks
                p.copy_(torch.randn_like(p) * float(os.environ.get("TINY_GATE_STD", "0.06")))
            elif name.endswith("conv.conv.weight"):
                p.copy_(torch.randn_like(p) * 0.5)
            elif "norm" in name:
                p.copy_(1.0 + 0.1 * torch.randn_like(p))
    m = m.to(torch.bfloat16)
    m.save_pretrained(out, max_shard_size="50MB")
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"):
        shutil.copy(f"{src}/{f}", f"{out}/{f}")
    print("saved", out, sum(p.numel() for p in m.parameters()), "params")


if __name__ == "__main__":
    main(*sys.argv[1:3])
