"""Count tokens whose top-k expert set differs between FreeToken's and exllamav3's chained
MoE inputs (same router math on both: sigmoid, + expert_bias for selection).

    python routing_flips.py <model_dir> <ft.pt> <exl3.pt>
"""
import json
import sys

import torch
from safetensors import safe_open


def main(model_dir, ft_path, ex_path):
    cfg = json.load(open(f"{model_dir}/config.json"))
    k = cfg["num_experts_per_tok"]
    ft, ex = torch.load(ft_path), torch.load(ex_path)
    with safe_open(f"{model_dir}/model.safetensors", "pt") as f:
        for i in range(cfg["num_dense_layers"], cfg["num_hidden_layers"]):
            w = f.get_tensor(f"model.layers.{i}.feed_forward.gate.weight").float()
            b = f.get_tensor(f"model.layers.{i}.feed_forward.expert_bias").float()
            sel = lambda x: (torch.sigmoid(x @ w.T) + b).topk(k, -1).indices.sort(-1).values
            s_ft, s_ex = sel(ft["layers"]["ffn_in"][i]), sel(ex["ffn_in"][i])
            sc = torch.sigmoid(ex["ffn_in"][i] @ w.T) + b
            top = sc.topk(k + 1, -1).values
            print(f"layer {i}: {int((s_ft != s_ex).any(-1).sum())}/{len(s_ft)} tokens route differently; "
                  f"min gap between expert k and k+1 {float((top[:, k - 1] - top[:, k]).min()):.2e}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
