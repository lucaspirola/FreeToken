"""Isolated MoE rows with > 5% error: are they all routing near-ties?

    python iso_moe_ties.py <model_dir> <ft_iso.pt> <exl3.pt> [tie_gap]

A row is a near-tie when the k-th and (k+1)-th selection scores (sigmoid + bias, fp32, on
exllamav3's input) differ by less than tie_gap (default 2e-3, about fp16 resolution near 1.0,
the precision exllamav3 routes in), or when rounding that input to bf16 (FreeToken's
activation dtype) changes the selected set. Prints, per MoE layer, the large-error rows, how many are
near-ties, and the worst isolated error over the rows that are not near-ties.
"""
import json
import sys

import torch
from safetensors import safe_open


def main(model_dir, ft_path, ex_path, tie_gap="2e-3"):
    cfg = json.load(open(f"{model_dir}/config.json"))
    k, tie = cfg["num_experts_per_tok"], float(tie_gap)
    ft, ex = torch.load(ft_path), torch.load(ex_path)
    L = ft["layers"]
    worst = 0.0
    with safe_open(f"{model_dir}/model.safetensors", "pt") as f:
        for i in range(cfg["num_dense_layers"], cfg["num_hidden_layers"]):
            w = f.get_tensor(f"model.layers.{i}.feed_forward.gate.weight").float()
            b = f.get_tensor(f"model.layers.{i}.feed_forward.expert_bias").float()
            x = ex["ffn_in"][i]
            s = (torch.sigmoid(x @ w.T) + b).topk(k + 1, -1).values
            pick = lambda v: (torch.sigmoid(v @ w.T) + b).topk(k, -1).indices.sort(-1).values
            near = ((s[:, k - 1] - s[:, k]) < tie) | (pick(x) != pick(x.bfloat16().float())).any(-1)
            ref = ex["ffn_out"][i]
            err = (L["iso_ffn"][i] - ref).abs().max(-1).values / ref.abs().max()
            bad = err > 0.05
            clean = float(err[~near].max())
            worst = max(worst, clean)
            print(f"layer {i}: {int(bad.sum())} rows > 5%, {int((bad & near).sum())} of them near-ties "
                  f"({int(near.sum())}/{len(near)} near-tie rows); worst rel on the other rows {clean:.2e}")
    print(f"worst isolated MoE rel outside near-ties: {worst:.2e}")


if __name__ == "__main__":
    main(*sys.argv[1:])
