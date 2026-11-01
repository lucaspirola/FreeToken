"""Score the lfm2_moe parity runs.

    python score.py <ft_iso.pt> <exl3.pt>

``ft_iso.pt`` is an ft_run.py output made with ``--ref exl3.pt``. Per layer kind it reports the
isolated error (FreeToken's sub-module on exllamav3's exact input), the chained error (each
side's own forward) and, at the output, logits max |d|, top-1 agreement and KL.
rel = max|ft - ex| / max|ex|.
"""
import sys

import torch


def rel(a, b):
    return float((a - b).abs().max() / b.abs().max())


def main(ft_path, ex_path):
    ft, ex = torch.load(ft_path), torch.load(ex_path)
    L, kinds = ft["layers"], ft["kinds"]
    n_dense = int(ft.get("n_dense", 2))
    groups = {}
    for i, kind in enumerate(kinds):
        groups.setdefault("conv mixer" if kind == "conv" else "attention mixer", []).append(("mixer", i))
        groups.setdefault("dense FFN" if i < n_dense else "MoE FFN", []).append(("ffn", i))
    print(f"prompt {len(ft['prompt_ids'])} tokens, {len(ft['output_ids'])} generated; layers {kinds}")
    print(f"{'layer kind':16s} {'layers':14s} {'isolated rel':>13s} {'isolated max|d|':>16s} {'chained rel':>12s}")
    for name, items in groups.items():
        iso = [rel(L[f'iso_{s}'][i], ex[f'{s}_out'][i]) for s, i in items]
        iso_abs = [float((L[f'iso_{s}'][i] - ex[f'{s}_out'][i]).abs().max()) for s, i in items]
        ch = [rel(L[f'{s}_out'][i], ex[f'{s}_out'][i]) for s, i in items]
        print(f"{name:16s} {str([i for _, i in items]):14s} {max(iso):13.2e} {max(iso_abs):16.2e} {max(ch):12.2e}")
    stream = ex["embed"].clone()
    worst = 0.0
    for i in range(len(kinds)):
        stream = stream + ex["mixer_out"][i] + ex["ffn_out"][i]
        worst = max(worst, rel(L["stream"][i], stream))
    print(f"residual stream after each layer (chained), worst rel {worst:.2e}; last layer {rel(L['stream'][len(kinds) - 1], stream):.2e}")
    a, b = ft["logits"].float(), ex["logits"].float()
    d = (a - b).abs()
    top1 = (a.argmax(-1) == b.argmax(-1)).float()
    kl = torch.nn.functional.kl_div(a.log_softmax(-1), b.log_softmax(-1), log_target=True, reduction="none").sum(-1)
    gap = b.topk(2, -1).values
    print(f"output logits over {a.shape[0]} positions: max|d| {d.max():.4f}  rel {d.max() / b.abs().max():.2e}  "
          f"top-1 {int(top1.sum())}/{len(top1)}  KL(ex||ft) max {kl.max():.2e} mean {kl.mean():.2e}  "
          f"min top1-top2 gap {float((gap[:, 0] - gap[:, 1]).min()):.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
