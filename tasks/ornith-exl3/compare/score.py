"""Compare FreeToken's per-step logits with exllamav3's teacher-forced logits.

    python score.py <ft.pt> <exl3.pt>

Reports max |dlogit|, the same relative to the largest |logit|, top-1 agreement, top-5 overlap
and the max KL(exl3 || ft) over the generated positions.
"""
import sys

import torch


def main(ft_path, ex_path):
    ft, ex = torch.load(ft_path), torch.load(ex_path)
    a, b = ft["logits"].float(), ex["logits"].float()
    assert a.shape == b.shape, (a.shape, b.shape)
    d = (a - b).abs()
    top1 = (a.argmax(-1) == b.argmax(-1)).float()
    t5a, t5b = a.topk(5, -1).indices, b.topk(5, -1).indices
    overlap = torch.tensor([len(set(x.tolist()) & set(y.tolist())) / 5 for x, y in zip(t5a, t5b)])
    kl = torch.nn.functional.kl_div(a.log_softmax(-1), b.log_softmax(-1), log_target=True, reduction="none").sum(-1)
    print(f"positions {a.shape[0]}  vocab {a.shape[1]}")
    print(f"max|dlogit| {d.max():.4f}  mean|dlogit| {d.mean():.5f}  max|dlogit|/max|logit| {d.max() / b.abs().max():.2e}")
    print(f"top-1 agreement {top1.mean():.3f} ({int(top1.sum())}/{len(top1)})  first disagreement {int((top1 == 0).nonzero()[0]) if (top1 == 0).any() else None}")
    print(f"top-5 overlap {overlap.mean():.3f}  KL(exl3||ft) max {kl.max():.2e} mean {kl.mean():.2e}")
    if "-v" in sys.argv:
        for i in range(a.shape[0]):
            print(f"  pos {i:3d}  max|d| {d[i].max():.4f}  mean|d| {d[i].mean():.5f}  KL {kl[i]:.2e}  top1 {int(top1[i])}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
