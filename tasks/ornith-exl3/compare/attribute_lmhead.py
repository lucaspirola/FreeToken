# usage: python attribute_lmhead.py <model_dir> <layer0_ref.pt>
# lm_head 1-row: FreeToken and exllamav3 each vs an fp32 exact reference (computed in N slices)
import sys, torch
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from layer0_ref import Reader
from freetoken.kernel.triton.exl3 import Exl3Parts, linear_reference
from freetoken.layers.quantization.linear.exl3 import exl3_forward
root, refp = sys.argv[1:3]
ref = torch.load(refp)["linears"]["lm_head"]
t = {k: v.cuda() for k, v in Reader(root).module("lm_head").items()}
x = ref["x1"].cuda()
k, n = t["suh"].shape[0], t["svh"].shape[0]
step = 128 * 64
yt = torch.cat([linear_reference(x.float(), t["trellis"][:, j // 16:(j + step) // 16], t["suh"], t["svh"][j:j + step], "mul1")
                for j in range(0, n, step)], dim=1)
parts = Exl3Parts.build(k, (n,), t["trellis"].shape[-1] // 16, "mul1", "cuda")
yf = exl3_forward(x, t["trellis"].reshape(-1), t["suh"].view(1, -1), t["svh"], parts, torch.float32)
ye = ref["y1"].cuda().float()
m = yt.abs().max()
print(f"lm_head 1-row: FreeToken vs exact {float((yf - yt).abs().max() / m):.2e}   exllamav3 vs exact {float((ye - yt).abs().max() / m):.2e}")
i = int((ye - yt).abs().argmax())
print(f"worst exllamav3 column {i}: exllamav3 {float(ye.flatten()[i]):.4f} exact {float(yt.flatten()[i]):.4f} FreeToken {float(yf.flatten()[i]):.4f}")
