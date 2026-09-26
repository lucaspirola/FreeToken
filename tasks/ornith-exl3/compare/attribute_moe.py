# usage: python attribute_moe.py <model_dir> <layer0_ref.pt> <layer0_ft outputs .pt (layer0_ft.py 3rd arg)>
# attribute layer-0 MoE and lm_head 1-row differences: FreeToken and exllamav3 each vs an fp32 exact reference
import sys, torch, torch.nn.functional as F
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from layer0_ref import Reader, P
from freetoken.kernel.triton.exl3 import linear_reference, Exl3Parts
from freetoken.layers.quantization.linear.exl3 import exl3_forward
root, refp, ftp = sys.argv[1:4]
ref = torch.load(refp); rd = Reader(root)
m = ref["moe"]; x, ids, w = m["x"].float().cuda(), m["ids"], m["w"]
exact = torch.zeros(x.shape[0], x.shape[1], device="cuda")
cache = {}
def lin(name, xi):
    if name not in cache:
        t = rd.module(name); cache[name] = {k: v.cuda() for k, v in t.items()}
    t = cache[name]
    return linear_reference(xi, t["trellis"], t["suh"], t["svh"], "mul1").float()
for tkn in range(x.shape[0]):
    for k in range(ids.shape[1]):
        e = int(ids[tkn, k]); b = f"{P}.0.mlp.experts.{e}"
        xi = x[tkn:tkn + 1]
        a = F.silu(lin(f"{b}.gate_proj", xi)) * lin(f"{b}.up_proj", xi)
        exact[tkn] += float(w[tkn, k]) * lin(f"{b}.down_proj", a)[0]
mx = exact.abs().max()
ex = m["y"].cuda()
print("MoE: exllamav3(per-expert fp16) vs exact", float((ex - exact).abs().max() / mx))
ft = torch.load(ftp)
print("MoE: FreeToken prefill vs exact", float((ft["yp"].cuda().float() - exact).abs().max() / mx),
      " decode vs exact", float((ft["yd"].cuda().float() - exact[:3]).abs().max() / exact[:3].abs().max()))
