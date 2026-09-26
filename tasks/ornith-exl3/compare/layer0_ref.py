"""exllamav3 reference outputs on REAL checkpoint tensors (exllamav3 venv, CUDA).

    python layer0_ref.py <model_dir> <out.pt>

For every EXL3 linear of layer 0 (GatedDeltaNet + shared expert), layer 3 (full attention) and
lm_head: seeded fp16 inputs through exllamav3's ``LinearEXL3`` (its small-batch kernel for 1 row,
reconstruct + hgemm for 300 rows). For layer 0's routed experts: a seeded routing of 37 tokens
(the config's top-k of its experts) through exllamav3's own per-expert linears, fp16 intermediates as in its MoE, so
the output is the routed sum ``layer0_ft.py`` compares ``fused_experts_exl3`` with.
"""
import json
import os
import sys

import torch
import torch.nn.functional as F
from safetensors import safe_open

P = "model.language_model.layers"
LINEARS = [f"{P}.0.linear_attn.{n}" for n in ("in_proj_qkv", "in_proj_z", "out_proj")] + \
          [f"{P}.0.mlp.shared_expert.{n}" for n in ("gate_proj", "up_proj", "down_proj")] + \
          [f"{P}.3.self_attn.{n}" for n in ("q_proj", "k_proj", "v_proj", "o_proj")] + ["lm_head"]
KINDS = ("trellis", "suh", "svh", "mul1", "mcg")
# the fused layouts FreeToken loads: target -> its checkpoint parts (one shared input per group)
FUSED = {
    f"{P}.0.linear_attn.in_proj_qkvz": [f"{P}.0.linear_attn.in_proj_qkv", f"{P}.0.linear_attn.in_proj_z"],
    f"{P}.3.self_attn.qkv_proj": [f"{P}.3.self_attn.{n}" for n in ("q_proj", "k_proj", "v_proj")],
    f"{P}.0.mlp.shared_expert.gate_up_proj": [f"{P}.0.mlp.shared_expert.{n}" for n in ("gate_proj", "up_proj")],
}


class Reader:
    def __init__(self, root):
        self.root = root
        self.wm = json.load(open(os.path.join(root, "model.safetensors.index.json")))["weight_map"]
        self.files = {}

    def module(self, name):
        out = {}
        for kind in KINDS:
            key = f"{name}.{kind}"
            if key in self.wm:
                f = self.files.setdefault(self.wm[key], safe_open(os.path.join(self.root, self.wm[key]), "pt"))
                out[kind] = f.get_tensor(key)
        return out


def linear(t):
    from exllamav3.modules.quant.exl3 import LinearEXL3  # imported here: layer0_ft.py reuses this module without exllamav3

    k, n = t["suh"].shape[0], t["svh"].shape[0]
    flags = {f: t[f].cuda() for f in ("mul1", "mcg") if f in t}
    return LinearEXL3(None, k, n, suh=t["suh"].cuda(), svh=t["svh"].cuda(), trellis=t["trellis"].cuda(), **flags)


def main(root, out):
    rd = Reader(root)
    res = {"linears": {}, "moe": {}}
    g = torch.Generator().manual_seed(7)
    with torch.inference_mode():
        for name in LINEARS:
            lin = linear(rd.module(name))
            k = lin.in_features
            x1 = torch.randn(1, k, generator=g).half()
            x300 = torch.randn(300, k, generator=g).half()
            res["linears"][name] = {
                "x1": x1, "y1": lin.forward(x1.cuda(), {}).cpu(),
                "x300": x300, "y300": lin.forward(x300.cuda(), {"reconstruct": True}).cpu(),
            }
            del lin
            torch.cuda.empty_cache()
        res["fused"] = {}
        for target, names in FUSED.items():
            lins = [linear(rd.module(nm)) for nm in names]
            for rows in (1, 300):
                xg = torch.randn(rows, lins[0].in_features, generator=g).half()
                yg = torch.cat([ln.forward(xg.cuda(), {} if rows == 1 else {"reconstruct": True}) for ln in lins], dim=1)
                res["fused"][(target, rows)] = {"x": xg, "y": yg.cpu()}
            del lins
        cfg = json.load(open(os.path.join(root, "config.json")))["text_config"]
        T, K, E, H = 37, cfg["num_experts_per_tok"], cfg["num_experts"], cfg["hidden_size"]
        x = torch.randn(T, H, generator=g).half()
        ids = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(T)])
        w = torch.softmax(torch.randn(T, K, generator=g), -1)
        acc = torch.zeros(T, H, dtype=torch.float32, device="cuda")
        for e in ids.unique().tolist():
            base = f"{P}.0.mlp.experts.{e}"
            gate, up, down = (linear(rd.module(f"{base}.{p}")) for p in ("gate_proj", "up_proj", "down_proj"))
            rows = (ids == e).nonzero()
            xe = x[rows[:, 0]].cuda()
            a = F.silu(gate.forward(xe, {})) * up.forward(xe, {})          # fp16, as exllamav3's MoE
            o = down.forward(a.contiguous(), {}).float()
            acc.index_add_(0, rows[:, 0].cuda(), o * w[rows[:, 0], rows[:, 1]].cuda()[:, None])
        res["moe"] = {"x": x, "ids": ids, "w": w, "y": acc.cpu()}
    torch.save(res, out)
    print("saved", len(res["linears"]), "linears + layer-0 MoE to", out)


if __name__ == "__main__":
    main(*sys.argv[1:3])
