"""FreeToken's EXL3 kernels on the same REAL tensors as ``layer0_ref.py`` (FreeToken venv, CUDA).

    python layer0_ft.py <model_dir> <ref.pt>

Dense: ``exl3_forward`` per linear (1-row GEMV, 300-row GEMM), and the fused layouts the model
actually loads (GDN in_proj_qkv|z, attention q|k|v, shared gate|up) through
``Exl3Config.fuse_parts``. MoE: layer 0's routed experts packed into banks through the real
reader + ``pack`` (prefill over all 256, decode over a slot subset) vs exllamav3's routed sum.
Metric: rel = max|y - ref| / max|ref|.
"""
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from layer0_ref import P, Reader  # noqa: E402


def rel(y, ref):
    return float((y.float().cpu() - ref.float()).abs().max() / ref.float().abs().max())


def main(root, ref_path):
    import freetoken.distributed.info as di
    from freetoken.kernel.triton.exl3 import Exl3Parts
    from freetoken.layers.quantization.linear.exl3 import exl3_forward
    from freetoken.models.register import _load_attr, checkpoint_quant_config, get_model_spec
    from freetoken.layers.quantization import set_quant_config
    from freetoken.utils import cached_load_hf_config

    di.set_tp_info(0, 1)
    ref = torch.load(ref_path)
    rd = Reader(root)
    worst = 0.0
    print("dense linears (rel err vs exllamav3):")
    for name, r in ref["linears"].items():
        t = rd.module(name)
        cb = "mul1" if "mul1" in t else ("mcg" if "mcg" in t else "3inst")
        k, n = t["suh"].shape[0], t["svh"].shape[0]
        parts = Exl3Parts.build(k, (n,), t["trellis"].shape[-1] // 16, cb, "cuda")
        args = (t["trellis"].cuda().reshape(-1), t["suh"].cuda().view(1, -1), t["svh"].cuda(), parts, torch.float16)
        e1 = rel(exl3_forward(r["x1"].cuda(), *args), r["y1"])
        e300 = rel(exl3_forward(r["x300"].cuda(), *args), r["y300"])
        worst = max(worst, e1, e300)
        print(f"  {name:60s} K={k:5d} N={n:6d} bits={t['trellis'].shape[-1] // 16}  1-row {e1:.2e}  300-row {e300:.2e}")

    hf = cached_load_hf_config(root)
    spec = get_model_spec(hf.architectures[0])
    quant = checkpoint_quant_config(root, hf, spec)
    set_quant_config(quant)
    print("fused layouts (Exl3Config.fuse_parts, as the model loads them) vs exllamav3 per part:")
    from layer0_ref import FUSED

    for target, names in FUSED.items():
        mods = [rd.module(nm) for nm in names]
        fused = quant.fuse_parts(target, quant.scheme_for_name(names[0]), mods)
        k, sizes = mods[0]["suh"].shape[0], tuple(m["svh"].shape[0] for m in mods)
        parts = Exl3Parts.build(k, sizes, mods[0]["trellis"].shape[-1] // 16, "mul1", "cuda")
        for rows in (1, 300):
            r = ref["fused"][(target, rows)]
            y = exl3_forward(r["x"].cuda(), fused["trellis"].cuda(), fused["suh"].cuda(), fused["svh"].cuda(), parts, torch.float16)
            e = rel(y, r["y"])
            worst = max(worst, e)
            print(f"  {target:60s} parts {sizes} {rows:3d}-row {e:.2e}")

    mc = _load_attr(spec.module, spec.parse_config)(hf)
    from freetoken.layers.quantization.moe.base import MoEConfig
    from freetoken.layers.quantization.moe.exl3 import TritonExl3MoEKernel
    from freetoken.layers.quantization.scheme import exl3_scheme
    from freetoken.models.exl3_banks import empty_piece, expert_index_for, _pread_file, read_expert
    from freetoken.moe.fused_exl3 import fused_experts_exl3

    idx = expert_index_for(root, mc)
    cfg = MoEConfig(num_experts=mc.num_experts, hidden=mc.hidden_size, intermediate=mc.moe_intermediate_size,
                    top_k=mc.num_experts_per_tok if hasattr(mc, "num_experts_per_tok") else ref["moe"]["ids"].shape[1], scheme=exl3_scheme(idx.bits, idx.codebook), strategy="offload")
    kernel = TritonExl3MoEKernel()
    layout = kernel.layout(cfg)
    E = mc.num_experts
    banks = {nm: torch.empty((E, *s.shape), dtype=s.dtype) for nm, s in layout.items()}
    read, close = _pread_file(root)
    for e in range(E):
        piece = empty_piece(idx, 1)
        read_expert(idx, e, read, piece, 0)
        kernel.pack(piece, cfg, {nm: b[e:e + 1] for nm, b in banks.items()})
    close()
    order = ("gate_up_trellis", "gate_up_suh", "gate_up_svh", "down_trellis", "down_suh", "down_svh")
    m = ref["moe"]
    x, ids, w = m["x"].cuda().to(torch.bfloat16), m["ids"].to(torch.int32).cuda(), m["w"].cuda()
    gpu = tuple(banks[nm].cuda() for nm in order)
    yp = fused_experts_exl3(x, gpu, w, ids, bits=idx.bits, codebook=idx.codebook, is_prefill=True, num_experts=E)
    ep = rel(yp, m["y"])
    # decode: 3 tokens through a cache of len(used)+8 slots holding their experts, ids remapped to slots
    used = m["ids"][:3].unique()
    slot_of = {int(e): s for s, e in enumerate(used.tolist())}
    slots = tuple(torch.cat([banks[nm][used], torch.zeros_like(banks[nm][:8])]).cuda() for nm in order)
    sid = torch.tensor([[slot_of[int(e)] for e in row] for row in m["ids"][:3]], dtype=torch.int32).cuda()
    yd = fused_experts_exl3(x[:3], slots, w[:3], sid, bits=idx.bits, codebook=idx.codebook, is_prefill=False)
    ed = rel(yd, m["y"][:3])
    worst = max(worst, ep, ed)
    print(f"layer-0 routed MoE (bf16 in/out) vs exllamav3's per-expert linears: prefill {x.shape[0]} tok {ep:.2e}  decode 3 tok via slots {ed:.2e}")
    print(f"WORST rel err {worst:.2e}")
    if len(sys.argv) > 3:  # keep the MoE outputs for attribution against an exact reference
        torch.save({"yp": yp.cpu(), "yd": yd.cpu()}, sys.argv[3])


if __name__ == "__main__":
    main(*sys.argv[1:3])
