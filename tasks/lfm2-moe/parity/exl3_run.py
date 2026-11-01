"""exllamav3 side of the lfm2_moe parity check (exllamav3 v1.6.6 venv).

    python exl3_run.py <model_dir> <ft.pt> <out.pt>

Pass 1 runs the prompt through exllamav3's forward and records, per TransformerBlock, the input
and output of the mixer (ShortConv / Attention) and of the MLP (GatedMLP / BlockSparseMLP), plus
the embedding output. Pass 2 runs prompt + FreeToken's generated ids teacher-forced and keeps
the logits at the positions that predicted each generated token.
"""
import sys

import torch
from exllamav3 import Config, Model
from exllamav3.modules.transformer import TransformerBlock


def main(model_dir, ft_path, out):
    ft = torch.load(ft_path)
    prompt, gen = ft["prompt_ids"], ft["output_ids"]
    config = Config.from_directory(model_dir)
    model = Model.from_config(config)
    model.load(device="cuda:0", max_chunk_size=4096)
    rec = {"mixer_in": {}, "mixer_out": {}, "ffn_in": {}, "ffn_out": {}, "embed": None}
    on = [False]

    def wrap(mod, i, kind):
        orig = mod.forward

        def fwd(x, params, *a, **k):
            y = orig(x, params, *a, **k)
            if on[0]:
                rec[f"{kind}_in"][i] = x.float().reshape(-1, x.shape[-1]).cpu()
                rec[f"{kind}_out"][i] = y.float().reshape(-1, y.shape[-1]).cpu()
            return y

        mod.forward = fwd

    emb = model.modules[0]
    emb_orig = emb.forward

    def emb_fwd(x, params, *a, **k):
        y = emb_orig(x, params, *a, **k)
        if on[0]:
            rec["embed"] = y.float().reshape(-1, y.shape[-1]).cpu()
        return y

    emb.forward = emb_fwd
    blocks = [m for m in model.modules if isinstance(m, TransformerBlock)]
    for i, b in enumerate(blocks):
        wrap(b.attn, i, "mixer")
        wrap(b.mlp, i, "ffn")
    with torch.inference_mode():
        on[0] = True
        model.forward(torch.tensor([prompt], dtype=torch.long), {})
        on[0] = False
        n = len(gen)
        logits = model.forward(torch.tensor([prompt + gen], dtype=torch.long), {"last_tokens_only": n + 1})
    sel = logits[0, -(n + 1):-1].float().cpu()
    rec["logits"] = sel
    torch.save(rec, out)
    print("saved", out, tuple(sel.shape), "layers", len(rec["mixer_out"]))


if __name__ == "__main__":
    main(*sys.argv[1:4])
