"""exllamav3 side of the whole-model logits comparison (exllamav3 venv).

    python exl3_logits.py <model_dir> <ft.pt> <out.pt>

Runs exllamav3's own forward over FreeToken's prompt + generated ids in one pass (teacher-forced)
and saves the logits at the positions that predicted each generated token.
"""
import sys

import torch
from exllamav3 import Config, Model


def main(model_dir, ft_path, out):
    ft = torch.load(ft_path)
    ids = ft["prompt_ids"] + ft["output_ids"]
    config = Config.from_directory(model_dir)
    model = Model.from_config(config)
    # the whole 35B model does not fit a 16 GB card: set EXL3_MOE_CPU_OFFLOAD=<n layers> in the
    # environment to run the routed experts of the first n MoE layers on exllamav3's CPU worker
    model.load(device="cuda:0", max_chunk_size=max(2048, len(ids)))
    n = len(ft["output_ids"])
    with torch.inference_mode():
        # logits of the last n + 1 positions only (a full [T, 248320] would not fit); drop the last
        logits = model.forward(torch.tensor([ids], dtype=torch.long), {"last_tokens_only": n + 1})
    sel = logits[0, -(n + 1):-1].float().cpu()
    torch.save({"logits": sel, "ids": ids}, out)
    print("saved", tuple(sel.shape))


if __name__ == "__main__":
    main(*sys.argv[1:4])
