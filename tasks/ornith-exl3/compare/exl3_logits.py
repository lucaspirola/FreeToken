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
    model.load(device="cuda:0")
    with torch.inference_mode():
        logits = model.forward(torch.tensor([ids], dtype=torch.long), {})
    p = len(ft["prompt_ids"])
    sel = logits[0, p - 1: len(ids) - 1].float().cpu()
    torch.save({"logits": sel, "ids": ids}, out)
    print("saved", tuple(sel.shape))


if __name__ == "__main__":
    main(*sys.argv[1:4])
