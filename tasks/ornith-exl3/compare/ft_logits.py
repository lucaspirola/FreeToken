"""FreeToken side of the whole-model logits comparison (FreeToken venv, owns the GPU while it runs).

    python ft_logits.py <out.pt> <n_new> -- <ft serve flags, e.g. --model DIR --text-model-only ...>

Parses the flags exactly as ``ft serve`` does, runs the engine offline on a fixed chat prompt,
decodes ``n_new`` tokens greedily and records the full-vocabulary logits of every step (the
sampler's input: prefill's last row, then each decode step, graphed or not). Saves
``{"prompt_ids", "output_ids", "logits" [n_new, V] fp32, "argv"}``; ``exl3_logits.py`` then
scores the same ids with exllamav3 and ``score.py`` compares.
"""
import dataclasses
import os
import sys

import torch

# FT_PROMPT_REPEAT=n prepends n copies of a filler paragraph (long prefills: chunking, the MoE
# prefill path at thousands of tokens, the mirror's prefill assembly)
_FILLER = "The quick brown fox jumps over the lazy dog while the committee reviews the budget. " * 8
PROMPT = [{"role": "user", "content": _FILLER * int(os.environ.get("FT_PROMPT_REPEAT", "0"))
           + "Write three short sentences about the history of the bicycle."}]


def main(out, n_new, flags):
    from freetoken.core import SamplingParams
    from freetoken.engine import sample as sample_mod
    from freetoken.llm.llm import LLM
    from freetoken.scheduler import SchedulerConfig
    from freetoken.server.args import parse_args

    sa, _ = parse_args(flags)
    skip = {"model_path", "tp_info", "dtype", "offline_mode"}
    kwargs = {f.name: getattr(sa, f.name) for f in dataclasses.fields(SchedulerConfig) if f.init and f.name not in skip}
    captured = []
    real_sample = sample_mod.Sampler.sample

    # FT_FORCE=<run.pt>:<start> teacher-forces that run's output ids from sampler call <start> on
    # (its logged "emitted window starts at"), so two code versions are compared on the same ids
    force = os.environ.get("FT_FORCE", "")
    forced, force_start = [], 0
    if force:
        path, force_start = force.rsplit(":", 1)
        forced, force_start = torch.load(path)["output_ids"], int(force_start)

    def sample(self, logits, args):
        k = len(captured)
        captured.append(logits[:1].float().cpu())
        tokens = real_sample(self, logits, args)
        if forced and force_start <= k < force_start + len(forced):
            tokens = torch.full_like(tokens, forced[k - force_start])
        return tokens

    sample_mod.Sampler.sample = sample
    llm = LLM(sa.model_path, dtype=sa.dtype, **kwargs)
    try:
        prompt_ids = llm.tokenizer.apply_chat_template(PROMPT, add_generation_prompt=True, tokenize=False)
        prompt_ids = llm.tokenizer.encode(prompt_ids, add_special_tokens=False)
        captured.clear()
        res = llm.generate([prompt_ids], SamplingParams(temperature=0.0, max_tokens=n_new, ignore_eos=True))
    finally:
        llm.shutdown()
    out_ids = res[0]["token_ids"]
    # A chunked prefill also samples at its non-final chunks and the overlap scheduler may run
    # one step past the end, so locate the window whose argmax is exactly the emitted ids.
    n, want = len(out_ids), torch.tensor(out_ids)
    tops = torch.cat(captured).argmax(-1)
    starts = [force_start] if forced else [k for k in range(len(captured) - n + 1) if torch.equal(tops[k:k + n], want)]
    assert starts, f"no window of {len(captured)} samples reproduces the {n} greedy ids: {tops.tolist()} vs {out_ids}"
    logits = torch.cat(captured[starts[-1]:starts[-1] + n])
    print(f"{len(captured)} sampler calls; emitted window starts at {starts[-1]}")
    torch.save({"prompt_ids": prompt_ids, "output_ids": out_ids, "logits": logits, "argv": flags}, out)
    print(f"saved {len(prompt_ids)} prompt + {len(out_ids)} new tokens to {out}")
    print("text:", repr(llm.tokenizer.decode(out_ids)))


if __name__ == "__main__":
    sep = sys.argv.index("--")
    main(sys.argv[1], int(sys.argv[2]), sys.argv[sep + 1:])
