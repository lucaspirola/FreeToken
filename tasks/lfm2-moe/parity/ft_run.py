"""FreeToken side of the lfm2_moe parity check (FreeToken venv, owns the GPU while it runs).

    python ft_run.py <out.pt> <n_new> [--ref exl3.pt] -- <ft serve flags>

Runs the offline engine on a fixed chat prompt, decodes n_new tokens greedily and saves the
sampler's logits per step (as tasks/ornith-exl3/compare/ft_logits.py). During the prompt's
prefill forward it also records, per decoder layer, the input and output of the mixer (conv
or attention) and of the feed-forward (dense or MoE), and the residual stream after the layer.
With --ref, each mixer / feed-forward is ALSO run on exllamav3's input for that sub-module
(isolated parity); the chained call runs last, so the engine's state stays the chained one.
"""
import dataclasses
import os
import sys

import torch

_FILLER = "The quick brown fox jumps over the lazy dog while the committee reviews the budget. " * 8
PROMPT = [{"role": "user", "content": _FILLER * int(os.environ.get("FT_PROMPT_REPEAT", "0"))
           + "Write three short sentences about the history of the bicycle."}]


def main(out, n_new, ref_path, flags):
    from freetoken.core import SamplingParams, get_global_ctx
    from freetoken.engine import sample as sample_mod
    from freetoken.llm.llm import LLM
    from freetoken.scheduler import SchedulerConfig
    from freetoken.server.args import parse_args

    sa, _ = parse_args(flags)
    skip = {"model_path", "tp_info", "dtype", "offline_mode"}
    kwargs = {f.name: getattr(sa, f.name) for f in dataclasses.fields(SchedulerConfig) if f.init and f.name not in skip}
    captured = []
    real_sample = sample_mod.Sampler.sample

    def sample(self, logits, args):
        captured.append(logits[:1].float().cpu())
        return real_sample(self, logits, args)

    sample_mod.Sampler.sample = sample
    llm = LLM(sa.model_path, dtype=sa.dtype, **kwargs)
    ref = torch.load(ref_path) if ref_path else None
    rec = {"mixer_in": {}, "mixer_out": {}, "ffn_in": {}, "ffn_out": {}, "stream": {}, "iso_mixer": {}, "iso_ffn": {}}
    prompt_len = [0]

    def recording():
        b = get_global_ctx().batch
        return (not b.is_decode) and int(b.input_ids.numel()) == prompt_len[0]

    def wrap(op, i, kind):
        orig = op.forward

        def fwd(x, *a, **k):
            if not recording() or i in rec[f"{kind}_out"]:
                return orig(x, *a, **k)
            if ref is not None:
                r = ref[f"{kind}_in"][i].to(device=x.device, dtype=x.dtype)
                assert r.shape == x.shape, (kind, i, r.shape, x.shape)
                rec[f"iso_{kind}"][i] = orig(r, *a, **k).float().cpu()
            y = orig(x, *a, **k)
            rec[f"{kind}_in"][i] = x.float().cpu()
            rec[f"{kind}_out"][i] = y.float().cpu()
            return y

        op.forward = fwd

    def wrap_layer(layer, i):
        orig = layer.forward

        def fwd(x, residual=None):
            y, res = orig(x, residual)
            if recording() and i not in rec["stream"]:
                rec["stream"][i] = (y.float() + res.float()).cpu()
            return y, res

        layer.forward = fwd

    model = llm.engine.model
    for i, layer in enumerate(model.model.layers.op_list):
        wrap(layer.conv if layer._is_conv else layer.self_attn, i, "mixer")
        wrap(layer.feed_forward, i, "ffn")
        wrap_layer(layer, i)
    try:
        text = llm.tokenizer.apply_chat_template(PROMPT, add_generation_prompt=True, tokenize=False)
        prompt_ids = llm.tokenizer.encode(text, add_special_tokens=False)
        prompt_len[0] = len(prompt_ids)
        captured.clear()
        res = llm.generate([prompt_ids], SamplingParams(temperature=0.0, max_tokens=n_new, ignore_eos=True))
    finally:
        llm.shutdown()
    out_ids = res[0]["token_ids"]
    n, want = len(out_ids), torch.tensor(out_ids)
    tops = torch.cat(captured).argmax(-1)
    starts = [k for k in range(len(captured) - n + 1) if torch.equal(tops[k:k + n], want)]
    assert starts, f"no window of {len(captured)} samples reproduces the {n} greedy ids"
    logits = torch.cat(captured[starts[-1]:starts[-1] + n])
    if len(rec["stream"]) != len(model.model.layers.op_list):
        print(f"no per-layer record: the prompt was not prefilled in one forward ({sorted(rec['stream'])})")
    torch.save({"prompt_ids": prompt_ids, "output_ids": out_ids, "logits": logits, "layers": rec,
                "kinds": ["conv" if l._is_conv else "attn" for l in model.model.layers.op_list],
                "argv": flags}, out)
    print(f"{len(captured)} sampler calls; window at {starts[-1]}; {len(prompt_ids)} prompt + {n} new tokens -> {out}")
    print("text:", repr(llm.tokenizer.decode(out_ids)))


if __name__ == "__main__":
    sep = sys.argv.index("--")
    head = sys.argv[1:sep]
    ref = None
    if "--ref" in head:
        k = head.index("--ref")
        ref = head[k + 1]
        del head[k:k + 2]
    main(head[0], int(head[1]), ref, sys.argv[sep + 1:])
