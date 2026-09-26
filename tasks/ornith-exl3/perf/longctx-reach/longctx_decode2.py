"""Long-context decode driver (any model): where does decode time go as the context grows?

    longctx_decode.py N_CTX N_GEN OUT_JSON -- <ft serve flags>

It runs in-process, through the scheduler's overlap loop, as ft serve runs it. For one context size N_CTX:

  warm   8K prompt, 8 tokens
  pass1  probe prompt (scripts/probe_decode.py's text and chat framing) at N_CTX, N_GEN tokens. This pass
         grows the KV; it is timed, not of record.
  pass2  the same with a fresh prefix. Timed, OF RECORD: steady decode tok/s from per-token arrival times.
         Also records the mirror counters (swaps = decode misses) between the first and the last token.
  pass3  the same with another prefix, under nsys. cudaProfilerStart is called at the FIRST token, so the
         capture holds decode only; cudaProfilerStop is called when the request ends. Run it under
         `nsys profile --capture-range=cudaProfilerApi --cuda-graph-trace=node` (job-longctx.sh), then
         split it with longctx_split.py.
  pass4  natural text (this tree's Python sources) at N_CTX. Timed, with its own decode miss counters, to
         check whether the probe's repeated sentence flatters the expert cache.

"Decode misses" are the mirror's SWAPS counter differenced between the first token and the end, so
prefill materialization is excluded (--expert-residency mirror only). The hit rate is
1 - swaps / (steps * moe_layers * top_k). --moe-collect-stats is NOT used: it adds per-batch timing events
and would perturb the timeline.
"""
import hashlib
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROBE_SENT = "The quick brown fox jumps over the lazy dog near the riverbank while the miller counts his sacks of grain. "
PLACE = "\x00FILL\x00"


def build(flags):
    import dataclasses

    from freetoken.llm.llm import LLM
    from freetoken.scheduler import SchedulerConfig
    from freetoken.server.args import parse_args

    sa, _ = parse_args(flags)
    skip = {"model_path", "tp_info", "dtype", "offline_mode"}
    kw = {f.name: getattr(sa, f.name) for f in dataclasses.fields(SchedulerConfig) if f.init and f.name not in skip}
    return LLM(sa.model_path, dtype=sa.dtype, **kw)


def framed(tok, head, filler, tail, n):
    """Chat-templated prompt of exactly n tokens: head + filler (truncated) + tail, thinking off."""
    msgs = [{"role": "user", "content": head + PLACE + tail}]
    try:
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    pre, post = s.split(PLACE)
    enc = lambda t: tok.encode(t, add_special_tokens=False)  # noqa: E731
    a, b = enc(pre), enc(post)
    body = enc(filler)
    need = n - len(a) - len(b)
    assert len(body) >= need, f"filler too short: {len(body)} < {need}"
    return a + body[:need] + b


def corpus(root, n_chars):
    parts, got = [], 0
    for d, _, files in sorted(os.walk(root)):
        for f in sorted(files):
            if f.endswith(".py"):
                t = open(os.path.join(d, f), encoding="utf-8", errors="replace").read()
                parts.append(f"# file {os.path.relpath(os.path.join(d, f), root)}\n{t}\n")
                got += len(parts[-1])
                if got >= n_chars:
                    return "".join(parts)
    return "".join(parts)


def moe_shape(llm):
    cache = llm.engine.moe_offload_cache
    layers = getattr(cache, "num_layers", None) if cache is not None else None
    cfg = getattr(llm.engine, "model_config", None) or llm.engine.config.model_config
    top_k = None
    for obj in (cfg, getattr(cfg, "hf_config", None), getattr(cfg, "hf_text_config", None)):
        for name in ("num_experts_per_tok", "moe_topk", "top_k", "num_experts_per_token"):
            v = getattr(obj, name, None) if obj is not None else None
            if isinstance(v, int):
                top_k = v
                break
        if top_k:
            break
    return layers, top_k


def main(n_ctx, n_gen, out_json, flags):
    from freetoken.core import SamplingParams
    from freetoken.message import DetokenizeMsg

    llm = build(flags)
    tok = llm.tokenizer
    cache = llm.engine.moe_offload_cache
    layers, top_k = moe_shape(llm)
    stamps, marks = [], {}
    orig = llm.send_result

    def mirror():
        try:
            return dict(cache.mirror_stats()) if cache is not None else {}
        except Exception as e:  # noqa: BLE001
            return {"error": repr(e)}

    def hook(reply):
        now = time.perf_counter()
        n = sum(1 for m in reply if isinstance(m, DetokenizeMsg))
        if n:
            if not stamps and marks.get("at_first") is not None:
                marks["at_first"]()
            stamps.extend([now] * n)
        return orig(reply)

    llm.send_result = hook
    sp = SamplingParams(temperature=0.0, max_tokens=n_gen, ignore_eos=True)
    probe = lambda tag: framed(tok, f"Run {tag}{n_ctx}. ", PROBE_SENT * (n_ctx // 20 + 16),  # noqa: E731
                               "\n\nWrite a long story about the fox.", n_ctx)
    # LONGCTX_CORPUS pins the natural-text corpus, so an A/B of two trees reads the same prompt.
    src = os.environ.get("LONGCTX_CORPUS") or os.path.join(os.environ.get("PYTHONPATH", ".").split(":")[0], "freetoken")
    natural = framed(tok, "Here is a Python code base.\n\n", corpus(src, 6 * n_ctx),
                     "\n\nSummarize what this code base does, file by file.", n_ctx)
    res = {"n_ctx": n_ctx, "n_gen": n_gen, "moe_layers": layers, "top_k": top_k, "passes": {}}

    def run(name, ids, nsys=False):
        stamps.clear()
        snap = {}

        def at_first():
            snap["first"] = mirror()  # one device read at the first token (prefill done)
            if nsys:
                torch.cuda.synchronize()
                torch.cuda.profiler.start()
        marks["at_first"] = at_first
        t0 = time.perf_counter()
        out = llm.generate([ids], sp)
        torch.cuda.synchronize()
        if nsys:
            torch.cuda.profiler.stop()
        end = mirror()
        marks["at_first"] = None
        toks = len(out[0]["token_ids"])
        ttft = stamps[0] - t0 if stamps else None
        dec = (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 2 and stamps[-1] > stamps[0] else None
        first = snap.get("first", {})
        delta = {k: end[k] - first.get(k, 0) for k in end if isinstance(end.get(k), int) and isinstance(first.get(k), int)}
        steps = max(toks - 1, 1)
        r = {"prompt_tokens": len(ids), "tokens": toks, "ttft_s": ttft, "prefill_tok_s": len(ids) / ttft if ttft else None,
             "decode_tok_s": dec, "decode_ms_per_token": 1e3 / dec if dec else None, "mirror_decode_delta": delta}
        r["token_ids"] = [int(t) for t in out[0]["token_ids"]]
        r["prompt_sha1"] = hashlib.sha1(str(list(ids)).encode()).hexdigest()[:10]
        r["out_sha1"] = hashlib.sha1(str(r["token_ids"]).encode()).hexdigest()[:10]
        if layers and top_k and "swaps" in delta:
            r["decode_swaps_per_token"] = delta["swaps"] / steps
            r["decode_miss_rate"] = delta["swaps"] / (steps * layers * top_k)
            r["decode_hit_rate"] = 1 - r["decode_miss_rate"]
        res["passes"][name] = r
        print(f"{name}: {json.dumps(r)}", flush=True)

    try:
        run("warm", framed(tok, "Warm up. ", PROBE_SENT * 600, "\n\nWrite a long story about the fox.", 8000))
        run("pass1", probe("a"))
        run("pass2", probe("b"))
        run("pass3_nsys", probe("c"), nsys=True)
        run("pass4_natural", natural)
    finally:
        llm.shutdown()
        json.dump(res, open(out_json, "w"), indent=1)


if __name__ == "__main__":
    sep = sys.argv.index("--")
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[sep + 1:])
