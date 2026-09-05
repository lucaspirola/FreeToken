"""Go/no-go probe for prompt-lookup (n-gram) speculative decoding on an offload MoE.

Two quantities decide whether n-gram speculation can pay on this host, and both are
measurable WITHOUT implementing the verify step:

1. **lambda(k), the mean accepted length.** Under greedy decoding a prompt-lookup drafter
   is verified against exactly the greedy continuation, so the acceptance distribution is
   a deterministic function of an ordinary greedy transcript. This script records real
   transcripts as *token ids* (the offline ``LLM`` returns ids; the HTTP API does not) and
   ``benchmarks/ngram_spec_analysis.py`` replays the drafter over them.

2. **touched(m), the number of distinct experts a m-token verify step activates per MoE
   layer.** Task 2B4 measured this for two *independent* sequences (11.61 of 12) and noted
   that consecutive tokens of ONE sequence should be more correlated -- but did not
   measure it. A verify step's m tokens are consecutive tokens of one stream, which is
   exactly what bs=1 decode produces, so hooking the router during an ordinary greedy
   generation and taking sliding-window unions over the recorded top-k sets gives
   touched(m) directly. No speculative code required.

Writes a transcripts JSONL and a routing summary JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch


def build_prompts(repo: Path) -> dict[str, str]:
    """Three prompt classes: code-heavy, prose, and copy-heavy agent tool output."""
    code_ctx = (repo / "python/freetoken/engine/sample.py").read_text()
    prose = (
        "Write a careful, self-contained essay of at least 700 words on why memory "
        "bandwidth, not arithmetic throughput, has become the binding constraint on "
        "single-stream inference for large language models. Use concrete numbers where "
        "you can and avoid bullet lists."
    )
    code = (
        "Write a complete, production-quality Python module `ringbuf.py` implementing a "
        "lock-free single-producer single-consumer ring buffer over a preallocated "
        "`bytearray`, with `try_push`, `try_pop`, `__len__`, capacity rounding to a power "
        "of two, and full docstrings. Then write pytest tests for it. Output code only."
    )
    # The canonical agent-session shape: a tool has dumped a file into the context and the
    # model is asked to emit it back with a small edit. Most output tokens are copies of
    # prompt tokens, which is precisely where prompt lookup should win if it ever does.
    copy = (
        "Here is the current contents of `sample.py`:\n\n```python\n"
        + code_ctx
        + "\n```\n\nRename the function `sample_impl` to `sample_logits_impl` everywhere "
        "it appears, and output the COMPLETE updated file inside one ```python fence. "
        "Do not abbreviate or elide any part of the file."
    )
    return {"code": code, "prose": prose, "copy": copy}


def render_prompt(tok, content: str) -> list[int]:
    """Chat-template the prompt and return token ids.

    ``apply_chat_template`` returns a str, a list, a nested list or a tensor depending on
    the tokenizer wrapper, so normalise all four rather than assume one.
    """
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=True
    )
    for _ in range(3):
        if isinstance(rendered, str):
            rendered = tok(rendered, add_special_tokens=False)
            continue
        if hasattr(rendered, "keys") and "input_ids" in rendered:
            rendered = rendered["input_ids"]
            continue
        if hasattr(rendered, "tolist"):
            rendered = rendered.tolist()
            continue
        if isinstance(rendered, (list, tuple)) and rendered and isinstance(rendered[0], (list, tuple)):
            rendered = rendered[0]
            continue
        break
    return [int(x) for x in rendered]


class RouterProbe:
    """Records the routed top-k expert ids of every token of every MoE layer.

    Hooks ``NemotronHMoE.forward`` rather than the router module so the recorded ids are
    the ones the expert gather actually uses (post-bias top-k), and keeps only decode
    batches (one token per running request) so the recorded sequence is the stream's own
    consecutive tokens.
    """

    def __init__(self, decode_only: bool = False):
        self.decode_only = decode_only
        self.rows: list[torch.Tensor] = []
        self.layer_ids: list[int] = []
        self._orig = None

    def install(self) -> None:
        from freetoken.core import get_global_ctx
        from freetoken.models.nemotron_h.model import NemotronHMoE

        probe = self
        orig = NemotronHMoE.forward
        self._orig = orig

        def patched(self, x):  # noqa: ANN001
            scores, choice = self.gate.forward(x)
            ids = torch.topk(choice, self.top_k, dim=-1, sorted=False).indices
            batch = get_global_ctx().batch
            if (not probe.decode_only) or batch.is_decode:
                probe.rows.append(ids.detach().to("cpu", torch.int16, non_blocking=False))
                probe.layer_ids.append(id(self))
            weights = scores.gather(1, ids)
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
            weights = (weights * self.scale).float()
            latent = self.fc1_latent_proj.forward(x) if self.has_latent else x
            routed = self.experts.routed_forward(latent, weights, ids.to(torch.int32)).to(x.dtype)
            if self.has_latent:
                routed = self.fc2_latent_proj.forward(routed)
            return routed + self.shared_experts.forward(x)

        NemotronHMoE.forward = patched

    def remove(self) -> None:
        from freetoken.models.nemotron_h.model import NemotronHMoE

        if self._orig is not None:
            NemotronHMoE.forward = self._orig
            self._orig = None

    def per_layer_streams(self) -> dict[int, list[list[int]]]:
        """Regroup the flat capture into one token-ordered list of expert sets per layer."""
        streams: dict[int, list[list[int]]] = {}
        for key, ids in zip(self.layer_ids, self.rows):
            arr = ids.tolist()
            streams.setdefault(key, []).extend(row for row in arr)
        return streams


def touched_curve(streams: dict[int, list[list[int]]], widths: list[int]) -> dict:
    """Mean distinct experts per layer over sliding windows of m consecutive tokens,
    and, as the independent-token control, over m tokens sampled far apart."""
    import random

    rng = random.Random(0)
    out = {"consecutive": {}, "independent": {}}
    for m in widths:
        cons, indep = [], []
        for rows in streams.values():
            n = len(rows)
            if n < m:
                continue
            for i in range(0, n - m + 1):
                cons.append(len(set().union(*(set(r) for r in rows[i : i + m]))))
            for _ in range(min(256, n)):
                picks = rng.sample(range(n), m) if n >= m else range(n)
                indep.append(len(set().union(*(set(rows[j]) for j in picks))))
        out["consecutive"][m] = sum(cons) / len(cons) if cons else None
        out["independent"][m] = sum(indep) / len(indep) if indep else None
    return out


class ForwardTimer:
    """CUDA-event timing of every eager model forward, keyed on the batch's token count.

    The offline ``generate()`` call carries ~280 ms of fixed harness cost (tokenize,
    scheduler round trip, detokenize), which is 40x the quantity being measured, so wall
    clock around a one-step request cannot see a 2 ms difference. Events around the model
    forward are overhead-free. Decode steps are CUDA-graph replays and are not hooked --
    their cost is measured separately from a long generation's steady-state rate.
    """

    def __init__(self):
        self.samples: list[tuple[tuple, torch.cuda.Event, torch.cuda.Event]] = []
        self.host: dict[tuple, list[float]] = {}
        self._orig = None

    def install(self):
        from freetoken.core import get_global_ctx
        from freetoken.models.nemotron_h.model import NemotronHForCausalLM

        timer = self
        orig = NemotronHForCausalLM.forward
        self._orig = orig

        def patched(self):
            batch = get_global_ctx().batch
            ntok = (int(batch.input_ids.numel()), "decode" if batch.is_decode else "extend")
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            t0 = time.perf_counter()
            out = orig(self)
            host_ms = 1000 * (time.perf_counter() - t0)
            end.record()
            timer.samples.append((ntok, start, end))
            timer.host.setdefault(ntok, []).append(host_ms)
            return out

        NemotronHForCausalLM.forward = patched

    def remove(self):
        from freetoken.models.nemotron_h.model import NemotronHForCausalLM

        if self._orig is not None:
            NemotronHForCausalLM.forward = self._orig
            self._orig = None

    def by_tokens(self) -> dict[tuple, list[float]]:
        torch.cuda.synchronize()
        out: dict[tuple, list[float]] = {}
        for ntok, start, end in self.samples:
            out.setdefault(ntok, []).append(start.elapsed_time(end))
        return out


def measure_verify_cost(llm, base: list[int], tails: list[list[int]], widths: list[int],
                        repeats: int, decode_steps: int) -> dict:
    """Wall cost of an m-token extend on a warm session, i.e. a verify step's real price.

    A verify step is exactly this shape: one running request contributing m query tokens
    against its own already-cached KV and recurrent state. The prefix cache reproduces it
    with no engine change -- resend the cached prefix plus m fresh tokens and the engine
    forwards only those m. Every cost the model cannot see is in here: the eager (ungraphed)
    launch of 52 blocks, the extend attention kernel instead of the split-K decode kernel,
    the chunked SSD scan instead of the decode SSU, and the expert misses the m tokens add.

    Each width gets a DISTINCT continuation so the radix tree matches ``base`` and nothing
    longer; reusing one continuation would let width m+1 match width m's cached tokens.
    """
    from freetoken.core import SamplingParams

    one = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
    llm.generate([list(base)], one)  # seat `base` in the radix tree

    # Baseline graphed decode step: (wall at D+1 tokens - wall at 1 token) / D.
    t = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        llm.generate([list(base)], one)
        t.append(time.perf_counter() - t0)
    t_zero = min(t)
    t = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        llm.generate([list(base)], SamplingParams(temperature=0.0, max_tokens=decode_steps + 1,
                                                  ignore_eos=True))
        t.append(time.perf_counter() - t0)
    decode_ms = 1000 * (min(t) - t_zero) / decode_steps

    timer = ForwardTimer()
    # Like-for-like: the SAME hook over the decode path. With CUDA graphs on this records
    # only the prefill forwards (a graph replay never enters Python); with FT_PROBE_EAGER=1
    # it records the decode forwards too, which is what isolates the extend path's cost.
    timer.install()
    llm.generate([list(base)], SamplingParams(temperature=0.0, max_tokens=decode_steps + 1,
                                              ignore_eos=True))
    timer.remove()
    baseline = {f"{k[0]}/{k[1]}": round(min(v), 3) for k, v in sorted(timer.host.items())}

    out = {
        "decode_ms": round(decode_ms, 3),
        "t_zero_ms": round(1000 * t_zero, 3),
        "baseline_host_ms": baseline,
        "widths": {},
    }
    for i, m in enumerate(widths):
        tail = tails[i % len(tails)][:m]
        if len(tail) < m:
            continue
        prompt = list(base) + list(tail)
        llm.generate([prompt], one)  # warm this width's kernels/experts
        timer.samples.clear()
        timer.host.clear()
        timer.install()
        t = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            llm.generate([prompt], one)
            t.append(time.perf_counter() - t0)
        timer.remove()
        buckets = timer.by_tokens()
        gpu = sorted(buckets.get((m, "extend"), []))
        ms = 1000 * min(t)
        # ``marginal_ms`` is overhead-free: the offline generate() call, tokenization,
        # scheduling and detokenization are identical in both terms and cancel. The
        # absolute verify cost is that marginal plus one eager 1-token forward, which is
        # measured separately (CUDA graphs off) rather than inferred from here.
        out["widths"][m] = {
            "wall_ms": round(ms, 3),
            "marginal_ms": round(ms - 1000 * t_zero, 3),
            "gpu_ms_min": round(gpu[0], 3) if gpu else None,
            "gpu_ms_median": round(gpu[len(gpu) // 2], 3) if gpu else None,
            "gpu_n": len(gpu),
            "gpu_buckets": {f"{k[0]}/{k[1]}": round(min(v), 3) for k, v in sorted(buckets.items())},
            "host_ms_min": {f"{k[0]}/{k[1]}": round(min(v), 3) for k, v in sorted(timer.host.items())},
        }
        print("m", m, out["widths"][m], flush=True)
    return out


def measure_chunk_cost(model_kwargs: dict, pool: list[int], widths: list[int],
                       total_tokens: int) -> dict:
    """Cost of an m-token extend step, measured by chunking a prefill into m-token steps.

    A verify step IS an m-token extend on a running request: same extend attention kernel,
    same varlen conv + chunked SSD scan, same eager (ungraphed) launch, same expert routing
    over m consecutive tokens. Setting ``max_extend_tokens = m`` makes an N-token prefill
    run N/m of them back to back inside ONE offline call, so the ~280 ms fixed cost of a
    ``generate()`` round trip amortises to nothing instead of burying a 2 ms signal.

    A fresh engine per width, because ``max_extend_tokens`` is construction-time.
    """
    from freetoken.core import SamplingParams
    from freetoken.llm.llm import LLM

    one = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
    out: dict[int, dict] = {}
    for m in widths:
        kwargs = dict(model_kwargs)
        kwargs["max_extend_tokens"] = m
        llm = LLM(**kwargs)
        try:
            # Warm kernels/autotune/expert cache on a disjoint prompt of the same shape.
            llm.generate([pool[total_tokens : 2 * total_tokens]], one)
            t = []
            for rep in range(2):
                # A distinct prompt per repeat: the radix tree would otherwise serve the
                # second one from cache and measure nothing.
                lo = 2 * total_tokens + rep * total_tokens
                t0 = time.perf_counter()
                llm.generate([pool[lo : lo + total_tokens]], one)
                t.append(time.perf_counter() - t0)
            wall = min(t)
            steps = total_tokens / m
            out[m] = {
                "wall_s": round(wall, 4),
                "steps": round(steps, 1),
                "ms_per_step": round(1000 * wall / steps, 4),
                "ms_per_token": round(1000 * wall / total_tokens, 4),
            }
            print("m", m, out[m], flush=True)
        finally:
            del llm
            torch.cuda.empty_cache()
    return out


class LayerTimer:
    """Host-side wall time inside each mixer kind, for extend forwards only.

    Attributes the extend path's fixed per-forward cost to Mamba-2 / attention / MoE
    without a profiler: the cost is host-side and serialized, so ``perf_counter`` around
    each mixer's ``forward`` sums to the model forward.
    """

    KINDS = ("mamba", "attention", "moe")

    def __init__(self):
        self.totals: dict[str, list[float]] = {k: [] for k in self.KINDS}
        self._orig: dict[str, object] = {}
        self._acc: dict[str, float] = {}

    def install(self):
        from freetoken.core import get_global_ctx
        from freetoken.models.nemotron_h import model as M

        timer = self
        pairs = {
            "mamba": M.NemotronHMamba2Mixer,
            "attention": M.NemotronHAttention,
            "moe": M.NemotronHMoE,
        }
        for kind, cls in pairs.items():
            orig = cls.forward
            timer._orig[kind] = orig

            def make(kind=kind, orig=orig):
                def patched(self, x):
                    if get_global_ctx().batch.is_decode:
                        return orig(self, x)
                    t0 = time.perf_counter()
                    out = orig(self, x)
                    timer._acc[kind] = timer._acc.get(kind, 0.0) + (time.perf_counter() - t0)
                    return out

                return patched

            cls.forward = make()

    def remove(self):
        from freetoken.models.nemotron_h import model as M

        for kind, cls in (("mamba", M.NemotronHMamba2Mixer), ("attention", M.NemotronHAttention),
                          ("moe", M.NemotronHMoE)):
            if kind in self._orig:
                cls.forward = self._orig[kind]
        self._orig.clear()

    def take(self) -> dict[str, float]:
        out = {k: round(1000 * v, 3) for k, v in self._acc.items()}
        self._acc.clear()
        return out


def profile_extend_layers(llm, base: list[int], tails: list[list[int]], widths: list[int],
                          repeats: int) -> dict:
    from freetoken.core import SamplingParams

    one = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
    llm.generate([list(base)], one)
    out = {}
    timer = LayerTimer()
    ftimer = ForwardTimer()
    for i, m in enumerate(widths):
        prompt = list(base) + list(tails[i % len(tails)][:m])
        llm.generate([prompt], one)
        best = None
        for _ in range(repeats):
            ftimer.samples.clear()
            ftimer.host.clear()
            timer.take()
            ftimer.install()
            timer.install()
            llm.generate([prompt], one)
            timer.remove()
            ftimer.remove()
            parts = timer.take()
            total = min(min(v) for v in ftimer.host.values()) if ftimer.host else None
            if total is not None and (best is None or total < best[0]):
                best = (total, parts)
        out[m] = {"forward_host_ms": round(best[0], 3), "by_mixer_ms": best[1]}
        print("m", m, out[m], flush=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--widths", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 8, 9, 12, 17])
    p.add_argument("--moe-cache-rate", type=float, default=None)
    p.add_argument("--moe-cache-auto", action="store_true")
    p.add_argument("--memory-ratio", type=float, default=0.85)
    p.add_argument("--num-tokens", type=int, default=65536)
    p.add_argument("--max-seq-len", type=int, default=65536)
    p.add_argument("--host-ram-reserve-gb", type=float, default=6.0)
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--layer-profile", action="store_true",
                   help="attribute the extend forward host cost to each mixer kind")
    p.add_argument("--chunk-cost-from", default=None,
                   help="measure m-token extend step cost by chunking a prefill")
    p.add_argument("--chunk-total-tokens", type=int, default=4096)
    p.add_argument("--verify-cost-from", default=None,
                   help="measure verify-step wall cost using this transcripts JSONL as token source")
    p.add_argument("--verify-base-tokens", type=int, default=2048)
    p.add_argument("--verify-repeats", type=int, default=7)
    p.add_argument("--routing-from", default=None,
                   help="skip generation; capture routing by re-prefilling this transcripts JSONL")
    args = p.parse_args()

    from freetoken.core import SamplingParams
    from freetoken.llm.llm import LLM

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parent.parent

    kwargs = dict(
        model_path=args.model,
        dtype=torch.bfloat16,
        attention_backend="triton",
        moe_backend="offload",
        nvfp4_backend="triton",
        max_running_req=1,
        max_extend_tokens=8192,
        memory_ratio=args.memory_ratio,
        max_seq_len_override=args.max_seq_len,
        num_token_override=args.num_tokens,
        kv_cache_dtype="q8_0",
        host_ram_reserve_gb=args.host_ram_reserve_gb,
        cuda_graph_bs=([] if os.environ.get("FT_PROBE_EAGER") else None),
        cuda_graph_max_bs=1,
        session_spill_dir=None,
    )
    if args.moe_cache_auto:
        kwargs["moe_cache_auto"] = True
    if args.moe_cache_rate is not None:
        kwargs["moe_cache_rate"] = args.moe_cache_rate

    llm = None if args.chunk_cost_from else LLM(**kwargs)
    tok = llm.tokenizer if llm is not None else None
    prompts = build_prompts(repo)
    names = args.only or list(prompts)

    # Natural stopping: ignore_eos would let the model run past EOS into repetition,
    # which inflates n-gram acceptance with text no real request would decode.
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, ignore_eos=False)
    transcripts = out_dir / "transcripts.jsonl"
    routing_out = out_dir / "routing.json"
    timing: dict[str, dict] = {}
    routing: dict[str, dict] = {}

    if args.chunk_cost_from:
        rows = [json.loads(x) for x in Path(args.chunk_cost_from).read_text().splitlines() if x.strip()]
        pool: list[int] = []
        while len(pool) < 4 * args.chunk_total_tokens:
            for row in rows:
                pool += list(row["prompt_tokens"]) + list(row["output_tokens"])
        res = measure_chunk_cost(kwargs, pool, args.widths, args.chunk_total_tokens)
        (out_dir / "chunk_cost.json").write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))
        return

    if args.verify_cost_from:
        rows = [json.loads(x) for x in Path(args.verify_cost_from).read_text().splitlines() if x.strip()]
        src = rows[0]
        pool = list(src["prompt_tokens"]) + list(src["output_tokens"])
        base = pool[: args.verify_base_tokens]
        rest = pool[args.verify_base_tokens :]
        widths = [m for m in args.widths]
        # One distinct continuation per width, so each matches `base` and nothing longer.
        tails, off = [], 0
        for m in widths:
            tails.append(rest[off : off + m])
            off += max(m, 1)
        if args.layer_profile:
            res = profile_extend_layers(llm, base, tails, widths, args.verify_repeats)
            (out_dir / "layer_profile.json").write_text(json.dumps(res, indent=2))
            print(json.dumps(res, indent=2))
            return
        res = measure_verify_cost(llm, base, tails, widths, args.verify_repeats, 64)
        res["base_tokens"] = len(base)
        (out_dir / "verify_cost.json").write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))
        return

    if args.routing_from:
        rows = [json.loads(x) for x in Path(args.routing_from).read_text().splitlines() if x.strip()]
        for row in rows:
            name = row["name"]
            if args.only and name not in args.only:
                continue
            ids = list(row["prompt_tokens"]) + list(row["output_tokens"])
            gen_start = len(row["prompt_tokens"])
            probe = RouterProbe(decode_only=False)
            probe.install()
            llm.generate([ids], SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True))
            probe.remove()
            streams = probe.per_layer_streams()
            # Drop the prompt positions: only the generated stream is what a verify step
            # would ever cover.
            streams = {k: v[gen_start:] for k, v in streams.items()}
            routing[name] = touched_curve(streams, args.widths)
            routing[name]["moe_layers"] = len(streams)
            routing[name]["tokens_per_layer"] = len(next(iter(streams.values()))) if streams else 0
            print(name, "layers", len(streams), "tokens",
                  routing[name]["tokens_per_layer"], flush=True)
            (out_dir / "routing.json").write_text(json.dumps(routing, indent=2))
        print(json.dumps(routing, indent=2))
        return

    with transcripts.open("w") as fh:
        for name in names:
            ids = render_prompt(tok, prompts[name])
            # Warm the expert cache and the graphs on this prompt before measuring.
            llm.generate([list(ids)], SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True))
            t0 = time.perf_counter()
            res = llm.generate([list(ids)], sp)
            wall = time.perf_counter() - t0
            out_ids = res[0]["token_ids"]
            fh.write(
                json.dumps({"name": name, "prompt_tokens": list(ids), "output_tokens": out_ids})
                + "\n"
            )
            timing[name] = {
                "prompt_tokens": len(ids),
                "output_tokens": len(out_ids),
                "wall_s": round(wall, 3),
                "decode_tok_s": round(len(out_ids) / wall, 2) if wall else None,
                "ms_per_step_incl_prefill": round(1000 * wall / max(len(out_ids), 1), 3),
            }
            print(name, timing[name], flush=True)
            (out_dir / "timing.json").write_text(json.dumps(timing, indent=2))

    print(json.dumps({"timing": timing}, indent=2))
    print("wrote", transcripts, routing_out)


if __name__ == "__main__":
    main()
