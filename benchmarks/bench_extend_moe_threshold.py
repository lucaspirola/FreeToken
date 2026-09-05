"""Decide ``--moe-extend-cache-tokens`` against the scheduler's real chunk sizes.

The gate (``OffloadMoeCache.use_cached_extend``) sends an extend forward of at most
``extend_cache_tokens`` tokens down the DECODE slot cache -- gather only the experts the
tokens route to, multiply with the decode GEMV -- and everything larger down the full-layer
prefill stream, which moves every expert of every layer over PCIe (23 x 128 x ~5.35 MiB =
15.4 GiB, ~290 ms on Nemotron 3.5 Lightning) whatever the token count. The default is 64,
chosen from a sweep that stopped at m = 32.

The scheduler's smallest *normal* chunk is not 32. Under an interleave share
``PrefillScheduler`` divides the token budget by the seatable lanes
(``scheduler/prefill.py``, ``chunk_limit = adder.token_budget // seatable``), which on the
16-lane Switchyard profile lands at 512 tokens. Those chunks pay the whole 15.4 GiB stream
to compute over ~128 routed experts per layer. The cached path would move only the misses --
and would evict the decode working set once per chunk. Which one wins, and where the
crossover is, is the question this driver answers.

One model load; for each m and each arm it measures the extend forward, splits it across the
mixer kinds, and then runs a plain decode burst so the *other* half of the question -- what
the cached arm costs the following decode's working set -- is on the same row.

Two traps this driver is built around, both paid for once already:

* **The radix tree serves the prompt.** An m-token extend is reproduced by re-sending a
  seated prefix plus m fresh tokens. Reuse a tail and the second call forwards ONE token
  while the harness happily labels it "m". Every timed call therefore gets its own tail,
  each tail starts with an id no other tail starts with, and the token count that actually
  reached the model forward is read back out of the forward hook and asserted.
* **The arm is not what the flag says.** ``use_cached_extend`` has six other terms (format,
  decode target, size classes, CPU layers, unpinned layers). The gate is wrapped in a
  counting proxy and each row carries its own ``hits/calls`` path proof; a cached arm with
  zero hits, or a stream arm with any, fails the run.

Invoke through ``benchmarks/extend_moe/run_threshold.sh`` under ``scripts/gpu_lock.sh``.
Never pipe gpu_lock.sh into anything: its exit trap runs ``pkill -9 -g $$``.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import date
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_ngram_spec import ForwardTimer, LayerTimer  # noqa: E402

ARMS = ("stream", "cached")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class GateProbe:
    """Counting proxy over ``use_cached_extend``: the per-row path proof.

    Records ``(layer_id, num_tokens, decision)`` for every MoE layer of every extend
    forward. ``num_tokens`` is the second independent witness that the forward carried m
    tokens (the first is the ``ForwardTimer`` bucket), and ``decision`` is what proves the
    arm -- the flag alone does not, because the gate has six other terms.
    """

    def __init__(self, cache):
        self.cache = cache
        self._unbound = type(cache).use_cached_extend
        self.calls: list[tuple[int, int, bool]] = []

    def install(self) -> None:
        probe, cache, unbound = self, self.cache, self._unbound

        def wrapper(layer_id: int, num_tokens: int, *rest) -> bool:
            # *rest carries num_routed (topk_ids.numel()), which the gate grew when it
            # learned to refuse widths the slot cache cannot compile a dedup block for.
            decision = bool(unbound(cache, layer_id, num_tokens, *rest))
            probe.calls.append((int(layer_id), int(num_tokens), decision))
            return decision

        cache.use_cached_extend = wrapper  # instance attribute shadows the method

    def remove(self) -> None:
        self.cache.__dict__.pop("use_cached_extend", None)

    def reset(self) -> None:
        self.calls.clear()

    def summary(self) -> dict:
        hits = sum(1 for _, _, d in self.calls if d)
        widths = sorted({n for _, n, _ in self.calls})
        return {"hits": hits, "calls": len(self.calls), "token_counts": widths}


def lru_counters(cache) -> tuple[int, int, int]:
    """``(active, missing, layer_calls)`` summed over layers, as ``decode_miss_stats`` reads
    them. ``missing`` is the rows that crossed PCIe: for the cached extend arm that IS the
    eviction pressure the ticket asks about, because ``ensure_experts`` bumps these counters
    on extend forwards too (write-up §7 ticket 3)."""
    active, missing, calls = (int(x) for x in cache.lru_stats.sum(0)[:3])
    return active, missing, calls


def _rate(after: tuple[int, int, int], before: tuple[int, int, int]) -> dict:
    active = after[0] - before[0]
    missing = after[1] - before[1]
    calls = after[2] - before[2]
    return {
        "layer_calls": calls,
        "active_per_layer": round(active / calls, 3) if calls else None,
        "missing_per_layer": round(missing / calls, 3) if calls else None,
        "miss_rate": round(missing / active, 4) if active else None,
    }


class TailSource:
    """Fresh, mutually divergent extend tails.

    Each tail's first id is unique across the whole run, so the radix tree can match the
    seated base and nothing beyond it -- which is what makes an m-token request forward m
    tokens instead of one. ``base_child`` (the token the model generates after the base) is
    excluded: a tail starting with it would match one token deeper.
    """

    def __init__(self, pool: list[int], markers: list[int]):
        self.pool = pool
        self.markers = markers
        self.at = 0
        self.issued = 0

    def take(self, m: int) -> list[int]:
        if self.issued >= len(self.markers):
            raise RuntimeError("ran out of unique tail markers; raise --pool-tokens")
        if self.at + m > len(self.pool):
            raise RuntimeError("ran out of tail pool tokens; raise --pool-tokens")
        tail = list(self.pool[self.at : self.at + m])
        self.at += m
        tail[0] = self.markers[self.issued]
        self.issued += 1
        return tail


def build_pool(tok, repo: Path, target: int) -> list[int]:
    text = (repo / "docs/nemotron.md").read_text()
    ids: list[int] = []
    while len(ids) < target:
        ids += tok(text, add_special_tokens=False)["input_ids"]
    return ids[:target]


def timed_extend(llm, prompt, sp, ftimer, ltimer, probe) -> dict:
    """One timed extend forward: wall, GPU, per-mixer split, gate proof."""
    ftimer.samples.clear()
    ftimer.host.clear()
    ltimer.take()
    probe.reset()
    ftimer.install()
    ltimer.install()
    t0 = time.perf_counter()
    llm.generate([prompt], sp)
    wall_ms = 1000 * (time.perf_counter() - t0)
    ltimer.remove()
    ftimer.remove()
    parts = ltimer.take()
    buckets = ftimer.by_tokens()
    extend = {k: v for k, v in buckets.items() if k[1] == "extend"}
    return {
        "wall_ms": round(wall_ms, 3),
        "extend_buckets": {f"{k[0]}/{k[1]}": round(min(v), 3) for k, v in sorted(extend.items())},
        "forward_host_ms": (
            round(min(min(v) for k, v in ftimer.host.items() if k[1] == "extend"), 3)
            if any(k[1] == "extend" for k in ftimer.host)
            else None
        ),
        "forward_gpu_ms": (
            round(min(min(v) for v in extend.values()), 3) if extend else None
        ),
        "by_mixer_ms": parts,
        "gate": probe.summary(),
    }


def run_cell(llm, cache, probe, ftimer, ltimer, tails, base, args, *, m_val, arm_name,
             expected_tokens) -> dict:
    """One (m, arm) cell. ``m_val``/``arm_name`` are deliberately distinct from every
    enclosing loop variable: a previous harness collapsed a whole sweep by shadowing one."""
    from freetoken.core import SamplingParams

    one = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
    threshold = 0 if arm_name == "stream" else expected_tokens
    cache.extend_cache_tokens = threshold
    log(f"  cell m={m_val} arm={arm_name} extend_cache_tokens={threshold}")

    # Prime the decode working set BEFORE the extend, so `--decode-after` measures what the
    # extend did to an established working set rather than to an empty one.
    if args.prime_decode:
        llm.generate(
            [list(base)],
            SamplingParams(temperature=0.0, max_tokens=args.prime_decode + 1, ignore_eos=True),
        )

    prompt = list(base) + tails.take(m_val)
    llm.generate([prompt], one)  # warm this width's kernels/experts; untimed

    before = lru_counters(cache) if cache.collect_stats else None
    reps: list[dict] = []
    observed: set[int] = set()
    for _rep in range(args.repeats):
        prompt = list(base) + tails.take(m_val)
        row = timed_extend(llm, prompt, one, ftimer, ltimer, probe)
        reps.append(row)
        observed.update(row["gate"]["token_counts"])
        for key in row["extend_buckets"]:
            observed.add(int(key.split("/")[0]))
    after = lru_counters(cache) if cache.collect_stats else None

    # --- path proof, loudly ------------------------------------------------------------
    if observed != {expected_tokens}:
        raise AssertionError(
            f"m={m_val} arm={arm_name}: the forward carried {sorted(observed)} tokens, "
            f"expected exactly [{expected_tokens}]. A repeated tail (the radix tree serving "
            f"the prompt) or a prefix-match shift silently turns an m-token extend into a "
            f"1-token one -- the whole point of this driver is that this cannot pass."
        )
    hits = sum(r["gate"]["hits"] for r in reps)
    calls = sum(r["gate"]["calls"] for r in reps)
    if arm_name == "cached" and hits == 0:
        raise AssertionError(
            f"m={m_val} cached arm took the gate 0/{calls} times at "
            f"extend_cache_tokens={threshold}: quant_format={cache.quant_format!r}, "
            f"decode_target={cache.decode_target!r}, size_classes="
            f"{getattr(cache, '_size_class_enabled', None)}, "
            f"cpu_layers={sorted(cache.cpu_layer_ids)}, "
            f"unpinned={sorted(getattr(cache, '_unpinned_layers', ()))}"
        )
    if arm_name == "stream" and hits:
        raise AssertionError(f"m={m_val} stream arm took the cached gate {hits}/{calls} times")

    host = sorted(r["forward_host_ms"] for r in reps if r["forward_host_ms"] is not None)
    gpu = sorted(r["forward_gpu_ms"] for r in reps if r["forward_gpu_ms"] is not None)
    wall = sorted(r["wall_ms"] for r in reps)
    best = min(reps, key=lambda r: r["forward_host_ms"] or float("inf"))
    moe_ms = best["by_mixer_ms"].get("moe")
    moe_layers = max(1, best["gate"]["calls"])

    row = {
        "m": m_val,
        "arm": arm_name,
        "extend_cache_tokens": threshold,
        "expected_tokens": expected_tokens,
        "observed_tokens": sorted(observed),
        "gate_hits": hits,
        "gate_calls": calls,
        "gate_per_forward": f"{best['gate']['hits']}/{best['gate']['calls']}",
        "moe_layers": moe_layers,
        "repeats": args.repeats,
        "forward_host_ms_list": [r["forward_host_ms"] for r in reps],
        "forward_gpu_ms_list": [r["forward_gpu_ms"] for r in reps],
        "wall_ms_list": [r["wall_ms"] for r in reps],
        "forward_host_ms_median": round(statistics.median(host), 3) if host else None,
        "forward_host_ms_min": round(host[0], 3) if host else None,
        "forward_gpu_ms_median": round(statistics.median(gpu), 3) if gpu else None,
        "forward_gpu_ms_min": round(gpu[0], 3) if gpu else None,
        "wall_ms_median": round(statistics.median(wall), 3),
        "by_mixer_ms": best["by_mixer_ms"],
        "moe_ms_per_layer": round(moe_ms / moe_layers, 4) if moe_ms is not None else None,
        "extend_cache_pressure": (
            _rate(after, before) if before is not None and after is not None else None
        ),
    }

    if args.decode_after:
        row["decode"] = measure_decode_after(llm, cache, prompt, args)
    log(
        f"    m={m_val} {arm_name}: forward {row['forward_host_ms_median']} ms "
        f"(min {row['forward_host_ms_min']}), MoE/layer {row['moe_ms_per_layer']} ms, "
        f"gate {row['gate_per_forward']}, decode "
        f"{(row.get('decode') or {}).get('ms_per_step')} ms/step"
    )
    return row


def measure_decode_after(llm, cache, prompt, args) -> dict:
    """Plain decode steps immediately after the cell's extends.

    ``ms_per_step`` is a difference of two ``generate`` calls on the SAME prompt, so the
    offline round trip, the tokenizer, the scheduler pass and the one re-forwarded token are
    identical in both terms and cancel. The miss counters are differenced the same way, so
    what is left is the decode steps' own slot-cache traffic -- i.e. how much of the decode
    working set the extend arm just evicted.
    """
    from freetoken.core import SamplingParams

    one = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
    many = SamplingParams(
        temperature=0.0, max_tokens=args.decode_steps + 1, ignore_eos=True
    )
    stats = cache.collect_stats

    t_one = []
    d_one = []
    for _probe_rep in range(2):
        before = lru_counters(cache) if stats else None
        t0 = time.perf_counter()
        llm.generate([list(prompt)], one)
        t_one.append(time.perf_counter() - t0)
        if stats:
            d_one.append(tuple(a - b for a, b in zip(lru_counters(cache), before)))

    before = lru_counters(cache) if stats else None
    t0 = time.perf_counter()
    llm.generate([list(prompt)], many)
    t_many = time.perf_counter() - t0
    after = lru_counters(cache) if stats else None

    out = {
        "steps": args.decode_steps,
        "ms_per_step": round(1000 * (t_many - min(t_one)) / args.decode_steps, 3),
        "one_token_ms": round(1000 * min(t_one), 3),
        "burst_ms": round(1000 * t_many, 3),
    }
    if stats:
        # Subtract the re-forwarded extend token's own slot traffic (measured by the
        # 1-token calls) so what remains is the decode steps'.
        overhead = min(d_one, key=lambda d: d[2])
        net_after = tuple(a - o for a, o in zip(after, overhead))
        out.update(_rate(net_after, before))
    return out


def print_table(rows: list[dict]) -> None:
    head = (
        f"{'m':>5} {'arm':>7} {'thr':>5} {'gate':>7} {'fwd ms':>9} {'min':>8} "
        f"{'mamba':>7} {'attn':>6} {'MoE':>8} {'MoE/lyr':>8} {'miss/lyr':>9} "
        f"{'dec ms':>7} {'dec miss':>9}"
    )
    print("\n" + head, flush=True)
    print("-" * len(head), flush=True)
    for row in rows:
        mixer = row.get("by_mixer_ms") or {}
        pressure = row.get("extend_cache_pressure") or {}
        decode = row.get("decode") or {}
        print(
            f"{row['m']:>5} {row['arm']:>7} {row['extend_cache_tokens']:>5} "
            f"{row['gate_per_forward']:>7} "
            f"{_fmt(row['forward_host_ms_median']):>9} {_fmt(row['forward_host_ms_min']):>8} "
            f"{_fmt(mixer.get('mamba')):>7} {_fmt(mixer.get('attention')):>6} "
            f"{_fmt(mixer.get('moe')):>8} {_fmt(row['moe_ms_per_layer']):>8} "
            f"{_fmt(pressure.get('missing_per_layer')):>9} "
            f"{_fmt(decode.get('ms_per_step')):>7} {_fmt(decode.get('miss_rate')):>9}",
            flush=True,
        )


def _fmt(v) -> str:
    return "-" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)


def summarise_crossover(rows: list[dict]) -> dict:
    """Where the cached arm stops being cheaper, on the extend forward alone and on the
    extend plus the decode burst it degrades."""
    by_m: dict[int, dict[str, dict]] = {}
    for row in rows:
        by_m.setdefault(row["m"], {})[row["arm"]] = row
    out: dict[str, object] = {"per_m": {}}
    cached_wins: list[int] = []
    for m_key in sorted(by_m):
        pair = by_m[m_key]
        if set(pair) != set(ARMS):
            continue
        # WALL, not ``forward_host_ms_median``: the host number is only the launch leg,
        # and the two arms put their cost in different places -- the stream arm blocks
        # the host on its PCIe copies (host ~= wall), the cached arm returns in ~60 ms
        # and leaves the GPU gathering scattered expert rows for another ~250. Scoring
        # on host time picked m=96 as the crossover; on wall (and on the CUDA-event GPU
        # span, which agrees with it to ~2 %) the crossover is between 64 and 80.
        fwd = {a: pair[a]["wall_ms_median"] for a in ARMS}
        if any(fwd[a] is None for a in ARMS):
            continue
        dec = {a: (pair[a].get("decode") or {}).get("ms_per_step") for a in ARMS}
        # One chunk's forward plus the decode burst it hands off to: the cached arm can win
        # the forward and lose here, which is exactly the trade §7 ticket 2 asks about.
        total = {
            a: (fwd[a] + (dec[a] or 0.0) * (pair[a].get("decode") or {}).get("steps", 0))
            for a in ARMS
        }
        entry = {
            "forward_wall_ms": fwd,
            "forward_host_ms": {a: pair[a]["forward_host_ms_median"] for a in ARMS},
            "forward_gpu_ms": {a: pair[a]["forward_gpu_ms_median"] for a in ARMS},
            "decode_ms_per_step": dec,
            "forward_plus_decode_ms": {a: round(total[a], 3) for a in ARMS},
            "cached_forward_speedup": (
                round(fwd["stream"] / fwd["cached"], 3) if fwd["cached"] else None
            ),
            "cached_wins_forward": fwd["cached"] < fwd["stream"],
            "cached_wins_with_decode": total["cached"] < total["stream"],
        }
        out["per_m"][m_key] = entry
        if entry["cached_wins_with_decode"]:
            cached_wins.append(m_key)
    out["cached_wins_with_decode_at"] = cached_wins
    out["suggested_extend_cache_tokens"] = max(cached_wins) if cached_wins else 0
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="/home/lucas/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4")
    p.add_argument("--m", type=int, nargs="+", default=[64, 128, 256, 512],
                   help="extend widths; 512 is the scheduler's interleave-share chunk")
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    p.add_argument("--repeats", type=int, default=7, help="timed forwards per cell (>= 5)")
    p.add_argument("--base-tokens", type=int, default=4096,
                   help="seated prefix each timed extend continues")
    p.add_argument("--pool-tokens", type=int, default=65536,
                   help="corpus tokens to draw the base and every fresh tail from")
    p.add_argument("--decode-after", dest="decode_after", action="store_true", default=True)
    p.add_argument("--no-decode-after", dest="decode_after", action="store_false")
    p.add_argument("--decode-steps", type=int, default=32)
    p.add_argument("--prime-decode", type=int, default=16,
                   help="untimed decode steps before each cell's extends (0 disables)")
    p.add_argument("--json", default=None)
    p.add_argument("--moe-cache-auto", action="store_true", default=True)
    p.add_argument("--no-moe-cache-auto", dest="moe_cache_auto", action="store_false")
    p.add_argument("--no-collect-stats", dest="collect_stats", action="store_false",
                   default=True, help="skip the slot-cache miss counters")
    p.add_argument("--memory-ratio", type=float, default=0.85)
    # The whole run seats one base plus ~15K tokens of fresh tails; 64K leaves the radix
    # tree room to keep the base resident, which is what makes the extends stay m tokens.
    p.add_argument("--num-tokens", type=int, default=65536)
    p.add_argument("--max-seq-len", type=int, default=65536)
    p.add_argument("--host-ram-reserve-gb", type=float, default=6.0)
    args = p.parse_args()

    assert args.repeats >= 5, "the write-up reports median and min over >= 5 repeats"
    repo = Path(__file__).resolve().parent.parent
    out_path = Path(
        args.json
        or repo / "benchmarks/results" / f"extend_moe_threshold_{date.today().isoformat()}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from freetoken.core import SamplingParams
    from freetoken.llm.llm import LLM

    log(f"loading {args.model}")
    t_load = time.perf_counter()
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
        cuda_graph_max_bs=1,
        session_spill_dir=None,
        host_ram_reserve_gb=args.host_ram_reserve_gb,
        moe_collect_stats=args.collect_stats,
    )
    if args.moe_cache_auto:
        kwargs["moe_cache_auto"] = True
    llm = LLM(**kwargs)
    log(f"loaded in {time.perf_counter() - t_load:.1f} s")

    cache = llm.engine.moe_offload_cache
    assert cache is not None, "no MoE offload cache: --moe-backend offload did not take"
    original_threshold = cache.extend_cache_tokens
    page_size = max(1, int(getattr(llm.config, "page_size", 1)))
    log(
        f"cache: quant_format={cache.quant_format} decode_target={cache.decode_target} "
        f"size={cache.cache_size} experts={cache.num_experts} layers={cache.num_layers} "
        f"collect_stats={cache.collect_stats} cpu_layers={sorted(cache.cpu_layer_ids)} "
        f"unpinned={sorted(getattr(cache, '_unpinned_layers', ()))} "
        f"page_size={page_size} extend_cache_tokens={original_threshold}"
    )

    tok = llm.tokenizer
    pool = build_pool(tok, repo, args.pool_tokens)
    # Align the base to a page so the prefix match is not truncated to a page boundary and
    # every extend forwards exactly m tokens.
    base_len = max(page_size, args.base_tokens // page_size * page_size)
    base = pool[:base_len]
    tail_pool = pool[base_len:]

    one = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
    log(f"seating a {len(base)}-token base")
    first = llm.generate([list(base)], one)[0]["token_ids"]
    base_child = first[0] if first else None

    # Unique first ids for the tails, drawn from the corpus so they are always valid ids,
    # and never the token the model itself continues the base with.
    needed = len(args.m) * len(args.arms) * (args.repeats + 1) + 8
    markers: list[int] = []
    seen: set[int] = {base_child} if base_child is not None else set()
    for tid in tail_pool:
        if tid not in seen:
            seen.add(tid)
            markers.append(tid)
        if len(markers) >= needed:
            break
    assert len(markers) >= needed, (
        f"only {len(markers)} distinct tail markers in the corpus, need {needed}"
    )
    tails = TailSource(tail_pool, markers)

    probe = GateProbe(cache)
    probe.install()
    ftimer, ltimer = ForwardTimer(), LayerTimer()

    # Calibration: how many tokens does a base+m request actually forward? Anything but m
    # means the prefix match moved, and every later row would be mislabelled.
    calib_m = min(args.m)
    cache.extend_cache_tokens = 0
    calib = timed_extend(llm, list(base) + tails.take(calib_m), one, ftimer, ltimer, probe)
    observed = sorted({int(k.split("/")[0]) for k in calib["extend_buckets"]})
    assert len(observed) == 1, f"calibration saw several extend widths: {observed}"
    expected_tokens = observed[0]
    log(f"calibration: a base+{calib_m} request forwards {expected_tokens} tokens")
    if expected_tokens != calib_m:
        raise AssertionError(
            f"the prefix match is off by {expected_tokens - calib_m} tokens "
            f"(base_len={base_len}, page_size={page_size}); every row would be mislabelled"
        )

    results = {
        "config": {
            "model": args.model,
            "m": args.m,
            "arms": args.arms,
            "repeats": args.repeats,
            "base_tokens": len(base),
            "page_size": page_size,
            "decode_steps": args.decode_steps if args.decode_after else 0,
            "prime_decode": args.prime_decode,
            "moe_cache_auto": args.moe_cache_auto,
            "collect_stats": bool(cache.collect_stats),
            "cache_size": cache.cache_size,
            "num_experts": cache.num_experts,
            "num_layers": cache.num_layers,
            "quant_format": cache.quant_format,
            "decode_target": cache.decode_target,
            "cpu_layers": sorted(cache.cpu_layer_ids),
            "unpinned_layers": sorted(getattr(cache, "_unpinned_layers", ())),
            "default_extend_cache_tokens": original_threshold,
            "env": {
                k: os.environ.get(k)
                for k in ("FREETOKEN_PIN_BUDGET_GB", "FREETOKEN_MOE_EXTEND_CACHE_TOKENS")
            },
        },
        "rows": [],
    }

    try:
        for m_val in args.m:
            for arm_name in args.arms:
                results["rows"].append(
                    run_cell(
                        llm, cache, probe, ftimer, ltimer, tails, base, args,
                        m_val=m_val, arm_name=arm_name, expected_tokens=m_val,
                    )
                )
                out_path.write_text(json.dumps(results, indent=2))
    finally:
        cache.extend_cache_tokens = original_threshold
        probe.remove()

    results["summary"] = summarise_crossover(results["rows"])
    out_path.write_text(json.dumps(results, indent=2))
    print_table(results["rows"])
    print("\n" + json.dumps(results["summary"], indent=2), flush=True)
    log(f"wrote {out_path}")


if __name__ == "__main__":
    main()
