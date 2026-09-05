"""Offline analysis of prompt-lookup (n-gram) speculative decoding.

Under *greedy* decoding the accepted-length distribution of a prompt-lookup drafter is a
deterministic function of the token sequence the model produces without speculation: the
verified continuation is exactly the greedy continuation. So the acceptance side of the
go/no-go can be answered from ordinary greedy transcripts, with no engine change and no
second GPU run.

Input: JSONL, one object per transcript, with integer token id lists::

    {"name": "code", "prompt_tokens": [...], "output_tokens": [...]}

Output: mean accepted length lambda(n, k) per transcript and per prompt class, the fraction
of steps that issue a draft at all, and the projected decode speedup under the measured
step-cost model ``T(m)/T(1) = 1 + c*(m-1)`` where ``m = k+1`` is the verify width.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict


def build_index(tokens: list[int], n: int) -> dict[tuple, int]:
    """Map every n-gram to the index just past its most recent occurrence."""
    index: dict[tuple, int] = {}
    for i in range(len(tokens) - n + 1):
        index[tuple(tokens[i : i + n])] = i + n
    return index


class NgramDrafter:
    """Most-recent-occurrence prompt lookup over prompt + generated tokens.

    ``n_list`` is tried longest-first, which is the usual refinement over a single n: a
    5-gram match is a much better predictor than a 3-gram match, and falling back only
    when the long match is absent keeps the hit rate of the short one.
    """

    def __init__(self, n_list: list[int]):
        self.n_list = sorted(n_list, reverse=True)
        self.index: dict[int, dict[tuple, int]] = {n: {} for n in self.n_list}
        self.cursor = {n: 0 for n in self.n_list}

    def observe(self, tokens: list[int], upto: int) -> None:
        """Index every n-gram of ``tokens`` that *ends before* ``upto``.

        The n-gram ending exactly at ``upto`` is the query itself. Indexing it would
        overwrite the most recent real occurrence with a self-match, so it is held back
        until the position advances -- this is the difference between a drafter that
        never fires and one that does.
        """
        for n in self.n_list:
            idx = self.index[n]
            i = self.cursor[n]
            while i + n <= upto - 1:
                idx[tuple(tokens[i : i + n])] = i + n
                i += 1
            self.cursor[n] = i

    def draft(self, tokens: list[int], pos: int, k: int) -> list[int]:
        """Propose up to k tokens to follow ``tokens[:pos]``. Empty list = no match."""
        for n in self.n_list:
            if pos < n:
                continue
            key = tuple(tokens[pos - n : pos])
            hit = self.index[n].get(key)
            # A match at the very end is the query itself; it predicts nothing.
            if hit is None or hit >= pos:
                continue
            return tokens[hit : hit + k]
        return []


def simulate(
    tokens: list[int],
    start: int,
    n_list: list[int],
    k_max: int,
    adaptive: bool,
) -> dict:
    """Replay greedy decoding with an n-gram drafter over a known greedy transcript."""
    drafter = NgramDrafter(n_list)
    pos = start
    k = k_max
    steps_drafted = 0
    steps_plain = 0
    accepted_total = 0
    drafted_total = 0
    verify_width_total = 0
    hist: dict[int, int] = defaultdict(int)
    widths: dict[int, int] = {}
    emitted = 0
    while pos < len(tokens):
        drafter.observe(tokens, pos)
        draft = drafter.draft(tokens, pos, k) if k > 0 else []
        remaining = len(tokens) - pos
        if not draft:
            steps_plain += 1
            pos += 1
            emitted += 1
            continue
        accepted = 0
        for j, tok in enumerate(draft):
            if pos + j >= len(tokens) or tokens[pos + j] != tok:
                break
            accepted += 1
        steps_drafted += 1
        drafted_total += len(draft)
        accepted_total += accepted
        verify_width_total += len(draft) + 1
        # The verify forward covers len(draft)+1 positions and always yields one token
        # beyond the accepted prefix (the bonus token), capped by what is left.
        step_emit = min(accepted + 1, remaining)
        hist[step_emit] += 1
        widths[len(draft) + 1] = widths.get(len(draft) + 1, 0) + 1
        pos += step_emit
        emitted += step_emit
        if adaptive:
            k = min(k_max, k + 2) if accepted == len(draft) else max(1, accepted + 1)
    return {
        "generated": emitted,
        "steps_drafted": steps_drafted,
        "steps_plain": steps_plain,
        "accepted_total": accepted_total,
        "drafted_total": drafted_total,
        "verify_width_total": verify_width_total,
        "hist": dict(hist),
        "widths": widths,
    }


def cost_from_routing(routing: dict, anchor: tuple[float, float, float, float]) -> dict[int, float]:
    """Relative verify-step cost per width m, from measured expert-routing overlap.

    ``anchor`` is ``(touched_1, touched_2, cost_1, cost_2)`` -- two measured points of
    step cost against distinct experts touched per MoE layer (task 2B4 measured a 1-token
    and a 2-token step directly). On this offload path the step is dominated by expert
    PCIe traffic, so cost is taken linear in touched experts and the ``touched(m)`` curve
    supplies the rest. This is what makes the model sublinear in m: consecutive tokens
    share experts, so the m-th token costs less than the first extra one did.
    """
    t1, t2, c1, c2 = anchor
    slope = (c2 - c1) / (t2 - t1)
    return {int(m): c1 + slope * (v - t1) for m, v in routing.items() if v is not None}


def project(sim: dict, cost: dict[int, float] | float) -> dict:
    """Projected speedup. ``cost`` is either a flat marginal coefficient c (meaning
    T(m)/T(1) = 1 + c*(m-1)) or a measured {m: relative cost} table."""
    if isinstance(cost, dict):
        verify_cost = sum(cost[m] * n for m, n in sim["widths"].items())
    else:
        c = cost
        verify_cost = sim["steps_drafted"] + c * (
            sim["verify_width_total"] - sim["steps_drafted"]
        )
    total = sim["steps_plain"] + verify_cost
    baseline = sim["generated"]
    steps = sim["steps_drafted"] + sim["steps_plain"]
    return {
        "steps": steps,
        "draft_rate": sim["steps_drafted"] / steps if steps else 0.0,
        "accept_rate": (
            sim["accepted_total"] / sim["drafted_total"] if sim["drafted_total"] else 0.0
        ),
        "lambda": sim["generated"] / steps if steps else 0.0,
        "speedup": baseline / total if total else 0.0,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("transcripts", help="JSONL with prompt_tokens/output_tokens")
    p.add_argument("--n", type=int, nargs="+", default=[5, 4, 3])
    p.add_argument("--k", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8, 12, 16])
    p.add_argument(
        "--marginal-cost",
        type=float,
        nargs="+",
        default=[0.64, 0.42],
        help="c in T(m)/T(1) = 1 + c*(m-1); measure it, do not guess it",
    )
    p.add_argument("--adaptive", action="store_true")
    p.add_argument(
        "--routing",
        help="routing.json from benchmarks/probe_ngram_spec.py; replaces the flat "
        "--marginal-cost model with one derived from measured expert overlap",
    )
    p.add_argument(
        "--anchor",
        type=float,
        nargs=4,
        default=[6.0, 11.61, 1.0, 1.63],
        metavar=("TOUCHED1", "TOUCHED2", "COST1", "COST2"),
        help="two measured (experts touched per MoE layer, relative step cost) points",
    )
    p.add_argument("--json-out")
    args = p.parse_args()

    rows = []
    with open(args.transcripts) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    routing = None
    if args.routing:
        raw = json.loads(open(args.routing).read())
        routing = {name: d["consecutive"] for name, d in raw.items()}

    out = []
    for row in rows:
        tokens = list(row["prompt_tokens"]) + list(row["output_tokens"])
        start = len(row["prompt_tokens"])
        for k in args.k:
            sim = simulate(tokens, start, args.n, k, args.adaptive)
            rec = {
                "name": row.get("name", "?"),
                "prompt_tokens": start,
                "output_tokens": len(row["output_tokens"]),
                "n": args.n,
                "k": k,
                "adaptive": args.adaptive,
            }
            models: list[tuple[str, dict | float]] = []
            if routing is not None:
                table = routing.get(row.get("name", "?")) or next(iter(routing.values()))
                models.append(("measured", cost_from_routing(table, tuple(args.anchor))))
            models += [(f"c={c}", c) for c in args.marginal_cost]
            for label, model in models:
                proj = project(sim, model)
                rec.update(
                    {key: proj[key] for key in ("steps", "draft_rate", "accept_rate", "lambda")}
                )
                rec[f"speedup@{label}"] = round(proj["speedup"], 3)
            rec["draft_rate"] = round(rec["draft_rate"], 3)
            rec["accept_rate"] = round(rec["accept_rate"], 3)
            rec["lambda"] = round(rec["lambda"], 3)
            out.append(rec)

    hdr = ["name", "k", "draft_rate", "accept_rate", "lambda"] + [
        key for key in out[0] if key.startswith("speedup@")
    ]
    print("\t".join(hdr))
    for rec in out:
        print("\t".join(str(rec[h]) for h in hdr))
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
