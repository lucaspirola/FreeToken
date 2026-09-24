#!/usr/bin/env python3
"""Prefill-transient A/B (checkpoint arms <ck>-<x>-def vs <ck>-<x>-t0): arena slots, decode
(monotonic clock) and the decode expert hit rate, side by side.

  compare_transient_ab.py <results dir> <ck> [t0|st]   (second side: -t0, or -st = the static
                                                  reservation, FREETOKEN_DYNAMIC_PREFILL_HEADROOM=0)

-def holds the measured prefill transient + margin free below the ratio plan through decode
(fix 82207c8); -t0 is the pre-fix cushion-only headroom (FREETOKEN_PREFILL_TRANSIENT_MEASURE=0,
FREETOKEN_PREFILL_TRANSIENT_MB=0). Both run with --moe-collect-stats.
  slots     "expert arena A -> B of C" at startup (B = what decode starts with) and every
            "MoE slots X -> Y" the run went through (min over the run)
  decode    (gen_tokens - 1) / (total_mono_s - ttft_mono_s) per size and pass
  hit rate  1 - missing/active from scheduler.moe.decode (cumulative over the arm's probes,
            warmups and posts included); mirror swaps per completion token for pool arms
"""
import json
import re
import sys
from pathlib import Path


def arm(d: Path, name: str):
    probe = d / f"{name}-probe.jsonl"
    if not probe.exists():
        return None
    dec = {}
    for line in probe.read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        m = r.get("total_mono_s", 0) - r.get("ttft_mono_s", 0)
        dec[(r["target"], r["pass"])] = (r["gen_tokens"] - 1) / m if m > 0 else None
    j = re.sub(r"\x1b\[[0-9;]*m", "", (d / f"{name}-journal.txt").read_text(errors="replace"))
    start = j.rfind("ServerArgs(model_path")
    j = j[start:] if start >= 0 else j
    a = re.search(r"expert arena (\d+) -> (\d+) of (\d+) slots, ([0-9.]+) GiB free", j)
    t = re.search(r"Prefill headroom: transient ([0-9.]+) GiB (\w+)", j)
    moves = [int(x) for x in re.findall(r"MoE slots \d+ -> (\d+)", j)]
    st = {}
    sp = d / f"{name}-stats.json"
    if sp.exists():
        st = json.loads(sp.read_text())
    moe = ((st.get("scheduler") or {}).get("moe") or {})
    dstat = moe.get("decode") or {}
    mirror = moe.get("mirror") or {}
    comp = ((st.get("requests") or {}).get("completion_tokens_total")) or 0
    hit = None
    if dstat.get("active"):
        hit = 1 - dstat.get("missing", 0) / dstat["active"]
    return {
        "dec": dec, "arena": a.groups() if a else None, "transient": t.groups() if t else None,
        "min_slots": min(moves) if moves else None, "max_slots": max(moves) if moves else None,
        "releases": len(re.findall(r"Prefill headroom released to decode", j)), "hit": hit, "decode_stats": dstat,
        "swaps_per_tok": (mirror.get("swaps", 0) / comp) if (mirror and comp) else None,
        "comp": comp,
    }


def main():
    d, ck = Path(sys.argv[1]), sys.argv[2]
    other = sys.argv[3] if len(sys.argv) > 3 else "t0"
    bases = sorted({p.name[len(ck) + 1:-len("-def-probe.jsonl")]
                    for p in d.glob(f"{ck}-*-def-probe.jsonl")})
    for b in bases:
        A, B = arm(d, f"{ck}-{b}-def"), arm(d, f"{ck}-{b}-{other}")
        print(f"== {ck}-{b}: default (-def) vs -{other}")
        for lab, r in (("def", A), (other, B)):
            if r is None:
                print(f"  {lab}: no probe"); continue
            ar = r["arena"]
            print(f"  {lab:3} transient {r['transient']} arena "
                  f"{(ar[1] + ' of ' + ar[2] + ' (free ' + ar[3] + ' GiB)') if ar else '-'}"
                  f", slots over run min {r['min_slots']} max {r['max_slots']} (releases {r['releases']}), decode hit rate "
                  f"{'%.4f' % r['hit'] if r['hit'] is not None else '-'}"
                  f", mirror swaps/token {'%.1f' % r['swaps_per_tok'] if r['swaps_per_tok'] is not None else '-'}"
                  f"  decode stats {r['decode_stats']}")
        if A and B:
            for k in sorted(set(A["dec"]) | set(B["dec"])):
                a, bb = A["dec"].get(k), B["dec"].get(k)
                ratio = f"{100 * bb / a:6.1f}%" if a and bb else "   -"
                print(f"  decode {k[0] // 1000:>4}K p{k[1]}  def {a or 0:6.1f}  {other} {bb or 0:6.1f}  {other}/def {ratio}")


if __name__ == "__main__":
    main()
