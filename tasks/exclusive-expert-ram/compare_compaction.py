#!/usr/bin/env python3
"""Arena compaction A/B: <ck>-<x>-{def,dc,st,t0} side by side, one block per base (whole, mirror).

  compare_compaction.py <results dir> <ck>

  def  dynamic prefill headroom + compaction (the branch default)
  dc   dynamic, compaction off (FREETOKEN_ARENA_COMPACTION=0)
  st   static reservation (FREETOKEN_DYNAMIC_PREFILL_HEADROOM=0)
  t0   no prefill transient reserved at all (FREETOKEN_PREFILL_TRANSIENT_MEASURE=0 / _MB=0)

Per arm: decode hit rate and mirror swaps per completion token (from the arm's stats
snapshot, compare_transient_ab.arm), the journal's arena compactions (calls, experts moved,
MiB and ms per call, over reserves and KV-growth shrinks alike), the mirror's coverage
refills (rows, and whether written back from the GPU or re-read from the checkpoint), and
the "Decode window since release" lines: expert misses per request decode window. Then
decode tok/s (monotonic) per size and pass, with each arm's ratio to def.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_transient_ab import arm  # noqa: E402

VARIANTS = ("def", "dc", "st", "t0")


def journal_extras(d: Path, name: str) -> dict:
    p = d / f"{name}-journal.txt"
    if not p.exists():
        return {}
    j = re.sub(r"\x1b\[[0-9;]*m", "", p.read_text(errors="replace"))
    start = j.rfind("ServerArgs(model_path")
    j = j[start:] if start >= 0 else j
    comp = [tuple(float(x) for x in m) for m in re.findall(
        r"Arena compaction \d+ -> \d+: (\d+) doomed, (\d+) moved \(([0-9.]+) MiB D2D, ([0-9.]+) ms\), "
        r"(\d+) dropped with a host copy, (\d+) left for write-back", j)]
    refill = re.findall(r"mirror restored coverage for (\d+) experts(?: \(([^)]*)\))?", j)
    windows = [tuple(int(x) for x in m) for m in re.findall(
        r"Decode window since release: (\d+) layer calls, (\d+) active, (\d+) missing", j)]
    return {"comp": comp, "refill": refill, "windows": windows}


def fmt(v, f="%.4f"):
    return f % v if v is not None else "-"


def main():
    d, ck = Path(sys.argv[1]), sys.argv[2]
    for base in ("whole", "mirror"):
        arms = {v: arm(d, f"{ck}-{base}-{v}") for v in VARIANTS}
        arms = {v: a for v, a in arms.items() if a is not None}
        if not arms:
            continue
        print(f"== {ck}-{base}: " + ", ".join(arms))
        for v, a in arms.items():
            x = journal_extras(d, f"{ck}-{base}-{v}")
            comp = x.get("comp", [])
            ncomp = len(comp)
            moved = sum(c[1] for c in comp)
            mib = sum(c[2] for c in comp)
            ms = sum(c[3] for c in comp)
            left = sum(c[5] for c in comp)
            refill = x.get("refill", [])
            rows = sum(int(r[0]) for r in refill)
            disk = sum(int(r[0]) for r in refill if "checkpoint" in (r[1] or "checkpoint"))
            win = x.get("windows", [])
            wmiss = [w[2] for w in win]
            print(f"  {v:3} hit {fmt(a['hit'])}  swaps/tok {fmt(a['swaps_per_tok'], '%.1f')}"
                  f"  releases {a['releases']}  slots {a['min_slots']}..{a['max_slots']}")
            if ncomp:
                print(f"      compactions {ncomp}: moved {moved:.0f} ({moved / ncomp:.1f}/call), "
                      f"{mib / ncomp:.1f} MiB and {ms / ncomp:.2f} ms per call, "
                      f"{left:.0f} left for write-back")
            if refill:
                print(f"      mirror refills {len(refill)}: {rows} rows ({disk} re-read from the checkpoint)")
            if win:
                print(f"      decode windows {len(win)}: misses per window {wmiss} "
                      f"(mean {sum(wmiss) / len(wmiss):.0f})")
        keys = sorted({k for a in arms.values() for k in a["dec"]})
        ref = arms.get("def")
        print("  decode tok/s      " + "".join(f"{v:>16}" for v in arms))
        for k in keys:
            row = f"  {k[0] // 1000:>5}K p{k[1]}       "
            for v, a in arms.items():
                val = a["dec"].get(k)
                r = ref["dec"].get(k) if ref else None
                pct = f" ({100 * val / r:5.1f}%)" if (val and r and v != "def") else "         "
                row += f"{(val or 0):7.1f}{pct}"
            print(row)


if __name__ == "__main__":
    main()
