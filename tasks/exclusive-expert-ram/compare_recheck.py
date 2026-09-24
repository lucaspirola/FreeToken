#!/usr/bin/env python3
"""recheck-local.sh verdicts.

  compare_recheck.py <results dir> <label> ["size ...", default "8000"]

8K part: <label>-whole-{1,2,3} and <label>-mirror-{1,2,3} (8K, 512 tokens, passes 1-2).
Per pass: each arm's decode (monotonic), the ratio of each mirror arm to the whole arm
run before it, and the median of those three ratios against 91% (the gate). Also per arm:
the first inter-chunk gap vs the median gap (the release stall right after token 1), and
decode excluding that first gap.
RAM part: <label>-mirror-1m ram_gib against 12.26 +/- 0.6 with its attribution (RSS,
anon, file, pool rows) next to ck4's mirror-1m values.
"""
import json
import re
import statistics
import sys
from pathlib import Path

CK4 = {"ram_gib": 12.65, "rss_ready_gib": 13.71, "rss_gib": 14.05, "anon_gib": 12.92,
       "file_gib": 1.50, "pool": "1830 rows, 9.58 GiB"}


def probes(d: Path, name: str):
    p = d / f"{name}-probe.jsonl"
    out = {}
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        m = r.get("total_mono_s", 0) - r.get("ttft_mono_s", 0)
        r["dec"] = (r["gen_tokens"] - 1) / m if m > 0 else None
        out[(r["target"], r["pass"])] = r
    return out


def pool(d: Path, name: str):
    p = d / f"{name}-journal.txt"
    if not p.exists():
        return None
    m = re.findall(r"mirror expert pool attached: (\d+) host rows, ([0-9.]+) GiB", p.read_text(errors="replace"))
    return f"{m[-1][0]} rows, {m[-1][1]} GiB" if m else None


def main():
    d, lab = Path(sys.argv[1]), sys.argv[2]
    sizes = [int(x) for x in sys.argv[3].split()] if len(sys.argv) > 3 else [8000]
    runs = [(probes(d, f"{lab}-whole-{i}"), probes(d, f"{lab}-mirror-{i}")) for i in (1, 2, 3)]
    for size in sizes:
      if any(w or m for w, m in runs):
          print(f"== {size // 1000}K, 512 tokens: mirror vs the whole arm run before it")
          for p in (1, 2):
              ratios = []
              for i, (w, m) in enumerate(runs, 1):
                  a, b = w.get((size, p)), m.get((size, p))
                  if not (a and b and a["dec"] and b["dec"]):
                      print(f"  p{p} run {i}: missing"); continue
                  ratios.append(b["dec"] / a["dec"])
                  print(f"  p{p} run {i}: whole {a['dec']:6.1f} (gap1 {a.get('gap1_ms')} ms, median gap "
                        f"{a.get('median_gap_ms')} ms)  mirror {b['dec']:6.1f} (gap1 {b.get('gap1_ms')} ms, "
                        f"median gap {b.get('median_gap_ms')} ms)  ratio {100 * ratios[-1]:5.1f}%  "
                        f"after gap1: whole {a.get('decode_tok_s_after_gap1')} mirror {b.get('decode_tok_s_after_gap1')}")
              if ratios:
                  med = statistics.median(ratios)
                  print(f"  p{p} median ratio {100 * med:5.1f}%  {'PASS' if med >= 0.91 else 'FAIL'} (>= 91%)")
    one = d / f"{lab}-mirror-1m-record.json"
    if one.exists():
        r = json.loads(one.read_text())
        print(f"== 1M RAM: {lab}-mirror-1m vs ck4 mirror-1m")
        for k in ("ram_gib", "rss_ready_gib", "rss_gib", "anon_gib", "file_gib"):
            print(f"  {k:14} {r.get(k)}  (ck4 {CK4[k]})")
        print(f"  pool           {pool(d, f'{lab}-mirror-1m')}  (ck4 {CK4['pool']})")
        ram = r.get("ram_gib")
        if ram is not None:
            print(f"  ram 12.26 +/- 0.6  {'PASS' if abs(ram - 12.26) <= 0.6 else 'OUT'}")
        pr = probes(d, f"{lab}-mirror-1m")
        for k in sorted(pr):
            print(f"  decode {k[0] // 1000}K p{k[1]}: {pr[k]['dec']:.1f}")


if __name__ == "__main__":
    main()
