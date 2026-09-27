#!/usr/bin/env python3
"""Bracket table: decode tok/s per probe point, with the 1M decode window's median gap and the
host load / SM clock sampled closest to the start of that window (load5s.txt, every 5 s).
  brtab.py <bracket dir> <arm>..."""
import datetime, json, sys
B = sys.argv[1]
load = [l.split() for l in open(f"{B}/load5s.txt")]
def near(ts):
    t = datetime.datetime.fromtimestamp(ts)
    def d(r):
        x = datetime.datetime.combine(t.date(), datetime.datetime.strptime(r[0], "%H:%M:%S").time())
        return abs((x - t).total_seconds())
    r = min(load, key=d)
    return f"load {r[1]} sm {r[4]}"
for a in sys.argv[2:]:
    try:
        rows = [json.loads(l) for l in open(f"{B}/{a}-probe.jsonl")]
    except OSError:
        continue
    out = []
    for r in rows:
        s = f"{r['target'] // 1000}K p{r['pass']} {r['decode_tok_s']}"
        if r["target"] >= 1000000:
            s += f" (gap {r.get('median_gap_ms')} ms, {near(r['t_start_wall'] + r['ttft_s'])})"
        out.append(s)
    print(f"{a:18s} " + " | ".join(out))
