#!/usr/bin/env python3
"""Prefill chunk A/B (chunk-local.sh arms ch-<model>-<kind>-<chunk>): measured transient,
arena slots at the decode and prefill levels, prefill tok/s (monotonic) and decode tok/s per
size and pass, each chunk's ratio to 8192.

  compare_chunks.py <results dir>
"""
import json
import re
import sys
from pathlib import Path


def load(d: Path, name: str):
    probe = d / f"{name}-probe.jsonl"
    if not probe.exists():
        return None
    pre, dec = {}, {}
    for line in probe.read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        k = (r["target"], r["pass"])
        pre[k] = r.get("prefill_tok_s_mono") or r.get("prefill_tok_s")
        m = r.get("total_mono_s", 0) - r.get("ttft_mono_s", 0)
        dec[k] = (r["gen_tokens"] - 1) / m if m > 0 else None
    jp = d / f"{name}-journal.txt"
    j = jp.read_text(errors="replace") if jp.exists() else ""
    start = j.rfind("ServerArgs(model_path")
    j = j[start:] if start >= 0 else j
    t = re.search(r"Prefill headroom: transient ([0-9.]+) GiB measured on a (\d+)-token chunk", j)
    reserved = [int(x) for x in re.findall(r"Prefill headroom reserved: MoE slots \d+ -> (\d+)", j)]
    released = [int(x) for x in re.findall(r"Prefill headroom released to decode: MoE slots \d+ -> (\d+)", j)]
    oom = len(re.findall(r"out of memory|OutOfMemory", j))
    return {"pre": pre, "dec": dec, "transient": t.groups() if t else None,
            "prefill_slots": min(reserved) if reserved else None,
            "decode_slots": max(released) if released else None, "oom": oom}


def main():
    d = Path(sys.argv[1])
    names = sorted({p.name[: -len("-probe.jsonl")] for p in d.glob("ch-*-probe.jsonl")})
    groups = {}
    for n in names:
        _, model, kind, chunk = n.split("-")
        groups.setdefault((model, kind), {})[int(chunk)] = load(d, n)
    for (model, kind), arms in sorted(groups.items()):
        print(f"== {model} {kind}")
        for c, a in sorted(arms.items()):
            print(f"  chunk {c:>5}: transient {a['transient']}, arena prefill level "
                  f"{a['prefill_slots']} / decode level {a['decode_slots']} slots, OOM lines {a['oom']}")
        ref = arms.get(8192)
        keys = sorted({k for a in arms.values() for k in a["pre"]})
        for what in ("pre", "dec"):
            print(f"  {'prefill' if what == 'pre' else 'decode'} tok/s " + "".join(f"{c:>18}" for c in sorted(arms)))
            for k in keys:
                row = f"  {k[0] // 1000:>5}K p{k[1]}     "
                for c in sorted(arms):
                    v = arms[c][what].get(k)
                    r = ref[what].get(k) if ref else None
                    pct = f" ({100 * v / r:5.1f}%)" if (v and r and c != 8192) else "         "
                    row += f"{(v or 0):9.1f}{pct}"
                print(row)


if __name__ == "__main__":
    main()
