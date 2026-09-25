#!/usr/bin/env python3
"""Tabulate the dh2 decode-target A/B arms from FT-dh results dirs."""
import glob, json, os, re, sys
R = sys.argv[1] if len(sys.argv) > 1 else "/root/FT-dh/tasks/exclusive-expert-ram/results"
for d in sorted(glob.glob(f"{R}/dt*-box")):
    if not re.search(r"/dt(def|128)(-\d|m)-box$", d):
        continue
    for probe in sorted(glob.glob(f"{d}/*-probe.jsonl")):
        arm = os.path.basename(probe)[:-len("-probe.jsonl")]
        rows = [json.loads(l) for l in open(probe) if l.strip()]
        dec = {(r["target"], r["pass"]): r.get("decode_tok_s") for r in rows}
        st = json.load(open(f"{d}/{arm}-stats.json")) if os.path.exists(f"{d}/{arm}-stats.json") else {}
        moe = (st.get("scheduler") or {}).get("moe") or {}
        dcd = moe.get("decode") or {}
        act, miss = dcd.get("active"), dcd.get("missing")
        hit = (1 - miss / act) if act else None
        mir = moe.get("mirror") or {}
        j = open(f"{d}/{arm}-journal.txt", errors="replace").read()
        win = re.findall(r"Decode memory window closed[^\n]*", j)
        drops = [float(x) for x in re.findall(r"\(drop ([0-9.]+) GiB\)", "\n".join(win))]
        rel = re.findall(r"Prefill headroom released to decode: MoE slots (\d+) -> (\d+)", j)
        faults = mir.get("coverage_faults", st.get("coverage_faults"))
        print(f"{arm}: decode " + " ".join(f"{t//1000}K p{p} {v}" for (t, p), v in sorted(dec.items())) +
              f" | hit {hit:.4f}" if hit is not None else f"{arm}: hit n/a", end="")
        print(f" | windows {len(win)} max drop {max(drops) if drops else 0:.2f} GiB"
              f" | release slots {rel[-1] if rel else '-'} ({len(rel)} releases)"
              f" | faults {faults} starved {mir.get('starved_writebacks')} captures {j.count('Start capturing CUDA graphs')}"
              f" tracebacks {j.count('Traceback')} ooms {j.lower().count('out of memory')}")
