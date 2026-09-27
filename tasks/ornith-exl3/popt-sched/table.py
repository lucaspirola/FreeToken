#!/usr/bin/env python3
"""Tables for the final-numbers README from results/ (records, probes, natural json, journals).

    table.py [RESULTS_DIR]   -> markdown on stdout

Numbers are pass 2 of the probe unless marked p1; prefill and TTFT from the monotonic clock
(prefill_tok_s_mono = prompt tokens / TTFT). Journal facts come from the arm's LAST start (a unit
name reused after an aborted run keeps the older invocation in the journal)."""
import glob, json, os, re, sys
from collections import defaultdict

R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def probe(arm):
    out = {}
    try:
        for line in open(f"{R}/{arm}-probe.jsonl"):
            d = json.loads(line)
            out[(d["target"], d["pass"])] = d
    except FileNotFoundError:
        pass
    return out


def record(arm):
    try:
        return json.load(open(f"{R}/{arm}-record.json"))
    except FileNotFoundError:
        return {}


def natural(arm):
    try:
        t = json.load(open(f"{R}/{arm}-natural.json"))["tasks"]
    except (FileNotFoundError, KeyError):
        return None, {}
    return sum(x["completion_tokens"] for x in t) / sum(x["s"] for x in t), {x["task"]: x["md5"] for x in t}


def journal(arm):
    try:
        txt = open(f"{R}/{arm}-journal.txt").read().splitlines()
    except FileNotFoundError:
        return {}
    starts = [i for i, l in enumerate(txt) if "ServerArgs(model_path" in l]
    txt = txt[starts[-1]:] if starts else txt
    j = {}
    for l in txt:
        m = re.search(r"Startup geometry: .*expert arena (\d+) slots; KV (\d+) tokens mapped of (\d+) \(([\d.]+) GiB\)", l)
        if m:
            j["slots_start"] = int(m[1]); j["kv_ceiling"] = int(m[3])
        m = re.search(r"Committed growable KV through (\d+) tokens \(([\d.]+) GiB physical\); MoE slots (\d+) -> (\d+)", l)
        if m and int(m[1]) >= j.get("kv_max", 0):
            j["kv_max"] = int(m[1]); j["kv_gib"] = float(m[2]); j["slots_at_max"] = int(m[4])
        m = re.search(r"Prefill headroom: transient ([\d.]+) GiB", l)
        if m:
            j["transient_gib"] = float(m[1])
        m = re.search(r"Mirror pool: (\d+) rows pinned \(([\d.]+) GiB\)", l)
        if m:
            j["pool_rows"] = int(m[1]); j["pool_gib"] = float(m[2])
    return j


def load_of(arm):
    """Mean and max 1-min load over the arm's window (status.txt start/done, load5s.txt samples)."""
    st = open(f"{R}/status.txt").read().splitlines()
    t0 = next((l.split()[2][11:19] for l in st if l.startswith(f"{arm} start ")), None)
    t1 = next((l.split()[2][11:19] for l in st if l.startswith(f"{arm} done ")), None)
    if not (t0 and t1):
        return None
    ld = [float(l.split()[1]) for l in open(f"{R}/load5s.txt") if t0 <= l[:8] <= t1]
    return (sum(ld) / len(ld), max(ld)) if ld else None


def lf(arm):
    x = load_of(arm)
    return "–" if x is None else f"{x[0]:.1f} / {x[1]:.1f}"


def f(x, nd=1):
    return "–" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def arms(prefix):
    return sorted(os.path.basename(p)[:-len("-record.json")] for p in glob.glob(f"{R}/{prefix}-record.json"))


# ---- Part 1
SIZES = [8000, 32000, 80000, 128000, 256000]
print("## Part 1: probe, q8_0 KV, 262144 ceiling (pass 2; p1 in brackets)\n")
for res in ("whole", "saver"):
    names = arms(f"p1-{res}-*")
    print(f"### {res} ({', '.join(names)})\n")
    print("| prompt | " + " | ".join(f"{n} prefill tok/s" for n in names) + " | " + " | ".join(f"{n} TTFT s" for n in names)
          + " | " + " | ".join(f"{n} decode tok/s" for n in names) + " |")
    print("|---" * (1 + 3 * len(names)) + "|")
    for s in SIZES:
        ps = [probe(n) for n in names]
        cell = lambda d, k, p1k: f"{f(d.get(k))} ({f(p1k.get(k))})"
        pre = [cell(p.get((s, 2), {}), "prefill_tok_s_mono", p.get((s, 1), {})) for p in ps]
        tt = [cell(p.get((s, 2), {}), "ttft_mono_s", p.get((s, 1), {})) for p in ps]
        de = [cell(p.get((s, 2), {}), "decode_tok_s", p.get((s, 1), {})) for p in ps]
        print(f"| {s // 1000}K | " + " | ".join(pre) + " | " + " | ".join(tt) + " | " + " | ".join(de) + " |")
    for n in names:
        r, j = record(n), journal(n)
        print(f"\n`{n}`: ram_gib {r.get('ram_gib')}, rss_ready_gib {r.get('rss_ready_gib')}, gpu_mib {r.get('gpu_mib')}, "
              f"coverage_faults {r.get('coverage_faults')}, starved {r.get('starved')}; expert slots at start "
              f"{j.get('slots_start')}, at the 256K KV commit {j.get('slots_at_max')} (KV {j.get('kv_gib')} GiB); "
              f"prefill transient {j.get('transient_gib')} GiB; host load mean/max {lf(n)}", end="")
    print("\n")
# saver/whole ratio of the means
# Period split: arms 1-2 ran 06:49-07:35, arms 3-4 10:22-10:54, after the ~10% host-wide slowdown
# (README "Drift"); the ratio is taken within a period so the drift cancels.
def period(k):
    return lambda pattern: [n for n in arms(pattern) if n.rsplit("-", 1)[1] in k]


for title, pick in (("arms 1-2 (06:49-07:35)", period("12")), ("arms 3-4 (10:22-10:54, slow period)", period("34"))):
  print(f"### saver / whole, pass 2, mean of {title} "
        f"(whole: {', '.join(pick('p1-whole-*')) or 'none'}; saver: {', '.join(pick('p1-saver-*')) or 'none'})\n")
  print("| prompt | prefill | TTFT | decode |\n|---|---:|---:|---:|")
  for s in SIZES:
    m = {}
    for res in ("whole", "saver"):
        ps = [probe(n).get((s, 2), {}) for n in pick(f"p1-{res}-*")]
        m[res] = {k: sum(p.get(k) or 0 for p in ps) / max(1, len(ps)) for k in ("prefill_tok_s_mono", "ttft_mono_s", "decode_tok_s")}
    if m["whole"]["decode_tok_s"] and m["saver"]["decode_tok_s"]:
        print(f"| {s // 1000}K | {m['saver']['prefill_tok_s_mono']:.0f} / {m['whole']['prefill_tok_s_mono']:.0f} = "
              f"{100 * m['saver']['prefill_tok_s_mono'] / m['whole']['prefill_tok_s_mono']:.1f}% | "
              f"{m['saver']['ttft_mono_s']:.2f} / {m['whole']['ttft_mono_s']:.2f} s | "
              f"{m['saver']['decode_tok_s']:.1f} / {m['whole']['decode_tok_s']:.1f} = {100 * m['saver']['decode_tok_s'] / m['whole']['decode_tok_s']:.1f}% |")
  print()
print("\n### natural text (5 tasks, 3500 max tokens, 8K warm-up only)\n")
nat = arms("p1nat-*")
ref = next((n for n in nat if "whole" in n), None)
_, rmd5 = natural(ref) if ref else (None, {})
print("| arm | decode tok/s | md5 same as " + f"{ref} | host load mean / max |\n|---|---:|---|---|")
for n in nat:
    tps, md5 = natural(n)
    print(f"| {n} | {f(tps)} | {sum(md5[k] == rmd5.get(k) for k in md5)}/{len(md5)} | {lf(n)} |")

# ---- Part 2
print("\n## Part 2: KV lanes, saver\n")
for c, big in (("256k", 256000), ("384k", 384000)):
    def quietest(base):
        # The first run of every lane (07:35-09:58, one period); the -r2 repeats ran in the slow
        # period and are listed in the appendix below.
        return base
    ref_arm = quietest(f"kv-q8q8-{c}"); ref = probe(ref_arm)
    _, rn = natural(quietest(f"kvnat-q8q8-{c}"))
    print(f"### ceiling {c} ({'--rope-yarn-factor 2, 393216' if c == '384k' else '262144'})\n")
    print(f"q8q8 reference arms: `{ref_arm}`, `{quietest(f'kvnat-q8q8-{c}')}`\n")
    print(f"| lane (arm) | {big // 1000}K prefill tok/s | {big // 1000}K TTFT s | {big // 1000}K decode tok/s | 8K decode tok/s | "
          f"natural tok/s | output vs q8q8 (probe 8K/{big // 1000}K p1,p2; natural) | KV GiB at {big // 1000}K | expert slots start / at max KV | pool rows (GiB) | ram_gib / rss_ready_gib | gpu_mib | load probe / natural arm |")
    print("|---|---:|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---|")
    for l in ("q8q8", "q8q6", "q6q5", "q4q4"):
        n = quietest(f"kv-{l}-{c}"); nn = quietest(f"kvnat-{l}-{c}"); p = probe(n); r = record(n); j = journal(n)
        if not p and not r:
            continue
        same = []
        for key in ((8000, 1), (8000, 2), (big, 1), (big, 2)):
            a, b = p.get(key, {}).get("out_sha1"), ref.get(key, {}).get("out_sha1")
            same.append("=" if a and a == b else "≠")
        tps, md5 = natural(nn)
        ns = f"{sum(md5[k] == rn.get(k) for k in md5)}/{len(md5)}" if md5 else "–"
        g2 = p.get((big, 2), {}); g1 = p.get((big, 1), {})
        print(f"| {l} (`{n}`, `{nn}`) | {f(g2.get('prefill_tok_s_mono'))} ({f(g1.get('prefill_tok_s_mono'))}) | {f(g2.get('ttft_mono_s'), 2)} | "
              f"{f(g2.get('decode_tok_s'))} ({f(g1.get('decode_tok_s'))}) | {f(p.get((8000, 2), {}).get('decode_tok_s'))} | {f(tps)} | "
              f"{''.join(same)}; {ns} | {j.get('kv_gib')} | {j.get('slots_start')} / {j.get('slots_at_max')} | {j.get('pool_rows')} ({j.get('pool_gib')}) | {r.get('ram_gib')} / {r.get('rss_ready_gib')} | {r.get('gpu_mib')} | {lf(n)}; {lf(nn)} |")
    print()
print("### every part-2 probe arm (pass 2 of the big prompt; load = window mean / max)\n")
print("| arm | prefill tok/s | TTFT s | decode tok/s | 8K decode | out_sha1 (big p2) | load |\n|---|---:|---:|---:|---:|---|---|")
for n in arms("kv-*"):
    p = probe(n); big = max((k[0] for k in p), default=0); g = p.get((big, 2), {})
    print(f"| {n} | {f(g.get('prefill_tok_s_mono'))} | {f(g.get('ttft_mono_s'), 2)} | {f(g.get('decode_tok_s'))} | "
          f"{f(p.get((8000, 2), {}).get('decode_tok_s'))} | {(g.get('out_sha1') or '')[:10]} | {lf(n)} |")
print("\n| natural arm | tok/s | load |\n|---|---:|---|")
for n in arms("kvnat-*"):
    print(f"| {n} | {f(natural(n)[0])} | {lf(n)} |")
print()
rep = probe("kv-q8q8-256k-rep")
if rep:
    a, b = probe("kv-q8q8-256k").get((256000, 2), {}), rep.get((256000, 2), {})
    print(f"Drift check, q8q8 256K repeated at the end: decode {f(a.get('decode_tok_s'))} -> {f(b.get('decode_tok_s'))}, "
          f"prefill {f(a.get('prefill_tok_s_mono'))} -> {f(b.get('prefill_tok_s_mono'))}, output "
          f"{'identical' if a.get('out_sha1') == b.get('out_sha1') else 'DIFFERENT'}.")
