#!/usr/bin/env python3
"""Tables for ctx393/README.md from results/ (probe jsonl, records, journals, samplers).

    report.py [RESULTS_DIR]   -> markdown on stdout

Prefill and TTFT from the monotonic clock (prefill_tok_s_mono = prompt tokens / TTFT)."""
import json, os, re, sys

R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def jl(p):
    try:
        return [json.loads(l) for l in open(p) if l.strip().startswith("{")]
    except FileNotFoundError:
        return []


def probe(arm):
    return {(d["target"], d["pass"]): d for d in jl(f"{R}/{arm}-probe.jsonl")}


def rec(arm):
    try:
        return json.load(open(f"{R}/{arm}-record.json"))
    except FileNotFoundError:
        return {}


def journal(arm):
    try:
        txt = open(f"{R}/{arm}-journal.txt").read().splitlines()
    except FileNotFoundError:
        return {}
    starts = [i for i, l in enumerate(txt) if "ServerArgs(model_path" in l]
    txt = txt[starts[-1]:] if starts else txt
    j = {"pred": [], "kv_max": 0}
    for l in txt:
        m = re.search(r"Memory prediction check: (.*)", l)
        if m:
            j["pred"].append(m[1])
        m = re.search(r"rope_yarn_factor=([\w.]+), rope_yarn_original_context=(\w+)", l)
        if m:
            j["yarn"] = f"{m[1]}/{m[2]}"
        m = re.search(r"Startup geometry: .*expert arena (\d+) slots; KV (\d+) tokens mapped of (\d+) \(([\d.]+) GiB\)", l)
        if m:
            j["slots_start"], j["kv_start"], j["kv_ceiling"], j["kv_ceiling_gib"] = int(m[1]), int(m[2]), int(m[3]), float(m[4])
        m = re.search(r"Committed growable KV through (\d+) tokens \(([\d.]+) GiB physical\); MoE slots (\d+) -> (\d+)", l)
        if m and int(m[1]) >= j["kv_max"]:
            j["kv_max"], j["kv_gib"], j["slots_at_max"] = int(m[1]), float(m[2]), int(m[4])
        m = re.search(r"\|(\d\d:\d\d:\d\d)\|.*(API server is ready|Uvicorn running|Application startup complete)", l)
        if m and "ready_t" not in j:
            j["ready_t"] = m[1]
        m = re.search(r"Mirror pool: (\d+) rows pinned \(([\d.]+) GiB\)", l)
        if m:
            j["pool_rows"], j["pool_gib"] = int(m[1]), float(m[2])
    return j


def status(arm):
    st = open(f"{R}/status.txt").read().splitlines()
    s = next((l for l in st if l.startswith(f"{arm} start ")), "")
    d = next((l for l in st if l.startswith(f"{arm} done ")), "")
    return s, d


def mem(arm):
    """(idle gpu MiB, idle MemAvailable GiB) at the first sample after readiness; peak gpu, min avail over the arm."""
    s, d = status(arm)
    if not (s and d):
        return None
    t0, t1 = s.split()[2][11:19], d.split()[2][11:19]
    rows = []
    for l in open(f"{R}/mem5s.txt"):
        t = l[:8]
        if t0 <= t <= t1:
            kv = dict(x.split("=") for x in l.split()[1:])
            rows.append((t, int(kv["gpu_mib"]), float(kv["avail_gib"])))
    rt = journal(arm).get("ready_t")
    idle = next((r for r in rows if rt and r[0] >= rt), None)
    return {"idle_gpu": idle and idle[1], "idle_avail": idle and idle[2],
            "peak_gpu": max(r[1] for r in rows), "min_avail": min(r[2] for r in rows)} if rows else None


def fmt(x, f="{:.1f}"):
    return "–" if x is None else f.format(x)


def probe_table(arms):
    sizes = sorted({k[0] for a in arms for k in probe(a)})
    print("| target | " + " | ".join(f"{a} p1 | {a} p2" for a in arms) + " |")
    print("|---" * (1 + 2 * len(arms)) + "|")
    for what, key, f in (("prompt tok", "prompt_tokens", "{:d}"), ("TTFT s", "ttft_mono_s", "{:.2f}"),
                         ("prefill tok/s", "prefill_tok_s_mono", "{:.0f}"), ("decode tok/s", "decode_tok_s", "{:.1f}")):
        for s in sizes:
            cells = []
            for a in arms:
                p = probe(a)
                for ps in (1, 2):
                    v = p.get((s, ps), {}).get(key)
                    cells.append(fmt(v, f))
            print(f"| {s} {what} | " + " | ".join(cells) + " |")


def ratio_table(num, den, label):
    pn, pd = probe(num), probe(den)
    print(f"| target | {label} prefill p2 | {label} decode p2 | {label} decode p1 | out_sha1 p2 equal |")
    print("|---|---:|---:|---:|---|")
    for s in sorted({k[0] for k in pn} & {k[0] for k in pd}):
        a2, b2, a1, b1 = pn.get((s, 2), {}), pd.get((s, 2), {}), pn.get((s, 1), {}), pd.get((s, 1), {})
        r = lambda x, y, k: fmt(100 * x[k] / y[k], "{:.1f}%") if x.get(k) and y.get(k) else "–"
        print(f"| {s} | {r(a2, b2, 'prefill_tok_s_mono')} | {r(a2, b2, 'decode_tok_s')} | {r(a1, b1, 'decode_tok_s')} | "
              f"{'yes' if a2.get('out_sha1') == b2.get('out_sha1') else 'no'} |")


def arm_table(arms):
    print("| arm | start (load, GPU, avail, SM) | yarn | KV ceiling (GiB) | KV max committed (GiB) | slots start -> at max | "
          "idle GPU MiB / avail GiB | peak GPU MiB / min avail GiB | ram_gib / rss_ready / peak cgroup | cov faults / starved | R3 |")
    print("|---" * 11 + "|")
    for a in arms:
        s, d = status(a)
        j, r, m = journal(a), rec(a), mem(a) or {}
        st = re.sub(r".*load=(\S+) \S+ \S+ gpu=(\S+) avail=(\S+) sm=(\S+ \S+).*", r"\1, \2, \3, \4", s)
        print(f"| {a} | {st} | {j.get('yarn', '–')} | {j.get('kv_ceiling', '–')} ({j.get('kv_ceiling_gib', '–')}) | "
              f"{j.get('kv_max') or '–'} ({j.get('kv_gib', '–')}) | {j.get('slots_start', '–')} -> {j.get('slots_at_max', '–')} | "
              f"{m.get('idle_gpu', '–')} / {m.get('idle_avail', '–')} | {m.get('peak_gpu', '–')} / {m.get('min_avail', '–')} | "
              f"{r.get('ram_gib', '–')} / {r.get('rss_ready_gib', '–')} / {r.get('peak_current_gib', '–')} | "
              f"{r.get('coverage_faults', '–')} / {r.get('starved', '–')} | {d.split(': ', 1)[-1] if d else '–'} |")
    print()
    for a in arms:
        print(f"* {a} prediction checks: " + "; ".join(journal(a).get("pred", [])))


def natural(arm):
    try:
        t = json.load(open(f"{R}/{arm}-natural.json"))["tasks"]
    except (FileNotFoundError, KeyError):
        return None
    return sum(x["completion_tokens"] for x in t) / sum(x["s"] for x in t), {x["task"]: x["md5"] for x in t}


def needle_table(arms):
    rows = {}
    for a in arms:
        for d in jl(f"{R}/{a}-depth.jsonl"):
            rows.setdefault((d["kind"], d["size"], str(d["depth"])), {})[a] = d
    print("| kind | size | depth | planted | " + " | ".join(f"{a} prompt tok / s / answer / pass" for a in arms) + " | answers equal |")
    print("|---" * (5 + len(arms)) + "|")
    for k in sorted(rows):
        v = rows[k]; first = next(iter(v.values()))
        cells = [f"{x.get('prompt_tokens')} / {x.get('seconds')} / `{(x.get('answer') or x.get('error', ''))[:80].replace(chr(10), ' ')}` / "
                 f"{'PASS' if x.get('pass') else 'FAIL'}" if (x := v.get(a)) else "–" for a in arms]
        eq = len({x.get("answer") for x in v.values()}) == 1 and len(v) == len(arms)
        print(f"| {k[0]} | {k[1]} | {k[2]} | {first['planted'] if k[0] == 'single' else ', '.join(first['planted'])} | "
              + " | ".join(cells) + f" | {'yes' if eq else 'no'} |")


if __name__ == "__main__":
    Y = ["y-sp", "y-wp"]; C = ["c-sp", "c-wp"]
    print("## Arms\n"); arm_table([a for a in ["c-sp", "y-sp", "y-wp", "c-wp", "y-sq", "y-wq", "c-sn", "y-sn", "y-wn"]
                                    if os.path.exists(f"{R}/{a}-journal.txt")])
    print("\n## Probe, YaRN 393216\n"); probe_table(Y)
    print("\n## Probe, control 262144\n"); probe_table(C)
    print("\n## saver / whole, YaRN\n"); ratio_table("y-sp", "y-wp", "saver/whole")
    print("\n## saver / whole, control\n"); ratio_table("c-sp", "c-wp", "saver/whole")
    print("\n## YaRN / control, saver\n"); ratio_table("y-sp", "c-sp", "yarn/ctrl")
    print("\n## YaRN / control, whole\n"); ratio_table("y-wp", "c-wp", "yarn/ctrl")
    print("\n## Natural text\n")
    for a in ["c-sn", "y-sn", "y-wn"]:
        n = natural(a)
        if n:
            print(f"* {a}: {n[0]:.1f} tok/s, md5 {n[1]}")
    print("\n## Depth needles\n"); needle_table(["y-sq", "y-wq"])
