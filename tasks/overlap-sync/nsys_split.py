"""Split a decode window from an nsys sqlite export (job-nsys.sh) into CUDA-graph replay, eager GPU work
outside the graphs, and GPU idle, per token.

    nsys_split.py decode-8k.sqlite N_GEN

With --cuda-graph-trace=graph a replay is one record (CUPTI_ACTIVITY_KIND_GRAPH_TRACE); eager kernels,
memcpys and memsets are their own records. The decode window starts after the last prefill-only kernel
(same rule as profile_step.py). Idle = no record active on any stream.
"""
import sqlite3
import sys
from collections import defaultdict

PREFILL_ONLY = ("_extend_attention", "_had_cols_kernel", "_reconstruct_experts_kernel", "_exl3_gemm_kernel", "chunk_")

db, n_gen = sys.argv[1], int(sys.argv[2])
con = sqlite3.connect(db)
tables = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
print("tables:", sorted(t for t in tables if t.startswith("CUPTI") or t.startswith("CUDA")))
strings = {}
if "StringIds" in tables:
    strings = dict(con.execute("select id, value from StringIds"))


def cols(t):
    return [r[1] for r in con.execute(f"pragma table_info({t})")]


recs = []  # (start ns, end ns, kind, name)
for t, kind in (("CUPTI_ACTIVITY_KIND_KERNEL", "kernel"), ("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy"),
                ("CUPTI_ACTIVITY_KIND_MEMSET", "memset"), ("CUPTI_ACTIVITY_KIND_GRAPH_TRACE", "graph")):
    if t not in tables:
        continue
    c = cols(t)
    name_col = next((x for x in ("shortName", "demangledName", "name") if x in c), None)
    q = f"select start, end{', ' + name_col if name_col else ''} from {t}"
    for row in con.execute(q):
        name = strings.get(row[2], str(row[2])) if name_col else kind
        if kind == "memcpy":
            name = "memcpy"
        recs.append((row[0], row[1], kind, name))
recs.sort()
print("records by kind:", dict(sorted(defaultdict(int, {k: sum(1 for r in recs if r[2] == k) for k in {r[2] for r in recs}}).items())))
last_pf = max((i for i, r in enumerate(recs) if r[2] == "kernel" and r[3].startswith(PREFILL_ONLY)), default=-1)
dec = recs[last_pf + 1:]
t0, t1 = dec[0][0], max(r[1] for r in dec)
span = t1 - t0

# sweep: attribute each instant to graph if any graph is running, else eager if anything runs, else idle
pts = sorted([(r[0], 0, i) for i, r in enumerate(dec)] + [(r[1], 1, i) for i, r in enumerate(dec)], key=lambda p: (p[0], -p[1]))
active, last = set(), t0
acc = defaultdict(float)
eager_by_name = defaultdict(float)
for t, k, i in pts:
    dt = t - last
    if dt > 0:
        if not active:
            acc["idle"] += dt
        elif any(dec[j][2] == "graph" for j in active):
            acc["graph replay"] += dt
        else:
            acc["eager (outside graphs)"] += dt
            for j in active:
                eager_by_name[dec[j][3]] += dt / len(active)
    last = t
    (active.discard if k else active.add)(i)
graphs = [r for r in dec if r[2] == "graph"]
print(f"decode window {span / 1e6:.2f} ms = {span / 1e6 / n_gen:.3f} ms/token ({n_gen / (span / 1e9):.1f} tok/s); "
      f"{len(graphs) / n_gen:.2f} graph replays/token, {sum(1 for r in dec if r[2] != 'graph') / n_gen:.1f} eager records/token")
for k in ("graph replay", "eager (outside graphs)", "idle"):
    print(f"  {k:24s} {acc[k] / 1e6 / n_gen:7.3f} ms/token  {100 * acc[k] / span:5.1f}%")
print("eager work by name, ms/token:")
for k, v in sorted(eager_by_name.items(), key=lambda kv: -kv[1])[:15]:
    print(f"  {v / 1e6 / n_gen:7.4f}  {k[:110]}")
if graphs:
    d = sorted(g[1] - g[0] for g in graphs)
    print(f"graph replay duration: median {d[len(d) // 2] / 1e3:.1f} us, min {d[0] / 1e3:.1f}, max {d[-1] / 1e3:.1f}")
    gaps = sorted(graphs[i + 1][0] - graphs[i][1] for i in range(len(graphs) - 1))
    if gaps:
        print(f"end of one replay -> start of the next: median {gaps[len(gaps) // 2] / 1e3:.1f} us, "
              f"p10 {gaps[len(gaps) // 10] / 1e3:.1f}, p90 {gaps[9 * len(gaps) // 10] / 1e3:.1f}")
