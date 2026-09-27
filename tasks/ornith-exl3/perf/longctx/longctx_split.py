"""Split a decode-only nsys capture (longctx_decode.py pass3, --cuda-graph-trace=node) into per-token
components, and describe the full-attention decode kernels.

    longctx_split.py decode.sqlite RUN_LOG N_CTX [--json out.json]

Window: from the first kernel of the SECOND decode graph launch to the end of the last launch's work.
The first launch may have started before cudaProfilerStart. Graph-node kernels carry the correlationId of
their cudaGraphLaunch, so every launch is one step. The window's wall clock is shared among the records
active at each instant (1/n each when n overlap: side streams, eager DtoH writebacks). Instants with nothing
running count as idle. The components add up to the window, as in perf/profile_step.py wall_table().

Attention detail: stage1 grid (splits = gridZ) and blocks per launch, median us per call, us per
attention layer per token, and the KV bandwidth that implies. The KV bytes are
ctx * kv_heads * head_dim * 2 (K and V) * 34/32 (q8_0: 32 int8 + one fp16 scale per block). kv_heads,
head_dim and kv_splits come from the "Triton decode launch:" line in RUN_LOG.
"""
import json
import re
import sqlite3
import sys
from collections import defaultdict
from statistics import median

CATS = [  # first match wins, lower-cased kernel name
    ("attn split-KV (stage1+2)", ("_decode_grouped_stage1", "_decode_stage2", "_paged_attention")),
    ("attn KV write + rope", ("_store_kv_quant", "_store_one_quant", "rotary", "rope")),
    ("recurrent (GDN/Mamba)", ("selective_state_update", "causal_conv1d", "gated_delta", "fused_sigmoid_gating",
                               "fused_recurrent", "gated_norm", "gated_rmsnorm", "_state_update")),
    ("expert/EXL3 GEMV", ("exl3_gemv", "_splitk_silu_had", "_splitk_combine", "had_rows", "_decode_nvfp4_moe",
                          "_decode_nvfp4_marlin", "_decode_fp8_moe")),
    ("dense GEMV/GEMM", ("_nvfp4_gemv", "gemvx", "splitkreduce", "gemm", "nvjet", "cublas", "cutlass", "sm120")),
    ("miss loads (HtoD)", ("fast_index_copy", "memcpy htod")),
    ("write-backs (DtoH)", ("memcpy dtoh", "_writeback_buffer")),
    ("mirror/cache bookkeeping", ("_ensure_experts", "_resolve_swaps", "_publish_freed", "_begin_layer",
                                 "_reset_cache", "_materialize")),
    ("router/top-k", ("_router", "topk", "moe_align", "sigmoid")),
]
COPY_KIND = {1: "memcpy HtoD", 2: "memcpy DtoH", 8: "memcpy DtoD", 10: "memcpy PtoP"}


def cat(name):
    n = name.lower()
    for label, keys in CATS:
        if any(k in n for k in keys):
            return label
    return "rest (norms, elementwise, other)"


def main(db, run_log, n_ctx, out_json=None):
    con = sqlite3.connect(db)
    tables = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
    strings = dict(con.execute("select id, value from StringIds")) if "StringIds" in tables else {}
    cols = lambda t: [r[1] for r in con.execute(f"pragma table_info({t})")]  # noqa: E731

    launches = set()
    for t in ("CUPTI_ACTIVITY_KIND_RUNTIME",):
        if t in tables:
            for cid, nid in con.execute(f"select correlationId, nameId from {t}"):
                if "cudaGraphLaunch" in strings.get(nid, ""):
                    launches.add(cid)
    kc = cols("CUPTI_ACTIVITY_KIND_KERNEL")
    name_col = "demangledName" if "demangledName" in kc else "shortName"
    recs = []  # start, end, name, correlationId, grid
    for s, e, nid, cid, gx, gy, gz in con.execute(
            f"select start, end, {name_col}, correlationId, gridX, gridY, gridZ from CUPTI_ACTIVITY_KIND_KERNEL"):
        recs.append((s, e, strings.get(nid, str(nid)), cid, (gx, gy, gz)))
    for t in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
        if t in tables:
            has_kind = "copyKind" in cols(t)
            q = f"select start, end, correlationId{', copyKind, bytes' if has_kind else ''} from {t}"
            for row in con.execute(q):
                nm = COPY_KIND.get(row[3], f"memcpy kind{row[3]}") if has_kind else "memset"
                recs.append((row[0], row[1], nm, row[2], None))
    if "CUPTI_ACTIVITY_KIND_GRAPH_TRACE" in tables:  # --cuda-graph-trace=graph: one record per replay
        for s, e, cid in con.execute("select start, end, correlationId from CUPTI_ACTIVITY_KIND_GRAPH_TRACE"):
            recs.append((s, e, "graph replay (not split: use --cuda-graph-trace=node)", cid, None))
    recs.sort()
    by_launch = defaultdict(list)
    for i, r in enumerate(recs):
        if r[3] in launches:
            by_launch[r[3]].append(i)
    order = sorted(by_launch, key=lambda c: recs[by_launch[c][0]][0])
    if len(order) < 3:
        sys.exit(f"only {len(order)} graph launches found in the capture")
    t0 = recs[by_launch[order[1]][0]][0]
    t1 = max(recs[i][1] for i in by_launch[order[-1]])
    steps = len(order) - 1
    win = [r for r in recs if r[0] >= t0 and r[1] <= t1 + 1]
    span = t1 - t0

    pts = sorted([(r[0], 0, i) for i, r in enumerate(win)] + [(r[1], 1, i) for i, r in enumerate(win)], key=lambda p: (p[0], -p[1]))
    by_cat, by_name, idle, active, last = defaultdict(float), defaultdict(float), 0.0, set(), t0
    for t, kind, i in pts:
        dt = t - last
        if dt > 0:
            if active:
                sh = dt / len(active)
                for j in active:
                    by_name[win[j][2]] += sh
                    by_cat[cat(win[j][2])] += sh
            else:
                idle += dt
        last = t
        (active.discard if kind else active.add)(i)

    ms = lambda v: v / 1e6 / steps  # noqa: E731  ns -> ms/token
    out = {"n_ctx": n_ctx, "steps": steps, "window_ms_per_token": ms(span), "tok_s": steps / (span / 1e9),
           "components_ms_per_token": {k: ms(v) for k, v in sorted(by_cat.items(), key=lambda kv: -kv[1])},
           "idle_ms_per_token": ms(idle)}
    gaps = [recs[by_launch[order[k + 1]][0]][0] - max(recs[i][1] for i in by_launch[order[k]]) for k in range(1, len(order) - 1)]
    out["gap_between_steps_us_median"] = median(gaps) / 1e3 if gaps else None

    # attention detail
    s1 = [r for r in win if "_decode_grouped_stage1" in r[2]]
    s2 = [r for r in win if "_decode_stage2" in r[2]]
    kvw = [r for r in win if "_store_kv_quant" in r[2] or "_store_one_quant" in r[2]]
    m = re.search(r"Triton decode launch: kv_splits=(\d+) block_n=(\d+) warps=(\d+) \(q_heads=(\d+) kv_heads=(\d+) "
                  r"head_dim=(\d+) quant=(\S+) sms=(\S+)\)", open(run_log, errors="replace").read())
    att = {"launch_line": m.group(0) if m else None}
    if s1:
        n_attn = len(s1) / steps
        d1 = [(r[1] - r[0]) / 1e3 for r in s1]
        d2 = [(r[1] - r[0]) / 1e3 for r in s2] or [0.0]
        grids = sorted({r[4] for r in s1})
        blocks = [g[0] * g[1] * g[2] for g in grids]
        att.update({"attn_layers_per_token": n_attn, "stage1_grid": grids, "stage1_blocks": blocks,
                    "stage1_us_median": median(d1), "stage2_us_median": median(d2),
                    "kv_write_us_median": median([(r[1] - r[0]) / 1e3 for r in kvw]) if kvw else None,
                    "attn_us_per_layer_per_token": (sum(d1) + sum(d2)) / len(s1)})
        if m:
            sms = int(m.group(8)) if m.group(8).isdigit() else None
            kv_heads, head_dim = int(m.group(5)), int(m.group(6))
            kv_bytes = n_ctx * kv_heads * head_dim * 2 * 34 / 32
            att["kv_bytes_per_layer_MB"] = kv_bytes / 1e6
            att["stage1_GB_s"] = kv_bytes / (median(d1) * 1e-6) / 1e9
            if sms:
                att["stage1_waves"] = [b / sms for b in blocks]
    out["attention"] = att
    out["top_kernels_ms_per_token"] = [(k[:110], ms(v)) for k, v in sorted(by_name.items(), key=lambda kv: -kv[1])[:25]]

    print(f"ctx {n_ctx}: {steps} decode steps, window {ms(span):.3f} ms/token = {out['tok_s']:.1f} tok/s under nsys; "
          f"gap between steps median {out['gap_between_steps_us_median']:.1f} us")
    for k, v in out["components_ms_per_token"].items():
        print(f"  {k:32s} {v:7.3f} ms/token")
    print(f"  {'idle':32s} {out['idle_ms_per_token']:7.3f} ms/token")
    print("attention:", json.dumps(att))
    print("top kernels:")
    for k, v in out["top_kernels_ms_per_token"]:
        print(f"  {v:7.3f}  {k}")
    if out_json:
        json.dump(out, open(out_json, "w"), indent=1)


if __name__ == "__main__":
    a = sys.argv[1:]
    oj = None
    if "--json" in a:
        k = a.index("--json")
        oj = a[k + 1]
        a = a[:k] + a[k + 2:]
    main(a[0], a[1], int(a[2]), oj)
