"""Kernel time per bucket from an nsys sqlite (the popt-exl3 view: routed experts, EXL3 dense,
GDN, norms, attention, copies, other), plus the top kernels. Usage: python nsys_buckets.py X.sqlite"""
import sqlite3, sys, collections
db = sqlite3.connect(sys.argv[1])
names = dict(db.execute("select id, value from StringIds"))
rows = db.execute("select k.start, k.end, k.demangledName, k.streamId from CUPTI_ACTIVITY_KIND_KERNEL k").fetchall()
span = (max(r[1] for r in rows) - min(r[0] for r in rows)) / 1e6
def bucket(n):
    if any(s in n for s in ("_exl3_gemm", "_reconstruct_experts", "_had_rows", "_splitk_combine", "act_and_mul", "moe_align", "count_and_sort", "_group_blocks", "_router")):
        return "moe/exl3-routed"
    if any(s in n for s in ("_reconstruct_folded", "Kernel2", "gemm", "cutlass_80_tensorop", "sm80_xmma", "sm90", "cublas")):
        return "dense (cuBLAS + folded reconstruct)"
    if any(s in n for s in ("chunk_", "recompute_w_u", "causal_conv1d", "l2norm", "fused_recurrent", "solve_tril", "kkt")):
        return "GDN + conv"
    if "norm" in n.lower():
        return "norms"
    if any(s in n for s in ("Prefill", "Decode", "MergeState", "Rotary", "_gather_dequant", "flashinfer")):
        return "attention"
    if any(s in n for s in ("fast_index_copy", "_writeback", "_prefetch")):
        return "layer stream (side)"
    if any(s in n for s in ("elementwise", "copy", "Copy")):
        return "copies/casts"
    return "other"
agg = collections.Counter(); per = collections.Counter(); cnt = collections.Counter()
for s, e, nid, st in rows:
    n = names.get(nid, str(nid))
    agg[bucket(n)] += (e - s) / 1e6; per[n[:70]] += (e - s) / 1e6; cnt[n[:70]] += 1
print(f"{sys.argv[1].split('/')[-1]}: span {span:.1f} ms, kernel sum {sum(agg.values()):.1f} ms")
for b, t in agg.most_common():
    print(f"  {b:40s} {t:9.1f} ms")
for n, t in per.most_common(14):
    print(f"     {t:9.1f} ms  x{cnt[n]:5d}  {n}")
