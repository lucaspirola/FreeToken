"""Plan 7.3 on a real capture: compare two nsys cuda_gpu_trace CSVs of the same request
(one prefill + N decode steps) by per-kernel-name launch counts, memory-operation counts
and bytes, and the GPU span; plus the /v1/stats counter deltas from instrument.sh's
result files. Identical launches + identical bytes = no new GPU work.

(instrument_analyze.py's per-step clustering splits 128 decode steps into ~1500 "steps"
on this GPU's kernel mix -- its gap threshold does not separate steps -- so its median
is not used; the whole-request totals below need no step boundary.)

usage: compare_traces.py <results_dir> <label>   (reads instrument-<label>-{before,after}*)
"""
import collections, csv, json, os, sys

D, LABEL = sys.argv[1:3]

def trace(s):
    rows = list(csv.DictReader(open(os.path.join(D, f"instrument-{LABEL}-{s}_cuda_gpu_trace.csv"))))
    start = next(c for c in rows[0] if c.startswith("Start"))
    dur = next(c for c in rows[0] if c.startswith("Duration"))
    nbytes = next((c for c in rows[0] if c.startswith("Bytes")), None)
    k, m, b = collections.Counter(), collections.Counter(), collections.Counter()
    ev = []
    for r in rows:
        n = r["Name"]; ev.append((int(r[start]), int(r[dur])))
        if n.startswith("[CUDA mem"):
            m[n] += 1; b[n] += float(r[nbytes] or 0) if nbytes else 0
        else:
            k[n] += 1
    t0 = min(a for a, _ in ev); span = (max(a + d for a, d in ev) - t0) / 1e9
    return k, m, b, span

def counters(s):
    t = open(os.path.join(D, f"instrument-{LABEL}-{s}.txt")).read()
    return json.loads(t[t.index("{"):])["byte_counters"]

kb, mb, bb, sb = trace("before"); ka, ma, ba, sa = trace("after")
cb, ca = counters("before"), counters("after")
bad = 0
names = sorted(set(kb) | set(ka))
kd = [(n, kb[n], ka[n]) for n in names if kb[n] != ka[n]]
print(f"kernel launches: before {sum(kb.values())} after {sum(ka.values())}; "
      f"{len(names)} kernel names, {len(kd)} with different counts")
for n, x, y in kd[:20]: print(f"  {x:7d} {y:7d}  {n[:120]}")
bad += bool(kd)
for n in sorted(set(mb) | set(ma)):
    same = mb[n] == ma[n] and round(bb[n], 3) == round(ba[n], 3)
    print(f"{'SAME' if same else 'DIFF'} {n}: count {mb[n]} / {ma[n]}, bytes {bb[n]:.3f} / {ba[n]:.3f}")
    bad += not same
cd = {k: (cb.get(k), ca.get(k)) for k in sorted(set(cb) | set(ca)) if cb.get(k) != ca.get(k)}
print("counters: " + ("IDENTICAL " + json.dumps(cb) if not cd else "DIFF " + json.dumps(cd)))
bad += bool(cd)
print(f"GPU span of the request: before {sb:.3f} s, after {sa:.3f} s ({(sa / sb - 1) * 100:+.1f}%)")
print("RESULT:", "IDENTICAL" if not bad else "DIFFERENT")
sys.exit(1 if bad else 0)
