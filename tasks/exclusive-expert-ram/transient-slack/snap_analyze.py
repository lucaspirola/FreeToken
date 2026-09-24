"""Where does the caching allocator's reservation exceed live tensors during one prefill chunk?
Replays the recorded device trace of a torch.cuda.memory._snapshot() pickle."""
import collections, pickle, sys

MiB = 1 << 20


def frame_str(frames, n=3):
    out = []
    for f in frames or []:
        fn = f.get("filename", "")
        if "freetoken" in fn or "flashinfer" in fn:
            out.append(f"{fn.split('python/')[-1]}:{f.get('line')}:{f.get('name')}")
        if len(out) >= n:
            break
    return " < ".join(out) or "?"


for path in sys.argv[1:]:
    snap = pickle.load(open(path, "rb"))
    ft = snap.get("freetoken", {})
    tr = [e for dev in snap["device_traces"] for e in dev]
    print(f"== {path}")
    print(f"   measured: reserved rise {ft.get('reserved_rise', 0) / MiB:.0f} MiB, allocated rise "
          f"{ft.get('allocated_rise', 0) / MiB:.0f} MiB (length {ft.get('length')}, prefix {ft.get('cached_len')})")
    st = ft.get("stats", {})
    for k in ("num_alloc_retries", "num_device_alloc", "num_device_free", "segment.large_pool.peak",
              "segment.small_pool.peak", "oversize_allocations.peak", "oversize_segments.peak"):
        if k in st:
            print(f"   stats {k} = {st[k]}")
    reserved = allocated = 0
    peak_r = peak_a = 0
    peak_r_at = None
    segs = {}
    seg_log = []
    live = {}
    pending_seg = None
    for i, e in enumerate(tr):
        a = e["action"]
        if a in ("segment_alloc", "segment_map"):
            reserved += e["size"]; segs[e["addr"]] = e["size"]
            pending_seg = (i, e, reserved - allocated)
        elif a in ("segment_free", "segment_unmap"):
            reserved -= e["size"]; segs.pop(e["addr"], None)
        elif a == "alloc":
            allocated += e["size"]; live[e["addr"]] = e
            if pending_seg is not None:
                j, se, free_before = pending_seg
                seg_log.append((se["size"], e["size"], free_before, frame_str(e.get("frames"))))
                pending_seg = None
        elif a == "free_completed":
            allocated -= e["size"]; live.pop(e["addr"], None)
        if reserved > peak_r:
            peak_r, peak_r_at = reserved, (i, allocated, dict(live))
        peak_a = max(peak_a, allocated)
    print(f"   trace replay: peak reserved (in-window segments) {peak_r / MiB:.0f} MiB, peak allocated {peak_a / MiB:.0f} MiB,"
          f" allocated at the reserved peak {peak_r_at[1] / MiB:.0f} MiB")
    print(f"   {len(seg_log)} segments created in the window, {sum(s for s, *_ in seg_log) / MiB:.0f} MiB:")
    by = collections.Counter(); bysz = collections.defaultdict(int)
    for s, req, free_before, fr in seg_log:
        k = "20MiB (1-10 MiB request)" if s == 20 * MiB else ("2MiB small" if s == 2 * MiB else ">=10 MiB request")
        by[k] += 1; bysz[k] += s
    for k in by:
        print(f"     {k:28s} x{by[k]:4d}  {bysz[k] / MiB:8.0f} MiB")
    print("   largest segments (size, the request that created it, cached-free bytes at that moment, site):")
    for s, req, free_before, fr in sorted(seg_log, reverse=True)[:25]:
        print(f"     {s / MiB:8.1f} MiB for {req / MiB:8.2f} MiB  free-cached {free_before / MiB:7.1f} MiB  {fr}")
    slack = collections.Counter()
    for s, req, free_before, fr in seg_log:
        slack[fr] += s - req
    print("   rounding slack by site (segment - request), top:")
    for fr, v in slack.most_common(10):
        print(f"     {v / MiB:8.1f} MiB  {fr}")
    # at the end: segments of the window vs blocks
    print("   end-of-run segments:", len(snap["segments"]),
          f"total {sum(s['total_size'] for s in snap['segments']) / MiB:.0f} MiB, active {sum(s['active_size'] for s in snap['segments']) / MiB:.0f} MiB")
