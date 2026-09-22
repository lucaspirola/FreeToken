#!/usr/bin/env python3
"""Parsing/analysis logic for the section 7.3 instrument (plan
`2026-09-22-refactor-plan-final.md`, sections 5/S6/S7 and 7.3), split out of
`instrument.sh` so it can be unit-tested without a GPU.

Two jobs, both pure functions of already-captured data:

1. ``cluster_decode_steps`` -- turn an `nsys stats --report cuda_gpu_trace
   --format csv` export of the decode window into per-decode-step kernel-launch
   groups. Decode runs from a captured CUDA graph (CLAUDE.md, "single lane"),
   so every step launches the SAME graph; `--cuda-graph-trace=node` (verified
   against `nsys profile --help` / `nsys launch --help` on this host, Nsight
   Systems 2025.6.3.541) makes the trace list every kernel inside each replay
   instead of one opaque "graph launch" event, but nsys's own report scripts
   (checked directly: `/opt/nvidia/nsight-systems/2025.6.3/*/reports/
   cuda_gpu_trace.py`) carry no per-replay grouping column -- no graph-exec id,
   no launch-instance id. The only signal left to regroup by step is time: one
   step's kernels fire back-to-back (microseconds apart, same stream), and the
   next step's kernels start only after a python-side host round trip (at
   record decode speed, ~13 ms/token -- three-plus orders of magnitude bigger
   than the intra-step gaps). Steps are therefore recovered by clustering
   launch start times on the gap between consecutive kernels: gaps far above
   the local median are step boundaries. This is a heuristic, stated as such;
   see `_choose_gap_threshold` for the exact rule and `tests()` below for the
   synthetic case it is verified against.

2. ``compare_results`` -- diff two `instrument-<label>.txt` JSON-bodied result
   files (see `instrument.sh`'s writer) and report IDENTICAL or the deltas.

Run standalone for self-test (no nsys, no GPU, no torch needed):
    python3 instrument_analyze.py --selftest
"""
from __future__ import annotations

import csv
import io
import json
import statistics
import sys
from dataclasses import dataclass, field


@dataclass
class Kernel:
    start_ns: int
    dur_ns: int
    name: str


@dataclass
class StepGroup:
    kernels: list = field(default_factory=list)

    def count(self) -> int:
        return len(self.kernels)

    def histogram(self) -> dict:
        h: dict = {}
        for k in self.kernels:
            h[k.name] = h.get(k.name, 0) + 1
        return h


def _find_col(fieldnames, *substrings) -> str:
    """nsys CSV headers carry a unit suffix, e.g. 'Start:ts_ns' (see
    cuda_gpu_trace.py's query_stub on this host) -- match by substring, not
    exact name, so a future nsys version's unit tag doesn't silently break
    the parse (it would instead raise ValueError below, loud not silent)."""
    for fn in fieldnames:
        low = fn.lower()
        if all(s in low for s in substrings):
            return fn
    raise ValueError(
        f"no column matches {substrings!r} in header {fieldnames!r} "
        "-- nsys's cuda_gpu_trace column names may have changed; re-check "
        "`nsys stats --help` / the reports/cuda_gpu_trace.py on this host"
    )


def parse_cuda_gpu_trace_csv(text: str) -> list:
    """Parse an `nsys stats --report cuda_gpu_trace --format csv` export into
    a time-sorted list of `Kernel`. Only Start/Duration/Name are load-bearing
    here; every other cuda_gpu_trace column (grid/block dims, registers,
    stream, ...) is read straight off the CSV for a human but not needed for
    the count/histogram this instrument reports."""
    if not text.strip():
        # Verified live on this host: `nsys stats --report=cuda_gpu_trace` on a
        # capture with no CUDA activity prints "SKIPPED: ... does not contain
        # GPU trace data" and still writes a 0-byte CSV. That is not a parse
        # bug, it means the capture window caught no GPU work at all -- e.g.
        # the decode request never ran, or --cuda-graph-trace tracing was not
        # actually active in the target process.
        raise ValueError(
            "the CSV is empty -- nsys likely reported 'SKIPPED: ... does not "
            "contain GPU trace data' when exporting; the capture window "
            "caught no CUDA activity (check that the probed request actually "
            "ran and that CUDA tracing was active in the target process)"
        )
    reader = csv.DictReader(io.StringIO(text))
    fieldnames = reader.fieldnames or []
    start_col = _find_col(fieldnames, "start")
    dur_col = _find_col(fieldnames, "duration")
    name_col = _find_col(fieldnames, "name")
    out = []
    for row in reader:
        s = row.get(start_col, "").strip()
        d = row.get(dur_col, "").strip()
        n = row.get(name_col, "").strip()
        if not s or not n:
            continue
        out.append(Kernel(start_ns=int(float(s)), dur_ns=int(float(d or 0)), name=n))
    out.sort(key=lambda k: k.start_ns)
    return out


def _choose_gap_threshold(gaps: list) -> int:
    """Gaps within one CUDA-graph replay are microseconds; the gap between one
    decode step's replay and the next is milliseconds at any plausible decode
    speed (record: 76.9 tok/s => ~13 ms/token; even a 20x regression is still
    ~0.65 ms, far above intra-replay spacing). Threshold = max(10x the median
    gap, 50 microseconds): the 10x multiplier survives runs where most gaps
    are near-zero (median could be 0-1 ns on a fast GPU), and the 50 us floor
    stops a run of truly back-to-back kernels with a few-ns median from
    splitting on jitter alone. Verified against the synthetic case in
    `tests()`, not against a real capture (no GPU here -- see instrument.sh's
    "unverified until the owner's first run" note)."""
    if not gaps:
        return 50_000
    med = statistics.median(gaps)
    return max(int(med * 10), 50_000)


def cluster_decode_steps(kernels: list) -> list:
    """Split a time-sorted kernel list into per-decode-step groups by the gap
    heuristic above. The FIRST group is dropped by the caller when it looks
    like prefill/warmup (see `summarize`), not here: this function only knows
    about timing, not about what prefill looks like."""
    if not kernels:
        return []
    gaps = [b.start_ns - a.start_ns for a, b in zip(kernels, kernels[1:])]
    threshold = _choose_gap_threshold(gaps)
    groups = [StepGroup([kernels[0]])]
    for prev, cur in zip(kernels, kernels[1:]):
        if cur.start_ns - prev.start_ns > threshold:
            groups.append(StepGroup([]))
        groups[-1].kernels.append(cur)
    return groups


def summarize(kernels: list, drop_first_n_groups: int = 1) -> dict:
    """Cluster, drop the leading group(s) (prefill / the pre-decode warmup
    request; see instrument.sh's capture-window comment for why the window
    intentionally starts a little before the first decode token), and report
    median/spread of the per-step kernel count plus a histogram of one
    representative (the median-count) step."""
    groups = cluster_decode_steps(kernels)
    decode_groups = groups[drop_first_n_groups:] if len(groups) > drop_first_n_groups else groups
    counts = [g.count() for g in decode_groups]
    if not counts:
        return {"n_steps": 0, "counts": [], "median": None, "spread": None, "histogram": {}}
    med = statistics.median(counts)
    # representative step: the one closest to the median count, ties -> first
    rep = min(decode_groups, key=lambda g: (abs(g.count() - med), decode_groups.index(g)))
    return {
        "n_steps": len(decode_groups),
        "counts": counts,
        "median": med,
        "spread": max(counts) - min(counts),
        "stdev": statistics.pstdev(counts) if len(counts) > 1 else 0.0,
        "histogram": rep.histogram(),
    }


def compare_results(a: dict, b: dict) -> tuple:
    """Diff two result documents (the JSON `instrument.sh` writes into each
    `instrument-<label>.txt`). Returns (identical: bool, lines: list[str]).
    "Identical" is judged on the launch-count median/spread/histogram and the
    byte-counter deltas -- not on wall-clock timing fields, which are
    expected to jitter run to run."""
    lines = []
    identical = True

    def cmp_field(path, va, vb):
        nonlocal identical
        if va != vb:
            identical = False
            lines.append(f"  {path}: before={va!r} after={vb!r}")

    ka = a.get("kernel_launches", {})
    kb = b.get("kernel_launches", {})
    cmp_field("kernel_launches.median", ka.get("median"), kb.get("median"))
    cmp_field("kernel_launches.spread", ka.get("spread"), kb.get("spread"))
    cmp_field("kernel_launches.histogram", ka.get("histogram"), kb.get("histogram"))

    ba = a.get("byte_counters", {})
    bb = b.get("byte_counters", {})
    for key in sorted(set(ba) | set(bb)):
        cmp_field(f"byte_counters.{key}", ba.get(key), bb.get(key))

    return identical, lines


def _cli_cluster(args) -> int:
    text = sys.stdin.read() if args.csv == "-" else open(args.csv).read()
    kernels = parse_cuda_gpu_trace_csv(text)
    summary = summarize(kernels, drop_first_n_groups=args.drop_first)
    print(json.dumps(summary, indent=2, default=str))
    return 0


def _cli_compare(args) -> int:
    a = json.load(open(args.before))
    b = json.load(open(args.after))
    identical, lines = compare_results(a, b)
    if identical:
        print("IDENTICAL")
        return 0
    print("DIFFERENT")
    for line in lines:
        print(line)
    return 1


def tests() -> None:
    """Self-test on synthetic data shaped like a real `cuda_gpu_trace` CSV
    export (columns and header text taken from
    `/opt/nvidia/nsight-systems/2025.6.3/target-linux-x64/reports/
    cuda_gpu_trace.py`'s `query_stub` on this host, 2025.6.3.541-256337736014v0).
    No nsys run, no GPU, no network."""
    header = (
        "Start:ts_ns,Duration:dur_ns,CorrId,GrdX,GrdY,GrdZ,BlkX,BlkY,BlkZ,"
        "Reg/Trd,StcSMem:mem_B,DymSMem:mem_B,Bytes:mem_B,Throughput:thru_B,"
        "SrcMemKd,DstMemKd,Device,Ctx,GreenCtx,Strm,Name\n"
    )
    rows = []
    t = 1_000_000  # ns
    kernel_names = [f"kernel_{i}" for i in range(20)]
    # 3 decode steps of 20 kernels each, back-to-back within a step (200 ns
    # apart), 13 ms between steps (record decode speed => ~13 ms/token).
    for step in range(3):
        for i, name in enumerate(kernel_names):
            rows.append(
                f"{t},500,{step*20+i},1,1,1,32,1,1,32,0,0,,,,,\"GPU (0)\",1,,7,{name}\n"
            )
            t += 200
        t += 13_000_000
    # a leading "prefill" burst: many more kernels, should end up as group 0
    prefill_rows = []
    tp = 0
    for i in range(200):
        prefill_rows.append(
            f"{tp},2000,{9000+i},1,1,1,32,1,1,32,0,0,,,,,\"GPU (0)\",1,,7,prefill_kernel\n"
        )
        tp += 300
    csv_text = header + "".join(prefill_rows) + "".join(rows)
    # shift decode timestamps so the whole file is monotonic (prefill first)
    csv_text = header
    tp = 0
    for i in range(200):
        csv_text += f"{tp},2000,{9000+i},1,1,1,32,1,1,32,0,0,,,,,\"GPU (0)\",1,,7,prefill_kernel\n"
        tp += 300
    t = tp + 5_000_000
    for step in range(3):
        for i, name in enumerate(kernel_names):
            csv_text += f"{t},500,{step*20+i},1,1,1,32,1,1,32,0,0,,,,,\"GPU (0)\",1,,7,{name}\n"
            t += 200
        t += 13_000_000

    kernels = parse_cuda_gpu_trace_csv(csv_text)
    assert len(kernels) == 200 + 3 * 20, len(kernels)
    groups = cluster_decode_steps(kernels)
    assert len(groups) == 4, f"expected 4 groups (1 prefill + 3 decode), got {len(groups)}"
    assert groups[0].count() == 200
    for g in groups[1:]:
        assert g.count() == 20, g.count()

    summary = summarize(kernels, drop_first_n_groups=1)
    assert summary["n_steps"] == 3
    assert summary["median"] == 20
    assert summary["spread"] == 0
    assert summary["histogram"] == {name: 1 for name in kernel_names}

    # compare_results: identical case
    doc_a = {
        "kernel_launches": {"median": 20, "spread": 0, "histogram": {"k": 20}},
        "byte_counters": {"swaps": 100, "writebacks": 5},
    }
    doc_b = dict(doc_a)
    identical, lines = compare_results(doc_a, doc_b)
    assert identical and not lines

    doc_c = json.loads(json.dumps(doc_a))
    doc_c["kernel_launches"]["median"] = 21
    doc_c["byte_counters"]["swaps"] = 101
    identical, lines = compare_results(doc_a, doc_c)
    assert not identical
    assert any("kernel_launches.median" in l for l in lines)
    assert any("byte_counters.swaps" in l for l in lines)

    # a run with an irregular gap pattern (simulates real jitter): the
    # threshold heuristic must still separate 3 steps from a noisy prefill.
    csv_text2 = header
    t = 0
    for step in range(3):
        for i in range(15):
            csv_text2 += f"{t},400,{i},1,1,1,32,1,1,32,0,0,,,,,\"GPU (0)\",1,,7,k{i%5}\n"
            t += 150 + (i % 3) * 50  # small jitter inside a step
        t += 9_000_000  # inter-step gap
    kernels2 = parse_cuda_gpu_trace_csv(csv_text2)
    groups2 = cluster_decode_steps(kernels2)
    assert len(groups2) == 3, len(groups2)
    for g in groups2:
        assert g.count() == 15, g.count()

    print("all self-tests passed")


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selftest", action="store_true", help="run tests() and exit")
    sub = p.add_subparsers(dest="cmd")

    pc = sub.add_parser("cluster", help="parse a cuda_gpu_trace CSV, print the step summary as JSON")
    pc.add_argument("csv", help="path to the CSV, or - for stdin")
    pc.add_argument("--drop-first", type=int, default=1, help="leading groups to drop as prefill/warmup")
    pc.set_defaults(func=_cli_cluster)

    pcmp = sub.add_parser("compare", help="diff two instrument result JSON files")
    pcmp.add_argument("before")
    pcmp.add_argument("after")
    pcmp.set_defaults(func=_cli_compare)

    args = p.parse_args(argv)
    if args.selftest:
        tests()
        return 0
    if not getattr(args, "cmd", None):
        p.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
