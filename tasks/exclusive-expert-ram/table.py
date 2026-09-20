#!/usr/bin/env python3
"""Build results/sweep.tsv from the per-arm record files.

measure.sh used to append straight to the TSV, writing the header from the
FIRST arm's keys and each row from its own. A baseline arm has no mirror
counters and a mirror arm has three, so the header ended up 19 columns wide
over 22-column rows and every mirror number was read against the wrong name.
Columns are fixed here instead, and a missing value is an empty cell.

Usage: table.py [results_dir]
"""
from __future__ import annotations

import json
import pathlib
import sys

COLUMNS = [
    "arm", "model", "rows", "ratio",
    # The RAM answer, then its attribution (results/README.md).
    "ram_gib", "rss_ready_gib", "devzero_gib", "ram_after_80k_gib", "rss_gib", "locked_gib", "anon_gib", "file_gib", "shmem_gib",
    "unevict_gib", "current_gib", "peak_current_gib", "gpu_mib",
    "decode_8k", "ttft_8k", "decode_32k", "ttft_32k", "decode_80k", "ttft_80k",
    "swaps", "free_evict_rate", "retained_rows", "coverage_faults", "starved",
    "probe_error",
]


def main(root: str = None) -> None:
    out = pathlib.Path(root or pathlib.Path(__file__).with_name("results"))
    records = []
    for path in sorted(out.glob("*-record.json")):
        records.append(json.loads(path.read_text()))
    records.sort(key=lambda r: (r.get("model", ""), int(r.get("rows") or 0)))
    lines = ["\t".join(COLUMNS)]
    for rec in records:
        lines.append("\t".join(str(rec.get(c, "")) for c in COLUMNS))
    (out / "sweep.tsv").write_text("\n".join(lines) + "\n")
    print(f"{len(records)} arms -> {out / 'sweep.tsv'}")


if __name__ == "__main__":
    main(*sys.argv[1:])
