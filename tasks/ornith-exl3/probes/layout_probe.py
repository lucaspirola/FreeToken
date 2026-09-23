"""Check an EXL3 checkpoint's safetensors headers against quantization_config.json and
report whether each routed expert is one contiguous extent (stdlib only).

usage: python3 layout_probe.py ~/ai/models/Ornith-1.5-35B-A3B-exl3-5.0bpw-hq
"""
import collections
import glob
import json
import os
import re
import struct
import sys

root = sys.argv[1]
q = json.load(open(os.path.join(root, "quantization_config.json")))["tensor_storage"]
st = {k: s for v in q.values() for k, s in v["stored_tensors"].items()}
hdrs = {}
for f in sorted(glob.glob(os.path.join(root, "*.safetensors"))):
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        h = json.loads(fh.read(n))
    for k, v in h.items():
        if k != "__metadata__":
            hdrs[k] = (f, 8 + n + v["data_offsets"][0], v["data_offsets"][1] - v["data_offsets"][0], v["dtype"], v["shape"])
dt = {"I16": "torch.int16", "F16": "torch.float16", "BF16": "torch.bfloat16", "I32": "torch.int32", "F32": "torch.float32"}
bad = [k for k, s in st.items() if dt[hdrs[k][3]] != s["dtype"] or list(hdrs[k][4]) != list(s["shape"]) or hdrs[k][2] != s["n_bytes"]]
print("storage entries", len(st), "header mismatches", len(bad))
ex = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.(trellis|suh|svh|mul1)$")
groups = collections.defaultdict(list)
for k, v in hdrs.items():
    m = ex.match(k)
    if m:
        groups[(int(m[1]), int(m[2]))].append((v[0], v[1], v[2], m[3], m[4]))
files, contiguous, orders, sizes = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
for t in groups.values():
    files[len({x[0] for x in t})] += 1
    t = sorted(t, key=lambda x: (x[0], x[1]))
    total = sum(x[2] for x in t)
    sizes[total] += 1
    contiguous[t[-1][1] + t[-1][2] - t[0][1] == total and len({x[0] for x in t}) == 1] += 1
    orders[tuple((x[3], x[4]) for x in t)] += 1
print("experts", len(groups), "files/expert", dict(files), "contiguous", dict(contiguous), "extent bytes", dict(sizes))
for o, n in orders.most_common(2):
    print(n, o)
