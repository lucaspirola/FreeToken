"""Compare greedy transcripts of job-greedy.sh arms (first arm = reference, e.g. tri; then fi; tile =
the noise yardstick).

    greedy_compare.py MODEL_DIR RESULT_DIR [REF OTHER ...]   (default arms: tri fi tile)

Per prompt size in the transcripts: whether each arm's two repeats agree (run-to-run determinism), the
first divergent generated token of every arm against the reference (tokenized with the model's
tokenizer), and whether the three records the prompt asks to quote come out with the right facts.
Full texts follow the summary.
"""
import json
import os
import sys
from collections import defaultdict

from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from greedy_client import CITIES, ITEMS, NAMES  # noqa: E402

mdir, rdir = sys.argv[1], sys.argv[2]
ARMS = sys.argv[3:] or ["tri", "fi", "tile"]
tok = AutoTokenizer.from_pretrained(mdir)


def truth(size):
    n = 0
    while n * 16 < size:
        n += 1
    # the fact, whatever markup the model wraps around "Record i:"
    return [f"{NAMES[i % 10]} left a {ITEMS[(i * 3) % 8]} in {CITIES[(i * 5) % 8]} on day {(i * 37) % 365}."
            for i in (n // 7, n // 2, n - 3)]


runs = defaultdict(dict)  # size -> arm -> rows
for arm in ARMS:
    p = os.path.join(rdir, f"{arm}.jsonl")
    if os.path.exists(p):
        for line in open(p):
            r = json.loads(line)
            runs[r["target"]].setdefault(arm, []).append(r)


def ids(row):
    return tok.encode(row["text"].replace("\x00", ""), add_special_tokens=False)


def first_div(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


summary = defaultdict(list)
for size in sorted(runs):
    by = runs[size]
    facts = truth(size)
    print(f"##### target {size}")
    for arm, rows in by.items():
        body = rows[0]["text"].replace("\x00", "")
        ok = [f in body for f in facts]
        rep = f"reps agree: {first_div(ids(rows[0]), ids(rows[1])) is None}" if len(rows) > 1 else "1 rep"
        print(f"  {arm:5s} prompt {rows[0]['prompt_tokens']} gen {rows[0]['completion_tokens']} "
              f"sha1 {rows[0]['sha1'][:10]} {rep}; quoted facts correct {ok}")
    if ARMS[0] not in by:
        continue
    ref = ids(by[ARMS[0]][0])
    for arm in ARMS[1:]:
        if arm not in by:
            continue
        other = ids(by[arm][0])
        d = first_div(ref, other)
        summary[arm].append(d)
        print(f"  {ARMS[0]} vs {arm}: first divergent token {d} of {len(ref)} / {len(other)}")
        if d is not None:
            print(f"     {ARMS[0]:5s} ..{tok.decode(ref[max(0, d - 12):d])!r} || {tok.decode(ref[d:d + 40])!r}")
            print(f"     {arm:5s} ..{tok.decode(other[max(0, d - 12):d])!r} || {tok.decode(other[d:d + 40])!r}")
print("\nfirst divergence per prompt (None = identical):")
for arm, ds in summary.items():
    print(f"  {ARMS[0]} vs {arm}: {ds}")
print("\n--- full texts (rep0) ---")
for size in sorted(runs):
    for arm, rows in runs[size].items():
        print(f"##### {size} {arm}\n{rows[0]['text'].replace(chr(0), '[/think]')}\n")
