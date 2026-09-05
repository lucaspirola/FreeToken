"""Rebuild the per-class arm table from a probe session log (the JSON lost its class keys
in session 1's shadowed loop variable)."""
import ast
import re
import sys

for path in sys.argv[1:]:
    cls = None
    print(f"## {path}")
    for line in open(path):
        line = line.rstrip("\n")
        m = re.match(r"^=== (\w+): (\d+) prompt tokens", line)
        if m:
            cls = m.group(1)
            print(f"\n[{cls}] prompt={m.group(2)}")
            continue
        m = re.match(r"^  (off2?|on/\S+):\s*(\d+) tok, ([\d.]+) tok/s(.*)$", line)
        if m:
            arm, ntok, tps, rest = m.groups()
            extra = ""
            d = re.search(r"lambda=([\d.]+|None), (\{.*\})$", rest)
            if d:
                st = ast.literal_eval(d.group(2))
                c = st["cost_ms"]
                extra = (f" lam={d.group(1)} verify={st['verify_steps']} "
                         f"rate={st['draft_rate']} acc={st['accept_rate']} "
                         f"tpv={st['tokens_per_verify_step']} uneco={st['declined_uneconomic']} "
                         f"| step={c['total']} prep={c['prep']} launch={c['launch']} "
                         f"sync={c['sync']} commit={c['commit']} gpu_fwd={c['gpu_forward']} "
                         f"gpu_commit={c['gpu_commit']} drain={c['drain']}")
            print(f"  {arm:<12} {tps:>8} tok/s ({ntok} tok){extra}")
            continue
        if line.startswith("  identical="):
            print("  " + line.strip())
