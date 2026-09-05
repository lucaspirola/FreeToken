"""Engagement policy on a FIXED transcript — the comparison the throughput arms cannot make.

Speculation perturbs its own token stream, so an end-to-end copy-class arm measures "where
did this run start copying" as much as it measures the drafter. Replaying the *baseline*
(non-speculative) transcript through the shipped `NgramDrafter` removes that: the greedy
continuation is known, so acceptance is exact and both policies see the same stream.

Policies, all with the same n and k:
  post-drain-exact   the ideal: engage whenever the drafter would fire on the current list
  stale-exact        the shipped 2026-09-05 peek: ask the EXACT question of the one-token-
                     stale list (the answer for the previous position)
  stale-superset     the new peek: ask whether any indexed n-gram starts with the n-1
                     tokens already held, then re-test exactly after the drain
"""
import json, sys
from pathlib import Path

sys.path.insert(0, "python")
sys.path.insert(0, "benchmarks")
from freetoken.scheduler.spec_ngram import NgramDrafter, accepted_count  # noqa: E402

DECODE_MS = 7.3          # graphed decode step, bs=1, measured
# End-to-end verify step, measured on the copy class (SpecStats.cost_ms["total"], 20-120
# samples per point) for the optimised path; the shipped path is the same forward plus a
# 280-launch commit and the general batch prep.
VERIFY_V1 = {4: 33.0, 8: 36.0, 12: 45.0, 16: 55.0}
VERIFY_V0 = {k: v + 18.4 for k, v in VERIFY_V1.items()}   # measured v0-v1 gap at k=8
VERIFY = VERIFY_V1
def VERIFY_MS(k):
    ks = sorted(VERIFY)
    return VERIFY[min(ks, key=lambda x: abs(x - k))]


def replay(tokens, start, n, k_max, policy, drain_ms=0.0):
    d = NgramDrafter(n)
    pos = start
    k = k_max
    drained = True          # the first step after prefill has nothing in flight
    peeks = plain = verify = drained_plain = 0
    accepted_tot = drafted_tot = emitted = 0
    ms = 0.0
    while pos < len(tokens):
        stale = tokens[: pos if drained else pos - 1]
        d.observe(stale)
        peeks += 1
        if policy == "post-drain-exact":
            d.observe(tokens[:pos])
            engaged = d.has_match(tokens[:pos])
        elif policy == "stale-exact":
            engaged = d.has_match(stale)
        else:
            engaged = d.could_match(stale)
        if not engaged:
            plain += 1; pos += 1; emitted += 1; ms += DECODE_MS; drained = False
            continue
        d.observe(tokens[:pos])
        draft = d.draft(tokens[:pos], k)
        if not draft:                      # false positive: paid the drain, no verify step
            drained_plain += 1; plain += 1; pos += 1; emitted += 1
            ms += DECODE_MS + drain_ms; drained = True
            continue
        acc = accepted_count(draft, tokens[pos : pos + len(draft) + 1])
        step = min(acc + 1, len(tokens) - pos)
        verify += 1; drafted_tot += len(draft); accepted_tot += acc
        ms += VERIFY_MS(len(draft)) + (0.0 if drained else drain_ms)
        pos += step; emitted += step; drained = True
        if acc == len(draft):
            k = k_max
        elif acc == 0:
            k = max(1, k // 2)
    steps = plain + verify
    return {
        "steps": steps, "verify": verify, "draft_rate": verify / steps if steps else 0,
        "lambda": emitted / steps if steps else 0,
        "accept_rate": accepted_tot / drafted_tot if drafted_tot else 0,
        "tok_per_verify": (accepted_tot + verify) / verify if verify else 0,
        "false_pos": drained_plain,
        "ms_per_token": ms / emitted if emitted else 0,
    }


def main():
    from transformers import AutoTokenizer
    repo = Path(".").resolve()
    from probe_ngram_spec import build_prompts, render_prompt
    tok = AutoTokenizer.from_pretrained(sys.argv[1])
    prompts = build_prompts(repo)
    data = json.load(open(sys.argv[2]))
    global VERIFY
    drain = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    for cls, row in data["classes"].items():
        if cls not in prompts:
            continue
        ids = render_prompt(tok, prompts[cls])
        stream = list(ids) + list(row["off"]["token_ids"])
        print(f"\n[{cls}] prompt {len(ids)} + output {len(row['off']['token_ids'])} "
              f"(drain penalty {drain} ms)")
        print(f"  {'n':>2} {'k':>2}  {'policy':<16} {'rate':>6} {'lam':>5} {'acc':>6} "
              f"{'tpv':>6} {'fp':>4} {'ms/tok':>7} {'speedup':>8}")
        for n in (6, 8, 10):
            for k in (4, 8, 12, 16):
                for arm, table in (("v0", VERIFY_V0), ("v1", VERIFY_V1)):
                    VERIFY = table
                    pol = "stale-exact" if arm == "v0" else "stale-superset"
                    r = replay(stream, len(ids), n, k, pol, drain_ms=drain)
                    print(f"  {n:>2} {k:>2}  {arm + '/' + pol:<16} {r['draft_rate']:>6.3f} "
                          f"{r['lambda']:>5.2f} {r['accept_rate']:>6.3f} "
                          f"{r['tok_per_verify']:>6.2f} {r['false_pos']:>4} "
                          f"{r['ms_per_token']:>7.2f} "
                          f"{DECODE_MS / r['ms_per_token']:>8.3f}")


main()
