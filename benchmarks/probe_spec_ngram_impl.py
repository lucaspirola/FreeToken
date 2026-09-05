"""Measure the shipped `--speculative ngram` path: greedy equivalence, tok/s, acceptance.

One model load, two arms per prompt, toggled on the live scheduler (``Scheduler._spec``)
so both arms share the same weights, the same expert cache and the same warm prefix tree.
The prefix tree is warmed with a 1-token generation before either arm, so neither arm pays
a prefill the other does not -- what is being compared is the decode phase.

Arms:
  off   the ordinary overlapped graphed decode step
  on    the same, plus a drained verify step whenever the n-gram drafter fires

Reported per prompt class: decode tok/s, the token ids (compared for greedy equivalence),
and the drafter's own accounting -- verify steps, draft rate, per-token acceptance, and
lambda (tokens emitted per scheduler step).

    FREETOKEN_PIN_BUDGET_GB=17 PYTHONPATH=python scripts/gpu_lock.sh .venv/bin/python \\
      benchmarks/probe_spec_ngram_impl.py --model <lightning> --out results.json \\
      --moe-cache-auto
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_ngram_spec import build_prompts, render_prompt  # noqa: E402

PASSCODE = "The passcode for the vault is 8371-QUARTZ-9042."


def needle_prompt(tok, repo: Path, target_tokens: int) -> tuple[list[int], str]:
    """A long-context haystack with one recallable fact a third of the way in."""
    filler = (repo / "docs/nemotron.md").read_text()
    unit = len(tok(filler, add_special_tokens=False)["input_ids"])
    # FLOOR, not ceil: overshooting max_seq_len makes the engine drop the request, and the
    # offline LLM turns that into an assertion 200 lines away from the cause.
    body = [filler] * max(1, target_tokens // unit)
    text = "\n\n".join(body)
    cut = len(text) // 3
    text = text[:cut] + f"\n\n{PASSCODE}\n\n" + text[cut:]
    content = (
        "Read the following document, then answer the question at the end.\n\n"
        + text
        + "\n\nQuestion: what is the passcode for the vault? Answer with the passcode only."
    )
    ids = render_prompt(tok, content)
    return ids, PASSCODE


def _stats(spec):
    s = spec.stats
    steps = s.verify_steps + s.plain_peeks
    return {
        "verify_steps": s.verify_steps,
        "plain_peeks": s.plain_peeks,
        "draft_rate": round(s.verify_steps / steps, 4) if steps else 0.0,
        "drafted_tokens": s.drafted_tokens,
        "accepted_tokens": s.accepted_tokens,
        "accept_rate": round(s.accept_rate, 4),
        "tokens_per_verify_step": round(s.tokens_per_verify, 3),
        "declined_shape": s.declined_shape,
        "declined_no_slot": s.declined_no_slot,
        "declined_budget": s.declined_budget,
        "declined_stale_match": s.declined_stale_match,
        "declined_uneconomic": s.declined_uneconomic,
        # Seeded gate. Without these the only evidence that the probes ran is arithmetic on
        # ``drafted_tokens``, which is unambiguous only while the step count is tiny.
        "seed_probe_steps": s.seed_probe_steps,
        "seed_fits": s.seed_fits,
        "cost_ms": s.cost_ms,
    }


# Each optimisation of the 2026-09-05 verify-step work, switchable so one model load can
# measure the shipped path and the new one against the same off/off2 control.
VARIANTS = {
    "v0": dict(post_drain=False, fused_commit=False, fast_prep=False, gate_seed=False),
    "v1": dict(post_drain=True, fused_commit=True, fast_prep=True, gate_seed=False),
    "drain": dict(post_drain=True, fused_commit=False, fast_prep=False, gate_seed=False),
    "commit": dict(post_drain=False, fused_commit=True, fast_prep=True, gate_seed=False),
}
# The break-even gate priced from two NARROW verify steps (m = 2, m = 4) fitted to
# t(m) = a + b*m, instead of _GATE_MIN_SAMPLES full-width ones. ``gate_seed`` is carried by
# EVERY variant above, not only this one: a key absent from VARIANTS["v1"] inherits the
# previous arm's value, which is how a "v0, k = 8" arm once ran at the preceding arm's k = 16.
VARIANTS["seed"] = dict(VARIANTS["v1"], gate_seed=True)
# Repeats of v1 under different labels: the copy class's draft rate depends on the token
# stream, which speculation itself perturbs, so "is this arm reproducible?" needs arms that
# differ in nothing at all.
VARIANTS["v1b"] = dict(VARIANTS["v1"])
VARIANTS["v1c"] = dict(VARIANTS["v1"])
for _k in (4, 12, 16, 24):
    VARIANTS[f"k{_k}"] = dict(VARIANTS["v1"])   # draft_len filled in from the CLI in main()


def run_arm(llm, spec, ids: list[int], sp, *, on: bool, label: str | None = None,
            flags: dict | None = None) -> dict:
    from freetoken.scheduler.spec_ngram import SpecStats

    llm._spec = spec if on else None
    if spec is not None:
        # EVERY tunable, every arm: a variant that names only some of them would otherwise
        # inherit the previous arm's values (measured the hard way -- a "v0" arm ran at the
        # preceding arm's --spec-draft-len 16 and reported 15.5 tokens per verify step at
        # k = 8, which is impossible).
        for k, v in {**VARIANTS["v1"], **(flags or {})}.items():
            setattr(spec, k, v)
    if spec is not None:
        spec.stats = SpecStats()
        spec._state = None
        spec._last_peek_at = None
        spec._last_peek_hit = False
    t0 = time.perf_counter()
    out = llm.generate([list(ids)], sp)[0]
    wall = time.perf_counter() - t0
    n = len(out["token_ids"])
    row = {
        "arm": label or ("on" if on else "off"),
        "tokens": n,
        "wall_s": round(wall, 3),
        "tok_per_s": round(n / wall, 2) if wall > 0 else 0.0,
        "token_ids": out["token_ids"],
        "text": out["text"],
    }
    if on and spec is not None:
        row["spec"] = _stats(spec)
        st = row["spec"]
        steps = st["verify_steps"] + st["plain_peeks"]
        row["lambda"] = round(n / steps, 3) if steps else None
    return row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--needle-tokens", type=int, default=131072)
    p.add_argument("--needle-max-tokens", type=int, default=48)
    p.add_argument("--spec-ngram-n", type=int, default=8)
    p.add_argument("--spec-draft-len", type=int, default=8)
    p.add_argument("--no-spec-adaptive", action="store_true")
    p.add_argument("--moe-cache-auto", action="store_true")
    p.add_argument("--memory-ratio", type=float, default=0.85)
    p.add_argument("--num-tokens", type=int, default=140000)
    p.add_argument("--max-seq-len", type=int, default=140000)
    p.add_argument("--host-ram-reserve-gb", type=float, default=6.0)
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument(
        "--variants", nargs="*", default=["v1"],
        help=f"speculative arms to run, in order; one of {sorted(VARIANTS)}",
    )
    p.add_argument(
        "--sweep-k", nargs="*", type=int, default=None,
        help="draft lengths to sweep (one arm each, all on the v1 path)",
    )
    p.add_argument(
        "--sweep-n", nargs="*", type=int, default=None,
        help="n-gram widths to sweep; crossed with --sweep-k",
    )
    p.add_argument("--skip-needle", action="store_true")
    args = p.parse_args()

    from freetoken.core import SamplingParams
    from freetoken.llm.llm import LLM

    repo = Path(__file__).resolve().parent.parent
    kwargs = dict(
        model_path=args.model,
        dtype=torch.bfloat16,
        attention_backend="triton",
        moe_backend="offload",
        nvfp4_backend="triton",
        max_running_req=1,
        max_extend_tokens=8192,
        memory_ratio=args.memory_ratio,
        max_seq_len_override=args.max_seq_len,
        num_token_override=args.num_tokens,
        kv_cache_dtype="q8_0",
        cuda_graph_max_bs=1,
        session_spill_dir=None,
        host_ram_reserve_gb=args.host_ram_reserve_gb,
        speculative="ngram",
        spec_ngram_n=args.spec_ngram_n,
        spec_draft_len=args.spec_draft_len,
        spec_adaptive=not args.no_spec_adaptive,
    )
    if args.moe_cache_auto:
        kwargs["moe_cache_auto"] = True

    llm = LLM(**kwargs)
    spec = llm._spec
    assert spec is not None, "speculation was refused by the engine; see the warning above"
    tok = llm.tokenizer

    warm = SamplingParams(temperature=0.0, max_tokens=1)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    # A sweep is a set of v1 arms differing only in (n, k); the decoder reads both off
    # itself when run_arm resets its per-request state, so one model load covers the grid.
    # Every named variant carries every tunable, seeded from the CLI, so no arm can inherit
    # a value from the arm before it.
    for _v in VARIANTS.values():
        _v["n"] = args.spec_ngram_n
        _v["draft_len"] = args.spec_draft_len
    for _k in (4, 12, 16, 24):
        VARIANTS[f"k{_k}"]["draft_len"] = _k
    if args.sweep_k or args.sweep_n:
        ks = args.sweep_k or [args.spec_draft_len]
        ns = args.sweep_n or [args.spec_ngram_n]
        args.variants = []
        for nn in ns:
            for k in ks:
                label = f"n{nn}k{k}"
                VARIANTS[label] = dict(VARIANTS["v1"], n=nn, draft_len=k)
                args.variants.append(label)

    prompts = build_prompts(repo)
    names = args.only or list(prompts)
    results: dict = {
        "config": {
            "variants": {k: VARIANTS[k] for k in args.variants},
            "n": args.spec_ngram_n,
            "draft_len": args.spec_draft_len,
            "adaptive": not args.no_spec_adaptive,
            "max_tokens": args.max_tokens,
        },
        "classes": {},
    }

    cases = [(name, render_prompt(tok, prompts[name]), sp) for name in names]
    if not args.skip_needle:
        ids, _ = needle_prompt(tok, repo, args.needle_tokens)
        cases.append(
            ("needle", ids, SamplingParams(temperature=0.0, max_tokens=args.needle_max_tokens))
        )

    for name, ids, params in cases:
        print(f"=== {name}: {len(ids)} prompt tokens", flush=True)
        llm._spec = None
        # A 1-token call warms the prefix tree; a short decode warms the expert cache and the
        # decode graph for THIS prompt, so the first timed arm is not measuring a cold cache
        # (the first run had off=84 tok/s against on=123 for exactly that reason).
        llm.generate([list(ids)], warm)
        llm.generate([list(ids)], SamplingParams(temperature=0.0, max_tokens=16))
        off = run_arm(llm, spec, ids, params, on=False)
        print(f"  off: {off['tokens']} tok, {off['tok_per_s']} tok/s", flush=True)
        arms = {}
        for vname in args.variants:  # NOT `name`: that is the prompt class, still in scope
            arm = run_arm(
                llm, spec, ids, params, on=True, label=vname, flags=VARIANTS[vname]
            )
            arms[vname] = arm
            print(
                f"  on/{vname}: {arm['tokens']} tok, {arm['tok_per_s']} tok/s, "
                f"lambda={arm.get('lambda')}, {arm.get('spec')}",
                flush=True,
            )
        on = arms[args.variants[-1]]
        # Control arm: spec OFF again, run last. Without it an "on != off" verdict cannot
        # distinguish speculation from ordinary run-to-run nondeterminism of the engine.
        off2 = run_arm(llm, spec, ids, params, on=False, label="off2")
        print(f"  off2:{off2['tokens']} tok, {off2['tok_per_s']} tok/s", flush=True)
        identical = off["token_ids"] == on["token_ids"]
        first_diff = None
        if not identical:
            for i, (a, b) in enumerate(zip(off["token_ids"], on["token_ids"])):
                if a != b:
                    first_diff = i
                    break
            if first_diff is None:
                first_diff = min(len(off["token_ids"]), len(on["token_ids"]))
        control_identical = off["token_ids"] == off2["token_ids"]
        results["classes"][name] = {
            "prompt_tokens": len(ids),
            "off": off,
            "on": on,
            "arms": arms,
            "off2": off2,
            "identical": identical,
            "control_identical": control_identical,
            "first_diff": first_diff,
            "speedup": (
                round(on["tok_per_s"] / off["tok_per_s"], 3) if off["tok_per_s"] else None
            ),
            "speedups": {
                k: (round(a["tok_per_s"] / off["tok_per_s"], 3) if off["tok_per_s"] else None)
                for k, a in arms.items()
            },
        }
        if name == "needle":
            code = PASSCODE.split()[-1].rstrip(".")
            results["classes"][name]["passcode_off"] = code in off["text"]
            results["classes"][name]["passcode_on"] = code in on["text"]
        print(
            f"  identical={identical} control_identical={control_identical} "
            f"first_diff={first_diff} speedup={results['classes'][name]['speedup']}",
            flush=True,
        )

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}", flush=True)

    summary = copy.deepcopy(results)
    for row in summary["classes"].values():
        row.pop("arms", None)
        for arm in ("off", "on", "off2"):
            row[arm].pop("token_ids", None)
            row[arm]["text"] = row[arm]["text"][:120]
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
