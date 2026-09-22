"""Shared NVFP4 MoE geometry presets for the prefill benchmark and tuner scripts.

`bench_moe_prefill_gemm.py` (measures) and `tune_nvfp4_moe.py` (measures *and writes*
the tuned JSON configs the server loads) each need the same "which model has which
shape" table. A single copy here is what stops the two scripts from drifting into two
different ideas of what "Ornith" means -- the exact failure mode that produced a stale
docstring in `freetoken.moe.fused_nvfp4` before this module existed.

Deliberately kept under `benchmarks/`, not `python/freetoken/`: both scripts already
own this directory, and this table is a benchmark/tuner-only convenience, not
something the runtime server package needs at serve time (the server reads the
*written JSON*, never this module).
"""

from __future__ import annotations

import argparse

# Nemotron-3.5-Lightning MoE geometry (ungated ReLU^2): gate_up is [I, H], down is
# [H, I]. This is the historical hardcoded geometry both scripts shipped with, and
# stays the no-flags default for both.
NEMOTRON = dict(hidden=2688, intermediate=1856, experts=128, top_k=6,
                 activation="relu2", gated=False)

# Ornith-1.5-35B-A3B-NVFP4 (gated SiLU): gate_up is [2*I, H] (gate and up
# concatenated on N), down is [H, I].
ORNITH = dict(hidden=2048, intermediate=512, experts=256, top_k=8,
               activation="silu", gated=True)

MODEL_GEOMETRIES = {"nemotron": NEMOTRON, "ornith": ORNITH}


def add_geometry_args(p: argparse.ArgumentParser, defaults: dict = NEMOTRON) -> None:
    """Register ``--model`` plus the individual override flags on ``p``.

    ``--model`` sets the whole geometry group in one go (so a sweep or a write cannot
    end up mixing fields from two different models by a forgotten override); any
    explicit individual flag wins over the preset -- see :func:`resolve_geometry`.
    """
    p.add_argument("--model", choices=sorted(MODEL_GEOMETRIES), default=None,
                   help="convenience preset for the whole MoE geometry (nemotron, the "
                        "default, or ornith); explicit --hidden/--intermediate/--experts/"
                        "--top-k/--activation/--gated override individual fields of it")
    p.add_argument("--hidden", type=int, default=None,
                   help=f"hidden size H (default: model preset; nemotron={defaults['hidden']})")
    p.add_argument("--intermediate", type=int, default=None,
                   help="MoE intermediate size I (default: model preset; "
                        f"nemotron={defaults['intermediate']})")
    p.add_argument("--experts", type=int, default=None,
                   help=f"number of experts E (default: model preset; nemotron={defaults['experts']})")
    p.add_argument("--top-k", dest="top_k", type=int, default=None,
                   help=f"routed top-k (default: model preset; nemotron={defaults['top_k']})")
    p.add_argument("--activation", choices=("relu2", "silu"), default=None,
                   help=f"activation (default: model preset; nemotron={defaults['activation']!r})")
    p.add_argument("--gated", action=argparse.BooleanOptionalAction, default=None,
                   help="gate_up bank N = 2*intermediate (gate and up concatenated) "
                        "when set; ungated (N = intermediate) when --no-gated "
                        f"(default: model preset; nemotron={'gated' if defaults['gated'] else 'ungated'})")


def resolve_geometry(args: argparse.Namespace) -> dict:
    """``--model`` preset, then any explicit individual flag overrides it field-by-field.

    No flags at all -> the "nemotron" preset, i.e. the historical hardcoded geometry --
    the no-flags path is unchanged in both scripts that call this.
    """
    preset = MODEL_GEOMETRIES[args.model or "nemotron"]
    return dict(
        hidden=args.hidden if args.hidden is not None else preset["hidden"],
        intermediate=args.intermediate if args.intermediate is not None else preset["intermediate"],
        experts=args.experts if args.experts is not None else preset["experts"],
        top_k=args.top_k if args.top_k is not None else preset["top_k"],
        activation=args.activation if args.activation is not None else preset["activation"],
        gated=args.gated if args.gated is not None else preset["gated"],
    )


def gate_up_n(intermediate: int, gated: bool) -> int:
    """gate_up bank's N: ``2*intermediate`` (gate and up concatenated) when gated,
    else ``intermediate``."""
    return 2 * intermediate if gated else intermediate
