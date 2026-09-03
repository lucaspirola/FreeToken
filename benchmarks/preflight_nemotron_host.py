#!/usr/bin/env python3
"""Host preflight for serving NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4.

Deliberately torch-free (stdlib + ``nvidia-smi``) so it can run *before* the venv
imports CUDA: a stray llama-server or a leftover FreeToken worker has to be found
while there is still a terminal to report it to, not after 15.4 GiB of expert banks
have been read into a host that cannot hold them.

Checks, in the order a failed boot hits them:

1. host RAM -- MemAvailable must cover the resident expert banks plus the engine's
   own ``--host-ram-reserve-gb`` plus the non-bank process footprint;
2. the CUDA host-registration (pin) quota -- on WSL this is ~0.4x RAM, below the
   banks, so some MoE layers must go pageable (``--moe-pageable-gpu``), which costs
   the decode CUDA graphs. The report says how many layers, and what
   ``FREETOKEN_PIN_BUDGET_GB`` would have to be to avoid it;
3. the GPU -- free VRAM and every process holding any;
4. leftovers -- stale torch-extension build locks and stray FreeToken workers.

Exit status is 0 when the host is ready, 1 when something must be fixed first
(and 2 on a usage error).

    python benchmarks/preflight_nemotron_host.py [--max-running-requests 16]
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

GIB = 1 << 30

# --- Lightning geometry (config.json), mirrored in kernel/aot_models.py ------------
HIDDEN_SIZE = 2688
MOE_INTERMEDIATE_SIZE = 1856
NUM_EXPERTS = 128
NUM_MOE_LAYERS = 23
TOP_K = 6

# Engine defaults these mirror: EngineConfig.host_ram_reserve_gb, and the non-bank
# resident footprint measured at bring-up (dense weights + CUDA context + the
# tokenizer/scheduler processes).
DEFAULT_RESERVE_GIB = 3.0
PROCESS_FOOTPRINT_GIB = 4.0
# Weights 2.26 GiB + Mamba slots + KV + the smallest useful MoE slot cache.
MIN_GPU_FREE_GIB = 13.0
# Anything below this is measurement noise (a compositor, nvidia-smi itself).
FOREIGN_VRAM_LIMIT_MIB = 1024

VENV_PYTHON = "/home/lucas/ai/FreeToken/.venv/bin/python"


def _expert_bank_bytes() -> int:
    """Host bytes of the 23 x 128 NVFP4 routed-expert banks.

    Ungated ReLU^2 experts: the first bank is ``[I, H]``, not the gated ``[2I, H]``.
    Layout per moe/offload_cache.py ``_BANK_SCHEMAS`` (packed e2m1 pairs, per-16
    fp8-e4m3 scales, fp16 per-row globals).
    """
    h, i = HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE
    per_expert = (
        i * (h // 2)  # gate_up_packed  (ungated: I rows)
        + i * (h // 16)  # gate_up_scale
        + h * (i // 2)  # down_packed
        + h * (i // 16)  # down_scale
        + i * 2  # gate_up_global
        + h * 2  # down_global
    )
    return per_expert * NUM_EXPERTS * NUM_MOE_LAYERS


# --- host readings -----------------------------------------------------------------


def _meminfo() -> dict[str, int]:
    out: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if parts:
            out[key] = int(parts[0]) * 1024  # kB -> bytes
    return out


def _is_wsl() -> bool:
    return hasattr(os, "uname") and "microsoft" in os.uname().release.lower()


def _pin_budget_bytes() -> int | None:
    """Bytes this process can safely cudaHostRegister.

    Replicates ``engine/engine.py::_pin_budget_bytes``: WSL's WDDM-backed CUDA caps
    pinning near half of RAM, shared across processes, so the engine budgets 40%.
    ``FREETOKEN_PIN_BUDGET_GB`` overrides anywhere; plain Linux is uncapped (None).
    """
    if env := os.environ.get("FREETOKEN_PIN_BUDGET_GB"):
        return int(float(env) * GIB)
    if not _is_wsl():
        return None
    return int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") * 0.4)


def _pageable_layers(
    bank_bytes: int, budget: int | None, max_running_req: int
) -> tuple[int, list[int], int]:
    """(count, layer ids, staging bytes) the engine would move to pageable staging.

    Mirrors ``_auto_pageable_gpu_layers``: the decode gather arena is charged to the
    same budget, then a head+tail split covers the overflow.
    """
    if budget is None or bank_bytes <= budget:
        return 0, [], 0
    stage_rows = 1 << max(0, (max_running_req * TOP_K - 1).bit_length())
    row_bytes = math.ceil(bank_bytes / (NUM_MOE_LAYERS * NUM_EXPERTS))
    stage_bytes = stage_rows * row_bytes
    budget = max(0, budget - stage_bytes)
    n = min(NUM_MOE_LAYERS, math.ceil(NUM_MOE_LAYERS * (1 - budget / bank_bytes)))
    head = (n + 1) // 2
    ids = sorted(
        set(range(head)) | set(range(NUM_MOE_LAYERS - (n - head), NUM_MOE_LAYERS))
    )
    return n, ids, stage_bytes


def _nvidia_smi(query: str, extra: list[str] | None = None) -> list[list[str]]:
    cmd = ["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits"]
    if extra:
        cmd[1:1] = extra
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []
    return [
        [f.strip() for f in line.split(",")]
        for line in out.stdout.splitlines()
        if line.strip()
    ]


def _stale_extension_locks() -> list[Path]:
    root = Path.home() / ".cache" / "torch_extensions"
    if not root.is_dir():
        return []
    return sorted(root.glob("*/lock")) + sorted(root.glob("*/*/lock"))


def _stray_workers() -> list[tuple[str, str]]:
    try:
        out = subprocess.run(
            ["pgrep", "-a", "-f", VENV_PYTHON],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in out.stdout.splitlines():
        pid, _, cmd = line.partition(" ")
        if pid.strip() and int(pid) != os.getpid():
            rows.append((pid.strip(), cmd.strip()))
    return rows


# --- report ------------------------------------------------------------------------


def _gib(n: float) -> str:
    return f"{n / GIB:.2f} GiB"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--max-running-requests",
        type=int,
        default=16,
        help="the --max-running-requests the server will be launched with (sizes the "
        "pinned decode gather arena; default 16)",
    )
    ap.add_argument(
        "--host-ram-reserve-gb",
        type=float,
        default=DEFAULT_RESERVE_GIB,
        help="mirrors the server flag of the same name (default 3)",
    )
    args = ap.parse_args(argv)
    if args.max_running_requests < 1:
        ap.error("--max-running-requests must be >= 1")

    failures: list[str] = []
    warnings: list[str] = []
    mem = _meminfo()
    bank_bytes = _expert_bank_bytes()
    available = mem.get("MemAvailable", 0)
    reserve = int(args.host_ram_reserve_gb * GIB)
    required = bank_bytes + reserve + int(PROCESS_FOOTPRINT_GIB * GIB)

    print("Nemotron-3.5 Lightning host preflight")
    print("=" * 68)
    print(f"platform                : {'WSL2' if _is_wsl() else 'native Linux'}")
    print(f"MemTotal                : {_gib(mem.get('MemTotal', 0))}")
    print(f"MemAvailable            : {_gib(available)}")
    print(
        f"SwapFree                : {_gib(mem.get('SwapFree', 0))} "
        f"of {_gib(mem.get('SwapTotal', 0))}"
    )
    print(
        f"expert banks (23x128)   : {_gib(bank_bytes)} "
        f"(ungated NVFP4, H={HIDDEN_SIZE} I={MOE_INTERMEDIATE_SIZE})"
    )
    print(
        f"required host RAM       : {_gib(required)} "
        f"(banks + {args.host_ram_reserve_gb:g} GiB reserve + "
        f"{PROCESS_FOOTPRINT_GIB:g} GiB process)"
    )
    if available < required:
        failures.append(
            f"MemAvailable {_gib(available)} < required {_gib(required)}: free host RAM "
            "(stop other services, drop caches) or the bank load will OOM."
        )

    budget = _pin_budget_bytes()
    src = (
        "FREETOKEN_PIN_BUDGET_GB"
        if os.environ.get("FREETOKEN_PIN_BUDGET_GB")
        else ("0.4 x MemTotal (WSL)" if _is_wsl() else "uncapped")
    )
    print(
        f"CUDA pin budget         : {'unlimited' if budget is None else _gib(budget)} [{src}]"
    )
    n, ids, stage_bytes = _pageable_layers(
        bank_bytes, budget, args.max_running_requests
    )
    if n:
        print(
            f"pinned / pageable layers: {NUM_MOE_LAYERS - n} pinned, {n} pageable "
            f"(head+tail {ids}); {stage_bytes / (1 << 20):.1f} MiB gather arena at "
            f"--max-running-requests {args.max_running_requests}"
        )
        need_gib = (bank_bytes + stage_bytes) / GIB
        warnings.append(
            f"{n}/{NUM_MOE_LAYERS} MoE layers need --moe-pageable-gpu (which disables the "
            f"decode CUDA graphs). FREETOKEN_PIN_BUDGET_GB >= {math.ceil(need_gib)} would "
            "pin every layer instead -- only if MemTotal can back it."
        )
    else:
        print(f"pinned / pageable layers: {NUM_MOE_LAYERS} pinned, 0 pageable")

    gpus = _nvidia_smi("gpu=index,name,memory.free,memory.total")
    if not gpus:
        failures.append("nvidia-smi produced no GPU rows; is the driver up?")
    for idx, name, free_mib, total_mib in gpus:
        print(f"GPU {idx} {name:<28}: {free_mib} MiB free of {total_mib} MiB")
        if float(free_mib) < MIN_GPU_FREE_GIB * 1024:
            failures.append(
                f"GPU {idx} has only {free_mib} MiB free (need "
                f"{int(MIN_GPU_FREE_GIB * 1024)} MiB); stop whatever holds VRAM."
            )

    apps = _nvidia_smi("compute-apps=pid,process_name,used_memory")
    if apps:
        print("VRAM holders:")
        for pid, pname, used_mib in apps:
            print(f"  pid {pid:<8} {used_mib:>7} MiB  {pname}")
            if float(used_mib) > FOREIGN_VRAM_LIMIT_MIB:
                failures.append(
                    f"pid {pid} ({pname}) holds {used_mib} MiB of VRAM; kill it before serving."
                )
    else:
        print("VRAM holders           : none")

    locks = _stale_extension_locks()
    print(f"torch_extensions locks  : {len(locks)}")
    for lock in locks:
        print(f"  {lock}")
    if locks:
        warnings.append(
            f"{len(locks)} stale ~/.cache/torch_extensions lock file(s); a JIT build that "
            "died holding one blocks the next build forever. Remove them."
        )

    strays = _stray_workers()
    print(f"stray FreeToken workers : {len(strays)}")
    for pid, cmd in strays:
        print(f"  pid {pid}  {cmd[:96]}")
    if strays:
        warnings.append(
            f"{len(strays)} process(es) still running out of {VENV_PYTHON}; kill them by pid "
            '(never `pkill -f "ft serve"`, which matches this shell).'
        )

    print("=" * 68)
    for warning in warnings:
        print(f"WARN: {warning}")
    for failure in failures:
        print(f"FAIL: {failure}")
    if failures:
        print(f"NOT READY ({len(failures)} blocking issue(s))")
        return 1
    print("READY")
    return 0


if __name__ == "__main__":
    sys.exit(main())
