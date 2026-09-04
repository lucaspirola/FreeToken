# CPU checks

Most of FreeToken's test suite needs an NVIDIA GPU, so it has always been run by hand.
A meaningful slice of it does not, and that slice now runs on every push and pull
request in [`.github/workflows/cpu-checks.yml`](../.github/workflows/cpu-checks.yml)
on a GitHub-hosted runner — no GPU, no CUDA toolkit, no self-hosted node, no secrets.

The GPU tests, the AIME gates and the real-hardware A/B numbers a PR needs are
unchanged and still yours to run (see [CONTRIBUTING.md](../CONTRIBUTING.md) and
[tests/README.md](../tests/README.md)). This page covers only what CI does for you.

## What runs

| check | what it protects |
|-------|------------------|
| `ruff check .` | syntax errors, undefined names, redefinitions, broken format strings |
| `pytest` over the CPU-runnable directories | scheduler, server, kvcache, DSV4, Nemotron-H |
| `benchmarks/scheduler_replay.py --gate` | prefill admission throughput under sustained mixed-length traffic |

The whole job is a few minutes, almost all of it dependency installation.

### 1. `ruff check`

Ruff's default rule set (`E4`, `E7`, `E9`, `F`) minus the codes listed under
`[tool.ruff.lint] ignore` in `pyproject.toml`. That ignore list is *exactly* the set of
rules the tree was already violating when the gate was introduced, with the violation
count recorded next to each one — so the gate fires only on something newly introduced,
and never on pre-existing style. What stays enabled is the part that catches real bugs:
syntax errors (`E9`), undefined names and redefinitions (`F821`, `F811`), `is` against a
literal (`F632`), and malformed `%`/`.format` calls (`F50x`).

To pay down a line of that debt, clear the violations and delete the line in the same PR.

### 2. Unit tests

```
tests/scheduler  tests/server  tests/kvcache  tests/dsv4
tests/models/test_nemotron_h{,_chunked_prefill,_hidden_states,_mamba2_path}.py
```

All 1,290 tests collected from those paths run on a CUDA-less box: the GPU-dependent
ones skip themselves (51 skips) and the rest pass. There is no deselect list and no new
marker — the existing `torch.cuda.is_available()` guards already do the job.

Directories deliberately left out for now:

- **`tests/moe`** — one failure without flashinfer installed
  (`test_offload.py::test_adjust_config_converts_moe_cache_rate_to_cache_size`, which
  asks for the `fi` attention backend). The other 154 tests across `tests/moe` and
  `tests/kernels` do pass on CPU, so this is a cheap follow-up once that one test skips
  instead of failing.
- **`tests/e2e`** — boots a real server against a local checkpoint (`needs_weights`).
- **`tests/engine`, `tests/daemon`, `tests/tokenizer`, the rest of `tests/models`** —
  not yet measured on a CPU-only runner. Add them once they are.

### 3. Scheduler replay gate

`benchmarks/scheduler_replay.py` replays the Switchyard stage route — 16 concurrent
clients, prompt lengths from 2K to 118K, ~75% prefix reuse, a 262,144-token growable
pool — through the *real* `PrefillManager` / `CacheManager` / `TableManager` /
`DecodeManager`. No GPU, no model, no kernels: it drives the scheduling logic directly
and counts what got prefilled and what finished.

`--gate` runs two fixed scenarios at seed 7 for 20,000 forwards and fails if throughput
or completions fall under the recorded floors, or if the scheduler raises at all:

| profile | prefilled tokens | completions |
|---------|------------------|-------------|
| `stage` (the soak's scenario mix) | ≥ 6,500,000 | ≥ 350 |
| `pressure` (long prompts crowd the queue) | ≥ 8,500,000 | ≥ 85 |

Measured at `81ab30e`, seed 7, 20,000 forwards, torch 2.11 CPU / Python 3.12:
stage **7,049,549 tokens / 373 completions**, pressure **10,071,808 / 99**. The floors
sit ~8–13% under that, which is slack enough for scheduling jitter and far tighter than
the regression class they exist for: the pre-fix trees managed roughly half the stage
throughput, or raised out of `_prepare_batch` outright.

The run is deterministic — identical counts on 2 cores and on 12 — and takes about
4 seconds wall on two cores, so it costs nothing to keep in CI.

This is the only check here that catches the prefill admission-gate starvation class of
bug. No unit test sees it: it needs sustained mixed-length traffic against a pool under
real pressure before the scheduler stops admitting.

Investigating a failure:

```bash
uv run --no-project python benchmarks/scheduler_replay.py \
  --ticks 20000 --seed 7 --profile stage --diagnose
```

`--diagnose` prints one JSON line with a per-refusal breakdown — which check refused
each fresh admit, how much pool headroom there was at the time, and how many queued
requests behind the refused one the pools could actually have seated. That breakdown is
what identified the original starvation.

If a deliberate scheduling change moves these numbers, update `GATE_CASES` in the script
**and** the measured values in the comment above it, in the same commit that moves them.

## Running the checks locally

The workflow does not `pip install -e .`: `setup.py` links
`freetoken.kernel._pinned_tensor`, `_cpu_moe` and `_pageable_stage` against `libcudart`
and hard-requires `CUDA_HOME`, so an editable install cannot work on a machine with no
CUDA toolkit. Nothing in these checks needs the compiled extensions, so CI installs the
dependencies only and puts `python/` on `PYTHONPATH`.

If you already have a normal `[accel]` development install, just run the three commands
against it — `ruff check .`, the `pytest` line, and the replay gate. To reproduce CI's
environment exactly:

```bash
uv venv --python 3.12
export VIRTUAL_ENV="$PWD/.venv" PYTHONPATH="$PWD/python"

# [project.dependencies] is the single source of truth; the project version is dynamic,
# so uv cannot read the list without building the project.
python3 - <<'PY' > /tmp/cpu-requirements.txt
import tomllib
with open("pyproject.toml", "rb") as fh:
    print("\n".join(tomllib.load(fh)["project"]["dependencies"]))
PY

# --no-sources drops the [tool.uv.sources] cu130 pin (a ~3 GB CUDA torch this never
# uses); --torch-backend=cpu then resolves torch from PyTorch's CPU index.
uv pip install --no-sources --torch-backend=cpu \
  -r /tmp/cpu-requirements.txt "pytest>=6.0" "ruff==0.15.12"

uv run --no-project ruff check .
uv run --no-project pytest -q tests/scheduler tests/server tests/kvcache tests/dsv4 \
  tests/models/test_nemotron_h.py tests/models/test_nemotron_h_chunked_prefill.py \
  tests/models/test_nemotron_h_hidden_states.py tests/models/test_nemotron_h_mamba2_path.py
uv run --no-project python benchmarks/scheduler_replay.py --gate
```

`--no-project` on `uv run` matters: without it uv discovers `pyproject.toml` and tries to
build and install the project, which is the thing that cannot work here.
