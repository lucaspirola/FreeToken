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

`--gate` runs four fixed scenarios at seed 7 for 20,000 forwards and fails if throughput
or completions fall under the recorded floors, if the error rate exceeds its ceiling, if
the scheduler raises, if it **deadlocks**, or if the **finishability invariant** is
violated on any pass:

| profile | prefilled tokens | completions | error rate |
|---------|------------------|-------------|------------|
| `stage` (the soak's scenario mix) | ≥ 2,670,000 | ≥ 171 | — |
| `pressure` (long prompts crowd the queue) | ≥ 4,750,000 | ≥ 57 | — |
| `switchyard-stage` (adds session residency) | ≥ 1,779,000 | ≥ 208 | ≤ 0.376 |
| `switchyard-deadlock` (soak report T geometry) | ≥ 857,000 | ≥ 112 | ≤ 0.276 |

The floors sit ~5% under `d685e99`, the only tree that has passed the live 16-way
Switchyard soak (stage route: 471 requests / 0 errors / 1 STALLED interval). They are
deliberately **not** set to the best replay numbers ever recorded: `81ab30e` scored
7,049,549 / 373 on `stage` and `ea7ed7c` 6,194,304 / 375, and both then failed the live
soak — `ea7ed7c` deadlocking a 262,144-token pool permanently after 5 m 35 s.

That is why the two non-throughput checks exist, and why they are the ones that matter:

* **`deadlock` / `trailing_silence`.** A deadlock produces zero gaps *between* batch
  lines, because the silence starts after the last one and never ends. The gate reports
  the trailing half of the measurement too.
* **`invariant_violations` / `slack_min`.** Every tick, the replay checks that the set of
  requests **already admitted** is still finishable:

  ```
  owed = SUM over in-flight chunked prefills of (input_len - forwarded) + output_len
       + DecodeManager.inflight_tokens
  owed <= CacheManager.available_size + reclaimable idle lease tokens
  ```

  A violation says the pool has promised more than it can ever hand over. `ea7ed7c`
  passes every throughput floor above and fails this on 566 `switchyard-stage` passes
  (short by up to 192,242 tokens) and 2,181 `switchyard-deadlock` passes. The scheduler
  carries the same statement as an env-gated debug assertion, without the lease term (it
  has no hook for it, so its check is strictly tighter):
  `FREETOKEN_SCHEDULER_INVARIANT=warn` logs each violation (safe for a live soak),
  `=raise` fails fast.

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

### The `ornith-ada` profile (not gated)

The four gated profiles all run with `PrefillManager.max_batch_seqs == 0`, because
`_resolve_max_prefill_seqs` only caps prefill lanes for growable *quantized-GGUF MoE*
serving — the Ornith path, not Nemotron. `--profile ornith-ada` is that geometry: four
agents, a 4,096-token chunk and a 65,536-token pool (the configuration of
`benchmarks/results/ornith_ada_prefill_chunk_2026-08-31.md` and
`ornith_ada_multi_agent_scheduler_2026-08-31.md`), with the one-lane cap and the
short-prompt grouping crossover that lifts it.

```bash
uv run --no-project python benchmarks/scheduler_replay.py \
  --ticks 20000 --seed 7 --profile ornith-ada
```

It exists to show that the seatable-lanes divisor and the GGUF lane cap compose: at seed 7
the grouping arm fires (`lanes_max` 2 against 1 with the crossover at 0) and halves the
error rate, 0.0079 against 0.0159, with `invariant_violations` 0 and `deadlock` false
either way. It is deliberately **not** in `--gate`: no live Ornith soak has been run
against this tree, so there is no measured floor to set — only this tree's own numbers,
which is exactly the mistake the floors above are documented as avoiding.

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
