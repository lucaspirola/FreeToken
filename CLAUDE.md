# FreeToken — operating notes for agents on the owner's machines

Read this before touching the GPU or the server. It is the same on every host (RTX 5080
WSL2 box, the Ada box); per-host differences live in `$HOME/.config/freetoken/serve.env`.

## The server and its ONE default configuration

* Model served: NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 on `http://127.0.0.1:1919`
  (`nemotron-3.5-lightning`, aliases `-judge`, `-collect`).
* The configuration is `scripts/serve-default.sh`. **Never hand-type `ft serve` flags** —
  edit that file if the profile must change, and keep its comments truthful. Measured
  numbers behind it: `benchmarks/results/nemotron35_lightning_5080_single_lane_2026-09-17.md`,
  design notes in `docs/nemotron.md` ("Default profile").
* What it is: single lane (one session decoding on the GPU, every other session checkpointed
  to RAM/disk and swapped back), growable KV up to 1M tokens, expert VMM arena + overlap
  scheduling (no CUDA-graph recaptures), **whole model in RAM** (`FREETOKEN_PIN_BUDGET_GB`
  ≥ expert banks, all banks mlock'd), q8_0 KV, prefill chunk 8192.
* It runs as the **system** unit `freetoken-serve` (template
  `scripts/systemd/freetoken-serve.service.in`, installed by `sudo scripts/systemd/install.sh`).
  A system unit because only PID 1 grants `LimitMEMLOCK=infinity`; the user manager's cap
  leaves the banks pageable.

## Bring FreeToken up / down

```
ft-up      # = sudo systemctl start freetoken-serve + wait for readiness; starts are never
           #   rate-limited (no reset-failed ritual)
ft-down    # = sudo systemctl stop freetoken-serve, nothing more
```
Both are symlinks in `~/.local/bin` to `scripts/ft-up` / `scripts/ft-down` (installed by
`install.sh`). **Neither touches anything else on the GPU**: `ft-up` leaves whatever is
running there running (the memory ratio is a fraction of the FREE VRAM, so the server sizes
itself to what is left) and `ft-down` brings nothing back up. Stopping another GPU service
(on the RTX 5080 box the piro-board embedder holds 3–10 GB) is the owner's decision, taken
explicitly and separately; never do it as part of "bring FreeToken up". `nvidia-smi` shows
who holds VRAM. Readiness, if you watch the log yourself: `API server is ready` AFTER
the last `ServerArgs(model_path` line in `~/.cache/freetoken/logs/ft_serve.log` (the log
appends across starts; `/v1/stats` answers with nulls while loading). Startup takes 1–3 min
(serial expert-bank build when free RAM is low); the first requests after a start are slow
(prefill < 1K tok/s) until the bank build finishes, ~3 min.
Do not start the server from an agent shell: the harness can kill shells during the load
and a server started there dies with them.

Before starting, `free -g` must show MemAvailable ≥ expert banks + ~4 GiB (≈ 20 GiB for
this model); with `--host-ram-reserve-gb 0` (the owner's choice) the server keeps only
~6–7 GiB of headroom on a 28 GiB host, and it dies first in a host OOM (OOMScoreAdjust=1000).
Never run torch-backed pytest beside the live model; stop the server first
(`tests/scheduler` etc. need ~1 GiB, the OOM sweep of 2026-09-06 killed a server this way).

## New machine (e.g. the Ada box)

1. `git clone` the fork and `uv sync`; put the model under `~/ai/models/` (or set
   `FREETOKEN_MODEL` in `~/.config/freetoken/serve.env`).
2. `sudo scripts/systemd/install.sh` — renders the unit for this user/repo path and makes
   **memlock unlimited for this user, now and after every reboot**: `user@UID` drop-in,
   `system.conf.d`/`user.conf.d` `DefaultLimitMEMLOCK=infinity`, `limits.d` for shells/ssh,
   plus `prlimit` on the running user manager so the current session already pins. Add
   `--enable` only if the server should take the GPU at boot. On WSL it also reminds you
   that the VM's RAM cap (`[wsl2] memory=` in `%USERPROFILE%\.wslconfig`) must hold the
   banks + ~4 GiB, and that `/etc/wsl.conf` keeps `[boot] systemd=true`.
   Verify: `sudo systemctl show -p LimitMEMLOCK freetoken-serve` → `infinity`, and after a
   start the log must not contain "settled pageable" (`acceptance.sh R6`).
3. Check `scripts/serve-default.sh` knobs for the host: `FREETOKEN_PIN_BUDGET_GB` must stay
   ≥ 15.41 GiB (banks) so the whole model is in RAM — lower it only if the host cannot spare
   the RAM, accepting CPU-decode layers. CUDA arch is auto-detected
   (`TVM_FFI_CUDA_ARCH_LIST`, 8.9 on Ada, 12.0 on Blackwell).
4. Start as above, then verify with `benchmarks/switchyard_soak/checks/acceptance.sh R3`
   (one graph capture, KV growth, no tracebacks) after a couple of long requests, and
   `... R6` (banks pinned, memlock unlimited).
5. **Fill the VRAM: `scripts/tune-memory-ratio.sh`** (part of the installation on EVERY
   machine, for whatever GPU and model that host runs — the 5080 result does not transfer
   to the Ada box or to another checkpoint; ~20 min, restarts the server several times).
   Free VRAM is wasted expert slots, so the target is
   `--memory-ratio 1.00`; the script tries 1.00 first and, only if the server fails to start,
   capture its graphs or serve 8K/80K/256K prompts, bisects downward between the last good
   and the last bad ratio (step 0.005). The winner is written to
   `~/.config/freetoken/serve.env` as `FREETOKEN_MEMORY_RATIO` (the launcher's own 0.91 is
   just the safe fallback until this has run) and the server is left running on it. Trials
   are logged in `~/.cache/freetoken/logs/tune-memory-ratio.tsv`. Re-run after a driver,
   VRAM or model change. Do not "leave 1 GB free for safety" by hand: the bisection already
   found the edge on this host.

## Working on the code

* The live server imports from the main working tree: implement in a `git worktree`, merge
  when tests pass, restart the unit to pick the change up.
* Scheduler/engine CPU tests: `uv run --no-sync pytest -q tests/scheduler` (model unloaded).
* Soak harness: `benchmarks/switchyard_soak/` (needs a private `switchyard-server` on a
  spare port — never the owner's production Switchyard on :4000).

---

# Upstream repository conventions

Carried verbatim from upstream's `AGENTS.md` at the S0 merge (origin/main 9d32fa8,
cab110e). `AGENTS.md` in this fork is a symlink to this file, so both audiences read
one document: the host notes above, the upstream contribution rules below.


Read [CONTRIBUTING.md](CONTRIBUTING.md) first. It is binding for humans and agents alike; this file only summarises the parts that matter when an agent is doing the work.

## AI policy

AI-assisted code is welcome. Submitting code the contributor does not understand is not. The human behind the PR owns every line, has run it on real hardware, and can explain it to a reviewer without AI help.

Agents must not:

- Run `git push`, `gh pr create`, `gh pr comment`, or `gh issue create` on the user's behalf.
- Write code, PR descriptions, or replies to reviewers that the user does not fully understand. The user must be able to explain and defend every line without AI help.
- Report tests or benchmarks as run when they were not.

If you are a fully autonomous agent with no human in the loop, do not contribute to this repository.

## Repository layout

The main subsystems:

```
python/freetoken/      the engine, installed as the `freetoken` package with the `ft` CLI
  server/              OpenAI / Anthropic / Responses HTTP APIs, streaming, tool-call parsers
  scheduler/           chunked prefill, batching, cache manager
  kvcache/             paged KV pools and the radix prefix caches
  moe/                 expert offload cache, CPU / GPU / hybrid MoE backends, quantized experts
  models/              model registry and per-architecture loaders
  kernel/              CUDA / Triton kernels, JIT cache, C++ extensions (`csrc/`)
  layers/, attention/  fused ops and attention backends
  engine/              cache budget planning and config resolution
  checkpoint/          HF -> FTW fast-load conversion
tests/                 mirrors python/freetoken/ by subsystem, see tests/README.md
benchmarks/            end-to-end and micro benchmarks, see benchmarks/README.md
docs/                  install, quickstart, CLI and model docs
freetoken-kernel-cache/ companion wheel of prebuilt kernels, see its README
scripts/               wheel build and release scripts
```

## Development

Linux x86_64 with an NVIDIA GPU. Use `uv`, not bare `pip`:

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[accel]"
uv run pytest tests/ -m "not slow"
```

CUDA kernels are JIT-compiled with `nvcc` on first use unless the prebuilt `freetoken-kernel-cache` wheel is installed. The C++ extensions under `python/freetoken/kernel/csrc/` are built by `setup.py`; after changing them run `python setup.py build_ext --inplace`.

Put a new test in the `tests/` directory that mirrors the module it protects, and extend an existing file before creating a new one. Bug fixes come with a test that fails before and passes after. Performance changes come with A/B numbers against `main`.

## Issues and PRs

- Search existing issues and PRs before starting. Items on the [Roadmap](https://github.com/FlashML-org/FreeToken/issues/79) are discussed with maintainers before implementation; features not on it start as an issue.
- When helping the user draft an issue, follow the matching template in `.github/ISSUE_TEMPLATE/` (engine bug, model checkpoint, feature request) and fill in every required field: hardware, driver, FreeToken version, checkpoint ID, exact command, and the full log.
- One change per PR, linked to its issue, with the hardware, checkpoint ID and exact command it was tested with.

## Code comments

Comments explain a non-obvious "why", never restate the code. Write the code first, then add a comment only where a reader would otherwise be confused. Keep them to one or two lines. Configuration files get no comments. Use ASCII: `-` not em-dash, `->` not arrows.

## Commits

[Conventional Commits](https://www.conventionalcommits.org/), one line, imperative, lowercase, no trailing period:

```
fix(kvcache): size the SWA radix pool for chunked prefill
```

PRs are squash-merged, so the PR title follows the same format. The subject line is usually enough; add a body only when the change needs a why that the diff does not show, and keep it to a few lines. Only commit when the user asks. If the user wants attribution, use `Assisted-by: <agent name>`, not `Co-authored-by`.
