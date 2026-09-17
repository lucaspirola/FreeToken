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

## "Clean the GPU and bring FreeToken up"

```
nvidia-smi                                   # who holds VRAM? stop THEIR service, don't kill blindly
systemctl --user stop piro-board-embedder     # the unit also does this itself (ExecStartPre)
sudo systemctl reset-failed freetoken-serve   # always, a SIGTERM stop leaves it 'failed'
sudo systemctl start freetoken-serve
```
Readiness: wait for `API server is ready` AFTER the last `ServerArgs(model_path` line in
`~/.cache/freetoken/logs/ft_serve.log` (the log appends across starts; `/v1/stats` answers
with nulls while loading). Startup takes 1–3 min (serial expert-bank build when free RAM is
low). Stop with `sudo systemctl stop freetoken-serve` (the embedder is restored on stop).
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
