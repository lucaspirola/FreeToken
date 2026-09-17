# FreeToken — operating notes for agents on the owner's machines

Read this before touching the GPU or the server. It is the same on every host (RTX 5080
WSL2 box, the Ada box); per-host differences live in `$HOME/.config/freetoken/serve.env`.

## The server and its ONE default configuration

* Default model: `~/ai/models/Ornith-1.5-35B-Q6_K.gguf` on `http://127.0.0.1:8080`
  (`ornith1.5-35b`, aliases `-judge`, `-collect`), for the RTX 2000 Ada 16 GB.
* The configuration is `scripts/serve-default.sh`. **Never hand-type `ft serve` flags** —
  edit that file if the profile must change, and keep its comments truthful.
  Model-specific context guidance: `docs/models.md`. No host env file is needed for
  the Ornith Ada defaults; other models require explicit host overrides.
* What it is: single lane (one session decoding on the GPU, every other session checkpointed
  to RAM/disk and swapped back), growable KV up to 524288 tokens (YaRN x2 over 262144),
  expert VMM arena + overlap scheduling (no CUDA-graph recaptures), all ~24.61 GiB of
  expert banks CUDA-pinned in RAM (32 GiB pin budget), q8_0 KV, prefill chunk 4096.
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

Before starting, allow the ~24.61 GiB expert banks, process overhead and a 4 GiB host
reserve in MemAvailable. This profile was tested with `[wsl2] memory=64GB`; swap does
not substitute for resident expert-bank RAM. The unit uses OOMScoreAdjust=1000.
Never run torch-backed pytest beside the live model; stop the server first
(`tests/scheduler` etc. need ~1 GiB, the OOM sweep of 2026-09-06 killed a server this way).

## New machine (e.g. the Ada box)

1. `git clone` the fork and `uv sync`; put the model under `~/ai/models/` (or set
   `FREETOKEN_MODEL` in `~/.config/freetoken/serve.env`).
   Ornith Q6_K / Ada is the fallback without `serve.env`. The optional
   `scripts/serve-env.examples/ornith-ada.env` records the same settings; copy it only
   when you want host overrides. The installer does not download the checkpoint.
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
   ≥ 24.61 GiB (Ornith Q6_K banks; default budget 32) so the whole model is in RAM — lower it only if the host cannot spare
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
   `~/.config/freetoken/serve.env` as `FREETOKEN_MEMORY_RATIO`. The launcher defaults
   to 1.00, but each new GPU/driver/model must pass tuning; this machine's result is not a guarantee.
   Trials are logged in `~/.cache/freetoken/logs/tune-memory-ratio.tsv`. Re-run after a driver,
   VRAM or model change. Do not "leave 1 GB free for safety" by hand: the bisection already
   found the edge on this host.

## Working on the code

* The live server imports from the main working tree: implement in a `git worktree`, merge
  when tests pass, restart the unit to pick the change up.
* Scheduler/engine CPU tests: `uv run --no-sync pytest -q tests/scheduler` (model unloaded).
* Soak harness: `benchmarks/switchyard_soak/` (needs a private `switchyard-server` on a
  spare port — never the owner's production Switchyard on :4000).
