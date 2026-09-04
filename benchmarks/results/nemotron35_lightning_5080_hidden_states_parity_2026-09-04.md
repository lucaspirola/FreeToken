# Phase 3H hidden-state export — GPU parity check (PASS)

2026-09-04, RTX 5080 (16 GiB) + 34 GiB WSL host, repo at `c4486b6` (plus the probe fix
below). Checkpoint `~/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`
(modelopt `MIXED_PRECISION`: FP8 mamba `in_proj`/`out_proj`, W4A16 NVFP4 experts, 52
blocks, hidden 2688).

**Verdict: PASS.** Every one of the 52 exported layers matches transformers' own
`NemotronHBlock` stack at cosine **> 0.9988** on the mean-pooled residual Switchyard
consumes (gate: > 0.99). Worst layer 32 at 0.998840, median 0.999760, best 0.999996.

## What had to change before the check could run at all

`benchmarks/probe_hidden_states_parity.py` as merged in `1f2de67` could not have worked
on this checkpoint on any host:

- Its reference was `AutoModelForCausalLM.from_pretrained(model, dtype=bf16,
  device_map="cpu")`. The release is a **modelopt** checkpoint and transformers 5.15.1 has
  no `modelopt` quantizer (`AUTO_QUANTIZER_MAPPING` has none); its tensor names are
  `backbone.*` against HF's `model.*`; its 128 experts are per-expert 2-D NVFP4 tensors
  against HF's fused 3-D bf16 parameter; `lm_head.weight` is `[131072, 1344] U8`. A
  meta-device skeleton diffs **400 missing / 18 486 unexpected** keys against the index.
- A dense bf16 `NemotronHForCausalLM` is **31.58 B params = 58.8 GiB**, against 34 GiB of
  WSL RAM — before the server's ~20 GiB of pinned expert banks.

Fix, in the same script (the only file changed):

- `reference_hidden_states` now builds the model on `meta` and **streams one block at a
  time**: a forward pre-hook materializes that block's weights from the shards
  (`assign=True`), dequantizing per the sibling scales the checkpoint carries
  (`weight_scale_2` ⇒ NVFP4 via FreeToken's own dequant kernel, `weight_scale` ⇒ FP8,
  else passthrough), and the forward hook records the block's output and puts the
  parameters back on `meta`. transformers still owns the forward — masks, position ids,
  the pure-torch Mamba-2 scan, the residual adds — and the hook records exactly what
  FreeToken exports (`residual + mixer`, before the next input norm and before `norm_f`),
  so the comparison no longer depends on HF's `output_hidden_states` indexing at all.
  Peak ~3.5 GiB VRAM, ~10-22 s for the whole 52-block forward.
- `--capture-only` / `--artifact <path>` split the run in two phases, because the served
  model and the reference cannot be resident at once.
- `--reference-dt-min` (default `0.0`) sets the reference scan's `dt` floor. See below.

## Commands

```bash
systemctl --user stop piro-board-embedder.service   # already inactive this session
mkdir -p /tmp/ft-hidden-states

# Phase A — server up, capture one artifact, stop the server; all inside ONE lock hold
scripts/gpu_lock.sh /tmp/claude-1000/-home-lucas-ai-FreeToken/f4e2e9e3-f4f5-40d0-9980-b3b09d1ef47d/scratchpad/hs/run_capture.sh
#   .venv/bin/python -m freetoken.cli serve --model <ckpt> --port 1919 \
#     --max-running-requests 1 --moe-backend offload --moe-pageable-gpu --moe-cache-auto \
#     --num-tokens 65536 --memory-ratio 0.85 --max-prefill-length 8192 \
#     --host-ram-reserve-gb 6 --hidden-states-dir /tmp/ft-hidden-states
#   .venv/bin/python benchmarks/probe_hidden_states_parity.py --model <ckpt> \
#     --base-url http://127.0.0.1:1919 --hidden-states-dir /tmp/ft-hidden-states \
#     --prompt-tokens 300 --capture-only

# Phase B — server stopped, reference streamed on the (now idle) GPU
scripts/gpu_lock.sh /tmp/claude-1000/-home-lucas-ai-FreeToken/f4e2e9e3-f4f5-40d0-9980-b3b09d1ef47d/scratchpad/hs/run_reference.sh
#   .venv/bin/python benchmarks/probe_hidden_states_parity.py --model <ckpt> \
#     --hidden-states-dir /tmp/ft-hidden-states \
#     --artifact /tmp/ft-hidden-states/227ffb0a24874f2b9b968277e472ea05.safetensors \
#     --keep-artifact --reference-dt-min 0.0
```

Both phases ran under `scripts/gpu_lock.sh`, one at a time; the driver script `exec`s to
its own log because the lock's exit trap kills the invoking pipeline.

## Timings

| step | wall |
|---|---|
| P1 server launch → `/health` `{"status":"ok"}` | 42 s (16:22:30 → 16:23:12) |
| probe request end to end (incl. tokenizer load, 316-token prefill, 88.3 MB artifact) | 45.2 s |
| streamed HF reference, 52 blocks, 316 tokens, RTX 5080 | 22.2 s cold / 10.3 s warm |
| scoring (mean-pool + 52 cosines) | < 1 s |

## Artifact

`/tmp/ft-hidden-states/227ffb0a24874f2b9b968277e472ea05.safetensors`, 88 341 128 bytes:
`hidden_states [316, 52, 2688] BF16`, `token_ids [316] I64` — the contract in
`docs/switchyard.md` §6 exactly. The response carried the path in
`kv_transfer_params.hidden_states_path`. The server logged
`Prefill batch, #new-seq: 1, #new-token: 316, #cached-token: 0`, i.e. the probe's
`no_prefix_cache` bypass held (316 = 300 filler tokens + chat template).

## Per-layer cosine

`cosine(mean_t reference[t, i], mean_t artifact[t, i])`, all 316 prompt tokens. The
"dt-floor control" column is the same comparison with the reference left as transformers
ships it (`time_step_limit = (config.time_step_min, inf)` = 1e-3).

| layer | cosine | dt-floor control | layer | cosine | dt-floor control |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.999996 | 0.973700 | 26 | 0.999582 | 0.992014 |
| 1 | 0.999991 | 0.954911 | 27 | 0.999673 | 0.994278 |
| 2 | 0.999984 | 0.946224 | 28 | 0.999603 | 0.993393 |
| 3 | 0.999950 | 0.940622 | 29 | 0.999677 | 0.994992 |
| 4 | 0.999929 | 0.952212 | 30 | 0.999686 | 0.993618 |
| 5 | 0.999899 | 0.965677 | 31 | 0.999643 | 0.994140 |
| 6 | 0.999893 | 0.971796 | 32 | 0.998840 | 0.995610 |
| 7 | 0.999901 | 0.975657 | 33 | 0.999557 | 0.995161 |
| 8 | 0.999868 | 0.979398 | 34 | 0.999548 | 0.994989 |
| 9 | 0.999903 | 0.985486 | 35 | 0.999589 | 0.995572 |
| 10 | 0.999901 | 0.987384 | 36 | 0.999620 | 0.995791 |
| 11 | 0.999878 | 0.990429 | 37 | 0.999628 | 0.995952 |
| 12 | 0.999834 | 0.990055 | 38 | 0.999618 | 0.995927 |
| 13 | 0.999785 | 0.989615 | 39 | 0.999691 | 0.996713 |
| 14 | 0.999847 | 0.992715 | 40 | 0.999752 | 0.997394 |
| 15 | 0.999830 | 0.992346 | 41 | 0.999759 | 0.997667 |
| 16 | 0.999761 | 0.992199 | 42 | 0.999775 | 0.997841 |
| 17 | 0.999705 | 0.993167 | 43 | 0.999708 | 0.997120 |
| 18 | 0.999727 | 0.994380 | 44 | 0.999771 | 0.997683 |
| 19 | 0.999653 | 0.992670 | 45 | 0.999766 | 0.997662 |
| 20 | 0.999651 | 0.992881 | 46 | 0.999790 | 0.997925 |
| 21 | 0.999594 | 0.992079 | 47 | 0.999783 | 0.997869 |
| 22 | 0.999669 | 0.993618 | 48 | 0.999844 | 0.998500 |
| 23 | 0.999630 | 0.992952 | 49 | 0.999833 | 0.998333 |
| 24 | 0.999730 | 0.994283 | 50 | 0.999895 | 0.999012 |
| 25 | 0.999695 | 0.993260 | 51 | 0.999857 | 0.998847 |

| run | min | median | max | layers below 0.99 |
|---|---:|---:|---:|---:|
| `--reference-dt-min 0.0` (matches the engine) | **0.998840** (L32) | 0.999760 | 0.999996 | **0** |
| `--reference-dt-min 1e-3` (transformers as shipped) | 0.940622 (L3) | 0.993618 | 0.999012 | 12 (0–10, 13) |

## The dt floor is the whole gap — and an independent confirmation of item 1

The first reference run failed 12 layers, and the failures were the **shallow** ones
(worst at layer 3, monotonically improving to 0.9990 at layer 51) — the signature of a
fixed absolute perturbation injected in the first blocks and then diluted as the residual
norm grows with depth, not of accumulating quantization error.

The cause is `modeling_nemotron_h.py:381`, `self.time_step_limit = (config.time_step_min,
float("inf"))` — the same 1e-3 `dt` clamp that item 1 of the handover identified as
FreeToken's own 262K recall bug (`time_step_min` is HF's *initializer* range for
`dt_bias`, not a runtime bound; vLLM passes `(0.0, inf)` and llama.cpp does not clamp).
FreeToken no longer clamps, so the shipped reference was the one introducing the error.
Setting the reference's floor to 0.0 moves the worst layer from **0.9406 → 0.9988** and
every layer above the gate. Nothing else changed between the two runs — same artifact,
same weights, same forward.

That is a second, independent confirmation of item 1's root cause: an HF reference that
keeps the floor disagrees with the fixed engine by 6 % of the residual at layer 3, and
one that drops it agrees to 1e-3.

## What the check does and does not prove

Proved: layer indexing (row `i` is block `i`'s output, not the embedding and not an
off-by-one), no `norm_f` leak, no dropped prefill chunk, correct token ids, BF16 dtype,
`[tokens, layers, hidden]` order, prefix-cache bypass, and that the FP8/NVFP4 serving path
tracks a dequantized bf16 reference to ≥ 0.9988 per layer on a 316-token prompt.

Not covered: prompts long enough to be chunked (the probe cap is
`--hidden-states-max-tokens 4096`, a single 8192-token chunk here), `layer_ids` subsets,
concurrent probes, and the router's own scoring head.

## Reproducing

The two driver scripts are in the session scratchpad
(`.../scratchpad/hs/{run_capture.sh,run_reference.sh}`); everything they call is in the
repo. The artifact was kept for the run and removed afterwards (Switchyard's reader
normally consumes it; FreeToken never cleans the directory).
