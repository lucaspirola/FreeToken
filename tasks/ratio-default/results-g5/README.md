# exp/ratio-default on ft-g5: verify at 1.00, prediction vs measurement

Box: Vast 52354198 (RTX 5080, Core Ultra 9 285K, 62 GB, PCIe gen 5, driver 580, native
Linux). Code: exp/ratio-default 29db773 (python/ scripts/ tests/) applied on 9dac3b5 as
2fe2fff. The fixed-size reruns used the verify script of 52d8e59. Run 2026-09-24,
06:57Z to 07:18Z.
Launcher: `VERIFY_LAUNCHER=nohup scripts/verify-memory-ratio.sh` (port 1920), sizes 8K / 80K /
256K, driven by `ratio-evidence.sh`. The fixed-size reruns came from the first half of
`g5-next.sh`.

| run | KV | result | decode 8K/80K/256K tok/s | runtime reserve | free after capture |
|---|---|---|---|---|---|
| nemotron-growable | serve-default profile, grows to 1M | PASS, 1 capture | 148.7 / 165.9 / 141.7 | 0 (arena fills around the measured transient) | 10.61 GiB before arena fill, 0.77 GiB after |
| nemotron-fixed | 262144 tokens, no growth | PASS, 1 capture | 146.3 / 167.0 / 140.4 | 0.83 GiB | 0.75 GiB |
| ornith-growable | grows to 256K | PASS, 1 capture | 130.4 / 127.7 / 97.7 | 0 | 1.12 GiB |
| ornith-fixed | 262144 tokens, no growth | PASS, 1 capture | 129.0 / 128.6 / 99.1 | 1.03 GiB | 0.99 GiB |

## Prediction vs measurement ("Memory prediction check" lines, predicted vs measured)

| term | nemotron growable | nemotron fixed | ornith growable | ornith fixed |
|---|---|---|---|---|
| prefill transient, 8192 chunk | 0.68 / 0.65 (+5%) | 0.68 / 0.59 (+15%) | 0.88 / 0.98 (-11%) | 0.88 / 0.97 (-10%) |
| non-expert weights | 2.20 / 2.26 (-2%) | 2.20 / 2.26 (-2%) | 2.52 / 2.84 (-11%) | 2.52 / 2.84 (-11%) |
| linear-state pool | 0.59 / 0.59 (0%) | 0.59 / 0.59 (0%) | 0.78 / 0.78 (0%) | 0.78 / 0.78 (0%) |
| CUDA graph pool (bs 1) | 0.03 / 0.04 | 0.03 / 0.05 | 0.03 / 0.00 | 0.03 / 0.00 |

0 WARN lines on all four runs. Every term is inside the 25% band except the graph pool,
where the relative error is large (-41%, +703%) but the absolute error is tens of MiB.
The 64 MiB floor on the WARN exists for that case.
The fixed-size transient on Nemotron measures 0.59 GiB, not 0.65 GiB: the growable start
measures after the arena is formatted. The prediction sits between the two, and it covers
both on a fixed start, where the reserve is taken out of the budget.

## The first fixed-size attempt (files `*-grow0-rejected*`)

The first nemotron-fixed and ornith-fixed runs passed `--kv-grow-step-tokens 0` through
FREETOKEN_EXTRA_ARGS. `ft serve` rejects 0 (`must be >= 1`), so the server exited at argument
parsing. The verify then waited out its 300 s start timeout and reported FAIL-start with no
note. This is a harness problem, not an engine one. 52d8e59 fixes the verify script:
* a dead start ends the wait;
* the note matches `error:`;
* `VERIFY_SERVE` runs a copy of serve-default.sh without `--kv-grow-step-tokens 65536`
  (`scripts/serve-fixed.sh` on the box, made by sed in `g5-next.sh`).
The reruns above used that copy.
