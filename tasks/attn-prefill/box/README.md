# triton vs flashinfer extend on the NVFP4 models (box ft-dev, RTX 5080 gen4)

These are box numbers. The tree is FT-exl3p4, which holds the P4 tree (its python is exp/ornith-exl3 c355278: the six flashinfer-extend commits on top of P2). The only thing that changes between arms is `FREETOKEN_EXTEND_BACKEND=triton|flashinfer`.

- **`job-ab.sh`** runs each model the way production runs it. It serves the whole model in RAM through `scripts/serve-default.sh` at ratio 1.00 on port 1920, with nothing else on the GPU. It does an 8K warm-up, then two probe passes at 32K, 80K and 256K. Pass 2 is the pass of record.
- **`job-logits.sh`** produces teacher-forced logits at ≥ 80K. The prompt is 83K tokens: filler x650, in 8192-token chunks, at ratio 0.85 with 131072-token caps. The triton run decodes 32 greedy tokens. flashinfer is then teacher-forced on those ids. A second triton tile (M64 N64 w8 s1) is also forced on them; it is the same math reordered, so it serves as the kernel-noise yardstick.

## Prefill / decode (pass 2, tok/s)

| model | prompt | prefill triton | prefill flashinfer | change | decode triton / flashinfer |
|---|---:|---:|---:|---:|---:|
| Nemotron 3.5 Lightning NVFP4 | 32K | 9009 | 9659 | +7% | 155.0 / 156.9 |
| | 80K | 7121 | 8143 | +14% | 154.5 / 153.1 |
| | 256K | 3805 | 4982 | +31% | 136.2 / 136.0 |
| Ornith 1.5 NVFP4 | 32K | 8382 | 9846 | +17% | 140.2 / 143.0 |
| | 80K | 5193 | 7421 | +43% | 131.1 / 131.4 |
| | 256K | 2024 | 3760 | +86% | 98.4 / 98.0 |

- Every arm had captures=1, no tracebacks and no OOM.
- The flashinfer arms logged `extend attention: flashinfer`, and the triton arms did not.
- Decode does not change: decode attention does not use the extend path.
- Pass 1 shows the same picture (`results/attnab-*/probe.jsonl`).

## Logits at ≥ 80K (teacher-forced, 32 positions; reference = triton)

| model | vs triton | top-1 | top-5 | KL mean / max | max abs dlogit |
|---|---|---:|---:|---:|---:|
| Nemotron (83227-token prompt) | flashinfer | 32/32 | 0.962 | 1.32e-3 / 1.57e-2 | 3.45 |
| | other triton tile (noise) | 32/32 | 0.975 | 7.53e-4 / 1.03e-2 | 2.16 |
| Ornith NVFP4 (83221-token prompt) | flashinfer | 32/32 | 0.931 | 9.95e-3 / 9.86e-2 | 4.01 |
| | other triton tile (noise) | 31/32 | 0.925 | 6.68e-3 / 5.86e-2 | 3.06 |

- **Nemotron's attention layers do not regress in speed:**
  - Prefill is 7-31% faster.
  - Decode is equal.
  - Top-1 matches at every position.
- **Numerically, flashinfer is close to the noise yardstick but a little above it:**
  - Its KL mean against triton is 1.5x (Ornith) to 1.75x (Nemotron) that of the second triton tile.
  - There is only one noise sample per model, so this band is thin.
- On EXL3 Ornith (`tasks/ornith-exl3/perf/p4-extend`), the same check gave 6.39e-3 against 6.21e-3 for the noise tile.

The `.pt` logit dumps (17-46 MB each) are not committed. The owner's machine holds them, md5-verified against the box, in `~/ai/box-archive/ft-dev-attnab/`.

## Greedy generation (the output gate): `job-greedy.sh`, `greedy_compare.py`

Setup:
- Every arm is served the production way.
- The prompt is `greedy_client.py`'s haystack: numbered records, then "quote records N/7, N/2 and N-3 exactly, then write a long story that visits every city in them, in order". SIZE 83000 comes out at 107113 tokens on Nemotron and 114112 on Ornith.
- 1024 tokens are generated at temperature 0, with thinking off.
- Each arm ran twice. The two runs were token-identical in every arm, so any cross-arm difference is the kernel's.
- The reference is triton.
- "First divergence" is the first differing generated token.

| model | arm | first divergence | quoted facts right | the story |
|---|---|---:|---|---|
| Nemotron (stops at 75 tokens) | flashinfer | none (identical) | 1 of 3 | — |
| | other triton tile | 36 (a digit of a day) | 1 of 3 | — |
| Ornith NVFP4 | flashinfer | 5 ("records:" vs "records you requested:") | 3 of 3 | same title, Tromsø -> Cusco -> Oaxaca (right) |
| | other triton tile | 83 | 3 of 3 | adds Porto, which is not a quoted city |

Nemotron gets two of the three day numbers wrong in every arm, identically. That is the model, not the kernel. `results/attngreedy-*/compare.txt` holds the full texts.

### Four more prompts (`TAG=multi`, SIZE 62000/70000/78000/90000; 80K-124K tokens; 1 repeat each)

First divergent generated token against triton, one entry per prompt, with the 107K/114K prompt above listed first:

| model | flashinfer | other triton tile (noise) |
|---|---|---|
| Nemotron | none, 63, 12, 173, 85 (median 63) | 36, 63, 42, 11, 0 (median 36) |
| Ornith NVFP4 | 5, 100, 101, 0, 7 (median 7) | 83, 100, 20, 0, 7 (median 20) |

Quoted facts right, out of 15 (5 prompts x 3 records):

| model | triton | flashinfer | noise tile |
|---|---:|---:|---:|
| Ornith | 13 | 14 | 13 |
| Nemotron | 1 | 1 | 1 |

- Nemotron misquotes the records in every arm. It invents the item and day around the right record number, the same way in every arm, so the facts check does not discriminate there.
- flashinfer's divergence is no earlier than the noise tile's:
  - On Nemotron it is later.
  - On Ornith, three of the five prompts diverge at the same token in both arms, and the medians are 7 against 20.
- In substance, flashinfer is equal or better: on Ornith it gets one more fact right than triton or the tile.
- `results/attngreedy-multi-*/compare.txt` holds the full texts.
