# Nemotron 3.5 Lightning NVFP4 — 1M multi-needle recall, ONE prefill

Host: RTX 5080 16 GiB, WSL2 34 GiB RAM. Tree `508ea32` (= `81ab30e` plus a CI workflow, a
replay harness and a `TYPE_CHECKING` import — **no engine change**; `git diff 81ab30e..508ea32`
touches `python/freetoken/` only in `kvcache/cache_status.py`, annotations only). Model
`/home/lucas/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`. 2026-09-04 21:55–22:29
local, under `scripts/gpu_lock.sh`, `piro-board-embedder.service` inactive. GPU 0 MiB at the
end, no leftover venv process, 0 tracebacks, 0 warnings in the server log.

**One 1,039,994-token prefill; eight graded questions.** Six needles at depths
0.05 / 0.25 / 0.50 / 0.60 / 0.75 / 0.95, one control for a key that is not in the text, and one
combined question that needs two needles at once. Turns 2–8 ran on the same chat prefix and hit
the prefix cache for **99.9954 %+** of their prompt, so the whole experiment cost **34 minutes
of GPU** instead of the ~4 hours eight separate 1M prompts would have.

| # | question | depth | expected | verdict | answer |
|---|---|---|---|---|---|
| 1 | orchard ledger code | 0.0501 | 5663623 | **PASS** | `5663623 The <name> ledger code is 5663623.` |
| 2 | harbour ledger code | 0.2500 | 4190877 | **FAIL** | `The orchard ledger code is 5663623.` |
| 3 | quarry ledger code | 0.5000 | 8324516 | **FAIL** | `The orchard ledger code is 5663623.` |
| 4 | cavern ledger code | 0.6000 | 6082735 | **FAIL** | 128 tokens of haystack filler, no answer |
| 5 | meadow ledger code | 0.7500 | 7218459 | **PASS** | `The meadow ledger code is 7218459.` |
| 6 | thicket ledger code | 0.9500 | 3947162 | **PASS** | `The thicket ledger code is 3947162.` |
| 7 | **control**: belfry ledger code | — | absent | **PASS** | `No belfry ledger code found.` |
| 8 | **combined**: larger of orchard/harbour + their sum | 0.05 & 0.25 | orchard, 9854500 | **PASS** | `The orchard ledger code is larger: 5663623. The sum of the two codes is 9854500.` |

**5/8.** The control did not fabricate (no 7-digit string anywhere in its answer) and did not
substitute another needle's code — the two failure modes a needle gate cannot see.

## The result that matters: question 8 contradicts question 2

Turn 2 asked for the harbour code directly and got the *orchard* code. Turn 8 asked which of
the orchard and harbour codes is larger and for their sum, and answered
**9,854,500 = 5,663,623 + 4,190,877** — exactly right, and unobtainable without the harbour
needle at depth 0.25. Nothing earlier in the conversation contains 4,190,877: turn 2's own
answer was the orchard code.

So the depth-0.25 needle **is** in the recurrent state and **is** reachable at 1.04M tokens;
what failed at turn 2 was answering, not retention. A single-needle gate would have recorded
"1M recall fails at depth 0.25" and closed the case on the model or the kernels. It is the same
class of mistake the 262K bisect made before the `dt`-floor fix, in the opposite direction.

Practical consequence: **grade long-context recall with more than one question shape per
needle.** The cheap version costs one extra cached turn (~5 s here) per needle.

## Depth profile at 1.04M

```
depth  0.05   0.25   0.50   0.60   0.75   0.95
       PASS   FAIL*  FAIL   FAIL   PASS   PASS      (* recovered by the combined question)
```

Both ends of the haystack are recalled; the 0.50–0.60 band is not. Turn 4 (cavern, 0.60) did
not answer wrongly — it degenerated, emitting 128 tokens of `The warden notes that the ... seal
remains inactive.` filler, i.e. it continued the haystack instead of answering the question.
That is a coherence break, not a retrieval miss, and it is the only turn in the run that spent
its whole decode budget.

Caveats, stated so the table is not over-read: n = 1 per depth, one phrasing per needle, greedy
decoding, and all six needles share one prompt (a six-needle haystack is a harder retrieval
problem than a one-needle haystack — the model must bind code to key, not merely find digits).
`benchmarks/bench_long_context.py`'s single-needle gate at 262K/524K depth 0.50 passed after
the `dt` fix; this run says the same prompt shape at 1.04M does not.

## Timing and the prefix cache

| turn | prompt tokens | cached | fresh | TTFT s | decode tok/s | completion |
|---|---|---|---|---|---|---|
| 1 | 1,039,994 | **0** | 1,039,994 | **1,815.20** | 19.19 | 25 |
| 2 | 1,040,054 | 1,040,018 | 36 | 7.03 | 20.20 | 17 |
| 3 | 1,040,106 | 1,040,070 | 36 | 4.87 | 20.20 | 17 |
| 4 | 1,040,158 | 1,040,122 | 36 | 4.95 | 19.49 | 128 |
| 5 | 1,040,322 | 1,040,285 | 37 | 4.94 | 19.49 | 16 |
| 6 | 1,040,374 | 1,040,337 | 37 | 4.99 | 20.29 | 17 |
| 7 | 1,040,435 | 1,040,390 | 45 | 5.04 | 19.49 | 10 |
| 8 | 1,040,493 | 1,040,444 | 49 | 4.70 | 19.69 | 35 |

* Turn 1: **1.04M tokens in 1,815 s** = **573 tok/s** whole-prompt (the marginal rate decays
  from 1,850 tok/s on the first 8 K chunk to ~270 tok/s on the last), and the KV pool grew
  65,536 → 1,048,576 in seven `KV grow` steps. Final occupancy 1,040,527 / 1,048,576 pages,
  8,049 free — the same ~8 K headroom the growth run measured.
* Turns 2–8: `cached_tokens` = prompt − 36…49 on every turn, i.e. the entire haystack, the
  previous questions and every previous answer were served from the prefix cache. TTFT
  4.70–7.03 s for a 1.04M-token conversation; decode holds **19.2–20.3 tok/s**, matching the
  18.8 tok/s the incremental growth run measured at 1.04M.
* Whole run 34 min wall (server start 22 s + 30.3 min prefill + 8 turns). Eight independent 1M
  prompts would have been ~4 h — outside the 4 h `gpu_lock` cap on its own.
* VRAM 3.82 GiB reported by `/v1/stats` for the 1.04M q8_0 pool; peak process GPU ~13.9 GiB.

## Profile

```bash
FREETOKEN_PIN_BUDGET_GB=17 \
uv run ft serve --model .../NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --host 127.0.0.1 --port 8124 \
  --kv-cache-dtype q8_0 --attention-backend triton --memory-ratio 0.85 \
  --max-prefill-length 8192 --max-running-requests 1 \
  --max-seq-len-override 1048576 --num-tokens 1048576 --kv-grow-step-tokens 131072 \
  --session-spill-limit-gb 50 --session-spill-ram-gb 0 \
  --session-spill-dir ~/.cache/freetoken/multineedle-1m \
  --linear-state-slots 5 --enable-cache-report
```

Driver: **`benchmarks/bench_multi_needle.py`** (new, this run). It builds the haystack with a
`Filler` sliced to an exact token count (the same digit-free construction
`bench_long_context.py` uses — the filler names no ledger at all, so neither a needle key nor
the control key appears anywhere but in its own needle line), places each needle at its
requested depth (realised depths 0.0501 / 0.2500 / 0.5000 / 0.6000 / 0.7500 / 0.9500), then
drives `/v1/chat/completions` turn by turn, re-sending the whole conversation. It grades the
concatenated SSE `content` fields, never the raw stream; `enable_thinking=False`; no
`ignore_eos` (a forced-length reply ends in tokens the next turn cannot resend, which breaks
the exact-prefix match the design depends on); replies are truncated at `</think>`/`<|im_*|>`
before being echoed back for the same reason. It talks to an already-running server, so the
server stays under `gpu_lock.sh` for the whole session. `--build-only` token-counts the prompt
with no GPU at all.

```bash
uv run benchmarks/bench_multi_needle.py --base-url http://127.0.0.1:8124 \
  --model-dir .../NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --target-prompt-tokens 1040000 --decode 128 --session-id multineedle-1m \
  --out <scratch>/mn1/turns.jsonl
```

## Artifacts

`/tmp/claude-1000/-home-lucas-ai-FreeToken/f4e2e9e3-f4f5-40d0-9980-b3b09d1ef47d/scratchpad/mn1/`
— `run.sh`, `serve.sh`, `server.log`, `driver.log`, `bench.log`, `turns.jsonl`,
`stats_after.json`.
