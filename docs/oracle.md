# Cross-engine oracle

`benchmarks/oracle_cross_engine.py` runs the same long-context needle suite through
FreeToken and through llama.cpp and produces one report with a verdict per question.

It exists because a single-engine bisect gets the wrong answer. On 2026-09-04 a 262K
depth sweep concluded "model/quant retrieval limit"; llama.cpp then recalled the needle
at every depth and length where FreeToken had missed it
(`benchmarks/results/nemotron35_lightning_5080_262k_crossengine_2026-09-04.md`), which
reopened the verdict as an engine bug. A second engine is the cheapest available oracle
for "is this us or is this the model".

## The verdicts

| verdict | meaning | what to do |
|---|---|---|
| `agree` | both engines answered correctly | nothing |
| `both-miss` | neither answered | a model or prompt limit; do **not** file an engine bug |
| `freetoken-only-miss` | llama.cpp answered, FreeToken did not | reopen / file the engine bug |
| `llamacpp-only-miss` | FreeToken answered, llama.cpp did not | usually a llama.cpp context or quant limit; note it and move on |
| `missing` | one recording has no row, or the engine errored | the run is incomplete; re-record before reading anything else |

**The standing confound.** FreeToken serves NVFP4 safetensors; llama.cpp serves a Q4_0
GGUF. FreeToken has no GGUF loader for `nemotron_h` and llama.cpp has no NVFP4 path, so
on this host engine and quantization cannot be varied independently. Read every
`freetoken-only-miss` as "engine **or** NVFP4". It is still the useful signal — it says
the *model* can answer at that depth and length — it just is not proof the bug is in the
scheduler rather than the weights. The report prints this paragraph every time.

## Reading a miss

Answer-level pass/fail is not enough: at 1M the direct question for the depth-0.25
needle returned the depth-0.05 needle's code, and a combined question then proved the
depth-0.25 needle had been resident all along. So each needle is asked three ways
(**direct** `key -> code`, **combined** with a neighbour, **reverse** `code -> key`) and
each needle has a near-duplicate twin planted half a haystack away (`orchard ledger`
vs `orchard register`). The per-needle table then carries a class, not just a colour:

| class | means | reads as |
|---|---|---|
| `recall` | every probe passed | fine |
| `recall-partial` | direct passed, a composed probe did not | the binding is there; composition is weak |
| `interference-near` | a probe returned the same key's `register` twin | selection failed *between two similar keys* |
| `interference-cross` | a probe returned another key's answer | selection failed *across the context* |
| `selection` | direct missed, a leak-free combined/reverse probe recovered the code | the needle **is in state**; addressing failed |
| `retention` | nothing recovered it | the needle is genuinely not in state |
| `incoherent` | the direct answer carried neither a code nor a denial | decode ran off; not a retrieval result |

`interference-*` and `selection` both mean the state was intact. Only `retention` is
evidence that the KV/SSM state lost the token. Filing a retention bug on an
interference result is the exact mistake this suite was built to stop.

**`in state` and `leak-free`.** Every turn's text joins the conversation, so a code that
has already been printed can be read back out of the transcript. Questions run
`direct -> control -> combined -> reverse` (reverse *states* the code, so it goes last)
and each row records whether that needle's code was still unseen when the turn ran. Only
leak-free probes count as evidence in the `in state` column. A `no` in the leak-free
column does not invalidate a pass — it just means that pass is not independent evidence.

## Host rules

- **One model-loading job at a time**, under `scripts/gpu_lock.sh`. The lock refuses to
  start below 22 GiB `MemAvailable` and kills the job's whole process group on exit.
- **One engine at a time.** At 262K and up neither engine leaves room for the other on
  the 5080. This is why recording and comparing are three separate commands: the
  FreeToken recording finishes and its server is stopped *before* llama-server starts.
- **Stop the host GPU embedder first**: `systemctl --user stop piro-board-embedder`
  (or whatever is holding VRAM) before either engine.
- **The lock caps a job at 4 h** (`FREETOKEN_GPU_LOCK_MAX_HOLD`). A 1M llama.cpp
  recording has never been run on this host and may not fit inside it; see the budget
  table below.
- `--filler-cursor` rotates the haystack so a previous run's session checkpoint cannot
  match. The filler repeats every 64 lines, so **use a value that is not a multiple of
  64** or you get a byte-identical prompt and the stale checkpoint matches anyway.

## The three commands

Set a scratch directory first; all three phases write into it.

```bash
export ORACLE_OUT=~/ai/bench/oracle/$(date +%F)
mkdir -p "$ORACLE_OUT"
export FT_MODEL=~/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4
export GGUF=~/ai/models/nemotron35-gguf/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf
export LEN=262144            # one of 131072 / 262144 / 524288 / 1048576
```

### Phase A — FreeToken

The oracle does **not** start the FreeToken server; start it yourself with the profile
for the length you are testing (`docs/nemotron.md`), then drive it.

```bash
# terminal 1 -- the server, under the lock
FREETOKEN_PIN_BUDGET_GB=17 scripts/gpu_lock.sh \
  ft serve --model $FT_MODEL \
  --max-running-requests 1 --max-seq-len-override 1048576 --num-tokens 1048576 \
  --kv-grow-step-tokens 131072 --kv-cache-dtype q8_0 --attention-backend triton \
  --moe-backend offload --moe-cache-auto --linear-state-slots 6 \
  --memory-ratio 0.85 --max-prefill-length 8192 --host-ram-reserve-gb 6 \
  --session-spill-ram-gb 12 --session-spill-dir /mnt/nvme/ft-spill \
  --port 8123

# terminal 2 -- the recording (CPU-side; the GPU work happens in the server)
uv run benchmarks/oracle_cross_engine.py record \
  --engine freetoken --base-url http://127.0.0.1:8123 \
  --model-dir $FT_MODEL --target-prompt-tokens $LEN --filler-cursor 0 \
  --label "freetoken nvfp4 $LEN" --out "$ORACLE_OUT/ft_$LEN.json"
```

Then **stop the FreeToken server** before phase B.

At 524K and 1M add `--no-generic`. The five haystack-free prompts are cheap in tokens
but they open a second session, and with a 1M lease resident that costs a checkpoint
spill and restore each way.

To hang the Switchyard hidden-state export off the same run, serve with
`--hidden-states-dir /tmp/ft-hidden-states` and pass the same path to `record`; turn 1
then exports its per-layer residuals and the report notes the directory. Feed it to
`benchmarks/probe_hidden_states_parity.py --artifact <path>` afterwards. llama.cpp
exports nothing comparable, so this is a FreeToken-vs-transformers hook on the same
prompt, not a cross-engine tensor diff.

### Phase B — llama.cpp

This one starts and stops `llama-server` itself. Nothing else may hold the GPU.

```bash
scripts/gpu_lock.sh uv run benchmarks/oracle_cross_engine.py record \
  --engine llama.cpp --gguf $GGUF --model-dir $FT_MODEL \
  --target-prompt-tokens $LEN --filler-cursor 0 \
  --n-cpu-moe 14 --llama-log "$ORACLE_OUT/llama_$LEN.log" \
  --label "llama.cpp q4_0 $LEN" --out "$ORACLE_OUT/lc_$LEN.json"
```

`--model-dir` is still the **HF checkpoint**: the haystack is sized and trimmed with the
HF tokenizer for both engines so the prompt is byte-identical, exactly as the 262K
cross-engine run did. llama.cpp retokenizes it with the GGUF vocabulary and the
recording keeps its reported `prompt_tokens` so any drift is visible.

Defaults reproduce the 262K reference invocation: `-c <target+8192> -np 1
--no-context-shift --cache-ram 0 -ngl 999 --n-cpu-moe 14 -fa on -ctk q8_0 -ctv q8_0
-b 4096 -ub 512 -t 16 --jinja --no-warmup`. `--n-cpu-moe 14` is what makes a 17.6 GiB
Q4_0 fit a 16 GiB card; raise it if the server OOMs, lower it for speed if it fits.
Anything else goes through repeated `--llama-arg`.

### Phase C — compare (CPU only, no lock)

```bash
uv run benchmarks/oracle_cross_engine.py compare \
  --freetoken "$ORACLE_OUT/ft_$LEN.json" --llamacpp "$ORACLE_OUT/lc_$LEN.json" \
  --markdown "$ORACLE_OUT/report_$LEN.md" --json "$ORACLE_OUT/merged_$LEN.json"
```

Exit codes: `0` no FreeToken-only miss, `2` at least one (read the report), `3` the two
recordings were not asked the same prompt — the report says so at the top and every
verdict in it is void.

## The standard sweep

**Depth is not a sweep dimension here.** All six needles and all six near-duplicates
live in *one* prompt, at depths 0.05 / 0.25 / 0.50 / 0.60 / 0.75 / 0.95, and the twins
land at 0.58 / 0.78 / 0.03 / 0.13 / 0.28 / 0.48. The suite is 19 graded turns -- 6 direct, 6 combined, 6 reverse, 1 control -- plus 5
short haystack-free prompts. One prefill covers the whole depth sweep; the earlier
design's five separate 262K prefills for five depths cost about an hour of GPU for the
same information. The sweep dimension is **length**:

```bash
for LEN in 131072 262144 524288 1048576; do ... phases A, B, C ... done
```

Do not loop it unattended. Each length is two lock acquisitions with a manual server
stop between them, and 524K/1M should be run only after the shorter lengths agree.

### Budget

FreeToken numbers are measured (`docs/nemotron.md`, the 1M multi-needle result);
llama.cpp is measured at 262K only.

| length | FreeToken prefill | FreeToken 19 turns | FreeToken total | llama.cpp prefill | llama.cpp total |
|---:|---:|---:|---:|---:|---:|
| 131,072 | ~45 s (3,007 tok/s) | ~2 min | **~5 min** | ~60 s | **~10 min** |
| 262,144 | ~140 s (1,860 tok/s) | ~3 min | **~8 min** | 141 s (2,230 tok/s) | **~12 min** |
| 524,288 | ~500 s (1,064 tok/s) | ~5 min | **~15 min** | not measured | budget 30 min |
| 1,048,576 | 1,815 s (573 tok/s) | ~5 min | **~35 min** | not measured | **may not fit the 4 h cap** |

Add ~25 s for FreeToken server start and several minutes for `llama-server` to load an
18.9 GiB GGUF with 14 CPU-MoE blocks. A full four-length sweep of both engines is
roughly **2.5–3 h of GPU** if 1M llama.cpp behaves, and the 1M llama.cpp leg is the one
to attempt last and alone.

The whole point of the one-prefill-many-turns design is that the 19 graded turns are
nearly free: at 1M they ride the prefix cache at TTFT 4.7–7.0 s and 19–20 tok/s decode,
against ~30 min for a cold re-prefill each. If a report shows `cached_tokens` collapsing
to near zero on turns 2+, the prefix match broke and the run is measuring re-prefill,
not recall — check for a stray `</think>` in the previous reply before trusting anything
in it.

## Logprobs: the current gap

The oracle probes both servers at startup with a one-token
`logprobs: true, top_logprobs: 2` request and records what came back.

**FreeToken does not support logprobs on any endpoint.** `/v1/chat/completions` rejects
`top_logprobs > 0` outright (`python/freetoken/server/openai_api.py`, "logprobs are not
supported") and `/v1/completions` rejects `logprobs` the same way. This is not a
serialization gap that could be patched in the response builder: `SamplingParams` in
`python/freetoken/core.py` has no `logprobs` / `top_logprobs` / `return_logprob` field,
the `Req` object carries no slot for them, and nothing in the scheduler or sampler
gathers per-token or top-k probabilities at any point. Exposing them means adding
collection to the sampling path — a feature, not a wiring fix — so it was deliberately
**not** done as part of this tool.

Consequently the oracle compares at the **answer level** today and prints the gap in the
report's Logprobs section. The comparison itself (top-1 token agreement by decoded
string, mean absolute logprob delta, first divergent position, capped at
`--logprob-positions`) is implemented and unit-tested against mocked engines, and starts
producing that table the day `SamplingParams` grows the field — no change to this tool.

Tokens are joined by decoded **string**, not id: both engines carry the same vocabulary
but reach it through different tokenizer implementations, so ids are not a safe key.

## Tests

```bash
uv run pytest tests/benchmarks/test_bench_multi_needle.py \
              tests/benchmarks/test_oracle_cross_engine.py
```

No GPU, no model, no network: the transport is exercised against a throwaway
`http.server` that speaks streamed chat completions, and the end-to-end test drives a
full record → compare → render pipeline through two fake engines, one of which
reproduces the 1M failure shape.
