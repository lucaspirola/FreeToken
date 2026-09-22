# How to read nemotron-{baseline,lever1}-needles.json (and why they must be re-run)

Both files were written under commit `1ed6372`, before the output cap was
raised (`8a16cf3`, `--max-output-tokens` 16384 -> 65536). Read as-is they
understate recall badly.

## The two arms are byte-identical

Across both sizes and all seven question types, `nemotron-baseline-needles.json`
(whole model in RAM) and `nemotron-lever1-needles.json` (pool) agree on every
field that describes the model's output - the answer text, `completion_tokens`,
`reasoning_chars`, `finish_reason`. Only wall-clock `seconds` differs.

That is the correctness evidence this branch needs, and it is the kind the plan
asks for: the OUTPUT is identical, not merely the fault counters. A pool serving
a wrong expert would not reproduce 125 119 reasoning characters exactly.

## Five of the six "correct": false are the old cap, not recall

`needles.py` gives thinking questions `NEEDLES_THINK_MAX_TOKENS` (default
16 384) and the server then capped at 16 384 too, so a thinking question that
wanted more simply stopped mid-sentence and scored 0:

| size | question | finish_reason | completion | verdict |
|---|---|---|---|---|
| 21 000  | multihop      | **length** | 16 383 | truncated - says nothing about recall |
| 120 000 | multihop      | **length** | 16 383 | truncated |
| 120 000 | ordering      | **length** | 16 383 | truncated |
| 120 000 | arithmetic    | **length** | 16 383 | truncated |
| 21 000  | counting      | stop       |    276 | genuine miss |
| 120 000 | counting      | stop       |    154 | genuine miss |

`codes` (the hard gate) and `negative` pass at both sizes in both arms, thinking
off, in 18-27 tokens. `contradiction` passes everywhere. `ordering` and
`arithmetic` pass at 21 000 and only fail at 120 000 where they ran out of
budget.

**`counting` is the one real miss** - it stopped on its own and got the number
wrong, identically in both arms at both sizes. That is a model limitation on
counting scattered occurrences, not a pool defect: the whole model in RAM
misses it exactly the same way.

## Therefore

These two files stay as the correctness comparison (identical output), and are
**not** a recall score. R5 needs a re-run at the raised cap:

    NEEDLES_THINK_MAX_TOKENS=65536 tasks/exclusive-expert-ram/needles.py ...

against the final pool configuration and the whole-model reference, at the plan's
sizes (21 000 / 240 000 / 713 000 / 1 000 000), with the questions sent as
prefix-cached follow-ups to one prefill per haystack. `cached_tokens` is already
recorded per question (19 840 on the 21 000 haystack), so a cache miss is visible.

`needles.py` now also records `think_max_tokens` in the report header; the header
of these two files reports only `max_tokens: 1024`, which is the *non-thinking*
budget and is why the truncations were not obvious.
