# Prefix reuse across conversations (exp/prefix-reuse 99630d9)

Replay of captured Claude Code 2.1.282 / omp / Codex requests on :1920 (whole model, ratio 1.00,
Nemotron 3.5 Lightning), `replay.py`: cold (nonce), A (old session +110K chars), clear (first
request after /clear), repeat, next (the post-/clear conversation's next turn). Cached tokens:

| client | step | hoisted (switch off) | in place (default) |
|---|---|---|---|
| Claude Code | clear | 0 (TTFT 2.55 s) | 16609 = segment (0.37 s) |
| Claude Code | next  | 0 of 19032 (3.01 s) | 18913 of 19042 (0.81 s) |
| omp | clear / next | 8619 / 8619 | 8619 / 8619 |
| Codex | clear | 9143 (journal) | 9143 (journal) |

Before this branch every post-/clear request got 0; fixes 1-3 alone (prefixfix1) gave omp/Codex
8064 and Claude Code 0. omp/Codex now resume at the exact segment end (8619, 9143) instead of
the last chunk boundary. Claude Code's "next" resumes at the last 128-token track boundary
before turn N's generation prompt (turn N+1 renders the reply without its reasoning, so turn N's
end state cannot match; turn N's prompt minus "<think>\n" is a token prefix of turn N+1's).
TTFT of the ~100-token steps varies 0.3-0.8 s between arms with equal cached counts; the
counts are the result. The OFF arm's Claude Code A put the filler into a system message that
hoisting moved into the system text, so its A differs from the ON arm's.

Snapshots (trace, ON arm): 37 chunk + 7 segment + 22 end inserts, 56 snapshot evictions,
#mamba-slot 3-8/12, one idle-shrink lease release (spill validated), 0 fallbacks.

Tool use (`tooluse.py`, greedy, one sample per arm, Claude Code shape: environment block as a
system message after the first user turn, reminder system message after the tool result):
both arms call Bash with `ls -la /srv/demo-repo-7731` (the cwd from the environment block) and
list the files; only in-place followed the reminder ("end with FINISHED"); turn 2 cached 339
of 772 tokens in place, 0 hoisted.

ft-dev: tests/moe engine scheduler kernels kvcache server tokenizer, 3017 passed,
25 skipped (`ftdev-tests-99630d9.txt`). Local CPU: scheduler+kvcache+server+tokenizer 1811 passed.

## Captures (recaptured 2026-09-25, round 4)

The original captures lived in a /tmp scratchpad and were lost when WSL restarted. They are now
committed here, together with the harness that makes them:

* `capture.sh WORKDIR`: Claude Code 2.1.282, omp 18.3.0 and Codex 0.155.1 in a private tmux
  (`-L probe`) against `stub.py` on 127.0.0.1:18080. Each is redirected with environment
  variables only: a throwaway HOME / CODEX_HOME under WORKDIR and a dummy key, so ~/.claude*,
  ~/.omp and ~/.codex are only read and auth.json is never copied. The conversation is the six
  turns of `turns.txt` (turn 2 pastes `client_sessions.py`), then /clear, then two short turns.
  Codex gets its turns as a bracketed paste: typed fast, it folds them into one message.
* `redact.py WORKDIR/raw`: auth headers become `<dummy>`; the value of Claude Code's billing
  block is removed (its `x-anthropic-billing-header:` prefix stays, because the server strips
  the block by that prefix); UUIDs and hex ids get consistent pseudonyms, so session and
  prompt_cache_key sharing is kept. The script fails if anything key-shaped survives.
* `claude-code.jsonl` (16 requests), `omp.jsonl` (33), `codex.jsonl` (16): the redacted captures.
  `replay.py` now picks its requests by structure rather than by row number (main calls =
  the full tool list; /clear = where the conversation length drops). Its filler is this
  tree's `python/freetoken/server/*.py`.
* `arm.sh WT ARM OUTDIR [ROWS]`: one :1920 arm with `replay.py` as FT_POST, under the host
  lock.

Because the conversation before /clear is longer and the client prompts are newer, these
captures do not reproduce the absolute counts in the table above. Compare arms replayed from
the same captures.
