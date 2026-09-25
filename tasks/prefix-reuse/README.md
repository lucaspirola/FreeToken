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
