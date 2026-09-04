# Nemotron 3.5 Lightning NVFP4 — 1M single-session gate, remaining criteria

Host: RTX 5080 16 GiB, WSL2 34 GiB RAM + 4 GiB swap, NVMe `~/.cache`. Tree `d685e99`
(clean; the untracked `*_scheduler_bisect_*.md` belongs to another agent). Model
`/home/lucas/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`. Runs 2026-09-04
18:14–19:58 local, every model-loading process under `scripts/gpu_lock.sh`, one at a time.

This closes the four items the earlier 1M-gate session left open. Growth to 524K × 3 and
spill/restore-on-demand were already verified there
(`.../af23ede4-…/scratchpad/1m/growth2.log`); this run takes **one** session all the way to
1.04M and exercises the three residency behaviours the profile promises, then re-runs the
262K/524K needles now that the Mamba-2 `dt` floor is fixed (`3ac79ec`).

| # | Criterion | Verdict | Headline number |
|---|---|---|---|
| a | Sessions survive a server restart and restore | **PASS** | 1,040,020-token checkpoint adopted by a new process; turn took 9.8 s instead of ~31 min of prefill |
| b | Capacity/age eviction behaves per spec | **PASS** | 1.6 GiB cap, third spill evicted the older of two candidates; survivors still restore |
| c | ≥1M-size NVMe restore timing | **PASS** | 3.53 GiB restored from disk in **2.681 s (1.32 GiB/s)**; spill 2.980 s (1.18 GiB/s) |
| d | 262K/524K needles via `/v1/chat/completions` | **PASS** | 262,160 and 524,304 tokens at depth 0.50, both recall `5663623` |

## Profile

```bash
FREETOKEN_PIN_BUDGET_GB=17 \
uv run ft serve --model .../NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --host 127.0.0.1 --port 8123 \
  --kv-cache-dtype q8_0 --attention-backend triton --memory-ratio 0.85 \
  --max-prefill-length 8192 --max-running-requests 1 \
  --max-seq-len-override 1048576 --num-tokens 1048576 --kv-grow-step-tokens 131072 \
  --session-spill-dir <nvme> --session-spill-ram-gb 0 \
  --session-spill-limit-gb <50 | 1 | 1.6> \
  --linear-state-slots 5 --enable-cache-report
```

`--session-spill-ram-gb 0` is deliberate: it forces every checkpoint onto the **NVMe** tier,
which is what criteria (a) and (c) are about. The 4 GiB default would have kept the 3.5 GiB
checkpoint in RAM, where a restart destroys it.

Driver: `scratchpad/1m2/drive.py` (copied from the previous session) — each turn resends the
whole conversation plus a fresh block of digit-free filler through `/v1/chat/completions` with
an `x-switchyard-session-id` header, grades the concatenated SSE `content` fields (never the
raw stream), and persists the conversation to a state file so a turn can be taken against a
*different server process*.

## 1. Growth to 1.04M tokens in one session

`drive.py --step 130000 --turns 8 --sessions C --decode 96 --tag gate1m-v3`

| turn | prompt | cached | fresh | TTFT s | fresh prefill tok/s | decode tok/s | needle |
|---|---|---|---|---|---|---|---|
| 1 | 129,992 | 0 | 129,992 | 40.0 | 3,250 | 66.6 | PASS |
| 2 | 260,019 | 130,023 | 129,996 | 95.9 | 1,356 | 47.8 | PASS |
| 3 | 389,998 | 260,050 | 129,948 | 151.2 | 860 | 39.4 | PASS |
| 4 | 519,989 | 390,029 | 129,960 | 205.0 | 634 | 31.3 | PASS |
| 5 | 649,992 | 520,020 | 129,972 | 259.8 | 500 | 27.0 | PASS |
| 6 | 779,995 | 650,023 | 129,972 | 315.4 | 412 | 23.2 | PASS |
| 7 | 909,998 | 780,026 | 129,972 | 370.8 | 351 | 20.8 | PASS |
| 8 | 1,039,989 | 910,029 | 129,960 | 423.4 | 307 | 18.8 | PASS |

"fresh prefill tok/s" is the marginal rate for a 130K tail appended at that depth, not a
whole-prompt rate; it decays ≈ 1/context, so the eight turns cost 1,861 s of prefill in total.
Decode falls from 66.6 to 18.8 tok/s between 130K and 1.04M. A whole run was executed twice
(18:14 and 18:51, different filler tags) with per-turn rates agreeing inside 2 %; the needle
was recalled at every length in both.

The needle sits in the middle of turn 1's block, so its *relative* depth shrinks as the
conversation grows (0.50 at turn 1 → 0.06 at turn 8): this is a retention-at-distance result,
not a depth-0.5 result. §5 is the depth-0.5 gate.

## 2. Spill happens on demand, and only on demand

The first attempt's trigger was a 6-token request from a foreign session. It returned **200 in
0.347 s and spilled nothing**: the resident 1.04M session leaves ~8.5K tokens of the
1,048,576-token pool free, the tiny request fit, so no admission failed and the policy
correctly did not checkpoint anybody. That is criterion 3E ("no spill while nothing is waiting
on the space") demonstrated by accident, and it cost an hour — a demand test needs a prompt
that *cannot* fit.

The real trigger (`scratchpad/1m2/trigger.py`, 60,017 tokens, foreign session id):

```
trigger: 60017 prompt tokens, 15.2s, text='ok'
Spilled soft session auto:switchyard:…b87940d: 1040020 tokens, disk, 3.53 GiB in 2.980 s (1.18 GiB/s)
Released soft session auto:switchyard:…b87940d KV protection (admission pressure); cached prefix is now evictable
```

On-disk record after the server stopped: 3,793,835,648 bytes, `manifest.json` v2,
`num_tokens=1040020`, 1,552 chunks. 3.53 GiB for 1.04M tokens is 3.65 KiB/token — the q8_0 KV
plus 8 × 47 MiB of recurrent-state boundaries.

## 3. Criterion (a) restart persistence + (c) ≥1M NVMe restore — PASS

The phase-1 server was stopped (`SIGINT`, clean shutdown, GPU back to 0 MiB) and a **new**
`ft serve` process was started on the same spill root. Its startup adopted the checkpoint:

```
Session spill root /home/lucas/.cache/freetoken/gate-1m-spill: adopted 1 checkpoint(s), removed 0 stale entr(ies)
```

The session's next turn (the same conversation plus a 60-token question) then restored instead
of re-prefilling:

```
[gate] turn 9 C: prompt=1040080 cached=1040020 fresh=60 decode=18.8 recall=True wall=9.8s
KV grew 131072 -> 1048576 tokens to restore cold session auto:switchyard:…b87940d
Restored cold session auto:switchyard:…b87940d: 1040020/1040020 tokens from disk, 3.53 GiB in 2.681 s (1.32 GiB/s)
```

- **Restore rate 1.32 GiB/s at 1M size**, matching the 1.3 GiB/s measured at 393K, so the
  earlier "~2.5 s projected at 1M" estimate behind the 3F prefetch GO decision was right
  (measured 2.681 s). The RAM tier remains ~5–8 GiB/s, so 3F still pays for itself.
- Whole-prefix match (`1040020/1040020`), not a partial restore.
- The turn cost 9.8 s wall against the 1,861 s of prefill that built the same prefix — a 190×
  saving, and it crossed a process boundary.
- **Coherence survives the round trip**: the answer after the cold restore is byte-identical to
  the answer the resident session gave before the spill —
  `8324516\neight three two four five one six\nThe quarry ledger\nEvery marker record in the log
  was inactive.` (session C's needle is `8324516`/`quarry`), i.e. a correct 1.04M-token recall.

Two supporting observations:

- An old **v1** manifest (655,450 tokens, 2.04 GiB) left by the previous session was rejected on
  startup — `adopted 0 checkpoint(s), removed 1 stale entr(ies)` — because `b7242d2` bumped
  `MANIFEST_VERSION` to 2. The stale-record GC half of the criterion works too.
- A restore **consumes** its record (the spill root is empty again afterwards), and a resident
  session is never checkpointed while nothing needs its space. So a `SIGKILL`/crash while a
  session is resident still loses that session — see the ticket in §6.

## 4. Criterion (b) capacity + age eviction — PASS

131K-token checkpoints are 0.54 GiB each here (0.44 GiB of KV + the v2 boundary states).

**Run 1 — `--session-spill-limit-gb 1` (cap holds exactly one record), three sessions × 2 turns.**
Every spill evicted the previous record; both evicted sessions fell back to an ordinary
prefill on their next turn (`cached=0`, 262K re-prefilled at 1,914 tok/s, answers still
correct) and the one surviving record restored in 0.159 s (3.37 GiB/s). When a session's
turn-2 checkpoint grew to 1.08 GiB — larger than the whole 1.00 GiB cap — the store refused it
outright instead of evicting the world for it:
`Cold checkpoint for session … did not fit RAM/disk budgets; resume will recompute if its GPU
prefix is evicted`. The store therefore held at most one 0.54 GiB record at a time — which is
also why this cap could not show *which* record LRU picks, hence run 2.

**Run 2 — `--session-spill-limit-gb 1.6` (cap holds two of three), so the victim is a real
choice.** Sessions D, E, F take one 131K turn each, then a 60K foreign session forces F's spill:

```
19:41:18 Spilled soft session …05745d43 (D): 131090 tokens, disk, 0.54 GiB in 0.466 s (1.15 GiB/s)
19:42:03 Spilled soft session …1233ec58 (E): 131091 tokens, disk, 0.54 GiB in 0.482 s (1.11 GiB/s)
19:42:15 Evicting cold session checkpoint …05745d43 (D) (0.54 GiB, disk) to stay inside the session spill cap
19:42:16 Spilled soft session …acbe28a9 (F): 131102 tokens, disk, 0.54 GiB in 0.449 s (1.19 GiB/s)
19:42:27 Restored cold session …1233ec58 (E): 131091/131091 tokens from disk, 0.54 GiB in 0.255 s (2.10 GiB/s)
```

- The cap is enforced across the whole store: 3 × 0.54 = 1.62 GiB > 1.6 GiB, so one record had
  to go. Measured spill root peaked at 1,152,028,810 B (1.073 GiB).
- The victim was **D**, the older of the two candidates by `last_used_at`
  (1788536475.1 vs E's 1788536522.6) — oldest-by-last-use, as specified.
- The survivor is intact: E's next turn restored 131,091 cached tokens in 0.255 s and answered
  its own needle correctly; the evicted D would have re-prefilled.
- TTL is not involved: leases expire at 300 s while checkpoints live by capacity only.

## 5. Criterion (d) 262K / 524K needles after the `dt`-floor fix — PASS

`benchmarks/bench_long_context.py`, built-in synthetic needle (digit-free filler), depth 0.50,
graded on the concatenated chat SSE content:

```bash
FREETOKEN_PIN_BUDGET_GB=17 uv run benchmarks/bench_long_context.py --synthetic-needle \
  --model .../NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 --needle-depth 0.5 \
  --kv-cache-dtype q8_0 --target-prompt-tokens 262144|524288 --max-context 1048576 \
  --mem-ratio 0.85 --kv-grow-step-tokens 131072 --linear-state-slots 5 \
  --prefill-chunk 8192 --decode 128 --server-arg "--session-spill-dir off"
```

| prompt tokens | needle | TTFT s | prefill tok/s | decode tok/s | VRAM GiB |
|---|---|---|---|---|---|
| 262,160 | **found** | 136.2 | 1,924.8 | 56.33 | 2.59 |
| 524,304 | **found** | 492.7 | 1,064.1 | 34.54 | 2.59 |

Both answered `5663623` with the identical output hash (`bb7b67af7a9c`). This **retires the
cache study's coherence caveat** ("needle passes at 131K but is missed at 262K and 524K …
treat ~131K–256K as the coherent ceiling"): that run predates both the chat-endpoint gate and
the `dt`-floor fix. Prefill is also faster than the 2B4 rows measured on the same profile
(262K 1,790 → 1,925 tok/s; 524K 997 → 1,064) and decode is up (51.8 → 56.3; 32.0 → 34.5).

## 6. Open items found here (tickets, no fix applied)

1. **`_restore_cold_session` trusts a stale record reference.**
   `scheduler.py:1364` picks `record = session.spill or store.get(session_id)` without
   checking `record.valid`. `SessionSpillStore.discard()` clears `record.chunks` (and hence the
   state boundaries) but the lease still points at the object, so a restore attempted after a
   *capacity eviction* takes the `length <= 0` branch and logs
   `Discarded cold session …: client tokens diverge at 131103, before the first stored state
   boundary` — a prefix-divergence message for what was actually an eviction (observed twice in
   §4 run 1, with `matched` equal to the full record length, which cannot be a divergence).
   Behaviour is still correct (both paths fall back to prefill), but the log is misleading and
   the same line would shadow a *newer valid* record for that session id. Fix is one line
   (`session.spill if session.spill and session.spill.valid else store.get(...)`) plus a
   distinct message; not applied here because `scheduler.py` is being edited by another agent.
2. **A resident session is not checkpointed at shutdown.** Restart persistence covers spilled
   checkpoints only; the session holding the GPU is by design never spilled while nothing is
   waiting, and a restore consumes its record. A clean `SIGINT` (or a crash) therefore loses the
   resident 1M session. `shutdown()` already demotes RAM records to disk — checkpointing the
   resident lease there too (opt-in flag) would make "survive restart" true for the session that
   matters most. Cost: one 3.5 GiB write, ~3 s.
3. **Eviction can pick the session that is about to be restored.** `_evict_one_lru` excludes only
   the session currently *spilling*, so under a tight cap the victim can be the very record the
   pending admission needs (§4 run 1: cap 1 GiB, every turn evicted its predecessor). Excluding
   the pending session id would turn a guaranteed re-prefill into a hit.
4. Hygiene: `~/.cache/freetoken/session-spill` still holds 28 GiB / 245 record dirs from earlier
   runs. Adoption only scans the *configured* root, so a root that a later run does not use is
   never trimmed.

## 7. Reproduction

Scratch tree (drivers, per-phase server logs, JSONL rows):
`/tmp/claude-1000/-home-lucas-ai-FreeToken/f4e2e9e3-…/scratchpad/1m2/` —
`serve.sh`, `drive.py`, `trigger.py`, `wait_ready.sh`, and one script per gpu_lock hold:
`hold1b.sh` (growth → demand spill → restart → restore), `hold3_evict.sh` (cap = 1 GiB),
`hold4_lru.sh` (cap = 1.6 GiB, real victim choice), `hold2_needles.sh` (262K/524K).
Logs: `hold1b.log`, `hold3.log`, `hold4.log`, `hold2.log`, `needles.jsonl`, `p3/p4.jsonl`.
Neither the scratch tree nor `/tmp` survives a WSL restart.

```bash
scripts/gpu_lock.sh <scratch>/1m2/hold1b.sh       # ~35 min, criteria (a) and (c)
scripts/gpu_lock.sh <scratch>/1m2/hold4_lru.sh    # ~6 min,  criterion (b)
scripts/gpu_lock.sh <scratch>/1m2/hold2_needles.sh # ~20 min, criterion (d)
```

GPU at 0 MiB and no FreeToken venv processes after every hold.
