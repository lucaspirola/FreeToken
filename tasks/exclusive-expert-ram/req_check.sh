#!/usr/bin/env bash
# Campaign requirement checks for the reorganisation (plan section 9.1), one per id:
#   tasks/exclusive-expert-ram/req_check.sh R-S3
# Runs from the reorg worktree root whatever the caller's cwd is (the campaign runner starts in
# the main checkout, where the first recipes silently measured main's code), and ASSERTS the
# plan's expected value: a bare `grep | wc -l` exits 0 whatever it counts, so it proves nothing.
# Exit 0 = the requirement's evidence holds on this tree; non-zero = it does not (message says why).
# Torch-backed pytest here needs the model unloaded (CLAUDE.md); run it only with the GPU empty.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 2
PY=/home/lucas/ai/FreeToken/.venv/bin/python
R=tasks/exclusive-expert-ram/results
export PYTHONPATH=$PWD/python

fail() { echo "FAIL $ID: $*"; exit 1; }
ok() { echo "OK $ID: $*"; }
expect_count() {  # expect_count <label> <expected> <actual>
  [ "$3" = "$2" ] || fail "$1 is $3, expected $2"; }
pytest_q() { "$PY" -m pytest -q "$@" || fail "pytest $*"; }

ID=${1:?usage: req_check.sh R-<id>}
case "$ID" in
R-S0)
  git fetch -q origin || fail "git fetch origin"
  if git merge-tree --write-tree origin/main HEAD > /tmp/req_check_mt.$$ 2>&1; then
    ok "origin/main merges into HEAD with 0 conflicts ($(git rev-parse --short origin/main))"
  else
    n=$(grep -c '^CONFLICT' /tmp/req_check_mt.$$); rm -f /tmp/req_check_mt.$$
    fail "$n conflicts merging origin/main"
  fi
  rm -f /tmp/req_check_mt.$$ ;;
R-S1)
  expect_count 'stats"].tolist() in offload_cache.py' 0 "$(grep -c 'stats"\].tolist()' python/freetoken/moe/offload_cache.py)"
  pytest_q tests/moe/test_mirror_stats.py
  ok "named counter schema; mirror_stats tests pass" ;;
R-S2)
  # Code only: the plan and review documents name these symbols to record their deletion, so a
  # tree-wide grep can never reach the plan's literal 0 (the plan's own Verify line matches it).
  n=$( { grep -rn '_mirror_restore_coverage\|_mirror_needs_coverage' python tests --include=*.py
         grep -rn '_mirror_restore_coverage\|_mirror_needs_coverage' tasks --include=*.py | grep -v '^\S*:[0-9]*: *#'; } | wc -l)
  expect_count "code references to the deleted coverage-restore paths" 0 "$n"
  ls $R/s2-test_mirror_device-*.log >/dev/null 2>&1 || fail "no test_mirror_device.py log in $R"
  grep -q ' passed' "$(ls -t $R/s2-test_mirror_device-*.log | head -1)" || fail "test_mirror_device log has no pass"
  ok "0 code references; $(tail -1 "$(ls -t $R/s2-test_mirror_device-*.log | head -1)")" ;;
R-S3)
  expect_count "def _arena_chunk_boundaries in python/" 1 "$(grep -rn 'def _arena_chunk_boundaries' python | wc -l)"
  ok "one _arena_chunk_boundaries" ;;
R-S4)
  expect_count "false radix claim in STATUS.md" 0 "$(grep -n "resolves to .cache_type='radix'" tasks/exclusive-expert-ram/STATUS.md | wc -l)"
  ok "STATUS.md no longer claims radix" ;;
R-S5a)
  expect_count "'H // 16' in mirror_pool.py" 0 "$(grep -c 'H // 16' python/freetoken/moe/mirror_pool.py)"
  pytest_q tests/moe/test_nvfp4_row_layout.py
  ok "single NVFP4 row layout; byte-equality tests pass" ;;
R-S5b)
  pytest_q tests/moe/test_bank_bytes_estimate_gated.py
  ok "gated bank-bytes estimate tests pass" ;;
R-S6)
  n=$(grep -c '_mirror' python/freetoken/moe/offload_cache.py); [ "$n" -le 15 ] || fail "_mirror count $n > 15"
  expect_count 'getattr(self, "_mirror"' 0 "$(grep -c 'getattr(self, "_mirror"' python/freetoken/moe/offload_cache.py)"
  "$PY" -c "import sys, freetoken.moe.offload_cache; assert 'freetoken.moe.mirror_pool' not in sys.modules" \
    || fail "whole-model path imports the mirror pool"
  grep -q 'RESULT: IDENTICAL' $R/instrument-s6s7-compare.txt || fail "plan 7.3 instrument not IDENTICAL"
  ok "_mirror $n (<=15), no getattr, pool not imported, 7.3 IDENTICAL" ;;
R-S7)
  # S7 rewrote the source-parsing tests: *_transaction_source.py / *_handoff_policy_source.py are
  # now test_growable_kv_transaction.py / test_growable_handoff_policy.py (commit e85c4e9), both
  # inside the globs below.
  pytest_q tests/moe/test_offload_kernels_usable_slots_source.py tests/scheduler/test_growable_*.py tests/engine/test_growable_*.py
  n=$(wc -l < python/freetoken/engine/engine.py); base=$(git show cde7ace:python/freetoken/engine/engine.py | wc -l)
  [ $((base - n)) -ge 600 ] || fail "engine.py shrank by $((base - n)) lines (< 600)"
  ok "growable suites pass; engine.py $base -> $n (-$((base - n)))" ;;
R-S8)
  "$PY" -m freetoken.models.check_experts ~/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 || fail "check_experts Nemotron"
  "$PY" -m freetoken.models.check_experts ~/ai/models/Ornith-1.5-35B-A3B-NVFP4 || fail "check_experts Ornith"
  pytest_q tests/models/test_expert_source_conformance.py
  ok "check_experts accepts Nemotron and Ornith; conformance tests pass" ;;
R-S9)
  scripts/upstream-sync-check.sh || fail "upstream-sync-check.sh"
  ls $R/sync-check-*.txt >/dev/null 2>&1 || fail "no filed sync-check result"
  ok "sync check passes" ;;
R-S10)
  expect_count "destroy_cuda_graphs in growable_kv.py" 0 "$(grep -c 'destroy_cuda_graphs' python/freetoken/engine/growable_kv.py)"
  expect_count "graph captures in the checkpoint-2 1M journal" 1 "$(grep -c 'Start capturing CUDA graphs' $R/ck2-mirror-1m-journal.txt)"
  ok "legacy recapture path gone; one capture at checkpoint 2" ;;
R-S11)
  expect_count "'elastic' in python/freetoken" 0 "$(grep -rn 'elastic' python/freetoken --include=*.py | wc -l)"
  ok "--elastic-initial-requests retired" ;;
R-S12a)
  # Plan target: prefill >= 5000 tok/s at 32K AND 80K (prompt tokens / pass-2 TTFT), plus the
  # roofline and the profile. Asserted as written; a miss fails this check.
  f=$R/ornith-s12a/ornith-s12a-record.json
  [ -f "$f" ] && [ -f $R/ornith-s12a/roofline-bw.txt ] && [ -d $R/ornith-s12a/profile ] || fail "record/roofline/profile missing"
  "$PY" - "$f" <<'EOF' || fail "prefill target not met (see above)"
import json, sys
d = json.load(open(sys.argv[1]))
bad = 0
for label, toks in (("32k", 32000), ("80k", 80000)):
    rate = toks / d[f"ttft_{label}"]
    print(f"  prefill {label}: {rate:,.0f} tok/s (target 5,000)")
    bad |= rate < 5000
sys.exit(bad)
EOF
  ok "prefill >= 5000 at 32K and 80K; roofline and profile present" ;;
R-S13)
  pytest_q tests/kvcache/radix/test_hybrid_radix_pins.py -k session_scope
  "$PY" - $R/s13-session-pins.json <<'EOF' || fail "measured receipt misses the target"
import json, sys
d = json.load(open(sys.argv[1]))
hay = max(s["prompt_tokens"] for s in d["steps"])
qs = [s for s in d["steps"] if s["label"] in ("A2", "A3")]
assert len(qs) == 2, "no A2/A3 steps"
for s in qs:
    print(f"  {s['label']}: cached {s['cached_tokens']:,} of {s['prompt_tokens']:,}, TTFT {s['ttft_s']:.2f} s")
    assert s["cached_tokens"] >= hay - 8192 and s["ttft_s"] < 3.0
EOF
  ok "session-scope tests pass; questions 2-3 hit the haystack under 3 s" ;;
R-CKPT-1|R-CKPT-2)
  ck=$( [ "$ID" = R-CKPT-1 ] && echo ck1 || echo ck2 )
  for a in mirror mirror-1m; do
    grep -q 'captures=1 .*tracebacks=0' $R/$ck-$a-acceptance-R3.txt || fail "$ck-$a R3"
    grep -q '^R6(arm) ok' $R/$ck-$a-acceptance-R6.txt || fail "$ck-$a R6"
  done
  grep -q '^0 difference' $R/$ck-needles-compare.txt && ! grep -q '^DIFF' $R/$ck-needles-compare.txt \
    || fail "needle/recall answers differ from the whole-model reference ($ck-needles-compare.txt)"
  [ $ck = ck2 ] && { grep -q 'RESULT: IDENTICAL' $R/instrument-s6s7-compare.txt || fail "7.3 instrument"; }
  cmp=$R/$ck-records-compare.txt
  # Plan 7.2: a point failing only with the delivery-stall signature is re-measured on the same
  # commit. Checkpoint 1's 1M pass 1 (30.7) was re-measured by diag-1m-p1stall (commit 83e5efb).
  allowed=""
  [ $ck = ck1 ] && "$PY" - $R/diag-1m-p1stall-probe.jsonl <<'PYEOF' && allowed=decode_1000k_p1
import json, sys
p = [json.loads(l) for l in open(sys.argv[1])]
v = [x["decode_tok_s"] for x in p if x["target"] == 1000000 and x["pass"] == 1]
print(f"  ck1 1M pass-1 re-measure: {v[0]} tok/s (record 76.9, band >= {0.91*76.9:.1f})")
sys.exit(0 if v and v[0] >= 0.91 * 76.9 else 1)
PYEOF
  # decode pass by pass, faults, starved: every line of the comparison must read ok; the only
  # accepted OUT is the documented ram_gib readiness delta of ck2-mirror-1m (commit 9e67f45:
  # anon 12.69 vs record 12.81, geometry byte-identical)
  grep -E '^\s+(decode|coverage|starved)' $cmp | grep -v ' ok' | grep -v "^\s*${allowed:-NONE}\s" | grep -q . \
    && fail "a decode/fault/starved line is outside the band in $cmp"
  "$PY" - $R/$ck-mirror-1m-record.json $R/nemotron-reserve-2e-1m-record.json <<'EOF' || fail "1M RAM outside 12.26 +/- 0.6 GiB on ram_gib and anon_gib"
import json, sys
a, r = (json.load(open(p)) for p in sys.argv[1:])
okram = abs(a["ram_gib"] - 12.26) <= 0.6 or abs(a["anon_gib"] - r["anon_gib"]) <= 0.6
print(f"  1M RAM: ram_gib {a['ram_gib']}, anon {a['anon_gib']} (record {r['anon_gib']})")
sys.exit(0 if okram else 1)
EOF
  ok "$ck: R3/R6 on both pool arms, decode/faults/starved in band, RAM in band" ;;
*) echo "unknown requirement $ID"; exit 2 ;;
esac
