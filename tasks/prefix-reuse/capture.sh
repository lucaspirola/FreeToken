#!/usr/bin/env bash
# Record Claude Code, omp and Codex requests across /clear for replay.py (CPU only, no server).
#
#   capture.sh WORKDIR [claude-code|omp|codex ...]
#
# Each client runs in a private tmux server (-L probe) against stub.py on 127.0.0.1:18080.
# It is redirected with environment variables only: a throwaway HOME (Claude Code, omp) or
# CODEX_HOME (Codex) under WORKDIR, and a dummy API key. The owner's ~/.claude*, ~/.omp and ~/.codex
# are only read (omp's model list and Codex's config are copied into the throwaway homes), never
# written. auth.json is never copied.
# Conversation: the six turns in turns.txt (turn 2 pastes python/freetoken/server/client_sessions.py),
# then /clear, then two short turns. Raw captures go to WORKDIR/raw/<client>.jsonl. Keep WORKDIR
# outside git: it holds the copied configs. Then redact.py WORKDIR/raw writes the redacted
# <client>.jsonl next to replay.py.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
W=${1:?workdir}; shift
CLIENTS=${*:-claude-code omp codex}
mkdir -p "$W/raw"; W=$(cd "$W" && pwd)
PORT=18080
SPATH=/home/lucas/.local/bin:/usr/local/bin:/usr/bin:/bin
T="tmux -L probe"

paste_text() {   # client_sessions.py flattened to one line (a newline would submit the turn)
    tr '\n' ' ' < "$REPO/python/freetoken/server/client_sessions.py" | tr -s ' ' | cut -c1-6000
}

rows() { wc -l < "$F" 2>/dev/null || echo 0; }

wait_rows() {    # until the stub logged a request beyond $1, then let side calls settle
    local n=$1
    timeout 90 bash -c "until [ \$(wc -l < '$F' 2>/dev/null || echo 0) -gt $n ]; do sleep 1; done" \
        || echo "  no request after row $n" >&2
    sleep 8
}

say() {          # enter one turn and submit it. Codex takes fast typing for a paste burst and
    local n; n=$(rows)  # folds the next turns into it, so it gets a bracketed paste instead.
    if [ "$S" = codex ]; then
        printf '%s' "$1" | $T load-buffer -b turn -; $T paste-buffer -p -d -b turn -t "$S"; sleep 3
    else
        $T send-keys -t "$S" -l "$1"; sleep 1.5
    fi
    $T send-keys -t "$S" Enter
    wait_rows "$n"
}

slash() {        # a slash command: type, let the completion menu settle, submit
    $T send-keys -t "$S" -l "$1"; sleep 2
    $T send-keys -t "$S" Enter; sleep 4
    $T capture-pane -p -t "$S" | grep -v '^\s*$' | tail -4 | cut -c1-140
}

start_stub() {
    ss -ltn "sport = :$PORT" | grep -q LISTEN && { echo "port $PORT busy" >&2; exit 1; }
    CAPTURE=$F PORT=$PORT python3 "$HERE/stub.py" & STUB=$!
    sleep 1
}

stop_all() { $T kill-session -t "$S" 2>/dev/null; kill "$STUB" 2>/dev/null; wait "$STUB" 2>/dev/null; }

run_client() {   # $1 name, $2 launcher script
    S=$1; F=$W/raw/$1.jsonl; : > "$F"
    start_stub
    $T new-session -d -s "$S" -x 220 -y 50 "$2; sleep 600"
    sleep 12
    $T capture-pane -p -t "$S" | grep -v '^\s*$' | tail -6 | cut -c1-140
    local i=0
    while IFS= read -r turn; do
        i=$((i + 1))
        say "${turn//PASTE_SESSIONS/$(paste_text)}"
        echo "  turn $i: $(rows) requests"
    done < "$HERE/turns.txt"
    echo "  $S before /clear: $(rows) requests"
    slash /clear
    say "After the clear: in one sentence, what is a prefix snapshot?"
    say "And in one more sentence: when should it be evicted?"
    echo "  $S end: $(rows) requests"
    $T capture-pane -p -t "$S" | grep -v '^\s*$' | tail -8 | cut -c1-140
    stop_all
}

cc_launcher() {
    local H=$W/cc-home WD=$W/work-cc KEY=sk-ant-dummy-000000000000000000000000clearprobe
    mkdir -p "$H" "$WD"
    cat > "$H/.claude.json" <<EOF
{"hasCompletedOnboarding": true, "theme": "dark", "numStartups": 5, "customApiKeyResponses": {"approved": ["${KEY: -20}"], "rejected": []},
 "projects": {"$WD": {"hasTrustDialogAccepted": true, "allowedTools": []}}, "bypassPermissionsModeAccepted": true}
EOF
    cat > "$W/cc-run.sh" <<EOF
#!/bin/bash
cd $WD
exec env -i HOME=$H PATH=$SPATH TERM=xterm-256color USER=$USER ANTHROPIC_BASE_URL=http://127.0.0.1:$PORT ANTHROPIC_API_KEY=$KEY CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 DISABLE_AUTOUPDATER=1 claude
EOF
    chmod +x "$W/cc-run.sh"; echo "$W/cc-run.sh"
}

omp_launcher() {
    local H=$W/omp-home WD=$W/work-omp
    mkdir -p "$H/.omp/agent" "$WD"
    (cd ~/.omp/agent && cp -r config.yml models.yml settings.json trust.json agents prompts "$H/.omp/agent/")
    sed -i "s#http://127.0.0.1:1919/v1#http://127.0.0.1:$PORT/v1#" "$H/.omp/agent/models.yml"
    grep -q "127.0.0.1:$PORT/v1" "$H/.omp/agent/models.yml" || { echo "omp models.yml not redirected" >&2; exit 1; }
    cat > "$W/omp-run.sh" <<EOF
#!/bin/bash
cd $WD
exec env -i HOME=$H PATH=$SPATH TERM=xterm-256color USER=$USER omp --model freetoken/nemotron-3.5-lightning --no-lsp
EOF
    chmod +x "$W/omp-run.sh"; echo "$W/omp-run.sh"
}

codex_launcher() {
    local C=$W/codex-home U=$W/codex-userhome WD=$W/work-codex
    mkdir -p "$C" "$U" "$WD"
    cp ~/.codex/models_cache.json "$C/" 2>/dev/null
    python3 - "$C/config.toml" "$WD" "$PORT" <<'EOF'
import re, sys
out_path, wd, port = sys.argv[1:]
out, skip = [], False
for line in open("/home/lucas/.codex/config.toml").read().splitlines():
    m = re.match(r"\[(.+)\]", line)
    if m:
        skip = m.group(1).startswith(("hooks", "plugins", "marketplaces", "mcp_servers", "projects", "model_providers"))
    if not skip and not re.match(r"\s*(model|model_provider)\s*=", line):
        out.append(line)
head = ['model = "nemotron-3.5-lightning"', 'model_provider = "stub_responses"']
tail = f'''
[model_providers.stub_responses]
name = "stub-responses"
base_url = "http://127.0.0.1:{port}/v1"
env_key = "STUB_API_KEY"
wire_api = "responses"

[projects."{wd}"]
trust_level = "trusted"
'''
open(out_path, "w").write("\n".join(head + out) + "\n" + tail)
EOF
    cat > "$W/codex-run.sh" <<EOF
#!/bin/bash
cd $WD
exec env -i HOME=$U CODEX_HOME=$C PATH=$SPATH TERM=xterm-256color USER=$USER STUB_API_KEY=sk-dummy-clearprobe codex
EOF
    chmod +x "$W/codex-run.sh"; echo "$W/codex-run.sh"
}

$T kill-server 2>/dev/null
for c in $CLIENTS; do
    echo "== $c"
    case $c in
        claude-code) run_client claude-code "$(cc_launcher)" ;;
        omp) run_client omp "$(omp_launcher)" ;;
        codex) run_client codex "$(codex_launcher)" ;;
    esac
done
$T kill-server 2>/dev/null
echo "captures in $W/raw"
