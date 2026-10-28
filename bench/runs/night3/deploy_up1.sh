#!/bin/bash
# deploy_up1.sh: the fast unit on apps-up1 (patches 01-16 on upstream 8359583) with patch 16's two
# cache switches and NInferno's default host cache (night 3 report section 4). Keeps the old unit as
# .bak, restarts when no request is in flight, waits until 8080 answers.
set -e
U=/etc/systemd/system/gemma4-31b-fast.service
BAK=$U.bak-$(date +%Y%m%d-%H%M%S)
NEW='ExecStart=/opt/NInferno-up/build/apps-up1/ninferno-serve /opt/models/dl2/turbo-qkv-vision-nvfp4.r31.ninferno --host 0.0.0.0 --port 8080 --model-id gemma4-31b-fast --vision --max-context 202112 --prefill-chunk 1024 --kv-dtype fp8 --sliding-kv-dtype fp8 --spec mtp --drafter /opt/models/dl2/assistant.q8full.ninferno --draft-tokens 7 --chat-template /opt/models/templates/gemma4-agent.jinja --auto-tools-shared-prefix --auto-shared-prefix-requested --host-state-slots 18 --host-kv-mib 8192'
cp -p $U $BAK
echo "backup: $BAK"
grep -q '^ExecStart=/opt/NInferno/build/apps-final9/ninferno-serve ' $U
python3 - "$U" "$NEW" <<'EOF'
import sys
path, new = sys.argv[1], sys.argv[2]
lines = open(path).read().splitlines()
lines = [new if l.startswith("ExecStart=") else l for l in lines]
open(path, "w").write("\n".join(lines) + "\n")
EOF
systemctl daemon-reload
diff $BAK $U || true

# No request in flight: the last req# line in the journal must be a "done" (or none at all),
# and the unit quiet for 20 s.
while :; do
    last=$(journalctl -u gemma4-31b-fast --since "-30 min" --no-pager -o cat | grep -E "req#[0-9]+ (started|done|failed|cancel)" | tail -1)
    recent=$(journalctl -u gemma4-31b-fast --since "-20 s" --no-pager -o cat | grep -c "req#" || true)
    if ! echo "$last" | grep -q " started " && [ "$recent" = 0 ]; then break; fi
    echo "$(date +%T) waiting: request in flight"; sleep 5
done
echo "$(date -u +%T) UTC restarting (last request line: ${last:0:80})"
systemctl restart gemma4-31b-fast
t0=$(date +%s)
for i in $(seq 1 300); do curl -sf http://127.0.0.1:8080/v1/models > /dev/null && break; sleep 1; done
echo "answering after $(( $(date +%s) - t0 )) s: $(systemctl is-active gemma4-31b-fast)"
journalctl -u gemma4-31b-fast --since "-6 min" --no-pager -o cat | grep -E "context cache|plan \| rung|listening|Engine failed" | tail -6
