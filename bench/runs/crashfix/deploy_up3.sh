#!/bin/bash
# deploy_up3.sh: the fast unit on apps-up3 (kvshare-up3: patches 01-16 on upstream 8359583, upstream
# #226 #246 #217 #234, patches 17 and 18; night 4 report section 5). Only the binary changes. Keeps
# the old unit as .bak, restarts when no request is in flight, waits until 8080 answers.
set -e
U=/etc/systemd/system/gemma4-31b-fast.service
BAK=$U.bak-$(date +%Y%m%d-%H%M%S)
cp -p $U $BAK
echo "backup: $BAK"
grep -q '^ExecStart=/opt/NInferno-up/build/apps-up1/ninferno-serve ' $U
sed -i 's#^ExecStart=/opt/NInferno-up/build/apps-up1/ninferno-serve #ExecStart=/opt/NInferno-up/build/apps-up3/ninferno-serve #' $U
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
