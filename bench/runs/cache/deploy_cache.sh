#!/bin/bash
# More room for the context cache: Host KV 8 -> 16 GiB, shared prefixes 7 -> 10, Host states 18 -> 21.
# Restarts only with no request in flight; rolls back by itself if the server or the checks fail.
U=/etc/systemd/system/gemma4-31b-fast.service
B=$U.bak-$(date -u +%Y%m%d-%H%M%S)
cd /root/gbench
log() { echo "$(date -u +%H:%M:%S) $*"; }
up() { for i in $(seq 1 300); do curl -sf -m 5 http://127.0.0.1:8080/v1/models >/dev/null && return 0; sleep 1; done; return 1; }
rollback() {
    log "ROLLBACK: $1"; cp $B $U; systemctl daemon-reload; systemctl restart gemma4-31b-fast
    up && log "rolled back, serving" || log "rolled back, NOT answering"
    echo DEPLOY_ROLLED_BACK; exit 1
}
cp $U $B; log "backup $B"; echo $B > n6/backup_path
grep -q -- "--host-state-slots 18 --host-kv-mib 8192$" $U || { log "unit not as expected"; echo DEPLOY_ABORTED; exit 1; }
sed -i 's/--host-state-slots 18 --host-kv-mib 8192$/--host-state-slots 21 --host-kv-mib 16384 --max-shared-prefixes 10/' $U
# quiet moment: nothing in flight and no request started in the last 90 s; give up after 40 min
for i in $(seq 1 480); do
    a=$(curl -s -m 5 http://127.0.0.1:8080/health | python3 -c "import json,sys; print(json.load(sys.stdin)['active_requests'])" 2>/dev/null)
    n=$(journalctl -u gemma4-31b-fast -o cat --since "-90s" | grep -c " started | ")
    [ "$a" = "0" ] && [ "$n" = "0" ] && break
    sleep 5
done
[ "$a" = "0" ] && [ "$n" = "0" ] || { cp $B $U; log "no quiet moment in 40 min, unit restored, nothing restarted"; echo DEPLOY_ABORTED; exit 1; }
log "quiet, restarting"; free -m | sed -n 2p
systemctl daemon-reload; T0=$(date -u "+%Y-%m-%d %H:%M:%S"); echo "$T0" > n6/deploy_time; systemctl restart gemma4-31b-fast
up || rollback "no answer within 300 s"
log "answering"; journalctl -u gemma4-31b-fast -o cat --since "$T0" | grep -E "pinned|context cache|capacity \|" | cut -c1-200
journalctl -u gemma4-31b-fast -o cat --since "$T0" | grep -q "host 21 states, 16.00 GiB KV" || rollback "cache line not as asked"
journalctl -u gemma4-31b-fast -o cat --since "$T0" | grep -q "capacity | KV 203,136 tokens" || rollback "context changed"
timeout 120 python3 verify.py > n6/verify.out 2>&1 || rollback "verify failed"; tail -4 n6/verify.out | cut -c1-150
timeout 120 python3 headprobe.py > n6/head.out 2>&1 || rollback "headprobe failed"; tail -3 n6/head.out | cut -c1-150
timeout 300 python3 agenttest.py --label "N6 cache" --runs 1 --steps 8 > n6/agent.out 2>&1 || rollback "agent loop failed"; tail -1 n6/agent.out | cut -c1-200
grep -q '"errors": 0' n6/agent.out || rollback "agent loop errors"
free -m | sed -n 2p; grep MemAvailable /proc/meminfo
echo DEPLOY_OK
