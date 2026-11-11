#!/bin/bash
# Second try within the engine's pin cap (60% of 48 GiB = 28.8 GiB): Host KV 8 -> 12 GiB, shared prefixes 7 -> 10,
# Host states stay 18 (16.0 + 12 = 28.0 GiB). Fails fast: any FATAL or restart rolls back at once.
U=/etc/systemd/system/gemma4-31b-fast.service
B=$U.bak-$(date -u +%Y%m%d-%H%M%S)
cd /root/gbench
log() { echo "$(date -u +%H:%M:%S) $*"; }
rollback() {
    log "ROLLBACK: $1"; cp $B $U; systemctl daemon-reload; systemctl restart gemma4-31b-fast
    for i in $(seq 1 120); do curl -sf -m 3 http://127.0.0.1:8080/v1/models >/dev/null && { log "rolled back, serving"; echo DEPLOY_ROLLED_BACK; exit 1; }; sleep 1; done
    log "rolled back, NOT answering yet"; echo DEPLOY_ROLLED_BACK; exit 1
}
cp $U $B; log "backup $B"; echo $B > n6/backup_path
grep -q -- "--host-state-slots 18 --host-kv-mib 8192$" $U || { log "unit not as expected"; echo DEPLOY_ABORTED; exit 1; }
sed -i 's/--host-state-slots 18 --host-kv-mib 8192$/--host-state-slots 18 --host-kv-mib 12288 --max-shared-prefixes 10/' $U
for i in $(seq 1 480); do
    a=$(curl -s -m 5 http://127.0.0.1:8080/health | python3 -c "import json,sys; print(json.load(sys.stdin)['active_requests'])" 2>/dev/null)
    n=$(journalctl -u gemma4-31b-fast -o cat --since "-90s" | grep -c " started | ")
    [ "$a" = "0" ] && [ "$n" = "0" ] && break
    sleep 5
done
[ "$a" = "0" ] && [ "$n" = "0" ] || { cp $B $U; log "no quiet moment in 40 min, unit restored"; echo DEPLOY_ABORTED; exit 1; }
systemctl daemon-reload; T0=$(date -u "+%Y-%m-%d %H:%M:%S"); echo "$T0" > n6/deploy_time
R0=$(systemctl show gemma4-31b-fast -p NRestarts --value); log "quiet, restarting (restarts so far $R0)"
systemctl restart gemma4-31b-fast
for i in $(seq 1 150); do
    journalctl -u gemma4-31b-fast -o cat --since "$T0" | grep -q "FATAL" && rollback "$(journalctl -u gemma4-31b-fast -o cat --since "$T0" | grep FATAL | head -1 | cut -c1-200)"
    journalctl -u gemma4-31b-fast -o cat --since "$T0" | grep -qE "Failed with result|status=[1-9]" && rollback "the server exited"
    curl -sf -m 3 http://127.0.0.1:8080/v1/models >/dev/null && break
    sleep 1
done
curl -sf -m 3 http://127.0.0.1:8080/v1/models >/dev/null || rollback "no answer within 150 s"
log "answering"; journalctl -u gemma4-31b-fast -o cat --since "$T0" | grep -E "pinned|context cache|capacity \|" | cut -c1-200
journalctl -u gemma4-31b-fast -o cat --since "$T0" | grep -qE "host 18 states, 12\.0+ GiB KV .*shared 10" || rollback "cache line not as asked"
journalctl -u gemma4-31b-fast -o cat --since "$T0" | grep -q "capacity | KV 203,136 tokens" || rollback "context changed"
timeout 120 python3 verify.py > n6/verify.out 2>&1 || rollback "verify failed"; tail -4 n6/verify.out | cut -c1-150
timeout 120 python3 headprobe.py > n6/head.out 2>&1 || rollback "headprobe failed"; tail -3 n6/head.out | cut -c1-150
timeout 300 python3 agenttest.py --label "N6 cache" --runs 1 --steps 8 > n6/agent.out 2>&1 || rollback "agent loop failed"; tail -1 n6/agent.out | cut -c1-200
grep -q '"errors": 0' n6/agent.out || rollback "agent loop errors"
free -m | sed -n 2p; grep MemAvailable /proc/meminfo
echo DEPLOY_OK
