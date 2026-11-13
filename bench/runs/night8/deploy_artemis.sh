#!/bin/bash
# Swap production's weights to Artemis v1.2 (imw NVFP4). Everything else stays: unit, port, model id
# gemma4-31b-fast, binary apps-up4, flags, drafter, template, host cache. Any failure rolls back at once.
U=/etc/systemd/system/gemma4-31b-fast.service
B=$U.bak-$(date -u +%Y%m%d-%H%M%S)
OLD=/opt/models/dl2/turbo-qkv-vision-nvfp4.r31.ninferno
SRC=/opt/models/n8/A2-imw.ninferno
NEW=/opt/models/artemis/artemis-v1.2-imw-qkv-vision-nvfp4.r31.ninferno
SIZE=20259164164
cd /root/gbench; mkdir -p n9
log() { echo "$(date -u +%H:%M:%S) $*"; }
rollback() {
    log "ROLLBACK: $1"; cp $B $U; systemctl daemon-reload; systemctl restart gemma4-31b-fast
    for i in $(seq 1 150); do curl -sf -m 3 http://127.0.0.1:8080/v1/models >/dev/null && { log "rolled back, serving"; echo DEPLOY_ROLLED_BACK; exit 1; }; sleep 1; done
    log "rolled back, NOT answering yet"; echo DEPLOY_ROLLED_BACK; exit 1
}
# a stable home for the weights (hard links: a later cleanup of n8 cannot pull them away)
mkdir -p /opt/models/artemis
[ -e $NEW ] || ln $SRC $NEW
[ -e $NEW.conversion.json ] || ln $SRC.conversion.json $NEW.conversion.json
[ "$(stat -c %s $NEW)" = "$SIZE" ] || { log "artifact size wrong"; echo DEPLOY_ABORTED; exit 1; }
cp $U $B; log "backup $B"; echo $B > n9/backup_path
grep -q -- "ninferno-serve $OLD " $U || { log "unit not as expected"; echo DEPLOY_ABORTED; exit 1; }
sed -i "s#ninferno-serve $OLD #ninferno-serve $NEW #" $U
sed -i 's#^Description=.*#Description=NInferno Gemma 4 31B Artemis v1.2 (imw NVFP4, Q/K/V packed) + vision + MTP drafter K7, K-only FP8 KV, FP8 rings, 203K - fast#' $U
grep -c "$NEW" $U | grep -q 1 || rollback "sed did not apply"
for i in $(seq 1 480); do
    a=$(curl -s -m 5 http://127.0.0.1:8080/health | python3 -c "import json,sys; print(json.load(sys.stdin)['active_requests'])" 2>/dev/null)
    n=$(journalctl -u gemma4-31b-fast -o cat --since "-90s" | grep -c " started | ")
    [ "$a" = "0" ] && [ "$n" = "0" ] && break
    sleep 5
done
[ "$a" = "0" ] && [ "$n" = "0" ] || { cp $B $U; systemctl daemon-reload; log "no quiet moment in 40 min, unit restored"; echo DEPLOY_ABORTED; exit 1; }
systemctl daemon-reload; T0=$(date -u "+%Y-%m-%d %H:%M:%S"); echo "$T0" > n9/deploy_time
log "quiet, restarting"
systemctl restart gemma4-31b-fast
for i in $(seq 1 150); do
    journalctl -u gemma4-31b-fast -o cat --since "$T0" | grep -q "FATAL" && rollback "$(journalctl -u gemma4-31b-fast -o cat --since "$T0" | grep FATAL | head -1 | cut -c1-200)"
    journalctl -u gemma4-31b-fast -o cat --since "$T0" | grep -qE "Failed with result|status=[1-9]" && rollback "the server exited"
    curl -sf -m 3 http://127.0.0.1:8080/v1/models >/dev/null && break
    sleep 1
done
curl -sf -m 3 http://127.0.0.1:8080/v1/models >/dev/null || rollback "no answer within 150 s"
log "answering"; journalctl -u gemma4-31b-fast -o cat --since "$T0" | grep -E "engine ready|pinned|context cache|capacity \||speculation \|" | cut -c1-200
curl -s http://127.0.0.1:8080/v1/models | python3 -c "
import json,sys; m=json.load(sys.stdin)['data'][0]; print('MODEL', m['id'], m['meta']['size'], m['max_model_len'], m['meta']['speculation']['config'])
sys.exit(0 if m['id']=='gemma4-31b-fast' and m['max_model_len']==203136 else 1)" || rollback "model id not as asked"
journalctl -u gemma4-31b-fast -o cat --since "$T0" | grep -q "engine ready | artemis-v1.2-imw-qkv-vision-nvfp4.r31 " || rollback "engine did not load Artemis"
journalctl -u gemma4-31b-fast -o cat --since "$T0" | grep -qE "host 18 states, 12\.0+ GiB KV .*shared 10" || rollback "cache line not as asked"
journalctl -u gemma4-31b-fast -o cat --since "$T0" | grep -q "capacity | KV 203,136 tokens" || rollback "context changed"
journalctl -u gemma4-31b-fast -o cat --since "$T0" | grep -q "serves mtp K=7" || rollback "drafter not serving"
timeout 120 python3 verify.py > n9/verify.out 2>&1 || rollback "verify failed"; tail -4 n9/verify.out | cut -c1-150
timeout 120 python3 textcalltest.py > n9/textcall.out 2>&1; tail -1 n9/textcall.out
grep -q "leaks 0" n9/textcall.out || rollback "tool markup leaked as text"
timeout 300 python3 visiontest.py --label "N9 live" > n9/vision.out 2>&1; tail -1 n9/vision.out
grep -q "7/7" n9/vision.out || rollback "vision below 7/7"
timeout 120 python3 headprobe.py > n9/head.out 2>&1 || rollback "headprobe failed"; tail -3 n9/head.out | cut -c1-150
timeout 300 python3 agenttest.py --label "N9 live" --runs 1 > n9/agent.out 2>&1 || rollback "agent loop failed"; tail -1 n9/agent.out | cut -c1-200
grep -q '"errors": 0' n9/agent.out || rollback "agent loop errors"
grep -q '"markup_as_text": 0' n9/agent.out || rollback "agent loop markup as text"
timeout 300 python3 -I n8/companionfit.py A2-live > n9/companionfit.out 2>&1; tail -1 n9/companionfit.out | cut -c1-300
free -m | sed -n 2p; grep MemAvailable /proc/meminfo
echo DEPLOY_OK
