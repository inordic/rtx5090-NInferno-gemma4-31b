#!/bin/bash
# Swap production's MTP drafter to the assistant retrained on Artemis (night 9, epoch 2). Everything else stays:
# unit, port, model id gemma4-31b-fast, binary apps-up4, weights, flags, K=7, template, host cache.
# Every check is enforced; any failure rolls back at once.
U=/etc/systemd/system/gemma4-31b-fast.service
B=$U.bak-$(date -u +%Y%m%d-%H%M%S)
OLD=/opt/models/dl2/assistant.nvfp4.ninferno
SRC=/opt/models/n9/assistant-artemis-ep2.nvfp4.ninferno
NEW=/opt/models/artemis/assistant-artemis-n9.nvfp4.ninferno
SIZE=296346628
cd /root/gbench; mkdir -p n9/deploy
log() { echo "$(date -u +%H:%M:%S) $*"; }
rollback() {
    log "ROLLBACK: $1"; cp $B $U; systemctl daemon-reload; systemctl restart gemma4-31b-fast
    for i in $(seq 1 150); do curl -sf -m 3 http://127.0.0.1:8080/v1/models >/dev/null && { log "rolled back, serving"; echo DEPLOY_ROLLED_BACK; exit 1; }; sleep 1; done
    log "rolled back, NOT answering yet"; echo DEPLOY_ROLLED_BACK; exit 1
}
# a stable home (hard links: a later cleanup of n9 cannot pull the file away)
[ -e $NEW ] || ln $SRC $NEW
[ -e $NEW.conversion.json ] || ln $SRC.conversion.json $NEW.conversion.json
[ "$(stat -c %s $NEW)" = "$SIZE" ] || { log "drafter size wrong"; echo DEPLOY_ABORTED; exit 1; }
log "drafter sha256 $(sha256sum $NEW | cut -c1-16)"
cp $U $B; log "backup $B"; echo $B > n9/deploy/backup_path
grep -q -- "--drafter $OLD " $U || { log "unit not as expected"; echo DEPLOY_ABORTED; exit 1; }
sed -i "s#--drafter $OLD #--drafter $NEW #" $U
sed -i 's#^Description=.*#Description=NInferno Gemma 4 31B Artemis v1.2 (imw NVFP4, Q/K/V packed) + vision + MTP drafter retrained on Artemis K7, K-only FP8 KV, FP8 rings, 203K - fast#' $U
grep -c -- "--drafter $NEW " $U | grep -q 1 || rollback "sed did not apply"
for i in $(seq 1 480); do
    a=$(curl -s -m 5 http://127.0.0.1:8080/health | python3 -c "import json,sys; print(json.load(sys.stdin)['active_requests'])" 2>/dev/null)
    n=$(journalctl -u gemma4-31b-fast -o cat --since "-90s" | grep -c " started | ")
    [ "$a" = "0" ] && [ "$n" = "0" ] && break
    sleep 5
done
[ "$a" = "0" ] && [ "$n" = "0" ] || { cp $B $U; systemctl daemon-reload; log "no quiet moment in 40 min, unit restored"; echo DEPLOY_ABORTED; exit 1; }
systemctl daemon-reload; T0=$(date -u "+%Y-%m-%d %H:%M:%S"); echo "$T0" > n9/deploy/deploy_time
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
PID=$(systemctl show -p MainPID --value gemma4-31b-fast)
tr '\0' ' ' < /proc/$PID/cmdline | grep -q -- "--drafter $NEW " || rollback "running server does not use the new drafter"
journalctl -u gemma4-31b-fast -o cat --since "$T0" | grep -q "engine ready | artemis-v1.2-imw-qkv-vision-nvfp4.r31 " || rollback "engine did not load Artemis"
journalctl -u gemma4-31b-fast -o cat --since "$T0" | grep -qE "host 18 states, 12\.0+ GiB KV .*shared 10" || rollback "cache line not as asked"
journalctl -u gemma4-31b-fast -o cat --since "$T0" | grep -q "capacity | KV 203,136 tokens" || rollback "context changed"
journalctl -u gemma4-31b-fast -o cat --since "$T0" | grep -q "serves mtp K=7" || rollback "drafter not serving"
timeout 180 python3 verify.py --model gemma4-31b-fast > n9/deploy/verify.out 2>&1 || rollback "verify failed: $(tail -1 n9/deploy/verify.out)"; tail -4 n9/deploy/verify.out | cut -c1-150
timeout 120 python3 textcalltest.py > n9/deploy/textcall.out 2>&1; tail -1 n9/deploy/textcall.out
grep -q "leaks 0" n9/deploy/textcall.out || rollback "tool markup leaked as text"
timeout 300 python3 visiontest.py --label "N9 drafter live" > n9/deploy/vision.out 2>&1; tail -1 n9/deploy/vision.out
grep -q "7/7" n9/deploy/vision.out || rollback "vision below 7/7"
timeout 360 python3 -I n8/companionfit.py N9-drafter-live > n9/deploy/companion.out 2>&1 || rollback "companion set: $(tail -1 n9/deploy/companion.out | cut -c1-200)"
tail -1 n9/deploy/companion.out | cut -c1-120
# the agent loop: transport errors and markup as text are enforced; the new counters (unknown tools, invalid
# arguments, tasks done) are recorded as the first baseline, not enforced yet
timeout 600 python3 agenttest.py --label "N9 drafter live" --runs 1 > n9/deploy/agent.out 2>&1; tail -1 n9/deploy/agent.out | cut -c1-260
grep "^AGENT" n9/deploy/agent.out | grep -q '"errors": 0' || rollback "agent loop errors"
grep "^AGENT" n9/deploy/agent.out | grep -q '"markup_as_text": 0' || rollback "agent loop markup as text"
free -m | sed -n 2p
echo DEPLOY_OK
