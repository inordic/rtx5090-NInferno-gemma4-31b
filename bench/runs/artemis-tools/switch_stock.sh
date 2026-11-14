#!/bin/bash
# switch_stock.sh (14 Nov 2026, on the owner's OK): the fast unit back to stock
# Gemma 4 31B (LilaRest NVFP4-turbo + Google's assistant drafter, unit .bak-20261113-120150), same binary and flags.
# The Artemis unit is kept as .bak-<now> (the rollback). If the stock server doesn't answer, Artemis comes back.
exec 9> /root/gbench/imgcall/switch.lock; flock -n 9 || { echo "already running"; exit 1; }
U=/etc/systemd/system/gemma4-31b-fast.service
step() { echo "=== $(date -u +%T) $*"; }
for i in $(seq 1 36); do   # at most 3 min for a request in flight to finish
    last=$(journalctl -u gemma4-31b-fast --since "-30 min" -o cat | grep -oE "req#[0-9]+ started" | tail -1 | cut -d' ' -f1)
    [ -z "$last" ] && break
    journalctl -u gemma4-31b-fast --since "-30 min" -o cat | grep -qE "$last (done|failed|cancelled)" && break
    sleep 5
done
B=$U.bak-$(date -u +%Y%m%d-%H%M%S)
cp -p $U $B; step "Artemis unit saved as $B (last request $last)"
cp -p $U.bak-20261113-120150 $U
systemctl daemon-reload; step "restarting on stock"; systemctl restart gemma4-31b-fast
ok=0; for i in $(seq 1 90); do curl -sf -m 5 http://127.0.0.1:8080/v1/models > /dev/null && { ok=1; break; }; sleep 5; done
if [ $ok = 0 ]; then
    step "stock did not answer: restoring Artemis"; cp -p $B $U; systemctl daemon-reload; systemctl restart gemma4-31b-fast
    for i in $(seq 1 90); do curl -sf -m 5 http://127.0.0.1:8080/v1/models > /dev/null && break; sleep 5; done
    step "production $(systemctl is-active gemma4-31b-fast) (Artemis)"; step "SWITCH_END: FAILED"; exit 1
fi
step "production $(systemctl is-active gemma4-31b-fast)"
journalctl -u gemma4-31b-fast --since "-5 min" -o cat | grep -E "sampling \|" | tail -1 | sed -E "s/^.*INFO +//"
pgrep -af "apps[^ ]*/ninferno-serve" | grep -oE "/opt/models/[^ ]+" | tr '\n' ' '; echo
cd /root/gbench
timeout 300 python3 verify.py --model gemma4-31b-fast > imgcall/switch_verify.out 2>&1; tail -1 imgcall/switch_verify.out
timeout 120 python3 textcalltest.py > imgcall/switch_textcall.out 2>&1; tail -1 imgcall/switch_textcall.out
timeout 600 python3 imgcall/imgcalltest.py stock-prod 10 plain,rule,echo default > imgcall/switch_imgcall.out 2>&1; echo "imgcall exit $?"; tail -1 imgcall/switch_imgcall.out
step "SWITCH_END: done"
