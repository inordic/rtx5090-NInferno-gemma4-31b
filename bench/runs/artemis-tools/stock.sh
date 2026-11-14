#!/bin/bash
# stock.sh: the picture test (imgcalltest.py) on stock Gemma (LilaRest NVFP4-turbo, the pre-Artemis production model),
# same binary, flags and template as production, on 127.0.0.1 only. Production is back whatever happens (trap).
exec 9> /root/gbench/imgcall/stock.lock; flock -n 9 || { echo "already running"; exit 1; }
cd /root/gbench/imgcall
step() { echo "=== $(date -u +%T) $*"; }
back() {
    trap - EXIT
    step "production back"; /root/gbench/n10/srv10.sh stop > /dev/null 2>&1
    pgrep -f "apps[^ ]*/ninferno-serve" > /dev/null && { pkill -9 -f "apps[^ ]*/ninferno-serve"; sleep 3; }
    systemctl start gemma4-31b-fast
    for i in $(seq 1 90); do curl -sf -m 5 http://127.0.0.1:8080/v1/models > /dev/null && break; sleep 5; done
    step "production $(systemctl is-active gemma4-31b-fast)"
    (cd /root/gbench && timeout 300 python3 verify.py --model gemma4-31b-fast > imgcall/stock_verify.out 2>&1); tail -1 stock_verify.out
    (cd /root/gbench && timeout 120 python3 textcalltest.py > imgcall/stock_textcall.out 2>&1); tail -1 stock_textcall.out
    step "STOCK_END"
}
trap back EXIT
# wait (at most 3 min) until no request is in flight
for i in $(seq 1 36); do
    last=$(journalctl -u gemma4-31b-fast --since "-30 min" -o cat | grep -oE "req#[0-9]+ started" | tail -1 | cut -d' ' -f1)
    [ -z "$last" ] && break
    journalctl -u gemma4-31b-fast --since "-30 min" -o cat | grep -qE "$last (done|failed|cancelled)" && break
    sleep 5
done
step "stopping production (last request $last)"
systemctl stop gemma4-31b-fast; sleep 3
export NINFERNO_GEMMA4_SHARED_VALUES=2 NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=1 NINFERNO_GEMMA4_EMBED_HOST=1 NINFERNO_GEMMA4_WIDE_PREFILL_MIN_T=64 NINFERNO_GEMMA4_SKIP_LAST_LAYER=1
SRV_BIN=/opt/NInferno-up/build/apps-up4/ninferno-serve /root/gbench/n10/srv10.sh start stock-img /opt/models/dl2/turbo-qkv-vision-nvfp4.r31.ninferno \
    --vision --max-context 203136 --prefill-chunk 1280 --kv-dtype fp8 --sliding-kv-dtype fp8 --spec mtp \
    --drafter /opt/models/dl2/assistant.nvfp4.ninferno --draft-tokens 7 --chat-template /opt/models/templates/gemma4-agent.jinja \
    --auto-tools-shared-prefix --auto-shared-prefix-requested > srv-stock.out 2>&1 || { step "stock server DIED"; tail -5 srv-stock.out; exit 1; }
grep -q "listening on" /root/gbench/logs/stock-img.log || { step "stock never listened"; exit 1; }
step "stock listening"
timeout 600 python3 imgcalltest.py stock 20 plain,rule,echo default
timeout 300 python3 imgcalltest.py stock-t07 20 plain,rule 0.7
grep -c "tool markup returned as text" /root/gbench/logs/stock-img.log | sed 's/^/stock malformed-markup warnings: /'
