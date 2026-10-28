#!/bin/bash
# night3_c: after night3_a, behind the idle gate. The entitlement stress in agent mode (long
# tool-result loops into a full cache, the shape of the 2026-10-27 exit) on production's binary
# (apps-final9) and on patch 15 (apps-final10), at production cache sizes and a smaller cache.
cd /root/gbench
step() { echo "=== $(date +%T) $*"; }
until grep -q "N3A DONE" night3_a.out 2>/dev/null; do sleep 30; done
./idle_gate_any.sh 5
step "idle: taking the GPU"
systemctl stop gemma4-31b-fast
for i in $(seq 1 90); do curl -sf http://127.0.0.1:8080/v1/models > /dev/null || break; sleep 1; done
export NINFERNO_GEMMA4_SHARED_VALUES=2 NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=1 NINFERNO_GEMMA4_EMBED_HOST=1 NINFERNO_GEMMA4_WIDE_PREFILL_MIN_T=64
M=/opt/models/dl2/turbo-qkv-vision-nvfp4.r31.ninferno
DR=/opt/models/dl2/assistant.q8full.ninferno
PROD="--vision --max-context 202112 --prefill-chunk 1024 --kv-dtype fp8 --sliding-kv-dtype fp8 --spec mtp --drafter $DR --draft-tokens 7 --chat-template /opt/models/templates/gemma4-agent.jinja"
efail() { grep -hE "Engine failed" logs/$1.log | sed -E "s/^.*ninferno: //" | cut -c1-400 | head -2; }
kinds() { grep -E "req#[0-9]+ done" logs/$1.log | grep -oE "cache [0-9,]+ \([0-9.]+%(, [a-z ]+)?\)" | sed -E "s/cache [0-9,]+ \([0-9.]+%,? ?//; s/\)//" | sort | uniq -c | tr "\n" " "; echo; }
for b in final9 final10; do
    BIN=/opt/NInferno/build/apps-$b/ninferno-serve
    for v in "prodcache|" "small|--host-state-slots 6 --host-kv-mib 4096"; do
        n=${v%%|*}; extra=${v#*|}; name=S-$b-$n
        step "agent-mode stress on $b, $n cache"
        SRV_BIN=$BIN ./srv_any.sh start $name $M $PROD $extra > /dev/null && \
            python3 stress_entitlement.py --mode agent --rounds 300 --seed $([ $n = small ] && echo 4 || echo 3) > stress_$name.log 2>&1
        tail -1 stress_$name.log; efail $name; kinds $name
        ./srv.sh stop
    done
done
step "restoring the fast unit"
./deploy_fast.sh
step "N3C DONE"
