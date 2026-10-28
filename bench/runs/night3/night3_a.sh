#!/bin/bash
# night3_a: behind the idle gate. (1) the GPU unit test of the context store; (2) the entitlement
# stress on production's binary (apps-final9) and on patch 15 (apps-final10) with a small host cache;
# (3) regression checks of apps-final10 at production flags; (4) the cache replay over five context-
# cache settings on apps-final10; then the fast unit back.
cd /root/gbench
step() { echo "=== $(date +%T) $*"; }
sed "s#^S=.*#S=\${SRV_BIN:?}#" srv.sh > srv_any.sh; chmod +x srv_any.sh
./idle_gate_any.sh 5
step "idle: taking the GPU"
systemctl stop gemma4-31b-fast
for i in $(seq 1 90); do curl -sf http://127.0.0.1:8080/v1/models > /dev/null || break; sleep 1; done
export NINFERNO_GEMMA4_SHARED_VALUES=2 NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=1 NINFERNO_GEMMA4_EMBED_HOST=1 NINFERNO_GEMMA4_WIDE_PREFILL_MIN_T=64
M=/opt/models/dl2/turbo-qkv-vision-nvfp4.r31.ninferno
DR=/opt/models/dl2/assistant.q8full.ninferno
PROD="--vision --max-context 202112 --prefill-chunk 1024 --kv-dtype fp8 --sliding-kv-dtype fp8 --spec mtp --drafter $DR --draft-tokens 7 --chat-template /opt/models/templates/gemma4-agent.jinja"
F9=/opt/NInferno/build/apps-final9/ninferno-serve
F10=/opt/NInferno/build/apps-final10/ninferno-serve
SMALL="--host-state-slots 4 --host-kv-mib 3072"
efail() { grep -hE "Engine failed" logs/$1.log | sed -E "s/^.*ninferno: //" | cut -c1-400 | head -2; }

step "context store unit test (GPU)"
/opt/NInferno/build/tests/ninferno_qwen3_5_context_store_test > t_context_store_gpu.log 2>&1 && echo "PASS context_store" || { echo "FAIL context_store"; tail -5 t_context_store_gpu.log; }

step "A1 stress on apps-final9 (production binary), small host cache, seed 1"
SRV_BIN=$F9 ./srv_any.sh start A1-stress-final9 $M $PROD $SMALL && python3 stress_entitlement.py --rounds 40 --seed 1 > stress_A1.log 2>&1
tail -2 stress_A1.log; efail A1-stress-final9
./srv.sh stop
step "A2 stress on apps-final10 (patch 15), same settings, seed 1"
SRV_BIN=$F10 ./srv_any.sh start A2-stress-final10 $M $PROD $SMALL && python3 stress_entitlement.py --rounds 40 --seed 1 > stress_A2.log 2>&1
tail -2 stress_A2.log; efail A2-stress-final10
./srv.sh stop
step "A3 stress on apps-final10 with the new cache flags, seed 2"
SRV_BIN=$F10 ./srv_any.sh start A3-stress-final10-flags $M $PROD $SMALL --auto-tools-shared-prefix --auto-shared-prefix-requested && \
    python3 stress_entitlement.py --rounds 40 --seed 2 > stress_A3.log 2>&1
tail -2 stress_A3.log; efail A3-stress-final10-flags
./srv.sh stop

step "R apps-final10 at production flags: verify, tool calls, vision, agent loop"
SRV_BIN=$F10 ./srv_any.sh start R-final10 $M $PROD && \
    python3 verify.py 2>&1 | tail -4 && python3 textcalltest.py 2>&1 | tail -3 && \
    python3 visiontest.py --label "R final10" 2>&1 | tail -2 && python3 agenttest.py --label "R final10" --runs 1 2>&1 | tail -3
efail R-final10
./srv.sh stop

for cfg in "C0|" "C1|--auto-tools-shared-prefix" "C2|--auto-tools-shared-prefix --auto-shared-prefix-requested" \
           "C3|--auto-tools-shared-prefix --auto-shared-prefix-requested --host-state-slots 18 --host-kv-mib 8192" \
           "C4|--host-state-slots 18 --host-kv-mib 8192"; do
    name=${cfg%%|*}; extra=${cfg#*|}
    step "cache replay $name: $extra"
    SRV_BIN=$F10 ./srv_any.sh start $name-replay $M $PROD $extra | grep -E "context cache" ; \
        python3 cachereplay.py --label "$name $extra" --slots 12 > replay_$name.log 2>&1
    tail -1 replay_$name.log; efail $name-replay
    grep -E "req#[0-9]+ done" logs/$name-replay.log | grep -oE "cache [0-9,]+ \([0-9.]+%(, [a-z ]+)?\)" | sed -E "s/cache [0-9,]+ \([0-9.]+%,? ?//; s/\)//" | sort | uniq -c | tr "\n" " "; echo
    ./srv.sh stop
done
step "restoring the fast unit"
./deploy_fast.sh
step "N3A DONE"
