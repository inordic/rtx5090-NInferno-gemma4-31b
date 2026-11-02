#!/bin/bash
# cf_run: which upstream change makes apps-up2 exit. Each build replays night 4's A1 sequence at
# production flags (verify, textcall, vision, speed, turns, edit, agent loop, cache replay) behind
# the idle gate, the fast unit back after each. The sequence stops as soon as the server is gone.
# Order: cf-t234 (#234 alone), cf-t3 (#226 #246 #217); cf-up2 only when neither exits.
cd /root/gbench
step() { echo "=== $(date +%T) $*"; }
export NINFERNO_GEMMA4_SHARED_VALUES=2 NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=1
B=/opt/NInferno-up/build
LILA=/opt/models/dl2/turbo-qkv-vision-nvfp4.r31.ninferno
DR=/opt/models/dl2/assistant.q8full.ninferno
TPL=/opt/models/templates/gemma4-agent.jinja
PROD="--vision --max-context 202112 --prefill-chunk 1024 --kv-dtype fp8 --sliding-kv-dtype fp8 --spec mtp --drafter $DR --draft-tokens 7 --chat-template $TPL --auto-tools-shared-prefix --auto-shared-prefix-requested --host-state-slots 18 --host-kv-mib 8192"
mkdir -p cf logs

take_gpu() {
    ./idle_gate_any.sh 5
    step "$1: idle, taking the GPU"
    systemctl stop gemma4-31b-fast
    for i in $(seq 1 90); do curl -sf http://127.0.0.1:8080/v1/models > /dev/null || break; sleep 1; done
}
give_back() { ./srv.sh stop; step "restoring the fast unit"; ./deploy_fast.sh; }
alive() { pgrep -f "apps[^ ]*/ninferno-serve" > /dev/null && curl -sf -m 10 http://127.0.0.1:8080/v1/models > /dev/null; }
run() {  # name command...: the command's last lines, or nothing when the server is already gone
    alive || { echo "--- $1: server gone, sequence stopped"; return 1; }
    echo "--- $1"; shift; "$@" 2>&1 | tail -2
}
seq_a1() {
    local L=$1
    run verify python3 verify.py && run textcall python3 textcalltest.py \
        && run vision python3 visiontest.py --label "CF $L" \
        && run speed python3 gbench.py --label "CF $L" --pp 2048,8192,32768,65536 --reps 2 \
        && run turns python3 turnprobe.py http://127.0.0.1:8080 \
        && run edit python3 editbench.py --label "CF $L" \
        && run agent python3 agenttest.py --label "CF $L" --runs 1 \
        && run replay python3 cachereplay.py --label "CF $L" --slots 12
    if alive; then echo "SURVIVED $L"; else echo "EXITED $L"; fi
    grep -hE "Engine failed|FATAL" logs/$L.log | head -2 | cut -c1-700
}
one() {  # label binary-dir
    take_gpu $1
    NINFERNO_GEMMA4_EMBED_HOST=1 NINFERNO_GEMMA4_WIDE_PREFILL_MIN_T=64 SRV_BIN=$B/$2/ninferno-serve \
        ./srv_any.sh start $1 $LILA $PROD > cf/$1.txt 2>&1 && seq_a1 $1 >> cf/$1.txt 2>&1
    grep -hE "^(SURVIVED|EXITED)|underflow|DIED" cf/$1.txt | cut -c1-700
    give_back
}
one CF_t234 apps-cf-t234
one CF_t3 apps-cf-t3
if grep -q SURVIVED cf/CF_t234.txt && grep -q SURVIVED cf/CF_t3.txt; then one CF_up2 apps-cf-up2; fi
curl -sf http://127.0.0.1:8080/v1/models > /dev/null && step "fast unit answering: $(systemctl is-active gemma4-31b-fast)"
step "CF_RUN_DONE"
