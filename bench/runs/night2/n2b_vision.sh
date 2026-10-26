#!/bin/bash
# n2b_vision: night 2, batch B - vision on both profiles, with both context levers (FP8 sliding
# rings, host-mapped token table): the planner's ceiling with --vision, then the image test, speed
# and a needle at that context. After batch A, behind a 5-minute idle gate; the fast unit comes back.
cd /root/gbench
step() { echo "=== $(date +%T) $*"; }
until grep -q "N2E DONE" n2e_ngram.log 2>/dev/null; do sleep 30; done
./idle_gate_any.sh 5
step "idle: taking the GPU"
systemctl stop gemma4-31b-fast
for i in $(seq 1 90); do curl -sf http://127.0.0.1:8080/v1/models > /dev/null || break; sleep 1; done
S=/opt/NInferno/build/apps-final5/ninferno-serve
DQ=/opt/models/dl2/qat-assistant.q40qat.ninferno
export NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=1 NINFERNO_GEMMA4_SHARED_VALUES=2 NINFERNO_GEMMA4_EMBED_HOST=1
vis() {  # vis <label> <artifact> <chunk>
    local label=$1 art=$2 ch=$3
    CTX=$($S $art --plan --vision --kv-dtype fp8 --sliding-kv-dtype fp8 --max-context 262144 --prefill-chunk $ch \
          --spec mtp --drafter $DQ --draft-tokens 7 2>&1 | grep -oE "with up to [0-9]+ tokens" | head -1 | grep -oE "[0-9]+")
    [ -z "$CTX" ] && CTX=100000
    [ "$CTX" -gt 262144 ] && CTX=262144
    step "$label: --vision ceiling $CTX at chunk $ch"
    ./srv_final5.sh start V-$label $art --vision --max-context $CTX --prefill-chunk $ch --kv-dtype fp8 --sliding-kv-dtype fp8 \
        --spec mtp --drafter $DQ --draft-tokens 7 && \
        grep -hE "plan \| rung|capacity \| KV" logs/V-$label.log | sed -E "s/^.*INFO +//" | cut -c1-200 && \
        python3 visiontest.py --label "V $label ctx$CTX" && \
        python3 gbench.py --label "V $label vision ctx$CTX chunk$ch K7" --pp 2048,8192,32768 --reps 1 && \
        python3 needle.py --label "V $label needle" --lengths $(( CTX * 9 / 10 )) --depths 0.5
    ./srv.sh stop
}
vis nvfp4 /opt/models/dl2/qat-nvfp4-vision.r31.ninferno 1024
vis q4 /opt/models/dl/qat-ud-q4-vision.ninferno 512
step "restoring the fast unit"
./deploy_fast.sh
step "N2B DONE"
