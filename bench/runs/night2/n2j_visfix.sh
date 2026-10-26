#!/bin/bash
# n2j_visfix: night 2, batch J - vision with MTP after the fix (apps-visfix): the shared planner starts an
# image's span one token early under MTP and Gemma 4 took that token as a soft token (261 for 260).
# Fast profile with both levers at its --vision ceiling, and Q4 with FP8 rings (tied table) at its own.
cd /root/gbench
step() { echo "=== $(date +%T) $*"; }
until grep -q "N2I DONE" n2i_rerun.log 2>/dev/null; do sleep 30; done
./idle_gate_any.sh 5
step "idle: taking the GPU"
systemctl stop gemma4-31b-fast
for i in $(seq 1 90); do curl -sf http://127.0.0.1:8080/v1/models > /dev/null || break; sleep 1; done
S=/opt/NInferno/build/apps-visfix/ninferno-serve
sed "s#^S=.*#S=$S#" srv.sh > srv_visfix.sh; chmod +x srv_visfix.sh
QNV=/opt/models/dl2/qat-nvfp4-vision.r31.ninferno
Q4V=/opt/models/dl/qat-ud-q4-vision.ninferno
DQ=/opt/models/dl2/qat-assistant.q40qat.ninferno
MTP="--spec mtp --drafter $DQ --draft-tokens 7"
export NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=1 NINFERNO_GEMMA4_SHARED_VALUES=2
full() {  # full <label> <artifact> <chunk> <embed host>
    local label=$1 art=$2 ch=$3 eh=$4
    CTX=$(NINFERNO_GEMMA4_EMBED_HOST=$eh $S $art --plan --vision --kv-dtype fp8 --sliding-kv-dtype fp8 --max-context 262144 \
          --prefill-chunk $ch $MTP 2>&1 | grep -oE "with up to [0-9]+ tokens" | head -1 | grep -oE "[0-9]+")
    [ -z "$CTX" ] && CTX=262144
    step "$label: --vision + MTP at $CTX, chunk $ch, embed host $eh"
    NINFERNO_GEMMA4_EMBED_HOST=$eh ./srv_visfix.sh start J-$label $art --vision --max-context $CTX --prefill-chunk $ch \
        --kv-dtype fp8 --sliding-kv-dtype fp8 $MTP && \
        python3 visiontest.py --label "J $label ctx$CTX mtp" && \
        python3 gbench.py --label "J $label vision mtp ctx$CTX chunk$ch" --pp 2048,8192,32768 --reps 1 && \
        python3 turnprobe.py http://127.0.0.1:8080 && \
        python3 needle.py --label "J $label needle" --lengths $(( CTX * 9 / 10 )) --depths 0.5
    grep -hE "Engine failed|ERROR" logs/J-$label.log | sed -E "s/^.*(ninferno: |ERROR +)//" | cut -c1-400 | head -3
    ./srv.sh stop
}
full nvfp4 $QNV 1024 1
full q4 $Q4V 512 0
step "restoring the fast unit"
./deploy_fast.sh
step "N2J DONE"
