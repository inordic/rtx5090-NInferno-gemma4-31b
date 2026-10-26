#!/bin/bash
# n2g_vision2: night 2, vision again after batch B failed on both profiles.
#  1. NVFP4: why an image request fails (apps-visdiag names the mismatch), probed at 32K with and
#     without the host token table and MTP; then the full run on the first setup that answers.
#  2. Q4: FP8 rings only. Its token table is tied to the head, so there is no copy to map to the host
#     (NINFERNO_GEMMA4_EMBED_HOST=1 refuses to start: "demanded both in device memory and mapped").
# After batch C, behind a 5-minute idle gate; the fast unit comes back after.
cd /root/gbench
step() { echo "=== $(date +%T) $*"; }
until grep -q "N2C DONE" n2c_uncensored.log 2>/dev/null; do sleep 30; done
./idle_gate_any.sh 5
step "idle: taking the GPU"
systemctl stop gemma4-31b-fast
for i in $(seq 1 90); do curl -sf http://127.0.0.1:8080/v1/models > /dev/null || break; sleep 1; done
S=/opt/NInferno/build/apps-visdiag/ninferno-serve
sed "s#^S=.*#S=$S#" srv.sh > srv_visdiag.sh; chmod +x srv_visdiag.sh
QNV=/opt/models/dl2/qat-nvfp4-vision.r31.ninferno
Q4V=/opt/models/dl/qat-ud-q4-vision.ninferno
DQ=/opt/models/dl2/qat-assistant.q40qat.ninferno
MTP="--spec mtp --drafter $DQ --draft-tokens 7"
export NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=1 NINFERNO_GEMMA4_SHARED_VALUES=2
probe() {  # probe <label> <embed host 0|1> <spec args...>
    local label=$1 eh=$2; shift 2
    step "probe $label"
    NINFERNO_GEMMA4_EMBED_HOST=$eh ./srv_visdiag.sh start VP-$label $QNV --vision --max-context 32768 --prefill-chunk 1024 \
        --kv-dtype fp8 --sliding-kv-dtype fp8 "$@" && \
        python3 visiontest.py --label "VP $label" 2>&1 | grep -E "^VISION|Error"
    grep -hE "Engine failed|ERROR" logs/VP-$label.log | sed -E "s/^.*(ninferno: |ERROR +)//" | cut -c1-400 | head -3
    ./srv.sh stop
}
probe nvfp4-host-mtp 1 $MTP
probe nvfp4-mtp 0 $MTP
probe nvfp4-plain 0 --spec none
full() {  # full <label> <artifact> <chunk> <embed host> <spec args...>
    local label=$1 art=$2 ch=$3 eh=$4; shift 4
    CTX=$(NINFERNO_GEMMA4_EMBED_HOST=$eh $S $art --plan --vision --kv-dtype fp8 --sliding-kv-dtype fp8 --max-context 262144 \
          --prefill-chunk $ch "$@" 2>&1 | grep -oE "with up to [0-9]+ tokens" | head -1 | grep -oE "[0-9]+")
    [ -z "$CTX" ] && CTX=262144
    step "$label: --vision at $CTX, chunk $ch, embed host $eh, $*"
    NINFERNO_GEMMA4_EMBED_HOST=$eh ./srv_visdiag.sh start V2-$label $art --vision --max-context $CTX --prefill-chunk $ch \
        --kv-dtype fp8 --sliding-kv-dtype fp8 "$@" && \
        python3 visiontest.py --label "V2 $label ctx$CTX" && \
        python3 gbench.py --label "V2 $label vision ctx$CTX chunk$ch" --pp 2048,8192,32768 --reps 1 && \
        python3 needle.py --label "V2 $label needle" --lengths $(( CTX * 9 / 10 )) --depths 0.5
    grep -hE "Engine failed|ERROR" logs/V2-$label.log | sed -E "s/^.*(ninferno: |ERROR +)//" | cut -c1-400 | head -3
    ./srv.sh stop
}
if grep -q "^VISION VP nvfp4-host-mtp [1-9]" n2g_vision2.log; then full nvfp4 $QNV 1024 1 $MTP
elif grep -q "^VISION VP nvfp4-mtp [1-9]" n2g_vision2.log; then full nvfp4 $QNV 1024 0 $MTP
elif grep -q "^VISION VP nvfp4-plain [1-9]" n2g_vision2.log; then full nvfp4-plain $QNV 1024 0 --spec none
else step "nvfp4: no setup answered an image"; fi
full q4 $Q4V 512 0 $MTP
step "restoring the fast unit"
./deploy_fast.sh
step "N2G DONE"
