#!/bin/bash
# n2d_nvfp4alt: night 2, batch D - does another NVFP4 close the knowledge gap? LilaRest NVFP4-turbo
# (from Google's BF16 weights) and yasu-oh QAT NVFP4 (QAT weights, error-minimising recipe), on the
# production config: the quality suite, the exact 140K needle production misses, needles, and the
# drafter's acceptance (the QAT assistant; for the BF16-based model also Google's regular assistant).
cd /root/gbench
step() { echo "=== $(date +%T) $*"; }
until grep -q "N2B DONE" n2b_vision.log 2>/dev/null && grep -q "PREP3 DONE" prep_night2c.log 2>/dev/null; do sleep 30; done
./idle_gate_any.sh 5
step "idle: taking the GPU"
systemctl stop gemma4-31b-fast
for i in $(seq 1 90); do curl -sf http://127.0.0.1:8080/v1/models > /dev/null || break; sleep 1; done
DQ=/opt/models/dl2/qat-assistant.q40qat.ninferno
DR=/opt/models/dl2/assistant.q8full.ninferno
export NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=1 NINFERNO_GEMMA4_SHARED_VALUES=2
alt() {  # alt <label> <artifact>
    local label=$1 art=$2
    [ -f $art ] || { step "$label: artifact missing"; return; }
    step "$label: quality (4 lanes)"
    ./srv_final5.sh start qD-$label $art --max-context 4096 --max-concurrency 4 --kv-dtype fp8 --spec mtp --drafter $DQ --draft-tokens 7 && \
        python3 geval.py --label "alt-$label" --suites gsm8k,mmlupro,mmlu,humaneval --conc 4
    ./srv.sh stop
    step "$label: the production miss (140K, depth 0.5, first draw), needles, speed with the QAT drafter"
    ./srv_final5.sh start D-$label $art --max-context 155000 --prefill-chunk 1024 --kv-dtype fp8 --spec mtp --drafter $DQ --draft-tokens 7 && \
        python3 needle.py --label "D $label repro" --lengths 140000 --depths 0.5 && \
        python3 needle.py --label "D $label needles" --lengths 32000,72000,140000 --depths 0.1,0.9 && \
        python3 gbench.py --label "D $label QAT drafter K7" --pp 2048,8192 --reps 1
    ./srv.sh stop
}
alt turbo /opt/models/dl3/turbo-nvfp4.r31.ninferno
step "turbo: speed with Google's regular assistant"
./srv_final5.sh start D-turbo-asst /opt/models/dl3/turbo-nvfp4.r31.ninferno --max-context 155000 --prefill-chunk 1024 --kv-dtype fp8 \
    --spec mtp --drafter $DR --draft-tokens 7 && \
    python3 gbench.py --label "D turbo regular assistant K7" --pp 2048 --reps 1
./srv.sh stop
alt yasu /opt/models/dl3/yasu-qat-nvfp4.r31.ninferno
step "restoring the fast unit"
./deploy_fast.sh
step "N2D DONE"
