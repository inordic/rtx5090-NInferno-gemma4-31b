#!/bin/bash
# n2c_uncensored: night 2, batch C - the uncensored candidates against the production model:
# refusal spot-check, speed, needles and the quality suite. Huihui QAT-abliterated NVFP4 on the fast
# config, Heretic QAT Q4 on the 262K config (both on apps-final5 without the new levers, so they
# compare with production as it runs). After batch B, behind a 5-minute idle gate.
cd /root/gbench
step() { echo "=== $(date +%T) $*"; }
until grep -q "N2D DONE" n2d_nvfp4alt.log 2>/dev/null; do sleep 30; done
step "refusal baseline: the production unit (censored)"
python3 refusalcheck.py --label "prod gemma4-31b-fast (QAT-NVFP4)"
./idle_gate_any.sh 5
step "idle: taking the GPU"
systemctl stop gemma4-31b-fast
for i in $(seq 1 90); do curl -sf http://127.0.0.1:8080/v1/models > /dev/null || break; sleep 1; done
DQ=/opt/models/dl2/qat-assistant.q40qat.ninferno
export NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=1 NINFERNO_GEMMA4_SHARED_VALUES=2
unc() {  # unc <label> <artifact> <ctx> <chunk> <needle lengths>
    local label=$1 art=$2 ctx=$3 ch=$4 nl=$5
    step "$label: speed, refusals, needles"
    ./srv_final5.sh start U-$label $art --max-context $ctx --prefill-chunk $ch --kv-dtype fp8 --spec mtp --drafter $DQ --draft-tokens 7 && \
        python3 refusalcheck.py --label "U $label" && \
        python3 gbench.py --label "U $label ctx$ctx chunk$ch K7" --pp 2048,8192 --reps 1 && \
        python3 needle.py --label "U $label needles" --lengths $nl --depths 0.5
    ./srv.sh stop
    step "$label: quality (4 lanes)"
    ./srv_final5.sh start qU-$label $art --max-context 4096 --max-concurrency 4 --kv-dtype fp8 --spec mtp --drafter $DQ --draft-tokens 7 && \
        python3 geval.py --label "unc-$label" --suites gsm8k,mmlupro,mmlu,humaneval --conc 4
    ./srv.sh stop
}
unc huihui-nvfp4 /opt/models/dl3/huihui-qat-abl-nvfp4.r31.ninferno 155000 1024 32000,72000,140000
unc heretic-q4 /opt/models/dl2/heretic-qat-q40qat.ninferno 262144 512 32000,140000,200000
step "restoring the fast unit"
./deploy_fast.sh
step "N2C DONE"
