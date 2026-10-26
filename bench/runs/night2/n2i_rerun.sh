#!/bin/bash
# n2i_rerun: night 2, batch I - Huihui QAT-abliterated NVFP4 quality, which batch C skipped (the start
# check missed a live server; srv.sh now looks twice). After batch H, behind a 5-minute idle gate.
cd /root/gbench
step() { echo "=== $(date +%T) $*"; }
until grep -q "N2H DONE" n2h_turbo.log 2>/dev/null; do sleep 30; done
./idle_gate_any.sh 5
step "idle: taking the GPU"
systemctl stop gemma4-31b-fast
for i in $(seq 1 90); do curl -sf http://127.0.0.1:8080/v1/models > /dev/null || break; sleep 1; done
DQ=/opt/models/dl2/qat-assistant.q40qat.ninferno
export NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=1 NINFERNO_GEMMA4_SHARED_VALUES=2
step "huihui-nvfp4: quality (4 lanes)"
./srv_final5.sh start qI-huihui-nvfp4 /opt/models/dl3/huihui-qat-abl-nvfp4.r31.ninferno --max-context 4096 --max-concurrency 4 --kv-dtype fp8 \
    --spec mtp --drafter $DQ --draft-tokens 7 && \
    python3 geval.py --label "unc-huihui-nvfp4" --suites gsm8k,mmlupro,mmlu,humaneval --conc 4
./srv.sh stop
step "restoring the fast unit"
./deploy_fast.sh
step "N2I DONE"
