#!/bin/bash
# night12: behind a 5-minute idle gate - 6 against 7 drafts on final3 (both verify at T <= 8).
cd /root/gbench
step() { echo "=== $(date +%T) $*"; }
./idle_gate_any.sh 5
step "idle: taking the GPU"
systemctl stop gemma4-31b-fast
for i in $(seq 1 90); do curl -sf http://127.0.0.1:8080/v1/models > /dev/null || break; sleep 1; done
QN=/opt/models/dl2/qat-nvfp4.r31.ninferno
DQ=/opt/models/dl2/qat-assistant.q40qat.ninferno
export NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=1 NINFERNO_GEMMA4_SHARED_VALUES=2
for k in 6 7 6 7; do
    step "K$k final3"
    ./srv_final3.sh start K$k-f3 $QN --max-context 155000 --prefill-chunk 1024 --kv-dtype fp8 --spec mtp --drafter $DQ --draft-tokens $k && \
        python3 gbench.py --label "K$k final3 155K" --pp "" --reps 1
    ./srv.sh stop
done
step "restoring the deployed unit"
./deploy_fast.sh
step "NIGHT12 DONE"
