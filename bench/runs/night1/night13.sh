#!/bin/bash
# night13: behind a 5-minute idle gate - the 262K QAT Q4 profile on final3 with 7 drafts: verify, a
# 200K needle and a 250K needle. Deployed unit back after.
cd /root/gbench
step() { echo "=== $(date +%T) $*"; }
./idle_gate_any.sh 5
step "idle: taking the GPU"
systemctl stop gemma4-31b-fast
for i in $(seq 1 90); do curl -sf http://127.0.0.1:8080/v1/models > /dev/null || break; sleep 1; done
QU=/opt/models/dl/.ninferno-cache/gemma-4-31B-it-qat-UD-Q4_K_XL.gemma4_published.ninferno
DQ=/opt/models/dl2/qat-assistant.q40qat.ninferno
export NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=1 NINFERNO_GEMMA4_SHARED_VALUES=2
./srv_final3.sh start U262-f3-k7 $QU --max-context 262144 --prefill-chunk 512 --kv-dtype fp8 --spec mtp --drafter $DQ --draft-tokens 7 && \
    python3 verify.py http://127.0.0.1:8080 80000 && \
    python3 needle.py --label "U262 final3 K7" --lengths 200000,250000 --depths 0.5
./srv.sh stop
step "restoring the deployed unit"
./deploy_fast.sh
step "NIGHT13 DONE"
