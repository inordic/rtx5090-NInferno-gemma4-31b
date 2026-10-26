#!/bin/bash
# night4: after night3, behind a 5-minute idle gate - the QAT drafter with a shortlist head (64K /
# 96K rows of gemma4_ranking2, QAT Q4_0 rows) against the full 262K head; final binary, K=8.
cd /root/gbench
step() { echo "=== $(date +%T) $*"; }
until grep -q "NIGHT3 DONE" night3.log 2>/dev/null; do sleep 20; done
step "waiting for 5 idle minutes on 8080"
./idle_gate.sh 5
step "idle: taking the GPU"
systemctl stop gemma4-31b-fast
for i in $(seq 1 90); do curl -sf http://127.0.0.1:8080/v1/models > /dev/null || break; sleep 1; done
QN=/opt/models/dl2/qat-nvfp4.r31.ninferno
export NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=1 NINFERNO_GEMMA4_SHARED_VALUES=2
for d in sl64k-q8 sl96k-q8 q40qat; do
    step "drafter $d"
    ./srv_final.sh start DR-$d $QN --max-context 155000 --prefill-chunk 1024 --kv-dtype fp8 --spec mtp \
        --drafter /opt/models/dl2/qat-assistant.$d.ninferno --draft-tokens 8 && \
        python3 gbench.py --label "DR $d QAT-NVFP4 final split1 K8 155K" --pp "" --reps 1
    grep -hE "mtp accepted" logs/DR-$d.log | tail -3 | sed -E "s/^.*(mtp accepted [^|]*).*/\1/"
    ./srv.sh stop
done
step "restoring the fast unit"
./deploy_fast.sh
step "NIGHT4 DONE"
