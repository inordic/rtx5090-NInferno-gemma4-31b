#!/bin/bash
# night7: behind a 5-minute idle gate - full validation of the candidate: apps-final2 (3036f83, qkv
# T8 schedule) with 7 drafts, otherwise the deployed fast config. Deployed unit back after.
cd /root/gbench
step() { echo "=== $(date +%T) $*"; }
./idle_gate_any.sh 5
step "idle: taking the GPU"
systemctl stop gemma4-31b-fast
for i in $(seq 1 90); do curl -sf http://127.0.0.1:8080/v1/models > /dev/null || break; sleep 1; done
QN=/opt/models/dl2/qat-nvfp4.r31.ninferno
DQ=/opt/models/dl2/qat-assistant.q40qat.ninferno
export NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=1 NINFERNO_GEMMA4_SHARED_VALUES=2
sed "s#^S=.*#S=/opt/NInferno/build/apps-final2/ninferno-serve#" srv.sh > srv_final2.sh; chmod +x srv_final2.sh
step "C2 candidate: final2, K7, 155K chunk1024"
./srv_final2.sh start C2-final2-k7 $QN --max-context 155000 --prefill-chunk 1024 --kv-dtype fp8 --spec mtp --drafter $DQ --draft-tokens 7 && \
    python3 gbench.py --label "C2 candidate final2 K7 155K chunk1024" --pp 2048,8192,32768,65536 --reps 1 && \
    python3 verify.py http://127.0.0.1:8080 80000 && \
    python3 needle.py --label "C2 candidate needles" --lengths 32000,72000,140000 --depths 0.1,0.5,0.9
./srv.sh stop
step "quality: candidate (4 lanes)"
./srv_final2.sh start q-c2 $QN --max-context 4096 --max-concurrency 4 --kv-dtype fp8 --spec mtp --drafter $DQ --draft-tokens 7 && \
    python3 geval.py --label nvfp4-final2-k7 --suites gsm8k,mmlupro,humaneval --conc 4
./srv.sh stop
step "restoring the deployed unit"
./deploy_fast.sh
step "NIGHT7 DONE"
