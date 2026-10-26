#!/bin/bash
# night11: behind a 5-minute idle gate - (1) full validation of candidate final3 (1a3d564: + Q6 head
# 8-column tile) with 7 drafts; (2) the 262K Q4 profile on final3 with 7 against 8 drafts.
cd /root/gbench
step() { echo "=== $(date +%T) $*"; }
./idle_gate_any.sh 5
step "idle: taking the GPU"
systemctl stop gemma4-31b-fast
for i in $(seq 1 90); do curl -sf http://127.0.0.1:8080/v1/models > /dev/null || break; sleep 1; done
QN=/opt/models/dl2/qat-nvfp4.r31.ninferno
QU=/opt/models/dl/.ninferno-cache/gemma-4-31B-it-qat-UD-Q4_K_XL.gemma4_published.ninferno
DQ=/opt/models/dl2/qat-assistant.q40qat.ninferno
export NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=1 NINFERNO_GEMMA4_SHARED_VALUES=2
sed "s#^S=.*#S=/opt/NInferno/build/apps-final3/ninferno-serve#" srv.sh > srv_final3.sh; chmod +x srv_final3.sh
step "C3 candidate: final3, K7, 155K chunk1024"
./srv_final3.sh start C3-final3-k7 $QN --max-context 155000 --prefill-chunk 1024 --kv-dtype fp8 --spec mtp --drafter $DQ --draft-tokens 7 && \
    python3 gbench.py --label "C3 candidate final3 K7 155K chunk1024" --pp 2048,8192,32768,65536 --reps 1 && \
    python3 verify.py http://127.0.0.1:8080 80000 && \
    python3 needle.py --label "C3 candidate needles" --lengths 32000,72000,140000 --depths 0.1,0.5,0.9
./srv.sh stop
step "quality: candidate final3 (4 lanes)"
./srv_final3.sh start q-c3 $QN --max-context 4096 --max-concurrency 4 --kv-dtype fp8 --spec mtp --drafter $DQ --draft-tokens 7 && \
    python3 geval.py --label nvfp4-final3-k7 --suites gsm8k,mmlupro,humaneval --conc 4
./srv.sh stop
for k in 7 8; do
    step "Q4 262K on final3, K$k"
    ./srv_final3.sh start Q4K$k $QU --max-context 262144 --prefill-chunk 512 --kv-dtype fp8 --spec mtp --drafter $DQ --draft-tokens $k && \
        python3 gbench.py --label "Q4K$k QAT-UD final3 K$k 262K chunk512" --pp 2048,8192 --reps 1
    ./srv.sh stop
done
step "restoring the deployed unit"
./deploy_fast.sh
step "NIGHT11 DONE"
