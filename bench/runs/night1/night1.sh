#!/bin/bash
# night1: behind the idle gate - W4A4 verify (NINFERNO_GEMMA4_VERIFY_A4=1) against A16 verify:
# decode on the fast config, then the quality suite with W4A4 verify. The fast unit comes back after.
cd /root/gbench
step() { echo "=== $(date +%T) $*"; }
step "waiting for 45 idle minutes on 8080"
./idle_gate.sh 45
step "idle: taking the GPU"
systemctl stop gemma4-31b-fast
for i in $(seq 1 90); do curl -sf http://127.0.0.1:8080/v1/models > /dev/null || break; sleep 1; done
QN=/opt/models/dl2/qat-nvfp4.r31.ninferno
DQ=/opt/models/dl2/qat-assistant.q40qat.ninferno
export NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=1 NINFERNO_GEMMA4_SHARED_VALUES=2
sed "s#^S=.*#S=/opt/NInferno/build/apps-va4/ninferno-serve#" srv.sh > srv_va4.sh; chmod +x srv_va4.sh
for a in 0 1; do
    step "V$a decode: verify A4=$a"
    NINFERNO_GEMMA4_VERIFY_A4=$a ./srv_va4.sh start V$a-verify $QN --max-context 155000 --prefill-chunk 1024 --kv-dtype fp8 --spec mtp --drafter $DQ --draft-tokens 8 && \
        python3 gbench.py --label "V$a QAT-NVFP4 final, verify A4=$a, K8 155K" --pp 2048 --reps 1
    ./srv.sh stop
done
step "quality with W4A4 verify (4 lanes)"
NINFERNO_GEMMA4_VERIFY_A4=1 ./srv_va4.sh start q-va4 $QN --max-context 4096 --max-concurrency 4 --kv-dtype fp8 --spec mtp --drafter $DQ --draft-tokens 8 && \
    python3 geval.py --label nvfp4-final-verifyA4 --suites gsm8k,mmlupro,humaneval --conc 4
./srv.sh stop
step "restoring the fast unit"
./deploy_fast.sh
step "NIGHT1 DONE"
