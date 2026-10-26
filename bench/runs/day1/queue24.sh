#!/bin/bash
# queue24: after queue23 - decode with the GeGLU split at every width (verify T=9 takes the NVFP4
# A16 register MMA for gate_up instead of the generic fused GeGLU), against the split at T >= 64.
cd /root/gbench
until grep -qE "QUEUE23 DONE" queue23.log 2>/dev/null; do sleep 10; done
step() { echo "=== $(date +%T) $*"; }
./srv.sh stop; sleep 2
QN=/opt/models/dl2/qat-nvfp4.r31.ninferno
DQ=/opt/models/dl2/qat-assistant.q40qat.ninferno
sed "s#^S=.*#S=/opt/NInferno/build/apps-qk8p1/ninferno-serve#" srv.sh > srv_qk8p1.sh; chmod +x srv_qk8p1.sh
for m in 64 1; do
    step "D$m decode: GeGLU split at T >= $m (qk8p1)"
    NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=$m NINFERNO_GEMMA4_SHARED_VALUES=2 ./srv_qk8p1.sh start D$m-split $QN --max-context 155000 --prefill-chunk 1024 --kv-dtype fp8 --spec mtp --drafter $DQ --draft-tokens 8 && \
        python3 gbench.py --label "D$m QAT-NVFP4 qk8p1 GeGLU split T>=$m K8 155K chunk1024" --pp 2048,8192 --reps 1
    ./srv.sh stop
done
step "QUEUE24 DONE"
