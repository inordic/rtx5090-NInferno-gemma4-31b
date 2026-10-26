#!/bin/bash
# queue18: after queue17 + the MSE NVFP4 conversion - does MSE weight scaling win back the NVFP4
# knowledge drop? Full quality suite, 32K perplexity against r31 (same build), serve speed.
cd /root/gbench
until grep -qE "QUEUE22 DONE" queue22.log 2>/dev/null && grep -qE "MSE_CONVERT_DONE" /opt/models/dl2/nvfp4mse.log; do sleep 20; done
step() { echo "=== $(date +%T) $*"; }
./srv.sh stop; sleep 2
QN=/opt/models/dl2/qat-nvfp4.r31.ninferno
QM=/opt/models/dl2/qat-nvfp4-mse.r31.ninferno
DQ=/opt/models/dl2/qat-assistant.q40qat.ninferno
export NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=64
ls -la $QM || { step "MSE ARTIFACT MISSING"; exit 1; }
step "quality: NVFP4-MSE split (4 lanes)"
NINFERNO_GEMMA4_SHARED_VALUES=2 ./srv.sh start q-nvfp4-mse $QM --max-context 4096 --max-concurrency 4 --kv-dtype fp8 --spec mtp --drafter $DQ --draft-tokens 8 && \
    python3 geval.py --label nvfp4-mse-split --suites gsm8k,mmlupro,mmlu,humaneval --conc 4
./srv.sh stop
for m in r31 mse; do
    A=$QN; [ $m = mse ] && A=$QM
    step "score $m window 32768 (split, shared2)"
    NINFERNO_GEMMA4_SHARED_VALUES=2 /opt/NInferno/build/apps-split/ninferno-gemma4 $A score --tokens /root/gbench/ppl64k.i32 --window 32768 \
        --kv-dtype fp8 --context 32768 --dump /root/gbench/score_${m}_split_32768.bin > /dev/null 2>&1
    python3 ppl.py /root/gbench/score_${m}_split_32768.bin
done
step "S1 serve: NVFP4-MSE split, K8, shared2 155K chunk1024"
NINFERNO_GEMMA4_SHARED_VALUES=2 ./srv.sh start S1-nvfp4-mse $QM --max-context 155000 --prefill-chunk 1024 --kv-dtype fp8 --spec mtp --drafter $DQ --draft-tokens 8 && \
    python3 gbench.py --label "S1 QAT-NVFP4-MSE split K8 shared2 155K chunk1024" --pp 2048,8192,32768 --reps 1 && \
    python3 needle.py --label "R140 nvfp4-mse" --lengths 140000 --depths 0.5 && \
    python3 needle.py --label "N140 nvfp4-mse" --lengths 140000 --depths 0.1,0.5,0.9
./srv.sh stop
step "QUEUE18 DONE"
