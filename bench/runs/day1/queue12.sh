#!/bin/bash
# queue12: after queue11 - isolate the MMLU-Pro drop: NVFP4 split (no INT8), NVFP4 i8c (repeat), Q4 (repeat)
cd /root/gbench
until grep -q "QUEUE11 DONE" queue11.log; do sleep 15; done
./srv.sh stop; sleep 2
QN=/opt/models/dl2/qat-nvfp4.r31.ninferno
QU=/opt/models/dl/.ninferno-cache/gemma-4-31B-it-qat-UD-Q4_K_XL.gemma4_published.ninferno
DQ=/opt/models/dl2/qat-assistant.q40qat.ninferno
export NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=64
step() { echo "=== $(date +%T) $*"; }
km() {  # km <label> <srv script> <artifact>
    NINFERNO_GEMMA4_SHARED_VALUES=2 ./$2 start "k-$1" $3 --max-context 4096 --max-concurrency 4 --kv-dtype fp8 --spec mtp --drafter $DQ --draft-tokens 8 || return
    python3 geval.py --label "$1" --suites mmlupro,mmlu --conc 4
    ./srv.sh stop
}
step "knowledge: NVFP4 split (no INT8)"; km nvfp4-split srv.sh $QN
step "knowledge: NVFP4 split+INT8 (repeat)"; km nvfp4-i8c srv_i8c.sh $QN
step "knowledge: Q4 QAT-UD (repeat)"; km qatud srv.sh $QU
step "PP: NVFP4 split+INT8 at chunk 1024, 155K context"
NINFERNO_GEMMA4_SHARED_VALUES=2 ./srv_i8c.sh start P7-nvfp4-i8c-155k $QN --max-context 155000 --prefill-chunk 1024 --kv-dtype fp8 --spec mtp --drafter $DQ --draft-tokens 8 && \
    python3 gbench.py --label "P7 QAT-NVFP4 split INT8 K8 shared2 155K chunk1024" --pp 2048,8192,32768,65536 --reps 1 --skip-chat && \
    python3 needle.py --label "P7 155K needle" --lengths 140000 --depths 0.5
./srv.sh stop
step "QUEUE12 DONE"
