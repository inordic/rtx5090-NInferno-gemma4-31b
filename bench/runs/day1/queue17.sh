#!/bin/bash
# queue17: after queue16 - the 140K needle P7 missed: which piece is it? Same needles (L 140000,
# depths 0.1 / 0.5 / 0.9) on Q4 QAT-UD, NVFP4 without INT8, NVFP4 + INT8 D512, NVFP4 + INT8 + FP8 PV.
cd /root/gbench
until grep -qE "QUEUE19 DONE|PV8B BUILD FAILED" queue19.log 2>/dev/null; do sleep 15; done
step() { echo "=== $(date +%T) $*"; }
./srv.sh stop; sleep 2
QN=/opt/models/dl2/qat-nvfp4.r31.ninferno
QU=/opt/models/dl/.ninferno-cache/gemma-4-31B-it-qat-UD-Q4_K_XL.gemma4_published.ninferno
DQ=/opt/models/dl2/qat-assistant.q40qat.ninferno
export NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=64
nd() {  # nd <label> <srv script> <artifact>
    NINFERNO_GEMMA4_SHARED_VALUES=2 ./$2 start "N-$1" $3 --max-context 155000 --prefill-chunk 1024 --kv-dtype fp8 --spec mtp --drafter $DQ --draft-tokens 8 && \
        python3 needle.py --label "N140 $1" --lengths 140000 --depths 0.1,0.5,0.9
    ./srv.sh stop
}
step "needle 140K: NVFP4 + INT8 D512 (i8c)"; nd nvfp4-i8c srv_i8c.sh $QN
step "needle 140K: NVFP4 no INT8 (split)"; nd nvfp4-split srv.sh $QN
step "needle 140K: Q4 QAT-UD"; nd qatud srv.sh $QU
step "needle 140K: NVFP4 + INT8 + FP8 PV (pv8)"; nd nvfp4-pv8 srv_pv8.sh $QN
step "QUEUE17 DONE"
