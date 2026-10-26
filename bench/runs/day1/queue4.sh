#!/bin/bash
# queue4: once queue3 reaches its quality matrix, take over: QAT drafter depth sweep, serving benches
# with the QAT drafter, Pajari vs QAT-UD, then the quality matrix with the new drafter.
cd /root/gbench
until grep -q "=== .* quality matrix" queue3.log; do sleep 10; done
pkill -f "^/bin/bash ./queue3.sh"; pkill -f "^/bin/bash ./qmatrix.sh"; pkill -f "geval.py"; sleep 1
./srv.sh stop; ./lsrv.sh stop; sleep 3
QU=/opt/models/dl/.ninferno-cache/gemma-4-31B-it-qat-UD-Q4_K_XL.gemma4_published.ninferno
QP=/opt/models/dl2/.ninferno-cache/pajari-Q4_0_V2.gemma4_published.ninferno
AK=/opt/models/.ninferno-cache/gemma-4-31B-abliterated-Q4_K.gemma4_published.ninferno
DQ8=/opt/models/dl2/qat-assistant.q8.ninferno
DQ=/opt/models/dl2/qat-assistant.q40qat.ninferno
G=/opt/NInferno/build/apps/ninferno-gemma4
step() { echo "=== $(date +%T) $*"; }
tune() {  # tune <artifact> <drafter> <configs> <tag>
    $G "$1" spec-tune --drafter "$2" --kv-dtype fp8 --configs "$3" --budget-seconds 4000 \
        --out "/root/gbench/$4.tune.json" > "/root/gbench/tune_$4.log" 2>&1
    grep -E "spec-tune (mtp|plain)|default" "/root/gbench/tune_$4.log" | sed "s/^/[$4] /"
}
step "QAT drafter Q4_0 (lossless) depth sweep on QAT-UD"
tune $QU $DQ mtp:3,mtp:4,mtp:5,mtp:6,mtp:8,mtp:4:2,mtp:6:2,mtp:8:2,mtp:10:3 qatasst_deep
step "Pajari Q4_0_V2 with QAT drafter"
tune $QP $DQ mtp:4 pajari_qatasst
step "C5 serve: QAT-UD + QAT drafter K=4 + shared2 262K chunk512"
NINFERNO_GEMMA4_SHARED_VALUES=2 ./srv.sh start C5-qatud-qatasst4-s2-262k $QU --max-context 262144 --prefill-chunk 512 --kv-dtype fp8 --spec mtp --drafter $DQ --draft-tokens 4 && \
  python3 gbench.py --label "C5 QAT-UD QATdrafter K4 shared2 262K"
./srv.sh stop
step "C6 serve: QAT-UD + QAT drafter K=3 + shared2 262K chunk512"
NINFERNO_GEMMA4_SHARED_VALUES=2 ./srv.sh start C6-qatud-qatasst3-s2-262k $QU --max-context 262144 --prefill-chunk 512 --kv-dtype fp8 --spec mtp --drafter $DQ --draft-tokens 3 && \
  python3 gbench.py --label "C6 QAT-UD QATdrafter K3 shared2 262K" --pp ""
./srv.sh stop
step "quality matrix (QAT drafter)"
sed -i "s#gemma-4-31B-it-assistant.gemma4_assistant_published.q8_sl98304.ninferno#X#; s#^DR=.*#DR=$DQ#" qmatrix.sh
./qmatrix.sh qatud-fp8 $QU 0 fp8
./qmatrix.sh qatud-shared2 $QU 2 fp8
./qmatrix.sh qatud-bf16 $QU 0 bf16
./qmatrix.sh pajari-shared2 $QP 2 fp8
./qmatrix.sh abl-q4k-fp8 $AK 0 fp8
step "QUEUE4 DONE"
