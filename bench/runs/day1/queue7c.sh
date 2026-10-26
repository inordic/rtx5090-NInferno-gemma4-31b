#!/bin/bash
# queue7c: once queue7 finishes its agent loops, take over: Gemma 31B QAT-NVFP4 (W4A4) prefill and
# decode, then the rest of queue7 (tuned profile, Heretic). Ends by marking queue7 done for queue8.
cd /root/gbench
./srv.sh stop; sleep 2
QU=/opt/models/dl/.ninferno-cache/gemma-4-31B-it-qat-UD-Q4_K_XL.gemma4_published.ninferno
QH=/opt/models/dl2/heretic-qat-q40qat.ninferno
QN=/opt/models/dl2/qat-nvfp4.r31.ninferno
DQ=/opt/models/dl2/qat-assistant.q40qat.ninferno
G=/opt/NInferno/build/apps/ninferno-gemma4
step() { echo "=== $(date +%T) $*" | tee -a queue7.log; }
step "F3 NVFP4 quality (QAT drafter)"
sed -i "s#^DR=.*#DR=$DQ#" qmatrix.sh
./qmatrix.sh qat-nvfp4-shared2 $QN 2 fp8
step "C7 serve: tuned profile (spec auto), shared2 262K chunk512"
NINFERNO_GEMMA4_SHARED_VALUES=2 ./srv.sh start C7-qatud-profile-s2-262k $QU --max-context 262144 --prefill-chunk 512 --kv-dtype fp8 --spec-profile /root/gbench/qatasst_deep.tune.json && \
    python3 gbench.py --label "C7 QAT-UD QATdrafter profile(mtp10:3) shared2 262K" --pp ""
./srv.sh stop
step "Heretic QAT-aware: drafter acceptance"
$G $QH spec-tune --drafter $DQ --kv-dtype fp8 --configs mtp:4,mtp:8,mtp:10:3 --budget-seconds 4000 \
    --out /root/gbench/heretic_qatasst.tune.json > /root/gbench/tune_heretic_qatasst.log 2>&1
grep -E "spec-tune (mtp|plain)|default" /root/gbench/tune_heretic_qatasst.log | sed "s/^/[heretic] /"
step "H1 serve: Heretic QAT-aware K=8 shared2 262K chunk512"
NINFERNO_GEMMA4_SHARED_VALUES=2 ./srv.sh start H1-heretic-k8-s2-262k $QH --max-context 262144 --prefill-chunk 512 --kv-dtype fp8 --spec mtp --drafter $DQ --draft-tokens 8 && \
    python3 gbench.py --label "H1 Heretic QATaware QATdrafter K8 shared2 262K" && python3 multiturn.py --label "H1 multiturn"
./srv.sh stop
step "quality: Heretic QAT-aware"
./qmatrix.sh heretic-qataware-shared2 $QH 2 fp8
step "QUEUE7 DONE"
