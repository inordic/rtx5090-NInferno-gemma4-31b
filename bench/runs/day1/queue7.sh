#!/bin/bash
# queue7: after queue6 - tuned-profile serve retry, Heretic QAT-aware speed/acceptance/quality,
# multi-turn re-prefill check.
cd /root/gbench
./srv.sh stop; ./lsrv.sh stop; sleep 2
QU=/opt/models/dl/.ninferno-cache/gemma-4-31B-it-qat-UD-Q4_K_XL.gemma4_published.ninferno
QH=/opt/models/dl2/heretic-qat-q40qat.ninferno
DQ=/opt/models/dl2/qat-assistant.q40qat.ninferno
G=/opt/NInferno/build/apps/ninferno-gemma4
step() { echo "=== $(date +%T) $*"; }
step "A1 agent loop: QAT-UD K8 shared2 262K chunk512"
NINFERNO_GEMMA4_SHARED_VALUES=2 ./srv.sh start A1-agent-262k $QU --max-context 262144 --prefill-chunk 512 --kv-dtype fp8 --spec mtp --drafter $DQ --draft-tokens 8 && \
  python3 agentbench.py --label "A1 262K chunk512 K8"
./srv.sh stop
step "A2 agent loop: QAT-UD K8 shared2 131K chunk1024"
NINFERNO_GEMMA4_SHARED_VALUES=2 ./srv.sh start A2-agent-131k $QU --max-context 131072 --prefill-chunk 1024 --kv-dtype fp8 --spec mtp --drafter $DQ --draft-tokens 8 && \
  python3 agentbench.py --label "A2 131K chunk1024 K8"
./srv.sh stop
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
sed -i "s#^DR=.*#DR=$DQ#" qmatrix.sh
./qmatrix.sh heretic-qataware-shared2 $QH 2 fp8
step "QUEUE7 DONE"
