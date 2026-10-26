#!/bin/bash
# queue2: after C4 needle, replace queue1 with plan steps 1-2 (drafter A/B) then quality matrix
cd /root/gbench
until grep -q "NEEDLE C4" queue1.log; do sleep 10; done
pkill -f "queue1.sh"; sleep 1; ./srv.sh stop; pkill -f "^/opt/NInferno/build/apps/ninferno-gemma4"; sleep 3
QU=/opt/models/dl/.ninferno-cache/gemma-4-31B-it-qat-UD-Q4_K_XL.gemma4_published.ninferno
QA=/opt/models/.ninferno-cache/gemma-4-31B_q4_0-it.gemma4_published.ninferno
AK=/opt/models/.ninferno-cache/gemma-4-31B-abliterated-Q4_K.gemma4_published.ninferno
step() { echo "=== $(date +%T) $*"; }
G=/opt/NInferno/build/apps/ninferno-gemma4
tuneD() { $G $QU spec-tune --drafter $1 --kv-dtype fp8 --configs mtp:3,mtp:4 --budget-seconds 3000 --out /root/gbench/$2.tune.json > /root/gbench/tune_$2.log 2>&1; grep -E "spec-tune (mtp|plain)" /root/gbench/tune_$2.log | sed "s/^/[$2] /"; }
step "drafter A/B: QAT-matched assistant, full head"
tuneD /opt/models/dl2/qat-assistant.q8.ninferno qatasst_full
step "drafter A/B: generic assistant, full head"
tuneD /opt/models/dl2/assistant.q8full.ninferno asst_full
step "quality matrix"
./qmatrix.sh qatud-fp8 $QU 0 fp8
./qmatrix.sh qatud-shared2 $QU 2 fp8
./qmatrix.sh qatud-bf16 $QU 0 bf16
./qmatrix.sh qatud-shared1 $QU 1 fp8
./qmatrix.sh qat-q40-fp8 $QA 0 fp8
./qmatrix.sh abl-q4k-fp8 $AK 0 fp8
step "QUEUE2 DONE"
