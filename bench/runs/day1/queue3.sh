#!/bin/bash
# queue3: after queue1's C4 needle, drafter A/B, llama.cpp reference runs, then the quality matrix.
cd /root/gbench
until grep -q "NEEDLE C4" queue1.log; do sleep 10; done
pkill -f "^/bin/bash ./queue1.sh"; sleep 1; ./srv.sh stop; pkill -f "^/opt/NInferno/build/apps/ninferno-gemma4"; sleep 3
QU=/opt/models/dl/.ninferno-cache/gemma-4-31B-it-qat-UD-Q4_K_XL.gemma4_published.ninferno
AK=/opt/models/.ninferno-cache/gemma-4-31B-abliterated-Q4_K.gemma4_published.ninferno
GU=/opt/models/dl/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf
LB=/opt/llama.cpp/build-cuda/bin
step() { echo "=== $(date +%T) $*"; }
G=/opt/NInferno/build/apps/ninferno-gemma4
tuneD() {
    $G $QU spec-tune --drafter "$1" --kv-dtype fp8 --configs mtp:3,mtp:4 --budget-seconds 3000 \
        --out "/root/gbench/$2.tune.json" > "/root/gbench/tune_$2.log" 2>&1
    grep -E "spec-tune (mtp|plain)" "/root/gbench/tune_$2.log" | sed "s/^/[$2] /"
}
step "drafter A/B: QAT-matched assistant, full head"
tuneD /opt/models/dl2/qat-assistant.q8.ninferno qatasst_full
step "drafter A/B: generic assistant, full head"
tuneD /opt/models/dl2/assistant.q8full.ninferno asst_full
step "llama.cpp L1 plain f16 KV 32K"
./lsrv.sh start L1-lcpp-plain $LB -m $GU -c 32768 && python3 gbench.py --label "L1 llama.cpp QAT-UD plain f16kv 32K"; ./lsrv.sh stop
step "llama.cpp L2 MTP3 f16 KV 32K"
./lsrv.sh start L2-lcpp-mtp3 $LB -m $GU -c 32768 --spec-type draft-mtp -md /opt/models/dl2/mtp-qat-Q8_0.gguf --spec-draft-n-max 3 && python3 gbench.py --label "L2 llama.cpp QAT-UD MTP3 f16kv 32K"; ./lsrv.sh stop
step "llama.cpp L3 MTP3 q8_0 KV 32K"
./lsrv.sh start L3-lcpp-mtp3-q8kv $LB -m $GU -c 32768 -ctk q8_0 -ctv q8_0 --spec-type draft-mtp -md /opt/models/dl2/mtp-qat-Q8_0.gguf --spec-draft-n-max 3 && python3 gbench.py --label "L3 llama.cpp QAT-UD MTP3 q8kv 32K" --skip-sampled; ./lsrv.sh stop
step "llama.cpp L4 DFlash7 f16 KV 32K"
./lsrv.sh start L4-lcpp-dflash $LB -m $GU -c 32768 --spec-type draft-dflash -md /opt/models/dl2/dflash-Q4_K_M.gguf --spec-draft-n-max 7 && python3 gbench.py --label "L4 llama.cpp QAT-UD DFlash7 f16kv 32K"; ./lsrv.sh stop
step "quality matrix"
./qmatrix.sh qatud-fp8 $QU 0 fp8
./qmatrix.sh qatud-shared2 $QU 2 fp8
./qmatrix.sh qatud-bf16 $QU 0 bf16
./qmatrix.sh abl-q4k-fp8 $AK 0 fp8
step "QUEUE3 DONE"
