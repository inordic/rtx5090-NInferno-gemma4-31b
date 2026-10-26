#!/bin/bash
# queue5: llama.cpp reference runs after queue4 (same gbench harness, port 8080).
cd /root/gbench
until grep -q "QUEUE4 DONE" queue4.log; do sleep 15; done
./srv.sh stop; sleep 2
GU=/opt/models/dl/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf
LB=/opt/llama.cpp/build-cuda/bin
step() { echo "=== $(date +%T) $*"; }
step "llama.cpp bench pp/tg"
$LB/llama-bench -m $GU -ngl 999 -fa 1 -p 512,2048,8192 -n 128 -r 2 2>&1 | grep -E "\| +gemma|pp|tg" | tail -6
step "llama.cpp L1 plain f16 KV 32K"
./lsrv.sh start L1-lcpp-plain $LB -m $GU -c 32768 && python3 gbench.py --label "L1 llama.cpp QAT-UD plain f16kv 32K" --skip-sampled; ./lsrv.sh stop
step "llama.cpp L2 MTP3 (QAT MTP Q8_0) f16 KV 32K"
./lsrv.sh start L2-lcpp-mtp3 $LB -m $GU -c 32768 --spec-type draft-mtp -md /opt/models/dl2/mtp-qat-Q8_0.gguf --spec-draft-n-max 3 && python3 gbench.py --label "L2 llama.cpp QAT-UD MTP3 f16kv 32K" --pp ""; ./lsrv.sh stop
step "llama.cpp L2b MTP4 (QAT MTP Q4_0 lossless) f16 KV 32K"
./lsrv.sh start L2b-lcpp-mtp4 $LB -m $GU -c 32768 --spec-type draft-mtp -md /opt/models/dl2/qat-assistant-Q4_0-qat.gguf --spec-draft-n-max 4 && python3 gbench.py --label "L2b llama.cpp QAT-UD MTP4 Q40qat f16kv 32K" --pp ""; ./lsrv.sh stop
step "llama.cpp L4 DFlash7 f16 KV 32K"
./lsrv.sh start L4-lcpp-dflash $LB -m $GU -c 32768 --spec-type draft-dflash -md /opt/models/dl2/dflash-Q4_K_M.gguf --spec-draft-n-max 7 && python3 gbench.py --label "L4 llama.cpp QAT-UD DFlash7 f16kv 32K" --pp ""; ./lsrv.sh stop
step "QUEUE5 DONE"
