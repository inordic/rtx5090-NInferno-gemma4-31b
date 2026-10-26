#!/bin/bash
# queue6: preempt queue4 at its quality matrix; run the prefill sweep and llama.cpp references
# first, then the quality matrix (QAT drafter, sizing that fits).
cd /root/gbench
until grep -q "=== .* quality matrix (QAT drafter)" queue4.log; do sleep 10; done
pkill -f "^/bin/bash ./queue4.sh"; pkill -f "^/bin/bash ./queue5.sh"; pkill -f "^/bin/bash ./qmatrix.sh"
pkill -f "^python3 geval.py"; sleep 1; ./srv.sh stop; ./lsrv.sh stop; sleep 3
QU=/opt/models/dl/.ninferno-cache/gemma-4-31B-it-qat-UD-Q4_K_XL.gemma4_published.ninferno
QP=/opt/models/dl2/.ninferno-cache/pajari-Q4_0_V2.gemma4_published.ninferno
AK=/opt/models/.ninferno-cache/gemma-4-31B-abliterated-Q4_K.gemma4_published.ninferno
DQ=/opt/models/dl2/qat-assistant.q40qat.ninferno
GU=/opt/models/dl/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf
LB=/opt/llama.cpp/build-cuda/bin
step() { echo "=== $(date +%T) $*"; }
pp() {  # pp <label> [serve args...]: prefill-only bench at 2K..64K
    local label=$1; shift
    NINFERNO_GEMMA4_SHARED_VALUES=2 ./srv.sh start "$label" $QU --max-context 70000 --kv-dtype fp8 \
        --spec mtp --drafter $DQ --draft-tokens 4 "$@" && \
        python3 gbench.py --label "$label" --pp 2048,8192,32768,65536 --tg "" --skip-chat --reps 1
    ./srv.sh stop
}
step "C7 serve: QAT drafter, tuned profile (mtp:10:3), shared2 262K chunk512"
NINFERNO_GEMMA4_SHARED_VALUES=2 ./srv.sh start C7-qatud-profile-s2-262k $QU --max-context 262144 --prefill-chunk 512 --kv-dtype fp8 --drafter $DQ --spec-profile /root/gbench/qatasst_deep.tune.json && \
  python3 gbench.py --label "C7 QAT-UD QATdrafter profile(mtp10:3) shared2 262K" --pp ""
./srv.sh stop
step "C8 serve: QAT drafter K=8, shared2 262K chunk512"
NINFERNO_GEMMA4_SHARED_VALUES=2 ./srv.sh start C8-qatud-k8-s2-262k $QU --max-context 262144 --prefill-chunk 512 --kv-dtype fp8 --spec mtp --drafter $DQ --draft-tokens 8 && \
  python3 gbench.py --label "C8 QAT-UD QATdrafter K8 shared2 262K" --pp ""
./srv.sh stop
step "PP sweep: chunk 1024 / 2048 / 4096, dense tuner, exact activations"
pp P1-chunk1024 --prefill-chunk 1024
pp P2-chunk2048 --prefill-chunk 2048
pp P3-chunk4096 --prefill-chunk 4096
pp P4-densetune-chunk2048 --prefill-chunk 2048 --dense-tune on
pp P5-densetune-again-chunk2048 --prefill-chunk 2048 --dense-tune on
pp P6-exact-chunk2048 --prefill-chunk 2048 --activations exact
step "llama.cpp bench pp/tg"
$LB/llama-bench -m $GU -ngl 999 -fa 1 -p 512,2048,8192 -n 128 -r 2 2>&1 | grep -E "\| +gemma|pp|tg" | tail -6
$LB/llama-bench -m $GU -ngl 999 -fa 1 -p 2048 -n 0 -b 4096 -ub 2048 -r 2 2>&1 | grep -E "pp" | tail -2
step "llama.cpp L1 plain f16 KV 32K"
./lsrv.sh start L1-lcpp-plain $LB -m $GU -c 32768 && python3 gbench.py --label "L1 llama.cpp QAT-UD plain f16kv 32K" --skip-sampled; ./lsrv.sh stop
step "llama.cpp L2 MTP4 (QAT MTP Q4_0 lossless) f16 KV 32K"
./lsrv.sh start L2-lcpp-mtp4 $LB -m $GU -c 32768 --spec-type draft-mtp -md /opt/models/dl2/qat-assistant-Q4_0-qat.gguf --spec-draft-n-max 4 && python3 gbench.py --label "L2 llama.cpp QAT-UD MTP4 Q40qat f16kv 32K" --pp ""; ./lsrv.sh stop
step "llama.cpp L4 DFlash7 f16 KV 32K"
./lsrv.sh start L4-lcpp-dflash $LB -m $GU -c 32768 --spec-type draft-dflash -md /opt/models/dl2/dflash-Q4_K_M.gguf --spec-draft-n-max 7 && python3 gbench.py --label "L4 llama.cpp QAT-UD DFlash7 f16kv 32K" --pp ""; ./lsrv.sh stop
step "quality matrix (QAT drafter)"
sed -i "s#^DR=.*#DR=$DQ#" qmatrix.sh
./qmatrix.sh qatud-fp8 $QU 0 fp8
./qmatrix.sh qatud-shared2 $QU 2 fp8
./qmatrix.sh qatud-bf16 $QU 0 bf16
./qmatrix.sh pajari-shared2 $QP 2 fp8
./qmatrix.sh abl-q4k-fp8 $AK 0 fp8
step "QUEUE6 DONE"
