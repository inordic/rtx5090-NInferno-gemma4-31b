#!/bin/bash
# queue10: leftovers after queue9 - abl Q4_K quality with 4 lanes, llama.cpp DFlash retry
cd /root/gbench
until grep -q "QUEUE9 DONE" queue9b.log; do sleep 15; done
./srv.sh stop; sleep 2
AK=/opt/models/.ninferno-cache/gemma-4-31B-abliterated-Q4_K.gemma4_published.ninferno
GU=/opt/models/dl/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf
LB=/opt/llama.cpp/build-cuda/bin
step() { echo "=== $(date +%T) $*"; }
step "quality abl Q4_K (4 lanes)"
sed -e "s/--max-concurrency 6/--max-concurrency 4/; s/--conc 6/--conc 4/" qmatrix.sh > qmatrix4.sh; chmod +x qmatrix4.sh
./qmatrix4.sh abl-q4k-fp8 $AK 0 fp8
QN=/opt/models/dl2/qat-nvfp4.r31.ninferno
step "quality QAT-NVFP4 (4 lanes, GeGLU split)"
NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=64 ./qmatrix4.sh qat-nvfp4-shared2 $QN 2 fp8
step "C7 serve: tuned profile (spec auto), shared2 262K chunk512"
QU=/opt/models/dl/.ninferno-cache/gemma-4-31B-it-qat-UD-Q4_K_XL.gemma4_published.ninferno
NINFERNO_GEMMA4_SHARED_VALUES=2 ./srv.sh start C7-qatud-profile-s2-262k $QU --max-context 262144 --prefill-chunk 512 --kv-dtype fp8 --spec-profile /root/gbench/qatasst_deep.tune.json && python3 gbench.py --label "C7 QAT-UD QATdrafter profile(mtp10:3) shared2 262K" --pp ""
./srv.sh stop
step "llama.cpp L4 DFlash7 (1 slot, 16K)"
./lsrv.sh start L4-lcpp-dflash $LB -m $GU -c 16384 -np 1 --spec-type draft-dflash -md /opt/models/dl2/dflash-Q4_K_M.gguf --spec-draft-n-max 7 && python3 gbench.py --label "L4 llama.cpp QAT-UD DFlash7 16K" --pp ""; ./lsrv.sh stop
step "QUEUE10 DONE"
