#!/bin/bash
# queue11: the NVFP4 + GeGLU split + INT8 (D512) candidate (apps-i8c): quality, max context, serve
# bench and needles; then the abl Q4_K quality leftover.
cd /root/gbench
./srv.sh stop; ./lsrv.sh stop; sleep 2
QN=/opt/models/dl2/qat-nvfp4.r31.ninferno
DQ=/opt/models/dl2/qat-assistant.q40qat.ninferno
AK=/opt/models/.ninferno-cache/gemma-4-31B-abliterated-Q4_K.gemma4_published.ninferno
S8=/opt/NInferno/build/apps-i8c/ninferno-serve
export NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=64
step() { echo "=== $(date +%T) $*"; }
sed "s#^S=.*#S=$S8#" srv.sh > srv_i8c.sh; chmod +x srv_i8c.sh
sed -e "s#\./srv.sh#./srv_i8c.sh#g; s#^DR=.*#DR=$DQ#; s/--max-concurrency 6/--max-concurrency 4/; s/--conc 6/--conc 4/" qmatrix.sh > qmatrix_i8c.sh; chmod +x qmatrix_i8c.sh
step "Q1 quality: NVFP4 + split + INT8 D512 (4 lanes)"
./qmatrix_i8c.sh nvfp4-split-i8c-shared2 $QN 2 fp8
step "M plan: max context at chunk 2048 / 1024 / 512 with MTP K8"
for ch in 2048 1024 512; do
    printf "chunk %s: " $ch
    NINFERNO_GEMMA4_SHARED_VALUES=2 $S8 $QN --plan --kv-dtype fp8 --max-context 262144 --prefill-chunk $ch --spec mtp --drafter $DQ --draft-tokens 8 2>&1 | grep -oE "with up to [0-9]+ tokens" | head -1
done
step "F9 serve: NVFP4 + split + INT8, K8, shared2 96K chunk2048"
NINFERNO_GEMMA4_SHARED_VALUES=2 ./srv_i8c.sh start F9-nvfp4-i8c-96k $QN --max-context 90000 --prefill-chunk 2048 --kv-dtype fp8 --spec mtp --drafter $DQ --draft-tokens 8 && \
    python3 gbench.py --label "F9 QAT-NVFP4 split INT8 K8 shared2 90K chunk2048" --pp 2048,8192,32768,65536 --reps 1 && \
    python3 needle.py --label "F9 i8c needle" --lengths 32000,72000 --depths 0.1,0.5,0.9
./srv.sh stop
step "quality abl Q4_K (4 lanes, reference)"
sed -e "s/--max-concurrency 6/--max-concurrency 4/; s/--conc 6/--conc 4/; s#^DR=.*#DR=$DQ#" qmatrix.sh > qmatrix4.sh; chmod +x qmatrix4.sh
./qmatrix4.sh abl-q4k-fp8 $AK 0 fp8
step "QUEUE11 DONE"
