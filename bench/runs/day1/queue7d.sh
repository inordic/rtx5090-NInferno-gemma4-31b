#!/bin/bash
# queue7d: after queue7c's NVFP4 quality run, take over: NVFP4 prefill with the GeGLU split,
# a decode check and a profile, then the rest of queue7 (tuned profile, Heretic).
cd /root/gbench
until grep -q "=== .* C7 serve" queue7c.log; do sleep 5; done
pkill -f "^/bin/bash ./queue7c.sh"; sleep 1; ./srv.sh stop; sleep 2
QU=/opt/models/dl/.ninferno-cache/gemma-4-31B-it-qat-UD-Q4_K_XL.gemma4_published.ninferno
QH=/opt/models/dl2/heretic-qat-q40qat.ninferno
QN=/opt/models/dl2/qat-nvfp4.r31.ninferno
DQ=/opt/models/dl2/qat-assistant.q40qat.ninferno
G=/opt/NInferno/build/apps/ninferno-gemma4
step() { echo "=== $(date +%T) $*" | tee -a queue7.log; }
step "F4 NVFP4 W4A4 + GeGLU split (min T 64), shared2 64K chunk2048: prefill"
NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=64 NINFERNO_GEMMA4_SHARED_VALUES=2 ./srv.sh start F4-nvfp4-split $QN --max-context 70000 --prefill-chunk 2048 --kv-dtype fp8 --spec none && \
    python3 gbench.py --label "F4 QAT-NVFP4 W4A4 GeGLU-split plain shared2 64K chunk2048" --pp 2048,8192,32768,65536 --tg "" --skip-chat --reps 1
./srv.sh stop
step "F5 NVFP4 + GeGLU split + QAT drafter K8: decode"
NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=64 NINFERNO_GEMMA4_SHARED_VALUES=2 ./srv.sh start F5-nvfp4-split-k8 $QN --max-context 70000 --prefill-chunk 2048 --kv-dtype fp8 --spec mtp --drafter $DQ --draft-tokens 8 && \
    python3 gbench.py --label "F5 QAT-NVFP4 GeGLU-split QATdrafter K8 shared2 64K" --pp 2048,8192
./srv.sh stop
step "F6 profile NVFP4 + GeGLU split prefill 6.6K"
NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=64 nsys profile -f true -o /root/gbench/prof_nv_split --trace=cuda --sample=none --cpuctxsw=none \
    $G $QN bench --tokens /root/gbench/prompt128k.i32 --prompt 6656 --decode 4 --kv-dtype fp8 --context 7168 --chunk 2048 > /root/gbench/prof_nv_split.out 2>&1
nsys stats --report cuda_gpu_kern_sum --format csv -o /root/gbench/prof_nv_split /root/gbench/prof_nv_split.nsys-rep > /dev/null 2>&1
python3 /root/gbench/kerntop.py /root/gbench/prof_nv_split_cuda_gpu_kern_sum.csv | head -14
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
    python3 gbench.py --label "H1 Heretic QATaware QATdrafter K8 shared2 262K"
./srv.sh stop
step "QUEUE7 DONE"
