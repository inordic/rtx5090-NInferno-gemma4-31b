#!/bin/bash
# queue14: after queue13 - where does long-context prefill time go on the candidate (INT8 D512)?
# nsys kernel breakdown of a 32K prefill with the lab tool.
cd /root/gbench
until grep -qE "QUEUE13 DONE|A8 BUILD FAILED" queue13.log 2>/dev/null; do sleep 15; done
step() { echo "=== $(date +%T) $*"; }
./srv.sh stop; sleep 2
QN=/opt/models/dl2/qat-nvfp4.r31.ninferno
G=/opt/NInferno/build/apps-a8/ninferno-gemma4
export NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=64 NINFERNO_GEMMA4_SHARED_VALUES=2
step "profile 32K prefill (INT8 D512, split, shared2)"
nsys profile -f true -o /root/gbench/prof_i8_32k --trace=cuda --sample=none --cpuctxsw=none \
    $G $QN bench --tokens /root/gbench/prompt128k.i32 --prompt 32768 --decode 4 --kv-dtype fp8 --context 34816 --chunk 2048 > /root/gbench/prof_i8_32k.out 2>&1
tail -3 /root/gbench/prof_i8_32k.out
nsys stats --report cuda_gpu_kern_sum --format csv -o /root/gbench/prof_i8_32k /root/gbench/prof_i8_32k.nsys-rep > /dev/null 2>&1
python3 /root/gbench/kerntop.py /root/gbench/prof_i8_32k_cuda_gpu_kern_sum.csv | head -24
step "QUEUE14 DONE"
