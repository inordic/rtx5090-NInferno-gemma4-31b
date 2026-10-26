#!/bin/bash
# Profile NInferno Gemma 31B QAT-NVFP4 (W4A4) prefill of ~6.6K tokens in the lab tool.
cd /root/gbench
QN=/opt/models/dl2/qat-nvfp4.r31.ninferno
G=/opt/NInferno/build/apps/ninferno-gemma4
nsys profile -f true -o /root/gbench/prof_nv_8k --trace=cuda --sample=none --cpuctxsw=none \
  $G $QN bench --tokens /root/gbench/prompt128k.i32 --prompt 6656 --decode 4 --kv-dtype fp8 --context 7168 --chunk 2048 > /root/gbench/prof_nv_8k.out 2>&1
tail -2 /root/gbench/prof_nv_8k.out
nsys stats --report cuda_gpu_kern_sum --format csv -o /root/gbench/prof_nv_8k /root/gbench/prof_nv_8k.nsys-rep > /dev/null 2>&1
ls /root/gbench/prof_nv_8k*
echo NV_PROF_DONE
