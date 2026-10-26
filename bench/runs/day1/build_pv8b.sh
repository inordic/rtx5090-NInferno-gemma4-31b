#!/bin/bash
# Build FP8 PV + INT8 D512 attention + W4A8 option + GeGLU split aside into apps-pv8b;
# restore apps (mode2 + GeGLU split) and the mode2 kernel afterwards.
set -e
B=/opt/NInferno/build
K=/opt/NInferno/src/ops/softmax_attention/gemma4/kernel.cuh
mkdir -p $B/apps-pv8b
cp /root/gbench/kernel_pv8b.cuh $K
cd $B && PATH=/usr/local/cuda-13.2/bin:$PATH nice -n 10 ninja -j4 ninferno-serve ninferno-gemma4 ninferno > /root/gbench/build_pv8b.log 2>&1 || { echo BUILD_FAILED; cp /root/gbench/kernel_mode2.cuh $K; cp -p $B/apps-split/* $B/apps/; exit 1; }
cp $B/apps/ninferno-serve $B/apps/ninferno-gemma4 $B/apps/ninferno $B/apps-pv8b/
cp -p $B/apps-split/ninferno-serve $B/apps-split/ninferno-gemma4 $B/apps-split/ninferno $B/apps/
cp /root/gbench/kernel_mode2.cuh $K
echo BUILD_PV8B_OK
