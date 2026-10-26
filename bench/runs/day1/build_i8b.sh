#!/bin/bash
# Build INT8-scores (D>=256) + GeGLU split aside into apps-i8b; restore apps (mode2 + GeGLU split).
set -e
B=/opt/NInferno/build
mkdir -p $B/apps-split $B/apps-i8b
cp -p $B/apps/ninferno-serve $B/apps/ninferno-gemma4 $B/apps/ninferno $B/apps-split/
cp /root/gbench/kernel_i8b.cuh /opt/NInferno/src/ops/softmax_attention/gemma4/kernel.cuh
cd $B && PATH=/usr/local/cuda-13.2/bin:$PATH nice -n 10 ninja -j4 ninferno-serve ninferno-gemma4 ninferno > /root/gbench/build_i8b.log 2>&1 || { echo BUILD_FAILED; cp /root/gbench/kernel_mode2.cuh /opt/NInferno/src/ops/softmax_attention/gemma4/kernel.cuh; cp -p $B/apps-split/* $B/apps/; exit 1; }
cp $B/apps/ninferno-serve $B/apps/ninferno-gemma4 $B/apps/ninferno $B/apps-i8b/
cp -p $B/apps-split/ninferno-serve $B/apps-split/ninferno-gemma4 $B/apps-split/ninferno $B/apps/
cp /root/gbench/kernel_mode2.cuh /opt/NInferno/src/ops/softmax_attention/gemma4/kernel.cuh
echo BUILD_I8B_OK
