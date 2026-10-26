#!/bin/bash
# Build Gemma 4 --activations int8 (W4A8) + INT8 D512 attention + GeGLU split aside into apps-a8;
# restore apps (mode2 + GeGLU split) and the mode2 kernel afterwards.
set -e
B=/opt/NInferno/build
K=/opt/NInferno/src/ops/softmax_attention/gemma4/kernel.cuh
mkdir -p $B/apps-a8
cp /root/gbench/kernel_i8.cuh $K
cd $B && PATH=/usr/local/cuda-13.2/bin:$PATH nice -n 10 ninja -j4 ninferno-serve ninferno-gemma4 ninferno > /root/gbench/build_a8.log 2>&1 || { echo BUILD_FAILED; cp /root/gbench/kernel_mode2.cuh $K; cp -p $B/apps-split/* $B/apps/; exit 1; }
cp $B/apps/ninferno-serve $B/apps/ninferno-gemma4 $B/apps/ninferno $B/apps-a8/
cp -p $B/apps-split/ninferno-serve $B/apps-split/ninferno-gemma4 $B/apps-split/ninferno $B/apps/
cp /root/gbench/kernel_mode2.cuh $K
echo BUILD_A8_OK
