#!/bin/bash
# Build the INT8-scores variant aside: back up current apps, build, copy to apps-i8, restore.
set -e
B=/opt/NInferno/build
mkdir -p $B/apps-mode2 $B/apps-i8
cp -p $B/apps/ninferno-serve $B/apps/ninferno-gemma4 $B/apps/ninferno $B/apps-mode2/
cp /opt/NInferno/src/ops/softmax_attention/gemma4/kernel.cuh /root/gbench/kernel_mode2.cuh
cp /root/gbench/kernel_i8.cuh /opt/NInferno/src/ops/softmax_attention/gemma4/kernel.cuh
cd $B && PATH=/usr/local/cuda-13.2/bin:$PATH nice -n 15 ninja -j4 ninferno-serve ninferno-gemma4 ninferno > /root/gbench/build_i8.log 2>&1 || { echo BUILD_FAILED; cp -p $B/apps-mode2/* $B/apps/; exit 1; }
cp $B/apps/ninferno-serve $B/apps/ninferno-gemma4 $B/apps/ninferno $B/apps-i8/
cp -p $B/apps-mode2/ninferno-serve $B/apps-mode2/ninferno-gemma4 $B/apps-mode2/ninferno $B/apps/
echo BUILD_I8_OK
