#!/bin/bash
# Build split-Q FP8 scores + FP8 PV aside into apps-qk8x2; the tree goes back to HEAD afterwards.
set -e
B=/opt/NInferno/build
K=src/ops/softmax_attention/gemma4/kernel.cuh
mkdir -p $B/apps-qk8x2
cp /root/gbench/kernel_qk8x2.cuh /opt/NInferno/$K
cd $B && PATH=/usr/local/cuda-13.2/bin:$PATH nice -n 10 ninja -j4 ninferno-serve ninferno-gemma4 ninferno > /root/gbench/build_qk8x2.log 2>&1 || { echo BUILD_FAILED; git -C /opt/NInferno checkout HEAD -- $K; exit 1; }
cp $B/apps/ninferno-serve $B/apps/ninferno-gemma4 $B/apps/ninferno $B/apps-qk8x2/
git -C /opt/NInferno checkout HEAD -- $K
echo BUILD_QK8X2_OK
