#!/bin/bash
# Build the promoted FP8-scores variants (p1: one E4M3 Q part, p2: split Q) aside, one after the
# other, into apps-qk8p1 / apps-qk8p2; the tree goes back to HEAD afterwards.
B=/opt/NInferno/build
K=src/ops/softmax_attention/gemma4/kernel.cuh
for v in qk8p1 qk8p2; do
    mkdir -p $B/apps-$v
    cp /root/gbench/kernel_$v.cuh /opt/NInferno/$K
    if (cd $B && PATH=/usr/local/cuda-13.2/bin:$PATH nice -n 10 ninja -j4 ninferno-serve ninferno-gemma4 ninferno > /root/gbench/build_$v.log 2>&1); then
        cp $B/apps/ninferno-serve $B/apps/ninferno-gemma4 $B/apps/ninferno $B/apps-$v/ && echo BUILD_${v}_OK
    else
        echo BUILD_${v}_FAILED; grep -m3 error /root/gbench/build_$v.log
    fi
done
git -C /opt/NInferno checkout HEAD -- $K
echo BUILD_QK8P_DONE
