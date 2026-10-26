#!/bin/bash
# build_aside.sh NAME: build the current tree and copy the binaries into build/apps-NAME.
B=/opt/NInferno/build
mkdir -p $B/apps-$1
if (cd $B && PATH=/usr/local/cuda-13.2/bin:$PATH nice -n 10 ninja -j4 ninferno-serve ninferno-gemma4 ninferno > /root/gbench/build_$1.log 2>&1); then
    cp $B/apps/ninferno-serve $B/apps/ninferno-gemma4 $B/apps/ninferno $B/apps-$1/ && echo BUILD_$1_OK
else
    echo BUILD_$1_FAILED; grep -m5 -E "error" /root/gbench/build_$1.log
fi
