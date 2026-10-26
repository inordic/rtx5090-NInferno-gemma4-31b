#!/bin/bash
# build_q6.sh: the Q6 head at T 7..8 on SIMT (A) or the sliced T8 MMA tile (B), each into its own
# linear bench binary; the tree goes back to HEAD afterwards.
cd /opt/NInferno
for v in A B; do
    git checkout -- src/ops/linear/q6
    python3 /root/gbench/patch_q6.py $v
    if (cd build && PATH=/usr/local/cuda-13.2/bin:$PATH nice -n 10 ninja -j6 bench/ninferno_linear_bench > /root/gbench/build_q6$v.log 2>&1); then
        cp build/bench/ninferno_linear_bench /root/gbench/lb_h$v && echo "Q6_${v}_OK"
    else
        echo "Q6_${v}_FAILED"; grep -m5 -E "error" /root/gbench/build_q6$v.log
    fi
done
git checkout -- src/ops/linear/q6
(cd build && PATH=/usr/local/cuda-13.2/bin:$PATH nice -n 10 ninja -j6 bench/ninferno_linear_bench > /root/gbench/build_q6base.log 2>&1) && echo Q6_BASE_REBUILT
echo Q6_DONE
