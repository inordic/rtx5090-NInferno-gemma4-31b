#!/bin/bash
# build_sched.sh: three A16T16 schedule variants for the weaker decode shapes, each built into its own
# linear bench binary (lb_v1..lb_v3); the shape files go back to HEAD afterwards.
cd /opt/NInferno
S=src/ops/linear/nvfp4/shapes
F="$S/n5376_k21504.cu $S/n5376_k8192.cu $S/n5376_k16384.cu $S/n16384_k5376.cu"
declare -A V=( [v1]="1, 4, 1, 16, 8, 2, 2" [v2]="1, 8, 1, 16, 16, 2, 1" [v3]="1, 4, 1, 16, 16, 3, 2" )
for v in v1 v2 v3; do
    for f in $F; do
        sed -i -E "s/^(using A16T16 +=) Nvfp4A16RegMmaSchedule<[^>]*>;/\1 Nvfp4A16RegMmaSchedule<${V[$v]}>;/" $f
    done
    grep -h "using A16T16" $F | head -4
    if (cd build && PATH=/usr/local/cuda-13.2/bin:$PATH nice -n 10 ninja -j6 bench/ninferno_linear_bench > /root/gbench/build_sched_$v.log 2>&1); then
        cp build/bench/ninferno_linear_bench /root/gbench/lb_$v && echo "SCHED_${v}_OK"
    else
        echo "SCHED_${v}_FAILED"; grep -m3 error /root/gbench/build_sched_$v.log
    fi
    git checkout -- $F
done
(cd build && PATH=/usr/local/cuda-13.2/bin:$PATH nice -n 10 ninja -j6 bench/ninferno_linear_bench > /root/gbench/build_sched_base.log 2>&1) && echo SCHED_BASE_REBUILT
echo SCHED_DONE
