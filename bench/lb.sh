#!/bin/bash
# lb.sh [bench binary]: NVFP4 A16 linear bench on the six Gemma 4 31B decode shapes at T 1 / 9 / 16.
B=${1:-/opt/NInferno/build/bench/ninferno_linear_bench}
for s in "16384 5376" "18432 5376" "5376 8192" "5376 16384" "43008 5376" "5376 21504"; do
    set -- $s
    for t in 1 9 16; do
        $B --qtype NVFP4 --policy a16 --n $1 --k $2 --t $t --execution graph --repeat 50 2>&1 | grep -vE "^\s*$" | tail -1
    done
done
