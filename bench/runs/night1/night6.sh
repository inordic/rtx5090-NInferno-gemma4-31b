#!/bin/bash
# night6: after night5, behind a 5-minute idle gate - the QKV shapes' 8-token schedule <1,4,1,8,...>
# (lb_q8) against the base <2,2,1,8,...> at T 1 and 8. Fast unit back after.
cd /root/gbench
step() { echo "=== $(date +%T) $*"; }
until grep -q "NIGHT5 DONE" night5.log 2>/dev/null; do sleep 15; done
./idle_gate_any.sh 5
step "idle: taking the GPU"
systemctl stop gemma4-31b-fast
for i in $(seq 1 90); do curl -sf http://127.0.0.1:8080/v1/models > /dev/null || break; sleep 1; done
for rep in 1 2; do
    for b in /root/gbench/lb_base /root/gbench/lb_q8; do
        for s in "16384 5376" "18432 5376"; do
            set -- $s
            for t in 1 8; do
                printf "Q8SCHED %-8s " $(basename $b)
                $b --qtype NVFP4 --policy a16 --n $1 --k $2 --t $t --execution graph --repeat 50 2>&1 | grep -E "^explicit" | tail -1 | cut -c40-110
            done
        done
    done
done
step "restoring the fast unit"
./deploy_fast.sh
step "NIGHT6 DONE"
