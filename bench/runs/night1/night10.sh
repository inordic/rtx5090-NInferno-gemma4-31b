#!/bin/bash
# night10: behind a 5-minute idle gate - the Q6 output head [262144,5376] at T 1..9: base (16-column
# MMA tile from T=7), A (SIMT through T=8), B (sliced 8-column MMA for T 7..8). Deployed unit back.
cd /root/gbench
step() { echo "=== $(date +%T) $*"; }
./idle_gate_any.sh 5
step "idle: taking the GPU"
systemctl stop gemma4-31b-fast
for i in $(seq 1 90); do curl -sf http://127.0.0.1:8080/v1/models > /dev/null || break; sleep 1; done
for rep in 1 2; do
    for b in lb_base lb_hA lb_hB; do
        for t in 1 6 7 8 9; do
            printf "HEAD %-8s T=%s " $b $t
            /root/gbench/$b --qtype Q6 --policy a16 --n 262144 --k 5376 --t $t --execution graph --repeat 30 2>&1 | grep -E "^explicit" | tail -1 | cut -c40-130
        done
    done
done
step "restoring the deployed unit"
./deploy_fast.sh
step "NIGHT10 DONE"
