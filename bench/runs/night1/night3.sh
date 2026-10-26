#!/bin/bash
# night3: behind a 15-minute idle gate, after build_sched - (1) MTP draft count 7 / 8 / 12 / 15 on the
# deployed binary and config (verify T = 8 rides the 8-token tile, 9..16 the 16-token one); (2) the
# A16T16 schedule variants against the base on the four weaker decode shapes. Fast unit back after.
cd /root/gbench
step() { echo "=== $(date +%T) $*"; }
until grep -q SCHED_DONE build_sched.out 2>/dev/null; do sleep 15; done
step "waiting for 15 idle minutes on 8080"
./idle_gate.sh 15
step "idle: taking the GPU"
systemctl stop gemma4-31b-fast
for i in $(seq 1 90); do curl -sf http://127.0.0.1:8080/v1/models > /dev/null || break; sleep 1; done
QN=/opt/models/dl2/qat-nvfp4.r31.ninferno
DQ=/opt/models/dl2/qat-assistant.q40qat.ninferno
export NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=1 NINFERNO_GEMMA4_SHARED_VALUES=2
for k in 7 8 12 15; do
    step "K$k decode (final binary, 155K chunk1024)"
    ./srv_final.sh start K$k-final $QN --max-context 155000 --prefill-chunk 1024 --kv-dtype fp8 --spec mtp --drafter $DQ --draft-tokens $k && \
        python3 gbench.py --label "K$k QAT-NVFP4 final split1 155K" --pp "" --reps 1
    ./srv.sh stop
done
for b in /opt/NInferno/build/bench/ninferno_linear_bench lb_v1 lb_v2 lb_v3; do
    for s in "5376 21504" "5376 8192" "5376 16384" "16384 5376"; do
        set -- $s
        for t in 8 9; do
            printf "SCHED %-10s " $(basename $b)
            $b --qtype NVFP4 --policy a16 --n $1 --k $2 --t $t --execution graph --repeat 50 2>&1 | grep -E "^explicit" | tail -1 | cut -c40-110
        done
    done
done
step "restoring the fast unit"
./deploy_fast.sh
step "NIGHT3 DONE"
