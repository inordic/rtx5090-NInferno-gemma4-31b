#!/bin/bash
# night5: behind a 5-minute idle gate - (1) the QAT drafter with a shortlist head (64K / 96K rows of
# gemma4_ranking2) against the full head, final binary, K=7; (2) the A16T16 schedule variants against
# the base at T 8 / 9 on the four weaker decode shapes. Fast unit back after.
cd /root/gbench
step() { echo "=== $(date +%T) $*"; }
./srv.sh stop > /dev/null 2>&1
step "waiting for 5 idle minutes on 8080"
./idle_gate_any.sh 5
step "idle: taking the GPU"
systemctl stop gemma4-31b-fast
for i in $(seq 1 90); do curl -sf http://127.0.0.1:8080/v1/models > /dev/null || break; sleep 1; done
QN=/opt/models/dl2/qat-nvfp4.r31.ninferno
export NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=1 NINFERNO_GEMMA4_SHARED_VALUES=2
for d in sl64k-q8 sl96k-q8 q40qat; do
    step "drafter $d (K7)"
    ./srv_final.sh start DR7-$d $QN --max-context 155000 --prefill-chunk 1024 --kv-dtype fp8 --spec mtp \
        --drafter /opt/models/dl2/qat-assistant.$d.ninferno --draft-tokens 7 && \
        python3 gbench.py --label "DR7 $d QAT-NVFP4 final split1 K7 155K" --pp "" --reps 1
    ./srv.sh stop
done
for b in /opt/NInferno/build/bench/ninferno_linear_bench /root/gbench/lb_v1 /root/gbench/lb_v2 /root/gbench/lb_v3; do
    for s in "5376 21504" "5376 8192" "5376 16384" "16384 5376"; do
        set -- $s
        for t in 8 9; do
            printf "SCHED %-24s " $(basename $b)
            $b --qtype NVFP4 --policy a16 --n $1 --k $2 --t $t --execution graph --repeat 50 2>&1 | grep -E "^explicit" | tail -1 | cut -c40-110
        done
    done
done
step "restoring the fast unit"
./deploy_fast.sh
step "NIGHT5 DONE"
