#!/bin/bash
# night8: behind a 5-minute idle gate - does a device-side cached state remove the ~260 ms of a
# cached agent turn? The candidate (final2, K7) at 155K / 120K / 100K context, turnprobe on each;
# then the Q6 output head [262144,5376] in the linear bench. Deployed unit back after.
cd /root/gbench
step() { echo "=== $(date +%T) $*"; }
./idle_gate_any.sh 5
step "idle: taking the GPU"
systemctl stop gemma4-31b-fast
for i in $(seq 1 90); do curl -sf http://127.0.0.1:8080/v1/models > /dev/null || break; sleep 1; done
QN=/opt/models/dl2/qat-nvfp4.r31.ninferno
DQ=/opt/models/dl2/qat-assistant.q40qat.ninferno
export NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=1 NINFERNO_GEMMA4_SHARED_VALUES=2
for ctx in 155000 120000 100000; do
    step "context $ctx"
    ./srv_final2.sh start T$ctx $QN --max-context $ctx --prefill-chunk 1024 --kv-dtype fp8 --spec mtp --drafter $DQ --draft-tokens 7 && \
        grep -hE "plan \| rung|context cache" logs/T$ctx.log | sed -E "s/^.*INFO +//" | cut -c1-230 && \
        python3 turnprobe.py http://127.0.0.1:8080
    ./srv.sh stop
done
step "Q6 head bench"
for t in 1 8; do
    printf "HEAD T=%s " $t
    /root/gbench/lb_base --qtype Q6 --policy a16 --n 262144 --k 5376 --t $t --execution graph --repeat 30 2>&1 | grep -E "^explicit" | tail -1 | cut -c40-130
done
step "restoring the deployed unit"
./deploy_fast.sh
step "NIGHT8 DONE"
