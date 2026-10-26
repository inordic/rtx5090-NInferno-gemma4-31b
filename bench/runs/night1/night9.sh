#!/bin/bash
# night9: behind a 5-minute idle gate - (1) the Q6 output head across T 1..16; (2) an nsys trace of
# the candidate serving cached agent turns, to see how much of a turn's ~250 ms the GPU is busy.
cd /root/gbench
step() { echo "=== $(date +%T) $*"; }
./idle_gate_any.sh 5
step "idle: taking the GPU"
systemctl stop gemma4-31b-fast
for i in $(seq 1 90); do curl -sf http://127.0.0.1:8080/v1/models > /dev/null || break; sleep 1; done
step "Q6 head sweep"
/root/gbench/lb_base --qtype Q6 --policy a16 --n 262144 --k 5376 --sweep 1:16:1 --execution graph --repeat 20 2>&1 | grep -E "^explicit|route" | cut -c40-150
step "nsys: cached turns on the candidate"
QN=/opt/models/dl2/qat-nvfp4.r31.ninferno
DQ=/opt/models/dl2/qat-assistant.q40qat.ninferno
export NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=1 NINFERNO_GEMMA4_SHARED_VALUES=2
cd /opt/models
PATH=/usr/local/cuda-13.2/bin:$PATH nsys profile -f true -o /root/gbench/prof_turn --trace=cuda --sample=none --cpuctxsw=none --cuda-graph-trace=node --duration=75 \
    /opt/NInferno/build/apps-final2/ninferno-serve $QN --host 0.0.0.0 --port 8080 --model-id prof --max-context 100000 --prefill-chunk 1024 \
    --kv-dtype fp8 --spec mtp --drafter $DQ --draft-tokens 7 > /root/gbench/logs/prof_turn.log 2>&1 &
cd /root/gbench
for i in $(seq 1 150); do grep -q "listening on" logs/prof_turn.log && break; sleep 1; done
python3 turnprobe.py http://127.0.0.1:8080 > prof_turn_probe.txt 2>&1
date +%s.%N > prof_turn_end.txt
cat prof_turn_probe.txt
wait
nsys stats --report cuda_gpu_trace --format csv -o /root/gbench/prof_turn /root/gbench/prof_turn.nsys-rep > /dev/null 2>&1
ls -la /root/gbench/prof_turn_cuda_gpu_trace.csv
step "restoring the deployed unit"
./deploy_fast.sh
step "NIGHT9 DONE"
