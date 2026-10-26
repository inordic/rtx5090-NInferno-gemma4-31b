#!/bin/bash
# queue20: after queue18 (graph nodes traced) - where does an MTP K=8 decode step go? nsys over the serve process while
# it answers one 512-token chat request (NVFP4 + split + INT8, shared2, 16K context).
cd /root/gbench
until grep -qE "QUEUE18 DONE|MSE ARTIFACT MISSING" queue18.log 2>/dev/null; do sleep 15; done
step() { echo "=== $(date +%T) $*"; }
./srv.sh stop; sleep 2
QN=/opt/models/dl2/qat-nvfp4.r31.ninferno
DQ=/opt/models/dl2/qat-assistant.q40qat.ninferno
export NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=64 NINFERNO_GEMMA4_SHARED_VALUES=2
step "profile decode: serve under nsys, one 512-token chat"
cd /opt/models
PATH=/usr/local/cuda-13.2/bin:$PATH nsys profile -f true -o /root/gbench/prof_decode2 --trace=cuda --sample=none --cpuctxsw=none --cuda-graph-trace=node --duration=90 \
    /opt/NInferno/build/apps-a8/ninferno-serve $QN --host 0.0.0.0 --port 8080 --model-id prof --max-context 16384 --kv-dtype fp8 \
    --spec mtp --drafter $DQ --draft-tokens 8 > /root/gbench/logs/prof_decode2.log 2>&1 &
cd /root/gbench
for i in $(seq 1 120); do grep -q "listening on" logs/prof_decode2.log && break; sleep 1; done
python3 - <<"PY"
import json, time, urllib.request
body = {"model": "x", "messages": [{"role": "user", "content": "Write a detailed explanation of how a hash map works, with a Python implementation."}], "max_tokens": 512, "temperature": 0}
for _ in range(2):
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})))
    print("chat", r["usage"], f"{time.time() - t:.2f}s")
PY
wait
grep -E "req#.*done" logs/prof_decode2.log | sed -E "s/^.*INFO +//" | cut -c1-260
nsys stats --report cuda_gpu_kern_sum --format csv -o /root/gbench/prof_decode2 /root/gbench/prof_decode2.nsys-rep > /dev/null 2>&1
python3 /root/gbench/kerntop.py /root/gbench/prof_decode2_cuda_gpu_kern_sum.csv | head -30
step "QUEUE20 DONE"
